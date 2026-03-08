# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark prefill latency with CPU 100% prefix-cache hits.

Workflow per prompt length:
1) Warmup prefill to populate GPU+LMCache.
2) Baseline prefill: run prompts with GPU prefix cache hot.
3) Reset only GPU prefix cache (LMCache remains), then run the same prompts.

Reports per length:
- baseline prefill latency (HBM hit)
- CPU hit prefill latency (LMCache->GPU)
- LMCache load latency and bandwidth
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import random
import statistics
import time
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:  # pragma: no cover - fallback for testing harnesses
    from backend_request_func import get_tokenizer


@dataclasses.dataclass
class Request:
    prompt: str
    prompt_len: int


class _BenchStatsLogger:
    """Capture per-run timing and kv-connector stats from vLLM v1 stat logs.

    Note: vLLM v1 does not currently attach per-request metrics to
    `RequestOutput.metrics` in offline `LLM.generate()`, so we rely on
    IterationStats/SchedulerStats emitted to StatLoggerManager.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ttft_s: list[float] = []
        self.kv_connector_stats: dict[str, float] = {}

    def record(self, scheduler_stats=None, iteration_stats=None, engine_idx: int = 0):
        if iteration_stats is not None:
            # Seconds. One entry per request that first-tokened in this iter.
            values = getattr(iteration_stats, "time_to_first_tokens_iter", None)
            if values:
                self.ttft_s.extend(values)

        if scheduler_stats is not None:
            kv_stats = getattr(scheduler_stats, "kv_connector_stats", None)
            if isinstance(kv_stats, dict):
                for k, v in kv_stats.items():
                    if isinstance(v, (int, float)):
                        self.kv_connector_stats[k] = self.kv_connector_stats.get(k, 0.0) + float(v)

    def log_engine_initialized(self):
        return

    def log(self):
        return


def _install_bench_stats_loggers(llm: LLM) -> dict[int, _BenchStatsLogger]:
    """Inject capture loggers into vLLM v1 StatLoggerManager (best-effort)."""

    if not hasattr(llm, "llm_engine"):
        return {}

    manager = getattr(llm.llm_engine, "logger_manager", None)
    if manager is None or not hasattr(manager, "per_engine_logger_dict"):
        return {}

    per_engine: dict[int, _BenchStatsLogger] = {}
    for engine_idx, engine_loggers in manager.per_engine_logger_dict.items():
        cap = _BenchStatsLogger()
        engine_loggers.append(cap)
        per_engine[int(engine_idx)] = cap
    return per_engine


def _parse_prompt_lens(arg: str) -> list[int]:
    values: list[int] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"Invalid prompt length '{piece}'") from exc
        if value < 1:
            raise ValueError(f"prompt length must be >= 1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one prompt length is required")
    return values


def _sample_tokens(
    tokenizer: PreTrainedTokenizerBase, length: int, rng: random.Random
) -> list[int]:
    vocab = tokenizer.get_vocab()
    specials = set(tokenizer.all_special_ids)
    candidates = [tid for token, tid in vocab.items() if tid not in specials]
    return rng.choices(candidates, k=length)


def _build_prompts(
    tokenizer: PreTrainedTokenizerBase,
    num_prompts: int,
    input_length: int,
    seed: int,
) -> list[Request]:
    rng = random.Random(seed)
    reqs: list[Request] = []
    for _ in range(num_prompts):
        ids = _sample_tokens(tokenizer, input_length, rng)
        reqs.append(Request(prompt=tokenizer.decode(ids), prompt_len=len(ids)))
    return reqs


def _prefill_ms_from_outputs(outputs) -> list[float]:
    latencies: list[float] = []
    for out in outputs:
        metrics = getattr(out, "metrics", None)
        if metrics is None:
            continue
        if metrics.first_token_time is None or metrics.arrival_time is None:
            continue
        latencies.append((metrics.first_token_time - metrics.arrival_time) * 1000.0)
    return latencies


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return values_sorted[0]
        idx = min(n - 1, max(0, int(math.ceil(p * (n - 1)))))
        return values_sorted[idx]

    return {
        "count": n,
        "mean_ms": statistics.fmean(values_sorted),
        "median_ms": pct(0.5),
        "p90_ms": pct(0.9),
        "p99_ms": pct(0.99),
        "min_ms": values_sorted[0],
        "max_ms": values_sorted[-1],
    }


def _summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return values_sorted[0]
        idx = min(n - 1, max(0, int(math.ceil(p * (n - 1)))))
        return values_sorted[idx]

    return {
        "count": n,
        "mean": statistics.fmean(values_sorted),
        "median": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def _normalize_lmcache_metrics(metrics: dict[str, float]) -> dict[str, float]:
    count = metrics.get("lmcache_load_count", 0) or 0
    total_time_s = metrics.get("lmcache_load_total_time_s", 0.0) or 0.0
    total_bytes = metrics.get("lmcache_load_total_bytes", 0) or 0

    if "lmcache_load_mean_time_ms" not in metrics:
        metrics["lmcache_load_mean_time_ms"] = (
            total_time_s / count * 1000.0 if count > 0 else 0.0
        )
    if "lmcache_load_agg_bw_gbps" not in metrics:
        metrics["lmcache_load_agg_bw_gbps"] = (
            (total_bytes / total_time_s) / (1024**3)
            if total_time_s > 0
            else 0.0
        )
    return metrics


def _get_kv_connector_metrics_from_capture(
    capture_loggers: dict[int, _BenchStatsLogger]
) -> Optional[dict[str, float]]:
    if not capture_loggers:
        return None
    # Prefer engine 0.
    cap = capture_loggers.get(0) or next(iter(capture_loggers.values()), None)
    if cap is None:
        return None
    if not cap.kv_connector_stats:
        return None
    return _normalize_lmcache_metrics(dict(cap.kv_connector_stats))


def _reset_gpu_prefix_cache(llm: LLM, retries: int = 5, delay_s: float = 0.25) -> bool:
    for attempt in range(1, max(retries, 1) + 1):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            reset_ret = llm.reset_prefix_cache()
        except Exception:
            reset_ret = False

        reset_ok = reset_ret is not False
        if reset_ok:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return True

        if attempt < retries:
            time.sleep(delay_s)

    return False


def _run_and_collect(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    capture_loggers: Optional[dict[int, _BenchStatsLogger]] = None,
) -> tuple[dict[str, float], Optional[dict[str, float]], float]:
    if capture_loggers:
        for cap in capture_loggers.values():
            cap.reset()

    start = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    wall = time.time() - start

    # vLLM v1 offline path typically does not populate RequestOutput.metrics.
    lat_ms: list[float] = []
    if capture_loggers:
        cap = capture_loggers.get(0) or next(iter(capture_loggers.values()), None)
        if cap is not None and cap.ttft_s:
            lat_ms = [t * 1000.0 for t in cap.ttft_s if t > 0]

    if not lat_ms:
        # Fallback: try old-style output metrics (v0), then wall-time.
        lat_ms = _prefill_ms_from_outputs(outputs)

    if not lat_ms:
        per_req_ms = (wall * 1000.0 / max(len(prompts), 1))
        lat_ms = [per_req_ms] * max(len(prompts), 1)

    kv = _get_kv_connector_metrics_from_capture(capture_loggers or {})
    return _summarize(lat_ms), kv, wall


def _warmup(llm: LLM, prompts: list[str], sampling_params: SamplingParams, runs: int) -> None:
    for _ in range(max(runs, 0)):
        llm.generate(prompts, sampling_params=sampling_params)


def _cleanup_engine(llm: Optional[LLM]) -> None:
    if llm is None:
        return
    try:
        if hasattr(llm, "llm_engine"):
            llm.llm_engine.shutdown()
    except Exception:
        pass
    try:
        llm.__del__()  # type: ignore[attr-defined]
    except Exception:
        pass
    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def create_argument_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="Prefill latency benchmark for CPU 100% prefix-cache hits."
    )
    parser.add_argument(
        "--prompt-lens",
        type=str,
        default="256,512,1024",
        help="Comma-separated prompt lengths to test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts per batch (same length per batch).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup passes to pre-populate GPU+LMCache.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help="Skip detokenization time in latency numbers.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=1)
    parser.add_argument(
        "--output-json",
        type=str,
        default="kv_cache_prefill_cpu_hit_results.json",
        help="Path to dump JSON results.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline HBM prefill phase.",
    )
    parser.add_argument(
        "--skip-cpu-hit",
        action="store_true",
        help="Skip CPU hit prefill phase.",
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    random.seed(args.shuffle_seed)

    prompt_lens = _parse_prompt_lens(args.prompt_lens)
    batch_size = args.batch_size
    if batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        detokenize=not args.disable_detokenize,
    )

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.disable_log_stats = False
    llm = LLM(**dataclasses.asdict(engine_args))

    capture_loggers = _install_bench_stats_loggers(llm)

    all_results = []

    for prompt_len in prompt_lens:
        print(f"\n=== Prompt length {prompt_len} ===")
        reqs = _build_prompts(
            tokenizer=tokenizer,
            num_prompts=batch_size,
            input_length=prompt_len,
            seed=args.shuffle_seed,
        )
        prompts = [r.prompt for r in reqs]
        print(f"Batch size={batch_size}, prompt_len={prompt_len}")

        _warmup(llm, prompts, sampling_params, runs=max(args.warmup_runs, 1))

        baseline_summary: Optional[dict[str, float]] = None
        baseline_kv: Optional[dict[str, float]] = None
        baseline_wall: Optional[float] = None

        cpu_summary: Optional[dict[str, float]] = None
        cpu_kv: Optional[dict[str, float]] = None
        cpu_wall: Optional[float] = None
        reset_ok: Optional[bool] = None

        if not args.skip_baseline:
            baseline_summary, baseline_kv, baseline_wall = _run_and_collect(
                llm, prompts, sampling_params, capture_loggers=capture_loggers
            )
            print(
                f"Baseline prefill mean={baseline_summary.get('mean_ms', float('nan')):.2f} ms, "
                f"p90={baseline_summary.get('p90_ms', float('nan')):.2f} ms"
            )

        if not args.skip_cpu_hit:
            reset_ok = _reset_gpu_prefix_cache(llm)
            if not reset_ok:
                print(
                    "[WARN] reset_prefix_cache reported failure after retries; CPU hit run may still see HBM hits."
                )

            cpu_summary, cpu_kv, cpu_wall = _run_and_collect(
                llm, prompts, sampling_params, capture_loggers=capture_loggers
            )
            print(
                f"CPU-hit prefill mean={cpu_summary.get('mean_ms', float('nan')):.2f} ms, "
                f"p90={cpu_summary.get('p90_ms', float('nan')):.2f} ms"
            )
            if cpu_kv:
                print(
                    "LMCache load: "
                    f"count={cpu_kv.get('lmcache_load_count', 0)}, "
                    f"mean={cpu_kv.get('lmcache_load_mean_time_ms', float('nan')):.2f} ms, "
                    f"bw={cpu_kv.get('lmcache_load_agg_bw_gbps', float('nan')):.2f} GB/s"
                )

        all_results.append(
            {
                "prompt_len": prompt_len,
                "batch_size": batch_size,
                "baseline": {
                    "summary": baseline_summary,
                    "lmcache_load": baseline_kv,
                    "wall_time_s": baseline_wall,
                }
                if baseline_summary
                else None,
                "cpu_hit": {
                    "summary": cpu_summary,
                    "lmcache_load": cpu_kv,
                    "wall_time_s": cpu_wall,
                }
                if cpu_summary
                else None,
                "lmcache_gpu_cache_reset_ok": reset_ok,
            }
        )

    def _num(val: Optional[float]) -> str:
        return f"{val:.2f}" if val is not None else "-"

    print("\nSummary (per prompt length):")
    header = (
        "len",
        "base_ms",
        "cpu_ms",
        "lm_load_ms",
        "lm_load_total_s",
        "lm_bw_gbps",
    )
    col_w = [8, 12, 12, 12, 16, 12]
    header_row = " ".join(h.ljust(w) for h, w in zip(header, col_w))
    print(header_row)
    print("-" * len(header_row))

    for res in all_results:
        base = res["baseline"]
        cpu = res["cpu_hit"]
        base_ms = base["summary"].get("mean_ms") if base else None
        cpu_ms = cpu["summary"].get("mean_ms") if cpu else None
        lm_load = cpu.get("lmcache_load") if cpu else None
        lm_load_ms = lm_load.get("lmcache_load_mean_time_ms") if lm_load else None
        lm_load_total_s = lm_load.get("lmcache_load_total_time_s") if lm_load else None
        lm_bw = lm_load.get("lmcache_load_agg_bw_gbps") if lm_load else None

        row = (
            str(res["prompt_len"]),
            _num(base_ms),
            _num(cpu_ms),
            _num(lm_load_ms),
            _num(lm_load_total_s),
            _num(lm_bw),
        )
        print(" ".join(val.ljust(w) for val, w in zip(row, col_w)))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {args.output_json}")

    # _cleanup_engine(llm)


if __name__ == "__main__":
    main()
