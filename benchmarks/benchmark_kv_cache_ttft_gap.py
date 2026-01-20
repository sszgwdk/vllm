# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Measure TTFT gap when prefix cache hits GPU HBM vs LMCache DRAM.

The script builds synthetic prompts with a controllable shared prefix ratio
(X%) and runs two phases per ratio:
- HBM hit: reuse the warmed GPU prefix cache in the same engine.
- LMCache hit: clear only the GPU prefix cache (no engine restart) so HBM is
    cold while LMCache DRAM still holds the prefix blocks; TTFT then includes
    LMCache->GPU load cost.

Example (mirrors the user-provided settings):

    export VLLM_USE_MODELSCOPE=true
    export LMCACHE_MAX_LOCAL_CPU_SIZE=150.0
    export LMCACHE_LOG_LEVEL=WARNING
    python3 benchmarks/benchmark_kv_cache_ttft_gap.py \
      --model mistralai/Mistral-Small-24B-Instruct-2501 \
      --enable-prefix-caching \
      --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
      --gpu-memory-utilization 0.95 \
      --compilation-config '{"level": 0, "cudagraph_mode": "NONE"}' \
    --num-prompts 1000 \
    --input-length 4096 \
      --output-len 1 \
    --hit-ratios 0.25,0.5,0.75,1.0

Notes:
- LMCache must persist across runs to test DRAM hits. If you run inside a
    container with ephemeral /dev/shm, point LMCache to a persistent path (e.g.,
    set LMCACHE_ROOT or similar) before running.
- To make LMCache hits valid, the script now clears only the GPU prefix cache
    via ``LLM.reset_prefix_cache``. LMCache entries remain intact so the LMCache
    phase measures DRAM->HBM transfer overhead without rebooting the engine.
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


# ---------------------- prompt construction helpers ---------------------- #


@dataclasses.dataclass
class Request:
    prompt: str
    prompt_len: int


def _parse_hit_ratios(arg: str) -> list[float]:
    values: list[float] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError as exc:  # pragma: no cover - CLI guard
            raise ValueError(f"Invalid hit ratio '{piece}'") from exc
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Hit ratio must be in [0,1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one hit ratio is required")
    return values


def _sample_tokens(
    tokenizer: PreTrainedTokenizerBase, length: int, rng: random.Random
) -> list[int]:
    vocab = tokenizer.get_vocab()
    specials = set(tokenizer.all_special_ids)
    candidates = [tid for token, tid in vocab.items() if tid not in specials]
    return rng.choices(candidates, k=length)


def _build_warm_test_triplets(
    tokenizer: PreTrainedTokenizerBase,
    num_prompts: int,
    input_length: int,
    hit_ratio: float,
    seed: int,
) -> tuple[list[Request], list[Request], list[Request]]:
    """为每个 test 构造 warm_up / hbm_test / lmcache_test 三元组。

    - 三者共享同一个前缀（长度 = input_length * hit_ratio 取整）。
    - 后缀各自独立随机生成，避免 hbm_test 把 LMCache 里的条目完全命中后
      再测 lmcache_test 出现 100% 的重复缓存。
    - 每个请求长度恒为 input_length。
    """
    rng = random.Random(seed)
    prefix_len = int(input_length * hit_ratio)
    suffix_len = max(input_length - prefix_len, 0)

    # shared_prefix = _sample_tokens(tokenizer, prefix_len, rng)

    warm_reqs: list[Request] = []
    hbm_reqs: list[Request] = []
    lmcache_reqs: list[Request] = []
    for _ in range(num_prompts):
        shared_prefix = _sample_tokens(tokenizer, prefix_len, rng)
        warm_suffix = _sample_tokens(tokenizer, suffix_len, rng)
        hbm_suffix = _sample_tokens(tokenizer, suffix_len, rng)
        lmcache_suffix = _sample_tokens(tokenizer, suffix_len, rng)

        warm_ids = shared_prefix + warm_suffix
        hbm_ids = shared_prefix + hbm_suffix
        lmcache_ids = shared_prefix + lmcache_suffix

        warm_reqs.append(Request(prompt=tokenizer.decode(warm_ids), prompt_len=len(warm_ids)))
        hbm_reqs.append(Request(prompt=tokenizer.decode(hbm_ids), prompt_len=len(hbm_ids)))
        lmcache_reqs.append(Request(prompt=tokenizer.decode(lmcache_ids), prompt_len=len(lmcache_ids)))

    return warm_reqs, hbm_reqs, lmcache_reqs


# --------------------------- metrics utilities --------------------------- #


def _ttft_ms_from_outputs(outputs) -> list[float]:
    ttfts: list[float] = []
    for out in outputs:
        metrics = getattr(out, "metrics", None)
        if metrics is None:
            continue
        if metrics.first_token_time is None or metrics.arrival_time is None:
            continue
        ttfts.append((metrics.first_token_time - metrics.arrival_time) * 1000.0)
    return ttfts


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


def _summarize_load_metrics(stats) -> Optional[dict[str, float]]:
    load_times = getattr(stats, "interval_load_time_s", None) or []
    load_bytes = getattr(stats, "interval_load_bytes", None) or []

    if not load_times or not load_bytes:
        return None

    pair_count = min(len(load_times), len(load_bytes))
    if pair_count == 0:
        return None

    load_times = load_times[:pair_count]
    load_bytes = load_bytes[:pair_count]

    time_ms = [t * 1000.0 for t in load_times if t > 0]
    bw_gbps = [
        (b / t) / (1024**3)
        for b, t in zip(load_bytes, load_times)
        if t > 0 and b > 0
    ]

    total_bytes = sum(load_bytes)
    total_time = sum(load_times)
    aggregate_bw_gbps = (
        (total_bytes / total_time) / (1024**3) if total_time > 0 else 0.0
    )

    return {
        "count": pair_count,
        "total_bytes": total_bytes,
        "total_time_s": total_time,
        "aggregate_bw_gbps": aggregate_bw_gbps,
        "time_ms": _summarize(time_ms),
        "bandwidth_gbps": _summarize_values(bw_gbps),
    }


def _get_prefix_cache_metrics(llm: LLM) -> Optional[dict[str, float]]:
    try:
        from vllm.v1.metrics.loggers import LoggingStatLogger
    except Exception:
        return None

    if not hasattr(llm, "llm_engine"):
        return None
    manager = getattr(llm.llm_engine, "logger_manager", None)
    if manager is None:
        return None

    for engine_loggers in manager.per_engine_logger_dict.values():
        for logger in engine_loggers:
            if isinstance(logger, LoggingStatLogger):
                metrics = {
                    "prefix_cache_hit_rate": logger.prefix_caching_metrics.hit_rate,
                    "gpu_prefix_cache_hit_rate": logger.prefix_caching_metrics.gpu_hit_rate,
                    "connector_prefix_cache_hit_rate": logger.prefix_caching_metrics.connector_hit_rate,
                }
                if hasattr(logger.prefix_caching_metrics, "total_hit_rate"):
                    metrics["prefix_cache_total_hit_rate"] = (
                        logger.prefix_caching_metrics.total_hit_rate
                    )
                if hasattr(logger.prefix_caching_metrics, "total_gpu_hit_rate"):
                    metrics["gpu_prefix_cache_total_hit_rate"] = (
                        logger.prefix_caching_metrics.total_gpu_hit_rate
                    )
                if hasattr(logger.prefix_caching_metrics, "total_connector_hit_rate"):
                    metrics["connector_prefix_cache_total_hit_rate"] = (
                        logger.prefix_caching_metrics.total_connector_hit_rate
                    )
                if hasattr(logger, "cumulative_prompt_tokens"):
                    metrics["total_prompt_tokens"] = logger.cumulative_prompt_tokens
                if hasattr(logger, "cumulative_generation_tokens"):
                    metrics["total_generation_tokens"] = (
                        logger.cumulative_generation_tokens
                    )
                return metrics
    return None


def _get_kv_connector_metrics(llm: LLM) -> Optional[dict[str, float]]:
    try:
        from vllm.v1.metrics.loggers import LoggingStatLogger
    except Exception:
        return None

    if not hasattr(llm, "llm_engine"):
        return None
    manager = getattr(llm.llm_engine, "logger_manager", None)
    if manager is None:
        return None

    for engine_loggers in manager.per_engine_logger_dict.values():
        for logger in engine_loggers:
            if isinstance(logger, LoggingStatLogger):
                kv_logging = getattr(logger, "kv_transfer_logging", None)
                if kv_logging is None:
                    continue
                stats = kv_logging.transfer_stats_accumulator
                if stats is not None:
                    reduced = stats.reduce()
                    kv_logging.reset()
                    return _normalize_lmcache_metrics(reduced)

                last_stats = getattr(logger, "last_scheduler_stats", None)
                if last_stats and last_stats.kv_connector_stats:
                    return _normalize_lmcache_metrics(
                        last_stats.kv_connector_stats)
                return None
    return None


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
            if total_time_s > 0 else 0.0
        )
    return metrics


def _reset_kv_connector_metrics(llm: LLM) -> None:
    try:
        from vllm.v1.metrics.loggers import LoggingStatLogger
    except Exception:
        return

    if not hasattr(llm, "llm_engine"):
        return
    manager = getattr(llm.llm_engine, "logger_manager", None)
    if manager is None:
        return

    for engine_loggers in manager.per_engine_logger_dict.values():
        for logger in engine_loggers:
            if isinstance(logger, LoggingStatLogger):
                kv_logging = getattr(logger, "kv_transfer_logging", None)
                if kv_logging is not None:
                    kv_logging.reset()


@dataclasses.dataclass
class RunResult:
    ttft_ms: list[float]
    summary: dict[str, float]
    wall_time_s: float
    prefix_cache_metrics: Optional[dict[str, float]]
    lmcache_load: Optional[dict[str, float]]


def _run_and_collect(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> RunResult:
    _reset_kv_connector_metrics(llm)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    wall = time.time() - start
    ttfts = _ttft_ms_from_outputs(outputs)
    kv_metrics = _get_kv_connector_metrics(llm)
    return RunResult(
        ttft_ms=ttfts,
        summary=_summarize(ttfts),
        wall_time_s=wall,
        prefix_cache_metrics=_get_prefix_cache_metrics(llm),
        lmcache_load=kv_metrics,
    )


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


def _reset_gpu_prefix_cache(llm: LLM, retries: int = 5, delay_s: float = 0.25) -> bool:
    """Reset only the GPU prefix cache so LMCache DRAM entries remain.

    Retries with a small delay to allow pending frees to complete before
    attempting ``LLM.reset_prefix_cache`` again.
    """

    for attempt in range(1, max(retries, 1) + 1):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            reset_ret = llm.reset_prefix_cache()
        except Exception:
            reset_ret = False

        # API currently returns None on success; treat only explicit False as
        # failure. This avoids false negatives when backend logs success.
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


# ------------------------------- CLI logic ------------------------------- #


def create_argument_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="TTFT gap benchmark for prefix cache hits in HBM vs LMCache DRAM."
    )
    # 注意 num_prompts * input_length 不要超过 gpu kv cache 容量，否则 kv cache 会被刷掉
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument(
        "--input-length",
        type=int,
        required=True,
        help="Fixed prompt length (tokens) for warm/test pairs.",
    )
    parser.add_argument(
        "--hit-ratios",
        type=str,
        default="0.5,1.0",
        help="Comma-separated shared-prefix ratios (0-1) to test.",
    )
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="How many warmup passes to pre-populate GPU+LMCache before HBM run.",
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
        default="kv_cache_ttft_gap_results.json",
        help="Path to dump JSON results (default: kv_cache_ttft_gap_results.json).",
    )
    parser.add_argument(
        "--skip-hbm",
        action="store_true",
        help="Skip the HBM reuse phase (only LMCache).",
    )
    parser.add_argument(
        "--skip-lmcache",
        action="store_true",
        help="Skip the LMCache DRAM phase.",
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser


def _validate_input_length(val: int) -> int:
    if val < 1:
        raise ValueError("input-length must be >= 1")
    return val


# ------------------------------- main flow ------------------------------- #


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    random.seed(args.shuffle_seed)

    hit_ratios = _parse_hit_ratios(args.hit_ratios)
    input_length = _validate_input_length(args.input_length)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.disable_log_stats = False  # keep detailed metrics available
    llm = LLM(**dataclasses.asdict(engine_args))

    all_results = []

    for hit_ratio in hit_ratios:
        print(f"\n=== Hit ratio {hit_ratio:.2%} ===")
        warm_reqs, hbm_reqs, lmcache_reqs = _build_warm_test_triplets(
            tokenizer=tokenizer,
            num_prompts=args.num_prompts,
            input_length=input_length,
            hit_ratio=hit_ratio,
            seed=args.shuffle_seed,
        )
        warm_prompts = [r.prompt for r in warm_reqs]
        hbm_prompts = [r.prompt for r in hbm_reqs]
        lmcache_prompts = [r.prompt for r in lmcache_reqs]
        print(
            f"Pairs: n={len(hbm_prompts)}, fixed_len={input_length}, shared_prefix={int(input_length*hit_ratio)}"
        )

        hbm_result: Optional[RunResult] = None
        lmcache_result: Optional[RunResult] = None
        reset_ok: Optional[bool] = None

        # Always warm up once so LMCache persists the prefix regardless of which
        # phases are enabled. Warmup also fills HBM prior to the HBM run.
        _warmup(llm, warm_prompts, sampling_params, runs=max(args.warmup_runs, 1))

        if not args.skip_hbm:
            hbm_result = _run_and_collect(llm, hbm_prompts, sampling_params)
            hbm_result.lmcache_load = hbm_result.lmcache_load
            print(
                f"HBM TTFT mean={hbm_result.summary.get('mean_ms', float('nan')):.2f} ms, "
                f"p90={hbm_result.summary.get('p90_ms', float('nan')):.2f} ms"
            )
            if hbm_result.lmcache_load:
                lm_load = hbm_result.lmcache_load
                print(
                    "HBM run LMCache load: "
                    f"count={lm_load.get('lmcache_load_count', 0)}, "
                    f"mean={lm_load.get('lmcache_load_mean_time_ms', float('nan')):.2f} ms, "
                    f"bw={lm_load.get('lmcache_load_agg_bw_gbps', float('nan')):.2f} GB/s"
                )

        if not args.skip_lmcache:
            # Invalidate GPU prefix cache without touching LMCache so the next run
            # measures DRAM->HBM transfer cost. Only reset_prefix_cache is allowed.
            reset_ok = _reset_gpu_prefix_cache(llm)
            if not reset_ok:
                print("[WARN] reset_prefix_cache reported failure after retries; LMCache run may still see HBM hits.")

            lmcache_result = _run_and_collect(llm, lmcache_prompts, sampling_params)
            lmcache_result.lmcache_load = lmcache_result.lmcache_load
            print(
                f"LMCache TTFT mean={lmcache_result.summary.get('mean_ms', float('nan')):.2f} ms, "
                f"p90={lmcache_result.summary.get('p90_ms', float('nan')):.2f} ms"
            )
            if lmcache_result.lmcache_load:
                lm_load = lmcache_result.lmcache_load
                print(
                    "LMCache load: "
                    f"count={lm_load.get('lmcache_load_count', 0)}, "
                    f"mean={lm_load.get('lmcache_load_mean_time_ms', float('nan')):.2f} ms, "
                    f"bw={lm_load.get('lmcache_load_agg_bw_gbps', float('nan')):.2f} GB/s"
                )

        gap_ms = None
        if hbm_result and lmcache_result and hbm_result.summary and lmcache_result.summary:
            gap_ms = lmcache_result.summary["mean_ms"] - hbm_result.summary["mean_ms"]
            print(f"TTFT gap (LMCache - HBM): {gap_ms:.2f} ms")

        all_results.append(
            {
                "hit_ratio": hit_ratio,
                "prompt_stats": {
                    "count": len(hbm_prompts),
                    "len": input_length,
                    "shared_prefix": int(input_length * hit_ratio),
                },
                "hbm": dataclasses.asdict(hbm_result) if hbm_result else None,
                "lmcache": dataclasses.asdict(lmcache_result) if lmcache_result else None,
                "ttft_gap_ms": gap_ms,
                "lmcache_gpu_cache_reset_ok": reset_ok,
            }
        )

        # _cleanup_engine(llm)

    def _num(val: Optional[float]) -> str:
        return f"{val:.2f}" if val is not None else "-"

    def _summary_val(run: Optional[dict], key: str) -> Optional[float]:
        if not run:
            return None
        summary = run.get("summary")
        if not summary:
            return None
        return summary.get(key)

    header = (
        "hit",
        "hbm_wall_s",
        "lmcache_wall_s",
        "lm_load_ms",
        "lm_load_total_s",
        "lm_bw_gbps",
    )
    col_w = [8, 12, 15, 12, 15, 12]
    header_row = " ".join(h.ljust(w) for h, w in zip(header, col_w))
    print("\nSummary (per hit ratio):")
    print(header_row)
    print("-" * len(header_row))
    for res in all_results:
        hbm_run = res["hbm"]
        lm_run = res["lmcache"]
        hbm_wall = hbm_run["wall_time_s"] if hbm_run else None
        lm_wall = lm_run["wall_time_s"] if lm_run else None
        lm_load = lm_run.get("lmcache_load") if lm_run else None
        lm_load_ms = None
        lm_bw = None
        lm_load_total_s = None
        if lm_load:
            lm_load_ms = lm_load.get("lmcache_load_mean_time_ms")
            lm_load_total_s = lm_load.get("lmcache_load_total_time_s")
            lm_bw = lm_load.get("lmcache_load_agg_bw_gbps")

        row = (
            f"{res['hit_ratio']:.2f}",
            _num(hbm_wall),
            _num(lm_wall),
            _num(lm_load_ms),
            _num(lm_load_total_s),
            _num(lm_bw),
        )
        print(" ".join(val.ljust(w) for val, w in zip(row, col_w)))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
