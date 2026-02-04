# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 测试不同前缀长度下，仅使用高带宽内存（HBM）缓存的性能提升情况。

"""
Benchmark prefix caching speedups with HBM-only KV cache.

This script generates random prompts and, for each prefix length, primes the
HBM prefix cache using extracted prefix prompts. It then measures throughput
and TTFT on the full prompts. Results are written to files with a date suffix.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
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
    token_ids: list[int]


def _parse_prefix_lens(arg: str, input_length: int) -> list[int]:
    values: list[int] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError as exc:  # pragma: no cover - CLI guard
            raise ValueError(f"Invalid prefix length '{piece}'") from exc
        if value < 0 or value > input_length:
            raise ValueError(
                f"prefix_len must be in [0, input_length], got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one prefix length is required")
    return values


def _sample_tokens(
    tokenizer: PreTrainedTokenizerBase, length: int, rng: random.Random
) -> list[int]:
    vocab = tokenizer.get_vocab()
    specials = set(tokenizer.all_special_ids)
    candidates = [tid for _, tid in vocab.items() if tid not in specials]
    return rng.choices(candidates, k=length)


def _build_requests(
    tokenizer: PreTrainedTokenizerBase,
    num_prompts: int,
    input_length: int,
    seed: int,
) -> list[Request]:
    rng = random.Random(seed)
    requests: list[Request] = []
    for _ in range(num_prompts):
        token_ids = _sample_tokens(tokenizer, input_length, rng)
        requests.append(
            Request(
                prompt=tokenizer.decode(token_ids),
                prompt_len=len(token_ids),
                token_ids=token_ids,
            )
        )
    return requests


def _build_prefix_prompts(
    requests: list[Request], prefix_len: int, tokenizer: PreTrainedTokenizerBase
) -> list[str]:
    if prefix_len <= 0:
        return []
    prompts = []
    for req in requests:
        prefix_ids = req.token_ids[:prefix_len]
        prompts.append(tokenizer.decode(prefix_ids))
    return prompts


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


def _get_prefix_cache_metrics(llm: LLM) -> Optional[dict[str, float]]:
    try:
        from vllm.v1.metrics.loggers import LoggingStatLogger
    except Exception:
        return None

    logger_manager = getattr(llm.llm_engine, "logger_manager", None)
    if logger_manager is None:
        return None

    for engine_loggers in logger_manager.per_engine_logger_dict.values():
        for logger in engine_loggers:
            if isinstance(logger, LoggingStatLogger):
                metrics: dict[str, float] = {
                    "prefix_cache_hit_rate":
                    logger.prefix_caching_metrics.hit_rate,
                    "gpu_prefix_cache_hit_rate":
                    logger.prefix_caching_metrics.gpu_hit_rate,
                    "connector_prefix_cache_hit_rate":
                    logger.prefix_caching_metrics.connector_hit_rate,
                }
                return metrics
    return None


def _get_prefix_cache_totals(llm: LLM) -> Optional[dict[str, int]]:
    try:
        from vllm.v1.metrics.loggers import LoggingStatLogger
    except Exception:
        return None

    logger_manager = getattr(llm.llm_engine, "logger_manager", None)
    if logger_manager is None:
        return None

    for engine_loggers in logger_manager.per_engine_logger_dict.values():
        for logger in engine_loggers:
            if isinstance(logger, LoggingStatLogger):
                pcm = logger.prefix_caching_metrics
                return {
                    "total_queries": int(getattr(pcm, "total_queries", 0)),
                    "total_hits": int(getattr(pcm, "total_hits", 0)),
                    "total_gpu_hits": int(getattr(pcm, "total_gpu_hits", 0)),
                    "total_external_hits": int(
                        getattr(pcm, "total_external_hits", 0)),
                }
    return None


def _delta_prefix_cache_hit_rate(
    before: Optional[dict[str, int]],
    after: Optional[dict[str, int]],
) -> Optional[dict[str, float]]:
    if before is None or after is None:
        return None

    queries = max(0, after.get("total_queries", 0) -
                  before.get("total_queries", 0))
    hits = max(0, after.get("total_hits", 0) - before.get("total_hits", 0))
    gpu_hits = max(0, after.get("total_gpu_hits", 0) -
                   before.get("total_gpu_hits", 0))
    ext_hits = max(0, after.get("total_external_hits", 0) -
                   before.get("total_external_hits", 0))

    if queries == 0:
        rate = 0.0
        gpu_rate = 0.0
        ext_rate = 0.0
    else:
        rate = hits / queries
        gpu_rate = gpu_hits / queries
        ext_rate = ext_hits / queries

    return {
        "prefix_cache_hit_rate": rate,
        "gpu_prefix_cache_hit_rate": gpu_rate,
        "connector_prefix_cache_hit_rate": ext_rate,
    }


def _get_ttft_histogram_snapshot(llm: LLM) -> Optional[dict[str, object]]:
    if not hasattr(llm, "get_metrics"):
        return None
    try:
        metrics = llm.get_metrics()
        from vllm.v1.metrics.reader import Histogram
    except Exception:
        return None

    total_count = 0
    total_sum = 0.0
    buckets: dict[str, int] = {}
    for metric in metrics:
        if not isinstance(metric, Histogram):
            continue
        if metric.name != "vllm:time_to_first_token_seconds":
            continue
        total_count += metric.count
        total_sum += metric.sum
        for bucket, value in metric.buckets.items():
            buckets[bucket] = buckets.get(bucket, 0) + value

    if total_count == 0 and total_sum == 0.0 and not buckets:
        return None
    return {"count": total_count, "sum": total_sum, "buckets": buckets}


def _diff_histogram(
    after: Optional[dict[str, object]],
    before: Optional[dict[str, object]],
) -> Optional[dict[str, object]]:
    if after is None:
        return None
    if before is None:
        return after

    count = max(0, int(after["count"]) - int(before["count"]))
    sum_val = max(0.0, float(after["sum"]) - float(before["sum"]))
    buckets_after = dict(after["buckets"])
    buckets_before = dict(before["buckets"])

    buckets: dict[str, int] = {}
    for key in set(buckets_after.keys()) | set(buckets_before.keys()):
        buckets[key] = max(0, buckets_after.get(key, 0) -
                          buckets_before.get(key, 0))

    return {"count": count, "sum": sum_val, "buckets": buckets}


def _bucket_to_float(bucket: str) -> float:
    if bucket in {"+Inf", "Inf", "inf", "infinity", "+inf"}:
        return math.inf
    try:
        return float(bucket)
    except ValueError:
        return math.inf


def _percentile_from_histogram(
    buckets: dict[str, int],
    total: int,
    percentile: float,
) -> Optional[float]:
    if total <= 0:
        return None
    target = max(1, int(math.ceil(percentile * total)))

    ordered = sorted(
        ((_bucket_to_float(b), count) for b, count in buckets.items()),
        key=lambda item: item[0],
    )
    max_finite = max((b for b, _ in ordered if math.isfinite(b)),
                     default=0.0)

    for upper, count in ordered:
        if count >= target:
            return max_finite if math.isinf(upper) else upper
    return max_finite


def _ttft_summary_from_histogram(
    before: Optional[dict[str, object]],
    after: Optional[dict[str, object]],
) -> dict[str, float]:
    diff = _diff_histogram(after, before)
    if diff is None:
        return {}

    count = int(diff["count"])
    if count <= 0:
        return {}

    sum_val = float(diff["sum"])
    buckets = dict(diff["buckets"])

    mean_ms = (sum_val / count) * 1000.0
    p50 = _percentile_from_histogram(buckets, count, 0.5)
    p90 = _percentile_from_histogram(buckets, count, 0.9)
    p99 = _percentile_from_histogram(buckets, count, 0.99)

    summary = {
        "count": float(count),
        "mean": mean_ms,
    }
    if p50 is not None:
        summary["median"] = p50 * 1000.0
    if p90 is not None:
        summary["p90"] = p90 * 1000.0
    if p99 is not None:
        summary["p99"] = p99 * 1000.0
    return summary


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
        "mean": statistics.fmean(values_sorted),
        "median": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


@dataclasses.dataclass
class RunResult:
    wall_time_s: float
    ttft_ms: list[float]
    ttft_summary: dict[str, float]
    throughput_tok_s: float


def _run_and_collect(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    total_tokens: int,
) -> RunResult:
    ttft_hist_before = _get_ttft_histogram_snapshot(llm)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    wall = time.perf_counter() - start
    ttft_hist_after = _get_ttft_histogram_snapshot(llm)
    ttfts = _ttft_ms_from_outputs(outputs)
    ttft_summary = _summarize(ttfts)
    if not ttft_summary:
        # v1 offline generate does not attach per-request metrics to
        # RequestOutput, so fall back to histogram-derived TTFT.
        ttft_summary = _ttft_summary_from_histogram(ttft_hist_before,
                                                    ttft_hist_after)

    throughput = total_tokens / wall if wall > 0 else 0.0
    return RunResult(
        wall_time_s=wall,
        ttft_ms=ttfts,
        ttft_summary=ttft_summary,
        throughput_tok_s=throughput,
    )


def _reset_kv_cache(llm: LLM, retries: int = 5, delay_s: float = 0.25) -> bool:
    for attempt in range(1, max(retries, 1) + 1):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            reset_ret = llm.reset_prefix_cache()
        except Exception:
            reset_ret = False

        if reset_ret is not False:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return True

        if attempt < retries:
            time.sleep(delay_s)

    return False


def create_argument_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="Benchmark prefix lengths with HBM-only prefix caching."
    )
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument(
        "--prefix-lens",
        "--prefix_lens",
        dest="prefix_lens",
        type=str,
        default=None,
        help="Comma-separated prefix lengths to test.",
    )
    parser.add_argument(
        "--prefix-gap",
        type=int,
        default=None,
        help="If set, test prefix lengths from 0 to input-length in steps of this gap.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=1)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup passes to load model and stabilize performance (run once).",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help="Skip detokenization time in latency numbers.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="prefix_lens_only_hbm",
        help="Output filename prefix (date suffix is appended).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for result files.",
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser


def _format_num(val: Optional[float]) -> str:
    return f"{val:.2f}" if val is not None else "-"


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.input_length < 1:
        raise ValueError("input-length must be >= 1")

    if args.prefix_lens:
        prefix_lens = _parse_prefix_lens(args.prefix_lens, args.input_length)
    elif args.prefix_gap is not None:
        if args.prefix_gap <= 0:
            raise ValueError("prefix-gap must be > 0")
        prefix_lens = list(range(0, args.input_length + 1, args.prefix_gap))
    else:
        raise ValueError("Either --prefix-lens or --prefix-gap is required")
    random.seed(args.shuffle_seed)

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.disable_log_stats = False
    llm = LLM(**dataclasses.asdict(engine_args))

    results = []

    base_requests = _build_requests(
        tokenizer=tokenizer,
        num_prompts=args.num_prompts,
        input_length=args.input_length,
        seed=args.shuffle_seed,
    )
    base_prompts = [r.prompt for r in base_requests]

    for _ in range(max(args.warmup_runs, 1)):
        llm.generate(base_prompts, sampling_params=sampling_params)

    _reset_kv_cache(llm)

    for prefix_len in prefix_lens:
        print(f"\n=== Prefix length {prefix_len} ===")
        reset_ok = _reset_kv_cache(llm)
        if not reset_ok:
            print("[WARN] reset_prefix_cache reported failure; results may be noisy.")

        prompts = base_prompts
        prefix_prompts = _build_prefix_prompts(
            requests=base_requests,
            prefix_len=prefix_len,
            tokenizer=tokenizer,
        )
        total_tokens = (args.input_length + args.output_len) * len(prompts)

        if prefix_prompts:
            llm.generate(prefix_prompts, sampling_params=sampling_params)

        # kv cache 命中率统计了前缀输入的影响，因此在测量正式输出前获取统计数据快照。
        prefix_cache_before = _get_prefix_cache_totals(llm)

        result = _run_and_collect(
            llm=llm,
            prompts=prompts,
            sampling_params=sampling_params,
            total_tokens=total_tokens,
        )
        prefix_cache_after = _get_prefix_cache_totals(llm)
        prefix_cache_metrics = _delta_prefix_cache_hit_rate(
            prefix_cache_before, prefix_cache_after)

        print(
            "TTFT mean="
            f"{result.ttft_summary.get('mean', float('nan')):.2f} ms"
        )
        print(
            "TTFT p90="
            f"{result.ttft_summary.get('p90', float('nan')):.2f} ms"
        )
        print(
            "Throughput="
            f"{result.throughput_tok_s:.2f} tok/s"
        )
        if prefix_cache_metrics:
            print(
                "Prefix cache hit rate="
                f"{prefix_cache_metrics.get('prefix_cache_hit_rate', 0.0) * 100:.2f}%"
            )
            if "gpu_prefix_cache_hit_rate" in prefix_cache_metrics:
                print(
                    "  GPU hit rate="
                    f"{prefix_cache_metrics['gpu_prefix_cache_hit_rate'] * 100:.2f}%"
                )
            if "connector_prefix_cache_hit_rate" in prefix_cache_metrics:
                print(
                    "  Connector hit rate="
                    f"{prefix_cache_metrics['connector_prefix_cache_hit_rate'] * 100:.2f}%"
                )

        results.append(
            {
                "prefix_len": prefix_len,
                "input_length": args.input_length,
                "num_prompts": args.num_prompts,
                "result": dataclasses.asdict(result),
                "kv_cache_reset_ok": reset_ok,
                "prefix_cache_metrics": prefix_cache_metrics,
            }
        )

    header = (
        "prefix",
        "throughput",
        "wall_time_s",
        "ttft_mean",
        "prefix_cache_hit_rate",
    )
    col_w = [8, 14, 12, 12, 22]
    header_row = " ".join(h.ljust(w) for h, w in zip(header, col_w))
    print("\nSummary:")
    print(header_row)
    print("-" * len(header_row))

    for res in results:
        result = res["result"]
        row = (
            str(res["prefix_len"]),
            _format_num(result.get("throughput_tok_s")),
            _format_num(result.get("wall_time_s")),
            _format_num(result.get("ttft_summary", {}).get("mean")),
            _format_num(
                (res.get("prefix_cache_metrics") or {}).get(
                    "prefix_cache_hit_rate")),
        )
        print(" ".join(val.ljust(w) for val, w in zip(row, col_w)))

    date_suffix = _dt.datetime.now().strftime("%Y%m%d")
    json_path = f"{args.output_prefix}_{date_suffix}.json"
    csv_path = f"{args.output_prefix}_{date_suffix}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "prefix_len,throughput_tok_s,wall_time_s,ttft_mean_ms,"
            "kv_cache_reset_ok,prefix_cache_hit_rate\n"
        )
        for res in results:
            result = res["result"]
            prefix_metrics = res.get("prefix_cache_metrics") or {}
            f.write(
                f"{res['prefix_len']},"
                f"{result.get('throughput_tok_s', 0.0):.6f},"
                f"{result.get('wall_time_s', 0.0):.6f},"
                f"{result.get('ttft_summary', {}).get('mean', 0.0):.6f},"
                f"{res.get('kv_cache_reset_ok', False)},"
                f"{prefix_metrics.get('prefix_cache_hit_rate', 0.0):.6f}\n"
            )

    print(f"\nSaved results to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()