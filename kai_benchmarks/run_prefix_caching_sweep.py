#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sequentially run benchmark_prefix_caching with different cache configs.

This script runs benchmark_prefix_caching.py for multiple combinations of
layer-wise cache, cold-hot LRU cache, and optional LMCACHE_CACHE_POLICY
settings, captures the stdout for each run, parses key metrics, and prints a
compact summary.
"""

from __future__ import annotations

import argparse
import json
import os
import errno
import pty
import re
import shlex
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_COMPILATION_CONFIG = '{"level":0,"cudagraph_mode":"NONE"}'
DEFAULT_KV_TRANSFER_CONFIG = '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'


@dataclass
class RunConfig:
    layerwise: int
    cold_hot_lru: bool
    cache_policy_enabled: bool


@dataclass
class RunResult:
    config: RunConfig
    exit_code: int
    metrics: Dict[str, Any]
    stdout: str


def parse_layerwise_options(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in {"0", "1"}:
            raise argparse.ArgumentTypeError("layerwise options must be 0 or 1")
        values.append(int(item))
    return values or [0, 1]


def parse_lru_options(raw: str) -> List[bool]:
    mapping = {"on": True, "off": False, "1": True, "0": False, "true": True, "false": False}
    values = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise argparse.ArgumentTypeError("cold-hot-lru options must be on/off or 1/0")
        values.append(mapping[key])
    return values or [True, False]


def parse_cache_policy_options(raw: str) -> List[bool]:
    mapping = {"on": True, "off": False, "1": True, "0": False, "true": True, "false": False}
    values: List[bool] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise argparse.ArgumentTypeError("cache-policy options must be on/off or 1/0")
        values.append(mapping[key])
    return values or [True, False]


def build_base_command(args: argparse.Namespace, bench_script: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(bench_script),
        "--model",
        args.model,
        "--dataset-path",
        args.dataset_path,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--compilation-config",
        args.compilation_config,
        "--kv-transfer-config",
        args.kv_transfer_config,
        "--num-prompts",
        str(args.num_prompts),
        "--input-length-range",
        args.input_length_range,
        "--output-len",
        str(args.output_len),
    ]

    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if args.use_zipf:
        cmd.append("--use-zipf")
        cmd.extend(["--zipf-scale", str(args.zipf_scale)])
    else:
        cmd.extend(["--repeat-count", str(args.repeat_count)])
    if args.enable_sort:
        cmd.append("--sort")
    if args.shuffle_seed is not None:
        cmd.extend(["--shuffle-seed", str(args.shuffle_seed)])
    if args.prefix_len is not None:
        cmd.extend(["--prefix-len", str(args.prefix_len)])
    if args.disable_detokenize:
        cmd.append("--disable-detokenize")
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def run_once(
    base_cmd: List[str],
    config: RunConfig,
    env_base: Dict[str, str],
    repo_root: Path,
    show_progress: bool,
) -> RunResult:
    cmd = list(base_cmd)
    if config.cold_hot_lru:
        cmd.append("--enable-cold-hot-lru-cache")

    env = env_base.copy()
    env["LMCACHE_USE_LAYERWISE"] = str(config.layerwise)
    if config.cache_policy_enabled:
        env["LMCACHE_CACHE_POLICY"] = env_base.get("LMCACHE_CACHE_POLICY", "FIFO_REINSERTION")
    else:
        env.pop("LMCACHE_CACHE_POLICY", None)

    if show_progress:
        # Use a PTY so tqdm-like progress bars are not disabled (isatty stays True).
        master_fd, slave_fd = pty.openpty()
        output_chunks: List[bytes] = []
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,
                close_fds=True,
            )
            os.close(slave_fd)

            while True:
                rlist, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in rlist:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError as exc:  # EOF on some platforms raises EIO
                        if exc.errno == errno.EIO:
                            break
                        raise
                    if not data:
                        break
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()
                    output_chunks.append(data)
                if proc.poll() is not None and not rlist:
                    break
            proc.wait()
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass

        stdout = b"".join(output_chunks).decode(errors="replace")
        exit_code = proc.returncode
    else:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout = proc.stdout
        exit_code = proc.returncode

    metrics = parse_metrics(stdout)
    return RunResult(config=config, exit_code=exit_code, metrics=metrics, stdout=stdout)


def parse_metrics(output: str) -> Dict[str, Any]:
    patterns = {
        "cost_seconds": r"cost time ([0-9.]+)s",
        "prefix_cache_hit_rate_total": r"Prefix cache hit rate \(total\): ([0-9.]+)%",
        "gpu_prefix_cache_hit_rate_total": r"GPU hit rate \(total\): ([0-9.]+)%",
        "connector_prefix_cache_hit_rate_total": r"Connector hit rate \(total\): ([0-9.]+)%",
        "total_prompt_tokens": r"Total prompt tokens: ([0-9]+)",
        "total_generation_tokens": r"Total generation tokens: ([0-9]+)",
        "total_tokens": r"Total tokens: ([0-9]+)",
        "avg_throughput": r"Average throughput: ([0-9.]+) tokens/s",
    }

    metrics: Dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            metrics[key] = float(value) if "." in value else int(value)
    return metrics


def print_summary(results: List[RunResult]) -> None:
    headers = [
        "layerwise",
        "cold_hot_lru",
        "cache_policy",
        "exit",
        "cost_s",
        "hit_total_%",
        "gpu_hit_%",
        "conn_hit_%",
        "throughput",
    ]

    rows: List[List[str]] = []
    for r in results:
        m = r.metrics
        rows.append([
            str(r.config.layerwise),
            "on" if r.config.cold_hot_lru else "off",
            "on" if r.config.cache_policy_enabled else "off",
            str(r.exit_code),
            fmt(m.get("cost_seconds")),
            fmt(m.get("prefix_cache_hit_rate_total")),
            fmt(m.get("gpu_prefix_cache_hit_rate_total")),
            fmt(m.get("connector_prefix_cache_hit_rate_total")),
            fmt(m.get("avg_throughput")),
        ])

    widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    def line(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("\n===== Summary =====")
    print(line(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(line(row))


def fmt(value: Optional[Any]) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VLLM_USE_MODELSCOPE", "true")
    env.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", str(args.max_local_cpu_size))
    env.setdefault("LMCACHE_LOG_LEVEL", args.lmcache_log_level)
    env["LMCACHE_CACHE_POLICY"] = args.cache_policy_value
    if args.vllm_logging_level:
        env.setdefault("VLLM_LOGGING_LEVEL", args.vllm_logging_level)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch prefix caching benchmark sweeps")
    parser.add_argument("--model", default="mistralai/Mistral-Small-24B-Instruct-2501")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--compilation-config", default=DEFAULT_COMPILATION_CONFIG)
    parser.add_argument("--kv-transfer-config", default=DEFAULT_KV_TRANSFER_CONFIG)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--use-zipf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zipf-scale", type=int, default=3)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-length-range", default="1024:8192")
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--enable-sort", action="store_true")
    parser.add_argument("--disable-detokenize", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=1)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--layerwise-options", type=parse_layerwise_options, default="0,1",
                        help="Comma list of layerwise settings to sweep (0,1)")
    parser.add_argument("--cold-hot-lru-options", type=parse_lru_options, default="on,off",
                        help="Comma list of cold-hot-lru settings to sweep (on,off)")
    parser.add_argument("--cache-policy-options", type=parse_cache_policy_options, default="on,off",
                        help="Comma list to toggle LMCACHE_CACHE_POLICY per run (on,off)")
    parser.add_argument("--cache-policy-value", default="FIFO_REINSERTION",
                        help="Value assigned to LMCACHE_CACHE_POLICY when enabled")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a PTY to preserve tqdm-style progress bars during sweeps (default: on)")
    parser.add_argument("--extra-args", default="", help="Extra args appended to benchmark command")
    parser.add_argument("--max-local-cpu-size", type=float, default=150.0)
    parser.add_argument("--lmcache-log-level", default="WARNING")
    parser.add_argument("--vllm-logging-level", default=None)
    parser.add_argument("--output-json", default=None, help="Optional path to dump raw results JSON")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    bench_script = repo_root / "kai_benchmarks" / "benchmark_prefix_caching.py"
    if not bench_script.exists():
        raise FileNotFoundError(f"Missing benchmark script at {bench_script}")

    env_base = build_env(args)
    base_cmd = build_base_command(args, bench_script)

    sweep: List[RunConfig] = [
        RunConfig(layerwise=l, cold_hot_lru=c, cache_policy_enabled=p)
        for l in args.layerwise_options
        for c in args.cold_hot_lru_options
        for p in args.cache_policy_options
    ]

    print("Base command:")
    print(" ".join(shlex.quote(part) for part in base_cmd))
    print("Env overrides (per-run LMCACHE_USE_LAYERWISE is set by sweep):")
    env_keys = [
        "VLLM_USE_MODELSCOPE",
        "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "LMCACHE_LOG_LEVEL",
        "VLLM_LOGGING_LEVEL",
        "LMCACHE_CACHE_POLICY",
    ]
    for key in env_keys:
        if key in env_base:
            print(f"  {key}={env_base[key]}")

    results: List[RunResult] = []
    for idx, config in enumerate(sweep, start=1):
        print(
            f"\n=== Run {idx}/{len(sweep)} | layerwise={config.layerwise} | "
            f"cold_hot_lru={'on' if config.cold_hot_lru else 'off'} | "
            f"cache_policy={'on' if config.cache_policy_enabled else 'off'} ==="
        )
        cmd_with_flags = list(base_cmd)
        if config.cold_hot_lru:
            cmd_with_flags.append("--enable-cold-hot-lru-cache")
        print("Command:")
        print(" ".join(shlex.quote(part) for part in cmd_with_flags))
        if args.dry_run:
            continue
        result = run_once(base_cmd, config, env_base, repo_root, args.show_progress)
        results.append(result)
        print("Exit code:", result.exit_code)
        print("Captured metrics:", json.dumps(result.metrics, indent=2))
        if result.exit_code != 0:
            print("--- stdout/stderr ---")
            print(result.stdout)

    if args.dry_run:
        print("Dry-run only; no benchmarks executed.")
        return

    print_summary(results)

    if args.output_json:
        payload = [
            {
                "layerwise": r.config.layerwise,
                "cold_hot_lru": r.config.cold_hot_lru,
                "cache_policy_enabled": r.config.cache_policy_enabled,
                "exit_code": r.exit_code,
                "metrics": r.metrics,
                "stdout": r.stdout,
            }
            for r in results
        ]
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON results to {args.output_json}")


if __name__ == "__main__":
    main()
