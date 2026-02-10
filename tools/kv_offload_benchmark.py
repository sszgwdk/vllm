# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import random
from dataclasses import dataclass
from typing import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.gflru_manager import GFLRUOffloadingManager
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager


@dataclass
class BenchmarkResult:
    name: str
    hits: int
    total: int
    stores: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def offload_ratio(self) -> float:
        return self.stores / self.total if self.total else 0.0


def to_hash(block_id: int) -> BlockHash:
    return BlockHash(str(block_id).encode())


def build_accesses(total: int, hotset_size: int, coldset_size: int,
                   hot_prob: float, rng: random.Random) -> Iterable[int]:
    hot = list(range(hotset_size))
    cold = list(range(hotset_size, hotset_size + coldset_size))
    for _ in range(total):
        if rng.random() < hot_prob:
            yield rng.choice(hot)
        else:
            yield rng.choice(cold)


def run_benchmark(name: str, manager, accesses: Iterable[int]) -> BenchmarkResult:
    hits = 0
    stores = 0
    evictions = 0
    total = 0
    for block_id in accesses:
        total += 1
        block_hash = to_hash(block_id)
        if manager.lookup([block_hash]) == 1:
            hits += 1
            manager.prepare_load([block_hash])
            manager.complete_load([block_hash])
            manager.touch([block_hash])
            continue

        store_output = manager.prepare_store([block_hash])
        if store_output is None:
            continue
        if store_output.block_hashes_to_store:
            stores += len(store_output.block_hashes_to_store)
            evictions += len(store_output.block_hashes_evicted)
            manager.complete_store(store_output.block_hashes_to_store)
        manager.touch([block_hash])

    return BenchmarkResult(name=name,
                           hits=hits,
                           total=total,
                           stores=stores,
                           evictions=evictions)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        "Policy        Hits   Total  HitRate  OffloadRatio  Evictions\n"
        "------------  -----  -----  -------  ------------  ---------"
    )
    print(header)
    for result in results:
        print(
            f"{result.name:<12}"
            f"  {result.hits:>5}"
            f"  {result.total:>5}"
            f"  {format_percent(result.hit_rate):>7}"
            f"  {format_percent(result.offload_ratio):>12}"
            f"  {result.evictions:>9}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small benchmark for LRU vs GFLRU offloading managers.")
    parser.add_argument("--total-accesses", type=int, default=5000)
    parser.add_argument("--hotset-size", type=int, default=128)
    parser.add_argument("--coldset-size", type=int, default=1024)
    parser.add_argument("--hot-prob", type=float, default=0.8)
    parser.add_argument("--capacity-blocks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    accesses = list(
        build_accesses(args.total_accesses, args.hotset_size,
                       args.coldset_size, args.hot_prob, rng))

    lru_backend = CPUBackend(block_size=256, num_blocks=args.capacity_blocks)
    gflru_backend = CPUBackend(block_size=256, num_blocks=args.capacity_blocks)

    lru_manager = LRUOffloadingManager(lru_backend, enable_events=False)
    gflru_manager = GFLRUOffloadingManager(gflru_backend, enable_events=False)

    results = [
        run_benchmark("LRU", lru_manager, accesses),
        run_benchmark("GFLRU", gflru_manager, accesses),
    ]
    print_results(results)


if __name__ == "__main__":
    main()
