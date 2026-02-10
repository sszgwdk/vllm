# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec, PrepareStoreOutput
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.gflru_manager import GFLRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


@dataclass
class ExpectedPrepareStoreOutput:
    block_hashes_to_store: list[int]
    store_block_ids: list[int]
    block_hashes_evicted: list[int]


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


def verify_store_output(
        prepare_store_output: Optional[PrepareStoreOutput],
        expected_prepare_store_output: ExpectedPrepareStoreOutput):
    assert prepare_store_output is not None
    assert (prepare_store_output.block_hashes_to_store == to_hashes(
        expected_prepare_store_output.block_hashes_to_store))
    assert (prepare_store_output.block_hashes_evicted == to_hashes(
        expected_prepare_store_output.block_hashes_evicted))
    store_spec = prepare_store_output.store_spec
    assert isinstance(store_spec, CPULoadStoreSpec)
    expected_array = np.array(expected_prepare_store_output.store_block_ids,
                              dtype=np.int64)
    assert np.array_equal(expected_array, store_spec.block_ids)


def verify_load_output(prepare_load_output: LoadStoreSpec,
                       expected_prepare_load_output: list[int]):
    assert isinstance(prepare_load_output, CPULoadStoreSpec)
    expected_array = np.array(expected_prepare_load_output, dtype=np.int64)
    assert np.array_equal(expected_array, prepare_load_output.block_ids)


def test_gflru_manager_admission():
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=2)
    cpu_manager = GFLRUOffloadingManager(cpu_backend, enable_events=True)

    # ghost empty -> admit [1]
    prepare_store_output = cpu_manager.prepare_store(to_hashes([1]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[1],
            store_block_ids=[0],
            block_hashes_evicted=[],
        ))
    cpu_manager.complete_store(to_hashes([1]))

    # ghost empty -> admit [2]
    prepare_store_output = cpu_manager.prepare_store(to_hashes([2]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[2],
            store_block_ids=[1],
            block_hashes_evicted=[],
        ))
    cpu_manager.complete_store(to_hashes([2]))

    # ghost empty -> admit [3], evict LRU [1]
    prepare_store_output = cpu_manager.prepare_store(to_hashes([3]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[3],
            store_block_ids=[0],
            block_hashes_evicted=[1],
        ))
    cpu_manager.complete_store(to_hashes([3]))

    # ghost not empty, [4] not in ghost -> reject
    prepare_store_output = cpu_manager.prepare_store(to_hashes([4]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[],
            store_block_ids=[],
            block_hashes_evicted=[],
        ))

    # ghost hit -> admit [1], evict LRU [2]
    prepare_store_output = cpu_manager.prepare_store(to_hashes([1]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[1],
            store_block_ids=[1],
            block_hashes_evicted=[2],
        ))
    cpu_manager.complete_store(to_hashes([1]))

    # load [3]
    prepare_load_output = cpu_manager.prepare_load(to_hashes([3]))
    verify_load_output(prepare_load_output, [0])
