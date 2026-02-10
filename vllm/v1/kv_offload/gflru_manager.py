# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable
from typing import Optional

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (LoadStoreSpec, OffloadingEvent,
                                         OffloadingManager, PrepareStoreOutput)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class _GhostCache:
    """FIFO ghost cache for recently evicted block hashes."""

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self._queue: OrderedDict[BlockHash, None] = OrderedDict()

    def add(self, block_hash: BlockHash) -> None:
        if block_hash in self._queue:
            return
        self._queue[block_hash] = None
        while len(self._queue) > self.capacity:
            self._queue.popitem(last=False)

    def remove(self, block_hash: BlockHash) -> None:
        self._queue.pop(block_hash, None)

    def exists(self, block_hash: BlockHash) -> bool:
        return block_hash in self._queue

    def is_empty(self) -> bool:
        return not self._queue


class GFLRUOffloadingManager(OffloadingManager):
    """
    OffloadingManager using a Ghost-First LRU admission policy.

    Two-tier logic is represented as:
    - HBM: GPU-resident blocks (implicit, not stored in this manager)
    - DRAM: CPU offloaded blocks (tracked here with LRU + ghost filter)

    The ghost cache records recently evicted CPU blocks. A block is admitted
    to CPU offload only if it is in the ghost cache or if the ghost cache is
    empty, matching the GFLRU two-tier policy.
    """

    def __init__(self,
                 backend: Backend,
                 enable_events: bool = False,
                 ghost_capacity_multiplier: int = 4):
        self.backend: Backend = backend
        # block_hash -> BlockStatus (DRAM tier only)
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.events: Optional[list[OffloadingEvent]] = \
            [] if enable_events else None

        ghost_capacity = self.backend.get_num_free_blocks(
        ) * ghost_capacity_multiplier
        self.ghost = _GhostCache(ghost_capacity)

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)
        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for block_hash in reversed(list(block_hashes)):
            if self.blocks.get(block_hash):
                self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def _evict_blocks(self, num_to_evict: int) -> Optional[list[BlockHash]]:
        to_evict: list[BlockHash] = []
        if num_to_evict <= 0:
            return to_evict

        for block_hash, block in self.blocks.items():
            if block.ref_cnt == 0:
                to_evict.append(block_hash)
                num_to_evict -= 1
                if num_to_evict == 0:
                    break
        else:
            return None

        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))
            self.ghost.add(block_hash)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(block_hashes=to_evict,
                                block_size=self.backend.block_size,
                                medium=self.backend.medium,
                                removed=True))
        return to_evict

    def prepare_store(
            self,
            block_hashes: Iterable[BlockHash]) -> Optional[PrepareStoreOutput]:
        block_hashes_to_store: list[BlockHash] = []
        for block_hash in block_hashes:
            if block_hash in self.blocks:
                continue

            if self.ghost.exists(block_hash) or self.ghost.is_empty():
                self.ghost.remove(block_hash)
                block_hashes_to_store.append(block_hash)
            else:
                self.ghost.add(block_hash)

        num_blocks_to_evict = (len(block_hashes_to_store) -
                               self.backend.get_num_free_blocks())
        to_evict = self._evict_blocks(num_blocks_to_evict)
        if to_evict is None:
            return None

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)
        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block

        store_spec = self.backend.get_load_store_spec(block_hashes_to_store,
                                                      blocks)
        return PrepareStoreOutput(block_hashes_to_store=block_hashes_to_store,
                                  store_spec=store_spec,
                                  block_hashes_evicted=to_evict)

    def complete_store(self,
                       block_hashes: Iterable[BlockHash],
                       success: bool = True):
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(block_hashes=stored_block_hashes,
                                block_size=self.backend.block_size,
                                medium=self.backend.medium,
                                removed=False))

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
