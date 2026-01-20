# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
import copy
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LMCacheKVConnectorStats(KVConnectorStats):
    """Container for LMCache load metrics."""

    def __post_init__(self):
        if "lmcache_load_count" not in self.data:
            self.data["lmcache_load_count"] = 0
        if "lmcache_load_total_time_s" not in self.data:
            self.data["lmcache_load_total_time_s"] = 0.0
        if "lmcache_load_total_bytes" not in self.data:
            self.data["lmcache_load_total_bytes"] = 0

    def reset(self):
        self.data = {
            "lmcache_load_count": 0,
            "lmcache_load_total_time_s": 0.0,
            "lmcache_load_total_bytes": 0,
        }

    def clone_and_reset(self) -> "LMCacheKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return self.data.get("lmcache_load_count", 0) == 0

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            self.data["lmcache_load_count"] += other.data.get(
                "lmcache_load_count", 0)
            self.data["lmcache_load_total_time_s"] += other.data.get(
                "lmcache_load_total_time_s", 0.0)
            self.data["lmcache_load_total_bytes"] += other.data.get(
                "lmcache_load_total_bytes", 0)
        return self

    def reduce(self) -> dict[str, Union[int, float]]:
        count = self.data.get("lmcache_load_count", 0)
        total_time_s = self.data.get("lmcache_load_total_time_s", 0.0)
        total_bytes = self.data.get("lmcache_load_total_bytes", 0)

        mean_time_ms = (total_time_s / count * 1000.0) if count > 0 else 0.0
        agg_bw_gbps = (
            (total_bytes / total_time_s) / (1024**3)
            if total_time_s > 0 else 0.0
        )

        return {
            "lmcache_load_count": count,
            "lmcache_load_total_time_s": total_time_s,
            "lmcache_load_total_bytes": total_bytes,
            "lmcache_load_mean_time_ms": mean_time_ms,
            "lmcache_load_agg_bw_gbps": agg_bw_gbps,
        }


class LMCacheConnectorV1(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._lmcache_engine = LMCacheConnectorV1Impl(vllm_config, role, self)

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
            
        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata",
                      **kwargs: Any) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(layer_name, kv_layer, attn_metadata,
                                           **kwargs)

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_kv_connector_stats(self) -> Optional[KVConnectorStats]:
        stats = self._lmcache_engine.get_kv_connector_stats()
        if stats is None:
            return LMCacheKVConnectorStats()
        return LMCacheKVConnectorStats(data=stats)

    @classmethod
    def build_kv_connector_stats(
            cls,
            data: Optional[dict[str, Any]] = None) -> Optional[KVConnectorStats]:
        return LMCacheKVConnectorStats(data=data) if data is not None \
            else LMCacheKVConnectorStats()

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the 
            external KV cache beyond what is already computed.
        """
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens), False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(request,
                                                      num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self._lmcache_engine.request_finished(request, block_ids)
