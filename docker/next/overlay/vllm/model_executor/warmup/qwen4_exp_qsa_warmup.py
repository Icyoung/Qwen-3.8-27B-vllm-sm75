# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect loaded Qwen4Exp QSA modules to their kernel-owned warmup."""

import sys
from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """Warm every reachable QSA specialization: indexer decode-query-length
    profiles plus the sparse attention split-K/merge configs."""

    qsa_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.indexer_qsa")
    attn_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa")
    if qsa_module is None or attn_module is None:
        return
    indexer = None
    owner = None
    for layer in worker.get_model().modules():
        if indexer is None and isinstance(layer, qsa_module.QSAIndexer):
            indexer = layer
        elif owner is None and isinstance(layer, attn_module.Qwen4ExpQSAAttention):
            owner = layer
    if indexer is None or owner is None:
        return

    runner = worker.model_runner

    def block_table_for(prefix: str) -> torch.Tensor:
        group_id = next(
            i
            for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            if prefix in group.layer_names
        )
        if worker.use_v2_model_runner:
            runner_v2 = cast("GPUModelRunnerV2", runner)
            return runner_v2.block_tables.input_block_tables[group_id]
        return runner.input_batch.block_table[group_id].get_device_tensor(
            runner.max_num_reqs
        )

    if worker.use_v2_model_runner:
        max_decode_query_len = cast("GPUModelRunnerV2", runner).decode_query_len
    else:
        max_decode_query_len = runner.uniform_decode_query_len

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        warmup_qsa_mqa_paged_decode,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    assert k_cache.numel()
    profiles = warmup_qsa_mqa_paged_decode(
        k_cache,
        block_table_for(indexer.compressed_key_cache.prefix),
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        max_decode_query_len=max_decode_query_len,
        max_num_reqs=runner.max_num_reqs,
        max_num_batched_tokens=runner.max_num_tokens,
    )
    logger.info("Warmed up Qwen4Exp QSA decode kernels: %s.", profiles)

    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        warmup_qsa_sparse_paged_attention,
    )

    kv_cache = owner.kv_cache
    assert kv_cache.numel()
    attention_profiles = warmup_qsa_sparse_paged_attention(
        kv_cache,
        block_table_for(owner.layer_name),
        num_query_heads=owner.num_heads,
        selection_width=indexer.output_width,
    )
    logger.info(
        "Warmed up Qwen4Exp QSA sparse attention kernels: %s.",
        attention_profiles,
    )

    from vllm.platforms import current_platform
    import os
    if (current_platform.is_device_capability(75)
            and owner.num_heads == 3 and owner.head_dim == 256
            and owner.kv_cache_torch_dtype == torch.float16
            and os.environ.get("VLLM_SM75_QSA_FLASHINFER_PREFILL", "1") == "1"):
        from vllm.models.qwen4_exp.nvidia.ops.qsa_flashinfer_prefill import warmup
        warmup()
        logger.info("SM75 QSA exact-selection FlashInfer Prefill startup warmup complete.")

    if (current_platform.is_device_capability(75)
            and owner.num_heads == 3 and owner.head_dim == 256
            and owner.kv_cache_torch_dtype == torch.float16
            and os.environ.get("VLLM_SM75_QSA_SELECTED_PAGES_PREFILL", "1") == "1"):
        from vllm.models.qwen4_exp.nvidia.ops.qsa_selected_prefill import warmup as warmup_selected
        warmup_selected()
        logger.info("SM75 QSA exact selected-page long Prefill startup warmup complete.")

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer_cublas import warmup_qsa_cublas_indexer
    if warmup_qsa_cublas_indexer(k_cache.device, k_cache.shape[1]):
        logger.info("SM75 QSA FP32-output cuBLAS indexer startup warmup complete.")

    if (current_platform.is_device_capability(75) and owner.kv_cache_torch_dtype == torch.uint8 and os.environ.get("VLLM_SM75_QSA_FP8_MAIN_KV", "0") == "1"):
        from vllm.models.qwen4_exp.nvidia.ops.qsa_selected_prefill_fp8 import warmup as warmup_fp8_selected
        warmup_fp8_selected()
        logger.info("SM75 QSA direct FP8 selected-page Prefill startup warmup complete.")
