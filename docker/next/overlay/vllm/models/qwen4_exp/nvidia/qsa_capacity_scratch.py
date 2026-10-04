# SPDX-License-Identifier: Apache-2.0
"""Opt-in scratch reuse for the sequential target QSA layers on SM75.

The target forward consumes a layer's indices before the next layer writes
them. CUDA kernels are issued on the same execution stream, so the write/read
ordering is retained in eager execution and CUDA graph replay. MTP layers are
created by a different model class and never call this helper: their step-0
selection remains private for speculative steps and compaction.
"""
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def share_target_qsa_scratch(model, vllm_config):
    if os.environ.get("VLLM_SM75_QSA_SHARED_TARGET_SCRATCH", "0") != "1":
        return
    parallel = vllm_config.parallel_config
    mode = getattr(vllm_config.compilation_config.mode, "value",
                   vllm_config.compilation_config.mode)
    if int(mode) != 0:
        raise RuntimeError("QSA scratch candidate requires compilation mode 0")
    if int(parallel.pipeline_parallel_size) != 1:
        raise RuntimeError("QSA scratch candidate requires pipeline size 1")
    if bool(getattr(parallel, "enable_dbo", False)):
        raise RuntimeError("QSA scratch candidate does not support DBO")
    # This is deliberately model-instance-local. No process-global pool can
    # leak aliasing between target, draft, reloaded models, or engine instances.
    owners = []
    for layer in model.layers:
        attention = getattr(layer, "self_attn", None)
        indexer = getattr(attention, "indexer", None)
        if indexer is not None:
            if indexer.skip_topk:
                raise RuntimeError("MTP/skip-topk owner cannot share target scratch")
            owners.append(attention)
    if not owners:
        return
    topk = owners[0].topk_indices_buffer
    if not topk.is_cuda or torch.cuda.get_device_capability(topk.device) != (7, 5):
        raise RuntimeError("QSA scratch candidate requires SM75 CUDA construction")
    if topk.dtype != torch.int32 or topk.ndim != 2:
        raise RuntimeError("Unexpected target QSA selection storage")
    first = owners[0].indexer
    block_width = first.token_topk // first.compress_ratio
    block_indices = topk.new_empty((topk.shape[0], block_width))
    bytes_before = 0
    for attention in owners:
        current = attention.topk_indices_buffer
        indexer = attention.indexer
        if current.shape != topk.shape or current.dtype != topk.dtype:
            raise RuntimeError("Target QSA scratch shapes must be identical")
        if current.device != topk.device:
            raise RuntimeError("Target QSA scratch devices must be identical")
        if indexer.token_topk // indexer.compress_ratio != block_width:
            raise RuntimeError("Target QSA selection widths must be identical")
        bytes_before += current.numel() * current.element_size()
        # Rebinding an existing nn.Module buffer preserves buffer registration.
        attention.topk_indices_buffer = topk
        indexer.register_buffer("_sm75_shared_block_indices", block_indices,
                                persistent=False)
    bytes_after = topk.numel() * topk.element_size() + block_indices.numel() * 4
    logger.info("SM75 shared target QSA scratch: layers=%d, rows=%d, "
                "topk_unique=1, block_unique=1, persistent_delta_bytes=%d; "
                "MTP buffers remain private", len(owners), topk.shape[0],
                bytes_after - bytes_before)
