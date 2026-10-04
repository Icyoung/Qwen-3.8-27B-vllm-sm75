"""One-time, per-owner validation of actual target/MTP FP8 cache bindings."""
import json
import math
import os
from pathlib import Path
import time

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)


def audit_fp8_layout(layer):
    if getattr(layer, "_sm75_fp8_layout_audited", False):
        return
    if torch.cuda.is_current_stream_capturing():
        return  # First ordinary prefill checks real values without capture sync.
    cache = layer.kv_cache
    indexer = layer.indexer
    assert layer.kv_cache_dtype in ("fp8", "fp8_e4m3")
    assert cache.dtype == torch.uint8 and cache.ndim == 4
    assert cache.shape[1] == layer.num_kv_heads == 1
    assert cache.shape[-1] == 2 * layer.head_dim == 512
    assert cache.stride(-1) == 1 and cache.is_cuda
    assert layer.impl.kv_cache_dtype == layer.kv_cache_dtype
    assert layer.kv_cache_torch_dtype == torch.uint8
    assert indexer.indexer_dtype == torch.float16
    assert indexer.raw_key_cache.kv_cache.dtype == torch.float16
    assert indexer.compressed_key_cache.kv_cache.dtype == torch.float16
    assert indexer.token_topk == 2048 and indexer.compress_ratio == 4
    scales, shadows = {}, {}
    for name in ("_k_scale", "_v_scale"):
        value = getattr(layer, name)
        assert value.dtype == torch.float32 and value.numel() == 1
        assert value.device == cache.device
        number = float(value.item())
        assert math.isfinite(number) and number == 1.0, (layer.layer_name, name, number)
        shadow = getattr(layer, name + "_float")
        assert isinstance(shadow, (int, float)) and math.isfinite(shadow)
        assert number == shadow, (layer.layer_name, name, number, shadow)
        scales[name] = number
        shadows[name + "_float"] = shadow
    role = "mtp" if ("mtp" in layer.layer_name.lower() or
                     indexer.layer_id >= int(layer.config.num_hidden_layers)) else "target"
    key, value = cache.transpose(1, 2).split(layer.head_dim, dim=-1)
    data = {"kind": "sm75_qsa_fp8_layout", "time": time.time(),
            "rank": get_tensor_model_parallel_rank(), "pid": os.getpid(),
            "tag": os.environ.get("VLLM_SM75_QSA_FP8_AUDIT_TAG", ""),
            "layer": layer.layer_name, "layer_id": indexer.layer_id, "role": role,
            "main_dtype": str(cache.dtype), "main_shape": list(cache.shape),
            "main_stride": list(cache.stride()), "key_stride": list(key.stride()),
            "value_stride": list(value.stride()), "main_scalar_bytes": cache.element_size(),
            "indexer_dtype": str(indexer.indexer_dtype),
            "raw_indexer_dtype": str(indexer.raw_key_cache.kv_cache.dtype),
            "compressed_indexer_dtype": str(indexer.compressed_key_cache.kv_cache.dtype),
            "token_budget": indexer.token_topk, "compress_ratio": indexer.compress_ratio,
            "scales": scales, "scale_shadows": shadows, "fixed_unit_scales": True,
            "topk_pointer": layer.topk_indices_buffer.data_ptr(),
            "block_shared": getattr(indexer, "_sm75_shared_block_indices", None) is not None,
            "passed": True}
    output = os.environ.get("VLLM_SM75_QSA_FP8_AUDIT_DIR")
    if output:
        root = Path(output).resolve()
        if not (root.is_relative_to(Path("/work/long256")) or
                root.is_relative_to(Path("/data/vllm-sm-next"))):
            raise ValueError("FP8 audit output must remain in test directories")
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"fp8-layout-rank{data['rank']}-pid{data['pid']}.jsonl").open("a") as f:
            f.write(json.dumps(data) + "\n")
    logger.info("SM75 QSA FP8 layout verified: layer=%s role=%s shape=%s stride=%s "
                "K/V scales=1/1 indexer=FP16", layer.layer_name, role,
                tuple(cache.shape), tuple(cache.stride()))
    layer._sm75_fp8_layout_audited = True
