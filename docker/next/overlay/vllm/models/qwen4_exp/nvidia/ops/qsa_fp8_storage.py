"""SM75 E4M3FN storage helpers: integer decode, FP16 attention arithmetic.

Quantization/cache scatter deliberately reuse vLLM's standard C++ cache op.
This module does not quantize QSA indexer keys, queries, logits, or selection.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def decode_e4m3fn(encoded):
    """Decode all 256 E4M3FN byte patterns without FP8 ISA instructions."""
    raw = encoded.to(tl.uint32)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    sign = (raw & 128) << 24
    normal_bits = sign | ((exponent + 120) << 23) | (mantissa << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)
    magnitude = mantissa.to(tl.float32) * 0.001953125
    subnormal_bits = magnitude.to(tl.uint32, bitcast=True) | sign
    subnormal = subnormal_bits.to(tl.float32, bitcast=True)
    value = tl.where(exponent == 0, subnormal, normal)
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), value)


@triton.jit(do_not_specialize=["n"])
def _decode_contiguous_kernel(Source, Scale, Output, n, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(Source + offset, mask=offset < n, other=0)
    value = decode_e4m3fn(raw) * tl.load(Scale)
    tl.store(Output + offset, value, mask=offset < n)


def decode_contiguous(source: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert source.dtype == torch.uint8 and source.is_contiguous()
    assert scale.numel() == 1 and scale.dtype == torch.float32
    assert scale.device == source.device
    output = torch.empty(source.shape, dtype=torch.float16, device=source.device)
    _decode_contiguous_kernel[(triton.cdiv(source.numel(), 256),)](
        source, scale, output, source.numel(), BLOCK=256, num_warps=4)
    return output


def store_standard(key, value, key_cache, value_cache, slots, k_scale, v_scale):
    from vllm._custom_ops import reshape_and_cache_flash
    assert key.dtype == value.dtype == torch.float16
    assert key_cache.dtype == value_cache.dtype == torch.uint8
    assert k_scale.dtype == v_scale.dtype == torch.float32
    assert k_scale.numel() == v_scale.numel() == 1
    reshape_and_cache_flash(key, value, key_cache, value_cache, slots,
                            "fp8_e4m3", k_scale, v_scale)
