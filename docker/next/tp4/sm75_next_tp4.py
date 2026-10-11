"""Activate the preserved four-card adapter only during Flash Next construction."""
import importlib
import os
from pathlib import Path
import sys

_active = False

def validate(config):
    if os.environ.get("TP4_NEXT_RUNTIME", "0") != "1":
        return False
    hf = config.model_config.hf_text_config
    pc = config.parallel_config
    if getattr(hf, "model_type", "") not in {"qwen4_exp", "qwen4_exp_text"}:
        raise ValueError("TP4 adapter requires Qwen3.8 Flash Next")
    if pc.tensor_parallel_size != 4 or getattr(pc, "pipeline_parallel_size", 1) != 1:
        raise ValueError("Flash Next TP4 adapter requires TP4 and PP1")
    if not getattr(pc, "enable_expert_parallel", False):
        raise ValueError("Flash Next TP4 adapter requires expert parallel")
    if config.speculative_config is not None:
        raise ValueError("The preserved Flash Next TP4 profile is target-only")
    if getattr(hf, "num_experts", 0) != 512 or getattr(hf, "hidden_size", 0) != 2560:
        raise ValueError("Unsupported Flash Next TP4 expert layout")
    if str(config.model_config.dtype) not in {"float16", "torch.float16", "half"}:
        raise ValueError("Flash Next TP4 adapter requires float16")
    if not os.environ.get("SM75_PLE_STORE") or not os.environ.get("SM75_PLE_MODEL"):
        raise ValueError("Set SM75_PLE_STORE and SM75_PLE_MODEL for TP4 DiskPLE")
    return True

def install(config):
    global _active
    if not validate(config):
        return False
    if _active:
        return True
    from vllm.platforms import current_platform
    if not current_platform.is_device_capability(75):
        raise ValueError("Flash Next TP4 adapter is restricted to SM75")
    root = str(Path(__file__).with_name("sm75_next_tp4_runtime"))
    if root not in sys.path:
        sys.path.insert(0, root)
    # Keep the validated adapter files byte-for-byte; no algorithm changes.
    for name in ("graph_boundary", "text_cache_compat", "tp4_bounded", "tp4_routed_hotcache"):
        importlib.import_module(name)
    _active = True
    return True
