"""SM75 DSpark integration contracts; no CUDA or model imports at module load.

Target verification and draft query graphs have different widths. All copies
are shallow so static_forward_context and the original target config survive.
The standard vLLM Markov sampler and rejection sampler are never replaced.
"""
from copy import copy

VERSION = "dspark-sm75-v1"
_MARKER = "_sm75_dspark_draft_query_len"


def is_dspark(config):
    return getattr(getattr(config, "speculative_config", None), "method", None) == "dspark"


def validate_dspark_config(config):
    """Fail closed for unsupported experimental SM75 modes, before allocation."""
    if not is_dspark(config):
        return
    spec = config.speculative_config
    if getattr(spec, "enable_adaptive_verification", False):
        raise ValueError("SM75 DSpark requires enable_adaptive_verification=false")
    if getattr(spec, "rejection_sample_method", "standard") != "standard":
        raise ValueError("SM75 DSpark requires standard rejection sampling")
    if getattr(spec, "draft_sample_method", None) != "probabilistic":
        raise ValueError("SM75 DSpark requires draft_sample_method=probabilistic")
    k = spec.num_speculative_tokens
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 7:
        raise ValueError("SM75 DSpark candidate supports explicit k in 1..7")
    if str(config.model_config.dtype) not in ("float16", "torch.float16", "half"):
        raise ValueError("SM75 DSpark candidate requires target/draft float16")
    pc = config.parallel_config
    if pc.tensor_parallel_size != 4 or getattr(spec, "draft_tensor_parallel_size", 4) != 4:
        raise ValueError("SM75 DSpark candidate requires target TP4 and draft TP4")
    if getattr(pc, "pipeline_parallel_size", 1) != 1 or getattr(pc, "decode_context_parallel_size", 1) != 1:
        raise ValueError("SM75 DSpark candidate requires PP1 and DCP1")
    if getattr(spec, "kv_cache_dtype", "auto") != "auto":
        raise ValueError("SM75 DSpark candidate requires draft kv_cache_dtype=auto")


def query_widths(spec):
    k = spec.num_speculative_tokens
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("num_speculative_tokens must be a positive integer")
    hf = spec.draft_model_config.hf_config
    anchor = getattr(hf, "sample_from_anchor", True)
    if not isinstance(anchor, bool):
        raise ValueError("sample_from_anchor must be a boolean")
    return k + 1, k if anchor else k + 1


def remap_capture_sizes(sizes, maximum, target_query, draft_query):
    """Preserve request counts, not token round-up (target 8 -> draft 7).

    Match the upstream uniform graph manager: round to whole target requests,
    discard sizes exceeding its maximum, then project those requests to draft.
    """
    if not maximum:
        return [], 0
    projected = {
        ((size + target_query - 1) // target_query) * draft_query
        for size in (sizes or [])
        if size > 0
        and ((size + target_query - 1) // target_query) * target_query <= maximum
    }
    mapped = sorted(projected)
    return mapped, max(mapped, default=0)


def draft_attention_config(config, actual_query_len):
    """Return a draft-only config for BOTH metadata builders and graph manager."""
    if not is_dspark(config):
        return config
    target_q, draft_q = query_widths(config.speculative_config)
    if actual_query_len != draft_q:
        raise ValueError(f"DSpark query mismatch: runtime={actual_query_len}, config={draft_q}")
    if getattr(config, _MARKER, None) == draft_q:
        return config
    result = copy(config)
    compilation = copy(config.compilation_config)
    sizes, maximum = remap_capture_sizes(
        compilation.cudagraph_capture_sizes,
        compilation.max_cudagraph_capture_size,
        target_q, draft_q,
    )
    compilation.cudagraph_capture_sizes = sizes
    compilation.max_cudagraph_capture_size = maximum
    result.compilation_config = compilation
    setattr(result, _MARKER, draft_q)
    # Keep the target's resolved layout, but use the draft's cache dtype.
    cache = copy(config.cache_config)
    cache.cache_dtype = config.speculative_config.kv_cache_dtype
    result.cache_config = cache
    return result


def graph_capture_plan(config):
    """Pure policy consumed by the existing feature-gated FA2 adapter."""
    spec = config.speculative_config
    target_q, draft_q = query_widths(spec)
    query = getattr(config, _MARKER, target_q)
    if query not in (target_q, draft_q):
        raise ValueError("Invalid DSpark attention role/query marker")
    cc = config.compilation_config
    maximum = cc.max_cudagraph_capture_size
    if not 1 <= spec.num_speculative_tokens <= 7 or not maximum:
        return query, frozenset()
    rows = frozenset(
        ((size + query - 1) // query) * query
        for size in (cc.cudagraph_capture_sizes or [])
        if size > 0
        and ((size + query - 1) // query) * query <= maximum
        and ((size + query - 1) // query) <= 8
    )
    return query, rows


def annotate_draft_cache_groups(configs, draft_names):
    """Tag complete draft groups using worker-reported names; reject mixed groups.

    Validate every rank before mutation. A mixed group cannot safely be marked
    as a draft group because that would change target prefix-cache handling.
    """
    names = set(draft_names)
    if not names:
        raise ValueError("DSpark cache annotation requires loaded draft layer names")
    pending = []
    for config in configs:
        found = set()
        for group in config.kv_cache_groups:
            layers = set(group.layer_names)
            hits = layers & names
            if hits:
                if hits != layers:
                    raise ValueError("DSpark target/draft layers share one KV group")
                found.update(hits)
                pending.append(group)
        missing = names - found
        if missing:
            raise ValueError(f"DSpark draft KV groups missing loaded layers: {sorted(missing)}")
    if not pending:
        raise ValueError("DSpark draft has no KV cache groups")
    for group in pending:
        group.is_eagle_group = True
    return len(pending)
