"""SM75 TP8 grouped QSA execution, preserving each request's learned selection.

Only non-captured multi-request target batches enter this path. Decode-only,
MTP draft, other head layouts and other devices retain their existing paths.
"""
import os
import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)
_logged = set()


def enabled():
    return (os.getenv('VLLM_SM75_QSA_MIXED_PREFILL', '0') == '1'
            and current_platform.is_device_capability(75)
            and not torch.cuda.is_current_stream_capturing())


def segments(query_start_loc, seq_lens, num_tokens):
    """Read exact GPU lengths once per forward, not CPU speculative upper bounds.

    The cache belongs to ForwardContext, so CUDA buffer reuse across iterations
    cannot reuse stale boundaries. It contains only small host tuples.
    """
    if query_start_loc.ndim != 1 or seq_lens.ndim != 1:
        return None
    requests = query_start_loc.numel()-1
    if requests < 2 or seq_lens.numel() < requests:
        return None
    context = get_forward_context()
    cache = getattr(context, '_sm75_qsa_mixed_segments', None)
    if cache is None:
        cache = {}
        context._sm75_qsa_mixed_segments = cache
    key = (query_start_loc.data_ptr(), seq_lens.data_ptr(), requests, num_tokens)
    if key not in cache:
        packed = torch.cat((query_start_loc, seq_lens[:requests])).detach().cpu().tolist()
        starts, lengths = packed[:requests+1], packed[requests+1:]
        if starts[0] != 0 or starts[-1] != num_tokens:
            return None
        spans = []
        for req, (start, stop, seq) in enumerate(zip(starts, starts[1:], lengths)):
            if not (0 <= start <= stop <= num_tokens and seq >= stop-start):
                return None
            if stop > start:
                spans.append((req, int(start), int(stop), int(seq)))
        cache[key] = tuple(spans)
    return cache[key]


def try_grouped_indexer(indexer, q, cache, metadata, visible_blocks, block_indices):
    if (not enabled() or not getattr(indexer, '_sm75_mixed_target', False)
            or metadata.num_prefills == 0
            or metadata.num_decodes + metadata.num_prefills < 2
            or q.dtype != torch.float16 or cache.dtype != torch.float16
            or q.shape[1:] != (4, 128)):
        return False
    spans = segments(metadata.query_start_loc, metadata.seq_lens, q.shape[0])
    if spans is None:
        return False
    from .qsa_indexer import qsa_select_paged_prefill
    from .qsa_indexer_cublas import try_qsa_select_paged_prefill_cublas
    fast_count = 0
    for req, start, stop, seq in spans:
        if req < metadata.num_decodes:
            continue  # Leading decode rows have already used the original path.
        count = stop-start
        local_starts = metadata.query_start_loc[req:req+2]-start
        table = metadata.block_table[req:req+1]
        kwargs = (q[start:stop], cache, table, local_starts,
                  visible_blocks[start:stop], indexer.token_topk,
                  indexer.compress_ratio, count, block_indices[start:stop], seq)
        used = try_qsa_select_paged_prefill_cublas(*kwargs, single_request_prefill=True)
        if not used:
            qsa_select_paged_prefill(*kwargs)
        fast_count += int(used)
    if 'indexer' not in _logged:
        logger.info('SM75 QSA mixed grouped indexer active: requests=%d, cuBLAS groups=%d',
                    len(spans), fast_count)
        _logged.add('indexer')
    return True


def try_grouped_attention(layer, q, k, v, indices, table, query_start_loc,
                          seq_lens, token_to_req, gate, output):
    if (not enabled() or q.dtype != torch.float16 or q.shape[1:] != (3, 256)
            or k.dtype not in (torch.float16, torch.uint8) or k.dtype != v.dtype
            or k.shape[2:] != (1, 256) or not gate.is_contiguous()
            or not output.is_contiguous()
            or not getattr(layer.indexer, '_sm75_mixed_target', False)):
        return False
    spans = segments(query_start_loc, seq_lens, q.shape[0])
    if spans is None or not any(stop-start >= 128 for _, start, stop, _ in spans):
        return False
    from .qsa import qsa_sparse_paged_attention
    from .qsa_flashinfer_prefill import run as short_prefill
    from .qsa_selected_prefill_fp8 import run as long_fp8
    from .qsa_selected_prefill import run as long_fp16
    fast_count = 0
    for req, start, stop, seq in spans:
        count = stop-start
        eligible = count >= 128 and 0 < seq <= 262144
        if eligible and seq <= 16384 and os.getenv('VLLM_SM75_QSA_FLASHINFER_PREFILL', '1') == '1':
            short_prefill(q[start:stop].contiguous(), k, v, indices[start:stop],
                          table[req:req+1], seq, gate[start:stop], output[start:stop],
                          k_scale=layer._k_scale, v_scale=layer._v_scale)
            fast_count += 1
        elif (eligible and seq > 16384 and k.shape[1] % 4 == 0
              and os.getenv('VLLM_SM75_QSA_SELECTED_PAGES_PREFILL', '1') == '1'):
            if k.dtype == torch.uint8:
                long_fp8(q[start:stop].contiguous(), k, v, indices[start:stop],
                         table[req:req+1], seq, gate[start:stop], output[start:stop],
                         k_scale=layer._k_scale_float, v_scale=layer._v_scale_float)
            else:
                long_fp16(q[start:stop].contiguous(), k, v, indices[start:stop],
                          table[req:req+1], seq, gate[start:stop], output[start:stop])
            fast_count += 1
        else:
            qsa_sparse_paged_attention(q[start:stop], k, v, indices[start:stop], table,
                token_to_req[start:stop], count > layer._max_decode_query_len,
                output[start:stop], output_gate=gate[start:stop],
                k_scale=layer._k_scale, v_scale=layer._v_scale)
    if 'attention' not in _logged:
        logger.info('SM75 QSA mixed grouped attention active: requests=%d, fast prefill groups=%d',
                    len(spans), fast_count)
        _logged.add('attention')
    return True
