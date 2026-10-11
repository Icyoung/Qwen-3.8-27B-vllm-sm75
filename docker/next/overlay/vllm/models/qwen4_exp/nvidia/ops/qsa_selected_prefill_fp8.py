"""SM75 QSA selected-page Prefill with FP8 KV and FP16 TensorCore arithmetic.

Each learned QSA selection group consists of four contiguous aligned tokens.
Reinterpret the native cache as four-token pages and make each query a separate
request whose page list is exactly that query's selected groups and causal tail.
This removes dense masked work while using FlashInfer's CUDA prefill kernels.
FP16 Q, FP8 KV storage; indexer and selected token order are unchanged. Uses the
public workspace_size/reset_workspace_buffer APIs and preserves plan copy guards.
"""
import torch
import logging
import math
from vllm.triton_utils import tl, triton
from vllm.models.qwen4_exp.nvidia.ops.qsa_flashinfer_prefill import _gate


@triton.jit
def _selected_subpages(Indices, Table, Indptr, Pages,
                       SI: tl.constexpr, PAGE: tl.constexpr, WIDTH: tl.constexpr,
                       PHYSICAL_BLOCK_SUBPAGES: tl.constexpr):
    row = tl.program_id(0)
    count = tl.load(Indices + row * SI + WIDTH)
    first = tl.load(Indptr + row)
    groups = tl.arange(0, 1024)
    col = groups * 4
    valid = col < count
    token = tl.load(Indices + row * SI + col, mask=valid, other=0)
    physical_page = tl.load(Table + token // PAGE, mask=valid, other=0)
    physical_subpage = physical_page * PHYSICAL_BLOCK_SUBPAGES + (token % PAGE) // 4
    tl.store(Pages + first + groups, physical_subpage, mask=valid)


class SelectedPagesRunner:
    """One request -> independent per-query sparse page lists.

    Assumes q rows are the contiguous suffix ending at seq-1. Intended for
    single-request prefill only, not mixed batches or speculative verification.
    Keep this object across layers. K/V views stay on their original storage.
    """
    def __init__(self, device, disable_split_kv=True, num_query_heads=3):
        assert num_query_heads in (3,6)
        self.num_query_heads=num_query_heads
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
        self.device = torch.device(device)
        self.workspace = torch.empty(256,
                                     dtype=torch.uint8, device=self.device)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="fa2")
        if not callable(getattr(self.wrapper, "workspace_size", None)):
            raise RuntimeError("FlashInfer lacks the required workspace_size API")
        # The public constructor transiently creates an 8 MiB int workspace.
        # Immediately release it through the public API; the CUDA allocator may
        # retain the freed block for reuse by subsequent model activations.
        self.int_workspace = torch.empty(256, dtype=torch.uint8, device=self.device)
        self.wrapper.reset_workspace_buffer(self.workspace, self.int_workspace)
        self.last_workspace_requirements = None
        self.disable_split_kv = disable_split_kv
        self.pages = None
        self.indptr = None
        self.expected_counts = None
        self.plan_key = None
        self.plan_ready = None
        self.stream = torch.cuda.current_stream(self.device).cuda_stream

    def _prepare(self, rows, seq):
        first = seq - rows
        assert first >= 0
        # At >=512 complete groups, page-list lengths only depend on tail mod4.
        # Avoid CPU plan work for every QSA layer and long-context chunk.
        key = (rows, "full", first % 4) if first >= 2047 else (rows, "short", seq)
        if key == self.plan_key:
            return
        if self.plan_ready is not None:
            # Protect FlashInfer's reusable pinned planning buffer from being
            # overwritten while an earlier plan H2D transfer is outstanding.
            self.plan_ready.synchronize()
        positions = torch.arange(first, seq, dtype=torch.int32)
        counts = torch.clamp((positions + 1) // 4, max=512) * 4 + (positions + 1) % 4
        page_counts = (counts + 3) // 4
        cpu_indptr = torch.zeros(rows + 1, dtype=torch.int32)
        cpu_indptr[1:] = torch.cumsum(page_counts, 0)
        total_pages = int(cpu_indptr[-1])
        if self.pages is None or self.pages.numel() < total_pages:
            self.pages = torch.empty(total_pages, dtype=torch.int32, device=self.device)
        self.indptr = cpu_indptr.to(self.device)
        self.expected_counts = counts
        plan_args = (
            torch.arange(rows + 1, dtype=torch.int32),
            cpu_indptr,
            self.pages[:total_pages],
            (counts - 1) % 4 + 1,
            self.num_query_heads, 1, 256, 4)
        plan_options = dict(causal=False, use_fp16_qk_reduction=False,
                            q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn,
                            o_data_type=torch.float16,
                            disable_split_kv=self.disable_split_kv)
        float_need, int_need = self.wrapper.workspace_size(*plan_args, **plan_options)
        float_capacity = max(256, triton.cdiv(int(float_need), 256) * 256)
        int_capacity = max(256, triton.cdiv(int(int_need), 256) * 256)
        changed = False
        if self.workspace.numel() < float_capacity:
            self.workspace = torch.empty(float_capacity, dtype=torch.uint8, device=self.device)
            changed = True
        if self.int_workspace.numel() < int_capacity:
            self.int_workspace = torch.empty(int_capacity, dtype=torch.uint8, device=self.device)
            changed = True
        if changed:
            # _prepare's event wait above protects the old pinned staging buffer
            # before reset_workspace_buffer replaces it.
            self.wrapper.reset_workspace_buffer(self.workspace, self.int_workspace)
        self.last_workspace_requirements = dict(
            rows=rows, seq=seq, float_required_bytes=int(float_need),
            int_required_bytes=int(int_need),
            float_allocated_bytes=self.workspace.numel(),
            int_allocated_bytes=self.int_workspace.numel(),
            selected_page_bytes=self.pages.numel() * self.pages.element_size())
        self.wrapper.plan(*plan_args, **plan_options)
        self.plan_ready = torch.cuda.Event()
        self.plan_ready.record(torch.cuda.current_stream(self.device))
        self.plan_key = key

    def run(self, q, k, v, indices, table, seq, gate, out, *, k_scale=1.0, v_scale=1.0):
        assert q.dtype == torch.float16 and k.dtype == v.dtype == torch.uint8
        assert q.shape[1:] == (self.num_query_heads, 256) and k.shape[2:] == (1, 256)
        assert q.is_contiguous() and out.is_contiguous() and gate.is_contiguous()
        assert indices.shape == (q.shape[0], 2052)
        assert indices.dtype == table.dtype == torch.int32
        assert table.shape[0] == 1 and table.stride(1) == 1
        assert k.shape[1] % 4 == 0
        assert k.stride(0) % (4 * k.stride(1)) == 0
        assert v.stride(0) % (4 * v.stride(1)) == 0
        physical_block_subpages = k.stride(0) // (4 * k.stride(1))
        assert physical_block_subpages == v.stride(0) // (4 * v.stride(1))
        assert torch.cuda.current_stream(self.device).cuda_stream == self.stream
        assert not torch.cuda.is_current_stream_capturing()
        self._prepare(q.shape[0], seq)
        _selected_subpages[(q.shape[0],)](
            indices, table, self.indptr, self.pages,
            SI=indices.stride(0), PAGE=k.shape[1], WIDTH=2051,
            PHYSICAL_BLOCK_SUBPAGES=physical_block_subpages, num_warps=4)
        # Split native pages into 4-token pages without copying or changing the
        # offset of the interleaved FP16 K and V storage views.
        shape = ((k.shape[0]-1)*physical_block_subpages + k.shape[1]//4, 4, 1, 256)
        kk = k.as_strided(shape, (4*k.stride(1), k.stride(1), k.stride(2), 1))
        vv = v.as_strided(shape, (4*v.stride(1), v.stride(1), v.stride(2), 1))
        self.wrapper.run(q, (kk.view(torch.float8_e4m3fn), vv.view(torch.float8_e4m3fn)),
                         out=out, k_scale=k_scale, v_scale=v_scale)
        _gate[(triton.cdiv(out.numel(), 256),)](
            out, gate, out, out.numel(), B=256, num_warps=4)
        return out


_RUNNERS = {}
_CALLS = 0


def run(q, k, v, indices, table, seq, gate, out, *, k_scale, v_scale):
    """Same learned selection, direct FP8 reads, no temporary FP16 KV cache.

    Scale arguments must be audited host shadows of the actual GPU cache scales.
    Refuse tensors to prevent hidden per-layer GPU scalar synchronization.
    """
    global _CALLS
    if not isinstance(k_scale, (float, int)) or not isinstance(v_scale, (float, int)):
        raise TypeError("FP8 selected-page Prefill requires validated host scalar scales")
    if not (math.isfinite(k_scale) and math.isfinite(v_scale) and k_scale > 0 and v_scale > 0):
        raise ValueError("FP8 cache scales must be finite and positive")
    device = q.device
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream, q.shape[1])
    runner = _RUNNERS.get(key)
    if runner is None:
        runner = SelectedPagesRunner(device, num_query_heads=q.shape[1])
        _RUNNERS[key] = runner
        logging.getLogger(__name__).info(
            "SM75 QSA FP8 selected-page Prefill active: native_cache_stride=%s, page_size=%d, chunk=1024, fp16_kv_staging=0",
            tuple(k.stride()), k.shape[1])
    rows = q.shape[0]
    for first in range(0, rows, 1024):
        stop = min(first+1024, rows)
        runner.run(q[first:stop].contiguous(), k, v, indices[first:stop], table,
                   seq - rows + stop, gate[first:stop], out[first:stop],
                   k_scale=float(k_scale), v_scale=float(v_scale))
    _CALLS += 1
    return out


def warmup():
    """Compile the CUDA FP8/FP16 path without retaining an extra layer cache."""
    device = torch.device("cuda", torch.cuda.current_device())
    rows, seq, page = 128, 16403, 400
    q = torch.zeros((rows, 3, 256), dtype=torch.float16, device=device)
    gate, out = torch.zeros_like(q), torch.empty_like(q)
    kv = torch.zeros((1, 1, page, 512), dtype=torch.uint8, device=device)
    k, v = kv.transpose(1, 2).split(256, dim=-1)
    table = torch.zeros((1, triton.cdiv(seq, page)), dtype=torch.int32, device=device)
    indices = torch.full((rows, 2052), -1, dtype=torch.int32, device=device)
    indices[:, :2048] = torch.arange(2048, dtype=torch.int32, device=device)
    positions = torch.arange(seq-rows, seq, dtype=torch.int32, device=device)
    tails = (positions+1) % 4
    for lane in range(3):
        indices[:, 2048+lane] = torch.where(tails > lane, ((positions+1)//4)*4+lane, -1)
    indices[:, -1] = 2048+tails
    runner = SelectedPagesRunner(device, num_query_heads=q.shape[1])
    runner.run(q, k, v, indices, table, seq, gate, out, k_scale=1., v_scale=1.)
    torch.cuda.synchronize(device)


