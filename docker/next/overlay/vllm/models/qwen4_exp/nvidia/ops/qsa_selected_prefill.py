"""V2: exact sparse QSA with planner-sized FlashInfer workspaces.

Each learned QSA selection group consists of four contiguous aligned tokens.
Reinterpret the native cache as four-token pages and make each query a separate
request whose page list is exactly that query's selected groups and causal tail.
This removes dense masked work while using FlashInfer's CUDA prefill kernels.
FP16 only; indexer outputs and selected token order are unchanged. Uses the
public workspace_size/reset_workspace_buffer APIs and preserves plan copy guards.
"""
import torch
from vllm.triton_utils import tl, triton
from vllm.models.qwen4_exp.nvidia.ops.qsa_flashinfer_prefill import _gate


@triton.jit
def _selected_subpages(Indices, Table, Indptr, Pages,
                       SI: tl.constexpr, PAGE: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    count = tl.load(Indices + row * SI + WIDTH)
    first = tl.load(Indptr + row)
    groups = tl.arange(0, 1024)
    col = groups * 4
    valid = col < count
    token = tl.load(Indices + row * SI + col, mask=valid, other=0)
    physical_page = tl.load(Table + token // PAGE, mask=valid, other=0)
    physical_subpage = physical_page * (PAGE // 4) + (token % PAGE) // 4
    tl.store(Pages + first + groups, physical_subpage, mask=valid)


class SelectedPagesRunner:
    """One request -> independent per-query sparse page lists.

    Assumes q rows are the contiguous suffix ending at seq-1. Intended for
    single-request prefill only, not mixed batches or speculative verification.
    Keep this object across layers. K/V views stay on their original storage.
    """
    def __init__(self, device, disable_split_kv=True):
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
            3, 1, 256, 4)
        plan_options = dict(causal=False, use_fp16_qk_reduction=False,
                            q_data_type=torch.float16, kv_data_type=torch.float16,
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

    def run(self, q, k, v, indices, table, seq, gate, out):
        assert q.dtype == k.dtype == v.dtype == torch.float16
        assert q.shape[1:] == (3, 256) and k.shape[2:] == (1, 256)
        assert q.is_contiguous() and out.is_contiguous() and gate.is_contiguous()
        assert indices.shape == (q.shape[0], 2052)
        assert indices.dtype == table.dtype == torch.int32
        assert table.shape[0] == 1 and table.stride(1) == 1
        assert k.shape[1] % 4 == 0
        assert k.stride(0) == k.shape[1] * k.stride(1)
        assert v.stride(0) == v.shape[1] * v.stride(1)
        assert torch.cuda.current_stream(self.device).cuda_stream == self.stream
        assert not torch.cuda.is_current_stream_capturing()
        self._prepare(q.shape[0], seq)
        _selected_subpages[(q.shape[0],)](
            indices, table, self.indptr, self.pages,
            SI=indices.stride(0), PAGE=k.shape[1], WIDTH=2051, num_warps=4)
        # Split native pages into 4-token pages without copying or changing the
        # offset of the interleaved FP16 K and V storage views.
        shape = (k.shape[0] * (k.shape[1] // 4), 4, 1, 256)
        kk = k.as_strided(shape, (4*k.stride(1), k.stride(1), k.stride(2), 1))
        vv = v.as_strided(shape, (4*v.stride(1), v.stride(1), v.stride(2), 1))
        self.wrapper.run(q, (kk, vv), out=out)
        _gate[(triton.cdiv(out.numel(), 256),)](
            out, gate, out, out.numel(), B=256, num_warps=4)
        return out


# One shared workspace per rank/device/stream, not one allocation per QSA layer.
_RUNNERS = {}


def run(q, k, v, indices, table, seq, gate, out):
    """Execute the unchanged selected token sets with bounded query chunks."""
    device = q.device
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    runner = _RUNNERS.get(key)
    if runner is None:
        runner = SelectedPagesRunner(device)
        _RUNNERS[key] = runner
    rows = q.shape[0]
    for first in range(0, rows, 1024):
        stop = min(first + 1024, rows)
        # Some projection outputs are contiguous within a head but carry a
        # wider row stride. A bounded copy preserves FlashInfer's input contract.
        qq = q[first:stop].contiguous()
        runner.run(qq, k, v, indices[first:stop], table,
                   seq - rows + stop, gate[first:stop], out[first:stop])
    return out


def warmup():
    """Compile without retaining a workspace on an unused startup CUDA stream."""
    device = torch.device("cuda", torch.cuda.current_device())
    rows, seq, page = 128, 16403, 400
    q = torch.zeros((rows, 3, 256), dtype=torch.float16, device=device)
    gate, out = torch.zeros_like(q), torch.empty_like(q)
    kv = torch.zeros((1, 1, page, 512), dtype=torch.float16, device=device)
    k, v = kv.transpose(1, 2).split(256, dim=-1)
    table = torch.zeros((1, triton.cdiv(seq, page)), dtype=torch.int32, device=device)
    indices = torch.full((rows, 2052), -1, dtype=torch.int32, device=device)
    indices[:, :2048] = torch.arange(2048, dtype=torch.int32, device=device)
    positions = torch.arange(seq - rows, seq, dtype=torch.int32, device=device)
    tails = (positions + 1) % 4
    for lane in range(3):
        indices[:, 2048+lane] = torch.where(
            tails > lane, ((positions + 1)//4)*4 + lane, -1)
    indices[:, -1] = 2048 + tails
    runner = SelectedPagesRunner(device)
    runner.run(q, k, v, indices, table, seq, gate, out)
    torch.cuda.synchronize(device)
