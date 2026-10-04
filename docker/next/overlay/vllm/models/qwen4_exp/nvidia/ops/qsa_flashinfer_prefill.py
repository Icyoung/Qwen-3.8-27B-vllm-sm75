"""SM75 TP8 QSA Prefill over the unchanged learned selection mask.

Only execution changes. Decode, mixed batches, and long contexts keep the
sparse fallback. See the image optimization manifest for validation evidence.
"""
import torch
from vllm.triton_utils import tl,triton

@triton.jit(do_not_specialize=["seq"])
def _flat_mask(Indices,Bits,stride:tl.constexpr,seq,
               TOPK:tl.constexpr,GROUPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);count=tl.load(Indices+row*stride+TOPK)
    group=tl.arange(0,BLOCK);complete=count//4
    tok=tl.load(Indices+row*stride+group*4,mask=(group<complete)&(group<GROUPS),other=-1)
    valid=(group<complete)&(group<GROUPS)&(tok>=0)&(tok+3<seq)
    bit=row.to(tl.int64)*seq+tl.maximum(tok,0);offset=(bit%32).to(tl.uint32)
    low=(tl.full((BLOCK,),15,tl.uint32)<<offset).to(tl.int32)
    tl.atomic_or(Bits+bit//32,low,mask=valid,sem='relaxed')
    high=(tl.full((BLOCK,),15,tl.uint32)>>(32-tl.maximum(offset,1))).to(tl.int32)
    tl.atomic_or(Bits+bit//32+1,high,mask=valid&(offset>28),sem='relaxed')
    lane=tl.arange(0,4);i=complete*4+lane
    tok=tl.load(Indices+row*stride+i,mask=(i<count)&(i<TOPK),other=-1)
    bit=row.to(tl.int64)*seq+tl.maximum(tok,0)
    value=(tl.full((4,),1,tl.uint32)<<(bit%32).to(tl.uint32)).to(tl.int32)
    tl.atomic_or(Bits+bit//32,value,mask=(i<count)&(i<TOPK)&(tok>=0)&(tok<seq),sem='relaxed')

@triton.jit(do_not_specialize=["N"])
def _gate(Attention,Gate,Out,N,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.load(Attention+i,mask=i<N,other=0).to(tl.float32)
    g=tl.load(Gate+i,mask=i<N,other=0).to(tl.float32)
    tl.store(Out+i,a*tl.sigmoid(g),mask=i<N)

def pack_mask(indices,seq):
    assert indices.shape[1]==2052 and indices.dtype==torch.int32
    bits=torch.zeros((triton.cdiv(indices.shape[0]*seq,32),),device=indices.device,dtype=torch.int32)
    _flat_mask[(indices.shape[0],)](indices,bits,indices.stride(0),seq,TOPK=2051,GROUPS=512,BLOCK=512,num_warps=4)
    return bits.view(torch.uint8)[:triton.cdiv(indices.shape[0]*seq,8)]

def run(q,k,v,indices,table,seq,gate,out,*,k_scale=None,v_scale=None):
    from flashinfer import single_prefill_with_kv_cache
    assert q.dtype==torch.float16 and k.dtype==v.dtype
    assert k.dtype in (torch.float16,torch.uint8)
    assert q.shape[1:]==(3,256) and k.shape[2:]==(1,256)
    assert 0<seq<=16384 and q.shape[0]>=128
    assert gate.is_contiguous() and out.is_contiguous()
    pages=table[0,:triton.cdiv(seq,k.shape[1])].long()
    kk=k[pages].reshape(-1,1,256)[:seq];vv=v[pages].reshape(-1,1,256)[:seq]
    if k.dtype==torch.uint8:
        from .qsa_fp8_storage import decode_contiguous
        kk=decode_contiguous(kk.contiguous(),k_scale)
        vv=decode_contiguous(vv.contiguous(),v_scale)
    mask=pack_mask(indices,seq)
    attention=single_prefill_with_kv_cache(q,kk,vv,packed_custom_mask=mask,
        backend='fa2',use_fp16_qk_reduction=False)
    assert attention.is_contiguous()
    _gate[(triton.cdiv(out.numel(),256),)](attention,gate,out,out.numel(),B=256,num_warps=4)
    return out


def warmup():
    """Compile the shape-independent bitmask/gate and CUDA attention at startup."""
    device=torch.device("cuda",torch.cuda.current_device())
    q=torch.zeros((128,3,256),device=device,dtype=torch.float16)
    k=torch.zeros((1,400,1,256),device=device,dtype=torch.float16)
    v=torch.zeros_like(k);gate=torch.zeros_like(q);out=torch.empty_like(q)
    indices=torch.full((128,2052),-1,device=device,dtype=torch.int32)
    indices[:,:4]=torch.arange(4,device=device,dtype=torch.int32)
    indices[:,-1]=4
    table=torch.zeros((1,1),device=device,dtype=torch.int32)
    run(q,k,v,indices,table,400,gate,out)
    torch.cuda.synchronize()
