import torch,triton,triton.language as tl
_buffers={}
_maps={}
@triton.jit
def remap_kernel(ids,lut,out,N:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 x=tl.load(ids+i,i<N,other=-1)
 valid=(x>=0)&(x<C)
 mapped=tl.load(lut+tl.maximum(0,tl.minimum(x,C-1)))
 tl.store(out+i,tl.where(valid,mapped,-1),i<N)

def select(ids,emap):
 raw=ids.detach().cpu().reshape(-1).tolist()
 if emap is None:return sorted(set(raw)-{-1})
 key=emap.data_ptr()
 if key not in _maps:_maps[key]=emap.detach().cpu().tolist()
 table=_maps[key]
 return sorted({table[x] for x in raw if x>=0}-{-1})

def remap(original_ids,slots,count,identity):
 if identity not in _buffers:
  host=torch.empty(count,dtype=torch.int32,pin_memory=True)
  gpu=torch.empty(count,dtype=torch.int32,device=original_ids.device)
  _buffers[identity]=(host,gpu,host.numpy())
 host,gpu,array=_buffers[identity]
 array.fill(-1)
 for expert,slot in slots.items():array[expert]=slot
 # select() synchronizes this stream before host storage is reused.
 gpu.copy_(host,non_blocking=True)
 out=torch.empty_like(original_ids)
 remap_kernel[(triton.cdiv(out.numel(),128),)](original_ids,gpu,out,out.numel(),count,128)
 return out
