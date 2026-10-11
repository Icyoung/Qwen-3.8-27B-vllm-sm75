"""Larger physical weight buffers with unchanged logical parameter views."""
import os
import torch
from vllm.model_executor.offloader.prefetch import StaticBufferPool,ParamInfo
MAX_CAPACITY=int(os.environ.get('TP4_EXPERT_CACHE_CAPACITY','128'))
assert MAX_CAPACITY in (128,192,256,384)
_wide={}
_init=StaticBufferPool.__init__
def init(self,param_infos,slot_capacity,device):
    if MAX_CAPACITY==128:
        return _init(self,param_infos,slot_capacity,device)
    infos={i.key:i for i in param_infos}
    assert infos and all(i.shape[0]==128 for i in infos.values())
    extended={k:ParamInfo(i.name,(MAX_CAPACITY,*i.shape[1:]),i.stride,i.dtype) for k,i in infos.items()}
    _init(self,list(extended.values()),slot_capacity,device)
    logical={}
    for key,info in extended.items():
        tensors=self._buffers[info.key]
        logical[key]=[t[:128] for t in tensors]
        for t in tensors:
            if t.numel():_wide[t.data_ptr()]=t
    self._buffers=logical
    print('TP4_PHYSICAL_EXPERT_CAPACITY',MAX_CAPACITY,'bytes',self.total_bytes,flush=True)
StaticBufferPool.__init__=init
def tensor(t):
    return _wide.get(t.data_ptr(),t) if t.numel() else t
def weight_kwargs(kw):
    out=dict(kw)
    for k in ('w1','w2','w1_scale','w2_scale','w1_zeros','w2_zeros','bias1','bias2'):
        if isinstance(out.get(k),torch.Tensor):out[k]=tensor(out[k])
    return out
