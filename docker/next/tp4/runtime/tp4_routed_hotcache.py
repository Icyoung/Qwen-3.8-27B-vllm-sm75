"""Eager GPU expert LRU using existing shared physical buffers, no extra weight VRAM."""
import os,json
import torch
from tp4_expert_slots import ExpertSlots
import wide_expert_cache as wide
from vllm.model_executor.offloader.prefetch import PrefetchOffloader,_ModuleOffloader
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as marlin
_active=None
_pools={}
_verified=set()
_stats={phase:dict(calls=0,hits=0,misses=0,evictions=0,bytes=0,reference_bytes=0,copies=0,full_loads=0,row_loads=0) for phase in ('prefill','decode')}
def hook(self,index,module):
    original=module.forward
    def run(*args,**kwargs):
        global _active
        assert _active is None
        _active=self.module_offloaders[index]
        try:return original(*args,**kwargs)
        finally:_active=None
    def forward(*args,**kwargs):
        return run(*args,**kwargs)
    module.forward=forward
PrefetchOffloader._hook_module_forward=hook
_ModuleOffloader.start_onload_to_static=lambda self:None
_native=marlin._fused_marlin_moe
def _original(*args,**kw):
    return _native(*args,**wide.weight_kwargs(kw))

def prepare(owner,kw):
    fast=os.environ.get('TP4_FAST_ROUTE','0')=='1'
    ids=kw['topk_ids']; emap=kw.get('expert_map')
    if fast:
        from fast_route import select
        selected=select(ids,emap)
    else:
        local=ids.long()
        if emap is not None:local=torch.where(local>=0,emap[local.clamp_min(0)],-1)
        selected=sorted(set(local.detach().cpu().flatten().tolist())-{-1})
    count=kw['w1'].shape[0];layer=owner.layer_idx
    assert all(0<=e<count for e in selected)
    params=sorted(owner._param_offloaders.items())
    assert all(p._cpu_storage.shape[0]==count for _,p in params)
    assert any(p._gpu_buffer.data_ptr()==kw['w1'].data_ptr() for _,p in params)
    identity=tuple(p._gpu_buffer.data_ptr() for _,p in params)
    phase='decode' if ids.shape[0]<=4 else 'prefill'
    from cache_policy import get_pool
    pool=get_pool(_pools,identity,count,phase)
    rowbytes=sum(p._cpu_storage[0].numel()*p._cpu_storage.element_size() for _,p in params)
    phase='decode' if ids.shape[0]<=4 else 'prefill';stat=_stats[phase]
    stat['calls']+=1
    from route_observe import observe
    observe(layer,selected,phase)
    needed=sum((layer,e) not in pool.entries for e in selected) if getattr(pool,'bulk_by_misses',False) else len(selected)
    if needed>=getattr(pool,'bulk_threshold',count//2):
        for _,p in params:p._gpu_buffer.copy_(p._cpu_storage,non_blocking=p._cpu_storage.is_pinned())
        pool.reset_full(layer);slots={e:e for e in selected};misses=[(e,e) for e in selected];evicted=0
        stat['copies']+=len(params);stat['full_loads']+=1
        stat['bytes']+=rowbytes*count
    else:
        slots,misses,evicted=pool.acquire(layer,selected)
        for _,p in params:
            for expert,slot in misses:
                wide.tensor(p._gpu_buffer)[slot].copy_(p._cpu_storage[expert],non_blocking=p._cpu_storage.is_pinned())
        stat['copies']+=len(params)*len(misses);stat['row_loads']+=len(misses)
        stat['bytes']+=rowbytes*len(misses)
    hit=len(selected)-len(misses)
    stat['hits']+=hit;stat['misses']+=len(misses);stat['evictions']+=evicted
    original_ids=kw['expert_ids']
    if fast:
        from fast_route import remap
        physical_ids=remap(original_ids,slots,count,identity)
    else:
        lut=[-1]*count
        for expert,slot in slots.items():lut[expert]=slot
        lut=torch.tensor(lut,dtype=torch.int32,device=kw['w1'].device)
        physical_ids=torch.where((original_ids>=0)&(original_ids<count),lut[original_ids.long().clamp(0,count-1)],-1)
    from cache_policy import verify
    physical_ids=verify(_original,kw,physical_ids,selected,count,layer,params,phase,stat,rowbytes,hit,pool,ids,emap,identity)
    if stat['calls']%130==0:
        print('TP4_HOT_STATS',os.getpid(),phase,json.dumps(stat),flush=True)
    return physical_ids,(selected,count,layer,params,phase,stat,rowbytes,hit,pool,ids,emap)

def routed(*args,**kw):
    owner=_active
    if owner is None:return _original(*args,**kw)
    assert not args
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    cap=BreakableCUDAGraphCapture.current()
    if cap is not None and cap._capturing:
        from graph_boundary import eager_output
        # Bind this layer's owner at capture time; replay does not execute hook().
        physical_ids=eager_output(lambda:prepare(owner,kw)[0])
        return _original(**dict(kw,expert_ids=physical_ids))
    physical_ids,meta=prepare(owner,kw)
    selected,count,layer,params,phase,stat,rowbytes,hit,pool,ids,emap=meta
    result=_original(**dict(kw,expert_ids=physical_ids))
    # Validate both first decode and a later cache-hit invocation for every layer.
    check=(layer,'hit' if hit else 'miss')
    if os.environ.get('TP4_VERIFY','1') == '1' and selected and (layer,phase,check[1]) not in _verified:
        before=result.clone()
        for _,p in params:p._gpu_buffer.copy_(p._cpu_storage,non_blocking=p._cpu_storage.is_pinned())
        pool.reset_full(layer)
        reference=_original(**kw)
        local=ids.long()
        if emap is not None:local=torch.where(local>=0,emap[local.clamp_min(0)],-1)
        valid=local.reshape(-1)>=0
        torch.testing.assert_close(before.reshape(-1,before.shape[-1])[valid],reference.reshape(-1,reference.shape[-1])[valid],rtol=0,atol=0)
        stat['reference_bytes']+=rowbytes*count
        _verified.add((layer,phase,check[1]))
        print('TP4_HOT_VERIFIED',os.getpid(),layer,phase,check[1],len(selected),flush=True)
    return result
marlin._fused_marlin_moe=routed

# A routed-expert eager break cannot bisect the shared-expert stream fork/join.
# Capture unloaded layers' shared experts on the main stream; resident layers
# and ordinary eager execution retain the native overlap behavior.
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts,SharedExpertsOrder
_shared_order=SharedExperts._determine_shared_experts_order
def shared_order(self,hidden_states):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    cap=BreakableCUDAGraphCapture.current()
    if _active is not None and cap is not None and cap._capturing:
        return SharedExpertsOrder.NO_OVERLAP
    return _shared_order(self,hidden_states)
SharedExperts._determine_shared_experts_order=shared_order
