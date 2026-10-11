"""Selectable cache policy and exact physical-weight mapping checks."""
import os
from perf_control import options
from tp4_expert_slots import ExpertSlots
from frequency_slots import FrequencySlots
_verified=set()
def get_pool(pools,identity,count,phase):
    from wide_expert_cache import MAX_CAPACITY
    opts=options()
    capacity=min(int(opts.get('cache_capacity',MAX_CAPACITY)),MAX_CAPACITY)
    assert capacity>=count
    threshold=int(opts.get('bulk_threshold',64));assert 32<=threshold<=128
    by_misses=bool(opts.get('bulk_by_misses',False))
    mode=opts.get('expert_policy','lru')
    assert mode in ('lru','lfu')
    pool=pools.get(identity)
    if pool is None or getattr(pool,'policy_name','lru')!=mode or pool.capacity!=capacity or getattr(pool,'bulk_threshold',64)!=threshold or getattr(pool,'bulk_by_misses',False)!=by_misses:
        import sys,json
        owner=sys.modules.get('tp4_routed_hotcache')
        if owner is not None:
            print('TP4_CACHE_BOUNDARY',os.getpid(),json.dumps({'previous_capacity':None if pool is None else pool.capacity,'capacity':capacity,'bulk_threshold':threshold,'bulk_by_misses':by_misses,'stats':owner._stats}),flush=True)
        pool=FrequencySlots(capacity,count) if mode=='lfu' else ExpertSlots(capacity,count)
        pool.bulk_threshold=threshold;pool.bulk_by_misses=by_misses
        pools[identity]=pool
        print('TP4_CACHE_POLICY',os.getpid(),mode,'capacity',capacity,flush=True)
    if mode=='lfu':pool.note_phase(phase)
    return pool

def verify(original,kw,physical_ids,selected,count,layer,params,phase,stat,rowbytes,hit,pool,ids,emap,identity):
    if options().get('verify_prefill') and phase=='prefill' and selected:
        return verify_prefill_once(original,kw,physical_ids,selected,count,layer,params,stat,rowbytes,pool,ids,emap,identity)
    mode=getattr(pool,'policy_name','lru');case=(mode,layer,'hit' if hit else 'miss')
    if not options().get('verify_cache') or phase!='decode' or not selected or case in _verified:return physical_ids
    import torch
    from fast_route import remap
    max_slot=int(physical_ids.max().item())
    before=original(**dict(kw,expert_ids=physical_ids)).clone()
    for _,p in params:p._gpu_buffer.copy_(p._cpu_storage,non_blocking=p._cpu_storage.is_pinned())
    pool.reset_full(layer)
    reference=original(**kw)
    local=ids.long()
    if emap is not None:local=torch.where(local>=0,emap[local.clamp_min(0)],-1)
    valid=local.reshape(-1)>=0
    torch.testing.assert_close(before.reshape(-1,before.shape[-1])[valid],reference.reshape(-1,reference.shape[-1])[valid],rtol=0,atol=0)
    stat['reference_bytes']+=rowbytes*count;_verified.add(case)
    # Full reference loading replaced the physical slot layout. Return a fresh
    # identity remap so the subsequent captured Marlin kernel reads valid rows.
    physical_ids=remap(kw['expert_ids'],{e:e for e in selected},count,identity)
    print('TP4_CACHE_VERIFIED',os.getpid(),mode,layer,case[-1],'bitwise','max_slot',max_slot,flush=True)
    return physical_ids

_prefill_verified=set()
def verify_prefill_once(original,kw,physical_ids,selected,count,layer,params,stat,rowbytes,pool,ids,emap,identity):
    if layer in _prefill_verified:return physical_ids
    import torch
    from fast_route import remap
    local=ids.long()
    if emap is not None:local=torch.where(local>=0,emap[local.clamp_min(0)],-1)
    valid=local.reshape(-1)>=0
    result=original(**dict(kw,expert_ids=physical_ids))
    before=result.reshape(-1,result.shape[-1])[valid].detach().cpu()
    for _,p in params:p._gpu_buffer.copy_(p._cpu_storage,non_blocking=p._cpu_storage.is_pinned())
    pool.reset_full(layer)
    result=original(**kw)
    reference=result.reshape(-1,result.shape[-1])[valid].detach().cpu()
    torch.testing.assert_close(before,reference,rtol=0.001,atol=0.01)
    exact=torch.equal(before,reference)
    error=float((before-reference).abs().max()) if before.numel() else 0.0
    stat['reference_bytes']+=rowbytes*count;_prefill_verified.add(layer)
    print('TP4_PREFILL_VERIFIED',os.getpid(),layer,'rows',ids.shape[0],'bitwise',exact,'max_abs',error,flush=True)
    return remap(kw['expert_ids'],{e:e for e in selected},count,identity)
