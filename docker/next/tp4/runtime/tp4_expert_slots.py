"""Physical-slot LRU; keys include layer identity. No GPU dependencies."""
from collections import OrderedDict
class ExpertSlots:
    def __init__(self, capacity, logical_count=None):
        self.logical_count=capacity if logical_count is None else logical_count
        self.capacity=capacity
        self.entries=OrderedDict()
        self.free=list(reversed(range(capacity)))
    def reset_full(self, layer):
        self.entries=OrderedDict(((layer,i),i) for i in range(self.logical_count))
        self.free=list(reversed(range(self.logical_count,self.capacity)))
    def acquire(self, layer, experts):
        keys=[(layer,e) for e in experts]
        if len(set(keys)) != len(keys) or len(keys)>self.capacity:
            raise ValueError('Invalid expert set')
        protected=set(keys)
        result={}; misses=[]; evictions=0
        for key in keys:
            if key in self.entries:
                slot=self.entries[key]
                self.entries.move_to_end(key)
            else:
                if self.free:slot=self.free.pop()
                else:
                    victim=next(k for k in self.entries if k not in protected)
                    slot=self.entries.pop(victim);evictions+=1
                self.entries[key]=slot;misses.append((key[1],slot))
            result[key[1]]=slot
        assert len(set(self.entries.values()))==len(self.entries)
        return result,misses,evictions
if __name__=='__main__':
    import random
    rng=random.Random(19); cache=ExpertSlots(16); physical={};hits=0;evictions=0
    for step in range(10000):
        layer=rng.randrange(13);experts=rng.sample(range(16),rng.randrange(1,17))
        if step%11==0:
            cache.reset_full(layer);physical={i:(layer,i) for i in range(16)}
        mapping,misses,evicted=cache.acquire(layer,experts)
        for expert,slot in misses:physical[slot]=(layer,expert)
        for expert,slot in mapping.items():assert physical[slot]==(layer,expert)
        hits+=len(experts)-len(misses);evictions+=evicted
    assert hits and evictions
    print('PASS 10000 interleaved layer requests, hits',hits,'evictions',evictions)
