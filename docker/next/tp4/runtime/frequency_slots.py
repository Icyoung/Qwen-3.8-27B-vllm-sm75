"""Same LFU eviction order with a bounded heap instead of full-slot scans."""
from collections import OrderedDict
import heapq
class FrequencySlots:
    policy_name='lfu'
    def __init__(self,capacity,logical_count=None):
        self.logical_count=capacity if logical_count is None else logical_count
        self.capacity=capacity;self.entries=OrderedDict();self.free=list(reversed(range(capacity)))
        self.frequency={};self.last={};self.clock=0;self.phase=None;self.decode_calls=0
        self.order={};self.serial=0;self.heap=[]
    def _item(self,key):
        return (self.frequency.get(key,0),self.last[key],self.order[key],key,self.entries[key])
    def _rebuild(self):
        self.heap=[self._item(k) for k in self.entries];heapq.heapify(self.heap)
    def _valid(self,item):
        freq,last,order,key,slot=item
        return (self.entries.get(key)==slot and self.order.get(key)==order and
                self.last.get(key)==last and self.frequency.get(key,0)==freq)
    def note_phase(self,phase):
        if phase=='prefill' and self.phase=='decode':
            self.frequency.clear();self.decode_calls=0;self._rebuild()
        self.phase=phase
    def reset_full(self,layer):
        self.entries=OrderedDict(((layer,i),i) for i in range(self.logical_count));self.free=list(reversed(range(self.logical_count,self.capacity)))
        self.last={key:self.clock for key in self.entries};self.order={}
        for key in self.entries:self.serial+=1;self.order[key]=self.serial
        self._rebuild()
    def acquire(self,layer,experts):
        keys=[(layer,e) for e in experts]
        if len(set(keys))!=len(keys) or len(keys)>self.capacity:raise ValueError('Invalid expert set')
        if any(e<0 or e>=self.logical_count for e in experts):raise ValueError('Invalid expert index')
        protected=set(keys);self.clock+=1
        if self.phase=='decode':
            self.decode_calls+=1
            if self.decode_calls%128==0:
                self.frequency={k:v*.5 for k,v in self.frequency.items()}
                self._rebuild()
            for k in keys:self.frequency[k]=self.frequency.get(k,0)+1
        result={};misses=[];evicted=0;held=[]
        for k in keys:
            if k in self.entries:
                slot=self.entries[k];self.entries.move_to_end(k)
            else:
                if self.free:slot=self.free.pop()
                else:
                    while True:
                        item=heapq.heappop(self.heap)
                        if not self._valid(item):continue
                        if item[3] in protected:held.append(item);continue
                        victim=item[3];slot=self.entries.pop(victim)
                        self.last.pop(victim,None);self.order.pop(victim,None);evicted+=1;break
                self.entries[k]=slot;misses.append((k[1],slot))
            self.last[k]=self.clock;self.serial+=1;self.order[k]=self.serial
            heapq.heappush(self.heap,self._item(k));result[k[1]]=slot
        for item in held:
            if self._valid(item):heapq.heappush(self.heap,item)
        if len(self.heap)>4*self.capacity:self._rebuild()
        assert len(set(self.entries.values()))==len(self.entries)
        return result,misses,evicted
