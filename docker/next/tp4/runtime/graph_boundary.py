import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture, BreakableCUDAGraphWrapper

def clone_tree(x):
 if isinstance(x,torch.Tensor):return x.clone()
 if isinstance(x,tuple):return tuple(clone_tree(v) for v in x)
 if isinstance(x,list):return [clone_tree(v) for v in x]
 if x is None:return None
 raise TypeError(type(x))

def copy_tree(dst,src):
 if isinstance(dst,torch.Tensor):
  assert dst.shape==src.shape and dst.dtype==src.dtype
  dst.copy_(src);return
 if dst is None:
  assert src is None;return
 assert type(dst)==type(src) and len(dst)==len(src)
 for a,b in zip(dst,src):copy_tree(a,b)

def eager_output(fn,*args,**kw):
 cap=BreakableCUDAGraphCapture.current()
 if cap is None or not cap._capturing:return fn(*args,**kw)
 # Bind owner/inputs through the closure and keep output storage stable.
 # Strong references are deliberate for the sole captured M=1 shape.
 held=[]
 def run():
  result=fn(*args,**kw)
  if not held:held.append(clone_tree(result))
  else:copy_tree(held[0],result)
  return held[0]
 return cap.add_eager(run)

_replay=BreakableCUDAGraphWrapper._replay

def replay(self,*args,**kwargs):
 self._tp4_replay_count=getattr(self,'_tp4_replay_count',0)+1
 return _replay(self,*args,**kwargs)
BreakableCUDAGraphWrapper._replay=replay

def status(model):
 result=[]
 for w in BreakableCUDAGraphWrapper._all_instances:
  result.append(dict(replays=getattr(w,'_tp4_replay_count',0),entries=[dict(graphs=e.capture._num_graphs,breaks=e.capture._num_eager_breaks) for e in w.entries.values() if e.capture is not None]))
 return result

# Host routing needs values produced by preceding captured segments even while
# the plan is being built. CUDA capture alone does not execute those kernels.
_end_segment=BreakableCUDAGraphCapture._end_segment

def end_segment_and_materialize(self):
 if not self._capturing:return _end_segment(self)
 _end_segment(self)
 self.segments[-1]()
BreakableCUDAGraphCapture._end_segment=end_segment_and_materialize


def reset_checks(model):
 import tp4_routed_hotcache as cache
 cache._verified.clear()
 return True

def finish_checks(model):
 import os
 import tp4_routed_hotcache as cache
 os.environ['TP4_VERIFY']='0'
 return len(cache._verified)
