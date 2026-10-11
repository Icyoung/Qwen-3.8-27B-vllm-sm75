"""TP4 runtime policy; optional local controls for bounded verification."""
import json,os,time
from pathlib import Path
_next=0.0
_options={}
_defaults={'expert_policy':'lfu','cache_capacity':256,'bulk_threshold':120,'bulk_by_misses':True,
           'verify_cache':False,'verify_prefill':False}
def options():
    global _next,_options
    now=time.monotonic()
    if now>=_next:
        _next=now+.25
        path=Path(os.environ.get('TP4_CONTROL_FILE','/data/tp4-control.json'))
        try:
            value=json.loads(path.read_text())
            _options={**_defaults,**(value if isinstance(value,dict) else {})}
        except (OSError,ValueError):_options=dict(_defaults)
    return _options
