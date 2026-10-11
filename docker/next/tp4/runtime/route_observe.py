"""Bounded diagnostic route capture, OFF during ordinary timing."""
import json,os
from pathlib import Path
from perf_control import options
_buffer=[]
def observe(layer,experts,phase):
    flags=options()
    if flags.get('route_trace'):
        _buffer.append((phase,layer,experts))
    if len(_buffer)>=512 or (_buffer and not flags.get('route_trace')):
        p=Path('/data/perf-test-20260927')/('routes-'+str(os.getpid())+'.jsonl')
        with p.open('a') as f:
            for row in _buffer:f.write(json.dumps(row,separators=(',',':'))+'\n')
        _buffer.clear()
