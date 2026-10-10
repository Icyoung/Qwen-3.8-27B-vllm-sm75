#!/usr/bin/env python3
"""CPU-only verification role of the same Ultra image."""
from pathlib import Path
import os,sys
ROLES = {'bugfind-15': 4010, 'cli-40': 4040, 'hermesagent-20': 4010}
def main():
 role=os.environ.get('SM75_BENCHLOCAL_ROLE','')
 if role not in ROLES:raise ValueError('Unknown BenchLocal role')
 root=Path('/opt/sm75-workbench/benchlocal-verifiers')
 if not (root/'runtime-lock.json').is_file():raise ValueError('BenchLocal runtime is not installed')
 if Path('/dev/nvidia0').exists():raise ValueError('BenchLocal workers must run without GPU devices')
 env=dict(os.environ)
 env.update(SM75_BENCH_VERIFIER_ROOT=str(root/role),SM75_VERIFIER_INNER_PORT=str(ROLES[role]),
  NVIDIA_VISIBLE_DEVICES='void',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
  PATH='/opt/hermes-venv/bin:/opt/sm75-bench/rust/bin:/usr/local/go/bin:'+env.get('PATH',''),
  AGENT_BROWSER_EXECUTABLE_PATH='/opt/sm75-bench/chrome/chrome-linux64/chrome')
 os.execvpe('/usr/local/bin/node',['node',str(root/'gateway.mjs')],env)
if __name__=='__main__':
 try:main()
 except ValueError as exc:print(str(exc),file=sys.stderr);sys.exit(2)
