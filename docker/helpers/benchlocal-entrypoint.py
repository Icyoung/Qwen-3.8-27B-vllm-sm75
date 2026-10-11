#!/usr/bin/env python3
"""Fixed task entry point from the shared Ultra image."""
from pathlib import Path
import os,sys
ROLES={'bugfind-15','cli-40','hermesagent-20'}
def main():
 if os.environ.get('SM75_BENCHLOCAL_ROLE') not in ROLES or not os.environ.get('SM75_JOB_TOKEN'):
  raise ValueError('Use the configured on-demand launcher to start a BenchLocal task')
 if Path('/dev/nvidia0').exists():raise ValueError('BenchLocal tasks must not expose GPU devices')
 os.execv('/usr/local/bin/node',['node','--disable-sigusr1','/opt/sm75-workbench/benchlocal-verifiers/ondemand-gateway.mjs'])
if __name__=='__main__':
 try:main()
 except ValueError as error:print(str(error),file=sys.stderr);sys.exit(2)
