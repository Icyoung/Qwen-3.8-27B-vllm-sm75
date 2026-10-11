#!/usr/bin/env python3
"""Pack a read-only immutable-image filesystem, retaining POSIX metadata.

Run inside the bounded CPU packer, with only /image-root:ro and /output mounted.
Never mount user model/data directories or the Docker socket in this helper.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root=Path('/image-root')
out=Path('/output')
if not (root/'opt/vllm-sm75/runtime-entrypoint.py').is_file():raise RuntimeError('Wrong immutable source')
for name in ['root/.ssh/id_rsa','root/.ssh/id_ed25519','root/.aws/credentials','root/.cache/huggingface/token','root/.huggingface/token']:
 p=root/name
 if p.is_file() and p.stat().st_size:raise RuntimeError('Refusing an image-owned credential file: '+name)
layer=out/'runtime.layer.tar'
if layer.exists():raise RuntimeError('Refusing to overwrite a previous layer')
subprocess.run(['tar','--format=pax','--pax-option=delete=atime,delete=ctime','--xattrs','--xattrs-include=*','--acls','--numeric-owner','--sort=name','--one-file-system','-C',str(root),'-cf',str(layer),'.'],check=True)
h=hashlib.sha256()
with layer.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
result={'diffId':'sha256:'+h.hexdigest(),'bytes':layer.stat().st_size,'sourceAccess':'read-only image filesystem','preserves':['uid','gid','mode','mtime','symlink','hardlink','xattrs','acl']}
(out/'runtime-layer.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
