#!/usr/bin/env python3
"""Assemble a portable Docker archive from a verified folded layer and OCI config.

Only the sanitized runtime configuration supplied on stdin is serialized, not
Docker inspect output. No Docker socket, user filesystem or credentials needed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import sys

p=argparse.ArgumentParser()
p.add_argument('--layer',type=Path,required=True)
p.add_argument('--layer-info',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--tag',required=True)
a=p.parse_args()
if not re.fullmatch(r'local/[a-z0-9][a-z0-9_./:-]+',a.tag):raise ValueError('Only a local candidate tag is permitted')
if a.output.exists():raise ValueError('Refusing to replace an existing artifact')
config=json.load(sys.stdin)
if set(config)-{'architecture','os','created','config','rootfs','history'}:raise ValueError('Unexpected OCI config key')
if config['architecture']!='amd64' or config['os']!='linux':raise ValueError('Wrong platform')
info=json.loads(a.layer_info.read_text())
if not re.fullmatch(r'sha256:[a-f0-9]{64}',info['diffId']):raise ValueError('Invalid layer digest')
if a.layer.stat().st_size!=info['bytes']:raise ValueError('Layer size changed')
config['rootfs']={'type':'layers','diff_ids':[info['diffId']]}
blob=json.dumps(config,sort_keys=True,separators=(',',':')).encode()
image='sha256:'+hashlib.sha256(blob).hexdigest()
config_name=image.split(':')[1]+'.json'
manifest=json.dumps([{'Config':config_name,'RepoTags':[a.tag],'Layers':['runtime/layer.tar']}],separators=(',',':')).encode()
partial=a.output.with_suffix(a.output.suffix+'.partial')
if partial.exists():raise ValueError('Previous incomplete artifact needs inspection')
with tarfile.open(partial,'x',format=tarfile.PAX_FORMAT) as archive:
    def add(name,data):
        entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o644
        archive.addfile(entry,io.BytesIO(data))
    add(config_name,blob);add('manifest.json',manifest)
    entry=tarfile.TarInfo('runtime/layer.tar');entry.size=info['bytes'];entry.mode=0o644
    with a.layer.open('rb') as source:archive.addfile(entry,source)
partial.rename(a.output)
result={'image':image,'tag':a.tag,'bytes':a.output.stat().st_size,'layerBytes':info['bytes'],'diffId':info['diffId'],'portable':True}
a.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
