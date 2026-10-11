#!/usr/bin/env python3
"""Reinstall the complete source overlay in one image layer."""
from pathlib import Path
import shutil,subprocess,sys,hashlib,json
from importlib import metadata
SOURCE=Path(sys.argv[1] if len(sys.argv)>1 else '/source').resolve()
def run(*args):subprocess.run(args,check=True)
def copytree(a,b):
 shutil.copytree(a,b,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
 # These are image-owned public program trees, not /data or /dsh. Archive
 # extraction under umask 077 must not make the non-root verifier unreadable.
 b.chmod(0o755)
 for directory in b.rglob('*'):
  if directory.is_dir() and not directory.is_symlink():directory.chmod(0o755)
helpers=SOURCE/'docker/helpers'
host=Path('/opt/vllm-sm75/benchlocal-host');(host/'helpers').mkdir(parents=True,exist_ok=True)
shutil.copyfile(SOURCE/'docker/benchlocal.sh',host/'benchlocal.sh')
shutil.copyfile(helpers/'benchlocal-service.sh',host/'helpers/benchlocal-service.sh')
run('python3',str(helpers/'verify_cuda_libraries.py'))
copytree(SOURCE/'docker/speculative',Path('/opt/vllm-sm75/speculative'))
run('python3',str(helpers/'install_sm75_overlay.py'),str(SOURCE/'vllm'),'--source-copy','/opt/vllm-sm75/source/vllm')
run('python3',str(helpers/'install_speculative.py'))
run('python3',str(helpers/'verify_sm75_image.py'))
copytree(SOURCE/'docker/next',Path('/opt/vllm-sm75/next'))
run('python3','/opt/vllm-sm75/next/install.py')
copytree(SOURCE/'docker/next/templates',Path('/opt/vllm-sm75/templates'))
copytree(SOURCE/'ultra/source/profiles',Path('/opt/vllm-sm75/profiles'))
copytree(SOURCE/'ultra/source/lmcache',Path('/opt/vllm-sm75/lmcache'))
lm_patch=json.loads((SOURCE/'ultra/source/lmcache/patches/packed4d-provenance.json').read_text())
try:
 lm_dist=metadata.distribution('lmcache')
 lm_target=Path(lm_dist.locate_file(lm_patch['module_path']))
 lm_ready=lm_dist.version=='0.5.5' and hashlib.sha256(lm_target.read_bytes()).hexdigest()==lm_patch['patched_file_sha256']
except (metadata.PackageNotFoundError,OSError):lm_ready=False
if not lm_ready:run('python3','/opt/vllm-sm75/lmcache/install.py','--install','--build-no-gpu')
for name in ['lmcache-entrypoint.sh','lmcache-launch.py','runtime-entrypoint.py','cpu-kv-check.py','benchlocal-entrypoint.py','pstate-entrypoint.sh']:
 shutil.copyfile(helpers/name,Path('/opt/vllm-sm75')/name)
# Replace image-owned application files; persistent /data and /dsh are untouched.
for name in ['console','profiles','benchlocal-native']:
 target=Path('/opt/sm75-workbench')/name
 if target.is_symlink():target.unlink()
 elif target.is_dir():shutil.rmtree(target)
copytree(SOURCE/'ultra/source',Path('/opt/sm75-workbench'))
for name in ['font-scale.py','pstate-supervisor.sh']:shutil.copyfile(SOURCE/'ultra/overlay'/name,Path('/opt/vllm-sm75')/name)
run('node','-e','const fs=require("fs");if(process.version!=="v22.23.2"||JSON.parse(fs.readFileSync("/opt/harness/node_modules/@deepseek-ai/dsh/package.json")).version!=="0.1.7-alpha.2")throw Error("Unreviewed Ultra dependencies")')
theme=Path('/opt/harness/node_modules/@deepseek-ai/dsh-client-ui-theme')
if theme.is_symlink():theme.unlink()
elif theme.is_dir():shutil.rmtree(theme)
theme.mkdir(parents=True)
run('tar','-xzf',str(SOURCE/'build-dependencies/harness-theme.tgz'),'-C',str(theme),'--strip-components=1')
run('python3','/opt/vllm-sm75/font-scale.py')
run('node','/opt/sm75-workbench/console/install-native-plugins.mjs')
run('node','/opt/sm75-workbench/console/verify-client-bundles.mjs')
for f in Path('/opt/vllm-sm75').glob('*entrypoint*'):f.chmod(0o755)
Path('/opt/vllm-sm75/pstate-supervisor.sh').chmod(0o755)
run('python3','/opt/sm75-workbench/benchlocal-verifiers/install-runtime.py')
print('ULTRA_BETA2_SOURCE_INSTALLED',flush=True)
