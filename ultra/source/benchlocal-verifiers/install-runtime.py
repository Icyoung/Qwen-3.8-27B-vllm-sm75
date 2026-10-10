#!/usr/bin/env python3
"""Install all BenchLocal execution dependencies inside the Ultra image.

Only the dedicated Hermes venv is modified; vLLM's Python packages are untouched.
Optional SM75_BENCH_DOWNLOAD_CACHE reuses verified public downloads at build time.
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile,tempfile,urllib.request,zipfile
HERE=Path(__file__).resolve().parent
TOOLS=Path('/opt/sm75-bench')
COMMIT='ea74f61d983ebdfd6a863c45761d1b38081f1d08'
REPOSITORY='https://github.com/nousresearch/hermes-agent.git'
def run(*args,**kw):
 print('+',args[0],args[1] if len(args)>1 else '',flush=True)
 subprocess.run(args,check=True,**kw)
def verify_versions(assets):
 checks=[('/usr/local/go/bin/go',['version'],'go'+assets['go']['version']),
  (str(TOOLS/'rust/bin/rustc'),['--version'],'rustc '+assets['rust']['version']),
  ('/usr/local/bin/agent-browser',['--version'],'agent-browser '+assets['agent-browser']['version']),
  (str(TOOLS/'chrome/chrome-linux64/chrome'),['--version'],assets['chrome']['version'])]
 for program,args,expected in checks:
  actual=subprocess.check_output([program,*args],text=True).strip()
  if expected not in actual:raise RuntimeError('Unexpected dependency version: '+program)
 return True
def install():
 if os.uname().machine!='x86_64':raise RuntimeError('The SM75 Ultra image targets Linux x86_64')
 assets=json.loads((HERE/'assets.lock.json').read_text())
 identity={'schema':1,'assets':assets,'hermesCommit':COMMIT,'pythonLockSha256':hashlib.sha256((HERE/'hermes-requirements.lock').read_bytes()).hexdigest()}
 marker=HERE/'runtime-lock.json'
 if marker.is_file() and json.loads(marker.read_text())==identity:
  verify_versions(assets)
  print('BenchLocal execution dependencies already installed');return
 TOOLS.mkdir(parents=True,exist_ok=True)
 env={**os.environ,'DEBIAN_FRONTEND':'noninteractive'}
 packages=['ca-certificates','git','gawk','jq','vim-common','python3-venv','python3-pytest','build-essential','xz-utils','unzip',
  'libnss3','libnspr4','libatk-bridge2.0-0t64','libatspi2.0-0t64','libcups2t64','libdrm2','libxkbcommon0','libxcomposite1',
  'libxdamage1','libxfixes3','libxrandr2','libgbm1','libasound2t64','libgtk-3-0t64','libxshmfence1','libx11-xcb1',
  'fonts-liberation','fonts-noto-color-emoji']
 run('apt-get','update',env=env)
 run('apt-get','install','--no-remove','-y','--no-install-recommends',*packages,env=env)
 with tempfile.TemporaryDirectory(prefix='sm75-bench-build-') as temp:
  temp=Path(temp)
  def fetch(key):
   item=assets[key];cached=Path(os.environ.get('SM75_BENCH_DOWNLOAD_CACHE',str(temp)))/item['file']
   if not cached.is_file():
    cached=temp/item['file']
    with urllib.request.urlopen(item['url'],timeout=120) as inp,cached.open('wb') as out:shutil.copyfileobj(inp,out)
   if hashlib.sha256(cached.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Download hash mismatch: '+key)
   return cached
  if not Path('/usr/local/go/bin/go').is_file():
   with tarfile.open(fetch('go')) as tar:tar.extractall('/usr/local',filter='data')
  if not (TOOLS/'rust/bin/rustc').is_file():
   dest=temp/'rust';dest.mkdir()
   with tarfile.open(fetch('rust')) as tar:tar.extractall(dest,filter='data')
   installer=next(dest.glob('*/install.sh'))
   run('bash',str(installer),'--prefix='+str(TOOLS/'rust'),'--disable-ldconfig','--components=rustc,rust-std-x86_64-unknown-linux-gnu,cargo')
  if not (TOOLS/'chrome/chrome-linux64/chrome').is_file():
   dest=TOOLS/'chrome';dest.mkdir(exist_ok=True)
   with zipfile.ZipFile(fetch('chrome')) as archive:
    for entry in archive.infolist():
     target=(dest/entry.filename).resolve()
     if not target.is_relative_to(dest.resolve()):raise RuntimeError('Invalid browser archive path')
     archive.extract(entry,dest)
     if target.is_file():target.chmod(((entry.external_attr>>16)&0o777) or 0o644)
  if not (TOOLS/'agent-browser/package/bin/agent-browser-linux-x64').is_file():
   dest=TOOLS/'agent-browser';dest.mkdir(exist_ok=True)
   with tarfile.open(fetch('agent-browser')) as tar:tar.extractall(dest,filter='data')
   for candidate in (dest/'package/bin').glob('agent-browser-*'):
    if candidate.name!='agent-browser-linux-x64':candidate.unlink()
  binary=TOOLS/'agent-browser/package/bin/agent-browser-linux-x64';binary.chmod(0o755)
  link=Path('/usr/local/bin/agent-browser')
  if not link.exists():link.symlink_to(binary)
 # A shallow, detached checkout retains upstream revision verification without full history.
 hermes=Path('/opt/hermes-agent')
 if not (hermes/'.git').is_dir():
  hermes.mkdir(exist_ok=True);run('git','init',str(hermes));run('git','-C',str(hermes),'remote','add','origin',REPOSITORY)
  run('git','-C',str(hermes),'fetch','--depth=1','origin',COMMIT)
  run('git','-C',str(hermes),'checkout','--detach','FETCH_HEAD')
 actual=subprocess.check_output(['git','-C',str(hermes),'rev-parse','HEAD'],text=True).strip()
 if actual!=COMMIT:raise RuntimeError('Hermes source version mismatch')
 run('git','config','--system','--add','safe.directory',str(hermes))
 run('python3','-m','venv','/opt/hermes-venv')
 python='/opt/hermes-venv/bin/python'
 run(python,'-m','pip','install','--no-cache-dir','-r',str(HERE/'hermes-requirements.lock'))
 run(python,'-m','pip','install','--no-cache-dir','--no-deps','--no-build-isolation','-e',str(hermes))
 run(python,'-m','pip','check')
 if subprocess.run(['getent','passwd','10001'],stdout=subprocess.DEVNULL).returncode:
  run('useradd','--create-home','--uid','10001','verifier')
 for name in ['/workspace','/home/verifier']:
  p=Path(name);p.mkdir(exist_ok=True);p.chmod(0o700);os.chown(p,10001,10001)
 run('/usr/local/go/bin/go','version');run(str(TOOLS/'rust/bin/rustc'),'--version');run('agent-browser','--version')
 run(str(TOOLS/'chrome/chrome-linux64/chrome'),'--version')
 verify_versions(assets)
 marker.write_text(json.dumps(identity,indent=2)+'\n')
 run('apt-get','clean')
 lists=Path('/var/lib/apt/lists')
 for f in lists.iterdir():
  if f.is_file():f.unlink()
 print('BENCHLOCAL_SINGLE_IMAGE_READY',flush=True)
if __name__=='__main__':install()
