"""Install the model-scoped Flash Next overlay, validating every file first."""
from pathlib import Path, PurePosixPath
import ast, hashlib, importlib.util, json, shutil

def install(payload, package):
    manifest=json.loads((payload/'manifest.json').read_text())
    changes=[]
    for name,meta in manifest['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='vllm':
            raise ValueError(f'Invalid overlay path: {name}')
        if not (name.startswith('vllm/models/qwen4_exp/') or name=='vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py'):
            raise ValueError(f'Next overlay cannot modify another model: {name}')
        source=payload/'overlay'/name; data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError(f'Payload mismatch: {name}')
        ast.parse(data.decode(),filename=name)
        target=package/Path(*relative.parts[1:])
        current=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
        if current not in (meta['sha256'],meta['base_sha256'],meta.get('previous_sha256',meta['base_sha256'])):
            raise ValueError(f'Unreviewed upstream file: {name}, SHA256={current}')
        changes.append((source,target))
    for source,target in changes:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    return manifest

if __name__=='__main__':
    package=Path(importlib.util.find_spec('vllm').origin).parent
    manifest=install(Path(__file__).resolve().parent,package)
    out=Path('/opt/vllm-sm75/next-manifest.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Installed {len(manifest["files"])} reviewed Flash Next files')
