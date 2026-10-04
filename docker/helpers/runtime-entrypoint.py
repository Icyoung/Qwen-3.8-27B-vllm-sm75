#!/usr/bin/env python3
"""Shared model profile launcher; raw vLLM arguments remain fully supported."""
from pathlib import Path
import json, os, re, sys

def profile_root():
    installed=Path('/opt/vllm-sm75/profiles')
    return installed if installed.is_dir() else Path(__file__).resolve().parents[2]/'ultra/source/profiles'

def load_profile(identifier, env):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}',identifier):raise ValueError('Invalid profile ID')
    data=json.loads((profile_root()/(identifier+'.json')).read_text(encoding='utf-8'))
    model=env.get('MODEL') or data['defaultModel']
    draft=env.get('DRAFT_MODEL') or data.get('defaultDraftModel') or model+'/runtime/mtp-int4-g32'
    def expand(value):
        if isinstance(value,str):return value.replace('${MODEL}',model).replace('${DRAFT_MODEL}',draft)
        if isinstance(value,list):return [expand(v) for v in value]
        if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
        return value
    data=expand(data)
    data['args']=[v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,separators=(',',':')) for v in data['args']]
    return data

def resolve(argv, env):
    argv=list(argv)
    if argv[:2]==['vllm','serve']:argv=argv[2:]
    elif argv[:1]==['serve']:argv=argv[1:]
    identifier=env.get('SM75_PROFILE','')
    if argv[:1]==['--profile']:
        if len(argv)<2:raise ValueError('--profile requires an ID')
        identifier=argv[1];argv=argv[2:]
    if identifier:
        data=load_profile(identifier,env)
        # Explicit CLI groups replace the preset's matching option. Preserve
        # multi-value arguments instead of guessing the vLLM parser's arity.
        groups=[]
        for arg in argv:
            if arg.startswith('--'):
                groups.append([arg])
            elif groups:
                groups[-1].append(arg)
            else:
                raise ValueError('Use MODEL to change a named profile model path')
        for group in groups:
            flag=group[0].split('=',1)[0]
            opposed='--'+flag[5:] if flag.startswith('--no-') else '--no-'+flag[2:]
            old=data['args']; keep=[old[0]]; i=1
            while i<len(old):
                if old[i].split('=',1)[0] in (flag,opposed):
                    i+=1
                    while i<len(old) and not old[i].startswith('--'):i+=1
                else:keep.append(old[i]);i+=1
            data['args']=keep+group
        # Template kernel values are defaults; explicitly supplied Docker
        # environment values must reach Standard and the Ultra profile alike.
        data['env']={k:env.get(k,v) for k,v in data['env'].items()}
        return data
    return {'id':'docker-startup','name':'Docker 启动参数','format':'custom','args':argv,'env':{}}

def main():
    env=dict(os.environ); edition=env.get('SM75_EDITION','standard')
    if edition=='ultra':env.setdefault('SM75_CONSOLE_ROOT','/data')
    supplied=bool(sys.argv[1:] or env.get('SM75_PROFILE'))
    if edition=='ultra' and not supplied:
        os.execvpe('/usr/local/bin/node',['node','/opt/sm75-workbench/console/server.mjs'],env)
    data=resolve(sys.argv[1:],env)
    args=data['args']
    if edition=='ultra' and args and not any(a in ('-h','--help','--version') for a in args):
        if any(a=='--api-key' or a.startswith('--api-key=') for a in args):
            raise ValueError('Set VLLM_API_KEY as an environment variable for Ultra')
        def value(flag,default):
            result=default
            for i,a in enumerate(args):
                if a==flag and i+1<len(args):result=args[i+1]
                elif a.startswith(flag+'='):result=a.partition('=')[2]
            return result
        port=int(value('--port',8000));tp=int(value('--tensor-parallel-size',1))
        if not 1024<=port<=65535 or not 1<=tp<=256:raise ValueError('Invalid port or tensor parallel size')
        data.setdefault('power',{'mode':'pstate','idleSeconds':1,'util':5,'confirm':60,'low':8,'high':16,'poll':5,'gpus':','.join(map(str,range(tp)))})
        data.update(backend='native',port=port,cacheRoot=env.get('SM75_CACHE_ROOT','/data/cache'),binds=[])
        env['SM75_LAUNCH_PROFILE']=json.dumps(data,ensure_ascii=False)
        os.execvpe('/usr/local/bin/node',['node','/opt/sm75-workbench/console/server.mjs'],env)
    # Template defaults belong to this engine, not to the Ultra manager or its
    # model download jobs. The manager applies data.env when it starts the model.
    for key,value in data['env'].items():env.setdefault(key,str(value))
    if env.get('LMCACHE_ENABLED','').lower() in ('1','true','yes','on'):
        os.execvpe('/opt/vllm-sm75/lmcache-entrypoint.sh',['lmcache-entrypoint.sh',*args],env)
    if env.get('POWER_MODE')=='pstate':
        for field,key in [('idleSeconds','PSTATE_IDLE_TIMEOUT'),('gpus','PSTATE_GPUS')]:
            if field in data.get('power',{}):env.setdefault(key,str(data['power'][field]))
        os.execvpe('/opt/vllm-sm75/pstate-entrypoint.sh',['pstate-entrypoint.sh',*args],env)
    os.execvpe('vllm',['vllm','serve',*args],env)

if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,json.JSONDecodeError) as exc:
        print(f'Model launch configuration error: {exc}',file=sys.stderr);sys.exit(2)
