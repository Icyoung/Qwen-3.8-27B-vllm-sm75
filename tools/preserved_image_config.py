"""Allowlisted runtime configuration for the preserved OCI artifact, not inspect dumps."""
from copy import deepcopy
import re

PUBLIC_ENV = set("""NVIDIA_VISIBLE_DEVICES PATH NVARCH NVIDIA_REQUIRE_CUDA NV_CUDA_CUDART_VERSION CUDA_VERSION LD_LIBRARY_PATH NVIDIA_DRIVER_CAPABILITIES DEBIAN_FRONTEND CUDA_HOME UV_HTTP_TIMEOUT UV_INDEX_STRATEGY UV_LINK_MODE UV_PYTHON_INSTALL_DIR UV_CACHE_DIR UV_OVERRIDE VLLM_ENABLE_CUDA_COMPATIBILITY TORCH_CUDA_ARCH_LIST VLLM_USAGE_SOURCE VLLM_BUILD_COMMIT VLLM_BUILD_PIPELINE VLLM_BUILD_URL VLLM_IMAGE_TAG FLASH_QLA_SM75_PREBUILT_EXTENSION_PATH FLASH_QLA_SM75_ALLOW_JIT PYTHONWARNINGS TRANSFORMERS_VERBOSITY VLLM_FIREFLY VLLM_USE_MODELSCOPE MODELSCOPE_CACHE SM75_FA2_SMALLQ_GRAPH SM75_GDN_B1_METADATA SM75_CONSOLE_HOST SM75_CONSOLE_PORT SM75_SINGLE_CONTAINER NCCL_P2P_LEVEL SM75_EDITION SM75_BENCHLOCAL_RUNTIME MAX_JOBS SM75_BENCH_DOWNLOAD_CACHE CUDA_VISIBLE_DEVICES OMP_NUM_THREADS""".split())
FIELDS={'User','ExposedPorts','Env','Entrypoint','Cmd','Volumes','WorkingDir','Labels','StopSignal','StopTimeout','Healthcheck','Shell','ArgsEscaped','OnBuild','AttachStdin','AttachStdout','AttachStderr','Tty','OpenStdin','StdinOnce','Hostname','Domainname','Image'}

def make_config(golden, prepared):
    if golden['Id']!='sha256:bb9f2af2b99be6eff3cbda9cad67d09a9d1218f2549af017f3c6e0d3e0aafe9f':raise ValueError('Unreviewed preservation donor')
    if prepared['RootFS']['Layers'][:len(golden['RootFS']['Layers'])]!=golden['RootFS']['Layers']:raise ValueError('Prepared image is not derived from the reviewed donor')
    source=golden['Config']
    if set(source)-FIELDS:raise ValueError('Unreviewed OCI field names')
    seen=set()
    for entry in source.get('Env',[]):
        key,sep,value=entry.partition('=')
        if not sep or key not in PUBLIC_ENV:raise ValueError('Unreviewed environment name: '+key)
        if re.search(r'https?://[^/@\s]+:[^/@\s]+@|[?&](?:token|key|secret|password)=|\bBearer\s|\bsk-[A-Za-z0-9]',value,re.I):raise ValueError('Credential-like value in image environment: '+key)
        seen.add(key)
    cfg=deepcopy(source)
    for key,value in cfg.get('Labels',{}).items():
        if re.search(r'(?:api.?key|password|secret|access.?token)',key,re.I):raise ValueError('Credential-like image label name')
        if re.search(r'https?://[^/@\s]+:[^/@\s]+@|[?&](?:token|key|secret|password)=',str(value),re.I):raise ValueError('Credential-like image label URL')
    labels=cfg.setdefault('Labels',{})
    labels.update({'org.opencontainers.image.version':'0.1.7-ultra-beta2-preserve-test','org.opencontainers.image.revision':'local-preserve-20261010','io.sm75.next.tp4':'preserved-20261010','io.sm75.preserved-from':golden['Id'],'io.sm75.prepared-from':prepared['Id'],'io.sm75.runtime.folded':'1'})
    return {'architecture':'amd64','os':'linux','created':prepared['Created'],'config':cfg,'history':[{'created':prepared['Created'],'created_by':'Folded immutable official-vLLM-derived Ultra; complete dependencies; preserved TP4 integration'}]}


def main():
    import argparse, hashlib, json, os, sys
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    images=json.load(sys.stdin)
    if not isinstance(images,list) or len(images)!=2:raise ValueError('Expected donor and prepared image metadata over stdin')
    golden,prepared=images
    config=make_config(golden,prepared)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'validated-runtime-config.json'
    with path.open('x') as stream:json.dump(config,stream,separators=(',',':'))
    path.chmod(0o600)
    chain=prepared['RootFS']['Layers'][0]
    for diff in prepared['RootFS']['Layers'][1:]:chain='sha256:'+hashlib.sha256((chain+' '+diff).encode()).hexdigest()
    (args.output/'source-chain-id.txt').write_text(chain.split(':')[1]+'\n')
    print(json.dumps({'donor':golden['Id'],'prepared':prepared['Id'],'chainId':chain,'configuration':'allowlisted and credential-checked'}))

if __name__=='__main__':main()
