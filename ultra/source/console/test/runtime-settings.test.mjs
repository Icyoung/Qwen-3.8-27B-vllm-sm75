import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {applyRuntimeEnvironment, hasRuntimeOverrides, runtimeArgumentEnvironment, runtimeLMCacheEnvironment, runtimeBooleanEnvironment, runtimeKernelEnvironment, runtimeEngineEnvironment} from '../runtime-settings.mjs';
import {argValue, readContextSettings, applyContextSettings} from '../public/context-config.js';
import {makeCommand} from '../config.mjs';
const profile = () => ({id:'test',port:8000,format:'fp8',cacheRoot:'/cache',power:{mode:'sleep'},
  args:['/models/model','--tensor-parallel-size','4','--tensor-parallel-size=3','--custom-unknown','keep',
    '--hf-overrides','{"dtype":"float16"}','--auto-sleep-idle-timeout','30',
    '--speculative-config','{"method":"mtp","num_speculative_tokens":5}',
    '--compilation-config','{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6]}'],env:{NCCL_P2P_LEVEL:'PHB'}});

test('startup import is pure, updates known flags once and preserves advanced arguments',()=>{
  const old=profile(), before=structuredClone(old);
  const next=applyRuntimeEnvironment(old,{TENSOR_PARALLEL_SIZE:'2',MAX_NUM_SEQS:'8',HF_OVERRIDES:'{"dtype":"float16","rope_scaling":{"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768}}'});
  assert.deepEqual(old,before);
  assert.equal(argValue(next.args,'--tensor-parallel-size'),'2');
  assert.equal(next.args.filter(x=>x==='--tensor-parallel-size').length,1);
  assert.equal(argValue(next.args,'--custom-unknown'),'keep');
  assert.equal(JSON.parse(argValue(next.args,'--hf-overrides')).rope_scaling.factor,4);
});

test('panel edits after import are not overwritten on the next engine launch',()=>{
  const next=applyRuntimeEnvironment(profile(),{TENSOR_PARALLEL_SIZE:'2'});
  next.args[next.args.indexOf('--tensor-parallel-size')+1]='1';
  const command=makeCommand(next,'test-key',{environment:{TENSOR_PARALLEL_SIZE:'2'}});
  assert.equal(command.args[command.args.indexOf('--tensor-parallel-size')+1],'1');
});

test('zero CPU/GPU KV removes limits; explicit connector JSON wins over convenience CPU setting',()=>{
  const source=profile();source.args.push('--kv-cache-memory-bytes','123','--kv-offloading-size','8','--kv-offloading-backend','native');
  const off=applyRuntimeEnvironment(source,{CPU_KV_GIB:'0',GPU_KV_BYTES:'0'});
  for(const flag of ['--kv-cache-memory-bytes','--kv-offloading-size','--kv-offloading-backend','--kv-transfer-config']) assert.equal(argValue(off.args,flag),null);
  const custom='{"kv_connector":"Custom","kv_role":"kv_both"}';
  assert.equal(argValue(applyRuntimeEnvironment(source,{CPU_KV_GIB:'8',KV_TRANSFER_CONFIG:custom}).args,'--kv-transfer-config'),custom);
});

test('speculative override invalidates only derived capture sizes and preserves explicit compilation JSON',()=>{
  const changed=applyRuntimeEnvironment(profile(),{SPECULATIVE_CONFIG:'{"method":"mtp","num_speculative_tokens":3}'});
  assert.equal(JSON.parse(argValue(changed.args,'--compilation-config')).cudagraph_capture_sizes,undefined);
  assert.match(argValue(changed.args,'--scheduler-cls'),/SM75Scheduler/);
  const off=applyRuntimeEnvironment(profile(),{SPECULATIVE_CONFIG:''});
  assert.equal(argValue(off.args,'--speculative-config'),null);assert.equal(argValue(off.args,'--scheduler-cls'),null);
});

test('power units and mode remain separate and explicit false values survive',()=>{
  const pstate=applyRuntimeEnvironment(profile(),{POWER_MODE:'pstate',PSTATE_IDLE_TIMEOUT:'1800',ASYNC_SCHEDULING:'0',ENABLE_AUTO_TOOL_CHOICE:'false'});
  assert.equal(pstate.power.idleSeconds,1800);assert.equal(argValue(pstate.args,'--auto-sleep-idle-timeout'),'0');
  assert.ok(pstate.args.includes('--no-async-scheduling'));
  assert.ok(!pstate.args.includes('--enable-auto-tool-choice'));
  const sleep=applyRuntimeEnvironment(profile(),{POWER_MODE:'sleep',AUTO_SLEEP_IDLE_TIMEOUT:'30',AUTO_SLEEP_OFFLOAD_TARGET:'cpu'});
  assert.ok(sleep.args.includes('--enable-sleep-mode'));assert.equal(argValue(sleep.args,'--auto-sleep-idle-timeout'),'30');
  assert.throws(()=>applyRuntimeEnvironment(profile(),{ASYNC_SCHEDULING:'maybe'}));
});

test('NCCL SYS is only a fallback and profile/container empty explicit values survive',()=>{
  assert.equal(runtimeEngineEnvironment({},{}).NCCL_P2P_LEVEL,'SYS');
  assert.equal(runtimeEngineEnvironment({},{NCCL_P2P_LEVEL:''}).NCCL_P2P_LEVEL,'');
  assert.equal(runtimeEngineEnvironment({NCCL_P2P_LEVEL:'PHB'},{NCCL_P2P_LEVEL:'PIX'}).NCCL_P2P_LEVEL,'PHB');
  assert.equal(runtimeEngineEnvironment({NCCL_P2P_LEVEL:''},{NCCL_P2P_LEVEL:'PIX'}).NCCL_P2P_LEVEL,'');
});

test('Docker allowlist covers every importable setting and image defaults alone do not import',()=>{
  const text=fs.readFileSync(new URL('../../../../docker/helpers/runtime-env.list',import.meta.url),'utf8');
  for(const key of [...Object.keys(runtimeArgumentEnvironment),...Object.keys(runtimeLMCacheEnvironment),...Object.keys(runtimeBooleanEnvironment),...runtimeKernelEnvironment.filter(x=>x!=='NCCL_P2P_LEVEL')]) assert.ok(text.split(/\r?\n/).includes(key),key);
  assert.equal(hasRuntimeOverrides({NCCL_P2P_LEVEL:'SYS',VLLM_FIREFLY:'1'}),false);
  assert.equal(hasRuntimeOverrides({CPU_KV_GIB:'0'}),true);
  assert.equal(hasRuntimeOverrides({SPECULATIVE_CONFIG:''}),true);
});


test('default-profile selection alone triggers startup import',()=>{
  assert.equal(hasRuntimeOverrides({SM75_DEFAULT_PROFILE:'second'}),true);
});
const cacheProfile = () => ({id:'cache',port:8000,power:{mode:'pstate'},env:{},
  args:['/models/model','--custom-unknown','keep','--auto-sleep-idle-timeout','0']});
test('LMCache environment uses the same explicit CPU/disk configuration and stays pure',()=>{
  const original=cacheProfile(), before=structuredClone(original);
  const result=applyRuntimeEnvironment(original,{LMCACHE_ENABLED:'1',LMCACHE_CPU_GIB:'12',
    LMCACHE_CHUNK_SIZE:'1568',LMCACHE_DISK_PATH:'/lmcache',LMCACHE_MIN_FREE_DISK_GIB:'3',
    LMCACHE_PORT:'5570',LMCACHE_HTTP_PORT:'5571'});
  assert.deepEqual(original,before);
  assert.deepEqual(result.lmcache,{enabled:true,cpuGiB:12,chunkSize:1568,diskPath:'/lmcache',minFreeDiskGiB:3,port:5570,httpPort:5571});
  assert.equal(argValue(result.args,'--custom-unknown'),'keep');
  assert.equal(hasRuntimeOverrides({LMCACHE_ENABLED:'1'}),true);
  assert.equal(applyRuntimeEnvironment(original,{}).lmcache,undefined);
  assert.equal(applyRuntimeEnvironment(original,{LMCACHE_ENABLED:'0'}).lmcache.enabled,false);
});
test('LMCache selection refuses native/custom/spec/sleep conflicts without deleting source settings',()=>{
  const enabled={LMCACHE_ENABLED:'1',LMCACHE_CHUNK_SIZE:'1568'};
  for(const env of [{CPU_KV_GIB:'8'},{KV_TRANSFER_CONFIG:'{}'},{POWER_MODE:'sleep'},
                    {AUTO_SLEEP_IDLE_TIMEOUT:'30'},{SPECULATIVE_CONFIG:'{}'}])
    assert.throws(()=>applyRuntimeEnvironment(cacheProfile(),{...enabled,...env}),/LMCache/);
  for(const flag of ['--kv-transfer-config','--speculative-config']) {
    const p=cacheProfile();p.args.push(flag,'{}');const before=structuredClone(p);
    assert.throws(()=>applyRuntimeEnvironment(p,enabled),/LMCache/);assert.deepEqual(p,before);
  }
  assert.throws(()=>applyRuntimeEnvironment(cacheProfile(),{LMCACHE_ENABLED:'1'}),/chunk size/);
  const p=cacheProfile();p.args.push('--kv-transfer-config','{}');
  assert.equal(applyRuntimeEnvironment(p,{...enabled,CPU_KV_GIB:'0'}).lmcache.enabled,true);
});


test('LMCache CPU convenience zero cannot erase an existing custom connector',()=>{
  const source=cacheProfile();source.args.push('--kv-transfer-config','{"kv_connector":"Custom"}');
  const env={LMCACHE_ENABLED:'1',LMCACHE_CHUNK_SIZE:'1568',CPU_KV_GIB:'0'};
  assert.throws(()=>applyRuntimeEnvironment(source,env),/自定义连接器/);
  assert.equal(applyRuntimeEnvironment(source,{...env,KV_TRANSFER_CONFIG:''}).lmcache.enabled,true);
});


test('decimal CPU GiB matches the panel byte rounding and zero disables',()=>{
  for(const value of ['10.25','.25','10.','0.1','0.0000000005','8388607.999999999']) {
    const args=['/models/model'];
    const panel=applyContextSettings(args,{...readContextSettings(args),cpuKvMode:'native',cpuKvGiB:value});
    const imported=applyRuntimeEnvironment(profile(),{CPU_KV_GIB:value});
    const bytes=args=>JSON.parse(argValue(args,'--kv-transfer-config')).kv_connector_extra_config.cpu_bytes_to_use;
    assert.equal(bytes(imported.args),bytes(panel),value);
    assert.ok(Number.isSafeInteger(bytes(imported.args)));
  }
  for(const value of ['0','0.0','.0','00.000']) {
    const source=profile();source.args.push('--kv-transfer-config','{"kv_connector":"OffloadingConnector"}');
    assert.equal(argValue(applyRuntimeEnvironment(source,{CPU_KV_GIB:value}).args,'--kv-transfer-config'),null);
  }
});
test('invalid, sub-byte and unsafe CPU GiB never change the source profile',()=>{
  for(const value of ['', '-1','+1','1e3','0x10','NaN','Infinity','10.25GiB','1,25','1.2.3',
                      ' 10.25 ','0.0000000001','8388608','999999999999999999999999999999999999999999']) {
    const source=profile(), before=structuredClone(source);
    assert.throws(()=>applyRuntimeEnvironment(source,{CPU_KV_GIB:value}),/CPU_KV_GIB/,value);
    assert.deepEqual(source,before);
  }
});
test('decimal CPU GiB remains explicit and conflicts with LMCache unless exactly zero',()=>{
  const cache={LMCACHE_ENABLED:'1',LMCACHE_CHUNK_SIZE:'1568'};
  assert.throws(()=>applyRuntimeEnvironment(cacheProfile(),{...cache,CPU_KV_GIB:'10.25'}),/LMCache/);
  assert.equal(applyRuntimeEnvironment(cacheProfile(),{...cache,CPU_KV_GIB:'0.0'}).lmcache.enabled,true);
});
