import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {nativeCpuKvBytes, recommendedShmGiB, preflightCpuKV} from '../cpu-kv-resources.mjs';
import {makeCommand} from '../config.mjs';
import {Standalone} from '../standalone.mjs';
const G = 2**30;
const profile = (bytes=16*G) => ({id:'cpu-kv-test',backend:'native',port:8000,format:'fp8',cacheRoot:'/tmp/cache',power:{mode:'sleep'},
 args:['/model','--tensor-parallel-size','4','--kv-transfer-config',JSON.stringify({kv_connector:'OffloadingConnector',kv_role:'kv_both',kv_connector_extra_config:{spec_name:'CPUOffloadingSpec',cpu_bytes_to_use:bytes}})]});
const statfs = (total,free) => () => ({blocks:BigInt(total/4096),bsize:4096n,bavail:BigInt(free/4096)});
test('16 GiB is total shared across TP4 and TP8, not multiplied by rank count',()=>{
 const p=profile(); assert.equal(nativeCpuKvBytes(p),16*G);p.args[2]='8';assert.equal(nativeCpuKvBytes(p),16*G);
});
test('equal shm capacity fails with exact free bytes and a 32g recommendation',()=>{
 assert.throws(()=>preflightCpuKV(profile(),{statfs:statfs(16*G,16*G-1048576)}), /32g/);
 assert.throws(()=>preflightCpuKV(profile(),{statfs:statfs(16*G,16*G-1048576)}), /17178820608 字节/);
});
test('32 GiB shm passes with CPU budget unchanged',()=>{
 const p=profile();const before=structuredClone(p); const r=preflightCpuKV(p,{statfs:statfs(32*G,32*G-1048576)});
 assert.equal(r.requiredBytes,16*G);assert.deepEqual(p,before);
});
test('capacity preflight permits reclaimable running cache but final free-space check rejects it',()=>{
 const options={statfs:statfs(32*G,8*G)};assert.ok(preflightCpuKV(profile(),{...options,capacityOnly:true}));
 assert.throws(()=>preflightCpuKV(profile(),options),/共享内存不足/);
});
test('off, custom and LMCache do not probe native shared memory',()=>{
 const list=[{args:['/model']},{...profile(),lmcache:{enabled:true}}, {args:['/model','--kv-transfer-config={"kv_connector":"Custom"}']}, {args:['/model','--kv-offloading-size','16','--kv-offloading-backend','lmcache']}];
 for(const p of list)assert.equal(preflightCpuKV(p,{statfs:()=>assert.fail('unexpected probe')}),null);
});
test('inline shorthand/native default budget and recommendation',()=>{
 assert.equal(nativeCpuKvBytes({args:['/model','--kv-offloading-size=16']}),16*G);
 assert.equal(nativeCpuKvBytes({args:['/model','--kv-transfer-config={}']}),8*G);
 assert.equal(recommendedShmGiB(profile()),32);assert.equal(recommendedShmGiB(profile(8*G)),16);
});
test('Docker model command provisions shm independently of CPU cache budget',()=>{
 const p={...profile(),backend:'docker'};const r=makeCommand(p,'synthetic');
 assert.equal(r.args[r.args.indexOf('--shm-size')+1],'32g');assert.equal(nativeCpuKvBytes(p),16*G);
 assert.equal(makeCommand({...p,args:['/model']},'synthetic').args.includes('16g'),true);
});
test('Standalone refuses before launching helpers, preparing caches or CUDA',async()=>{
 const x=new Standalone('/tmp','synthetic',{preflightCpuKV:()=>preflightCpuKV(profile(),{statfs:statfs(16*G,16*G-4096)})});
 x.prepareCache=()=>assert.fail('cache prepared');x.launch=()=>assert.fail('process launched');
 await assert.rejects(x.start(profile()),/32g/);assert.equal(x.engine,null);assert.equal(x.starting,false);
});
test('server checks before model switch and validation endpoint',()=>{
 const t=fs.readFileSync(new URL('../server.mjs',import.meta.url),'utf8');
 assert.equal((t.match(/preflightCpuKV\(p, \{capacityOnly:!!standalone\?\.engine\}\)/g)||[]).length,2);
 assert.ok(t.indexOf('preflightCpuKV(p, {capacityOnly')<t.indexOf('return await switchModel(p'));
});
