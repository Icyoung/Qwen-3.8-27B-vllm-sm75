import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {modelFormat,createModelDraft,preflightModel} from '../model-launch.mjs';
import {waitForEngine} from '../engine-ready.mjs';
import {argValue} from '../public/context-config.js';
const model={path:'/models/new',name:'New model',format:'int8',context:{maxModelLen:32768}};
const settings={cacheRoot:'/cache'};
test('newly registered model can create every generic draft without an existing profile',()=>{
 for(const preset of ['default','custom','single-long','multi-user','coding']) {
  const {profile}=createModelDraft(model,null,{settings,cards:[{index:0},{index:1},{index:2},{index:3}],preset});
  assert.equal(profile.args[0],model.path);assert.equal(profile.format,'int8');assert.equal(profile.backend,'native');
  assert.equal(argValue(profile.args,'--tensor-parallel-size'),'4');assert.equal(argValue(profile.args,'--dtype'),'float16');
  assert.equal(profile.power.gpus,'0,1,2,3');assert.equal(argValue(profile.args,'--speculative-config'),null);
 }
});
test('cross-format reuse preserves resource and explicit communication settings, clears old model wiring, never mutates source',()=>{
 const source={args:['/models/old','--quantization','awq','--speculative-config','{"method":"dflash","model":"/draft"}', '--scheduler-cls','OldScheduler','--tool-call-parser','old_parser','--hf-overrides','{"architectures":["OldModel"]}', '--tensor-parallel-size','4','--max-model-len','131072','--gpu-memory-utilization','0.92','--seed','17'],env:{NCCL_P2P_LEVEL:'PHB',VLLM_API_KEY:'private'},binds:['/old:/old:ro'],format:'awq'};
 const before=structuredClone(source),{profile,notices}=createModelDraft(model,source,{settings});
 assert.deepEqual(source,before);assert.equal(profile.args[0],model.path);assert.equal(profile.format,'int8');
 for(const flag of ['--quantization','--speculative-config','--scheduler-cls','--tool-call-parser','--hf-overrides'])assert.equal(argValue(profile.args,flag),null);
 assert.equal(argValue(profile.args,'--tensor-parallel-size'),'4');assert.equal(argValue(profile.args,'--seed'),'17');
 assert.equal(argValue(profile.args,'--max-model-len'),'32768');assert.deepEqual(profile.env,{NCCL_P2P_LEVEL:'PHB'});assert.deepEqual(profile.binds,[]);assert(notices.length);
});
test('same-model copy retains tested draft and custom parser configuration',()=>{
 const args=[model.path,'--speculative-config','{"method":"mtp","num_speculative_tokens":5}','--tool-call-parser','qwen3_coder'];
 const result=createModelDraft(model,{args,format:'int8'},{settings});
 assert.equal(argValue(result.profile.args,'--speculative-config'),argValue(args,'--speculative-config'));assert.equal(result.notices.length,0);
});
test('metadata detects newly supported catalog categories instead of treating everything as AWQ',()=>{
 for(const [config,format] of [[{torch_dtype:'bfloat16'},'bf16'],[{},'fp16'],[{quantization_config:{quant_method:'gptq'}},'gptq'],[{quantization_config:{quant_method:'compressed-tensors',config_groups:{group_0:{weights:{num_bits:8,type:'int'}}}}},'int8'],[{quantization_config:{quant_method:'unrecognized'}},'custom']])assert.equal(modelFormat(config),format);
});
test('preflight catches missing shards, excessive TP, unsupported precision and mismatched draft before launch',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'model-preflight-'));
 try {
  fs.writeFileSync(path.join(root,'config.json'),'{"vocab_size":100}');
  const p={args:[root,'--tensor-parallel-size','4','--dtype','float16']};
  assert.throws(()=>preflightModel(p),/权重/);
  fs.writeFileSync(path.join(root,'model.safetensors.index.json'),'{"weight_map":{"a":"part.safetensors"}}');
  assert.throws(()=>preflightModel(p),/分片/);fs.writeFileSync(path.join(root,'part.safetensors'),'fixture');
  assert.equal(preflightModel(p,{cards:[0,1,2,3]}).ok,true);
  assert.throws(()=>preflightModel(p,{cards:[0,1]}),/GPU/);
  assert.throws(()=>preflightModel({args:[root,'--dtype','bfloat16']}),/float16/);
  const draft=path.join(root,'draft');fs.mkdirSync(draft);fs.writeFileSync(path.join(draft,'config.json'),'{"vocab_size":99}');
  assert.throws(()=>preflightModel({args:[...p.args,'--speculative-config',JSON.stringify({method:'dflash',model:draft})]}),/词表/);
 } finally {fs.rmSync(root,{recursive:true,force:true});}
});
test('readiness waits for the requested model rather than an unrelated healthy server',async()=>{
 let time=0,calls=0;const p={port:8000,args:['/models/a','--served-model-name','a']};
 await waitForEngine(p,{key:'test',alive:()=>true,now:()=>time,sleep:async()=>{time+=1000;},timeout:4000,
  fetchImpl:async url=>({ok:true,json:async()=>({data:[{id:url.endsWith('/models')&&++calls<3?'old':'a'}]})})});
 assert.equal(calls,3);assert.equal(time,2000);
 await assert.rejects(waitForEngine(p,{alive:()=>false}),/退出/);
 time=0;await assert.rejects(waitForEngine(p,{alive:()=>true,now:()=>time,sleep:async()=>{time+=1000;},timeout:2000,fetchImpl:async()=>({ok:false})}),/超时/);
});
