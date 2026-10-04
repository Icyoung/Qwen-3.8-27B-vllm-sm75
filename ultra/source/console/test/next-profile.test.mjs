import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {inferenceProfile,inferenceProfiles} from '../inference-profiles.mjs';
import {createModelDraft} from '../model-launch.mjs';
import {argValue} from '../public/context-config.js';
import {applyRuntimeEnvironment} from '../runtime-settings.mjs';
const id='flash-next-tp8-256k-mtp3';
test('shared Next template contains all tested inference options',()=>{
 const p=inferenceProfile(id,'/models/Next');
 assert.equal(argValue(p.args,'--max-model-len'),'262144');
 assert.equal(argValue(p.args,'--max-num-seqs'),'4');
 assert.equal(argValue(p.args,'--kv-cache-dtype'),'fp8_e4m3');
 assert.deepEqual(JSON.parse(argValue(p.args,'--engram-config')),{cpu_offload:true});
 assert.equal(JSON.parse(argValue(p.args,'--speculative-config')).num_speculative_tokens,3);
 assert.equal(p.env.VLLM_SM75_QSA_FP8_MAIN_KV,'1');
 assert(!p.args.includes('--cpu-offload-params'));
});
test('Next profile CLI and panel resolve the identical model args, including quoted paths',()=>{
 const launcher=fileURLToPath(new URL('../../../../docker/helpers/runtime-entrypoint.py',import.meta.url));
 for(const profileId of inferenceProfiles().map(p=>p.id))for(const model of ['/models/Next','/models/a "quoted" path']){
  const script='import importlib.util,json,sys; s=importlib.util.spec_from_file_location("entry",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.load_profile(sys.argv[2],{"MODEL":sys.argv[3]})))';
  const got=JSON.parse(execFileSync(process.env.PYTHON||'python3',['-c',script,launcher,profileId,model],{encoding:'utf8'}));
  const wanted=inferenceProfile(profileId,model);
  assert.deepEqual(got.args,wanted.args);assert.deepEqual(got.env,wanted.env);
 }
});
test('Next template refuses 27B and insufficient GPU count; personal copies have unique IDs',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'next-profile-'));
 try{
  const model={path:root,name:'Next',modelType:'qwen4_exp',format:'awq'};
  const opts={settings:{cacheRoot:'/cache'},preset:id,cards:Array(8).fill({})};
  const a=createModelDraft(model,null,opts).profile,b=createModelDraft(model,null,opts).profile;
  assert.notEqual(a.id,b.id);assert.equal(a.args[0],root);
  assert.throws(()=>createModelDraft({...model,modelType:'qwen3_5'},null,opts),/Flash Next/);
  assert.throws(()=>createModelDraft(model,null,{...opts,cards:[{}]}),/GPU/);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('Docker Next settings survive startup import and explicit false values',()=>{
 let p=inferenceProfile(id,'/models/Next');
 p=applyRuntimeEnvironment(p,{ENGRAM_CONFIG:'{"cpu_offload":true}',ENABLE_EP_WEIGHT_FILTER:'false',MAX_NUM_SEQS:'2'});
 assert.equal(argValue(p.args,'--max-num-seqs'),'2');
 assert(!p.args.includes('--enable-ep-weight-filter'));assert(p.args.includes('--no-enable-ep-weight-filter'));
});
test('switching from a Next configuration to 27B removes Next-only runtime wiring',()=>{
 const source=inferenceProfile(id,'/models/Next');
 const result=createModelDraft({path:'/models/27B',name:'27B',format:'fp8'},source,{settings:{cacheRoot:'/cache'}}).profile;
 for(const flag of ['--engram-config','--enable-expert-parallel','--enable-ep-weight-filter','--language-model-only','--speculative-config'])assert(!result.args.includes(flag),flag);
 assert(!Object.keys(result.env).some(k=>k.startsWith('VLLM_SM75_QSA_')));
});
test('all six original 27B modes share the Docker/panel profile catalog',()=>{
 const profiles=inferenceProfiles().filter(p=>p.modelFamily==='qwen3_5');assert.equal(profiles.length,6);
 for(const spec of profiles){
  const p=inferenceProfile(spec.id,'/models/target');
  assert.equal(argValue(p.args,'--tensor-parallel-size'),'4');
  assert.equal(argValue(p.args,'--kv-cache-dtype'),'fp8_e4m3');
  assert(!p.args.includes('--engram-config'));assert(!Object.keys(p.env).some(k=>k.startsWith('VLLM_SM75_QSA_')));
  const draft=argValue(p.args,'--speculative-config');
  if(spec.id.includes('dflash2'))assert.equal(JSON.parse(draft).model,'/models/Qwen3.8-27B-DFlash2');
  else if(spec.id.includes('mtp5'))assert.equal(JSON.parse(draft).num_speculative_tokens,5);
  else assert.equal(draft,null);
  assert.equal(createModelDraft({path:'/models/target',modelType:'qwen3_5',name:'27B',format:spec.format},null,{settings:{cacheRoot:'/cache'},cards:Array(4).fill({}),preset:spec.id}).profile.format,spec.format);
 }
});


test('no-MTP Next uses the same context and QSA optimizations without a draft',()=>{
 const p=inferenceProfile('flash-next-tp8-256k-nomtp','/models/Next');
 assert.equal(argValue(p.args,'--speculative-config'),null);
 assert.equal(argValue(p.args,'--max-model-len'),'262144');
 assert.equal(argValue(p.args,'--max-num-seqs'),'4');
 assert.equal(p.env.VLLM_SM75_QSA_MIXED_PREFILL,'1');
 assert.deepEqual(JSON.parse(argValue(p.args,'--compilation-config')).cudagraph_capture_sizes,[1,2,4]);
});

test('mixed budget survives a Next copy and is cleared when selecting 27B',()=>{
 const p=inferenceProfile(id,'/models/Next');p.env.VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET='1024';
 const next=createModelDraft({path:'/models/Next',name:'Next',format:'awq',modelType:'qwen4_exp'},p,{settings:{cacheRoot:'/cache'}}).profile;
 assert.equal(next.env.VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET,'1024');
 const other=createModelDraft({path:'/models/27B',name:'27B',format:'fp8',modelType:'qwen3_5'},p,{settings:{cacheRoot:'/cache'}}).profile;
 assert(!Object.hasOwn(other.env,'VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET'));
});
