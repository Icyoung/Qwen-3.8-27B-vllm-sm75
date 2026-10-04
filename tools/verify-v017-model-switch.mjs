#!/usr/bin/env node
// Functional TP4 acceptance in a disposable container. No throughput baseline.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
const {setArg}=await import('/opt/sm75-workbench/console/public/context-config.js');
assert.equal(process.env.SM75_DISPOSABLE_AUDIT,'1');
assert(fs.existsSync('/.dockerenv') && fs.existsSync('/dev/nvidia0'));
const root=process.env.SM75_CONSOLE_ROOT || '/data';
const base='http://127.0.0.1:1615', checks=[];
const key=fs.readFileSync(root+'/key','utf8').trim();
const headers={Authorization:'Bearer '+key,'Content-Type':'application/json'};
const api=(route,body)=>new Promise((resolve,reject)=>{
 const request=http.request(base+'/console-api/'+route,{method:body===undefined?'GET':'POST',headers},res=>{
  let text='';res.on('data',chunk=>text+=chunk);res.on('end',()=>{try{const value=JSON.parse(text);if(res.statusCode>=400)throw Error(value.error||'HTTP '+res.statusCode);resolve(value);}catch(error){reject(error);}});
 });
 request.setTimeout(21*60*1000,()=>request.destroy(Error('Engine startup exceeded test deadline')));
 request.on('error',reject);request.end(body===undefined?undefined:JSON.stringify(body));
});
const check=(name,detail={})=>{checks.push({name,...detail});console.log(JSON.stringify(checks.at(-1)));};
const models={awq:process.env.SM75_TEST_AWQ_MODEL,fp8:process.env.SM75_TEST_FP8_MODEL};
for(const value of Object.values(models))assert(value&&fs.existsSync(value+'/config.json'));
const devices=(await api('system')).gpus;assert.equal(devices.length,4);
const engineKey=JSON.parse(fs.readFileSync(root+'/api-access.json','utf8')).key;
const engineHeaders={Authorization:'Bearer '+engineKey,'Content-Type':'application/json'};
const generate=async id=>{const r=await fetch('http://127.0.0.1:8000/v1/chat/completions',{method:'POST',headers:engineHeaders,body:JSON.stringify({model:id,messages:[{role:'user',content:'Reply with the word OK.'}],max_tokens:16,temperature:0})});const data=await r.json();assert(r.ok,JSON.stringify(data));assert(data.choices?.[0]?.message);assert(data.usage?.completion_tokens>0);return data.usage.completion_tokens;};
const dshPid=()=>fs.readdirSync('/proc').filter(n=>/^\d+$/.test(n)).find(n=>{try{return fs.readFileSync('/proc/'+n+'/cmdline','utf8').includes('/@deepseek-ai/dsh/lib/bin.js');}catch{return false;}});
let first,last;
try {
 for(const [format,dir] of Object.entries(models)) {
  const model=await api('models/register',{path:dir,name:'Beta '+format.toUpperCase()});
  const draft=await api('models/draft',{model:model.id,template:format==='awq'?'custom':first.id});
  const p=draft.profile;p.id='beta-'+format;p.name='Beta '+format.toUpperCase();
  for(const [flag,value] of [['--tensor-parallel-size','4'],['--max-model-len','2048'],['--max-num-batched-tokens','2048'],['--max-num-seqs','1'],['--served-model-name',p.id]])p.args=setArg(p.args,flag,value);
  if(!p.args.includes('--enforce-eager'))p.args.push('--enforce-eager');
  await api('profiles/validate',p);await api('profiles',p);
  if(!first)first=p;else last=p;
 }
 const started=await api('profiles/'+first.id+'/start',{});assert.equal(started.phase,'ready');
 check('AWQ TP4 is ready and generates',{tokens:await generate(first.id)});
 await api('harness/start',{profile:first.id});const before=dshPid();assert(before);
 const broken={...last,id:'beta-missing',args:['/models/does-not-exist','--dtype','float16']};await api('profiles',broken);
 await assert.rejects(api('profiles/'+broken.id+'/start',{}),/config.json/);
 assert.equal((await api('runtime')).profile,first.id);check('invalid target leaves the current engine running');
 const changed=await api('profiles/'+last.id+'/start',{});assert.equal(changed.phase,'ready');assert(!changed.warning,changed.warning);
 assert.equal((await api('runtime')).profile,last.id);assert.equal(dshPid(),before);
 const configured=await(await fetch('http://127.0.0.1:3085/sm75/configured-models')).json();assert(configured.models.some(m=>m.id==='sm75-local/'+last.id));
 check('one AWQ to FP8 switch preserves the Harness process',{tokens:await generate(last.id)});
} finally {
 const running=await api('runtime').catch(()=>({}));if(running.profile)await api('profiles/'+running.profile+'/stop',{});
 fs.writeFileSync(root+'/gpu-switch-results.json',JSON.stringify({checks,scope:'TP4 functional only; eager mode, 2048 context, no throughput benchmark'},null,2));
}
