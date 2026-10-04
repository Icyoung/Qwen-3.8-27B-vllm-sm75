import test from 'node:test';
import assert from 'node:assert/strict';
import {Readable} from 'node:stream';
import {apply} from '../plugins/sm75-workbench/lib/index.js';
import {managedHarnessConfig} from '../harness-config.mjs';
test('internal switch control rejects public API keys and busy tools; successful update preserves other providers',async()=>{
 const oldControl=process.env.SM75_WORKBENCH_CONTROL_KEY,oldEngine=process.env.SM75_ENGINE_KEY;
 process.env.SM75_WORKBENCH_CONTROL_KEY='synthetic-private-control';process.env.SM75_ENGINE_KEY='synthetic-public-engine';
 try {
  const routes=new Map(),providers={other:{models:[{id:'external'}]}};let busy=true,writes=0;
  const initial=managedHarnessConfig({args:['a'],port:8000});
  apply({root:{loader:{await:async()=>{}}},webServer:{register:r=>routes.set(r.path,r.handler)},
    get:()=>({list:()=>[{header:{id:'original-session'}}]}),waterfall:async(_event,input)=>{assert.equal(input.sessionId,'original-session');return busy?['tool']:[];},
    settings:{mutate:async(_ns,[op])=>{providers['sm75-local']=op.value;writes++;},replace:async()=>{},describe:()=>[{ns:'llm-pi-ai',value:{providers}}]},logger:{error:()=>assert.fail('unexpected setup error')}},initial);
  await new Promise(r=>setImmediate(r));const before=writes;
  const next={...managedHarnessConfig({args:['b'],port:8000}),engineKey:'rotated-engine-key'};
  const call=async(token,body=next)=>{const req=Readable.from([JSON.stringify(body)]);req.method='POST';req.headers={authorization:'Bearer '+token};let status,value;await routes.get('/sm75/model')(req,{writeHead:s=>status=s,end:s=>value=JSON.parse(s)});return {status,value};};
  assert.equal((await call('synthetic-public-engine')).status,401);assert.equal(writes,before);
  assert.equal((await call('synthetic-private-control')).status,409);assert.equal(writes,before);
  busy=false;
  assert.equal((await call('synthetic-private-control',{...next,provider:{...next.provider,baseURL:'http://other.invalid'}})).status,503);assert.equal(writes,before);
  assert.equal((await call('synthetic-private-control')).status,200);assert.equal(providers['sm75-local'].models[0].id,'b');
  assert.deepEqual(providers.other,{models:[{id:'external'}]});assert.equal(process.env.SM75_ENGINE_KEY,'rotated-engine-key');
 } finally {
  if(oldControl===undefined)delete process.env.SM75_WORKBENCH_CONTROL_KEY;else process.env.SM75_WORKBENCH_CONTROL_KEY=oldControl;
  if(oldEngine===undefined)delete process.env.SM75_ENGINE_KEY;else process.env.SM75_ENGINE_KEY=oldEngine;
 }
});
