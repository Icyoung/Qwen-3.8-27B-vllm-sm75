import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import http from 'node:http';
import {once,EventEmitter} from 'node:events';
import {PassThrough} from 'node:stream';
import {BenchLocal} from '../benchlocal-native.mjs';
import {redactBenchLocal,normalizeOptions,packSuiteScore,packResultScore,portableReport} from '../benchlocal-contract.mjs';
import {packSuiteScore as uiSuiteScore,packResultScore as uiResultScore} from '../public/benchlocal-native-ui.js';
const runtime=process.env.SM75_BENCHLOCAL_RUNTIME;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function until(check,timeout=15000){const end=Date.now()+timeout;while(Date.now()<end){if(check())return;await sleep(20);}throw Error('Timed out');}
function temporary(t){const dir=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-benchlocal-test-'));t.after(()=>{assert.equal(path.dirname(path.resolve(dir)),path.resolve(os.tmpdir()));assert.ok(path.basename(dir).startsWith('sm75-benchlocal-test-'));fs.rmSync(dir,{recursive:true,force:true});});return dir;}
test('settings do not invent output/temperature overrides and reject unsupported reasoning',()=>{
  assert.deepEqual(normalizeOptions({}).generation,{request_timeout_seconds:300});
  assert.equal(normalizeOptions({request_timeout_seconds:90}).generation.request_timeout_seconds,90);
  assert.equal(normalizeOptions({}).presetName,'题包默认');
  for(const value of[{temperature:NaN},{max_tokens:0},{top_p:2},{thinking:'yes'},{thinking:'on',effort:'high'}])assert.throws(()=>normalizeOptions(value));
});
test('nested secrets and authorization fields are removed from portable data',()=>{
  const secret='quotes"and\\slashes',value=redactBenchLocal({data:[{text:secret}],key:secret,api_key:secret,authorization:'Bearer x'},secret);
  assert.equal(value.data[0].text,'[redacted]');assert.equal(value.key,'[redacted]');assert.equal(value.api_key,undefined);assert.equal(value.authorization,undefined);
});

test('upstream partial, failed and stored summary scores remain visible and exportable',()=>{
  const suite={score:null,summary:{totalScore:51},results:[
    {scenarioId:'DE-01',status:'partial',score:79},
    {scenarioId:'DE-02',status:'error',score:null,scoringStatus:'fail',rawLog:'error=Request timed out after 30s.\nverdict=fail\nscore=0\nsummary=Request timed out after 30s.'},
    {scenarioId:'DE-03',status:'error',score:null},
    {scenarioId:'DE-04',status:'error',score:null,scoringScore:0},
    {scenarioId:'DE-05',status:'cancelled',score:null,scoringScore:0}
  ]};
  assert.equal(packSuiteScore(suite),51);assert.equal(uiSuiteScore(suite),51);
  assert.deepEqual(suite.results.map(packResultScore),[79,0,null,0,null]);
  assert.deepEqual(suite.results.map(uiResultScore),[79,0,null,0,null]);
  const exported=portableReport({suites:[suite]});
  assert.equal(exported.suites[0].score,51);
  assert.deepEqual(exported.suites[0].results.map(r=>r.score),[79,0,null,0,null]);
  assert.equal(exported.suites[0].results[1].rawLog,undefined);
  assert.equal(suite.score,null,'export must not rewrite original evidence');
  assert.equal(packSuiteScore({score:null,summary:null}),null);
});

test('request failures keep pack scores and successful retries keep the final verdict',async t=>{
  const root=temporary(t),children=[];
  const forkProcess=()=>{const c=new EventEmitter();c.stdout=new PassThrough();c.stderr=new PassThrough();c.kill=()=>{};c.send=()=>{};children.push(c);return c;};
  const manager=new BenchLocal(root,{runtime,forkProcess,verifiers:{}});
  const row=await manager.start({packIds:['dataextract-15']},{url:'http://127.0.0.1:8000',model:'fixture',engineKey:'fixture-key',capabilities:{}});
  const emit=m=>children[0].emit('message',{source:'sm75-benchlocal-native',packId:'dataextract-15',...m});
  manager.active.audit.set('dataextract-15/DE-01',[{status:499,response:{error:'aborted'}},{status:200}]);
  emit({kind:'scenario-result',result:{scenarioId:'DE-01',status:'partial',score:79,summary:'11/14 atomic fields correct (79%).'}});
  assert.equal(row.suites[0].results[0].status,'partial');assert.equal(row.suites[0].results[0].score,79);
  manager.active.audit.set('dataextract-15/DE-02',[{status:499,response:{error:'aborted'}}]);
  emit({kind:'scenario-result',result:{scenarioId:'DE-02',status:'fail',score:0,summary:'Request timed out after 30s.'}});
  assert.equal(row.suites[0].results[1].status,'error');assert.equal(row.suites[0].results[1].score,0);
  emit({kind:'suite-result',complete:true,summary:{totalScore:51}});
  assert.equal(row.suites[0].score,51);assert.equal(row.suites[0].state,'failed');
  children[0].emit('close',0,null);
});
test('progress survives restart, scoped credentials never persist, and an active job does not silently resume',async t=>{
  const root=temporary(t),children=[];
  const forkProcess=()=>{const c=new EventEmitter();c.stdout=new PassThrough();c.stderr=new PassThrough();c.kill=()=>{};c.send=()=>{};children.push(c);return c;};
  const manager=new BenchLocal(root,{runtime,forkProcess,verifiers:{}}),scenario=manager.catalog().packs[0].scenarios[0];
  const row=await manager.start({packIds:['toolcall-15']},{url:'http://127.0.0.1:8000',model:'fixture',modelName:'fixture',engineKey:'secret-fixture',capabilities:{}});
  await assert.rejects(manager.start({packIds:['toolcall-15']},{}));
  manager.active.audit.set('toolcall-15/'+scenario.id,[{status:200,settings:{temperature:0}}]);
  children[0].emit('message',{source:'sm75-benchlocal-native',kind:'scenario-result',packId:'toolcall-15',result:{scenarioId:scenario.id,status:'pass',score:100,summary:'secret-fixture'}});
  assert.equal(manager.detail(row.id).suites[0].results[0].summary,'[redacted]');assert.equal(manager.list()[0].suites[0].results,undefined);
  assert.throws(()=>manager.detail('../outside'));
  const restored=new BenchLocal(root,{runtime,forkProcess,verifiers:{}});assert.equal(restored.detail(row.id).state,'interrupted');assert.equal(restored.detail(row.id).done,1);
  children[0].emit('close',1,null);
  const files=fs.readdirSync(path.join(root,'benchlocal/jobs',row.id));
  for(const file of files)assert.ok(!fs.readFileSync(path.join(root,'benchlocal/jobs',row.id,file)).includes(Buffer.from('secret-fixture')));
});
test('cancelling aborts the in-flight model request and retains completed progress',async t=>{
  const root=temporary(t);let started=false,closed=false;
  const model=http.createServer(async(req,res)=>{for await(const part of req){}started=true;res.on('close',()=>{closed=true;});});model.listen(0,'127.0.0.1');await once(model,'listening');
  const manager=new BenchLocal(root,{runtime,verifiers:{}}),gateway=http.createServer((req,res)=>manager.proxy.handle(req,res));gateway.listen(0,'127.0.0.1');await once(gateway,'listening');
  manager.proxyLocal='http://127.0.0.1:'+gateway.address().port+'/benchlocal-inference/v1';manager.proxyDocker=manager.proxyLocal;
  t.after(()=>{manager.close();model.closeAllConnections();gateway.closeAllConnections();model.close();gateway.close();});
  const row=await manager.start({packIds:['reasonmath-15']},{url:'http://127.0.0.1:'+model.address().port,model:'fixture',modelName:'fixture',engineKey:'fixture-key',capabilities:{}});
  await until(()=>started);await manager.cancel(row.id);await until(()=>!manager.busy);
  assert.equal(manager.detail(row.id).state,'cancelled');assert.ok(closed);assert.equal(manager.proxy.leases.size,0);
});
