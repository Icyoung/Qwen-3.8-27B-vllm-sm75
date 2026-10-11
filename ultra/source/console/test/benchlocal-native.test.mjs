import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {once} from 'node:events';
import {fileURLToPath} from 'node:url';
import {BenchLocal} from '../benchlocal-native.mjs';
import {normalizeOptions,applyGeneration,validateImportedReport,reportCsv} from '../benchlocal-contract.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../benchlocal-native');
const runtime=process.env.SM75_BENCHLOCAL_RUNTIME||root;
const catalog=JSON.parse(fs.readFileSync(path.join(runtime,'catalog.json'),'utf8'));
const caps={toggle:true,budget:true};
test('native catalog preserves all scenario IDs and the 95/165 presets',()=>{
  assert.equal(catalog.packs.length,9);
  assert.equal(catalog.packs.filter(x=>x.defaultSelected).reduce((n,x)=>n+x.scenarioCount,0),95);
  assert.equal(catalog.packs.reduce((n,x)=>n+x.scenarioCount,0),165);
  for(const p of catalog.packs)assert.equal(new Set(p.scenarios.map(x=>x.id)).size,p.scenarioCount);
});
test('sampling preserves explicit zero, meaningful defaults and thinking constraints',()=>{
  assert.deepEqual(normalizeOptions({},caps).generation,{request_timeout_seconds:300});
  const opts=normalizeOptions({temperature:0,thinking:'on',effort:'medium',max_tokens:8192},caps);
  assert.equal(opts.generation.temperature,0);assert.equal(opts.generation.thinking_token_budget,4096);
  assert.throws(()=>normalizeOptions({thinking:'on',effort:'high',max_tokens:1024},caps));
  assert.throws(()=>normalizeOptions({thinking:'on'},{toggle:false}));
  assert.throws(()=>normalizeOptions({top_k:0},caps));
  const payload=applyGeneration({model:'x',messages:[],tools:[{name:'fixture'}],temperature:.9},catalog.packs[0],opts);
  assert.equal(payload.temperature,0);assert.equal(payload.thinking_token_budget,4096);
  assert.deepEqual(payload.tools,[{name:'fixture'}]);assert.equal(payload.chat_template_kwargs.enable_thinking,true);
});
test('six inline packs: native worker, model proxy, batch scoring and report round-trip use real pinned pack code',async t=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-bench-native-'));
  const secret='private-test-api-key-not-for-export';
  const requests=[];
  const model=http.createServer(async(req,res)=>{
    let text='';for await(const b of req)text+=b;const body=JSON.parse(text);requests.push({body,authorization:req.headers.authorization});
    res.writeHead(200,{'Content-Type':'application/json'});
    res.end(JSON.stringify({choices:[{message:{role:'assistant',content:'fixture answer',tool_calls:[]},finish_reason:'stop'}],usage:{prompt_tokens:11,completion_tokens:4,total_tokens:15}}));
  });model.listen(0,'127.0.0.1');await once(model,'listening');
  const manager=new BenchLocal(directory,{runtime,verifiers:{}});
  const gateway=http.createServer((req,res)=>manager.proxy.handle(req,res));gateway.listen(0,'127.0.0.1');await once(gateway,'listening');
  manager.proxyLocal='http://127.0.0.1:'+gateway.address().port+'/benchlocal-inference/v1';
  manager.proxyDocker=manager.proxyLocal;
  t.after(async()=>{manager.close();gateway.closeAllConnections();model.closeAllConnections();await Promise.all([new Promise(r=>gateway.close(r)),new Promise(r=>model.close(r))]);assert.equal(path.dirname(path.resolve(directory)),path.resolve(os.tmpdir()));assert.ok(path.basename(directory).startsWith('sm75-bench-native-'));fs.rmSync(directory,{recursive:true,force:true});});
  const row=await manager.start({packIds:catalog.packs.filter(p=>p.verifier!=='managed').map(p=>p.id),options:{temperature:.4,max_tokens:4096,thinking:'off'}},
    {model:'fixture-model',modelName:'FixtureModel',profileName:'Fixture',url:'http://127.0.0.1:'+model.address().port,engineKey:secret,capabilities:caps,isCurrent:()=>true});
  const child=manager.active.child;await once(child,'close');
  const report=manager.detail(row.id);
  assert.equal(report.state,'completed',report.error);
  assert.equal(report.done,90);assert.equal(report.suites.length,6);
  assert.equal(report.options.generation.request_timeout_seconds,300);
  assert.ok(report.suites.every(s=>s.results.every(r=>r.requestTimeoutSeconds===300)));
  assert.ok(report.suites.every(x=>Number.isFinite(x.score)));
  assert.ok(requests.length>=90);assert.ok(requests.every(x=>x.authorization==='Bearer '+secret));
  assert.ok(requests.every(x=>x.body.temperature===.4&&x.body.max_tokens===4096&&x.body.chat_template_kwargs.enable_thinking===false));
  const exported=manager.export(row.id,{detailed:true});
  assert.equal(JSON.stringify(exported).includes(secret),false);
  assert.ok(exported.suites[0].results[0].requests.length);
  const imported=manager.import(exported);assert.equal(imported.source,'imported');assert.notEqual(imported.id,row.id);
  assert.deepEqual(imported.suites.map(x=>x.score),report.suites.map(x=>x.score));
  const malformed=structuredClone(exported);malformed.suites[0].score=101;assert.throws(()=>validateImportedReport(malformed,catalog));
  const duplicate=structuredClone(exported);duplicate.suites[0].results.push(duplicate.suites[0].results[0]);assert.throws(()=>validateImportedReport(duplicate,catalog));
  assert.equal(reportCsv(report).includes('toolcall-15'),true);
  const denied=await fetch(manager.proxyLocal+'/models');assert.equal(denied.status,401);
});

test('pinned DataExtract accepts a response after 30 seconds and respects an explicit shorter timeout',async t=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-bench-native-timeout-'));
  let requestCount=0;
  const model=http.createServer(async(req,res)=>{
    for await(const chunk of req){}requestCount++;
    const timer=setTimeout(()=>{res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({choices:[{message:{role:'assistant',content:'{}'},finish_reason:'stop'}],usage:{prompt_tokens:1,completion_tokens:1,total_tokens:2}}));},31500);
    res.on('close',()=>clearTimeout(timer));
  });model.listen(0,'127.0.0.1');await once(model,'listening');
  const manager=new BenchLocal(directory,{runtime,verifiers:{}});
  const gateway=http.createServer((req,res)=>manager.proxy.handle(req,res));gateway.listen(0,'127.0.0.1');await once(gateway,'listening');
  manager.proxyLocal='http://127.0.0.1:'+gateway.address().port+'/benchlocal-inference/v1';manager.proxyDocker=manager.proxyLocal;
  t.after(async()=>{manager.close();gateway.closeAllConnections();model.closeAllConnections();await Promise.all([new Promise(r=>gateway.close(r)),new Promise(r=>model.close(r))]);assert.equal(path.dirname(path.resolve(directory)),path.resolve(os.tmpdir()));assert.ok(path.basename(directory).startsWith('sm75-bench-native-timeout-'));fs.rmSync(directory,{recursive:true,force:true});});
  const context={model:'fixture-model',url:'http://127.0.0.1:'+model.address().port,engineKey:'fixture-secret',capabilities:{},isCurrent:()=>true};
  for(const [options,timeout,expectedStatus] of [[{},300,200],[{request_timeout_seconds:5},5,499]]){
    const row=await manager.start({packIds:['dataextract-15'],scenarioSelection:{'dataextract-15':['DE-02']},options},context);
    await once(manager.active.child,'close');
    const result=row.suites[0].results[0];assert.equal(row.options.generation.request_timeout_seconds,timeout);
    assert.equal(result.requestTimeoutSeconds,timeout);assert.ok(result.requests.every(r=>r.status===expectedStatus));
    if(expectedStatus===200){assert.notEqual(result.status,'error');assert.equal(result.requests.length,1);assert.ok(result.durationMs>30000);}
    else {assert.equal(result.status,'error');assert.equal(result.requests.length,3);assert.match(result.upstreamSummary,/after 5s/);}
  }
  assert.equal(requestCount,4);
});
