import fs from 'node:fs';
import http from 'node:http';
import {once} from 'node:events';
import {BenchLocal} from '/opt/sm75-workbench/console/benchlocal-native.mjs';

const calls=[];
const model=http.createServer(async(req,res)=>{
  let text='';for await(const part of req)text+=part;const body=JSON.parse(text);
  calls.push({temperature:body.temperature,max_tokens:body.max_tokens,thinking:body.chat_template_kwargs?.enable_thinking,model:body.model,top_p:body.top_p,top_k:body.top_k,min_p:body.min_p,repetition_penalty:body.repetition_penalty,seed:body.seed,presence_penalty:body.presence_penalty,frequency_penalty:body.frequency_penalty});
  res.writeHead(200,{'Content-Type':'application/json'});
  res.end(JSON.stringify({id:'fixture',object:'chat.completion',model:'fixture-model',choices:[{index:0,message:{role:'assistant',content:'This is an isolated transport fixture. I cannot complete the task.',tool_calls:[]},finish_reason:'stop'}],usage:{prompt_tokens:20,completion_tokens:12,total_tokens:32}}));
});model.listen(18000,'127.0.0.1');await once(model,'listening');
const manager=new BenchLocal('/tmp/native-flow',{runtime:'/opt/sm75-workbench/benchlocal-native'});
const server=http.createServer((req,res)=>manager.proxy.handle(req,res));server.listen(1615,'0.0.0.0');await once(server,'listening');
const catalog=manager.catalog();
const selection=['bugfind-15','cli-40','hermesagent-20'];
const scenarios=Object.fromEntries(selection.map(id=>[id,[catalog.packs.find(p=>p.id===id).scenarios[0].id]]));
try {
  const row=await manager.start({packIds:selection,scenarioSelection:scenarios,options:{temperature:.3,top_p:.95,top_k:20,min_p:.05,repetition_penalty:1.1,seed:42,presence_penalty:.1,frequency_penalty:0,max_tokens:2048,thinking:'off',request_timeout_seconds:60}},
    {model:'fixture-model',modelName:'CPU transport fixture',profileName:'CPU fixture',url:'http://127.0.0.1:18000',engineKey:'fixture-private-key',capabilities:{toggle:true,budget:true},isCurrent:()=>true});
  await once(manager.active.child,'close');
  const report=manager.export(row.id,{detailed:true});
  fs.writeFileSync('/results/managed-flow-report.json',JSON.stringify(report,null,2));
  fs.writeFileSync('/results/managed-flow-requests.json',JSON.stringify(calls,null,2));
  console.log(JSON.stringify({state:report.state,done:report.done,suites:report.suites.map(s=>({id:s.id,state:s.state,results:s.results.map(r=>({id:r.scenarioId,status:r.status,summary:r.summary}))})),requests:calls.length}));
  if(report.done!==3||report.suites.some(s=>s.results.some(r=>r.status==='error'||!r.requests?.length))||!calls.length)process.exitCode=1;
  if(calls.some(r=>r.temperature!==.3||r.max_tokens!==2048||r.thinking!==false||r.model!=='fixture-model'||r.top_k!==20||r.top_p!==.95||r.min_p!==.05||r.seed!==42||r.repetition_penalty!==1.1||r.presence_penalty!==.1||r.frequency_penalty!==0))throw Error('Parameters were not applied on all verifier paths');
}finally{manager.close();server.closeAllConnections();model.closeAllConnections();server.close();model.close();}
