import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {redactBenchLocal,DEFAULT_REQUEST_TIMEOUT_SECONDS} from './benchlocal-contract.mjs';

const require=createRequire(import.meta.url);
const runtime=path.resolve(process.env.SM75_BENCHLOCAL_RUNTIME||path.join(path.dirname(fileURLToPath(import.meta.url)),'../benchlocal-native'));
const key=process.env.SM75_BENCH_VERIFIER_KEY||'';
const credentials=[key],abort=new AbortController(),waiting=new Map();
let seq=0;
const emit=message=>new Promise((resolve,reject)=>{
  const value={source:'sm75-benchlocal-native',...redactBenchLocal(message,credentials)};
  if(!process.send){process.stdout.write(JSON.stringify(value)+'\n');resolve();}
  else process.send(value,error=>error?reject(error):resolve());
});
async function lease(packId,scenarioId) {
  const rpcId=String(++seq);
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{waiting.delete(rpcId);reject(Error('评测授权分配超时'));},10000);
    waiting.set(rpcId,{resolve:value=>{clearTimeout(timer);resolve(value);},reject:error=>{clearTimeout(timer);reject(error);}});
    process.send({source:'sm75-benchlocal-native',kind:'lease',rpcId,packId,scenarioId});
  });
}
process.on('message',message=>{
  if(message?.source!=='sm75-benchlocal-parent'||message.kind!=='lease')return;
  const pending=waiting.get(message.rpcId);if(!pending)return;
  waiting.delete(message.rpcId);
  if(message.error)pending.reject(Error(message.error));else{credentials.push(message.token);pending.resolve(message.token);}
});
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>abort.abort());

async function builtinVerifier() {
  const {verifyAnswer}=await import(pathToFileURL(path.join(runtime,'packs/structoutput-15/verification/core.mjs')));
  const server=http.createServer(async(req,res)=>{
    const reply=(code,value)=>{res.writeHead(code,{'Content-Type':'application/json'});res.end(JSON.stringify(value));};
    try {
      if(req.method==='GET'&&req.url==='/health')return reply(200,{ok:true});
      if(req.method!=='POST'||req.url!=='/verify-answer')return reply(404,{});
      let size=0;const chunks=[];for await(const chunk of req){size+=chunk.length;if(size>2*1024*1024)return reply(413,{});chunks.push(chunk);}
      const value=JSON.parse(Buffer.concat(chunks));
      if(typeof value.scenarioId!=='string'||typeof value.answer!=='string')return reply(400,{});
      reply(200,await verifyAnswer(value.scenarioId,value.answer));
    } catch(error){reply(400,{error:error.message});}
  });
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
  return {url:'http://127.0.0.1:'+server.address().port,close:()=>new Promise(resolve=>{server.closeAllConnections();server.close(resolve);})};
}
async function main() {
  const job=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
  const catalog=JSON.parse(fs.readFileSync(path.join(runtime,'catalog.json'),'utf8'));
  const originalFetch=globalThis.fetch;
  const gatewayOrigins=new Set(Object.values(job.verifiers).map(value=>new URL(value).origin));
  let verifierError=null;
  globalThis.fetch=async(input,init={})=>{
    const url=new URL(input instanceof Request?input.url:String(input));
    if(gatewayOrigins.has(url.origin)) {
      const headers=new Headers(init.headers||(input instanceof Request?input.headers:undefined));
      headers.set('Authorization','Bearer '+key);init={...init,headers};
      try {const response=await originalFetch(input,init);if(!response.ok)verifierError='受管验证环境返回 '+response.status;return response;}
      catch(error){if(!abort.signal.aborted)verifierError='受管验证环境连接失败';throw error;}
    }
    return originalFetch(input,init);
  };
  let builtin;
  try {
    for(const id of job.selection) {
      if(abort.signal.aborted)break;
      const meta=catalog.packs.find(x=>x.id===id);if(!meta)throw Error('题包不存在');
      const pack=require(path.join(runtime,'packs',id,'dist/benchlocal/index.js')).default;
      const provider=['cli-40','hermesagent-20'].includes(id)?'custom':'llamacpp';
      const model={id:'current',label:job.model.name,provider,model:job.model.id};
      let verifier=job.verifiers[id];
      if(id==='structoutput-15'){builtin ||= await builtinVerifier();verifier=builtin.url;}
      const context={providers:[{id:provider,enabled:true,baseUrl:job.proxyLocal}],
        secrets:[{providerId:provider,value:''}],models:[model],
        inferenceEndpoints:[{modelId:'current',status:'running',baseUrl:job.proxyLocal,dockerBaseUrl:job.proxyDocker,authMode:'bearer',apiKey:'',exposedModel:job.model.id}],
        verifiers:verifier?[{id:'verifier',mode:'managed',status:'running',url:verifier}]:[]};
      const managed=job.verifierMode==='ondemand'&&['bugfind-15','cli-40','hermesagent-20'].includes(id);
      if(managed){const response=await globalThis.fetch(verifier+'/prepare',{method:'POST',signal:abort.signal});if(!response.ok){let value;try{value=await response.json();}catch{}throw Error(value?.error||'执行环境准备失败');}}
      const session=await pack.prepare(context),results=[];
      const libraryGeneration={request_timeout_seconds:DEFAULT_REQUEST_TIMEOUT_SECONDS,...meta.samplingDefaults,...job.options.generation};
      // Hermes uses the OpenAI SDK, which rejects vendor-only top-level kwargs.
      // The scoped model proxy applies every requested sampling field after
      // this client boundary and records the actual outgoing request.
      if(id==='hermesagent-20')for(const key of ['top_k','min_p','repetition_penalty'])delete libraryGeneration[key];
      await emit({kind:'suite-started',packId:id});
      try {
        const selectedScenarios=job.scenarioSelection?.[id];
        for(const scenario of meta.scenarios.filter(x=>!selectedScenarios||selectedScenarios.includes(x.id))) {
          if(abort.signal.aborted)break;
          verifierError=null;
          const token=await lease(id,scenario.id);
          context.secrets[0].value=token;context.inferenceEndpoints[0].apiKey=token;
          await emit({kind:'scenario-started',packId:id,scenarioId:scenario.id});
          const started=Date.now();let result;
          try {
            result=await session.runScenario({model,scenario,runId:id==='promptauthority-15'?job.fixtureId:job.id,
              generation:libraryGeneration,abortSignal:abort.signal},
              event=>emit({kind:'progress',packId:id,scenarioId:scenario.id,event}));
          } catch(error) {
            result={scenarioId:scenario.id,status:abort.signal.aborted?'cancelled':'error',score:null,summary:error.message};
          }
          result={...result,scoringScore:result.score,scenarioId:scenario.id,durationMs:Date.now()-started,requestTimeoutSeconds:libraryGeneration.request_timeout_seconds};
          if(verifierError&&!abort.signal.aborted)result={...result,status:'error',summary:verifierError};
          if(abort.signal.aborted)result={...result,status:'cancelled',score:null};
          results.push(result);
          await emit({kind:'scenario-result',packId:id,result});
        }
        const complete=results.length===meta.scenarioCount&&results.every(x=>x.status!=='cancelled'&&Number.isFinite(x.score));
        const summary=complete?await pack.scoreModelResults(results):null;
        await emit({kind:'suite-result',packId:id,summary,complete});
      } finally {await session.dispose?.();if(managed){const response=await globalThis.fetch(verifier+'/release',{method:'POST',signal:AbortSignal.timeout(20000)});if(!response.ok)throw Error('执行环境未正常回收');}}
    }
    await emit({kind:'finished',cancelled:abort.signal.aborted});
  } finally {
    if(builtin)await builtin.close();
    globalThis.fetch=originalFetch;
  }
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  main().catch(error=>emit({kind:'error',message:error.message})).finally(()=>{if(process.connected)process.disconnect();});
}
