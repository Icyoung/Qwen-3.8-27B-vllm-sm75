import fs from 'node:fs';
import http from 'node:http';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {connectChannel} from './ndjson.mjs';
const role=process.env.SM75_BENCHLOCAL_ROLE,token=process.env.SM75_JOB_TOKEN;
if(!['bugfind-15','cli-40','hermesagent-20'].includes(role)||!token||process.getuid()!==0)throw Error('Invalid execution container identity');
if(fs.existsSync('/dev/nvidia0'))throw Error('Execution container must not expose GPUs');
const innerPort=role==='cli-40'?4040:4010,root='/opt/sm75-workbench/benchlocal-verifiers',models=new Map();
const channel=await connectChannel('/run/sm75-job.sock');let child,stopping=false,busy=false,seq=0;
const requestBody=async req=>{let size=0;const chunks=[];for await(const chunk of req){size+=chunk.length;if(size>8*1024*1024)throw Error('Request too large');chunks.push(chunk);}return Buffer.concat(chunks);};
const relay=http.createServer(async(req,res)=>{
 if(!((req.method==='POST'&&req.url==='/benchlocal-inference/v1/chat/completions')||(req.method==='GET'&&req.url==='/benchlocal-inference/v1/models'))){res.writeHead(404).end();return;}
 if(models.size>=8){res.writeHead(429).end();return;}
 const id='m:'+String(++seq);let timer;
 try{
  const body=await requestBody(req);models.set(id,{res});
  timer=setTimeout(()=>{res.destroy(Error('Model proxy timeout'));},31*60*1000);
  res.on('close',()=>{clearTimeout(timer);if(models.delete(id))void channel.send({type:'model-abort',id}).catch(()=>{});});
  await channel.send({type:'model-open',id,method:req.method,path:req.url,headers:{authorization:String(req.headers.authorization||''),'content-type':'application/json'},data:body.toString('base64')});
 }catch{models.delete(id);clearTimeout(timer);if(!res.headersSent)res.writeHead(502);res.end();}
});
await new Promise((resolve,reject)=>{relay.once('error',reject);relay.listen(1615,'127.0.0.1',resolve);});
const env={PATH:'/opt/hermes-venv/bin:/opt/sm75-bench/rust/bin:/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin',HOME:'/home/verifier',USER:'verifier',LOGNAME:'verifier',LANG:'C.UTF-8',TMPDIR:'/tmp',PORT:String(innerPort),SM75_BENCH_VERIFIER_ROOT:root+'/'+role,PYTHONDONTWRITEBYTECODE:'1',NVIDIA_VISIBLE_DEVICES:'void',CUDA_VISIBLE_DEVICES:'',GOMAXPROCS:'2',HERMES_DISABLE_TELEMETRY:'1',NO_PROXY:'*',AGENT_BROWSER_EXECUTABLE_PATH:'/usr/bin/chromium',NODE_OPTIONS:'--disable-sigusr1 --max-old-space-size=768'};
child=spawn('/usr/local/bin/node',[root+'/'+role+'/server.mjs'],{cwd:root+'/'+role,uid:10001,gid:10001,env,detached:true,stdio:['ignore','ignore','pipe']});
let childError='';child.stderr.on('data',part=>{childError=(childError+String(part)).slice(-2000);});
child.on('error',()=>void shutdown());child.on('close',()=>{if(!stopping){console.error('评分进程退出');void shutdown();}});
async function shutdown(){if(stopping)return;stopping=true;for(const {res}of models.values())res.destroy();models.clear();relay.closeAllConnections();relay.close();channel.close();if(child?.pid)try{process.kill(-child.pid,'SIGKILL');}catch{}setTimeout(()=>process.exit(0),50);}
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>void shutdown());channel.on('close',()=>void shutdown());
channel.on('message',message=>{
 if(message.type==='call'){void run(message);return;}
 const entry=models.get(message.id);if(!entry)return;
 if(message.type==='model-head'){const status=Number(message.status);if(!Number.isInteger(status)||status<100||status>599){entry.res.destroy();return;}if(!entry.res.headersSent)entry.res.writeHead(status,{'Content-Type':message.headers?.['content-type']||'application/json'});}
 else if(message.type==='model-data'){if(typeof message.data!=='string'||message.data.length>256*1024){entry.res.destroy();return;}if(!entry.res.write(Buffer.from(message.data,'base64'))){channel.socket.pause();entry.res.once('drain',()=>channel.socket.resume());entry.res.once('close',()=>channel.socket.resume());}}
 else if(message.type==='model-end'){models.delete(message.id);entry.res.end();}
 else if(message.type==='model-error'){models.delete(message.id);if(!entry.res.headersSent)entry.res.writeHead(502);entry.res.end(JSON.stringify({error:'模型代理连接失败'}));}
});
async function run({id,route,body}){
 if(busy){await channel.send({type:'result',id,error:'Executor busy'});return;}busy=true;
 try{
  const routes=new Set(['/run-scenario','/verify-answer','/verify-canonical','/verify-oneshot','/verify-multiround']);if(!routes.has(route))throw Error('Unsupported execution route');
  if(body?.model){const url=new URL(body.model.inferenceBaseUrl);if(url.origin!=='http://127.0.0.1:1615'||url.pathname!=='/benchlocal-inference/v1')throw Error('Invalid scoped model endpoint');}
  const result=await fetch('http://127.0.0.1:'+innerPort+route,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(30*60*1000)});
  const text=await result.text();if(Buffer.byteLength(text)>16*1024*1024)throw Error('Result too large');
  let value;try{value=JSON.parse(text);}catch{throw Error('Invalid verifier response');}
  await channel.send({type:'result',id,value:{status:result.status,body:value}});
 }catch(error){await channel.send({type:'result',id,error:error.message||'Execution failed'}).catch(()=>{});}finally{busy=false;}
}
let ready=false;for(let attempt=0;attempt<80&&!stopping;attempt++){
 try{const response=await fetch('http://127.0.0.1:'+innerPort+'/health',{signal:AbortSignal.timeout(500)});if(response.ok){ready=true;break;}}catch{}
 await new Promise(resolve=>setTimeout(resolve,100));
}
if(!ready){console.error('执行依赖启动失败：'+childError.slice(-500));await shutdown();}else await channel.send({type:'ready',token,role});
