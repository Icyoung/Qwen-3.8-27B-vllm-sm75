import fs from 'node:fs';
import http from 'node:http';
import crypto from 'node:crypto';
import {connectChannel} from '../benchlocal-verifiers/ndjson.mjs';
export const EXECUTION_ROLES=['bugfind-15','cli-40','hermesagent-20'];
const routes=new Set(['/run-scenario','/verify-answer','/verify-canonical','/verify-oneshot','/verify-multiround']);
const reply=(res,status,value)=>{if(!res.destroyed&&!res.writableEnded){res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(value));}};
const closeServer=server=>{if(!server)return Promise.resolve();server.closeAllConnections?.();return new Promise(resolve=>server.close(()=>resolve()));};
const listen=server=>new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',()=>{server.removeListener('error',reject);resolve();});});
export class OnDemandVerifiers {
 constructor({proxyHandler,onFailure=()=>{},socketPath=process.env.SM75_BENCH_BROKER_SOCKET||'/run/sm75-benchlocal/control.sock'}={}){
  this.proxyHandler=proxyHandler;this.onFailure=onFailure;this.socketPath=socketPath;this.key=crypto.randomBytes(32).toString('hex');this.endpoints={};this.gateways=new Map();this.pending=new Map();this.models=new Map();this.channel=null;this.connecting=null;this.activeRole=null;this.jobId=null;this.runId=null;this.proxyServer=null;this.preparing=null;this.releasing=null;this.expectedClose=false;this.probeResult=null;
 }
 async connect(){
  if(this.channel&&!this.channel.closed&&!this.channel.socket.destroyed)return this.channel;
  if(this.connecting)return this.connecting;
  this.connecting=(async()=>{
   const channel=await connectChannel(this.socketPath);this.channel=channel;
   channel.on('message',m=>this.message(channel,m));
   channel.on('close',reason=>{
    if(this.channel!==channel)return;this.channel=null;clearInterval(this.heartbeat);this.probeResult=null;
    for(const {reject,timer}of this.pending.values()){clearTimeout(timer);reject(Error('按需执行启动器连接中断：'+reason));}this.pending.clear();
    for(const item of this.models.values())item.abort();this.models.clear();
    if(this.runId&&!this.expectedClose)void Promise.resolve().then(()=>this.onFailure('按需执行服务连接中断')).catch(()=>{});
   });
   const hello=await this.request('hello',{},6000,channel);this.image=hello.image;
   this.heartbeat=setInterval(()=>{void this.request('ping',{},5000).catch(()=>{});},10000);this.heartbeat.unref();return channel;
  })();try{return await this.connecting;}finally{this.connecting=null;}
 }
 async request(type,fields={},timeout=45000,knownChannel){
  const channel=knownChannel||await this.connect(),id='r:'+crypto.randomUUID();
  return new Promise((resolve,reject)=>{
   const timer=setTimeout(()=>{this.pending.delete(id);reject(Error('按需执行请求超时'));channel.close('操作超时');},timeout);
   this.pending.set(id,{resolve,reject,timer});
   void channel.send({type,id,...fields}).catch(e=>{clearTimeout(timer);this.pending.delete(id);reject(e);});
  });
 }
 message(channel,message){
  if(this.channel!==channel)return;
  if(message.type==='reply'){const p=this.pending.get(message.id);if(!p)return;this.pending.delete(message.id);clearTimeout(p.timer);message.error?p.reject(Error(message.error)):p.resolve(message.value);return;}
  if(message.type==='fatal'){channel.close(message.error||'执行容器已退出');return;}
  if(message.type==='model-open'){void this.forwardModel(channel,message);return;}
  if(message.type==='model-abort'){this.models.get(message.id)?.abort();this.models.delete(message.id);}
 }
 async forwardModel(channel,m){
  if(!this.proxyServer||m.jobId!==this.jobId||this.models.has(m.id)||this.models.size>=8){await channel.send({type:'model-error',id:m.id,jobId:m.jobId,error:'Invalid model lease'}).catch(()=>{});return;}
  if(!((m.method==='POST'&&m.path==='/benchlocal-inference/v1/chat/completions')||(m.method==='GET'&&m.path==='/benchlocal-inference/v1/models'))||typeof m.data!=='string'||m.data.length>12*1024*1024){channel.close('无效模型请求');return;}
  const abort=new AbortController();this.models.set(m.id,{abort:()=>abort.abort(),jobId:m.jobId});
  const send=value=>channel.send({...value,id:m.id,jobId:m.jobId});
  try{
   const response=await fetch('http://127.0.0.1:'+this.proxyServer.address().port+m.path,{method:m.method,headers:{'Content-Type':'application/json',Authorization:String(m.headers?.authorization||'')},body:m.method==='POST'?Buffer.from(m.data,'base64'):undefined,signal:abort.signal});
   await send({type:'model-head',status:response.status,headers:{'content-type':response.headers.get('content-type')||'application/json'}});
   if(response.body)for await(const part of response.body){const data=Buffer.from(part);for(let offset=0;offset<data.length;offset+=65536)await send({type:'model-data',data:data.subarray(offset,offset+65536).toString('base64')});}
   await send({type:'model-end'});
  }catch{await send({type:'model-error',error:'模型请求失败或已取消'}).catch(()=>{});}finally{this.models.delete(m.id);}
 }
 async readiness(force=false){
  if(!force&&this.probeResult&&Date.now()-this.probeResult.at<15000)return this.probeResult.states;
  let reason='';
  if(process.platform!=='linux')reason='按需执行需要 Linux 宿主启动器';
  else if(!fs.existsSync(this.socketPath))reason='按需执行启动器尚未安装或启动';
  else try{await this.connect();await this.request('hello',{},6000);}catch(error){reason='按需执行启动器未就绪：'+error.message;}
  const states=Object.fromEntries(EXECUTION_ROLES.map(id=>[id,{ready:!reason,reason,mode:'ondemand'}]));this.probeResult={at:Date.now(),states};return states;
 }
 async activate(selection,runId){
  await this.releasing;const selected=selection.filter(id=>EXECUTION_ROLES.includes(id));if(!selected.length)return;
  if(!/^[a-f0-9-]{36}$/.test(runId))throw Error('无效评测任务');
  await this.connect();this.runId=runId;
  this.proxyServer=http.createServer((req,res)=>void this.proxyHandler(req,res));await listen(this.proxyServer);
  for(const role of selected){const server=http.createServer((req,res)=>void this.handle(role,req,res));await listen(server);this.gateways.set(role,server);this.endpoints[role]='http://127.0.0.1:'+server.address().port;}
 }
 async prepare(role){
  if(this.activeRole===role&&this.jobId)return;
  if(this.preparing)await this.preparing;
  if(this.activeRole===role&&this.jobId)return;
  if(this.activeRole)throw Error('已有其他测试类占用执行环境');
  const operation=(async()=>{const result=await this.request('start',{runId:this.runId,role},60000);this.activeRole=role;this.jobId=result.jobId;})();this.preparing=operation;
  try{await operation;}finally{if(this.preparing===operation)this.preparing=null;}
 }
 async stopRole(role){
  if(this.activeRole&&role!==this.activeRole)return;
  if(this.runId&&this.channel&&!this.channel.closed){try{await this.request('stop',{runId:this.runId},20000);}catch(error){this.channel?.close('清理执行任务失败');throw error;}}
  for(const item of this.models.values())item.abort();this.models.clear();this.activeRole=null;this.jobId=null;
 }
 async handle(role,req,res){
  if(req.method==='GET'&&req.url==='/health'){reply(res,200,{ok:!!this.channel&&!this.channel.closed,managed:true,mode:'ondemand'});return;}
  const got=Buffer.from(String(req.headers.authorization||'')),expected=Buffer.from('Bearer '+this.key);
  if(got.length!==expected.length||!crypto.timingSafeEqual(got,expected)){reply(res,401,{error:'Unauthorized'});return;}
  if(req.method!=='POST'||!(routes.has(req.url)||['/prepare','/release','/cancel'].includes(req.url))){reply(res,404,{error:'Not found'});return;}
  try{
   if(req.url==='/prepare'){await this.prepare(role);reply(res,200,{ready:true});return;}
   if(req.url==='/release'||req.url==='/cancel'){await this.stopRole(role);reply(res,200,{ok:true});return;}
   if(this.activeRole!==role||!this.jobId)throw Error('执行环境尚未准备');
   let size=0;const chunks=[];for await(const chunk of req){size+=chunk.length;if(size>2*1024*1024)throw Error('执行请求过大');chunks.push(chunk);}
   const body=JSON.parse(Buffer.concat(chunks));if(body.model){const url=new URL(body.model.inferenceBaseUrl);if(url.origin!=='http://127.0.0.1:1615'||url.pathname!=='/benchlocal-inference/v1')throw Error('无效模型代理地址');}
   res.once('close',()=>{if(!res.writableEnded)void this.stopRole(role).catch(()=>{});});
   const result=await this.request('call',{runId:this.runId,route:req.url,body},31*60*1000);reply(res,result.status,result.body);
  }catch(error){reply(res,503,{error:error.message||'按需执行失败'});}
 }
 async release(){
  if(this.releasing)return this.releasing;
  const operation=(async()=>{
   this.expectedClose=true;
   try{if(this.runId)await this.stopRole(this.activeRole).catch(()=>{});if(this.preparing)await this.preparing.catch(()=>{});}
   finally{
    this.runId=null;this.activeRole=null;this.jobId=null;await Promise.all([...this.gateways.values()].map(closeServer));this.gateways.clear();this.endpoints={};await closeServer(this.proxyServer);this.proxyServer=null;this.expectedClose=false;
   }
  })();this.releasing=operation;try{await operation;}finally{if(this.releasing===operation)this.releasing=null;}
 }
 async close(){await this.release();this.expectedClose=true;clearInterval(this.heartbeat);this.channel?.close();this.channel=null;}
}
