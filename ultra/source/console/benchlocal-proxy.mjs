import crypto from 'node:crypto';
import {applyGeneration,requestSettings,redactBenchLocal} from './benchlocal-contract.mjs';

const PREFIX='/benchlocal-inference/v1';
export class BenchLocalProxy {
  constructor({fetchImpl=fetch}={}) {this.leases=new Map();this.fetch=fetchImpl;}
  allocate({runId,pack,scenarioId,options,context,onAudit}) {
    const token=crypto.randomBytes(32).toString('base64url');
    this.leases.set(token,{runId,pack,scenarioId,options,context,onAudit,requests:new Set(),count:0,expires:Date.now()+6*3600000});
    return token;
  }
  revoke(token) {const lease=this.leases.get(token);if(lease)for(const abort of lease.requests)abort.abort();this.leases.delete(token);}
  revokeRun(id) {for(const [token,lease]of this.leases)if(lease.runId===id)this.revoke(token);}
  async handle(req,res) {
    const url=new URL(req.url,'http://localhost');
    if(!url.pathname.startsWith('/benchlocal-inference/'))return false;
    const json=(code,value)=>{if(!res.destroyed&&!res.writableEnded){res.writeHead(code,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(value));}};
    const token=String(req.headers.authorization||'').replace(/^Bearer /,'');
    const lease=this.leases.get(token);
    if(!lease||lease.expires<Date.now()) {json(401,{error:'评测授权已失效'});return true;}
    const {context}=lease;
    if(context.isCurrent&&!context.isCurrent()){this.revokeRun(lease.runId);json(409,{error:'运行模型已变化'});return true;}
    if(req.method==='GET'&&url.pathname===PREFIX+'/models') {json(200,{object:'list',data:[{id:context.model,object:'model'}]});return true;}
    if(req.method!=='POST'||url.pathname!==PREFIX+'/chat/completions'||url.search) {json(404,{error:'Unsupported test endpoint'});return true;}
    const started=Date.now(),abort=new AbortController();lease.requests.add(abort);
    let request,status=0,received='',truncated=false;
    const decoder=new TextDecoder();
    const timer=setTimeout(()=>abort.abort(),(lease.options.generation.request_timeout_seconds||1800)*1000);
    res.once('close',()=>{if(!res.writableEnded)abort.abort();});
    try {
      if(++lease.count>1000)throw Error('本题模型请求次数超限');
      const parts=[];let bytes=0;
      for await(const part of req){bytes+=part.length;if(bytes>8*1024*1024)throw Error('评测请求过大');parts.push(part);}
      const payload=JSON.parse(Buffer.concat(parts));
      if(!Array.isArray(payload.messages))throw Error('评测请求缺少消息');
      request=applyGeneration(payload,lease.pack,lease.options);request.model=context.model;
      const response=await this.fetch(context.url.replace(/\/$/,'')+'/v1/chat/completions',{
        method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+context.engineKey},
        body:JSON.stringify(request),signal:abort.signal,
      });
      status=response.status;
      res.writeHead(status,{'Content-Type':response.headers.get('content-type')||'application/json','Cache-Control':'no-store'});
      if(response.body)for await(const chunk of response.body) {
        if(received.length<2*1024*1024)received+=decoder.decode(chunk,{stream:true});else truncated=true;
        if(!res.destroyed)res.write(chunk);
      }
      if(!res.destroyed)res.end();
    } catch(error) {
      status=status|| (abort.signal.aborted?499:502);
      if(!res.headersSent)json(status===499?504:502,{error:{message:error.message}});
      else if(!res.destroyed)res.destroy();
      received ||= JSON.stringify({error:error.message});
    } finally {
      clearTimeout(timer);lease.requests.delete(abort);
      if(!truncated)received+=decoder.decode();
      let response;
      try {response=JSON.parse(received);}catch{
        const events=received.split('\n').filter(line=>line.startsWith('data: ')&&line!=='data: [DONE]').slice(-2048)
          .map(line=>{try{return JSON.parse(line.slice(6));}catch{return null;}}).filter(Boolean);
        response={events,usage:events.findLast?.(item=>item.usage)?.usage};
      }
      const audit={scenarioId:lease.scenarioId,suiteId:lease.pack.id,started,durationMs:Date.now()-started,status,
        settings:requestSettings(request||{}),request:request?{messages:request.messages,tools:request.tools}:undefined,
        response,truncated,usage:response?.usage};
      lease.onAudit(redactBenchLocal(audit,[token,context.engineKey]));
    }
    return true;
  }
}
