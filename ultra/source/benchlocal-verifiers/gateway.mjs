import http from 'node:http';
import fs from 'node:fs';
import {spawn} from 'node:child_process';
import {timingSafeEqual} from 'node:crypto';

const key = process.env.SM75_BENCH_VERIFIER_KEY;
if (!key || key.length < 32) throw Error('Verifier credential missing');
const verifierRoot = process.env.SM75_BENCH_VERIFIER_ROOT;
if (!verifierRoot) throw Error('Verifier role missing');
const innerPort = Number(process.env.SM75_VERIFIER_INNER_PORT || 4010);
const port = Number(process.env.SM75_VERIFIER_PORT || 4200);
const routes = new Set(['/run-scenario','/verify-answer','/verify-canonical','/verify-oneshot','/verify-multiround']);
let child, active = null, restarting = null, stopping = false;
function authorized(req) {
  const actual = Buffer.from(String(req.headers.authorization || ''));
  const expected = Buffer.from('Bearer '+key);
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}
function reply(res, code, value) {
  if (res.destroyed || res.writableEnded) return;
  res.writeHead(code, {'Content-Type':'application/json','Cache-Control':'no-store'});
  res.end(JSON.stringify(value));
}
function startChild() {
  const env = {...process.env, PORT:String(innerPort), HOME:'/home/verifier'};
  delete env.SM75_BENCH_VERIFIER_KEY;
  delete env.SM75_VERIFIER_KEY;
  child = spawn(process.execPath, [verifierRoot+'/server.mjs'], {
    cwd:verifierRoot, env, uid:10001, gid:10001, detached:true,
    stdio:['ignore','ignore','pipe'],
  });
  child.stderr.on('data', () => {}); // Per-task errors return through the JSON result.
  child.once('error', error => console.error('verifier start failed:',error.code));
}
function stopExecutionProcesses() {
  // Only this isolated container's execution identity; never touches host PIDs.
  for (const name of fs.readdirSync('/proc')) {
    if (!/^\d+$/.test(name) || Number(name) < 2 || Number(name) === process.pid) continue;
    try {
      const status = fs.readFileSync('/proc/'+name+'/status','utf8');
      if (/^Uid:\s+10001\s/m.test(status)) process.kill(Number(name),'SIGKILL');
    } catch {}
  }
}
async function restart() {
  if (restarting) return restarting;
  restarting = (async () => {
    active?.abort.abort();
    stopExecutionProcesses();
    await new Promise(resolve => setTimeout(resolve,150));
    active = null;
    if (!stopping) startChild();
  })().finally(() => {restarting=null;});
  return restarting;
}
async function readBody(req) {
  let size=0;const chunks=[];
  for await (const chunk of req) {
    size+=chunk.length;
    if (size>2*1024*1024) throw Error('request too large');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}
startChild();
const server=http.createServer(async (req,res) => {
  try {
    if (req.method==='GET' && req.url==='/health') {
      const response=await fetch('http://127.0.0.1:'+innerPort+'/health',{signal:AbortSignal.timeout(1500)});
      return reply(res,response.ok?200:503,{ok:response.ok,busy:!!active,managed:true});
    }
    if (!authorized(req)) return reply(res,401,{error:'Unauthorized'});
    if (req.method==='POST' && req.url==='/cancel') {await restart();return reply(res,200,{ok:true});}
    if (req.method!=='POST' || !routes.has(req.url)) return reply(res,404,{error:'Not found'});
    if (active || restarting) return reply(res,409,{error:'Verifier is busy'});
    const raw=await readBody(req);
    const payload=JSON.parse(raw);
    if (payload.model) {
      const url=new URL(payload.model.inferenceBaseUrl);
      const allowed=new URL(process.env.SM75_BENCH_PROXY_ORIGIN || 'http://sm75-bench-controller:1615');
      if (url.origin!==allowed.origin || url.pathname!=='/benchlocal-inference/v1') return reply(res,400,{error:'Invalid managed model endpoint'});
    }
    const task={abort:new AbortController()};active=task;
    const timer=setTimeout(()=>{if(active===task)void restart();},30*60*1000);
    res.once('close',()=>{if(!res.writableEnded && active===task)void restart();});
    try {
      const response=await fetch('http://127.0.0.1:'+innerPort+req.url,{
        method:'POST',headers:{'Content-Type':'application/json'},body:raw,signal:task.abort.signal,
      });
      const text=await response.text();
      if(text.length>16*1024*1024)throw Error('Verifier output too large');
      if(!res.destroyed){res.writeHead(response.status,{'Content-Type':'application/json'});res.end(text);}
    } finally {clearTimeout(timer);if(active===task)active=null;}
  } catch(error) {reply(res,503,{error:error.name==='AbortError'?'Test cancelled':'Verifier unavailable'});}
});
server.listen(port,'0.0.0.0');
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>{
  stopping=true;active?.abort.abort();stopExecutionProcesses();server.close(()=>process.exit(0));
  setTimeout(()=>process.exit(0),2000).unref();
});
