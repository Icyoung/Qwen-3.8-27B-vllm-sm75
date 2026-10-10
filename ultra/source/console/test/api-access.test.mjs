import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';import os from 'node:os';import path from 'node:path';
import {spawn} from 'node:child_process';import {once} from 'node:events';import {fileURLToPath} from 'node:url';
import {Standalone} from '../standalone.mjs';

test('saved API keys cannot replace the active engine credential until the engine stops',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-key-lifecycle-'));
 try {
  const file=path.join(root,'api-access.json');fs.writeFileSync(file,JSON.stringify({key:'synthetic-first'}));
  const manager=new Standalone(root,'synthetic-web-login');assert.equal(manager.engineKey(),'synthetic-first');
  manager.engine={id:'model',apiKey:manager.engineKey()};
  fs.writeFileSync(file,JSON.stringify({key:'synthetic-next'}));
  assert.equal(manager.engineKey(),'synthetic-first');assert(!JSON.stringify(manager.status({id:'model'})).includes('synthetic-first'));
  manager.engine=null;assert.equal(manager.engineKey(),'synthetic-next');
 } finally {assert.equal(path.dirname(root),os.tmpdir());fs.rmSync(root,{recursive:true,force:true});}
});

test('API key reveal requires authenticated same-origin POST and never appears in ordinary metadata',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-key-api-')),base='http://127.0.0.1:19367';
 const child=spawn(process.execPath,[fileURLToPath(new URL('../server.mjs',import.meta.url))],{env:{...process.env,SM75_SINGLE_CONTAINER:'1',SM75_CONSOLE_ROOT:root,SM75_CONSOLE_HOST:'127.0.0.1',SM75_CONSOLE_PORT:'19367'},stdio:'ignore'});
 try {
  let ready=false;for(let i=0;i<100;i++){try{ready=(await fetch(base)).ok;if(ready)break;}catch{}await new Promise(r=>setTimeout(r,50));}assert(ready);
  const loginKey=fs.readFileSync(path.join(root,'key'),'utf8').trim(),engineKey=JSON.parse(fs.readFileSync(path.join(root,'api-access.json'))).key;
  assert.notEqual(loginKey,engineKey);
  const endpoint=base+'/console-api/api-access';
  assert.equal((await fetch(endpoint+'/reveal',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).status,401);
  const login=await fetch(base+'/console-api/login',{method:'POST',headers:{Origin:base,'Content-Type':'application/json'},body:JSON.stringify({token:loginKey})});assert.equal(login.status,200);
  const cookie=login.headers.get('set-cookie').split(';')[0],headers={Cookie:cookie,Origin:base,'Content-Type':'application/json'};
  const meta=await fetch(endpoint,{headers});assert.match(meta.headers.get('cache-control'),/no-store/);assert(!(await meta.text()).includes(engineKey));
  assert.equal((await fetch(endpoint+'/reveal',{headers})).status,405);
  assert.equal((await fetch(endpoint+'/reveal',{method:'POST',headers:{...headers,Origin:'http://evil.invalid'},body:'{}'})).status,403);
  let response=await fetch(endpoint+'/reveal',{method:'POST',headers,body:'{}'});assert.equal(response.status,200);assert.match(response.headers.get('cache-control'),/no-store/);assert.equal((await response.json()).key,engineKey);
  response=await fetch(endpoint,{method:'POST',headers,body:JSON.stringify({key:'synthetic-replacement-key'})});assert.equal(response.status,200);assert.equal((await response.json()).restartRequired,false);
  response=await fetch(endpoint+'/reveal',{method:'POST',headers,body:'{}'});assert.equal((await response.json()).key,'synthetic-replacement-key');
  assert.equal((await fetch(base+'/console-api/benchlocal')).status,401);
  let bench=await fetch(base+'/console-api/benchlocal',{headers});assert.equal(bench.status,200);bench=await bench.json();assert.equal(bench.packs.length,9);assert(!JSON.stringify(bench).includes(engineKey));
  const blocked=await fetch(base+'/console-api/benchlocal',{method:'POST',headers:{...headers,Origin:'http://evil.invalid'},body:JSON.stringify({enabled:false})});assert.equal(blocked.status,403);
  assert.equal((await fetch(base+'/benchlocal-native-ui.js')).status,200);
  const asset=await fetch(base+'/api-access-ui.js');assert.equal(asset.status,200);assert.match(asset.headers.get('content-type'),/javascript/);
 } finally {if(child.exitCode===null){const ended=once(child,'exit');child.kill();await ended;}assert.equal(path.dirname(root),os.tmpdir());fs.rmSync(root,{recursive:true,force:true});}
});
