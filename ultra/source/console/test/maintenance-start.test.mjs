import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {consumeStoppedModelGuard} from '../maintenance-start.mjs';
const filename='.preserve-stopped-model-once.json';
test('a maintenance restart suppresses autostart once without changing saved preferences',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-maintenance-'));
 try{fs.writeFileSync(path.join(root,'preferences.json'),'{"autoStart":true}');fs.writeFileSync(path.join(root,filename),JSON.stringify({expires:101000}));assert.equal(consumeStoppedModelGuard(root,100000),true);assert.equal(consumeStoppedModelGuard(root,100000),false);assert.equal(fs.readFileSync(path.join(root,'preferences.json'),'utf8'),'{"autoStart":true}');}
 finally{assert.equal(path.dirname(root),os.tmpdir());fs.rmSync(root,{recursive:true,force:true});}
});
test('stale or unbounded guards cannot suppress a later ordinary startup',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-maintenance-'));
 try{for(const expires of [1,10000000]){fs.writeFileSync(path.join(root,filename),JSON.stringify({expires}));assert.equal(consumeStoppedModelGuard(root,100000),false);assert.equal(fs.existsSync(path.join(root,filename)),false);}}
 finally{assert.equal(path.dirname(root),os.tmpdir());fs.rmSync(root,{recursive:true,force:true});}
});

test('a corrupt guard preserves the stopped state rather than crashing the console',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-maintenance-'));
 try{fs.writeFileSync(path.join(root,filename),'invalid');assert.equal(consumeStoppedModelGuard(root),true);}
 finally{assert.equal(path.dirname(root),os.tmpdir());fs.rmSync(root,{recursive:true,force:true});}
});
