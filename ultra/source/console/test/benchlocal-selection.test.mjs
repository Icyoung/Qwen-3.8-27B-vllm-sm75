import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {EventEmitter} from 'node:events';
import {fileURLToPath} from 'node:url';
import {BenchLocal} from '../benchlocal-native.mjs';
import {restoreBenchLocalSelection} from '../public/benchlocal-native-ui.js';
const runtime=fileURLToPath(new URL('../../benchlocal-native/',import.meta.url));
const {packs}=JSON.parse(fs.readFileSync(path.join(runtime,'catalog.json'),'utf8'));
const six=['toolcall-15','dataextract-15','instructfollow-15','reasonmath-15','bugfind-15','hermesagent-20'];
const oldSix=['toolcall-15','dataextract-15','instructfollow-15','promptauthority-15','reasonmath-15','structoutput-15'];
test('new defaults and legacy default preferences select six representative categories',()=>{
  assert.deepEqual(restoreBenchLocalSelection(packs),six);
  assert.deepEqual(restoreBenchLocalSelection(packs,{selected:oldSix}),six);
  assert.deepEqual(restoreBenchLocalSelection(packs,{selectionPreset:'default',selected:oldSix}),six);
});
test('explicit custom choices, all nine and empty selection survive preference restoration',()=>{
  assert.deepEqual(restoreBenchLocalSelection(packs,{selectionPreset:'custom',selected:oldSix}),oldSix);
  assert.deepEqual(restoreBenchLocalSelection(packs,{selected:['cli-40','toolcall-15','obsolete']}),['toolcall-15','cli-40']);
  assert.equal(restoreBenchLocalSelection(packs,{selected:packs.map(p=>p.id)}).length,9);
  assert.deepEqual(restoreBenchLocalSelection(packs,{selected:[]}),[]);
});
test('scheduler uses new defaults and puts CLI last even when selected first',async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-bench-order-'));let child,input;
  const manager=new BenchLocal(dir,{runtime,verifiers:{},forkProcess:(_file,args)=>{input=JSON.parse(fs.readFileSync(args[0],'utf8'));return child=new EventEmitter();}});
  try{
    const context={model:'fixture',engineKey:'fixture-only',capabilities:{},isCurrent:()=>true};
    const row=await manager.start({},context);
    assert.deepEqual(row.selection,six);assert.deepEqual(input.selection,six);assert.equal(row.total,95);
    child.emit('close');
    const all=await manager.start({packIds:packs.map(p=>p.id).reverse()},context);
    assert.equal(all.total,165);assert.equal(input.selection.at(-1),'cli-40');
    assert.deepEqual(all.suites.map(s=>s.id),input.selection);
    child.emit('close');
  }finally{
    if(manager.active)child.emit('close');manager.close();
    assert.equal(path.dirname(path.resolve(dir)),path.resolve(os.tmpdir()));assert.ok(path.basename(dir).startsWith('sm75-bench-order-'));
    fs.rmSync(dir,{recursive:true,force:true});
  }
});
