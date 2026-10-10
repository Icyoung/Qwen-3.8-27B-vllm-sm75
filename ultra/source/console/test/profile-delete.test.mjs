import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {Store} from '../store.mjs';
import {deleteRuntimeProfile,deletedBootProfile,persistProfiles} from '../profile-delete.mjs';

function fixture(t){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-profile-delete-'));
  const store=new Store(root),file=path.join(root,'profiles.json');
  const profiles=['one','two'].map(id=>({id,name:id,args:[path.join(root,'model')],format:'fp8',backend:'native',port:18000,cacheRoot:path.join(root,'cache')}));
  persistProfiles(file,profiles);
  const settings={...store.settings(),defaultProfile:'two',autoStart:true,defaultChatProfile:'two',appearance:{fontSize:17}};
  store.put('settings','main',settings);
  for(const name of ['model/weights','cache/keep','chats/two.json','workspace/keep']){const f=path.join(root,name);fs.mkdirSync(path.dirname(f),{recursive:true});fs.writeFileSync(f,'preserve');}
  store.saveTemplate('保存的模板',profiles[1]);
  const locks=new Set(),args={profiles,profilesPath:file,store,locks,status:async()=>({running:false})};
  t.after(()=>{store.close();assert.equal(path.dirname(root),os.tmpdir());assert.ok(path.basename(root).startsWith('sm75-profile-delete-'));fs.rmSync(root,{recursive:true,force:true});});
  return {root,store,file,profiles,settings,args,locks};
}

test('deleting a default configuration archives only its settings and preserves model, cache, conversations and templates',async t=>{
  const f=fixture(t),original=structuredClone(f.profiles[1]);
  const result=await deleteRuntimeProfile({...f.args,id:'two'});
  assert.deepEqual(result.profiles.map(p=>p.id),['one']);
  assert.deepEqual(JSON.parse(fs.readFileSync(f.file)).map(p=>p.id),['one']);
  assert.deepEqual(f.profiles.map(p=>p.id),['one']);
  assert.equal(f.store.settings().defaultProfile,'');assert.equal(f.store.settings().defaultChatProfile,'');assert.equal(f.store.settings().autoStart,false);
  assert.deepEqual(f.store.settings().appearance,{fontSize:17});
  assert.equal(f.store.list('template').length,1);
  for(const name of ['model/weights','cache/keep','chats/two.json','workspace/keep'])assert.equal(fs.readFileSync(path.join(f.root,name),'utf8'),'preserve');
  const archived=f.store.get('profile-deletion','two');
  const backup=JSON.parse(fs.readFileSync(path.join(f.root,'deleted-profiles',archived.backup)));
  assert.equal(backup.schema,'sm75-profile-v1');assert.deepEqual(backup.profile,original);
  assert.equal(f.locks.size,0);
});

test('unrelated defaults are preserved and deleting the last configuration persists an empty list',async t=>{
  const f=fixture(t);
  const first=await deleteRuntimeProfile({...f.args,id:'one'});
  assert.equal(f.store.settings().defaultProfile,'two');assert.equal(f.store.settings().autoStart,true);
  await deleteRuntimeProfile({...f.args,profiles:first.profiles,id:'two'});
  assert.deepEqual(JSON.parse(fs.readFileSync(f.file)),[]);assert.equal(f.store.settings().autoStart,false);
});

test('running, starting and unknown states refuse deletion without writing a backup',async t=>{
  const f=fixture(t),before=fs.readFileSync(f.file,'utf8');
  for(const state of [{running:true},{running:false,phase:'starting'},null]){
    await assert.rejects(deleteRuntimeProfile({...f.args,id:'two',status:async()=>state}),/正在运行|无法确认/);
    assert.equal(fs.readFileSync(f.file,'utf8'),before);assert.equal(f.locks.size,0);
  }
  assert.equal(fs.existsSync(path.join(f.root,'deleted-profiles')),false);
});

test('deletion holds the model lock across status checks and releases it after a failure',async t=>{
  const f=fixture(t);let resolve;
  const first=deleteRuntimeProfile({...f.args,id:'one',status:()=>new Promise(r=>{resolve=r;})});
  assert.equal(f.locks.has('model-switch'),true);
  await assert.rejects(deleteRuntimeProfile({...f.args,id:'two'}),/操作进行中/);
  resolve({running:true});await assert.rejects(first,/正在运行/);assert.equal(f.locks.size,0);
});

test('storage failure rolls back defaults and deletion markers while retaining the original list',async t=>{
  const f=fixture(t),before=fs.readFileSync(f.file,'utf8');
  await assert.rejects(deleteRuntimeProfile({...f.args,id:'two',write:()=>{throw Error('disk fixture');}}),/disk fixture/);
  assert.equal(fs.readFileSync(f.file,'utf8'),before);assert.equal(f.store.settings().defaultProfile,'two');
  assert.equal(f.store.settings().autoStart,true);assert.equal(f.store.get('profile-deletion','two'),null);assert.equal(f.locks.size,0);
});

test('deleted Docker startup profiles stay deleted and explicit profile recreation remains possible',async t=>{
  const f=fixture(t);await deleteRuntimeProfile({...f.args,id:'two'});
  f.store.put('startup','environment',{profile:'two'});
  assert.equal(deletedBootProfile(f.profiles,f.store,{id:'two'},{}),'two');
  assert.equal(deletedBootProfile(f.profiles,f.store,null,{SM75_DEFAULT_PROFILE:'two'}),'two');
  assert.equal(deletedBootProfile(f.profiles,f.store,null,{MODEL:path.join(f.root,'model')}),'two');
  assert.equal(deletedBootProfile(f.profiles,f.store,null,{MODEL:'/different/model'}),null);
  assert.equal(deletedBootProfile([...f.profiles,{id:'two'}],f.store,null,{SM75_DEFAULT_PROFILE:'two'}),null);
  await assert.rejects(deleteRuntimeProfile({...f.args,id:'../outside'}),/ID无效/);
});

const app=fs.readFileSync(new URL('../public/app.js',import.meta.url),'utf8');
const start=app.indexOf('  async function deleteProfile(p) {'),end=app.indexOf('  function closeEditor()',start);
function ui(confirmed){
  const elements={profileEditor:{hidden:false},pid:{value:'two'},notice:{textContent:''}};
  const events=[],context={profileState:{two:{running:false}},profilesSignature:'old',profiles:[{id:'one'},{id:'two'}],active:'two',runningProfile:'one',editorDirty:true,
    globalSettings:{defaultProfile:'two',defaultChatProfile:'two'},$:id=>elements[id],profileLabel:p=>p.name,
    confirm:message=>{events.push(['confirm',message]);return confirmed;},api:async(...args)=>{events.push(['api',...args]);},
    closeEditor:()=>events.push(['close']),load:async()=>events.push(['load']),populateModelSources:()=>events.push(['sources']),refreshProfileCards:async()=>events.push(['cards'])};
  vm.createContext(context);vm.runInContext(app.slice(start,end)+'\nglobalThis.remove=deleteProfile;',context);
  return {context,elements,events};
}
test('delete confirmation can be cancelled without changing editor or calling the API',async()=>{
  const f=ui(false);await f.context.remove({id:'two',name:'测试配置'});
  assert.equal(f.events.length,1);assert.equal(f.context.active,'two');assert.equal(f.elements.profileEditor.hidden,false);
});
test('confirmed deletion refreshes defaults and choices without issuing a model stop',async()=>{
  const f=ui(true);await f.context.remove({id:'two',name:'测试配置'});
  assert.match(f.events[0][1],/默认启动/);assert.match(f.events[0][1],/默认会话/);assert.match(f.events[0][1],/尚未保存/);
  assert.deepEqual(f.events.filter(x=>x[0]==='api').map(x=>x[1]),['/console-api/profiles/two/delete']);
  assert.equal(f.context.active,'one');assert.ok(f.events.some(x=>x[0]==='close'));assert.match(f.elements.notice.textContent,/已删除/);
});
