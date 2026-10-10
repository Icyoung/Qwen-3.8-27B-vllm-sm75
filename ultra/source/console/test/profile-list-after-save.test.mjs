import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../public/app.js',import.meta.url),'utf8');
const begin=source.indexOf('  async function refreshProfileCards()');
const end=source.indexOf('  $("backProfiles").onclick',begin);
assert(begin>=0&&end>begin);
function setup(origin='models') {
 const el=()=>({hidden:false,children:[],dataset:{},classList:{add(){},remove(){}},append(...x){this.children.push(...x);}});
 const editor=el(),cards=el();cards.replacements=0;cards.replaceChildren=(...x)=>{cards.children=x;cards.replacements++;};
 const state={editorOrigin:origin,profiles:[{id:'next',name:'Next',format:'awq',args:['/models/next']},{id:'27b',name:'27B',format:'fp8',args:['/models/27b']}],profileState:{},profilesSignature:'',
  $:id=>id==='profileEditor'?editor:cards,api:async()=>({running:false}),document:{createElement:el},profileLabel:p=>p.name,run:f=>f,editProfile(){},shareProfile(){},deleteProfile(){},launchProfile(){}};
 const ctx=vm.createContext(state);vm.runInContext(source.slice(begin,end)+'\nglobalThis.refreshCards=refreshProfileCards;',ctx);
 return {ctx,editor,cards};
}
test('saved model-library draft does not suppress the separate configuration list',async()=>{
 const {ctx,cards}=setup('models');await ctx.refreshCards();assert.equal(cards.children.length,2);assert.equal(cards.replacements,1);
});

test('running configurations keep a disabled delete action and stopped configurations can be deleted',async()=>{
 const {ctx,cards}=setup('models');ctx.api=async url=>({running:url.includes('/next/')});await ctx.refreshCards();
 const findDelete=el=>el.dataset.profileDelete?el:el.children.map(findDelete).find(Boolean);
 const actions=cards.children.map(findDelete);
 assert.equal(actions[0].disabled,true);assert.match(actions[0].title,/停止模型/);
 assert.equal(actions[1].disabled,false);assert.equal(actions[1].ariaLabel,'删除配置 27B');
});
test('live editor inside a configuration card remains protected',async()=>{
 const {ctx,cards}=setup('profiles');await ctx.refreshCards();assert.equal(cards.replacements,0);
});
test('opening a card editor during status requests prevents replacing its form',async()=>{
 const {ctx,cards}=setup('models');const pending=[];ctx.api=()=>new Promise(resolve=>pending.push(resolve));
 const task=ctx.refreshCards();ctx.editorOrigin='profiles';for(const resolve of pending)resolve({running:false});await task;assert.equal(cards.replacements,0);
});
