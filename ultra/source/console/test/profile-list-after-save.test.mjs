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
  $:id=>id==='profileEditor'?editor:cards,api:async()=>({running:false}),document:{createElement:el},profileLabel:p=>p.name,run:f=>f,editProfile(){},shareProfile(){},launchProfile(){}};
 const ctx=vm.createContext(state);vm.runInContext(source.slice(begin,end)+'\nglobalThis.refreshCards=refreshProfileCards;',ctx);
 return {ctx,editor,cards};
}
test('saved model-library draft does not suppress the separate configuration list',async()=>{
 const {ctx,cards}=setup('models');await ctx.refreshCards();assert.equal(cards.children.length,2);assert.equal(cards.replacements,1);
});
test('live editor inside a configuration card remains protected',async()=>{
 const {ctx,cards}=setup('profiles');await ctx.refreshCards();assert.equal(cards.replacements,0);
});
test('opening a card editor during status requests prevents replacing its form',async()=>{
 const {ctx,cards}=setup('models');const pending=[];ctx.api=()=>new Promise(resolve=>pending.push(resolve));
 const task=ctx.refreshCards();ctx.editorOrigin='profiles';for(const resolve of pending)resolve({running:false});await task;assert.equal(cards.replacements,0);
});
