import test from 'node:test';
import assert from 'node:assert/strict';
import {setManagedModel,resolveManagedSelection} from '../plugins/sm75-workbench/lib/managed-model.js';
import {patchClientModelSelection,patchServerModelSelection} from '../native-model-selection.mjs';
test('old local history follows the engine while explicit external and same-model settings survive',()=>{
 setManagedModel({provider:'sm75-local',model:'new'});
 const history=Object.freeze({provider:'sm75-local',model:'old',reasoningEffort:'high'});
 assert.deepEqual(resolveManagedSelection(history),{provider:'sm75-local',model:'new'});
 assert.equal(history.model,'old');
 const same={provider:'sm75-local',model:'new',reasoningEffort:'low'};
 assert.equal(resolveManagedSelection(same),same);
 const external={provider:'external',model:'old',reasoningEffort:'high'};
 assert.equal(resolveManagedSelection(external),external);
 assert.equal(resolveManagedSelection(undefined),undefined);
 assert.throws(()=>setManagedModel({provider:'external',model:'other'}));
 assert.equal(resolveManagedSelection(history).model,'new');
});
test('client restores cold history from the managed catalog even with an external global default',()=>{
 const input='const current = projected.next ?? catalog.value.default;';
 const patched=patchClientModelSelection(input);
 const run=new Function('projected','catalog',patched+';return current;');
 const catalog={value:{default:{provider:'external',model:'external'},groups:[{id:'sm75-local',models:[{id:'active'}]}]}};
 assert.deepEqual(run({next:{provider:'sm75-local',model:'old',reasoningEffort:'high'}},catalog),{provider:'sm75-local',model:'active'});
 assert.deepEqual(run({next:null},catalog),catalog.value.default);
 assert.equal(patchClientModelSelection(patched),patched);
 assert.throws(()=>patchClientModelSelection('unknown release'),/anchor/);
 assert.throws(()=>patchServerModelSelection('unknown release'),/anchor/);
});
