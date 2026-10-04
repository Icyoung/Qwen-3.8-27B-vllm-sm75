import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import vm from 'node:vm';
import {EventEmitter} from 'node:events';
import {Standalone} from '../standalone.mjs';

const appSource = fs.readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');
const serverSource = fs.readFileSync(new URL('../server.mjs', import.meta.url), 'utf8');
function sourceBetween(source, start, end) {
  const from = source.indexOf(start), to = source.indexOf(end, from + start.length);
  assert(from >= 0 && to > from, 'Actual function boundaries must remain available');
  return source.slice(from, to);
}
function editorHarness() {
  const elements = new Map();
  const $ = id => {
    if (!elements.has(id)) elements.set(id, {
      id, hidden: true, inert: false, disabled: false, value: '', textContent: '',
      parentElement: {hidden: false}, children: [], append() {}, scrollIntoView() {}, classList: {remove() {}},
    });
    return elements.get(id);
  };
  const state = {draft: {id: 'saved-profile', name: 'Original profile', args: ['/models/a']}, fills: 0};
  const sandbox = {
    $, console, document: {querySelectorAll: () => [], querySelector: () => null},
    modelSourceGeneration: 0, editorOrigin: 'profiles', modelLaunchId: null,
    editorDirty: false, editorTab: 'basic', profilesSignature: '', current: 'models',
    resolveDraft: async () => true,
    populateModelSources() { $('modelSourceTemplate').value = 'base:default'; },
    page: async () => {}, run: fn => fn, syncFeatures() {}, selectEditorTab() {}, requestAnimationFrame: fn => fn(),
    editedProfile: () => structuredClone(state.draft),
    fillProfile(profile) {
      state.draft = structuredClone(profile); state.fills++;
      $('mainModelSelect').value = profile.args[0];
    },
    catalog: [
      {id: 'b', path: '/models/b', name: 'Synthetic B'},
      {id: 'c', path: '/models/c', name: 'Synthetic C'},
      {id: 'a', path: '/models/a', name: 'Synthetic A'},
    ],
    api: async () => ({profile: {id: 'draft', args: ['/models/a']}, notices: []}),
  };
  vm.createContext(sandbox);
  vm.runInContext(
    sourceBetween(appSource, '  function closeEditor() {', '  async function refreshProfileCards(') +
    sourceBetween(appSource, '  function setModelSourceLoading(', '  $("localModelsTab").onclick') +
    sourceBetween(appSource, "  $('mainModelSelect').onchange =", '  $("chooseMtp").onclick'),
    sandbox,
  );
  return {$, sandbox, state};
}

test('model draft first-load failure recovers through the existing load-config button', async () => {
  const {$, sandbox, state} = editorHarness();
  let first = true;
  sandbox.api = async () => {
    if (first) {first = false; throw Error('synthetic temporary draft failure');}
    return {profile: {id: 'new-draft', args: ['/models/a']}, notices: []};
  };
  await assert.rejects(sandbox.openModelLaunch({id: 'a', name: 'Synthetic A'}), /synthetic/);
  assert.equal($('profileEditor').hidden, true);
  await $('loadModelSource').onclick();
  assert.equal(state.fills, 1);
  assert.equal($('profileEditor').hidden, false, 'Successful retry must reveal the populated editor');
  assert.equal($('profileEditor').inert, false);
  assert.equal($('loadModelSource').disabled, false);
});

test('model dropdown ignores an older result after the later choice has completed', async () => {
  const {$, sandbox, state} = editorHarness();
  const pending = new Map();
  sandbox.api = (_url, body) => new Promise(resolve => pending.set(body.model, resolve));
  $('profileEditor').hidden = false;
  $('mainModelSelect').value = '/models/b';
  const older = $('mainModelSelect').onchange();
  $('mainModelSelect').value = '/models/c';
  const newer = $('mainModelSelect').onchange();
  pending.get('c')({profile: {id: 'draft-c', args: ['/models/c']}, notices: []});
  await newer;
  assert.equal(state.draft.args[0], '/models/c');
  pending.get('b')({profile: {id: 'draft-b', args: ['/models/b']}, notices: []});
  await older;
  assert.equal(state.draft.args[0], '/models/c', 'An older response must not replace the last model choice');
  assert.equal($('mainModelSelect').value, '/models/c');
  assert.equal($('profileEditor').inert, false);
});

test('returning to the current model or closing the editor invalidates an outstanding retarget', async () => {
  for (const action of ['original-model', 'close']) {
    const {$, sandbox, state} = editorHarness();
    let resolve;
    sandbox.api = () => new Promise(done => {resolve = done;});
    $('profileEditor').hidden = false;
    $('mainModelSelect').value = '/models/b';
    const request = $('mainModelSelect').onchange();
    if (action === 'original-model') {
      $('mainModelSelect').value = '/models/a';
      await $('mainModelSelect').onchange();
    } else sandbox.closeEditor();
    resolve({profile: {id: 'draft-b', args: ['/models/b']}, notices: []});
    await request;
    assert.equal(state.draft.args[0], '/models/a', action + ' must invalidate stale model response');
    assert.equal(state.fills, 0);
    assert.equal($('profileEditor').inert, false);
  }
});

test('engine credential file follows actual start and exit transitions after key replacement', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-followup-credentials-'));
  const currentFile = path.join(root, 'engine-key.current'), savedFile = path.join(root, 'api-access.json');
  const originalUnlink = fs.unlinkSync;
  t.mock.method(fs, 'unlinkSync', file => {
    if (file === '/tmp/pstate-state.json') return;
    return originalUnlink(file);
  });
  try {
    fs.writeFileSync(savedFile, JSON.stringify({key: 'synthetic-first-engine-key'}));
    const keyContext = vm.createContext({fs, path, root, key: 'synthetic-web-login', standalone: null});
    vm.runInContext(
      sourceBetween(serverSource, 'const configuredEngineKey = () => {', 'const originalTests = ') +
      '\nglobalThis.syncKey = syncEngineKeyFile;', keyContext,
    );
    let transitions = 0;
    const manager = new Standalone(root, 'synthetic-web-login', {
      onEngineChange: () => {transitions++; keyContext.syncKey();},
    });
    keyContext.standalone = manager;
    manager.prepareCache = () => ({});
    const children = [], launchKeys = [];
    manager.launch = async (_bin, _args, env) => {
      const child = new EventEmitter(); child.exitCode = null; child.signalCode = null;
      children.push(child); launchKeys.push(env.VLLM_API_KEY); return child;
    };
    const profile = {id: 'synthetic-profile', args: ['synthetic-model'], port: 8000};
    await manager.start(profile);
    assert.equal(fs.readFileSync(currentFile, 'utf8'), launchKeys[0]);
    fs.writeFileSync(savedFile, JSON.stringify({key: 'synthetic-replacement-engine-key'}));
    keyContext.syncKey();
    assert.equal(fs.readFileSync(currentFile, 'utf8'), launchKeys[0], 'A live engine retains its accepted credential');
    children[0].emit('exit', 0);
    assert.equal(manager.engine, null);
    assert.equal(fs.readFileSync(currentFile, 'utf8'), 'synthetic-replacement-engine-key', 'Exiting the old engine must publish the pending credential');
    await manager.start(profile);
    assert.equal(launchKeys[1], 'synthetic-replacement-engine-key');
    assert.equal(fs.readFileSync(currentFile, 'utf8'), launchKeys[1], 'The upstream test adapter file must match the new engine');
    const beforeStaleExit = transitions;
    children[0].emit('exit', 0);
    assert.equal(manager.engine.child, children[1]);
    assert.equal(transitions, beforeStaleExit, 'A stale exit must not publish an engine transition');
    assert.equal(fs.readFileSync(currentFile, 'utf8'), launchKeys[1]);
    children[1].emit('exit', 0);
  } finally {
    assert.equal(path.dirname(root), os.tmpdir());
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test('a late failed model request cannot roll back a newer successful selection', async () => {
  const {$, sandbox, state} = editorHarness();
  const pending = new Map();
  sandbox.api = (_url, body) => new Promise((resolve, reject) => pending.set(body.model, {resolve, reject}));
  $('profileEditor').hidden = false;
  $('mainModelSelect').value = '/models/b';
  const older = $('mainModelSelect').onchange();
  $('mainModelSelect').value = '/models/c';
  const newer = $('mainModelSelect').onchange();
  pending.get('c').resolve({profile: {id: 'draft-c', args: ['/models/c']}, notices: []});
  await newer;
  pending.get('b').reject(Error('synthetic late failure'));
  await assert.doesNotReject(older, 'A stale failure must not surface as the latest action failure');
  assert.equal(state.draft.args[0], '/models/c');
  assert.equal($('mainModelSelect').value, '/models/c', 'A stale catch must not reset the dropdown');
  assert.equal($('profileEditor').inert, false);
});

test('opening another saved profile invalidates a pending model change in the previous editor', async () => {
  const {$, sandbox, state} = editorHarness();
  let resolve;
  sandbox.api = () => new Promise(done => {resolve = done;});
  $('profileEditor').hidden = false;
  $('mainModelSelect').value = '/models/b';
  const pending = $('mainModelSelect').onchange();
  const next = {id: 'different-saved-profile', name: 'Next saved profile', args: ['/models/d']};
  await sandbox.editProfile(next);
  resolve({profile: {id: 'draft-b', args: ['/models/b']}, notices: []});
  await pending;
  assert.deepEqual(state.draft, next, 'The response from the previous editor must not modify the next saved profile');
  assert.equal($('mainModelSelect').value, '/models/d');
  assert.equal($('profileEditor').hidden, false);
  assert.equal($('profileEditor').inert, false);
});
