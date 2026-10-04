import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');
function sourceBetween(start, end) {
  const from = source.indexOf(start), to = source.indexOf(end, from + start.length);
  assert(from >= 0 && to > from, 'Actual startup function boundaries must remain available');
  return source.slice(from, to);
}
function workspaceHarness(load) {
  const elements = new Map(), errors = [];
  const $ = id => {
    if (!elements.has(id)) elements.set(id, {id, value: '', textContent: '', hidden: false, disabled: false});
    return elements.get(id);
  };
  const sandbox = {
    $, AbortSignal, console, disposed: false, authTask: null, authGeneration: 0, authEpoch: 0,
    authState: 'checking', accountConfigured: true, abort: null,
    workspaceLoaded: false, workspaceLoadTask: null,
    shell: {dataset: {}, querySelectorAll: () => []},
    document: {querySelector: () => $('aside')},
    stopLiveSummary: {refresh() {}},
    location: {replace: () => assert.fail('Synthetic fixture has no native shell to replace')},
    fetch: async () => ({ok: true, json: async () => ({authenticated: true, accountConfigured: true})}),
    syncLoginMode() {}, report: error => errors.push(error.message), load,
  };
  vm.createContext(sandbox);
  vm.runInContext(
    sourceBetween('  function setAuth(', '  async function api(') +
    sourceBetween('  async function enterWorkspace() {', '  setAuth("checking");'),
    sandbox,
  );
  return {sandbox, errors};
}
function deferred() {
  let resolve;
  const promise = new Promise(done => {resolve = done;});
  return {promise, resolve};
}

test('authenticated initial-load failure retries successfully and later checks preserve loaded drafts', async () => {
  let attempts = 0;
  const {sandbox, errors} = workspaceHarness(async () => {
    if (++attempts === 1) throw Error('synthetic temporary settings failure');
  });
  await sandbox.checkSession();
  assert.equal(sandbox.authState, 'authenticated');
  assert.equal(sandbox.workspaceLoaded, false);
  assert.equal(errors.length, 1);
  await sandbox.checkSession();
  assert.equal(attempts, 2, 'A valid session must retry unfinished workspace initialization');
  assert.equal(sandbox.workspaceLoaded, true);
  await sandbox.checkSession();
  assert.equal(attempts, 2, 'A completed initialization must not reload and overwrite editing state');
});

test('concurrent workspace entry shares one initial data load', async () => {
  const pending = deferred(); let attempts = 0;
  const {sandbox} = workspaceHarness(async () => {attempts++; await pending.promise;});
  const first = sandbox.enterWorkspace(), second = sandbox.enterWorkspace();
  await Promise.resolve();
  assert.equal(attempts, 1, 'Concurrent entry must merge the initialization request');
  pending.resolve(); await Promise.all([first, second]);
  assert.equal(sandbox.workspaceLoaded, true);
  assert.equal(sandbox.workspaceLoadTask, null);
});

test('a load that finishes after logout cannot mark the workspace ready', async () => {
  const pending = deferred();
  const {sandbox} = workspaceHarness(() => pending.promise);
  const first = sandbox.enterWorkspace();
  sandbox.authGeneration++;
  sandbox.setAuth('anonymous', 'Synthetic signout');
  pending.resolve(); await first;
  assert.equal(sandbox.authState, 'anonymous');
  assert.equal(sandbox.workspaceLoaded, false);
  assert.equal(sandbox.workspaceLoadTask, null);
});

test('the previous login completion cannot clear or complete a new login load', async () => {
  const previous = deferred(), current = deferred(); let attempts = 0;
  const {sandbox} = workspaceHarness(() => (++attempts === 1 ? previous : current).promise);
  const oldEntry = sandbox.enterWorkspace();
  sandbox.authGeneration++;
  sandbox.setAuth('anonymous', 'Synthetic signout');
  const newEntry = sandbox.enterWorkspace();
  await Promise.resolve();
  assert.equal(attempts, 2);
  const currentTask = sandbox.workspaceLoadTask;
  previous.resolve(); await oldEntry;
  assert.equal(sandbox.workspaceLoaded, false, 'Old login completion must not mark new login ready');
  assert.equal(sandbox.workspaceLoadTask, currentTask, 'Old cleanup must retain the new initialization task');
  current.resolve(); await newEntry;
  assert.equal(sandbox.workspaceLoaded, true);
  assert.equal(sandbox.workspaceLoadTask, null);
});
