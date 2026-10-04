import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import net from 'node:net';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {Store} from '../store.mjs';
import {argValue, setArg} from '../public/context-config.js';

const serverFile = fileURLToPath(new URL('../server.mjs', import.meta.url));
const credentialEnvironment = {
  VLLM_API_KEY: 'test-only-engine-credential-do-not-persist',
  OPENAI_API_KEY: 'test-only-provider-credential-do-not-persist',
  HF_TOKEN: 'test-only-download-credential-do-not-persist',
};

async function freePort() {
  const probe = net.createServer();
  await new Promise((resolve, reject) => {
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', resolve);
  });
  const port = probe.address().port;
  await new Promise((resolve, reject) => probe.close(error => error ? reject(error) : resolve()));
  return port;
}

function cleanEnvironment() {
  // Keep OS process support, not the developer's Docker overrides or credentials.
  const allowed = /^(path|pathext|systemroot|windir|comspec|systemdrive|temp|tmp|tmpdir|home|userprofile|localappdata|appdata|lang|lc_all)$/i;
  return Object.fromEntries(Object.entries(process.env).filter(([key]) => allowed.test(key)));
}

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-runtime-startup-http-'));
  const managers = [];
  const savedProfiles = () => JSON.parse(fs.readFileSync(path.join(root, 'profiles.json'), 'utf8'));
  const record = (kind, id) => {
    const store = new Store(root);
    try { return store.get(kind, id); } finally { store.close(); }
  };
  t.after(async () => {
    for (const manager of managers) await manager.stop();
    fs.rmSync(root, {recursive: true, force: true, maxRetries: 5, retryDelay: 50});
  });
  return {
    root, savedProfiles, record,
    model(name, config = {model_type: 'llama', torch_dtype: 'float16', max_position_embeddings: 4096}) {
      const dir = path.join(root, name);
      fs.mkdirSync(dir, {recursive: true});
      fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify(config));
      return dir;
    },
    profile(id, model) {
      return {id, name: id, backend: 'native', format: 'fp16', port: 8000,
        cacheRoot: path.join(root, 'cache'),
        args: [model, '--tensor-parallel-size', '1', '--max-model-len', '4096', '--seed', '42'],
        env: {NCCL_P2P_LEVEL: 'SYS'}};
    },
    seed(profiles, settings = {}) {
      fs.writeFileSync(path.join(root, 'profiles.json'), JSON.stringify(profiles));
      const store = new Store(root);
      try {
        // Directly seed persisted legacy state so even an inconsistent autoStart
        // flag can prove failed import suppresses the startup path.
        store.put('settings', 'main', {modelRoot: path.join(root, 'models'),
          cacheRoot: path.join(root, 'cache'), defaultProfile: '', autoStart: false,
          defaultChatProfile: '', source: 'huggingface', ...settings});
      } finally { store.close(); }
    },
    async launch(overrides) {
      const port = await freePort();
      const base = `http://127.0.0.1:${port}`;
      let output = '', spawnError;
      const child = spawn(process.execPath, [serverFile], {cwd: path.dirname(serverFile),
        env: {...cleanEnvironment(), ...credentialEnvironment, SM75_SINGLE_CONTAINER: '1',
          SM75_CONSOLE_ROOT: root, SM75_CONSOLE_HOST: '127.0.0.1',
          SM75_CONSOLE_PORT: String(port), ...overrides},
        stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true});
      child.stdout.on('data', chunk => { output += chunk; });
      child.stderr.on('data', chunk => { output += chunk; });
      child.on('error', error => { spawnError = error; });
      const exited = new Promise(resolve => child.once('exit', resolve));
      const manager = {
        pid: child.pid,
        output: () => output,
        async stop() {
          if (!child.pid || child.exitCode !== null || child.signalCode !== null) return;
          child.kill('SIGTERM');
          let timer;
          const stopped = await Promise.race([exited.then(() => true),
            new Promise(resolve => { timer = setTimeout(() => resolve(false), 2000); })]);
          clearTimeout(timer);
          if (!stopped) { child.kill('SIGKILL'); await exited; }
        },
        async api(route, body) {
          const headers = {Authorization: 'Bearer ' + fs.readFileSync(path.join(root, 'key'), 'utf8').trim()};
          if (body !== undefined) headers['Content-Type'] = 'application/json';
          const response = await fetch(base + '/console-api/' + route, {
            method: body === undefined ? 'GET' : 'POST', headers,
            ...(body === undefined ? {} : {body: JSON.stringify(body)}), signal: AbortSignal.timeout(3000)});
          assert.equal(response.status, 200, `${route} HTTP status`);
          return response.json();
        },
      };
      managers.push(manager);
      const deadline = Date.now() + 8000;
      while (Date.now() < deadline) {
        assert.equal(spawnError, undefined, 'manager process must launch');
        assert.equal(child.exitCode, null, 'manager must stay alive until ready');
        assert.equal(child.signalCode, null, 'manager must not be terminated during startup');
        try {
          const response = await fetch(base + '/console-api/session', {signal: AbortSignal.timeout(300)});
          if (response.ok) return manager;
        } catch {}
        await delay(30);
      }
      assert.fail('isolated manager must become ready within 8 seconds');
    },
  };
}

function assertNoCredentialLeaks(manager, ...values) {
  const visible = JSON.stringify(values) + manager.output();
  for (const secret of Object.values(credentialEnvironment))
    assert.equal(visible.includes(secret), false, 'startup responses, diagnostics and saved profiles must not expose credentials');
}

async function assertStopped(manager, files) {
  assert.deepEqual(await manager.api('runtime'), {profile: null, phase: 'stopped', error: null});
  assert.equal(files.record('startup', 'last'), null, 'autoStart must not be attempted after import failure');
}

test('Docker TP imports once; panel POST remains authoritative in the same manager process', async t => {
  const files = fixture(t);
  const original = files.profile('existing-model', files.model('model'));
  files.seed([original]);
  const manager = await files.launch({TENSOR_PARALLEL_SIZE: '4'});
  const imported = await manager.api('profiles');
  assert.equal(imported.length, 1);
  assert.equal(imported[0].id, original.id);
  assert.equal(argValue(imported[0].args, '--tensor-parallel-size'), '4');
  assert.equal(argValue(imported[0].args, '--seed'), '42', 'unrelated advanced parameters must survive import');
  const settings = await manager.api('settings');
  assert.equal(settings.defaultProfile, original.id);
  assert.equal(settings.runtimeEnvironment.state, 'applied');
  assert.equal(settings.runtimeEnvironment.profile, original.id);

  const edited = {...imported[0], args: setArg(imported[0].args, '--tensor-parallel-size', '2')};
  const posted = await manager.api('profiles', edited);
  assert.equal(argValue(posted[0].args, '--tensor-parallel-size'), '2');
  for (let request = 0; request < 3; request++) {
    const current = await manager.api('profiles');
    assert.equal(argValue(current[0].args, '--tensor-parallel-size'), '2');
    assert.deepEqual((await manager.api('settings')).runtimeEnvironment, settings.runtimeEnvironment,
      'reading APIs must not reimport Docker overrides');
  }
  assert.equal(argValue(files.savedProfiles()[0].args, '--tensor-parallel-size'), '2');
  assertNoCredentialLeaks(manager, imported, posted, settings, files.savedProfiles());
});

test('MODEL on a fresh installation creates a persisted default and reuses it after manager restart', async t => {
  const files = fixture(t);
  const model = files.model('fresh-model', {model_type: 'llama', quantization_config: {quant_method: 'fp8'}});
  files.seed([]);
  const environment = {MODEL: model, TENSOR_PARALLEL_SIZE: '4', SERVED_MODEL_NAME: 'docker-selected-model'};
  let manager = await files.launch(environment);
  const first = await manager.api('profiles');
  assert.equal(first.length, 1);
  assert.equal(first[0].args[0], model);
  assert.equal(first[0].format, 'fp8');
  assert.equal(first[0].backend, 'native');
  assert.equal(argValue(first[0].args, '--tensor-parallel-size'), '4');
  assert.equal(argValue(first[0].args, '--served-model-name'), 'docker-selected-model');
  assert.equal(first[0].power.gpus, '0,1,2,3');
  assert.equal((await manager.api('settings')).defaultProfile, first[0].id);
  assert.deepEqual(files.savedProfiles(), first);
  await manager.stop();
  manager = await files.launch(environment);
  const second = await manager.api('profiles');
  assert.equal(second.length, 1);
  assert.equal(second[0].id, first[0].id, 'restart must reuse the same profile identity');
  assert.equal((await manager.api('settings')).defaultProfile, first[0].id);
  assertNoCredentialLeaks(manager, first, second, await manager.api('settings'), files.savedProfiles());
});

test('ambiguous TP import keeps profiles intact and blocks a persisted autoStart flag', async t => {
  const files = fixture(t);
  const profiles = ['first', 'second'].map(id => files.profile(id, files.model(id)));
  files.seed(profiles, {autoStart: true});
  const manager = await files.launch({TENSOR_PARALLEL_SIZE: '4'});
  const settings = await manager.api('settings');
  assert.equal(settings.runtimeEnvironment.state, 'failed');
  assert.equal(settings.defaultProfile, '');
  assert.equal(settings.autoStart, true, 'import failure must not rewrite the persisted preference');
  assert.equal(typeof settings.runtimeEnvironment.error, 'string');
  assert.deepEqual(await manager.api('profiles'), profiles);
  assert.deepEqual(files.savedProfiles(), profiles);
  await assertStopped(manager, files);
  assertNoCredentialLeaks(manager, settings, profiles, files.record('startup', 'environment'));
});

test('explicit new MODEL appends a default while retaining other profiles across restart', async t => {
  const files = fixture(t);
  const profiles = ['first', 'second'].map(id => files.profile(id, files.model(id)));
  const model = files.model('third-model');
  files.seed(profiles);
  const environment = {MODEL: model, TENSOR_PARALLEL_SIZE: '2'};
  let manager = await files.launch(environment);
  const imported = await manager.api('profiles');
  assert.equal(imported.length, 3);
  assert.deepEqual(imported.slice(0, 2), profiles);
  const selected = imported.find(row => row.args[0] === model);
  assert.ok(selected);
  assert.equal(argValue(selected.args, '--tensor-parallel-size'), '2');
  assert.equal((await manager.api('settings')).defaultProfile, selected.id);
  await manager.stop();
  manager = await files.launch(environment);
  const restarted = await manager.api('profiles');
  assert.equal(restarted.length, 3);
  assert.deepEqual(restarted.slice(0, 2), profiles);
  assert.equal(restarted.find(row => row.args[0] === model).id, selected.id);
  assertNoCredentialLeaks(manager, imported, restarted, await manager.api('settings'));
});

test('MODEL with one matching profile selects that identity without creating another configuration', async t => {
  const files = fixture(t);
  const profiles = ['first', 'second'].map(id => files.profile(id, files.model(id)));
  files.seed(profiles);
  const manager = await files.launch({MODEL: profiles[1].args[0], TENSOR_PARALLEL_SIZE: '4'});
  const imported = await manager.api('profiles');
  assert.equal(imported.length, 2);
  assert.deepEqual(imported[0], profiles[0]);
  assert.equal(imported[1].id, profiles[1].id);
  assert.equal(argValue(imported[1].args, '--tensor-parallel-size'), '4');
  const settings = await manager.api('settings');
  assert.equal(settings.defaultProfile, profiles[1].id);
  assert.equal(settings.runtimeEnvironment.state, 'applied');
  assertNoCredentialLeaks(manager, imported, settings);
});

test('MODEL matching multiple profiles without a default rejects ambiguity without changing profiles', async t => {
  const files = fixture(t);
  const model = files.model('shared-model');
  const profiles = ['first', 'second'].map(id => files.profile(id, model));
  files.seed(profiles);
  const manager = await files.launch({MODEL: model, TENSOR_PARALLEL_SIZE: '4'});
  const settings = await manager.api('settings');
  assert.equal(settings.runtimeEnvironment.state, 'failed');
  assert.equal(settings.defaultProfile, '');
  assert.deepEqual(await manager.api('profiles'), profiles);
  assert.deepEqual(files.savedProfiles(), profiles);
  await assertStopped(manager, files);
  assertNoCredentialLeaks(manager, settings, profiles, files.record('startup', 'environment'));
});

test('invalid Docker JSON reports an import error without echoing credential-bearing input', async t => {
  const files = fixture(t);
  const profiles = [files.profile('existing-model', files.model('model'))];
  files.seed(profiles);
  const manager = await files.launch({HF_OVERRIDES: '{"token":"' + credentialEnvironment.HF_TOKEN + '"'});
  const settings = await manager.api('settings');
  assert.equal(settings.runtimeEnvironment.state, 'failed');
  assert.equal(settings.defaultProfile, '');
  assert.deepEqual(await manager.api('profiles'), profiles);
  assert.deepEqual(files.savedProfiles(), profiles);
  await assertStopped(manager, files);
  assertNoCredentialLeaks(manager, settings, profiles, files.record('startup', 'environment'));
});

test('explicit unmatched MODEL preserves a lone existing profile when no default is selected', async t => {
  const files = fixture(t);
  const original = files.profile('existing-model', files.model('existing-model'));
  const model = files.model('new-model');
  files.seed([original]);
  const manager = await files.launch({MODEL: model, TENSOR_PARALLEL_SIZE: '2'});
  const imported = await manager.api('profiles');
  assert.equal(imported.length, 2, 'new MODEL must not retarget the only unrelated configuration');
  assert.deepEqual(imported[0], original);
  assert.notEqual(imported[1].id, original.id);
  assert.equal(imported[1].args[0], model);
  assert.equal(argValue(imported[1].args, '--tensor-parallel-size'), '2');
  const settings = await manager.api('settings');
  assert.equal(settings.defaultProfile, imported[1].id);
  assert.equal(settings.runtimeEnvironment.state, 'applied');
  assert.deepEqual(files.savedProfiles(), imported);
  assertNoCredentialLeaks(manager, imported, settings);
});

test('Docker API key imports at startup and panel replacement remains effective until restart', async t => {
  const files = fixture(t);
  files.seed([]);
  const dockerKey = 'test-only-explicit-docker-api-credential';
  const panelKey = 'test-only-panel-api-credential';
  let manager = await files.launch({VLLM_API_KEY: dockerKey});
  const beforeMetadata = await manager.api('api-access');
  assert.equal(beforeMetadata.configured, true);
  assert.equal(beforeMetadata.running, false);
  assert.equal(Object.hasOwn(beforeMetadata, 'key'), false);
  assert.equal((await manager.api('api-access/reveal', {source: 'saved'})).key, dockerKey);
  assert.equal((await manager.api('api-access/reveal', {source: 'current'})).key, dockerKey);

  const saved = await manager.api('api-access', {key: panelKey});
  assert.deepEqual(saved, {configured: true, restartRequired: false});
  for (let request = 0; request < 3; request++) {
    assert.equal((await manager.api('api-access/reveal', {source: 'saved'})).key, panelKey);
    assert.equal((await manager.api('api-access/reveal', {source: 'current'})).key, panelKey);
  }
  assert.equal(JSON.parse(fs.readFileSync(path.join(files.root, 'api-access.json'), 'utf8')).key, panelKey);
  assert.equal(fs.readFileSync(path.join(files.root, 'engine-key.current'), 'utf8'), panelKey);
  const afterMetadata = await manager.api('api-access');
  assert.equal(Object.hasOwn(afterMetadata, 'key'), false);
  const visible = JSON.stringify([beforeMetadata, afterMetadata, saved,
    await manager.api('settings'), await manager.api('profiles')]) + manager.output();
  for (const value of [dockerKey, panelKey, ...Object.values(credentialEnvironment)])
    assert.equal(visible.includes(value), false, 'ordinary metadata and startup diagnostics must not contain API keys');

  await manager.stop();
  manager = await files.launch({VLLM_API_KEY: dockerKey});
  assert.equal((await manager.api('api-access/reveal', {source: 'saved'})).key, dockerKey,
    'a new manager process intentionally reapplies the explicit Docker API key');
  const restartedVisible = JSON.stringify(await manager.api('api-access')) + manager.output();
  for (const value of [dockerKey, panelKey]) assert.equal(restartedVisible.includes(value), false);
});

test('SM75_DEFAULT_PROFILE alone imports the selected configuration without extra model or TP overrides', async t => {
  const files = fixture(t);
  const profiles = ['first', 'second'].map(id => files.profile(id, files.model(id)));
  files.seed(profiles, {defaultProfile: 'first'});
  const manager = await files.launch({SM75_DEFAULT_PROFILE: 'second'});
  const settings = await manager.api('settings');
  assert.equal(settings.defaultProfile, 'second');
  assert.equal(settings.runtimeEnvironment.state, 'applied');
  assert.equal(settings.runtimeEnvironment.profile, 'second');
  const imported = await manager.api('profiles');
  assert.equal(imported.length, 2);
  assert.deepEqual(imported[0], profiles[0], 'the previous default must remain unchanged');
  assert.equal(imported[1].id, profiles[1].id);
  assert.equal(imported[1].args[0], profiles[1].args[0]);
  assert.equal(argValue(imported[1].args, '--tensor-parallel-size'), '1');
  assert.equal(files.record('settings', 'main').defaultProfile, 'second');
  assertNoCredentialLeaks(manager, settings, imported, files.savedProfiles());
});
