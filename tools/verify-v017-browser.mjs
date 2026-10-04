#!/usr/bin/env node
/** Disposable, CPU-only real-browser integration audit. Never run on an installation. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

assert.equal(process.env.SM75_DISPOSABLE_AUDIT, '1', 'Requires explicit disposable-audit guard');
assert.equal(process.platform, 'linux');
assert(fs.existsSync('/.dockerenv'), 'Only run inside the isolated audit container');
assert(!fs.existsSync('/dev/nvidia0'), 'This audit must not have GPU access');
const source = '/opt/sm75-workbench/console';
const browserRoot = '/work/browser-audit';
assert(fs.existsSync(path.join(source, 'server.mjs')));
assert(fs.existsSync(path.join(browserRoot, 'node_modules/playwright/index.mjs')));
const { chromium } = await import(pathToFileURL(path.join(browserRoot, 'node_modules/playwright/index.mjs')));
const runId = new Date().toISOString().replace(/[:.]/g, '-');
const evidence = `/work/beta/evidence/browser-${runId}`;
const data = `${browserRoot}/runs/${runId}/data`;
const base = 'http://127.0.0.1:1616';
fs.mkdirSync(evidence, { recursive: true });
fs.mkdirSync(data, { recursive: true, mode: 0o700 });
const checks = [];
const errors = [];
const requests = [];
const ws = [];
const mockCalls = [];
const transportFailures = [];
const watcherVisuals = [];
const auditNonce = `browser-${runId}`;
const auditReply = `AUDIT_MOCK_REPLY: ${auditNonce}`;
let browser, page, manager, cookie = '', restarting = false, hardwareFailure, expectedConsoleErrors = 0;
const sensitive = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function clean(value) {
  let text = String(value).replace(/([?&]token=)[A-Za-z0-9_-]+/g, '$1[redacted]');
  for (const secret of sensitive) if (secret) text = text.split(secret).join('[redacted]');
  return text;
}
function check(name, detail = {}) {
  checks.push({ name, ...detail });
  console.log(JSON.stringify({ passed: name, ...detail }));
}
async function waitUntil(label, fn, timeout = 60000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (manager && manager.exitCode !== null) throw Error(`Manager exited (${manager.exitCode}); see redacted log`);
    try { const value = await fn(); if (value) return value; } catch {}
    await sleep(250);
  }
  throw Error(`Timed out: ${label}`);
}
async function api(route, body) {
  const response = await fetch(base + route, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { Cookie: cookie, Origin: base, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) throw Error(`${route}: HTTP ${response.status} ${clean(await response.text())}`);
  return response;
}
async function dump(page, name) {
  fs.writeFileSync(path.join(evidence, `${name}.txt`), clean(await page.locator('body').innerText()));
  await page.screenshot({ path: path.join(evidence, `${name}.png`), fullPage: true });
}
async function stopChild(child) {
  if (!child || child.exitCode !== null) return;
  const stopped = once(child, 'exit');
  child.kill('SIGTERM');
  await Promise.race([stopped, sleep(5000).then(() => { if (child.exitCode === null) child.kill('SIGKILL'); })]);
}
const mock = http.createServer(async (req, res) => {
  if (req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(req.url?.endsWith('/models') ? { object: 'list', data: [{ id: 'audit-model', object: 'model', owned_by: 'audit' }] } : { status: 'ok' }));
  }
  if (req.method !== 'POST' || req.url !== '/v1/chat/completions') { res.writeHead(404); return res.end(); }
  try {
    let body = ''; for await (const chunk of req) body += chunk;
    const payload = JSON.parse(body);
    mockCalls.push({ model: payload.model, stream: payload.stream === true, messages: payload.messages?.length ?? 0,
      temperature: payload.temperature, top_k: payload.top_k, max_tokens: payload.max_tokens,
      tools: payload.tools?.length ?? 0, auditNonce: (JSON.stringify(payload.messages?.filter(message => message.role === 'user')) || '').includes(auditNonce) });
    const reply = mockCalls.at(-1).auditNonce ? auditReply : 'AUDIT_BACKGROUND_REPLY';
    if (!payload.stream) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ id: 'chatcmpl-audit', object: 'chat.completion', model: 'audit-model', choices: [{ index: 0, message: { role: 'assistant', content: reply }, finish_reason: 'stop' }], usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 } }));
    }
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
    const write = (delta, finish_reason = null, usage) => res.write(`data: ${JSON.stringify({ id: 'chatcmpl-audit', object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model: 'audit-model', choices: [{ index: 0, delta, finish_reason }], ...(usage ? { usage } : {}) })}\n\n`);
    write({ role: 'assistant', content: '' });
    await sleep(100); write({ content: reply });
    await sleep(100); write({}, 'stop', { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 });
    res.end('data: [DONE]\n\n');
  } catch { res.writeHead(500); res.end('synthetic mock error'); }
});
try {
  for (const port of [1616, 1617, 3085, 8000]) {
    try { await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(500) }); throw Error(`Audit port ${port} is already occupied`); }
    catch (error) { if (error.message.startsWith('Audit port')) throw error; }
  }
  if (fs.existsSync('/dsh/home')) {
    const saved = fs.existsSync('/dsh/identity-home') ? `/dsh/browser-home-${runId}` : '/dsh/identity-home';
    fs.renameSync('/dsh/home', saved);
    check('previous synthetic home preserved', { destination: saved });
  }
  fs.mkdirSync('/dsh/home', { recursive: true, mode: 0o700 });
  fs.chownSync('/dsh/home', 1000, 1000);
  const legacy = {
    'ui-theme': { fontSize: 14, preference: 'light' }, 'locale': { preference: 'zh-CN' },
    'agent-default-model': { provider: 'legacy-local', model: 'legacy-model', reasoningEffort: 'maximum' },
    'llm-pi-ai': { providers: { 'legacy-local': { api: 'openai-completions', baseURL: 'http://127.0.0.1:8000/v1', models: [{ id: 'legacy-model', contextWindow: 8192 }] } } },
  };
  fs.writeFileSync('/dsh/home/settings.yaml', JSON.stringify(legacy), { mode: 0o600 });
  fs.chownSync('/dsh/home/settings.yaml', 1000, 1000);
  fs.writeFileSync(path.join(data, 'profiles.json'), JSON.stringify([{ id: 'audit', name: 'Synthetic audit', args: ['audit-model', '--max-model-len', '8192'], port: 8000, format: 'fp8', backend: 'native', cacheRoot: path.join(data, 'cache') }]));
  await new Promise((resolve, reject) => { mock.once('error', reject); mock.listen(8000, '127.0.0.1', resolve); });
  const output = fs.openSync(path.join(data, 'manager-private.log'), 'a', 0o600);
  const observer=path.join(data,'audit-observe-harness.mjs');
  fs.writeFileSync(observer,`import fs from 'node:fs';import {Standalone} from '${pathToFileURL(path.join(source,'standalone.mjs'))}';const original=Standalone.prototype.startHarness;Standalone.prototype.startHarness=async function(...args){await original.apply(this,args);fs.writeFileSync('${path.join(data,'control-private.json')}',JSON.stringify({key:this.harness.controlKey,pid:this.harness.child.pid}),{mode:0o600});};`);
  manager = spawn(process.execPath, ['--import',observer,path.join(source, 'server.mjs')], {
    env: { ...process.env, CUDA_VISIBLE_DEVICES: '', SM75_SINGLE_CONTAINER: '1', SM75_CONSOLE_ROOT: data,
      SM75_CONSOLE_HOST: '127.0.0.1', SM75_CONSOLE_PORT: '1616', SM75_HARNESS_PROXY_PORT: '1617', SM75_HARNESS_PORT: '3085', DSH_HOME: '/dsh/home' },
    stdio: ['ignore', output, output],
  });
  fs.closeSync(output);
  await waitUntil('manager', async () => (await fetch(base + '/')).ok);
  const token = fs.readFileSync(path.join(data, 'key'), 'utf8').trim(); sensitive.push(token);
  sensitive.push(JSON.parse(fs.readFileSync(path.join(data, 'api-access.json'), 'utf8')).key);
  const login = await fetch(base + '/console-api/login', { method: 'POST', headers: { Origin: base, 'Content-Type': 'application/json' }, body: JSON.stringify({ token }) });
  assert.equal(login.status, 200);
  cookie = login.headers.get('set-cookie').split(';')[0]; sensitive.push(cookie);
  await api('/console-api/harness/start', { profile: 'audit' });
  await waitUntil('native Harness readiness', async () => (await fetch('http://127.0.0.1:3085/sm75/ready')).ok, 120000);
  check('real manager and Harness started with synthetic profile');
  browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, locale: 'zh-CN' });
  const separator = cookie.indexOf('=');
  await context.addCookies([{ name: cookie.slice(0, separator), value: cookie.slice(separator + 1), url: base, httpOnly: true, sameSite: 'Strict' }]);
  page = await context.newPage();
  page.on('pageerror', error => errors.push({ kind: 'javascript', message: clean(error.message), restarting }));
  page.on('console', msg => { if (msg.type() === 'error') errors.push({ kind: 'console', message: clean(msg.text()), location: clean(msg.location().url || ''), restarting }); });
  page.on('response', async response => {
    const url = new URL(response.url());
    if (url.origin === base) requests.push({ path: url.pathname, status: response.status(), restarting });
    if (url.origin === base && url.pathname === '/console-api/hardware' && response.status() === 400) {
      const body = await response.json().catch(() => ({}));
      hardwareFailure = { path: url.pathname, status: response.status(), reason: clean(body.error || '') };
    }
  });
  page.on('requestfailed', request => { const url = new URL(request.url()); if (url.origin === base) transportFailures.push({ path: url.pathname, error: request.failure()?.errorText, restarting }); });
  page.on('websocket', socket => {
    const item = { path: new URL(socket.url()).pathname, receivedFrames: 0, closed: false, errors: [] }; ws.push(item);
    socket.on('framereceived', frame => { item.receivedFrames++;  });
    socket.on('socketerror', error => item.errors.push(clean(error))); socket.on('close', () => { item.closed = true; });
  });

  const go = async id => {await page.locator(`#nav [data-page="${id}"]`).click();await page.waitForFunction(id=>document.querySelector('main').dataset.page===id,id);};
  await page.goto(base+'/?page=harness',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>globalThis.__SM75_WORKBENCH__?.ready,{timeout:60000});
  const onboarding=page.getByRole('button',{name:/^(继续|Continue)$/});
  await onboarding.waitFor({state:'visible',timeout:2000}).catch(()=>{});
  if(await onboarding.isVisible()) await onboarding.click();
  const later=page.getByRole('button',{name:/^(稍后配置|Configure later)$/});
  if(await later.isVisible())await later.click();
  const composer=page.locator('[contenteditable="true"]').first();
  await composer.waitFor({state:'visible',timeout:30000});
  await dump(page,'01-workbench');
  check('unified workbench boots at 1920x1080');
  if(process.argv.includes('--probe')) {
    console.log(JSON.stringify({controls:await page.locator('#workspaceSidebar button,#workspaceSidebar input').evaluateAll(nodes=>nodes.map(n=>({tag:n.tagName,label:n.getAttribute('aria-label'),placeholder:n.getAttribute('placeholder'),text:n.innerText}))),evidence}));
  } else {
    await composer.click();
    await composer.fill('Return the synthetic audit reply for '+auditNonce+'. Do not use tools.');
    await page.waitForFunction(()=>document.querySelector('[contenteditable="true"]')?.textContent.includes('synthetic audit reply'));
    await page.getByRole('button',{name:'发送消息',exact:true}).click();
    await page.locator('#root p').filter({hasText:auditReply}).last().waitFor({timeout:60000});
    const registry=()=>JSON.parse(fs.readFileSync('/dsh/home/storages/workspace.json','utf8'));
    const snapshot=registry();
    await dump(page,'02-session');
    check('native conversation completes and original workspace registry persists');
    const search=page.locator('#workspaceSidebar input').first();
    await page.locator('#workspaceSidebar button[aria-label]').filter({has:page.locator('svg')}).first().click();
    await search.fill('no-such-session-'+auditNonce);
    await page.getByText('无匹配会话',{exact:true}).waitFor();
    assert(await page.locator('.sm75-workspace-filter').isVisible());
    await page.getByRole('button',{name:'恢复列表',exact:true}).click();
    await page.getByText('无匹配会话',{exact:true}).waitFor({state:'hidden'});
    assert.deepEqual(registry(),snapshot);
    await page.locator('#workspaceSidebar button[aria-label]').filter({has:page.locator('svg')}).first().click();
    await search.fill(auditNonce);
    await page.getByRole('tree',{name:'搜索结果',exact:true}).getByText(/AUDIT_MOCK_REPLY/).first().waitFor();
    assert(fs.existsSync('/dsh/home/indexes/session-search.sqlite'));
    await page.getByRole('button',{name:'恢复列表',exact:true}).click();
    // Search must recover via the main workbench navigation as well.
    await page.locator('#workspaceSidebar button[aria-label]').filter({has:page.locator('svg')}).first().click();
    await search.fill('no-such-session-'+auditNonce);
    await page.getByText('无匹配会话',{exact:true}).waitFor();
    await go('plugins');await go('harness');
    await page.getByText('无匹配会话',{exact:true}).waitFor({state:'hidden'});
    assert.equal(await search.inputValue(),'');
    check('visible search state and both reset paths restore the native list without registry changes');
    await composer.fill('unsent draft survives switching');
    await page.evaluate(()=>window.betaComposer=document.querySelector('[contenteditable="true"]'));
    for(const id of ['usage','plugins','settings','harness'])await go(id);
    assert.equal(await composer.innerText(),'unsent draft survives switching');
    assert(await page.evaluate(()=>window.betaComposer===document.querySelector('[contenteditable="true"]')));
    check('native history, Watcher, plugins and unsent draft survive navigation');
    await composer.fill('');
    const modelDir=path.join(data,'new-int8');fs.mkdirSync(modelDir);fs.writeFileSync(path.join(modelDir,'config.json'),JSON.stringify({model_type:'llama',max_position_embeddings:32768,quantization_config:{quant_method:'compressed-tensors',config_groups:{g:{weights:{num_bits:8,type:'int'}}}}}));
    await api('/console-api/models/register',{path:modelDir,name:'Synthetic new INT8'});
    const originalProfiles=await (await api('/console-api/profiles')).json();
    await go('models');
    const card=page.locator('#localModelsView article').filter({hasText:'Synthetic new INT8'});
    await card.getByRole('button',{name:'配置并启动',exact:true}).click();
    await page.locator('#profileEditor').waitFor({state:'visible'});
    assert.equal(await page.locator('#format').inputValue(),'int8');
    await page.locator('#modelSourceKind').selectOption('saved');
    assert(await page.locator('#modelSourceTemplate option[value="audit"]').count());
    await page.locator('#modelSourceTemplate').selectOption('audit');
    await page.locator('#loadModelSource').click();
    await page.waitForFunction(()=>document.querySelector('#modelDraftNotice').textContent.includes('已清除'));
    assert.equal(await page.locator('#modelPath').inputValue(),modelDir);
    await page.locator('#modelSourceKind').selectOption('custom');
    await page.locator('#loadModelSource').click();
    await page.waitForFunction(()=>document.querySelector('#profileName').value.endsWith('自定义'));
    assert.equal(await page.locator('#modelPath').inputValue(),modelDir);

    assert.equal((await (await api('/console-api/profiles')).json()).length,originalProfiles.length);
    assert.equal((await (await api('/console-api/runtime')).json()).profile,null);
    for(const tab of ['basic','acceleration','cache','power','advanced']) {
      await page.locator(`[data-editor-tab="${tab}"]`).click();
      assert.equal(await page.locator('[data-editor-tab][aria-selected="true"]').count(),1);
      const expected={basic:'modelSettings',acceleration:'quickConfiguration',cache:'contextSettings',power:'powerSettings',advanced:'advancedParameters'}[tab];
      assert(await page.locator('#'+expected).isVisible());
      for(const other of ['modelSettings','quickConfiguration','contextSettings','powerSettings','advancedParameters'])
        if(other!==expected)assert.equal(await page.locator('#'+other).isVisible(),false,other+' must hide in '+tab);
    }
    await page.locator('[data-editor-tab="basic"]').click();
    const widths=await page.evaluate(()=>['modelLaunch','profileEditor','modelSettings'].map(id=>document.getElementById(id).getBoundingClientRect().width));
    assert(Math.max(...widths)-Math.min(...widths)<2);
    await dump(page,'03-model-draft');
    await page.locator('#save').click();
    await page.waitForFunction(()=>document.getElementById('notice').textContent.includes('已保存'));
    const saved=(await (await api('/console-api/profiles')).json()).find(p=>p.args[0]===modelDir);
    assert(saved);assert.equal(saved.format,'int8');assert.equal(saved.backend,'native');
    const invalid=await fetch(base+'/console-api/profiles/'+saved.id+'/start',{method:'POST',headers:{Cookie:cookie,Origin:base,'Content-Type':'application/json'},body:'{}'});
    assert.equal(invalid.status,400);assert.match((await invalid.json()).error,/权重/);
    assert.equal((await (await api('/console-api/runtime')).json()).profile,null);
    check('new INT8 model has generic draft, full-width horizontal editor, independent save and preflight rejection');
    await page.locator('#backModels').click();
    await go('harness');
    assert(await page.evaluate(()=>window.betaComposer===document.querySelector('[contenteditable="true"]')));
    assert.deepEqual(registry(),snapshot);
    // Same live Harness receives a managed-model update without process restart or session writes.
    const engineKey=JSON.parse(fs.readFileSync(path.join(data,'api-access.json'),'utf8')).key;
    const {key:controlKey,pid:dshPid}=JSON.parse(fs.readFileSync(path.join(data,'control-private.json'),'utf8'));
    sensitive.push(controlKey);
    const control=(body,token=controlKey)=>fetch('http://127.0.0.1:3085/sm75/model',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify(body)});
    const {managedHarnessConfig}=await import(pathToFileURL(path.join(source,'harness-config.mjs')));
    const next=managedHarnessConfig({args:['b','--served-model-name','audit-next'],port:8000});
    assert.equal((await control(next,'wrong-token')).status,401);
    const switched=await control(next);assert.equal(switched.status,200,await switched.text());
    const configured=await (await fetch('http://127.0.0.1:3085/sm75/configured-models')).json();
    assert(configured.models.some(m=>m.id==='sm75-local/audit-next'));
    assert(configured.models.some(m=>m.provider==='legacy-local'));
    assert.deepEqual(registry(),snapshot);
    assert(await page.evaluate(()=>window.betaComposer===document.querySelector('[contenteditable="true"]')));
    check('live model connection update retains other providers, original registry and mounted conversation');
    const callsBefore=mockCalls.length;
    await composer.click();await composer.fill('Return the synthetic audit reply for '+auditNonce+' after the model switch.');
    await page.getByRole('button',{name:'发送消息',exact:true}).click();
    await waitUntil('existing conversation uses the new model',()=>mockCalls.slice(callsBefore).some(call=>call.auditNonce&&call.model==='audit-next'));
    check('the existing conversation sends its next message to the newly selected local model');
    await waitUntil('conversation finishes before restart',async()=>!(await(await fetch('http://127.0.0.1:3085/sm75/activity',{headers:{Authorization:'Bearer '+controlKey}})).json()).busy);
    const beforeRestart=registry();
    restarting=true;
    await api('/console-api/harness/stop',{});
    await api('/console-api/harness/start',{profile:'audit'});
    await waitUntil('restarted native Harness',async()=>(await fetch('http://127.0.0.1:3085/sm75/ready')).ok,120000);
    await page.reload({waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>globalThis.__SM75_WORKBENCH__?.ready,{timeout:60000});
    await page.locator('#workspaceSidebar').getByText(/^(Return the synthetic audit reply|AUDIT_MOCK_REPLY: browser-)/).first().click();
    await page.locator('#root p').filter({hasText:auditReply}).last().waitFor();
    assert.deepEqual(registry(),beforeRestart);
    const coldCalls=mockCalls.length;
    await composer.fill('Return the synthetic audit reply for '+auditNonce+' after service restart.');
    await page.getByRole('button',{name:'发送消息',exact:true}).click();
    await waitUntil('reopened history resolves current model',()=>mockCalls.slice(coldCalls).some(call=>call.auditNonce&&call.model==='audit-model'));
    restarting=false;
    await dump(page,'04-restored-history');
    check('service restart preserves the workspace and reopens the original conversation on the active model');


    assert.equal(errors.filter(e=>e.kind==='javascript').length,0,JSON.stringify(errors));
  }
  fs.writeFileSync(path.join(evidence,'report.json'),JSON.stringify({checks,errors,requests,ws,mockCalls},null,2));
  console.log(JSON.stringify({evidence,checks:checks.length,javascriptErrors:errors.filter(e=>e.kind==='javascript').length}));
} catch(error) {
  if(page)await dump(page,'failure').catch(()=>{});
  fs.writeFileSync(path.join(evidence,'failure.json'),JSON.stringify({error:clean(error.stack),checks,errors,requests},null,2));
  console.error(clean(error.stack));process.exitCode=1;
} finally {
  if(browser)await browser.close();
  try{if(cookie)await api('/console-api/harness/stop',{});}catch{}
  await stopChild(manager);mock.closeAllConnections();await new Promise(resolve=>mock.close(resolve));
}
