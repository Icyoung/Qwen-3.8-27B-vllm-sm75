import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {requestRates as summaryRequestRates} from '../live-summary.mjs';

const html = fs.readFileSync(new URL('../../vllm/entrypoints/serve/instrumentator/dashboard.html', import.meta.url), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const coreContext = vm.createContext({});
vm.runInContext(source, coreContext);
const core = coreContext.MonitorCore;
const plain = value => JSON.parse(JSON.stringify(value));
const gauges = (seconds, prefill = 240, decode = 25) => core.parsePrometheus([
  `vllm:sm75_request_prefill_tokens_per_second ${prefill}`,
  `vllm:sm75_request_decode_tokens_per_second ${decode}`,
  `vllm:sm75_request_timing_updated_seconds ${seconds}`,
  'vllm:sm75_request_timing_active 1',
].join('\n'));

test('dashboard request rates share the header freshness and exact-count contract', () => {
  for (const sample of [gauges(10), gauges(10, 0, 0), gauges(10, 'NaN', 50), gauges(10, -1, 12), {}]) {
    for (const now of [9999, 10000, 25000, 25001]) {
      assert.deepEqual(plain(core.requestRates(sample, now)), summaryRequestRates(sample, now));
    }
  }
});

test('ambiguous request series cannot be displayed as exact request rates', () => {
  const data = gauges(10);
  data['vllm:sm75_request_prefill_tokens_per_second'].push({labels: {process: 'other'}, value: 999});
  assert.equal(core.requestRates(data, 10000).prefill, null);
  assert.equal(core.requestRates(data, 10000).decode, 25);
  data['vllm:sm75_request_timing_updated_seconds'].push({labels: {process: 'other'}, value: 10});
  assert.equal(core.requestRates(data, 10000).decode, null);
});

async function dashboard({exact = true} = {}) {
  const listeners = new Map(), elements = new Map(), paints = [], requests = [];
  const css = new Map(Object.entries({'--blue': '#65b5ff', '--mint': '#58e0bc', '--amber': '#ffcb76', '--line': '#243142', '--muted': '#92a5ba'}));
  const addListener = (name, callback) => {
    const list = listeners.get(name) || [];
    list.push(callback); listeners.set(name, list);
  };
  function element(id = '') {
    const context = {
      setTransform() {}, beginPath() {}, moveTo() {}, lineTo() {}, arc() {}, fill() {}, fillRect() {}, save() {}, translate() {}, rotate() {}, restore() {},
      measureText: text => ({width: String(text).length * 7}),
      fillText(text) {paints.push({id, kind: 'text', text, color: this.fillStyle});},
      stroke() {paints.push({id, kind: 'stroke', color: this.strokeStyle});},
    };
    return {id, textContent: '', value: id === 'window' ? '300' : id === 'barMode' ? 'percent' : '', style: {},
      addEventListener() {}, append() {}, replaceChildren() {}, focus() {},
      getBoundingClientRect: () => ({width: 800, height: 230}), getContext: () => context,
      insertRow: () => ({append() {}, insertCell: () => element()})};
  }
  const root = {};
  const scope = {
    document: {documentElement: root, getElementById: id => {if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id);}, createElement: element},
    location: {origin: 'http://workbench.local'}, devicePixelRatio: 1,
    getComputedStyle: () => ({getPropertyValue: name => css.get(name) || ''}),
    addEventListener: addListener, setTimeout: () => 1, clearTimeout() {},
    AbortSignal: {timeout: () => ({})},
    fetch: async url => {
      requests.push(url);
      return {ok: true, json: async () => ({spec_configured: false}), text: async () => [
        'vllm:num_requests_running 1', 'vllm:num_requests_waiting 0',
        'vllm:prompt_tokens_total 90000', 'vllm:generation_tokens_total 1200',
        ...(exact ? [`vllm:sm75_request_timing_updated_seconds ${Date.now() / 1000}`, 'vllm:sm75_request_timing_active 1',
          'vllm:sm75_request_prefill_tokens_per_second 240', 'vllm:sm75_request_decode_tokens_per_second 25'] : []),
      ].join('\n')};
    },
  };
  scope.window = scope;
  vm.createContext(scope);
  vm.runInContext(source, scope);
  await new Promise(setImmediate);
  await new Promise(setImmediate);
  const message = (data, origin = scope.location.origin) => {
    for (const listener of listeners.get('message') || []) listener({origin, data});
  };
  return {elements, paints, requests, css, message};
}

test('paused monitor repaints its actual canvas colors on a theme message without fetching or unpausing', async () => {
  const page = await dashboard();
  assert.equal(page.elements.get('prefillNow').textContent, '240');
  assert.equal(page.elements.get('decodeNow').textContent, '25');
  page.elements.get('pause').onclick();
  const before = page.paints.length, fetched = page.requests.length;
  page.css.set('--line', '#d6e0ed'); page.css.set('--muted', '#53677e');
  page.css.set('--blue', '#176bc2'); page.css.set('--mint', '#087f69');
  assert.doesNotThrow(() => page.message({type: 'sm75-theme', mode: 'light'}));
  const repainted = page.paints.slice(before);
  assert(repainted.some(paint => paint.kind === 'stroke' && paint.color === '#d6e0ed'));
  assert(repainted.some(paint => paint.kind === 'text' && paint.color === '#53677e'));
  assert.equal(page.elements.get('status').textContent, '已暂停');
  assert.equal(page.elements.get('prefillNow').textContent, '—');
  assert.equal(page.elements.get('decodeNow').textContent, '—');
  assert.equal(page.requests.length, fetched);
  const count = page.paints.length;
  page.message({type: 'sm75-theme', mode: 'dark'}, 'https://unrelated.invalid');
  page.message({type: 'unrelated', mode: 'dark'});
  page.message({type: 'sm75-theme', mode: 'invalid'});
  assert.equal(page.paints.length, count);
});

test('head cards stay unavailable when only engine counters exist', async () => {
  const page = await dashboard({exact: false});
  assert.equal(page.elements.get('prefillNow').textContent, '—');
  assert.equal(page.elements.get('decodeNow').textContent, '—');
});
