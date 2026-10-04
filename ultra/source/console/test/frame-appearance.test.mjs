import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {injectFrameAppearance} from '../original-tests.mjs';

const bootstrap = fs.readFileSync(new URL('../public/frame-appearance.js', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');
const source = app.slice(app.indexOf('function installWorkbenchTheme() {'), app.indexOf('if (globalThis.document?.body) installWorkbenchTheme();'));

function declaration(values = {}) {
  const fields = new Map(Object.entries(values));
  return {getPropertyValue: key => fields.get(key) || '', getPropertyPriority: () => '', setProperty: (key, value) => fields.set(key, value)};
}
function node(tagName = 'DIV') {
  const attributes = new Map();
  return {tagName, nodeType: 1, dataset: {}, style: declaration(), children: [], textContent: '',
    setAttribute: (key, value) => attributes.set(key, value), hasAttribute: key => attributes.has(key),
    removeAttribute: key => attributes.delete(key), append(...nodes) {this.children.push(...nodes);},
    matches(selector) {return this.tagName === 'STYLE' && (selector.includes('data-sm75-theme') && this.dataset.sm75Theme || selector.includes('data-sm75-font-scale') && this.dataset.sm75FontScale);},
    addEventListener() {}, attributes};
}
function document({body = true} = {}) {
  const head = node('HEAD'), root = node('HTML'), listeners = new Map(), frames = [], inline = [];
  const doc = {head, documentElement: root, body: body ? node('BODY') : null, styleSheets: [], readyState: 'loading', frames, inline,
    createElement: tag => node(tag.toUpperCase()), getElementById: id => id === 'consoleSidebar' ? {} : null,
    querySelector: selector => head.children.find(child => child.matches(selector)) || null,
    querySelectorAll: selector => selector === '*' ? frames : selector === '[style]' ? inline : [],
    addEventListener(name, callback) {const items = listeners.get(name) || []; items.push(callback); listeners.set(name, items);},
    dispatchEvent(event) {for (const callback of listeners.get(event.type) || []) callback(event);},
  };
  return doc;
}
function workbench({storageBlocked = false} = {}) {
  const observers = [], timers = [], root = document();
  root.documentElement.style.setProperty('--ui-fs', String(17 / 14));
  root.documentElement.style.setProperty('--sm75-ui-font-family', '"Microsoft YaHei UI", sans-serif');
  class Observer {constructor(callback) {this.callback = callback;} observe(target) {this.target = target; observers.push(this);}}
  const computed = element => ({fontSize: '16px', getPropertyValue: key => element.style.getPropertyValue(key)});
  const storage = {getItem: () => {if (storageBlocked) throw Error('Storage blocked'); return 'dark';}, setItem() {if (storageBlocked) throw Error('Storage blocked');}};
  const scope = {document: root, localStorage: storage, MutationObserver: Observer, getComputedStyle: computed,
    location: {origin: 'http://workbench.local'}, __SM75_APPEARANCE_SNAPSHOT__: {colorMode: 'dark', fontSize: 17},
    CustomEvent: class {constructor(type, {detail} = {}) {this.type = type; this.detail = detail;}},
    addEventListener() {}, setInterval() {}, setTimeout(callback) {timers.push(callback);},
  };
  scope.window = scope;
  vm.createContext(scope);
  root.defaultView = vm.runInContext('globalThis', scope);
  vm.runInContext(source + '\ninstallWorkbenchTheme();', scope);
  function frame(id = 'benchFrame-speedtest') {
    const child = document({body: false}), messages = [], events = [];
    const win = {localStorage: storage, MutationObserver: Observer, getComputedStyle: computed,
      setTimeout: callback => timers.push(callback), Event: class {constructor(type) {this.type = type;}},
      dispatchEvent: event => events.push(event.type), postMessage: data => messages.push(data)};
    child.defaultView = win;
    const frame = {tagName: 'IFRAME', id, ownerDocument: root, contentDocument: child, contentWindow: win, addEventListener() {}};
    root.frames.push(frame);
    return {frame, child, win, messages, events};
  }
  return {scope, root, frame, observers, timers, initialize: frame => scope.__sm75InitializeFrameAppearance(frame)};
}

test('response bootstrap precedes upstream CSS and blocking remote chart scripts and remains idempotent', () => {
  const original = '<!doctype html><html><head><meta charset="utf-8"><script src="https://cdn.invalid/chart.js"></script><style>body{background:white}</style></head><body></body></html>';
  const prepared = injectFrameAppearance(original);
  assert(prepared.indexOf('data-sm75-frame-appearance') < prepared.indexOf('https://cdn.invalid'));
  assert(prepared.indexOf('data-sm75-frame-appearance') < prepared.indexOf('<style>'));
  assert.equal(injectFrameAppearance(prepared), prepared);
  assert(prepared.endsWith('</body></html>'));
});

test('child head calls the parent immediately and once again at DOM readiness without waiting for iframe load', () => {
  const calls = [], listeners = new Map(), frame = {};
  const context = {window: {frameElement: frame, parent: {__sm75InitializeFrameAppearance: value => calls.push(value)}},
    document: {addEventListener: (type, callback) => listeners.set(type, callback)}};
  vm.runInNewContext(bootstrap, context);
  assert.deepEqual(calls, [frame]);
  assert.equal(listeners.has('load'), false);
  listeners.get('DOMContentLoaded')();
  assert.deepEqual(calls, [frame, frame]);
});

test('parent installs the current theme and saved font while the child still has no body', () => {
  const parent = workbench(), child = parent.frame();
  parent.initialize(child.frame);
  assert.equal(child.child.body, null);
  assert(child.child.documentElement.hasAttribute('data-sm75-benchmark'));
  const theme = child.child.querySelector('style[data-sm75-theme]').textContent;
  assert(theme.includes('html{background:var(--bg)!important'));
  assert(theme.includes('html[data-sm75-benchmark] body'));
  assert.equal(theme.includes('@scope'), false);
  const fonts = child.child.querySelector('style[data-sm75-font-scale]').textContent;
  assert(fonts.includes('--ui-fs:' + 17 / 14));
  assert.equal(parent.timers.length, 0);
});

test('styles and inline font sizes added during parsing scale before the debounce and never compound', () => {
  const parent = workbench(), child = parent.frame();
  parent.initialize(child.frame);
  const rule = declaration({'font-size': '14px'}), inline = node();
  inline.style.setProperty('font-size', '12px'); child.child.inline.push(inline);
  child.child.styleSheets.push({ownerNode: node('STYLE'), cssRules: [{style: rule}]});
  const observer = parent.observers.find(item => item.target === child.child.documentElement);
  const mutations = [{type: 'childList', target: node('STYLE'), addedNodes: [node()]}];
  observer.callback(mutations);
  assert.equal(rule.getPropertyValue('font-size'), 'calc(14px * var(--ui-fs, 1))');
  assert.equal(inline.style.getPropertyValue('font-size'), 'calc(12px * var(--ui-fs, 1))');
  observer.callback(mutations);
  assert.equal(rule.getPropertyValue('font-size'), 'calc(14px * var(--ui-fs, 1))');
  assert.equal(parent.timers.length, 0);
});

test('denied storage and an unavailable chart cannot prevent initial appearance or later theme changes', () => {
  const parent = workbench({storageBlocked: true}), child = parent.frame();
  child.win.Chart = {defaults: {}, instances: {failed: {options: {}, update() {throw Error('Chart not ready');}}}};
  assert.doesNotThrow(() => parent.initialize(child.frame));
  assert(child.child.querySelector('style[data-sm75-theme]').textContent.includes('--bg:#151517'));
  assert(child.child.querySelector('style[data-sm75-font-scale]'));
  assert.doesNotThrow(() => parent.root.dispatchEvent(new parent.scope.CustomEvent('sm75-set-theme', {detail: 'light'})));
  assert(child.child.querySelector('style[data-sm75-theme]').textContent.includes('--bg:#f4f6f9'));
  assert.equal(child.messages.at(-1).mode, 'light');
});
test('frame text and chart labels receive the host sans family without replacing code fonts', () => {
  const parent = workbench(), child = parent.frame('benchFrame-sql');
  child.win.Chart = {defaults: {font: {family: 'Arial', size: 12}}, instances: {}};
  parent.initialize(child.frame);
  const theme = child.child.querySelector('style[data-sm75-theme]').textContent;
  assert(theme.includes('--sm75-ui-font-family:"Microsoft YaHei UI", sans-serif!important'));
  assert(theme.includes('--font-sans:"Microsoft YaHei UI", sans-serif!important'));
  assert(theme.includes('--font-serif:"Microsoft YaHei UI", sans-serif!important'));
  assert(theme.includes('body{font-family:var(--sm75-ui-font-family)!important}'));
  assert.equal(theme.includes('--font-mono:'), false);
  assert.equal(child.win.Chart.defaults.font.family, '"Microsoft YaHei UI", sans-serif');
});

test('a chart that failed while initializing is retried even when theme and font remain the same', () => {
  const parent = workbench(), child = parent.frame();
  let ready = false, calls = 0;
  child.win.Chart = {defaults: {}, instances: {chart: {options: {}, update() {
    calls += 1;
    if (!ready) throw Error('Chart not ready');
  }}}};
  parent.initialize(child.frame);
  assert.equal(calls, 1);
  ready = true;
  parent.initialize(child.frame);
  assert.equal(calls, 2);
  parent.initialize(child.frame);
  assert.equal(calls, 2);
});
