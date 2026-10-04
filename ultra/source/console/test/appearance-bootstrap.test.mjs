import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../public/appearance-bootstrap.js', import.meta.url), 'utf8');

function browser(storage = {}, customize = () => {}) {
  const styles = new Map(), dataset = {}, writes = [];
  const context = {
    document: {documentElement: {dataset, style: {setProperty: (key, value) => styles.set(key, value)}}},
    localStorage: {getItem: key => storage[key] ?? null, setItem: (...args) => writes.push(args)},
  };
  customize(context, styles);
  vm.createContext(context);
  const run = () => vm.runInContext(source, context);
  run();
  return {context, styles, dataset, writes, run};
}

test('confirmed font and theme are applied synchronously, without waiting for the native store', () => {
  for (const size of [12, 14, 17]) {
    for (const mode of ['light', 'dark']) {
      const page = browser({'sm75-font-size': String(size), 'sm75-color-mode': mode});
      assert.equal(page.styles.get('--ui-fs'), String(size / 14));
      assert.equal(page.styles.get('color-scheme'), mode);
      assert.equal(page.dataset.sm75ColorMode, mode);
      assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.fontSize, size);
      assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.colorMode, mode);
      assert(Object.isFrozen(page.context.__SM75_APPEARANCE_SNAPSHOT__));
      assert.deepEqual(page.writes, []);
    }
  }
});

test('missing preferences leave the existing CSS or server-provided values alone', () => {
  const page = browser({}, (_context, styles) => styles.set('--ui-fs', '1.125'));
  assert.equal(page.styles.get('--ui-fs'), '1.125');
  assert.equal(page.styles.has('color-scheme'), false);
  assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.fontSize, null);
  assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.colorMode, null);
  assert.deepEqual(page.writes, []);
});

test('invalid snapshots never become a font scale or overwrite stored preferences', () => {
  for (const invalid of ['', 'null', 'NaN', '0', '11', '14.5', '18', 'Infinity']) {
    const page = browser({'sm75-font-size': invalid, 'sm75-color-mode': 'system'});
    assert.equal(page.styles.size, 0, invalid);
    assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.fontSize, null);
    assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.colorMode, null);
    assert.deepEqual(page.writes, []);
  }
});

test('blocked storage access leaves a usable page with no persisted defaults', () => {
  const page = browser({}, context => {
    Object.defineProperty(context, 'localStorage', {get() {throw new Error('Storage blocked');}});
  });
  assert.equal(page.styles.size, 0);
  assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.fontSize, null);
  assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.colorMode, null);
  assert.deepEqual(page.writes, []);
});

test('a failure reading one setting does not discard the other confirmed setting', () => {
  const page = browser({}, context => {
    context.localStorage.getItem = key => {
      if (key === 'sm75-font-size') throw new Error('Unreadable font preference');
      return 'light';
    };
  });
  assert.equal(page.styles.has('--ui-fs'), false);
  assert.equal(page.styles.get('color-scheme'), 'light');
  assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__.fontSize, null);
});

test('a second bootstrap cannot replay old defaults over an already updated appearance', () => {
  const page = browser({'sm75-font-size': '17', 'sm75-color-mode': 'dark'});
  const snapshot = page.context.__SM75_APPEARANCE_SNAPSHOT__;
  page.styles.set('--ui-fs', String(12 / 14));
  page.styles.set('color-scheme', 'light');
  page.dataset.sm75ColorMode = 'light';
  page.run();
  assert.equal(page.context.__SM75_APPEARANCE_SNAPSHOT__, snapshot);
  assert.equal(page.styles.get('--ui-fs'), String(12 / 14));
  assert.equal(page.styles.get('color-scheme'), 'light');
  assert.equal(page.dataset.sm75ColorMode, 'light');
  assert.deepEqual(page.writes, []);
});
