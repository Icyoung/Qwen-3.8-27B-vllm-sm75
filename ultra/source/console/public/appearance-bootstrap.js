// Run as a classic script before stylesheets. Only confirmed local preferences
// may override CSS defaults while the native preference store is still loading.
(() => {
  if (globalThis.__SM75_APPEARANCE_SNAPSHOT__) return;
  const read = key => {
    try { return globalThis.localStorage?.getItem(key); }
    catch { return null; }
  };
  const savedSize = Number(read('sm75-font-size'));
  const savedMode = read('sm75-color-mode');
  const fontSize = Number.isInteger(savedSize) && savedSize >= 12 && savedSize <= 17 ? savedSize : null;
  const colorMode = savedMode === 'light' || savedMode === 'dark' ? savedMode : null;
  globalThis.__SM75_APPEARANCE_SNAPSHOT__ = Object.freeze({fontSize, colorMode});
  const root = globalThis.document?.documentElement;
  if (!root) return;
  if (fontSize !== null) root.style.setProperty('--ui-fs', String(fontSize / 14));
  if (colorMode !== null) {
    root.dataset.sm75ColorMode = colorMode;
    root.style.setProperty('color-scheme', colorMode);
  }
})();
