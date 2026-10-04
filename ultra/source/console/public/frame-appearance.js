// Embedded documents call the already loaded parent synchronously, before
// upstream CSS, scripts or remote chart libraries can paint their defaults.
(() => {
  const initialize = () => {
    try {
      const frame = window.frameElement;
      if (frame && window.parent !== window)
        window.parent.__sm75InitializeFrameAppearance?.(frame);
    } catch {}
  };
  initialize();
  document.addEventListener('DOMContentLoaded', initialize, {once: true});
})();
