import {applyWorkspaceSearch} from './native-workspace.mjs';
import fs from "node:fs";
import path from "node:path";

// Restore cached presentation before the first native snapshot, then expose
// the first accepted host settings read. Neither operation writes preferences.
export function patchNativeAppearanceRestore(source) {
  const marker = "/* sm75-appearance-restore */";
  if (source.includes(marker)) return source;
  const initialize = /(constructor\(ctx, host\) \{[\s\S]*?)this\.preference = DEFAULT_PREFERENCE;/;
  const adopt = /(const section = this\.host\.getSnapshot\(\)\.value;\s*if \(section === void 0\) return;)/;
  const unchanged = /if\s*\(this\.preference === section\.preference && this\.fontSize === fontSize\)\s*return;/;
  for (const pattern of [initialize, adopt, unchanged])
    if ([...source.matchAll(new RegExp(pattern.source, "g"))].length !== 1)
      throw Error("Harness appearance restore anchor changed");
  return source.replace(initialize, (_match, prefix) => prefix + marker + `
        this.sm75AppearanceRestored = false;
        const cached = globalThis.__SM75_APPEARANCE_SNAPSHOT__;
        this.preference = cached?.colorMode === "light" || cached?.colorMode === "dark"
          ? cached.colorMode : DEFAULT_PREFERENCE;
        if (Number.isInteger(cached?.fontSize) && cached.fontSize >= 12 && cached.fontSize <= 17)
          this.fontSize = cached.fontSize;`)
    .replace(adopt, "$1\n        const firstAppearance = this.sm75AppearanceRestored !== true;\n        this.sm75AppearanceRestored = true;")
    .replace(unchanged, "if (!firstAppearance && this.preference === section.preference && this.fontSize === fontSize) return;");
}

// Theme and font share one native ConfigForm write queue. A later theme write
// supersedes the mirror fold of an earlier font write, so settle the whole
// appearance queue before adopting either field from its accepted snapshot.
export function patchNativeAppearanceWrites(source) {
  const marker = "/* sm75-appearance-write-settlement */";
  if (source.includes(marker)) return source;
  const themeWrite = "if (isThemePreference(id)) this.host.set(THEME_PREFERENCE_FIELD, id);";
  const fontWrite = /\/\* sm75-font-write-settlement \*\/[\s\S]*?this\.sm75FontWrite = Promise\.resolve\(this\.host\.set\(FONT_SIZE_FIELD, px\)\)\.then\([\s\S]*?\n        \);/;
  const fontAdopt = "const fontSize = this.sm75FontPending ? this.fontSize : section.fontSize;";
  const preferenceCheck = "this.preference === section.preference && this.fontSize === fontSize";
  const preferenceAdopt = "this.preference = section.preference;";
  const adoptMethod = "adopt() {";
  for (const anchor of [themeWrite, fontAdopt, preferenceCheck, preferenceAdopt, adoptMethod])
    if (source.split(anchor).length !== 2) throw Error("Harness appearance write anchor changed");
  if ([...source.matchAll(new RegExp(fontWrite.source, "g"))].length !== 1)
    throw Error("Harness font write settlement anchor changed");
  source = source.replace(themeWrite, "if (isThemePreference(id)) this.sm75QueueAppearanceWrite(THEME_PREFERENCE_FIELD, id);")
    .replace(fontWrite, "/* sm75-font-write-settlement */\n        this.sm75FontWrite = this.sm75QueueAppearanceWrite(FONT_SIZE_FIELD, px);")
    .replace(fontAdopt, "const fontSize = this.sm75AppearancePending ? this.fontSize : section.fontSize;\n        const preference = this.sm75AppearancePending ? this.preference : section.preference;")
    .replace(preferenceCheck, "this.preference === preference && this.fontSize === fontSize")
    .replace(preferenceAdopt, "this.preference = preference;")
    .replace(adoptMethod, marker + `
      sm75QueueAppearanceWrite(field, value) {
        const generation = this.sm75AppearanceGeneration = (this.sm75AppearanceGeneration || 0) + 1;
        this.sm75AppearancePending = true;
        if (field === FONT_SIZE_FIELD) this.sm75FontPending = true;
        const finish = () => {
          if (generation !== this.sm75AppearanceGeneration) return;
          this.sm75AppearancePending = false;
          this.sm75FontPending = false;
          this.adopt();
        };
        this.sm75AppearanceWrite = Promise.resolve(this.host.set(field, value)).then(
          accepted => { finish(); return accepted; },
          error => { finish(); this.ctx.logger?.warn?.("Unable to save appearance settings", error); return false; }
        );
        return this.sm75AppearanceWrite;
      }
      adopt() {`);
  return source;
}

// Fixed-release adapter: retain each owner's service, store and child-slot
// declarations; replace only visual shell components at render time.
export function applyUnifiedShell(base) {
  applyWorkspaceSearch(base);
  for (const [pkg, name, renderer] of [
    ["dsh-client-ui-layout", "AppFrame", "frame"],
    ["dsh-client-ui-sidebar", "SidebarRoot", "sidebar"],
    ["dsh-client-ui-conversation", "ConversationHeader", "header"],
    ["dsh-client-ui-settings-general", "SettingsRoot", "settings"],
    ["dsh-client-ui-settings-general", "GeneralSection", "generalSettings"],
  ]) {
    const file = path.join(base, "@deepseek-ai", pkg, "lib/client.js");
    let source = fs.readFileSync(file, "utf8");
    const marker = "/* sm75-unified-shell:" + name + " */";
    if (source.includes(marker)) continue;
    const anchor = "function " + name + "(";
    if (source.split(anchor).length !== 2) throw Error("Harness unified shell anchor changed: " + name);
    source = source.replace(anchor, marker + "\nfunction " + name + "(props) {\n"
      + "  const shell = globalThis.__SM75_NATIVE_SHELL__;\n"
      + "  if (!shell) throw Error('SM75 unified shell is not ready');\n"
      + "  return shell." + renderer + "(props, Native" + name + ");\n}\nfunction Native" + name + "(");
    fs.writeFileSync(file, source);
  }
  // Read the owner's live store before issuing panel navigation: reselecting
  // the already-active conversation must not abort initial workspace restore.
  const layoutFile=path.join(base,"@deepseek-ai/dsh-client-ui-layout","lib/client.js");
  let layoutSource=fs.readFileSync(layoutFile,"utf8");
  if (!layoutSource.includes("/* sm75-active-panel-reader */")) {
    const anchor=/(const layout = new LayoutController\(instance\.actions,[^\n]+\);)/;
    if ([...layoutSource.matchAll(new RegExp(anchor.source,"g"))].length!==1)
      throw Error("Harness panel reader anchor changed");
    layoutSource=layoutSource.replace(anchor,"$1\n/* sm75-active-panel-reader */\nlayout.sm75ActivePanel = () => instance.getSnapshot().panelInfo.activePanelId;");
    fs.writeFileSync(layoutFile,layoutSource);
  }
  // AppFrame owns the native durable-session document title. Its visual
  // replacement must still mount that owner instead of leaving it unreachable.
  const titleMarker = "/* sm75-native-document-title */";
  if (!layoutSource.includes(titleMarker)) {
    const delegate = "return shell.frame(props, NativeAppFrame);";
    if (layoutSource.split(delegate).length !== 2 || layoutSource.split("function DocumentTitle(").length !== 2)
      throw Error("Harness native document title anchor changed");
    layoutSource = layoutSource.replace(delegate, titleMarker + "\n  return shell.frame({...props, sm75DocumentTitle: DocumentTitle}, NativeAppFrame);");
    fs.writeFileSync(layoutFile, layoutSource);
  }
  // This native control now scales the entire shared interface. Keep its
  // original settings store and actions, but describe the actual scope.
  const themeFile=path.join(base,"@deepseek-ai/dsh-client-ui-theme","lib/client.js");
  let theme=fs.readFileSync(themeFile,"utf8");
  for (const [pattern, value] of [
    [/("fontSize\.title"\s*:\s*)"(?:字号大小|全局字号)"/g,"全局字号"],
    [/("fontSize\.description"\s*:\s*)"(?:仅影响会话内容的字号|调整工作台与会话的字号|统一调整侧栏、表单、会话与使用统计的字号)"/g,"统一调整侧栏、表单、会话与使用统计的字号"],
    [/("fontSize\.title"\s*:\s*)"(?:Font size|Global font size)"/g,"Global font size"],
    [/("fontSize\.description"\s*:\s*)"(?:Only affects conversation content|Adjusts workbench and conversation text|Adjusts sidebar, forms, conversations and usage statistics)"/g,"Adjusts sidebar, forms, conversations and usage statistics"],
  ]) {
    if ([...theme.matchAll(pattern)].length!==1) throw Error("Harness global font copy anchor changed");
    theme=theme.replace(pattern, (_match, prefix)=>prefix+JSON.stringify(value));
  }
  // Older local bundles widened this UI-only range. Use the upstream 12..17
  // control and validation range; profile values are never rewritten here.
  theme=theme.replaceAll("disabled: fontSize >= 26","disabled: fontSize >= 17")
    .replaceAll("px > 26","px > 17")
    .replaceAll("outside 12..26","outside 12..17")
    .replaceAll("[FONT_SIZE_FIELD]: Schema.number().step(1).min(12).max(26)","[FONT_SIZE_FIELD]: Schema.number().step(1).min(12).max(17)")
    .replaceAll("parsed >= 12 && parsed <= 26","parsed >= 12 && parsed <= 17");
  const fontWriteMarker="/* sm75-font-write-settlement */";
  if (!theme.includes(fontWriteMarker)) {
    const write="this.host.set(FONT_SIZE_FIELD, px);";
    const adopt=/(const section = this\.host\.getSnapshot\(\)\.value;\s*if \(section === void 0\) return;)/;
    for (const needle of [write,"this.fontSize === section.fontSize","this.fontSize = section.fontSize;"])
      if (theme.split(needle).length!==2) throw Error("Harness font settlement anchor changed");
    if (!adopt.test(theme)) throw Error("Harness font adoption anchor changed");
    theme=theme.replace(write, fontWriteMarker+`
        const generation = this.sm75FontGeneration = (this.sm75FontGeneration || 0) + 1;
        this.sm75FontPending = true;
        const finish = () => {
          if (generation !== this.sm75FontGeneration) return;
          this.sm75FontPending = false;
          this.adopt();
        };
        this.sm75FontWrite = Promise.resolve(this.host.set(FONT_SIZE_FIELD, px)).then(
          accepted => { finish(); return accepted; },
          error => { finish(); this.ctx.logger?.warn?.("Unable to save global font size", error); return false; }
        );`)
      .replace(adopt,"$1\n        const fontSize = this.sm75FontPending ? this.fontSize : section.fontSize;")
      .replace("this.fontSize === section.fontSize","this.fontSize === fontSize")
      .replace("this.fontSize = section.fontSize;","this.fontSize = fontSize;");
  }
  fs.writeFileSync(themeFile,patchNativeAppearanceWrites(patchNativeAppearanceRestore(theme)));
  const file=path.join(base,"@deepseek-ai/dsh-client-ui-open-in-app","lib/client.js");
  let source=fs.readFileSync(file,"utf8");
  const route='"/open-in-app/';
  if (!source.includes('"/harness-ui/open-in-app/')) {
    if (!source.includes(route)) throw Error("Harness open-in-app route anchor changed");
    source=source.replaceAll(route,'"/harness-ui/open-in-app/');
    fs.writeFileSync(file,source);
  }

}
