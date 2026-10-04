import fs from 'node:fs';
import path from 'node:path';
import {resolveManagedSelection} from './plugins/sm75-workbench/lib/managed-model.js';
const marker='/* sm75-managed-model-selection */';
function replaceOnce(source, anchor, replacement) {
  if(source.split(anchor).length!==2)throw Error('Harness model selection anchor changed: '+anchor.slice(0,70));
  return source.replace(anchor,replacement);
}
export function patchServerModelSelection(source) {
  if(source.includes(marker))return source;
  source=replaceOnce(source,'if (picked !== void 0) return picked;','if (picked !== void 0) return resolveManagedSelection(picked);');
  source=replaceOnce(source,'if (loggedHeader === void 0) return defaultModel.currentSelection();','if (loggedHeader === void 0) return resolveManagedSelection(defaultModel.currentSelection());');
  source=replaceOnce(source,'const logged = loggedHeader.config;\n\t\t\t\treturn {','const logged = loggedHeader.config;\n\t\t\t\treturn resolveManagedSelection({');
  source=replaceOnce(source,'reasoningEffort: logged.reasoningEffort }\n\t\t\t\t};','reasoningEffort: logged.reasoningEffort }\n\t\t\t\t});');
  source=replaceOnce(source,'if (picked?.provider !== provider || picked.model !== model || picked.reasoningEffort !== reasoningEffort) return false;',
    'const effective = resolveManagedSelection(picked);\n                if (effective?.provider !== provider || effective?.model !== model || effective?.reasoningEffort !== reasoningEffort) return false;');
  return marker+'\nimport {resolveManagedSelection} from "sm75-workbench/managed-model";\n'+source;
}
export function patchClientModelSelection(source) {
  if(source.includes(marker))return source;
  return replaceOnce(source,'const current = projected.next ?? catalog.value.default;',marker+`
                const local = catalog.value.groups.find(group=>group.id === 'sm75-local')?.models[0];
                const current = (${resolveManagedSelection.toString()})(projected.next ?? catalog.value.default,
                  local ? {provider:'sm75-local',model:local.id} : null);`);
}
export function applyManagedModelSelection(base) {
  for(const [pkg,patch,file] of [
    ['dsh-api-session-controller',patchServerModelSelection,'index.js'],
    ['dsh-client-ui-model-selection',patchClientModelSelection,'client.js'],
  ]) {
    const target=path.join(base,'@deepseek-ai',pkg,'lib',file);
    fs.writeFileSync(target,patch(fs.readFileSync(target,'utf8')));
  }
}
