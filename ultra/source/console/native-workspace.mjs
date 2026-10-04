import fs from 'node:fs';
import path from 'node:path';
export function patchWorkspaceSearch(source) {
  const marker='/* sm75-workspace-search-state */';
  if (source.includes(marker)) return source;
  const replace=(anchor,value)=>{
    if(source.split(anchor).length!==2)throw Error('Harness workspace search anchor changed: '+anchor.slice(0,60));
    source=source.replace(anchor,value);
  };
  const query='const normalizedQuery = sanitizeSearchQuery(query).trim();';
  replace(query,query+`
${marker}
    (0, react.useEffect)(() => {
      if (normalizedQuery !== "") setSearchExpanded(true);
    }, [normalizedQuery]);
    (0, react.useEffect)(() => {
      const clear = () => {setQuery("");setSearchExpanded(false);};
      document.addEventListener("sm75-workspace-reset-search",clear);
      return () => document.removeEventListener("sm75-workspace-reset-search",clear);
    }, []);`);
  const list='(0, react_jsx_runtime.jsx)("div", {\n\t\t\t\t\t\tclassName: WorkspaceBrowser_module_css_default.listArea,';
  replace(list,`wide && (normalizedQuery !== "" || archivedFilter !== "default") && (0, react_jsx_runtime.jsxs)("div", {
    className:"sm75-workspace-filter",role:"status",children:[
      (0, react_jsx_runtime.jsx)("span", {children: normalizedQuery !== "" ? "搜索："+normalizedQuery : archivedFilter === "only" ? "仅显示归档会话" : "包含归档会话"}),
      (0, react_jsx_runtime.jsx)("button", {type:"button",children:"恢复列表",onClick:()=>{setQuery("");setSearchExpanded(false);actions.setArchivedFilter("default");}})
    ]
  }),
`+list);
  replace('children: t("search.noMatches")','children: currentRemote.status === "error" ? "会话搜索失败，请清除搜索后重试" : t("search.noMatches")');
  return source;
}
export function applyWorkspaceSearch(base) {
  const file=path.join(base,'@deepseek-ai/dsh-client-ui-workspace/lib/client.js');
  fs.writeFileSync(file,patchWorkspaceSearch(fs.readFileSync(file,'utf8')));
}
