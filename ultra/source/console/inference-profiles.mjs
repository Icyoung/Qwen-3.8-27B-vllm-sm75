import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const directory=fileURLToPath(new URL('../profiles/',import.meta.url));
export function inferenceProfiles() {
  if(!fs.existsSync(directory))return [];
  return fs.readdirSync(directory).filter(n=>n.endsWith('.json')).map(n=>JSON.parse(fs.readFileSync(path.join(directory,n),'utf8')));
}
export function inferenceProfile(id, model, settings={cacheRoot:'/data/cache'}) {
  const original=inferenceProfiles().find(p=>p.id===id);
  if(!original)throw Error('模型模板不存在');
  const variables={'${MODEL}':model,'${DRAFT_MODEL}':original.defaultDraftModel || model+'/runtime/mtp-int4-g32'};
  const expand=v=>typeof v==='string'?Object.entries(variables).reduce((s,[a,b])=>s.replaceAll(a,b),v)
    :Array.isArray(v)?v.map(expand):v&&typeof v==='object'?Object.fromEntries(Object.entries(v).map(([k,x])=>[k,expand(x)])):v;
  const p=expand(original);
  p.args=p.args.map(a=>typeof a==='string'?a:JSON.stringify(a));
  return {...p,backend:'native',port:8000,cacheRoot:settings.cacheRoot,binds:[]};
}
