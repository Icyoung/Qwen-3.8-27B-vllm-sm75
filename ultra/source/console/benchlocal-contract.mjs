import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

export const REPORT_KIND='sm75.benchlocal.report';
export const PROTOCOL='sm75-native-v1';
export const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
export const TERMINAL=new Set(['completed','cancelled','failed','interrupted']);
export const THINKING_BUDGETS={low:1024,medium:4096,high:16384};
export const DEFAULT_REQUEST_TIMEOUT_SECONDS=300;
export const digest=value=>crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
export const packSuiteScore=suite=>Number.isFinite(suite?.score)?suite.score:Number.isFinite(suite?.summary?.totalScore)?suite.summary.totalScore:null;
export function packResultScore(result){
  if(result?.status==='cancelled')return null;
  if(Number.isFinite(result?.score))return result.score;
  if(Number.isFinite(result?.scoringScore))return result.scoringScore;
  if(['pass','partial','fail'].includes(result?.scoringStatus)){
    const matches=[...String(result.rawLog||'').matchAll(/(?:^|\n)verdict=(pass|partial|fail)\nscore=(\d+(?:\.\d+)?)\nsummary=/g)];
    const last=matches.at(-1),score=last?Number(last[2]):NaN;
    if(last?.[1]===result.scoringStatus&&Number.isFinite(score)&&score>=0&&score<=100)return score;
  }
  return null;
}
export function restorePackScores(row){
  for(const suite of row.suites||[]){suite.score=packSuiteScore(suite);for(const result of suite.results||[])result.score=packResultScore(result);}
  return row;
}
export function redactBenchLocal(value,secrets=[]) {
  if(!Array.isArray(secrets))secrets=[secrets];
  if(typeof value==='string') {
    for(const key of secrets.filter(Boolean))value=value.replaceAll(key,'[redacted]');
    return value.replace(/Bearer\s+[A-Za-z0-9_./+\-=]{16,}/gi,'Bearer [redacted]')
      .replace(/https?:\/\/(?:127\.0\.0\.1|localhost|10\.\d+\.\d+\.\d+|sm75-bench-[a-z-]+)(?::\d+)?[^\s"'<>]*/g,'[local service]');
  }
  if(Array.isArray(value))return value.map(x=>redactBenchLocal(x,secrets));
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value)
    .filter(([key])=>!/(?:^|_)(?:api_?key|authorization|password|secret|access_token|auth_token|base_url|baseUrl|verifierUrl)$/i.test(key))
    .map(([key,v])=>[key,redactBenchLocal(v,secrets)]));
  return value;
}
const numeric={temperature:[0,2],top_p:[0.000001,1],top_k:[-1,1000000,true],min_p:[0,1],
  repetition_penalty:[0.000001,3],presence_penalty:[-2,2],frequency_penalty:[-2,2],
  max_tokens:[1,1048576,true],seed:[-2147483648,2147483647,true],
  request_timeout_seconds:[5,1800,true],thinking_token_budget:[1,1048575,true]};
export function normalizeOptions(value={},capabilities={}) {
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('测试参数格式错误');
  const generation={};
  for(const [key,[min,max,integer]] of Object.entries(numeric)) {
    const n=value[key];if(n===undefined||n===null||n==='')continue;
    if(typeof n!=='number'||!Number.isFinite(n)||n<min||n>max||(integer&&!Number.isInteger(n))||(key==='top_k'&&n===0))throw Error(`${key} 超出有效范围`);
    generation[key]=n;
  }
  const thinking=value.thinking??'default';
  if(!['default','on','off'].includes(thinking))throw Error('思考模式无效');
  const effort=value.effort??'default';
  if(!['default','low','medium','high','custom'].includes(effort))throw Error('思考深度无效');
  if(thinking!=='default'&&!capabilities.toggle)throw Error(capabilities.reason||'当前模型不支持已确认的思考开关');
  if(thinking==='on'&&effort!=='default') {
    if(capabilities.budget)generation.thinking_token_budget=effort==='custom'?generation.thinking_token_budget:THINKING_BUDGETS[effort];
    else if(capabilities.effort?.includes(effort))generation.reasoning_effort=effort;
    else throw Error('当前模型不支持所选思考深度');
    if(!generation.thinking_token_budget&&!generation.reasoning_effort)throw Error('请填写思考预算');
  } else delete generation.thinking_token_budget;
  if(generation.thinking_token_budget&&generation.max_tokens&&generation.max_tokens<=generation.thinking_token_budget)throw Error('最大生成 Token 需大于思考预算，为正文预留空间');
  const presetName=typeof value.presetName==='string'?value.presetName.trim().slice(0,100):'';
  const name=presetName||((Object.keys(generation).length||thinking!=='default')?'自定义':'题包默认');
  generation.request_timeout_seconds??=DEFAULT_REQUEST_TIMEOUT_SECONDS;
  return {presetName:name,thinking,effort,generation};
}
export function thinkingCapabilities(profile) {
  const args=profile?.args||[],get=key=>{const i=args.indexOf(key);return i>=0?args[i+1]:null;};
  const parser=get('--reasoning-parser');
  let template='';
  try {template=String(JSON.parse(fs.readFileSync(path.join(get('--tokenizer')||args[0],'tokenizer_config.json'),'utf8')).chat_template||'');}catch{}
  try {if(!template)template=fs.readFileSync(path.join(args[0],'chat_template.jinja'),'utf8');}catch{}
  const toggle=!!parser&&template.includes('enable_thinking');
  const budget=toggle&&['qwen3','deepseek_r1','deepseek_v3','nemotron_v3'].includes(parser);
  return {toggle,budget,budgets:THINKING_BUDGETS,effort:template.includes('reasoning_effort')?['low','medium','high']:[],
    parser:parser||null,reason:toggle?'':'当前模型模板未确认支持思考开关',verification:'template-and-pinned-engine'};
}
export function applyGeneration(payload,pack,options) {
  const body={...pack.samplingDefaults,...payload,...options.generation};
  delete body.request_timeout_seconds;
  if(options.thinking!=='default') {
    body.chat_template_kwargs={...body.chat_template_kwargs,enable_thinking:options.thinking==='on'};
    if(options.thinking==='off'){delete body.thinking_token_budget;delete body.reasoning_effort;}
  }
  if(body.max_tokens!==undefined)delete body.max_completion_tokens;
  const budget=body.thinking_token_budget,max=body.max_tokens??body.max_completion_tokens;
  if(budget&&max&&max<=budget)throw Error('题目实际输出上限小于思考预算，请设置更大的最大生成 Token');
  return body;
}
export function requestSettings(body) {
  return Object.fromEntries([...Object.keys(numeric),'reasoning_effort','chat_template_kwargs','max_completion_tokens']
    .filter(key=>Object.hasOwn(body,key)).map(key=>[key,body[key]]));
}
export function portableReport(row,{detailed=false}={}) {
  const report=restorePackScores(structuredClone(row));delete report.events;
  for(const suite of report.suites||[])for(const result of suite.results||[]) {
    if(!detailed){delete result.rawLog;delete result.output;delete result.requests;delete result.trace;}
  }
  const cleaned=redactBenchLocal(report);
  delete cleaned.integrity;
  cleaned.integrity={algorithm:'sha256',digest:digest(cleaned)};
  return cleaned;
}
export function validateImportedReport(value,catalog) {
  if(!value||value.kind!==REPORT_KIND||value.schemaVersion!==2||!Array.isArray(value.suites)||value.suites.length>9)throw Error('不支持的评测报告格式');
  if(!value.model||typeof value.model.name!=='string'||value.model.name.length>512)throw Error('报告缺少模型信息');
  if(!Number.isFinite(value.created)||value.created<0||!value.options||typeof value.options!=='object'||Array.isArray(value.options))throw Error('报告元数据无效');
  if(value.options.generation&&(!value.options.generation||typeof value.options.generation!=='object'||Array.isArray(value.options.generation)))throw Error('报告参数格式无效');
  for(const key of Object.keys(numeric)){const n=value.options.generation?.[key];if(n!=null&&(typeof n!=='number'||!Number.isFinite(n)))throw Error('报告参数值无效');}
  if(!(value.protocol==='legacy-unknown'&&value.fixtureId===null)&&(typeof value.fixtureId!=='string'||!UUID.test(value.fixtureId)))throw Error('报告测试变量标识无效');
  const ids=new Set();
  for(const suite of value.suites) {
    const pack=catalog.packs.find(x=>x.id===suite.id);
    if(!pack||ids.has(suite.id)||!Array.isArray(suite.results)||suite.results.length>pack.scenarioCount)throw Error('报告测试项目无效');
    ids.add(suite.id);
    if(suite.score!==null&&suite.score!==undefined&&(!Number.isFinite(suite.score)||suite.score<0||suite.score>100))throw Error('报告分数无效');
    if(suite.total!==pack.scenarioCount||typeof suite.version!=='string'||typeof suite.sourceDigest!=='string')throw Error('报告题库信息无效');
    if(suite.scenarios!==undefined){
      if(!Array.isArray(suite.scenarios)||suite.scenarios.length>pack.scenarioCount)throw Error('报告题目目录无效');
      const metaIds=new Set();
      for(const meta of suite.scenarios){if(!meta||typeof meta.id!=='string'||meta.id.length>128||metaIds.has(meta.id)||typeof meta.title!=='string')throw Error('报告题目描述无效');metaIds.add(meta.id);}
    }
    const scenarioIds=new Set();
    for(const result of suite.results) {
      if(typeof result.scenarioId!=='string'||result.scenarioId.length>128||scenarioIds.has(result.scenarioId))throw Error('报告题目ID重复或无效');
      if(suite.sourceDigest===pack.sourceDigest&&!pack.scenarios.some(x=>x.id===result.scenarioId))throw Error('报告题目与题库不符');
      scenarioIds.add(result.scenarioId);
      if(!['pass','partial','fail','error','cancelled'].includes(result.status))throw Error('报告题目状态无效');
      if(result.score!=null&&(!Number.isFinite(result.score)||result.score<0||result.score>100))throw Error('题目分数无效');
      if(result.durationMs!=null&&(!Number.isFinite(result.durationMs)||result.durationMs<0))throw Error('题目耗时无效');
      if(result.requests!==undefined&&(!Array.isArray(result.requests)||result.requests.length>1000))throw Error('题目请求轨迹无效');
    }
  }
  if(!Array.isArray(value.selection)||!value.selection.length||value.selection.length>9||new Set(value.selection).size!==value.selection.length||value.selection.some(id=>!ids.has(id)))throw Error('报告选择项目无效');
  if(value.integrity){const copy=structuredClone(value);delete copy.integrity;if(value.integrity.digest!==digest(copy))throw Error('报告完整性校验失败');}
  const encoded=JSON.stringify(value);
  if(encoded.length>24*1024*1024)throw Error('报告过大');
  function check(item,depth=0) {
    if(depth>30)throw Error('报告嵌套过深');
    if(typeof item==='string'&&item.length>2*1024*1024)throw Error('报告字段过大');
    if(item&&typeof item==='object')for(const [key,v]of Object.entries(item)){if(['__proto__','constructor','prototype'].includes(key))throw Error('报告字段不允许');check(v,depth+1);}
  }
  check(value);
  return redactBenchLocal(value);
}
export function reportCsv(report) {
  const quote=v=>'"'+String(v??'').replaceAll('"','""').replace(/^(?=[\s]*[=+@-])/,"'")+'"';
  const rows=[['model','preset','suite','suite_score','scenario','status','score','duration_ms','prompt_tokens','completion_tokens']];
  for(const suite of report.suites||[])for(const r of suite.results||[])rows.push([report.model.name,report.options?.presetName,suite.id,suite.score,r.scenarioId,r.status,r.score,r.durationMs,r.tokenUsage?.prompt_tokens,r.tokenUsage?.completion_tokens]);
  return '\uFEFF'+rows.map(row=>row.map(quote).join(',')).join('\r\n');
}
