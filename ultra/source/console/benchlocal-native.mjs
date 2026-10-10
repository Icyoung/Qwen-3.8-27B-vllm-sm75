import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {BenchLocalProxy} from './benchlocal-proxy.mjs';
import {REPORT_KIND,PROTOCOL,UUID,TERMINAL,normalizeOptions,portableReport,validateImportedReport,redactBenchLocal,digest,reportCsv,restorePackScores} from './benchlocal-contract.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const defaultVerifiers={'bugfind-15':'http://sm75-bench-bugfind:4200','cli-40':'http://sm75-bench-cli:4200','hermesagent-20':'http://sm75-bench-hermes:4200'};
export class BenchLocal {
  constructor(root,{runtime=process.env.SM75_BENCHLOCAL_RUNTIME||path.join(here,'../benchlocal-native'),
    forkProcess=fork,worker=path.join(here,'benchlocal-native-worker.mjs'),verifiers,verifierKey,
    proxyLocal,proxyDocker,fetchImpl=fetch}={}) {
    this.root=path.join(root,'benchlocal');this.runtime=path.resolve(runtime);this.worker=worker;this.forkProcess=forkProcess;
    this.verifiers=verifiers||JSON.parse(process.env.SM75_BENCH_VERIFIERS||JSON.stringify(defaultVerifiers));
    this.verifierKey=verifierKey??process.env.SM75_BENCH_VERIFIER_KEY??'';
    this.proxyLocal=proxyLocal||'http://127.0.0.1:'+(process.env.SM75_CONSOLE_PORT||1615)+'/benchlocal-inference/v1';
    this.proxyDocker=proxyDocker||process.env.SM75_BENCH_PROXY_BASE||'http://sm75-bench-controller:1615/benchlocal-inference/v1';
    this.fetch=fetchImpl;this.proxy=new BenchLocalProxy({fetchImpl});this.active=null;this.records=new Map();this.healthCache=null;
    fs.mkdirSync(path.join(this.root,'jobs'),{recursive:true,mode:0o700});this.own(this.root);this.own(path.join(this.root,'jobs'));
    for(const id of fs.readdirSync(path.join(this.root,'jobs'))) {
      if(!UUID.test(id))continue;
      try {
        let row=JSON.parse(fs.readFileSync(this.file(id,'meta.json'),'utf8'));
        if(row.kind!==REPORT_KIND)row=this.legacy(row,{fromDisk:true});
        restorePackScores(row);
        if(!TERMINAL.has(row.state)){row.state='interrupted';row.error='服务重启，评测中断';row.completed=Date.now();this.persist(row);}
        this.records.set(id,row);
      } catch {}
    }
  }
  own(file){if(process.getuid?.()===0)fs.chownSync(file,1000,1000);}
  file(id,name){if(!UUID.test(id))throw Error('评测记录不存在');return path.join(this.root,'jobs',id,name);}
  persist(row){const file=this.file(row.id,'meta.json');fs.writeFileSync(file+'.tmp',JSON.stringify(row),{mode:0o600});this.own(file+'.tmp');fs.renameSync(file+'.tmp',file);}
  catalog(){return JSON.parse(fs.readFileSync(path.join(this.runtime,'catalog.json'),'utf8'));}
  get busy(){return !!this.active;}
  list(){return [...this.records.values()].sort((a,b)=>b.created-a.created).slice(0,100).map(row=>({...row,events:undefined,suites:row.suites.map(({results,scenarios,...suite})=>({...suite,done:results?.length||0}))}));}
  detail(id){const row=this.records.get(id);if(!row)throw Error('评测记录不存在');return row;}
  async readiness(force=false){
    if(!force&&this.healthCache&&Date.now()-this.healthCache.at<15000)return this.healthCache.states;
    const states={};
    await Promise.all(Object.entries(this.verifiers).map(async([id,url])=>{
      if(!this.verifierKey){states[id]={ready:false,reason:'受管验证环境尚未配置'};return;}
      try{const response=await this.fetch(url+'/health',{signal:AbortSignal.timeout(2000)});const value=await response.json();states[id]={ready:response.ok&&value.ok,reason:response.ok?'':'验证环境尚未就绪'};}
      catch{states[id]={ready:false,reason:'验证环境暂时不可用'};}
    }));
    this.healthCache={at:Date.now(),states};return states;
  }
  legacy(old,{fromDisk=false}={}){
    const pack=this.catalog().packs.find(x=>x.id===old.packId);if(!pack)throw Error('未知旧题包');
    let summary=old.summary;if(fromDisk)try{summary ||= JSON.parse(fs.readFileSync(this.file(old.id,'result.json'),'utf8'));}catch{}
    return {kind:REPORT_KIND,schemaVersion:2,id:old.id,source:'legacy',protocol:'legacy-unknown',fixtureId:null,
      created:old.created||Date.now(),completed:old.completed,state:TERMINAL.has(old.state)?old.state:'interrupted',
      model:{id:old.model,name:old.model||old.profileName||'旧记录',profileName:old.profileName},snapshot:old.snapshot||{},
      options:{presetName:'旧版参数',generation:old.generation||{},thinking:'unknown'},selection:[pack.id],total:pack.scenarioCount,
      done:old.results?.length||0,error:old.error,legacyTiming:true,suites:[{id:pack.id,name:pack.name,label:pack.label,
        version:old.packVersion||'unknown',sourceDigest:'unknown',scorerVersion:'unknown',state:old.state,total:pack.scenarioCount,
        score:Object.values(old.scores||summary?.scores||{})[0]?.totalScore??null,
        results:(old.results||[]).map(r=>({...r,durationMs:r.timings?.durationMs??null}))}]};
  }
  export(id,{detailed=false,format='json'}={}) {const report=portableReport(this.detail(id),{detailed});return format==='csv'?reportCsv(report):report;}
  import(value){
    if(value?.packId&&!value.kind)value=this.legacy({...value,id:UUID.test(value.id||'')?value.id:crypto.randomUUID()});
    const report=restorePackScores(validateImportedReport(value,this.catalog()));
    const row={...report,source:'imported',sourceId:report.id,id:crypto.randomUUID(),importedAt:Date.now()};
    delete row.integrity;
    if(!TERMINAL.has(row.state))row.state='interrupted';
    fs.mkdirSync(this.file(row.id,''),{mode:0o700});this.own(this.file(row.id,''));this.records.set(row.id,row);this.persist(row);return row;
  }
  async start(body,context){
    if(this.busy)throw Error('已有BenchLocal评测正在运行');
    if(body.packId!==undefined)throw Error('评测界面已更新，请刷新页面后重试');
    const catalog=this.catalog();let selection=body.packIds??catalog.packs.filter(x=>x.defaultSelected).map(x=>x.id);
    let optionsInput=body.options||{},fixtureId=crypto.randomUUID(),replayOf;
    if(body.replayId){
      const previous=this.detail(body.replayId);
      if(previous.protocol!==PROTOCOL||!UUID.test(previous.fixtureId||''))throw Error('旧版或未知协议报告不能直接复用测试方案');
      for(const suite of previous.suites)if(!catalog.packs.some(x=>x.id===suite.id&&x.sourceDigest===suite.sourceDigest))throw Error('报告题库版本与本机不一致');
      selection=previous.selection;optionsInput={...previous.options.generation,thinking:previous.options.thinking,effort:previous.options.effort,presetName:previous.options.presetName};
      fixtureId=previous.fixtureId;replayOf=previous.id;
    }
    if(!Array.isArray(selection)||!selection.length||selection.length>9||new Set(selection).size!==selection.length||selection.some(id=>!catalog.packs.some(p=>p.id===id)))throw Error('请选择1–9个有效测试类目');
    selection=catalog.packs.filter(p=>selection.includes(p.id)).map(p=>p.id);
    const options=normalizeOptions(optionsInput,context.capabilities);
    const contextLimit=Number(context.snapshot?.context);
    if(contextLimit>0&&(options.generation.max_tokens>=contextLimit||options.generation.thinking_token_budget>=contextLimit))throw Error('生成上限和思考预算需小于当前模型上下文长度');
    let scenarioSelection;
    if(body.scenarioSelection){
      if(!body.scenarioSelection||typeof body.scenarioSelection!=='object'||Array.isArray(body.scenarioSelection))throw Error('复测题目格式无效');
      scenarioSelection={};
      for(const [id,ids] of Object.entries(body.scenarioSelection)){
        const pack=catalog.packs.find(p=>p.id===id);
        if(!selection.includes(id)||!Array.isArray(ids)||!ids.length||new Set(ids).size!==ids.length||ids.some(s=>!pack.scenarios.some(x=>x.id===s)))throw Error('复测题目不在题库中');
        scenarioSelection[id]=ids;
      }
    }
    const active={id:crypto.randomUUID(),context,cancelled:false,audit:new Map(),tokens:[],child:null};this.active=active;
    try {
      const readiness=await this.readiness(true);
      for(const id of selection)if(this.verifiers[id]&&!readiness[id]?.ready)throw Error(catalog.packs.find(x=>x.id===id).label+'：'+(readiness[id]?.reason||'验证环境不可用'));
      if(active.cancelled)throw Error('评测已取消');
      const row={kind:REPORT_KIND,schemaVersion:2,id:active.id,source:'native',protocol:PROTOCOL,fixtureId,replayOf,
        model:{id:context.model,name:context.modelName||context.model,profileName:context.profileName,identity:context.identity},
        snapshot:redactBenchLocal(context.snapshot||{}),capabilities:context.capabilities,options,selection,scenarioSelection,partialTest:!!scenarioSelection,
        created:Date.now(),state:'running',done:0,events:[],
        total:selection.reduce((n,id)=>n+(scenarioSelection?.[id]?.length||catalog.packs.find(p=>p.id===id).scenarioCount),0),
        suites:selection.map(id=>{const p=catalog.packs.find(x=>x.id===id);return {id,name:p.name,label:p.label,version:p.version,sourceDigest:p.sourceDigest,scorerVersion:p.scorerVersion,
          total:p.scenarioCount,scenarios:p.scenarios,state:'pending',score:null,results:[],defaults:p.samplingDefaults,defaultSource:'题包参数，未指定项沿用服务默认'};})};
      active.row=row;
      const dir=this.file(row.id,'');fs.mkdirSync(dir,{mode:0o700});this.own(dir);
      this.records.set(row.id,row);this.persist(row);
      const input={id:row.id,fixtureId,selection,scenarioSelection,model:row.model,options,verifiers:this.verifiers,proxyLocal:this.proxyLocal,proxyDocker:this.proxyDocker};
      const privateInput=this.file(row.id,'input.json');fs.writeFileSync(privateInput,JSON.stringify(input),{mode:0o600});this.own(privateInput);
      active.tempDir=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-bench-worker-'+row.id+'-'));this.own(active.tempDir);
      const inputFile=path.join(active.tempDir,'input.json');fs.writeFileSync(inputFile,JSON.stringify(input),{mode:0o600});this.own(inputFile);
      const childOptions={env:{PATH:process.env.PATH,HOME:active.tempDir,LANG:'C.UTF-8',NVIDIA_VISIBLE_DEVICES:'void',CUDA_VISIBLE_DEVICES:'',
        SM75_BENCH_VERIFIER_KEY:this.verifierKey,SM75_BENCHLOCAL_RUNTIME:this.runtime},stdio:['ignore','pipe','pipe','ipc'],execArgv:[]};
      if(process.getuid?.()===0){childOptions.uid=1000;childOptions.gid=1000;}
      active.child=this.forkProcess(this.worker,[inputFile],childOptions);
      active.timer=setTimeout(()=>void this.cancel(row.id,'测试达到6小时上限'),6*3600000);active.timer.unref();
      let logSize=0;
      const log=buffer=>{if(logSize>=65536)return;const text=redactBenchLocal(String(buffer),[context.engineKey,this.verifierKey,...active.tokens]).slice(0,65536-logSize);logSize+=text.length;fs.appendFileSync(this.file(row.id,'worker.log'),text,{mode:0o600});};
      active.child.stdout?.on('data',log);active.child.stderr?.on('data',log);
      active.child.on('message',message=>{
        if(message?.source!=='sm75-benchlocal-native'||this.active!==active)return;
        try {this.message(active,message,catalog);}catch(error){row.error=error.message;row.state='failed';this.persist(row);void this.cancel(row.id,error.message);}
      });
      active.child.once('error',error=>{row.error=redactBenchLocal(error.message,[context.engineKey,this.verifierKey]);row.state='failed';this.persist(row);});
      active.child.once('close',()=>{
        clearTimeout(active.timer);clearTimeout(active.force);this.proxy.revokeRun(row.id);
        if(!TERMINAL.has(row.state)){row.state=active.cancelled?'cancelled':'failed';row.error ||= '评测进程提前结束';}
        row.completed ||= Date.now();for(const suite of row.suites)if(['pending','running'].includes(suite.state))suite.state=active.cancelled?'cancelled':'interrupted';
        this.persist(row);if(this.active===active)this.active=null;
        this.cleanupWorker(active);
      });
      return row;
    }catch(error){if(active.row){active.row.state='failed';active.row.error=error.message;this.persist(active.row);}if(this.active===active)this.active=null;this.cleanupWorker(active);throw error;}
  }
  cleanupWorker(active){
    if(!active.tempDir)return;
    const target=path.resolve(active.tempDir),parent=path.resolve(os.tmpdir());
    if(path.dirname(target)!==parent||!path.basename(target).startsWith('sm75-bench-worker-'+active.id+'-'))throw Error('评测临时目录校验失败');
    fs.rmSync(target,{recursive:true,force:true});active.tempDir=null;
  }
  message(active,m,catalog){
    const row=active.row,suite=row.suites.find(x=>x.id===m.packId),pack=catalog.packs.find(x=>x.id===m.packId);
    if(m.kind==='lease'){
      if(!suite||!pack.scenarios.some(x=>x.id===m.scenarioId)||active.cancelled)throw Error('无效的题目授权');
      for(const token of active.tokens)this.proxy.revoke(token);
      const token=this.proxy.allocate({runId:row.id,pack,scenarioId:m.scenarioId,options:row.options,context:active.context,
        onAudit:audit=>{const key=pack.id+'/'+m.scenarioId;const list=active.audit.get(key)||[];if(list.length<200)list.push(audit);active.audit.set(key,list);}});
      active.tokens.push(token);
      active.child.send({source:'sm75-benchlocal-parent',kind:'lease',rpcId:m.rpcId,token});return;
    }
    m=redactBenchLocal(m,[active.context.engineKey,this.verifierKey,...active.tokens]);
    if(m.kind==='suite-started'&&suite)suite.state='running';
    else if(m.kind==='scenario-started'&&suite){row.current={suiteId:suite.id,scenarioId:m.scenarioId};suite.current=m.scenarioId;}
    else if(m.kind==='progress'){row.events.push({time:Date.now(),suiteId:m.packId,scenarioId:m.scenarioId,message:m.event?.message||''});row.events=row.events.slice(-60);}
    else if(m.kind==='scenario-result'&&suite){
      if(!pack.scenarios.some(x=>x.id===m.result.scenarioId))throw Error('题目结果不匹配');
      const requests=active.audit.get(suite.id+'/'+m.result.scenarioId)||[];
      const usages=requests.map(x=>x.usage).filter(Boolean);
      const sum=key=>usages.length===requests.length&&usages.length&&usages.every(u=>Number.isFinite(u[key]))?usages.reduce((n,u)=>n+u[key],0):null;
      let result={...m.result,requests,tokenUsage:{requests:requests.length,available:usages.length,prompt_tokens:sum('prompt_tokens'),completion_tokens:sum('completion_tokens'),total_tokens:sum('total_tokens')}};
      const requestFailed=x=>x.status>=400||!!x.response?.error;
      const terminalFailure=!requests.length||requests.every(requestFailed)||
        (requestFailed(requests.at(-1))&&/request timed out|provider returned error|aborted due to timeout/i.test(result.summary||''));
      if(terminalFailure&&!active.cancelled)result={...result,upstreamSummary:result.summary,scoringStatus:result.status,scoringScore:result.score,status:'error',summary:requests.length?'模型请求失败，请查看本题执行轨迹':'未收到本题模型请求，请检查执行环境与客户端参数'};
      const index=suite.results.findIndex(x=>x.scenarioId===result.scenarioId);if(index<0)suite.results.push(result);else suite.results[index]=result;
      delete suite.current;row.done=row.suites.reduce((n,s)=>n+s.results.length,0);
    }else if(m.kind==='suite-result'&&suite){
      const errors=suite.results.some(x=>['error','cancelled'].includes(x.status));
      suite.summary=m.summary;suite.score=m.complete&&Number.isFinite(m.summary?.totalScore)?m.summary.totalScore:null;
      suite.state=errors?'failed':m.complete?'completed':'partial';
      suite.protocolFingerprint=digest({protocol:PROTOCOL,source:suite.sourceDigest,options:row.options});
    }else if(m.kind==='finished'){
      row.state=active.cancelled||m.cancelled?'cancelled':row.suites.some(x=>x.state==='failed')?'failed':'completed';row.completed=Date.now();delete row.current;
    }else if(m.kind==='error'){row.state=active.cancelled?'cancelled':'failed';row.error=m.message;row.completed=Date.now();}
    this.persist(row);
  }
  async cancel(id,reason='测试已取消'){
    if(!this.active||this.active.id!==id)return this.detail(id);
    const active=this.active;active.cancelled=true;this.proxy.revokeRun(id);
    if(active.row){active.row.state='cancelling';active.row.error=reason;this.persist(active.row);}
    active.child?.kill('SIGTERM');
    const selection=active.row?.selection||[];
    await Promise.all(selection.filter(id=>this.verifiers[id]).map(async id=>{try{await this.fetch(this.verifiers[id]+'/cancel',{method:'POST',headers:{Authorization:'Bearer '+this.verifierKey},signal:AbortSignal.timeout(3000)});}catch{}}));
    if(active.child){active.force=setTimeout(()=>active.child.kill('SIGKILL'),10000);active.force.unref();}
    return active.row||{id,state:'cancelled'};
  }
  close(){if(this.active)void this.cancel(this.active.id,'服务停止，测试已取消');}
}
