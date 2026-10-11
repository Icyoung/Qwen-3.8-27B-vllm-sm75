const states={pending:'待测试',running:'正在测试',cancelling:'正在停止',completed:'已完成',cancelled:'已取消',failed:'存在执行错误',interrupted:'已中断',partial:'部分复测'};
const results={pass:'通过',partial:'部分得分',fail:'未通过',error:'执行异常',cancelled:'已取消',running:'正在测试',missing:'未测'};
const defaultRequestTimeout=300;
const legacyDefaultPacks=['toolcall-15','dataextract-15','instructfollow-15','promptauthority-15','reasonmath-15','structoutput-15'];
export function restoreBenchLocalSelection(packs,saved){
  const defaults=packs.filter(p=>p.defaultSelected).map(p=>p.id);
  if(!Array.isArray(saved?.selected)||saved.selectionPreset==='default')return defaults;
  const ids=packs.filter(p=>saved.selected.includes(p.id)).map(p=>p.id);
  const legacy=saved.selectionPreset===undefined&&ids.length===legacyDefaultPacks.length&&legacyDefaultPacks.every(id=>ids.includes(id));
  return legacy?defaults:ids;
}
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
function outcomeLabel(result,state=result?.status){
  if(state==='error'){
    const cause=[result?.upstreamSummary,result?.summary,...(result?.requests||[]).map(r=>JSON.stringify(r.response?.error||''))].join(' ');
    if(result?.errorKind==='timeout'||/timed?\s*out|timeout|请求超时/i.test(cause))return '请求超时';
    if(result?.requests?.some(r=>r.status>=400||r.response?.error))return '模型请求异常';
  }
  return results[state]||results.missing;
}
const budgets={low:1024,medium:4096,high:16384};
const number=value=>Number.isFinite(value)?value.toFixed(1):'—';
const duration=value=>!Number.isFinite(value)?'—':value<60000?(value/1000).toFixed(1)+'s':(value/60000).toFixed(1)+'m';
function node(tag,text,attributes={}){const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);for(const[k,v]of Object.entries(attributes))n.setAttribute(k,String(v));return n;}
function canonical(value){if(Array.isArray(value))return value.map(canonical);if(value&&typeof value==='object')return Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])]));return value;}
function parameters(report){const o=report?.options;if(!o)return '题包默认';return [o.presetName,o.thinking==='on'?'思考开启':o.thinking==='off'?'思考关闭':'默认思考',o.generation?.temperature!==undefined?'温度 '+o.generation.temperature:null,o.generation?.thinking_token_budget?'预算 '+o.generation.thinking_token_budget:null].filter(Boolean).join(' · ');}
export function mountBenchLocal({api,run}) {
  const $=id=>document.getElementById(id),root=$('benchLocalPanel');if(!root)return {refresh:async()=>{}};
  let data=null,selected=new Set(),selectionReady=false,record=null,selectedId='',detailCase=null;
  let loading=false,loadError=false,revision='',sequence=0,presetSignature='',catalogSignature='',savedView=null;const reports=new Map();
  try{savedView=JSON.parse(localStorage.getItem('sm75-benchlocal-native-view')||'null');}catch{}
  const rememberView=()=>{try{localStorage.setItem('sm75-benchlocal-native-view',JSON.stringify({id:selectedId,compare:$('blCompare').value,compareTwo:$('blCompareTwo').value,detail:detailCase}));}catch{}};
  const panels={speedtest:$('benchProject-speedtest'),sql:$('benchProject-sql'),benchlocal:$('benchLocalProject')};
  const projectButtons=[...document.querySelectorAll('[data-benchmark-project]')];
  function project(value){if(!Object.hasOwn(panels,value))value='benchlocal';for(const[k,p]of Object.entries(panels))if(p)p.hidden=k!==value;for(const button of projectButtons)button.setAttribute('aria-selected',String(button.dataset.benchmarkProject===value));try{localStorage.setItem('sm75-model-test-project',value);}catch{}if(value==='benchlocal')void refresh();}
  for(const button of projectButtons)button.onclick=()=>project(button.dataset.benchmarkProject);
  const fields=[...root.querySelectorAll('[data-bl-param]')];
  function readOptions(){
    const value={thinking:$('blThinking').value,effort:$('blEffort').value,presetName:$('blPresetName').value.trim()||($('blPreset').value==='default'?'题包默认':'自定义')};
    for(const input of fields){if(!input.checkValidity())throw Error('请检查 '+input.parentElement.textContent.trim());if(input.value!=='')value[input.dataset.blParam]=Number(input.value);}
    if(value.thinking==='on'&&value.effort==='custom')value.thinking_token_budget=Number($('blThinkingBudget').value);
    if(value.thinking==='on'&&budgets[value.effort])value.thinking_token_budget=budgets[value.effort];
    if(value.top_k===0)throw Error('Top K 使用 -1 或正整数');
    if(value.thinking_token_budget&&value.max_tokens&&value.max_tokens<=value.thinking_token_budget)throw Error('最大生成 Token 需大于思考预算');
    value.request_timeout_seconds??=defaultRequestTimeout;
    return value;
  }
  function setOptions(value={}){
    for(const input of fields)input.value=value[input.dataset.blParam]??(input.dataset.blParam==='request_timeout_seconds'?defaultRequestTimeout:'');
    $('blThinking').value=value.thinking||'default';$('blEffort').value=value.effort||'default';
    $('blThinkingBudget').value=value.thinking_token_budget??'';$('blPresetName').value=value.presetName||'';
    const saved=data?.presets.find(x=>x.name===value.presetName);
    $('blPreset').value=saved?'saved:'+saved.name:fields.some(input=>input.dataset.blParam!=='request_timeout_seconds'&&input.value!=='')||Number($('blRequestTimeout').value)!==defaultRequestTimeout||$('blThinking').value!=='default'?'custom':'default';paintThinking();
  }
  function remember(){try{localStorage.setItem('sm75-benchlocal-native-preferences',JSON.stringify({selected:[...selected],selectionPreset:selected.size===6&&data.packs.filter(p=>p.defaultSelected).every(p=>selected.has(p.id))?'default':'custom',options:readOptions()}));}catch{}}
  function paintThinking(){
    const cap=data?.currentModel?.capabilities||{},on=$('blThinking').value==='on';
    $('blThinking').disabled=!cap.toggle;
    $('blEffort').disabled=!on||!(cap.budget||cap.effort?.length);
    const custom=on&&$('blEffort').value==='custom'&&cap.budget;
    $('blThinkingBudget').disabled=!custom;
    if(!custom)$('blThinkingBudget').value=on?budgets[$('blEffort').value]||'':'';
    $('blParameterNote').textContent=cap.toggle?'参数用于下一次评测，历史记录保留原值。':cap.reason||'启动模型后读取思考控制能力。';
    for(const option of $('blEffort').options)if(option.value!=='default')option.disabled=!cap.budget&&!cap.effort?.includes(option.value);
  }
  for(const input of [...fields,$('blThinking'),$('blEffort'),$('blThinkingBudget')])input.onchange=()=>{$('blPreset').value='custom';paintThinking();remember();};
  $('blPreset').onchange=()=>{const value=$('blPreset').value;setOptions(value==='default'?{}:data?.presets.find(x=>'saved:'+x.name===value)?.options||{});remember();};
  $('blSavePreset').onclick=run(async()=>{const name=$('blPresetName').value.trim();if(!name)throw Error('请输入参数方案名称');await api('/console-api/benchlocal',{preset:{name,options:readOptions()}});presetSignature='';await refresh();$('blPreset').value='saved:'+name;$('blStatus').textContent='参数方案已保存';});
  function paintSelection(){
    const packs=data?.packs||[];let count=0;
    for(const pack of packs){const input=root.querySelector('input[data-pack="'+pack.id+'"]');if(input){input.checked=selected.has(pack.id);input.closest('.bl-category').classList.toggle('selected',input.checked);}if(selected.has(pack.id))count+=pack.scenarioCount;}
    $('blSelectionCount').textContent=`已选 ${selected.size} 类 · ${count} 题`;
    const unavailable=[...selected].some(id=>data?.readiness?.[id]&&!data.readiness[id].ready);
    $('blRun').disabled=!!data?.activeId||!selected.size||!data?.currentModel?.running||unavailable;
    $('blStop').disabled=!data?.activeId;
    for(const[id,yes]of [['blSelectSix',selected.size===6&&packs.filter(p=>p.defaultSelected).every(p=>selected.has(p.id))],['blSelectAll',selected.size===packs.length],['blSelectNone',!selected.size]])$(id).setAttribute('aria-pressed',String(yes));
  }
  function paintCatalog(){
    const signature=JSON.stringify(data.packs.map(x=>[x.id,x.sourceDigest,data.readiness?.[x.id]]));
    if(signature!==catalogSignature){catalogSignature=signature;const cards=[];
      for(const pack of data.packs){
        const label=node('label',undefined,{class:'bl-category',title:pack.name}),head=node('span',undefined,{class:'bl-category-head'}),check=node('input',undefined,{type:'checkbox','data-pack':pack.id,'aria-label':pack.label});
        head.append(check,node('span',pack.label),node('small',pack.name,{class:'bl-pack-name'}),node('small',pack.scenarioCount+'题'));label.append(head,node('span',pack.description,{class:'bl-description'}));
        if(data.readiness?.[pack.id]&&!data.readiness[pack.id].ready)label.append(node('small',data.readiness[pack.id].reason,{class:'bl-unavailable'}));
        check.onchange=()=>{check.checked?selected.add(pack.id):selected.delete(pack.id);paintSelection();remember();if(!record)renderResults();};cards.push(label);
      }$('blCategories').replaceChildren(...cards);
    }paintSelection();
  }
  for(const[id,which]of [['blSelectSix','six'],['blSelectAll','all'],['blSelectNone','none']])$(id).onclick=()=>{selected=new Set((data?.packs||[]).filter(p=>which==='all'||which==='six'&&p.defaultSelected).map(p=>p.id));paintSelection();remember();if(!record)renderResults();};
  function fillSelect(select,empty,value){const options=[node('option',empty,{value:''})];for(const item of data.jobs){options.push(node('option',`${new Date(item.created).toLocaleString()} · ${item.model.name} · ${item.options?.presetName||'默认'}${item.source==='imported'?' · 导入':''}${item.partialTest?' · 单题复测':''}`,{value:item.id}));}select.replaceChildren(...options);select.value=data.jobs.some(x=>x.id===value)?value:'';}
  async function loadReport(id){if(!id)return null;if(reports.has(id))return reports.get(id);const row=await api('/console-api/benchlocal/result?id='+encodeURIComponent(id));reports.set(id,row);return row;}
  function compared(){const ids=[selectedId,$('blCompare').value,$('blCompareTwo').value].filter(Boolean);return [...new Set(ids)].map(id=>id===selectedId?record:reports.get(id)).filter(Boolean);}
  function overviewPacks(){return (data?.packs||[]).filter(p=>(record?.selection||[...selected]).includes(p.id));}
  function comparability(rows){
    if(rows.length<2)return '';
    const first=rows[0],notes=[];
    for(const other of rows.slice(1)){
      if(other.source==='imported')notes.push('含导入记录');
      if(other.protocol!==first.protocol||other.suites.some(s=>first.suites.some(x=>x.id===s.id&&x.sourceDigest!==s.sourceDigest)))notes.push('题库或评分版本不同');
      if(JSON.stringify(canonical(other.options))!==JSON.stringify(canonical(first.options)))notes.push('参数方案不同');
      if(other.fixtureId!==first.fixtureId&&other.selection.includes('promptauthority-15')&&first.selection.includes('promptauthority-15'))notes.push('权限测试变量不同');
    }
    return [...new Set(notes)].join(' · ');
  }
  function drawRadar(){
    const svg=$('blRadar'),rows=compared(),font=parseFloat(getComputedStyle(root).fontSize)||14;
    const bounds=svg.getBoundingClientRect(),width=Math.max(240,bounds.width),height=bounds.height||260*font/16,cx=width/2,cy=height/2;
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.replaceChildren();
    const make=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;};
    svg.append(make('title',{},'每个测试项目使用原题包0至100分，缺测不计零分'));
    const axes=overviewPacks();
    svg.setAttribute('aria-label',`${axes.length}项能力雷达图，范围0至100分，缺测不计零分`);
    if(axes.length<3){svg.append(make('text',{x:cx,y:cy,'text-anchor':'middle'},'选择至少三项后显示雷达图'));return;}
    const r=Math.max(30,Math.min(height/2-font*2.1,width/2-font*6)),colors=['var(--blue)','var(--amber)','var(--green)'];
    const point=(i,percent)=>{const a=-Math.PI/2+i*2*Math.PI/axes.length;return[cx+Math.cos(a)*r*percent/100,cy+Math.sin(a)*r*percent/100];};
    const path=(points,close)=>points.map((p,i)=>(i?'L':'M')+p[0]+','+p[1]).join(' ')+(close?' Z':'');
    for(const level of [20,40,60,80,100]){svg.append(make('path',{d:path(axes.map((_,i)=>point(i,level)),true),fill:'none',stroke:'var(--line)','stroke-width':1}));svg.append(make('text',{x:cx+4,y:cy-r*level/100-3,class:'bl-radar-tick'},String(level)));}
    axes.forEach((p,i)=>{const outer=point(i,100),a=-Math.PI/2+i*2*Math.PI/axes.length,c=Math.cos(a);svg.append(make('line',{x1:cx,y1:cy,x2:outer[0],y2:outer[1],stroke:'var(--line)'}));svg.append(make('text',{x:cx+c*(r+16),y:cy+Math.sin(a)*(r+22)+4,'text-anchor':c>.25?'start':c<-.25?'end':'middle'},p.label));});
    rows.forEach((row,index)=>{
      const points=axes.map((p,i)=>{const score=packSuiteScore(row.suites.find(s=>s.id===p.id));return Number.isFinite(score)?point(i,score):null;});
      if(points.every(Boolean))svg.append(make('path',{d:path(points,true),fill:colors[index],'fill-opacity':.07,stroke:colors[index],'stroke-width':2,'stroke-dasharray':index===1?'5 3':index===2?'2 3':''}));
      else for(let i=0;i<points.length;i++){const next=(i+1)%points.length;if(points[i]&&points[next])svg.append(make('path',{d:path([points[i],points[next]],false),fill:'none',stroke:colors[index],'stroke-width':2}));}
      points.forEach((p,i)=>{if(!p)return;const mark=make(index?'rect':'circle',index?{x:p[0]-3,y:p[1]-3,width:6,height:6,fill:colors[index]}:{cx:p[0],cy:p[1],r:3,fill:colors[index]});mark.append(make('title',{},row.model.name+' · '+axes[i].label+' '+number(packSuiteScore(row.suites.find(s=>s.id===axes[i].id)))));svg.append(mark);});
    });
    if(!rows.length)svg.append(make('text',{x:cx,y:cy+5,'text-anchor':'middle'},'开始评测后显示结果'));
  }
  function renderScores(){
    root.style.setProperty('--bl-radar-height',overviewPacks().length>6?'300px':'230px');
    const rows=compared(),table=$('blScores');table.replaceChildren();const head=node('thead'),tr=node('tr');tr.append(node('th','测试项目'));
    for(const [i,row]of rows.entries())tr.append(node('th',i?'对照'+i:'本次',{title:row.model.name+' · '+parameters(row)}));if(!rows.length)tr.append(node('th','得分'));head.append(tr);table.append(head);
    const body=node('tbody');for(const p of overviewPacks()){const r=node('tr');r.append(node('td',p.label));for(const row of rows)r.append(node('td',number(packSuiteScore(row.suites.find(x=>x.id===p.id)))));if(!rows.length)r.append(node('td','—'));body.append(r);}table.append(body);
    $('blComparisonNote').textContent=comparability(rows);
    $('blRadarLegend').replaceChildren(...rows.map((row,i)=>{const label=node('span',undefined);label.append(node('i',undefined,{'data-series':i}),node('span',row.model.name+' · '+(row.options?.presetName||'默认')));return label;}));drawRadar();
  }
  function findResult(packId,id){return record?.suites.find(x=>x.id===packId)?.results.find(x=>x.scenarioId===id);}
  function renderMatrix(){
    if(!data)return;const scroll=$('blMatrixScroll'),previous=scroll.scrollLeft,table=$('blMatrix');table.replaceChildren();
    const font=parseFloat(getComputedStyle(root).fontSize)||14,scale=font/16,first=Math.round(160*scale),score=Math.round(56*scale),minimumCaseWidth=Math.ceil(34*scale);
    scroll.style.setProperty('--bl-project-width',first+'px');table.style.width='100%';table.style.minWidth=(first+score+minimumCaseWidth*20)+'px';
    const cols=node('colgroup');for(const width of[first,score,...Array(20).fill(null)]){const c=node('col');if(width!==null)c.style.width=width+'px';cols.append(c);}table.append(cols);
    const head=node('thead'),tr=node('tr');tr.append(node('th','测试项目',{class:'bl-project',scope:'col'}),node('th','得分',{class:'bl-score',scope:'col'}),node('th','逐题结果 · 每排20题',{colspan:20,scope:'colgroup'}));head.append(tr);table.append(head);
    for(const pack of data.packs){const suite=record?.suites.find(x=>x.id===pack.id),chosen=record?record.selection.includes(pack.id):selected.has(pack.id),count=Math.ceil(pack.scenarioCount/20),group=node('tbody',undefined,{'data-suite':pack.id,class:count>1?'bl-suite bl-suite-multi':'bl-suite'});
      const metas=suite?.scenarios||pack.scenarios;
      for(let line=0;line<count;line++){
        const row=node('tr');if(line===0){const label=node('th',undefined,{rowspan:count,scope:'rowgroup',class:'bl-project'});const heading=node('div',undefined,{class:'bl-suite-heading'});heading.append(node('span',pack.label),node('small',chosen?(suite?.results.length||0)+'/'+pack.scenarioCount:'未选择'));label.append(heading);const score=node('td',undefined,{rowspan:count,class:'bl-score'});score.append(node('div',number(packSuiteScore(suite)),{class:'bl-suite-heading'}));row.append(label,score);}
        for(let slot=0;slot<20;slot++){const index=line*20+slot,td=node('td'),meta=metas[index];if(!meta){td.className='bl-empty';row.append(td);continue;}
          const result=findResult(pack.id,meta.id),state=result?.status||(suite?.current===meta.id?'running':'missing'),label=outcomeLabel(result,state);
          const button=node('button',undefined,{type:'button',class:'bl-case','data-result':state,'aria-label':pack.label+' '+meta.id+' '+label,title:meta.id+' · '+label+(Number.isFinite(packResultScore(result))?' · '+number(packResultScore(result))+'分':'')});
          button.append(node('span',String(index+1).padStart(2,'0'),{class:'bl-case-number'}),node('small',duration(result?.durationMs)));
          button.setAttribute('aria-pressed',String(detailCase?.packId===pack.id&&detailCase?.id===meta.id));button.onclick=()=>{detailCase={packId:pack.id,id:meta.id};renderDetail();renderMatrix();rememberView();};td.append(button);row.append(td);
        }group.append(row);
      }table.append(group);
    }scroll.scrollLeft=previous;
  }
  function renderDetail(){
    $('blDetail').hidden=!detailCase;if(!detailCase)return;
    const pack=data.packs.find(x=>x.id===detailCase.packId),suite=record?.suites.find(x=>x.id===pack.id),meta=(suite?.scenarios||pack.scenarios).find(x=>x.id===detailCase.id),result=findResult(pack.id,detailCase.id);
    $('blDetailTitle').textContent=pack.label+' · '+detailCase.id+' · '+(meta?.title||'');
    const content=$('blDetailContent');content.replaceChildren();
    const add=(title,value,open=false)=>{if(value===undefined||value===null||value==='')return;const details=node('details'),summary=node('summary',title),pre=node('pre',typeof value==='string'?value:JSON.stringify(value,null,2));details.open=open;details.append(summary,pre);content.append(details);};
    if(result?.status==='error'){
      const timeout=(result.upstreamSummary||result.summary||'').match(/(?:after\s+|（)(\d+(?:\.\d+)?)\s*(?:s|秒)/i)?.[1]||result.requestTimeoutSeconds||record?.options?.generation?.request_timeout_seconds;
      add('异常原因',outcomeLabel(result)+(outcomeLabel(result)==='请求超时'&&timeout?`：单次等待 ${timeout} 秒，尝试 ${result.requests?.length||1} 次。可在开始前调整“请求超时（秒）”。`:'：'+(result.upstreamSummary||result.summary||'请查看实际请求记录。')),true);
    }
    add('题目与评分结果',[meta?.description,meta?.promptText,result?.summary,result?.note,result?'状态：'+outcomeLabel(result)+'；题目得分：'+number(packResultScore(result))+(result.points!==undefined?'；原始点数：'+result.points:'')+'；耗时：'+duration(result.durationMs):'尚未测试'].filter(Boolean).join('\n\n'),true);
    add('评分检查项',result?.checks||result?.evaluation||result?.output?.verification||meta?.detailCards);
    add('模型输出及工具轨迹',result?.output||result?.trace||result?.rawLog);
    add('实际模型请求与响应',result?.requests);
    add('Token 用量',result?.tokenUsage);
    add('参数快照',{model:record?.model,options:record?.options,suiteDefaults:suite?.defaults,sent:result?.requests?.map(x=>x.settings)});
    if(result&&!result.requests&&!result.rawLog&&!result.output)content.append(node('p','此报告没有附带完整执行轨迹。',{class:'muted'}));
  }
  function renderResults(){
    $('blResultTitle').textContent=record?record.model.name+' · '+parameters(record)+(record.source==='imported'?' · 导入记录':''):'选择测试类目后开始';
    $('blScore').textContent=record?(states[record.state]||record.state)+(record.partialTest?' · 单题复测':''):'';
    $('blProgressRow').hidden=!record;$('blProgress').hidden=!record;$('blProgress').max=record?.total||1;$('blProgress').value=record?.done||0;
    $('blProgressText').textContent=record?`${record.done||0}/${record.total} 题`+(record.current?' · '+(data.packs.find(p=>p.id===record.current.suiteId)?.label||'')+' · '+record.current.scenarioId:''):'';
    $('blError').textContent=record?.error||'';$('blError').hidden=!record?.error;
    $('blDownload').disabled=!record;$('blReplay').disabled=!record||!!data?.activeId||record.protocol!=='sm75-native-v1'||!data.currentModel?.running;
    $('blReportParameters').textContent=record?parameters(record):'';renderScores();renderMatrix();renderDetail();
  }
  async function choose(id){const current=++sequence;selectedId=id;const next=await loadReport(id);if(current!==sequence)return;record=next;revision='';$('blHistory').value=id;renderResults();rememberView();}
  async function refresh(){
    if(loading||$('tests')?.hidden||$('benchLocalProject').hidden)return;loading=true;
    try{
      data=await api('/console-api/benchlocal');if(loadError){$('blStatus').textContent='';loadError=false;}$('benchToggle-benchlocal').checked=data.enabled;root.hidden=!data.enabled;
      if(!selectionReady){let saved;try{saved=JSON.parse(localStorage.getItem('sm75-benchlocal-native-preferences')||'null');}catch{}selected=new Set(restoreBenchLocalSelection(data.packs,saved));if(saved)setOptions(saved.options||{});selectionReady=true;remember();}
      const ps=JSON.stringify(data.presets);if(ps!==presetSignature){presetSignature=ps;const before=$('blPreset').value;$('blPreset').replaceChildren(node('option','题包默认',{value:'default'}),node('option','自定义',{value:'custom'}),...data.presets.map(x=>node('option',x.name,{value:'saved:'+x.name})));$('blPreset').value=[...$('blPreset').options].some(o=>o.value===before)?before:'custom';}
      $('blCurrentModel').textContent=data.currentModel.running?'当前模型：'+data.currentModel.modelName+' · 名称、API 与凭据自动读取':data.currentModel.loading?'模型正在加载，完成后可以开始评测':'请先在运行配置中启动模型';paintThinking();paintCatalog();
      const view=savedView;savedView=null;
      if(view){if(data.jobs.some(x=>x.id===view.id))selectedId=view.id;const p=data.packs.find(x=>x.id===view.detail?.packId);if(p?.scenarios.some(x=>x.id===view.detail.id))detailCase=view.detail;}
      const a=view?.compare||$('blCompare').value,b=view?.compareTwo||$('blCompareTwo').value;
      if(!selectedId)selectedId=data.activeId||data.jobs[0]?.id||'';
      fillSelect($('blHistory'),'暂无记录',selectedId);fillSelect($('blCompare'),'不作对照',a);fillSelect($('blCompareTwo'),'不作对照',b);
      if(view)await Promise.all([$('blCompare').value,$('blCompareTwo').value].filter(Boolean).map(loadReport));
      const item=data.jobs.find(x=>x.id===selectedId),next=JSON.stringify(item);
      if(next!==revision){revision=next;if(selectedId){reports.delete(selectedId);const id=selectedId,row=await loadReport(id);if(id===selectedId)record=row;}else record=null;renderResults();}
    }catch(error){loadError=true;$('blStatus').textContent=error.message||'能力评测暂时不可用';}finally{loading=false;}
  }
  $('benchToggle-benchlocal').onchange=run(async()=>{await api('/console-api/benchlocal',{enabled:$('benchToggle-benchlocal').checked});await refresh();});
  async function start(body){remember();$('blRun').disabled=true;$('blStatus').textContent='正在准备评测…';try{const row=await api('/console-api/benchlocal/run',body);reports.set(row.id,row);selectedId=row.id;record=row;revision='';data.activeId=row.id;renderResults();await refresh();rememberView();$('blStatus').textContent='';}catch(error){$('blStatus').textContent=error.message;throw error;}finally{paintSelection();}}
  $('blRun').onclick=run(()=>start({packIds:[...selected],options:readOptions()}));
  $('blReplay').onclick=run(()=>start({replayId:selectedId}));
  $('blStop').onclick=run(async()=>{if(data?.activeId)await api('/console-api/benchlocal/cancel',{id:data.activeId});await refresh();});
  $('blHistory').onchange=run(()=>choose($('blHistory').value));
  for(const id of ['blCompare','blCompareTwo'])$(id).onchange=run(async()=>{await loadReport($(id).value);renderScores();rememberView();});
  $('blCloseDetail').onclick=()=>{detailCase=null;renderDetail();renderMatrix();rememberView();};
  $('blImport').onclick=()=>$('blImportFiles').click();
  $('blImportFiles').onchange=run(async()=>{
    for(const file of $('blImportFiles').files){if(file.size>24*1024*1024)throw Error('报告不能超过24 MiB');const value=JSON.parse(await file.text());const row=await api('/console-api/benchlocal/import',{report:value});reports.set(row.id,row);await refresh();if(!selectedId)await choose(row.id);else if(row.id!==selectedId){$('blCompare').value=row.id;renderScores();}}
    $('blImportFiles').value='';rememberView();$('blStatus').textContent='报告已导入；当前运行模型未改变';
  });
  $('blDownload').onclick=run(async()=>{
    if(!record)return;const mode=$('blExportFormat').value,result=await api('/console-api/benchlocal/export?id='+encodeURIComponent(record.id)+'&format='+(mode==='csv'?'csv':'json')+(mode==='detailed'?'&detailed=1':''));
    const blob=new Blob([typeof result.content==='string'?result.content:JSON.stringify(result.content,null,2)],{type:mode==='csv'?'text/csv;charset=utf-8':'application/json'}),url=URL.createObjectURL(blob),a=node('a',undefined,{href:url,download:result.filename});a.click();setTimeout(()=>URL.revokeObjectURL(url),0);
  });
  if(window.ResizeObserver)new ResizeObserver(()=>{if(data){drawRadar();renderMatrix();}}).observe($('blRadarWrap'));
  let saved='benchlocal';try{saved=localStorage.getItem('sm75-model-test-project')||saved;}catch{}project(saved);
  return {refresh};
}
