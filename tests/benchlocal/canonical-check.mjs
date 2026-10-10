import fs from 'node:fs';
const key=process.env.SM75_BENCH_VERIFIER_KEY;
const targets={bugfind:'http://sm75-bench-bugfind:4200',cli:'http://sm75-bench-cli:4200',hermes:'http://sm75-bench-hermes:4200'};
const catalog=JSON.parse(fs.readFileSync('/opt/sm75-workbench/benchlocal-native/catalog.json','utf8'));
async function post(url,value){const response=await fetch(url,{method:'POST',headers:{Authorization:'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify(value),signal:AbortSignal.timeout(20*60*1000)});if(!response.ok)throw Error(url+' HTTP '+response.status);return response.json();}
const result={health:{},bugfind:null,cli:[]};
for(const [id,url]of Object.entries(targets)){
  const response=await fetch(url+'/health',{signal:AbortSignal.timeout(3000)});result.health[id]=response.ok?await response.json():{ok:false};
  if(!result.health[id].ok)throw Error(id+' is not ready');
  const denied=await fetch(url+'/cancel',{method:'POST'});if(denied.status!==401)throw Error(id+' is not authenticated');
}
result.bugfind=await post(targets.bugfind+'/verify-canonical',{variant:'all'});
fs.writeFileSync('/results/canonical-check.json',JSON.stringify(result,null,2));
if(result.bugfind.status!=='passed'||result.bugfind.scenarioCount!==15)throw Error('BugFind canonical failed');
for(const scenario of catalog.packs.find(x=>x.id==='cli-40').scenarios){
  const value=await post(targets.cli+'/verify-canonical',{scenarioId:scenario.id});result.cli.push({id:scenario.id,...value});
  fs.writeFileSync('/results/canonical-check.json',JSON.stringify(result,null,2));
  console.log(scenario.id+' '+JSON.stringify(value).slice(0,120));
}
console.log(JSON.stringify({bugfind:result.bugfind.status,cliCount:result.cli.length,health:result.health}));
