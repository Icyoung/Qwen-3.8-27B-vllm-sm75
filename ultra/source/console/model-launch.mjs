import {inferenceProfile, inferenceProfiles} from './inference-profiles.mjs';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {argValue, setArg, retargetContextModel} from './public/context-config.js';

export function modelFormat(config) {
  const q = config.quantization_config || config.text_config?.quantization_config || {};
  const method = String(q.quant_method || '').toLowerCase();
  const scheme = JSON.stringify(q).toLowerCase();
  if (scheme.includes('nvfp4')) return 'nvfp4';
  if (method.includes('awq')) return 'awq';
  if (method.includes('gptq')) return 'gptq';
  if (method === 'compressed-tensors' && /"num_bits":4/.test(scheme) && scheme.includes('"type":"int"')) return 'awq';
  if (method.includes('fp8') || scheme.includes('float8') || scheme.includes('fp8')) return 'fp8';
  if (method.includes('int8') || (method === 'compressed-tensors' && /"num_bits":8/.test(scheme) && scheme.includes('"type":"int"'))) return 'int8';
  if (method) return 'custom';
  return String(config.dtype || config.torch_dtype || config.text_config?.dtype || '').includes('bfloat16') ? 'bf16' : 'fp16';
}
const modelFlags = ['--quantization','--tokenizer','--tokenizer-mode','--tokenizer-revision','--revision','--code-revision',
  '--speculative-config','--scheduler-cls','--tool-call-parser','--reasoning-parser','--chat-template',
  '--chat-template-content-format','--tool-parser-plugin','--enable-auto-tool-choice','--hf-overrides',
  '--rope-scaling','--rope-theta','--config-format','--load-format','--model-loader-extra-config'];
export function createModelDraft(model, source, {settings, cards = [], preset = 'default', standalone = true} = {}) {
  if (!model || model.role === 'draft') throw Error('请选择主模型；草稿在加速配置中选择');
  const selectedProfile=inferenceProfiles().find(p=>p.id===preset);
  if (selectedProfile) {
    let config={};
    try {config=JSON.parse(fs.readFileSync(path.join(model.path,'config.json'),'utf8'));} catch {}
    const identity=JSON.stringify({type:model.modelType,config});
    const match=selectedProfile.modelFamily==='qwen4_exp' ? /qwen4_exp|Qwen4Exp/.test(identity) : /qwen3_[58]|Qwen3_[58]/.test(identity);
    if(!match)throw Error(selectedProfile.modelFamily==='qwen4_exp'?'此模板仅适用于 Qwen3.8 Flash Next':'此模板适用于 Qwen3.8 27B');
    if(selectedProfile.modelFamily!=='qwen4_exp' && model.format && model.format!==selectedProfile.format)throw Error('模板与当前权重格式不匹配');
    const profile=inferenceProfile(preset,model.path,settings);
    const tp=Number(argValue(profile.args,'--tensor-parallel-size','1'));
    if (cards.length && cards.length < tp) throw Error(`此模板需要 ${tp} 张 GPU`);
    profile.id='personal-'+crypto.randomUUID().slice(0,12);
    return {profile,notices:[]};
  }
  const same = source?.args?.[0] === model.path;
  const tp = cards.length || 1;
  let args = source ? retargetContextModel(source.args, model.path, model.context) : [model.path,
    '--dtype','float16','--tensor-parallel-size',String(tp),'--max-num-seqs','1',
    '--max-num-batched-tokens','8192','--gpu-memory-utilization','0.87','--max-model-len','auto',
    '--enable-prefix-caching'];
  const notices = [];
  if (source && !same) {
    for (const flag of modelFlags) args = setArg(args, flag, null);
    notices.push('已清除原模型的量化、草稿、解析器及位置编码覆盖，请按当前模型配置。');
  }
  args = setArg(args, '--served-model-name', same ? argValue(args, '--served-model-name', model.name) : model.name);
  if (preset === 'single-long') args = setArg(args, '--max-num-seqs', 1);
  else if (preset === 'multi-user') args = setArg(args, '--max-num-seqs', 16);
  else if (preset === 'coding') args = setArg(args, '--max-num-seqs', 8);
  else if (!['default','custom'].includes(preset)) throw Error('场景模板不存在');
  const fromNext=source?.modelFamily==='qwen4_exp' || source?.env?.VLLM_SM75_QSA_FP8_MAIN_KV!==undefined;
  if(source && !same && fromNext) {
    for(const flag of ['--engram-config','--enable-expert-parallel','--enable-ep-weight-filter','--language-model-only','--cpu-offload-params']) args=setArg(args,flag,null);
  }
  const env = Object.fromEntries(Object.entries(source?.env || {}).filter(([key]) =>
    (key === 'VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET' || !/key|token|secret|password|authorization/i.test(key)) &&
    !(!same && fromNext && /^(VLLM_SM75_QSA_|VLLM_SM75_NEXT_|VLLM_SPARSE_INDEXER_|VLLM_SM75_QWEN38_)/.test(key))));
  const profile = {id:'personal-'+crypto.randomUUID().slice(0,12),name:model.name+' · '+({'single-long':'单人长上下文','multi-user':'多人并发',coding:'Coding',custom:'自定义',default:'默认'}[preset]),
    format:model.format, backend:standalone?'native':(source?.backend || 'docker'), port:8000,
    cacheRoot:settings.cacheRoot, args, env, binds:same ? structuredClone(source?.binds || []) : [],
    power:source?.power ? structuredClone(source.power) : {mode:'pstate',idleSeconds:1,util:5,confirm:60,low:8,high:16,poll:5,gpus:cards.map((g,i)=>String(g.index ?? i)).join(',') || '0'},
    ...(source?.lmcache ? {lmcache:structuredClone(source.lmcache)} : {})};
  if (same && source?.image) profile.image = source.image;
  return {profile, notices};
}
export function preflightModel(profile, {cards = [], requireWeights = true} = {}) {
  const dir = profile.args[0];
  if (!path.isAbsolute(dir)) throw Error('启动前请从模型库选择已下载的本地模型');
  const configFile = path.join(dir,'config.json');
  if (!fs.existsSync(configFile)) throw Error('模型目录缺少 config.json');
  const config = JSON.parse(fs.readFileSync(configFile,'utf8'));
  const tp = Number(argValue(profile.args,'--tensor-parallel-size','1'));
  if (!Number.isSafeInteger(tp) || tp < 1) throw Error('TP 卡数必须是正整数');
  if (cards.length && tp > cards.length) throw Error('TP 卡数超过当前可用 GPU 数量');
  if (['bf16','bfloat16'].includes(argValue(profile.args,'--dtype'))) throw Error('SM75 请使用 float16 计算精度');
  if (requireWeights) {
    const entries = fs.readdirSync(dir);
    const index = entries.find(name => /(?:safetensors|bin)\.index\.json$/.test(name));
    if (index) {
      const shards = Object.values(JSON.parse(fs.readFileSync(path.join(dir,index),'utf8')).weight_map || {});
      if (!shards.length) throw Error('模型权重索引为空');
      for (const name of new Set(shards)) {
        const full = path.resolve(dir,name), relative = path.relative(dir,full);
        if (relative.startsWith('..') || path.isAbsolute(relative) || !fs.existsSync(full)) throw Error('模型权重分片缺失或路径无效：'+name);
      }
    } else if (!entries.some(name => /\.(safetensors|bin|pt|pth)$/.test(name))) throw Error('模型权重尚未下载完整');
  }
  const raw = argValue(profile.args,'--speculative-config');
  if (raw) {
    const spec = JSON.parse(raw);
    if (spec.method === 'dflash') {
      if (!spec.model || !fs.existsSync(path.join(spec.model,'config.json'))) throw Error('请先选择已下载的匹配 DFlash2 草稿');
      const draft = JSON.parse(fs.readFileSync(path.join(spec.model,'config.json'),'utf8'));
      const targetVocab = config.text_config?.vocab_size ?? config.vocab_size;
      if (targetVocab && draft.vocab_size && targetVocab !== draft.vocab_size) throw Error('草稿与主模型词表大小不匹配');
    }
  }
  return {ok:true,model:dir,tp};
}
