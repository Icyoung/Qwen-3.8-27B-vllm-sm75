import {setArg, argValue, GiB} from './context-config.js';

export function isLMCacheEnabled(profile) {
  return profile?.lmcache?.enabled === true;
}
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const hasFlag = (args, key) => args.some(value => value === key || value.startsWith(key + '='));
function number(value, label, {integer = false, zero = false} = {}) {
  if (typeof value !== 'number' || !Number.isFinite(value) || (zero ? value < 0 : value <= 0) ||
      (integer && !Number.isSafeInteger(value))) throw Error(`${label}须为${zero ? '非负' : '正'}${integer ? '整数' : '数'}`);
  return value;
}
export function validateLMCacheProfile(profile) {
  const config = profile?.lmcache;
  if (config === undefined) return profile;
  if (!object(config) || typeof config.enabled !== 'boolean') throw Error('LMCache 配置须包含 enabled 布尔开关');
  // Disabled drafts may keep partially entered values. Enabling is the explicit
  // point where every managed field becomes required and is checked.
  if (!config.enabled) return profile;
  if (!Array.isArray(profile.args) || !profile.args.length || profile.args.some(value => typeof value !== 'string'))
    throw Error('LMCache 需要完整的模型启动参数数组');
  number(config.cpuGiB, 'LMCache CPU 总容量');
  if (!Number.isSafeInteger(Math.round(config.cpuGiB * GiB)) || Math.round(config.cpuGiB * GiB) <= 0)
    throw Error('LMCache CPU 总容量超出可用范围');
  number(config.chunkSize, 'LMCache chunk size', {integer: true});
  number(config.minFreeDiskGiB, '启动所需剩余空间', {zero: true});
  if (!Number.isSafeInteger(Math.round(config.minFreeDiskGiB * GiB))) throw Error('启动所需剩余空间超出可用范围');
  if (typeof config.diskPath !== 'string' || /[\x00-\x1f\x7f]/.test(config.diskPath) ||
      (config.diskPath !== '' && (!config.diskPath.startsWith('/') || /^\/+$/u.test(config.diskPath) || config.diskPath.split('/').includes('..'))))
    throw Error('LMCache 磁盘目录须为 Linux 绝对子目录；留空关闭磁盘层');
  for (const name of ['port', 'httpPort']) {
    number(config[name], name === 'port' ? 'LMCache RPC 端口' : 'LMCache HTTP 端口', {integer: true});
    if (config[name] < 1024 || config[name] > 65535) throw Error('LMCache 端口范围为 1024–65535');
  }
  if (config.port === config.httpPort || [config.port, config.httpPort].includes(profile.port))
    throw Error('LMCache 的 RPC、HTTP 与模型 API 端口不能相同');
  const sleepOrAllocator = profile.power?.mode === 'sleep' ||
    ['--enable-sleep-mode', '--enable-cumem-allocator'].some(flag => hasFlag(profile.args, flag)) ||
    Number(argValue(profile.args, '--auto-sleep-idle-timeout')) > 0 ||
    Number(profile.env?.VLLM_AUTO_SLEEP_IDLE_TIMEOUT) > 0 ||
    ['PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'].some(name => /expandable_segments\s*:\s*true/i.test(profile.env?.[name] || ''));
  if (sleepOrAllocator)
    throw Error('LMCache MP 暂不支持释放显存的休眠或 expandable_segments；请选 P-State，关闭休眠并移除相关 allocator 参数');
  if (hasFlag(profile.args, '--disable-hybrid-kv-cache-manager'))
    throw Error('LMCache MP 需要启用 hybrid KV cache manager；请移除 --disable-hybrid-kv-cache-manager 参数');
  if (hasFlag(profile.args, '--speculative-config'))
    throw Error('LMCache 实验模式暂不支持 DSpark/DFlash/MTP 等投机解码；请先关闭投机解码，草稿配置不会自动删除');
  if (['--kv-transfer-config', '--kv-offloading-size', '--kv-offloading-backend'].some(key => hasFlag(profile.args, key)))
    throw Error('LMCache 与原生 CPU KV 或自定义缓存连接器互斥；请先明确关闭原配置');
  // Backend capability is checked against the actual launch path on the server:
  // old ultra profiles may still carry backend="docker" while running in-container.
  return profile;
}

export function lmcacheConnectorArgs(profile) {
  validateLMCacheProfile(profile);
  let args = [...profile.args];
  if (!isLMCacheEnabled(profile)) return args;
  for (const flag of ['--kv-transfer-config', '--kv-offloading-size', '--kv-offloading-backend', '--no-enable-prefix-caching'])
    args = setArg(args, flag, null);
  const config = {
    kv_connector: 'LMCacheMPConnector',
    kv_role: 'kv_both',
    kv_connector_module_path: 'lmcache.integration.vllm.lmcache_mp_connector',
    kv_connector_extra_config: {'lmcache.mp.host': 'tcp://127.0.0.1', 'lmcache.mp.port': profile.lmcache.port, 'lmcache.mp.isolated_ipc': false},
  };
  args = setArg(args, '--kv-transfer-config', JSON.stringify(config));
  args = setArg(args, '--enable-prefix-caching', true);
  return setArg(args, '--mamba-cache-mode', 'align');
}


// UI suggestions are based on a matching completed runtime layout, not the
// unaligned --block-size request (hybrid models may inflate it at startup).
export function lmcacheLayoutKey(profile) {
  const args = profile?.args || [];
  let spec = null;
  try {
    const raw = argValue(args, '--speculative-config');
    if (raw) { const v = JSON.parse(raw); spec = {method:v.method, model:v.model, tokens:v.num_speculative_tokens}; }
  } catch { spec = 'invalid'; }
  const keys = ['--tensor-parallel-size','--dtype','--kv-cache-dtype','--block-size',
    '--mamba-cache-dtype','--mamba-ssm-cache-dtype','--mamba-block-size','--hf-overrides',
    '--kv-cache-dtype-skip-layers','--kv-cache-layout'];
  return JSON.stringify({model:args[0] || profile?.defaultModel || '', spec,
    options:keys.map(key => [key,argValue(args,key)])});
}
export function lmcacheDefaultDiskPath(profile) {
  const value = String(profile?.cacheRoot || '/data/cache');
  const root = value.startsWith('/') && !/[\x00-\x1f\x7f]/.test(value) && !value.split('/').includes('..')
    ? value.replace(/\/+$/,'') : '/data/cache';
  return `${root || '/data/cache'}/lmcache/disk`;
}
export function lmcacheHintsFromLog(profile, text) {
  const args = profile?.args || [], lines = String(text || '').split('\n');
  let start = -1;
  for (let i=0;i<lines.length;i++) if (lines[i].includes('Initializing a V1 LLM engine')) start=i;
  if (start<0) return null;
  const init=lines[start], stringField=(name, source=init) => {
    const m=source.match(new RegExp(`(?:^|[ ,(])${name}=['"]([^'"]*)['"]`));return m?.[1] ?? null;
  };
  if (stringField('model') !== (args[0] || profile.defaultModel)) return null;
  const tp=Number(init.match(/(?:^|[ ,])tensor_parallel_size=(\d+)/)?.[1]);
  if (tp !== Number(argValue(args,'--tensor-parallel-size') || 1)) return null;
  const kv=init.match(/(?:^|[ ,])kv_cache_dtype=([^,\s]+)/)?.[1];
  if (kv !== (argValue(args,'--kv-cache-dtype') || 'auto')) return null;
  const dtype=argValue(args,'--dtype');
  if (dtype && dtype!=='auto' && init.match(/(?:^|[ ,])dtype=torch\.([^,\s]+)/)?.[1] !== dtype) return null;
  let spec=null;
  try {const raw=argValue(args,'--speculative-config');if(raw)spec=JSON.parse(raw);} catch {return null;}
  const logged=init.match(/speculative_config=(.*?), tokenizer=/)?.[1];
  if (spec) {
    if (!logged || logged==='None' || stringField('method',logged)!==spec.method || stringField('model',logged)!==spec.model ||
      Number(logged.match(/num_spec_tokens=(\d+)/)?.[1])!==Number(spec.num_speculative_tokens)) return null;
  } else if (logged !== 'None') return null;
  let header='';
  for (let i=start-1;i>=0;i--) if(lines[i].includes('non-default args:')) {header=lines[i];break;}
  const requested=argValue(args,'--block-size'), prior=header.match(/["']block_size["']\s*:\s*(\d+)/)?.[1];
  if (requested && (!prior || Number(prior)!==Number(requested))) return null;
  // Explicit state/layout overrides need an exact logged match. Never reuse a
  // DSpark-derived default after changing the drafter or its lookahead width.
  for (const key of ['--mamba-cache-dtype','--mamba-ssm-cache-dtype','--mamba-block-size','--kv-cache-layout']) {
    const value=argValue(args,key);
    if (value && value!=='auto') {
      const field=key.slice(2).replaceAll('-','_');
      const found=header.match(new RegExp(`["']${field}["']\s*:\s*["']?([^,'"}\s]+)`))?.[1];
      if(found!==String(value)) return null;
    }
  }
  const skips=argValue(args,'--kv-cache-dtype-skip-layers');if(skips && skips!=='[]') return null;
  try {const overrides=JSON.parse(argValue(args,'--hf-overrides') || '{}');if(Object.keys(overrides).some(k=>k!=='dtype'))return null;} catch {return null;}
  let block=null;
  for(const line of lines.slice(start)) {
    const m=line.match(/Setting attention block size to (\d+) tokens/);
    if(m)block=Number(m[1]);
  }
  if(!Number.isSafeInteger(block) || block<=0) return null;
  return {blockSize:block, chunkSize:Math.ceil(256/block)*block, source:'matching-runtime-log',
    layoutKey:lmcacheLayoutKey(profile)};
}
