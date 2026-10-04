// Docker launch overrides are applied to a copy; stored profiles stay editable.
import {argValue, setArg, GiB} from './public/context-config.js';
import {validateLMCacheProfile} from './public/lmcache-config.js';
export const runtimeArgumentEnvironment = {
  SERVED_MODEL_NAME: '--served-model-name', TENSOR_PARALLEL_SIZE: '--tensor-parallel-size',
  MAX_NUM_SEQS: '--max-num-seqs', MAX_NUM_BATCHED_TOKENS: '--max-num-batched-tokens',
  GPU_MEMORY_UTILIZATION: '--gpu-memory-utilization', MAX_MODEL_LEN: '--max-model-len',
  KV_CACHE_DTYPE: '--kv-cache-dtype', BLOCK_SIZE: '--block-size', DTYPE: '--dtype',
  QUANTIZATION: '--quantization', HF_OVERRIDES: '--hf-overrides',
  ENGRAM_CONFIG: '--engram-config',
  ATTENTION_CONFIG: '--attention-config', GDN_PREFILL_BACKEND: '--gdn-prefill-backend',
  GENERATION_CONFIG: '--generation-config', OVERRIDE_GENERATION_CONFIG: '--override-generation-config',
  COMPILATION_CONFIG: '--compilation-config', SPECULATIVE_CONFIG: '--speculative-config',
  SCHEDULER_CLS: '--scheduler-cls', TOOL_CALL_PARSER: '--tool-call-parser',
  REASONING_PARSER: '--reasoning-parser', TOOL_PARSER_PLUGIN: '--tool-parser-plugin',
  REASONING_PARSER_PLUGIN: '--reasoning-parser-plugin', CHAT_TEMPLATE: '--chat-template',
  CHAT_TEMPLATE_CONTENT_FORMAT: '--chat-template-content-format',
  AUTO_SLEEP_IDLE_TIMEOUT: '--auto-sleep-idle-timeout',
  AUTO_SLEEP_OFFLOAD_TARGET: '--auto-sleep-offload-target',
  AUTO_SLEEP_RELOAD_PATH: '--auto-sleep-reload-path',
  AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL: '--auto-sleep-page-cache-keep-interval',
  AUTO_SLEEP_DISK_PATH: '--auto-sleep-disk-path',
};
export const runtimeBooleanEnvironment = {
  ENABLE_PREFIX_CACHING: ['--enable-prefix-caching', '--no-enable-prefix-caching'],
  ASYNC_SCHEDULING: ['--async-scheduling', '--no-async-scheduling'],
  DISABLE_CUSTOM_ALL_REDUCE: ['--disable-custom-all-reduce', '--no-disable-custom-all-reduce'],
  ENABLE_AUTO_TOOL_CHOICE: ['--enable-auto-tool-choice', ''],
  ENABLE_EXPERT_PARALLEL: ['--enable-expert-parallel','--no-enable-expert-parallel'],
  ENABLE_EP_WEIGHT_FILTER: ['--enable-ep-weight-filter','--no-enable-ep-weight-filter'],
  LANGUAGE_MODEL_ONLY: ['--language-model-only','--no-language-model-only'],
  ENABLE_PROMPT_TOKENS_DETAILS: ['--enable-prompt-tokens-details',''],
};
export const runtimeKernelEnvironment = [
  'VLLM_SM75_QSA_MIXED_PREFILL', 'VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET',
  'NCCL_P2P_LEVEL', 'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE', 'NCCL_DEBUG',
  'CUDA_VISIBLE_DEVICES', 'VLLM_USE_MODELSCOPE', 'VLLM_GDN_DECODE_KERNEL',
  'VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_MARLIN_USE_ATOMIC_ADD',
  'VLLM_USE_NCCL_SYMM_MEM', 'VLLM_ALLREDUCE_USE_SYMM_MEM',
  'VLLM_ALLREDUCE_USE_FLASHINFER', 'VLLM_FIREFLY', 'VLLM_FIREFLY_AR',
  'VLLM_FIREFLY_AR_BACKEND', 'VLLM_FIREFLY_AR_MIN_SIZE',
  'OMP_NUM_THREADS', 'MAX_JOBS', 'NVCC_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS',
  'VLLM_USE_MODELSCOPE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'VLLM_SM75_QWEN38_HC_GEMV', 'VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE', 'VLLM_USE_BREAKABLE_CUDAGRAPH', 'VLLM_WORKER_MULTIPROC_METHOD', 'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB', 'VLLM_SM75_QSA_FLASHINFER_PREFILL', 'VLLM_SM75_QSA_SHARED_TARGET_SCRATCH', 'VLLM_SM75_QSA_FP8_MAIN_KV', 'VLLM_SM75_QSA_CUBLAS_INDEXER', 'VLLM_SM75_QSA_SELECTED_PAGES_PREFILL', 'VLLM_ENFORCE_STRICT_TOOL_CALLING',
];
export const runtimeLMCacheEnvironment = {
  LMCACHE_ENABLED:'enabled', LMCACHE_CPU_GIB:'cpuGiB', LMCACHE_CHUNK_SIZE:'chunkSize',
  LMCACHE_DISK_PATH:'diskPath', LMCACHE_MIN_FREE_DISK_GIB:'minFreeDiskGiB',
  LMCACHE_PORT:'port', LMCACHE_HTTP_PORT:'httpPort',
};
const owns = (value, key) => Object.hasOwn(value, key) && value[key] !== undefined;
export function environmentBoolean(value, key) {
  const text = String(value).toLowerCase();
  if (['1','true','yes','on'].includes(text)) return true;
  if (['0','false','no','off'].includes(text)) return false;
  throw Error(`${key} 需要 true/false 或 1/0`);
}
export function runtimeEngineEnvironment(profileEnv = {}, env = process.env) {
  return {NCCL_P2P_LEVEL:'SYS', ...Object.fromEntries(runtimeKernelEnvironment
    .filter(key => owns(env, key)).map(key => [key, env[key]])), ...profileEnv};
}
export function hasRuntimeOverrides(env = {}) {
  return [...Object.keys(runtimeArgumentEnvironment), ...Object.keys(runtimeBooleanEnvironment),
    ...Object.keys(runtimeLMCacheEnvironment), 'SM75_DEFAULT_PROFILE','MODEL','GPU_KV_BYTES','CPU_KV_GIB','KV_TRANSFER_CONFIG','POWER_MODE',
    'PSTATE_IDLE_TIMEOUT','PSTATE_UTIL','PSTATE_CONFIRM','PSTATE_LOW','PSTATE_HIGH','PSTATE_POLL','PSTATE_GPUS']
    .some(key => owns(env, key));
}
export function applyRuntimeEnvironment(profile, env = {}) {
  const p = structuredClone(profile);
  p.env = runtimeEngineEnvironment(p.env, env);
  if (owns(env, 'MODEL') && env.MODEL !== '') p.args[0] = env.MODEL;
  for (const [key, flag] of Object.entries(runtimeArgumentEnvironment)) {
    if (owns(env, key)) p.args = setArg(p.args, flag, env[key] === '' || (key === 'SPECULATIVE_CONFIG' && env[key] === 'null') ? null : env[key]);
  }
  for (const [key, flags] of Object.entries(runtimeBooleanEnvironment)) {
    if (!owns(env, key)) continue;
    for (const flag of flags.filter(Boolean)) p.args = setArg(p.args, flag, null);
    const selected = flags[environmentBoolean(env[key], key) ? 0 : 1];
    if (selected) p.args.push(selected);
  }
  if (owns(env, 'GPU_KV_BYTES')) {
    if (!/^(0|[1-9][0-9]*)$/.test(env.GPU_KV_BYTES)) throw Error('GPU_KV_BYTES 需要非负整数');
    p.args = setArg(p.args, '--kv-cache-memory-bytes', env.GPU_KV_BYTES === '0' ? null : env.GPU_KV_BYTES);
  }
  const selectingLMCache = owns(env, 'LMCACHE_ENABLED')
    ? environmentBoolean(env.LMCACHE_ENABLED, 'LMCACHE_ENABLED') : p.lmcache?.enabled;
  if (selectingLMCache && owns(env, 'CPU_KV_GIB') &&
      !(owns(env, 'KV_TRANSFER_CONFIG') && env.KV_TRANSFER_CONFIG === '')) {
    const raw = argValue(p.args, '--kv-transfer-config');
    if (raw) {
      const kv = JSON.parse(raw), extra = kv.kv_connector_extra_config || {};
      if ((kv.kv_connector && kv.kv_connector !== 'OffloadingConnector') ||
          (extra.spec_name && extra.spec_name !== 'CPUOffloadingSpec') ||
          (kv.kv_role && kv.kv_role !== 'kv_both'))
        throw Error('LMCache 切换不会通过 CPU_KV_GIB 删除自定义连接器，请明确移除 KV_TRANSFER_CONFIG');
    }
    const backend = argValue(p.args, '--kv-offloading-backend');
    if (backend && backend !== 'native')
      throw Error('LMCache 切换不会删除自定义 KV offloading backend');
  }
  if (owns(env, 'CPU_KV_GIB')) {
    const text = String(env.CPU_KV_GIB);
    if (!/^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$/.test(text)) throw Error('CPU_KV_GIB 需要非负十进制 GiB 数值');
    const gib = Number(text), bytes = Math.round(gib * GiB);
    if (!Number.isSafeInteger(bytes) || bytes < 0 || (gib > 0 && bytes < 1))
      throw Error('CPU_KV_GIB 舍入后的字节数超出安全整数范围；0 表示关闭');
    for (const flag of ['--kv-offloading-size','--kv-offloading-backend','--kv-transfer-config']) p.args = setArg(p.args, flag, null);
    if (bytes > 0) p.args = setArg(p.args, '--kv-transfer-config', JSON.stringify({
      kv_connector:'OffloadingConnector', kv_role:'kv_both',
      kv_connector_extra_config:{spec_name:'CPUOffloadingSpec',cpu_bytes_to_use:bytes},
    }));
  }
  if (owns(env, 'KV_TRANSFER_CONFIG')) {
    for (const flag of ['--kv-offloading-size','--kv-offloading-backend']) p.args = setArg(p.args, flag, null);
    p.args = setArg(p.args, '--kv-transfer-config', env.KV_TRANSFER_CONFIG || null);
  }
  if (owns(env, 'SPECULATIVE_CONFIG') && !owns(env, 'SCHEDULER_CLS')) {
    p.args = setArg(p.args, '--scheduler-cls', argValue(p.args, '--speculative-config')
      ? 'vllm.v1.core.sched.scheduler_sm75.SM75Scheduler' : null);
    // Let vLLM infer capture sizes when the speculative batch shape is changed.
    if (!owns(env, 'COMPILATION_CONFIG')) {
      const raw = argValue(p.args, '--compilation-config');
      if (raw) { const config = JSON.parse(raw); delete config.cudagraph_capture_sizes;
        p.args = setArg(p.args, '--compilation-config', JSON.stringify(config)); }
    }
  }
  const powerKeys = {PSTATE_IDLE_TIMEOUT:'idleSeconds',PSTATE_UTIL:'util',PSTATE_CONFIRM:'confirm',
    PSTATE_LOW:'low',PSTATE_HIGH:'high',PSTATE_POLL:'poll',PSTATE_GPUS:'gpus'};
  if (owns(env, 'POWER_MODE') || Object.keys(powerKeys).some(key => owns(env, key))) {
    p.power = {...(p.power || {mode:'pstate'})};
    if (owns(env, 'POWER_MODE')) {
      if (!['pstate','sleep'].includes(env.POWER_MODE)) throw Error('POWER_MODE 需要 pstate 或 sleep');
      p.power.mode = env.POWER_MODE;
    }
    for (const [key, field] of Object.entries(powerKeys)) if (owns(env,key))
      p.power[field] = field === 'gpus' ? env[key] : Number(env[key]);
  }
  if (p.power?.mode === 'pstate') p.args = setArg(p.args, '--auto-sleep-idle-timeout', '0');
  else if (p.power?.mode === 'sleep') {
    const target = argValue(p.args, '--auto-sleep-offload-target', 'exit');
    if (!['exit','cpu','reload','disk'].includes(target)) throw Error('AUTO_SLEEP_OFFLOAD_TARGET 无效');
    const idle = Number(argValue(p.args, '--auto-sleep-idle-timeout', '0'));
    if (idle > 0 && ['cpu','reload','disk'].includes(target)) p.args = setArg(p.args, '--enable-sleep-mode', true);
  }
  if (Object.keys(runtimeLMCacheEnvironment).some(key => owns(env, key))) {
    p.lmcache = {enabled:false, cpuGiB:8, chunkSize:'', diskPath:'', minFreeDiskGiB:4,
      port:5555, httpPort:5556, ...(p.lmcache || {})};
    for (const [key, field] of Object.entries(runtimeLMCacheEnvironment)) {
      if (!owns(env, key)) continue;
      if (field === 'enabled') p.lmcache[field] = environmentBoolean(env[key], key);
      else if (field === 'diskPath') p.lmcache[field] = String(env[key]);
      else {
        if (String(env[key]).trim() === '') throw Error(`${key} 需要数值`);
        p.lmcache[field] = Number(env[key]);
      }
    }
  }
  if (p.lmcache?.enabled) {
    // An explicit cache-mode selection must never erase conflicting user options.
    if ((owns(env, 'CPU_KV_GIB') && Number(env.CPU_KV_GIB) > 0) ||
        (owns(env, 'KV_TRANSFER_CONFIG') && env.KV_TRANSFER_CONFIG))
      throw Error('LMCache 与显式 CPU_KV_GIB / KV_TRANSFER_CONFIG 互斥');
    if (owns(env, 'AUTO_SLEEP_IDLE_TIMEOUT') && Number(env.AUTO_SLEEP_IDLE_TIMEOUT) > 0)
      throw Error('LMCache 与显式自动休眠互斥');
    validateLMCacheProfile(p);
  }
  return p;
}
