import {setManagedModel} from './managed-model.js';
import {timingSafeEqual} from 'node:crypto';
// The managed local provider is written through Harness's native settings
// service. A CLI config override would prevent subsequent Models-page edits.
export const inject = ['settings', 'webServer'];
export function apply(ctx, config = {}) {
  let ready = false;
  let failed = false;
  let stage = 'waiting-for-plugins';
  let failure;
  let pending;
  let retryAfter = 0;
  const send = (res, value, status = 200) => {
    res.writeHead(status, {'content-type': 'application/json', 'cache-control': 'no-store'});
    res.end(JSON.stringify(value));
  };
  ctx.webServer.register({kind: 'exact', path: '/sm75/ready', handler: (_req, res) => {
    // A new open can retry a released writer lock without restarting Harness.
    // Never remove a lock: only its operator or owner can establish ownership.
    if (failed && failure?.code === 'CONFIG_LOCK_TIMEOUT' && Date.now() >= retryAfter)
      void initialize();
    const missing = typeof ctx.get === 'function' ? ['sessions','workspaceRegistry','sessionController'].filter(name=>{try{return !ctx.get(name);}catch{return true;}}) : [];
    const usable = ready && !missing.length;
    send(res, {ready:usable, failed, stage:ready && missing.length ? 'workbench-services:'+missing.join(',') : stage, ...(failure ? {failure} : {})}, usable ? 200 : 503);
  }});
  ctx.webServer.register({kind: 'exact', path: '/sm75/configured-models', handler: (_req, res) => {
    const section = ctx.settings.describe().find(row => row.ns === 'llm-pi-ai');
    const models = [];
    for (const [provider, value] of Object.entries(section?.value?.providers || {}))
      for (const model of value.models || []) models.push({
        id: `${provider}/${model.id}`, name: model.id, provider,
        // Deliberately whitelist presentation fields; never return credentials.
        api: model.api || value.api,
        model: {id: model.id, reasoning: model.reasoning, maxTokens: model.maxTokens},
        config: {api: value.api, defaultMaxTokens: value.defaultMaxTokens,
          ...(provider === 'sm75-local' ? {baseURL: config.provider?.baseURL} : {})},
      });
    send(res, {models});
  }});
  const authorized = req => {
    const expected = Buffer.from('Bearer '+(process.env.SM75_WORKBENCH_CONTROL_KEY || ''));
    const received = Buffer.from(req.headers.authorization || '');
    return !!process.env.SM75_WORKBENCH_CONTROL_KEY && expected.length === received.length && timingSafeEqual(expected,received);
  };
  const activity = async () => {
    const service = ctx.get('sessions');
    if (!service) throw Error('会话服务尚未就绪');
    const sessions = service.list();
    for (const session of sessions) {
      const active = await ctx.waterfall('workspace/session-activity',{sessionId:session.header.id},()=>Promise.resolve([]));
      if (active.length) return true;
    }
    return false;
  };
  ctx.webServer.register({kind:'exact',path:'/sm75/activity',handler:async (req,res)=>{
    if (req.method !== 'GET' || !authorized(req)) return send(res,{error:'Unauthorized'},401);
    try {send(res,{busy:await activity()});} catch {send(res,{error:'无法检查工作台任务状态'},503);}
  }});
  ctx.webServer.register({kind:'exact',path:'/sm75/model',handler:async (req,res)=>{
    if (req.method !== 'POST' || !authorized(req)) return send(res,{error:'Unauthorized'},401);
    try {
      if (pending) await pending;
      if (!ready) return send(res,{error:"工作台尚未就绪"},503);
      if (await activity()) return send(res,{error:'工作台仍有任务运行'},409);
      let text=''; for await (const chunk of req) {text+=chunk; if(text.length>65536) throw Error('模型配置过大');}
      const next=JSON.parse(text);
      if (next.defaultModel?.provider !== 'sm75-local' || next.provider?.baseURL !== 'http://127.0.0.1:8000/v1' || next.provider?.apiKeyEnv !== 'SM75_ENGINE_KEY') throw Error('无效的本地模型连接');
      if (next.engineKey !== undefined && (typeof next.engineKey !== "string" || !next.engineKey || next.engineKey.length>4096)) throw Error("无效凭据");
      const previous={provider:config.provider,defaultModel:config.defaultModel};
      try {
        await ctx.settings.mutate('llm-pi-ai',[{op:'set',path:['providers','sm75-local'],value:next.provider}]);
        await ctx.settings.replace('agent-default-model',next.defaultModel);
      } catch(error) {
        await ctx.settings.mutate('llm-pi-ai',[{op:'set',path:['providers','sm75-local'],value:previous.provider}]).catch(()=>{});
        throw error;
      }
      if (next.engineKey !== undefined) process.env.SM75_ENGINE_KEY=next.engineKey;
      config.provider=next.provider; config.defaultModel=next.defaultModel;
      setManagedModel(next.defaultModel);
      send(res,{ok:true});
    } catch {send(res,{error:'工作台模型连接更新失败'},503);}
  }});
  const settle = async () => {
    await ctx.root.loader.await();
    if (!config.provider || !config.defaultModel) throw Error('SM75 managed model configuration is missing');
    stage = 'local-provider';
    await ctx.settings.mutate('llm-pi-ai', [{op: 'set', path: ['providers', 'sm75-local'], value: config.provider}]);
    stage = 'default-model';
    await ctx.settings.replace('agent-default-model', config.defaultModel);
    setManagedModel(config.defaultModel);
    ready = true;
    stage = 'ready';
  };
  const initialize = () => {
    if (pending) return pending;
    failed = false;
    failure = undefined;
    stage = 'waiting-for-plugins';
    pending = settle().catch(error => {
    failure = {stage, kind:error?.name || 'Error',
      code:typeof error?.code==='string' && /^[A-Z_]+$/.test(error.code) ? error.code : 'SETUP_FAILED'};
    // Diagnostics describe the failing stage, never provider values or secrets.
    const message=String(error?.message || '');
    if(message.startsWith('atomic-write: timed out waiting for the writer lock at ')) {
      failure.code='CONFIG_LOCK_TIMEOUT';
      failure.reason='工作台配置写入锁被占用，请检查锁的持有进程';
    }
    if(/^(?:Configuration|No configurable plugin entry|Plugin entry|Config field|Service |Cannot access|Missing dependency)/.test(message))
      failure.reason=message.split('\n')[0].slice(0,200);
    failed = true;
    retryAfter = Date.now() + 3000;
    ctx.logger.error('SM75 local model setup failed at ' + stage + ' (' + failure.code + ')');
    }).finally(() => { pending = undefined; });
    return pending;
  };
  void initialize();
}
