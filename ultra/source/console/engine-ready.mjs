import {argValue} from './public/context-config.js';
export async function waitForEngine(profile, {key, alive, fetchImpl = fetch, sleep = ms => new Promise(r => setTimeout(r,ms)), timeout = 1200000, now = Date.now} = {}) {
  const expected = argValue(profile.args,'--served-model-name',profile.args[0]);
  const end = now()+timeout;
  while (now() < end) {
    if (!alive()) throw Error('模型进程已退出，请查看启动日志');
    try {
      const health = await fetchImpl(`http://127.0.0.1:${profile.port}/health`,{signal:AbortSignal.timeout(3000),headers:{Authorization:'Bearer '+key}});
      if (health.ok) {
        const response = await fetchImpl(`http://127.0.0.1:${profile.port}/v1/models`,{signal:AbortSignal.timeout(3000),headers:{Authorization:'Bearer '+key}});
        if (response.ok && (await response.json()).data?.some(model => model.id === expected)) return;
      }
    } catch {}
    await sleep(1000);
  }
  throw Error('模型启动等待超时，请查看日志后重试');
}
