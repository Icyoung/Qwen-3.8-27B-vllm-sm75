import fs from 'node:fs';
import {readContextSettings, GiB} from './public/context-config.js';

// CPUOffloadingSpec allocates one shared region for the entire TP group.
export function nativeCpuKvBytes(profile) {
  if (profile.lmcache?.enabled) return 0;
  const settings = readContextSettings(profile.args);
  if (settings.cpuKvMode !== 'native') return 0;
  const bytes = Math.round(Number(settings.cpuKvGiB) * GiB);
  if (!Number.isSafeInteger(bytes) || bytes < 0) throw Error('CPU KV 容量无效');
  return bytes;
}
export function recommendedShmGiB(profile) {
  const bytes = nativeCpuKvBytes(profile);
  return 2 ** Math.ceil(Math.log2(Math.max(16, bytes / GiB + 1)));
}
export function preflightCpuKV(profile, {capacityOnly = false, statfs = fs.statfsSync} = {}) {
  const required = nativeCpuKvBytes(profile);
  if (!required) return null;
  let stat;
  try { stat = statfs('/dev/shm', {bigint:true}); }
  catch { throw Error('无法检查 /dev/shm；原生 CPU KV 需要容器共享内存'); }
  const total = Number(stat.blocks) * Number(stat.bsize);
  const free = Number(stat.bavail) * Number(stat.bsize);
  if (total <= required || (!capacityOnly && free <= required)) {
    throw Error(`CPU KV 共享内存不足：申请 ${(required/GiB).toFixed(3)} GiB，` +
      `/dev/shm 总计 ${(total/GiB).toFixed(3)} GiB、可用 ${(free/GiB).toFixed(6)} GiB ` +
      `（${free} 字节）。请在 Docker/Unraid 将 --shm-size 设为 ${recommendedShmGiB(profile)}g 后重建容器；CPU KV 容量无需减小。`);
  }
  return {requiredBytes:required,totalBytes:total,availableBytes:free};
}
