import { metrics } from "./telemetry.mjs";

// Gauge selection is per request at the API server. Counter deltas remain
// useful engine-window throughput, but never substitute for request timing.
export function requestRates(sample, now) {
  const single = name => {
    const values = sample?.["vllm:sm75_request_" + name];
    return values?.length === 1 && Number.isFinite(values[0].value) ? values[0].value : null;
  };
  const timestamp = single("timing_updated_seconds");
  const sampledAt = timestamp === null ? null : timestamp * 1000;
  const fresh = sampledAt !== null && now >= sampledAt && now - sampledAt <= 15000;
  const rate = name => { const value = single(name + "_tokens_per_second"); return fresh && value !== null && value >= 0 ? value : null; };
  return {prefill: rate("prefill"), decode: rate("decode"), sampledAt,
    active: fresh ? single("timing_active") === 1 : false,
    contract: "llm-speedtest-client-v1", scope: "api-server-stream"};
}

export function liveSummary(
  rows,
  hardware,
  { now = Date.now(), failed = false } = {},
) {
  const last = rows.at(-1);
  const fresh = !!last && !failed && now - last.t <= 15000 && now >= last.t;
  const recent = fresh ? rows.filter((p) => p.t >= last.t - 60000) : [];
  const base = recent[0];
  let cacheHit = null;
  if (
    recent.length > 1 &&
    !recent.some((p) => p.gap) &&
    !metrics.counterReset(last.m, base.m)
  ) {
    const d = metrics.delta(last.m, base.m);
    const hits = metrics.sum(d, "prefix_cache_hits_total");
    const queries = metrics.sum(d, "prefix_cache_queries_total");
    if (hits !== null && queries > 0 && hits >= 0 && hits <= queries)
      cacheHit = (hits / queries) * 100;
  }
  const request = requestRates(fresh ? last.m : null, now);
  const hardwareFresh =
    hardware &&
    now - hardware.time * 1000 <= 15000 &&
    now >= hardware.time * 1000;
  const gpus = hardwareFresh ? hardware.gpus || [] : [];
  const complete = (key) =>
    gpus.length > 0 && gpus.every((g) => Number.isFinite(g[key]));
  const memoryPercent = (g) =>
    Number.isFinite(g.usedMiB) && Number.isFinite(g.totalMiB) &&
    g.totalMiB > 0 && g.usedMiB >= 0 && g.usedMiB <= g.totalMiB
      ? (100 * g.usedMiB) / g.totalMiB : null;
  const gpuDetails = gpus.map((g) => ({
    index: g.index, temp: g.temp, power: g.power,
    utilization: Number.isFinite(g.util) && g.util >= 0 && g.util <= 100 ? g.util : null,
    memory: memoryPercent(g),
  }));
  const maximum = (key) => gpuDetails.length && gpuDetails.every((g) => Number.isFinite(g[key]))
    ? Math.max(...gpuDetails.map((g) => g[key])) : null;
  return {
    fresh,
    sampledAt: last?.t ?? null,
    prefill: request.prefill,
    decode: request.decode,
    requestTiming: request,
    windowPrefill: fresh ? last.prefill : null,
    windowDecode: fresh ? last.decode : null,
    kv: fresh ? last.kv : null,
    cacheHit,
    cacheSeconds: recent.length > 1 ? (last.t - base.t) / 1000 : 0,
    temperature: complete("temp") ? Math.max(...gpus.map((g) => g.temp)) : null,
    power: complete("power") ? gpus.reduce((sum, g) => sum + g.power, 0) : null,
    utilization: maximum("utilization"),
    memory: maximum("memory"),
    gpus: gpuDetails,
  };
}
