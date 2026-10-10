'use strict';
const fs = require('node:fs');
const path = require('node:path');

// The pinned pack entry points only use these host bindings. The benchmark
// definitions, orchestration and scoring remain the upstream release code.
exports.defineBenchPack = pack => pack;
exports.loadBenchPackManifest = directory => JSON.parse(fs.readFileSync(path.resolve(directory, '../../benchlocal.pack.json'), 'utf8'));
exports.requireScoredResults = results => results.map(result => ({...result, score: result.score ?? 0}));
exports.createHostHelpers = context => ({
  getScenarioById(scenarios, id) {
    const value = scenarios.find(item => item.id === id);
    if (!value) throw Error(`未知题目：${id}`);
    return value;
  },
  getRequiredProvider(id) {
    const value = context.providers.find(item => item.id === id && item.enabled);
    if (!value) throw Error('当前模型连接不可用');
    return value;
  },
  getSecretValue(id) { return context.secrets.find(item => item.providerId === id)?.value; },
  getRequiredVerifier(id) {
    const value = context.verifiers?.find(item => item.id === id && item.status === 'running');
    if (!value?.url) throw Error('受管验证环境尚未就绪');
    return value;
  },
  getRequiredInferenceEndpoint(id) {
    const value = context.inferenceEndpoints?.find(item => item.modelId === id && item.status === 'running');
    if (!value) throw Error('本次评测的模型代理不可用');
    return value;
  },
  resolveGenerationRequest(generation = {}) {
    return Object.fromEntries(Object.entries(generation).filter(([, value]) => value !== undefined));
  },
});
