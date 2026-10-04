import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const html=fs.readFileSync(new URL('../vllm/entrypoints/serve/instrumentator/dashboard.html',import.meta.url),'utf8');
const context={};vm.createContext(context);vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const core=context.MonitorCore;
test('standard dashboard request speed never falls back to interval counters',()=>{
 const parse=text=>core.parsePrometheus(text);
 const engine=parse('vllm:prompt_tokens_total 100\nvllm:generation_tokens_total 50');
 assert.equal(core.requestRates(engine,100000).decode,null);
 const request=parse('vllm:sm75_request_timing_updated_seconds 100\nvllm:sm75_request_prefill_tokens_per_second 512\nvllm:sm75_request_decode_tokens_per_second 32');
 assert.equal(core.requestRates(request,100000).prefill,512);
 assert.equal(core.requestRates(request,100000).decode,32);
 assert.equal(core.requestRates(request,116000).decode,null);
 request['vllm:sm75_request_decode_tokens_per_second'][0].value=NaN;
 assert.equal(core.requestRates(request,100000).decode,null);
});


test('both dashboards withhold decode phase average without exact first-emission counts',()=>{
 for(const relative of ['../vllm/entrypoints/serve/instrumentator/dashboard.html','../ultra/source/vllm/entrypoints/serve/instrumentator/dashboard.html']){
  const source=fs.readFileSync(new URL(relative,import.meta.url),'utf8');
  const scope={};vm.createContext(scope);vm.runInContext(source.match(/<script>([\s\S]*?)<\/script>/)[1],scope);
  const metrics=scope.MonitorCore.parsePrometheus('vllm:request_generation_tokens_sum 10\nvllm:request_generation_tokens_count 1\nvllm:request_decode_time_seconds_sum 3\nvllm:request_prefill_kv_computed_tokens_sum 100\nvllm:request_prefill_time_seconds_sum 2');
  assert.equal(scope.MonitorCore.phaseRates(metrics).decode,null);
  assert.equal(scope.MonitorCore.phaseRates(metrics).prefill,50);
 }
});
