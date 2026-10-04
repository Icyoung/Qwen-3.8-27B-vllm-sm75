# 请求速度指标与精确计数（v0.1.7 beta）

监控头部与 Ultra 顶栏使用 `llm-speedtest-client-v1` 合同：Prefill = 精确 prompt tokens / 首个非空 content 或 reasoning 的等待时间；Decode = 全部精确 completion tokens / 从首个内容到当前采样或流结束的时间。两个分母最低 1 ms。首个 token、投机首批多个 token 都不扣除；结束时间包含尾部 usage 与 DONE。计算公式与固定 llm_speedtest 客户端一致，观察点位于 API 服务端，不包含客户端网络耗时，也不等于引擎内部阶段吞吐。

流仍在进行时，以当前已发送采样的时间作为临时 Decode 终点；结束后换成完整流终点。角色、空内容和控制帧不会在生成期间提前充当首 token 边界；整条流最终仍没有内容时，才使用基准中的首 JSON 回退。Prefix 命中的完整输入量仍计入分子，不把缓存命中后的请求速度称作 GPU 本地计算吞吐。

## 普通 DSH 与外部 API

`/v1/chat/completions`、`/v1/completions` 的单 choice/单 prompt SSE 请求均由同一个引擎 ASGI 中间件观察。DSH 直连、控制台代理和外部 Docker API 共用这个入口；不要求客户端添加 `include_usage` 或 `continuous_usage_stats`。

固定 vLLM 0.30 的两个流式生成器已经用 `len(output.token_ids)` 维护准确累计计数。`docker/helpers/patch_request_usage.py` 只在累计计数更新后，写入同一个 `request.state.request_metadata` 对象的私有 `_sm75_stream_usage` 字典。每次引擎输出仅复制两个整数，时间与额外存储均为 O(1)；没有 tokenizer、token 数组复制、字符或 SSE 包数估算。Pydantic 不将该未声明的内部字段序列化到 API。中间件在发送 SSE 后读取该计数，结束时优先采用现有 `final_usage_info`。

这个补丁不更改请求、响应、usage 开关或任何 generator 的计数表达式。初次安装必须匹配审查过的完整源码 SHA256；重复安装先移除已知补丁块检查原 SHA，确认一致后不再修改。未知版本、本地修改、残缺或重复补丁会拒绝安装，两份源码全部验证后才开始写入。Standard full 与 fast 的覆盖层安装均已接线，Ultra 继承同一引擎。

实际源码来自固定提交 [chat completion](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/openai/chat_completion/serving.py) 与 [completion](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/openai/completion/serving.py)，和隔离候选镜像的实际文件逐字节一致。压缩 fixture、来源及 SHA 位于 `tests/fixtures/request_timing/`，用于不加载 CUDA 的安装与计数回归。

## 显示和缺失规则

- 使用最新开始的活跃请求；其结束后仍有请求时选择其中最新开始者，全空闲时显示最近成功完成样本。
- 样本按其自身更新时间保留 15 秒。另一个较旧并发请求的输出不会让当前样本刷新计时。
- 缺少精确计数、尚无首内容边界、异常/取消、多 choice/多 prompt，或无法明确归属的多 API worker series 均显示 `—`，不合并或估算成单请求速度。TP GPU worker 不等于多 API worker。
- 重启后 gauge 初值为 NaN。中间件与 gauge 按应用实例唯一注册。
- 窗口速率仍用于原有趋势图，字段为 `windowPrefill/windowDecode`，不填充请求头部速率。内部 Decode 阶段缺精确首批计数时也显示 `—`。
- 历史 `sm75-release-v2` 基准有不同首批/终点合同，本轮没有修改历史报告。

专用 Prometheus gauges：`vllm:sm75_request_prefill_tokens_per_second`、`vllm:sm75_request_decode_tokens_per_second`、`vllm:sm75_request_timing_updated_seconds`、`vllm:sm75_request_timing_active`。前两项单位 tok/s，更新时间为 Unix 秒，active 为 0/1。

## 可执行的本地 CPU 回归

在完整工作树根目录执行，不需要 Docker、模型或 GPU：

```bash
python -m unittest tests.test_request_usage_patch tests.test_lmcache_standard tests.test_release_scripts tests.test_build_contract
python tests/entrypoints/serve/instrumentator/test_request_timing.py
python -m pytest --noconftest tests/test_lmcache_install.py -q
node --test tests/test_dashboard_request_timing.mjs tests/test_standard_monitor_chart.mjs ultra/source/console/test/runtime-settings.test.mjs ultra/source/console/test/lmcache-runtime.test.mjs ultra/source/console/test/lmcache-config.test.mjs ultra/source/console/test/live-summary.test.mjs ultra/source/console/test/config.test.mjs ultra/source/console/test/chart.test.mjs ultra/source/console/test/dashboard-theme.test.mjs
```

`prometheus_client` 缺失时，ASGI 测试中的真实注册项会跳过；安装依赖后应重跑。项目的 Python 测试环境还需要 pytest、Pydantic；LMCache 安装器测试只模拟依赖检查和文件操作，不安装 LMCache。Linux 专属的管理服务/磁盘条件测试需在获准的隔离环境复验。本轮不据此声称新镜像已构建或真实 GPU 性能已验收。
