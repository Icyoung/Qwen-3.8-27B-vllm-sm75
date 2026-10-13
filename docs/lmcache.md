# 可选 LMCache 扩展

本 fork 的标准镜像不安装本项目固定的 LMCache 扩展，也不启用分层缓存。相关源码已从旧工作台中移至 [`extensions/lmcache`](../extensions/lmcache/patches/README.md)，包含安装器、固定依赖、packed4D 补丁、运行前探针和许可证；不依赖 Node 或 Web 控制台。

```bash
docker build -f docker/Dockerfile.lmcache \
  --build-arg BASE_IMAGE=vllm-sm75:v$(cat docker/VERSION) \
  -t vllm-sm75:icy-lmcache .
# 扩展镜像内：只验证已安装依赖，不执行推理。
python3 /opt/vllm-sm75/lmcache/install.py
```

安装器锁定 LMCache 0.5.5、CUDA 12.9 和 vLLM 0.30.0 的兼容依赖，核对 wheel 和 patched module SHA。它不会解析升级 torch/vLLM/CUDA/FlashInfer。构建时使用 `--build-no-gpu` 验证原生 ABI，驱动相关的完整探针留给运行环境。探针为 `/opt/vllm-sm75/lmcache/probe.py`，同样不启动模型；配置 serverArgs 必须来自实际缓存服务设置。

LMCache 用于 prefix KV 的 CPU/磁盘复用，不能增加 active attention 的 GPU KV 容量。Qwen 混合模型的 opaque pages 必须使用随附 packed4D 修复；原始或未知 layout 的模块会被拒绝。依赖和精确哈希见[安装锁](../extensions/lmcache/install-lock.json)。

本 fork 只验过安装策略和源码迁移，没有 GPU 缓存回读验收。上游历史报告保留在[LMCache 验证记录](validation/v0.1.6-lmcache.md)，不作为本 fork 的性能或正确性保证。保持独立 opt-in；当前不推荐与 speculative decoding、GPU memory sleep 或原生 CPU KV connector 叠加。
