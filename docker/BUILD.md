# 标准版构建与启动

本 fork 只提供标准版。全量构建 `bash docker/build.sh`；开发基座 `BUILD_MODE=fast bash docker/build.sh`。fast 是源码迭代基座，不能直接作为正式推理镜像。`EDITION=ultra`、`BUILD_MODE=ui` 会在调用 Docker 前报错。完整依赖的固定版本、ABI/wheel RECORD 校验和 SM75 FlashQLA 构建检查保留。

完整镜像需要较大的磁盘及构建资源；使用独立 Linux 构建机、`MAX_JOBS=1`。生产 Unraid 未隔离完整导出入口继续拦截，原有事件与隔离流程保留在[上游历史记录](../docs/validation/2026-09-18-isolated-full-build.md)。本轮没有实际构建。

```bash
MAX_JOBS=1 bash docker/build.sh
# 只下载默认 ModelScope FP8 模型；没有 --gpus，不运行推理。
bash docker/download.sh
# 以下才会启动模型；本轮未执行。
export VLLM_API_KEY='replace-with-your-api-key'
bash docker/run.sh
```

版本来自 `docker/VERSION`；所有 variant 使用同一标准镜像。默认 ModelScope 模型下载与编译缓存持久化、TP、内存和通信配置见[配置](../docs/configuration-icy.md)。自有完整权重可以只读挂载 `MODEL_ROOT` 并设置 `MODEL=/models/<directory>`。默认 API 端口 8000，监控路径 `/monitor`。

默认 `POWER_MODE=sleep` 使用引擎 exit 休眠。可选 `POWER_MODE=pstate` 保留常驻权重，需要匹配宿主驱动的 `libnvidia-api.so.1`，用 `PSTATE_NVAPI_LIB` 显式指定；NVML 由 Toolkit 的 utility 能力注入。每组 GPU 只能由一个 P-State 管理实例控制。相关机制的历史说明见[休眠与缓存](../docs/sleep-and-cache.md)，以本 fork 的默认值为准。

源码归档：先提交修改，然后 `python3 tools/package-full-build.py /path/to/source.tar`。生成标准版的 `Dockerfile.full`，最终目标 `final`；不包含 Node、Harness 或额外 Web 工作台。归档 SHA 与 source revision 保存到旁边的 JSON。

可选 LMCache 扩展移到 `extensions/lmcache`，独立构建 `docker/Dockerfile.lmcache`，默认不安装、不启用。原有 packed4D 修复、精确哈希及许可证保留。见[LMCache](../docs/lmcache.md)。

本地命令构造检查通过不等于镜像已构建或 GPU 上能运行；检查记录见[本 fork 验证](../docs/releases/v0.1.6-icy.1.md)。
