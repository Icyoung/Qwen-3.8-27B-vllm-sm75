# Docker 入口

- `build.sh`：仅标准版 full/fast，版本来自 `VERSION`。
- `run.sh`：默认 ModelScope 官方 27B FP8，TP4，小内存候选配置。
- `download.sh`：复用推理的 ModelScope 缓存，仅下载，不访问 GPU。
- `Dockerfile.lmcache`：显式选择的独立缓存扩展。

详见[构建](BUILD.md)、[配置](../docs/configuration-icy.md)和[优化审查](../docs/optimization-review.md)。历史 upstream 验证记录不代表此 fork 已通过 GPU 验收。
