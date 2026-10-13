# Icy 标准版配置

所有设置都是 `docker/run.sh` 的环境变量；默认来源 ModelScope，完整下载由 vLLM 自动完成。`docker/download.sh` 提前下载同一模型/cache，不需要 API key 或 GPU。

| 变量 | 默认 | 含义 |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.8-27B-FP8` | ModelScope ID 或容器内完整 checkpoint 目录 |
| `MODEL_PROVIDER` | `modelscope` | 可选 `huggingface`；下载脚本仅支持 ModelScope |
| `MODEL_REVISION` | 未指定 | 可指定平台支持的固定 revision；下载脚本缺省 master |
| `VLLM_SM75_CACHE_ROOT` | `$HOME/.cache/vllm-sm75` | 编译缓存父目录 |
| `VLLM_SM75_MODEL_CACHE_ROOT` | `<cache>/models` | 模型父目录；modelscope/huggingface 分开 |
| `TP_SIZE` | `4` | 基础/MTP 的 TP；DFlash2 本版要求 TP4 |
| `GPUS` | `all` | Docker `--gpus` 的单个参数；选四卡可设 `GPUS='"device=0,1,2,3"'` |
| `CONTAINER_MEMORY` | `6g` | memory 和 memory-swap 相同，限制总内存并禁用额外 swap |
| `SHM_SIZE` | `512m` | tmpfs 容量上限，实际驻留计入容器 memory |
| `MAX_MODEL_LEN` | `32768` | base/MTP 目标；DFlash2 保留上游 262K 候选，可覆盖 |
| `MAX_NUM_SEQS` | `1` | FP8 基线；AWQ 保留上游 seq8 |
| `MAX_NUM_BATCHED_TOKENS` | `512` | FP8 基线；AWQ 保留上游 batch16384 |
| `CPU_KV_GIB` | `0` | 整个 TP 组的 prefix offload 缓存；不会扩大 active GPU KV |
| `GPU_KV_BYTES` | `0` | 0 使用自动预算，非零为每卡预算；DFlash2 有上游专用默认 |
| `POWER_MODE` | `sleep` | 可选 pstate，需匹配 NVAPI 驱动库 |
| `AUTO_SLEEP_IDLE_TIMEOUT` | `30` | 分钟，0 禁用 |
| `AUTO_SLEEP_OFFLOAD_TARGET` | `exit` | 退出引擎/worker；新请求重建。其他模式仍是显式选项 |
| `COMM_BACKEND` | `nccl` | 可选 firefly/auto，启用有损 FP8 reductions |
| `P2P_MODE` | `auto` | off 禁用 NCCL P2P 并将 Firefly 限定 SHM；on 只允许尝试，不创造硬件能力 |
| `NCCL_P2P_LEVEL` | 未设置 | 可显式 PIX/PXB/PHB/SYS 等，默认由 NCCL 按实际拓扑选择 |
| `VLLM_FIREFLY_AR_CHUNK_BYTES` | `16777216` | 可选 Firefly 分块；1–128MiB，4096 对齐 |

显式低层 `VLLM_FIREFLY_AR`、`VLLM_ALLREDUCE_USE_FLASHINFER` 可以覆盖通信候选。`NCCL_P2P_DISABLE=1` 也会约束 Firefly 使用 SHM；与显式 p2p 冲突时报错。custom all-reduce 保持禁用。

```bash
# 仅展示配置；本 fork 未启动以下模型。
CPU_KV_GIB=0 COMM_BACKEND=nccl P2P_MODE=off bash docker/run.sh
# 可选有损压缩通信，auto 仅在所有 rank 检查一致时选 P2P。
COMM_BACKEND=firefly P2P_MODE=auto bash docker/run.sh
# 已有 NAS/SSD 权重目录；宿主路径由用户选择。
MODEL_ROOT=/srv/models MODEL=/models/Qwen3.8-27B-FP8 bash docker/run.sh
```

NUC9 的 Thunderbolt/PEX 拓扑未完成 P2P payload 资格验证，不能沿用 Z4 的结论。32K、6GiB 启动预算和 FP8 checkpoint 的全模型数值质量都需后续 GPU 验收。这轮按用户要求只改源码。

## 一键入口

`bash start.sh` 使用上述环境变量，额外支持 `QWEN_SM75_CONFIG_DIR`（API key 的私有目录）。`VLLM_API_KEY` 优先于自动生成的 key；默认生成文件权限 600。默认容器名 `qwen38-27b-sm75`，入口会保存配置指纹作为容器 label，重复执行可复用相同运行容器，配置不同或已停止时保留现场并报错。

附加原生参数可用 `bash start.sh -- --enforce-eager`。API key、host/port、served model 和 TP 通过环境变量配置，禁止用附加参数覆盖，以保证就绪检查和配置匹配正确。默认等待模型列表，未做生成验收。`--dry-run` 不读取/生成凭据或访问 Docker/GPU。源码构建的主机资源预算独立于推理容器的 6GiB 限额；NUC9 应复用外部构建的完整镜像。
