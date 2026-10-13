# 原生配置

环境变量由 `bash start.sh` 读取。不存在容器内/外路径区别，`MODEL` 的本地目录使用真实宿主路径。

| 变量 | 默认 | 含义 |
|---|---|---|
| QWEN_SM75_HOME | HOME/.local/share/qwen-3.8-27b-vllm-sm75 | 专用 Python/CUDA/cache/state 目录 |
| MODEL | Qwen/Qwen3.8-27B-FP8 | ModelScope ID 或真实完整 checkpoint 目录 |
| MODEL_PROVIDER | modelscope | 可选 huggingface |
| MODEL_REVISION | 未指定 | 可固定平台 revision |
| TP_SIZE | 4 | 1/2/4/8，需满足模型维度与显存 |
| GPUS | all | 可设 device=0,1,2,3 或 UUID，原生转换为 CUDA_VISIBLE_DEVICES |
| HOST / PORT | 0.0.0.0 / 8000 | API监听地址/端口 |
| SERVED_MODEL_NAME | VLLM-Qwen3.8-27B | API模型名 |
| MAX_MODEL_LEN | 32768 | 目标，未做容量验收 |
| MAX_NUM_SEQS / MAX_NUM_BATCHED_TOKENS | 1 / 512 | 单请求、小块 prefill |
| CPU_KV_GIB | 0 | 整个TP组CPU prefix offload预算 |
| GPU_KV_BYTES | 未指定 | 每卡预算，默认自动 |
| NATIVE_MEMORY_MAX | 6G | systemd cgroup memory上限 |
| AUTO_SLEEP_IDLE_TIMEOUT | 30 | 分钟，0禁用；原生入口使用exit |
| COMM_BACKEND | nccl | firefly可选，有损FP8 reductions |
| P2P_MODE | auto | 可选off，同时约束NCCL/Firefly |
| VLLM_API_KEY | 自动生成 | 可显式提供，否则private/api-key权限600 |

附加参数：`bash start.sh -- --enforce-eager`。API身份、host/port和TP必须用环境变量配置，不允许附加参数覆盖就绪检查使用的值。默认首轮base模型；MTP/DFlash2等高级组合须自行显式配置匹配模型，尚未在原生环境验收。

`bash start.sh --dry-run` 完全不安装、不写凭据、不查询GPU。`--setup-only` 实际准备依赖，但不启动服务；本轮只验证它的代码，没有执行安装。
