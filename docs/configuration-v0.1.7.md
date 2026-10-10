# v0.1.7 模型配置与启动

v0.1.7 使用同一套推理实现运行 Qwen3.8 27B 和 Flash Next。标准镜像提供推理 API 与引擎监控，Ultra 增加完整管理面板。模型权重、推理参数、量化、MTP/DFlash2/DSpark 和工具解析支持共用。

对应 `v0.1.7-beta2` 预发布源码；版本摘要见[发布说明](releases/v0.1.7-beta2.md)，测试结果见[性能更新](flash-next-performance.md)与[整合验收](validation/v0.1.7-unified.md)。

## 模型模板

模板来源为 `ultra/source/profiles/*.json`，构建时同时装入公共运行核心和 Ultra。镜像后的完整 vLLM 参数仍然可用，未知参数交由 vLLM 校验。

| 模板 ID | 模型 | 主要参数 |
|---|---|---|
| `qwen38-27b-fp8-dspark5` | 27B FP8＋DSpark | TP4、k5、FP8 KV、256K、seq4、batch2048 |
| `qwen38-27b-nvfp4-dspark5-256k` | 27B NVFP4＋DSpark | TP4、k5、FP8 KV、256K、seq4、每卡 GPU KV 5 GiB |
| `qwen38-27b-fp8` | 27B FP8 | TP4、seq4、batch8192、FP8 KV |
| `qwen38-27b-fp8-mtp5` | 27B FP8＋MTP | 同上，MTP5、capture[6] |
| `qwen38-27b-fp8-dflash2` | 27B FP8＋DFlash2 | draft7、TP4、capture[8] |
| `qwen38-27b-awq` | 27B W4A16 AWQ | TP4、seq8、batch16384、FP8 KV |
| `qwen38-27b-awq-mtp5` | 27B AWQ＋MTP | 同上，MTP5、capture[6] |
| `qwen38-27b-awq-dflash2` | 27B AWQ＋DFlash2 | draft7、TP4、capture[8] |
| `flash-next-tp8-256k-nomtp` | Flash Next真正无MTP | TP8＋EP、QSA、FP8 KV、256K、seq4、batch4096 |
| `flash-next-tp8-256k-mtp3` | Flash Next W4A16 / FP8 PLE | TP8＋EP、QSA、FP8 KV、MTP3、256K、seq4、batch4096 |

27B FP8 使用每卡 3,288,334,336 bytes GPU KV、整组 8 GiB CPU KV；AWQ 使用每卡 4 GiB GPU KV、整组 8 GiB CPU KV。模板只作用于选定模型，不把 Next 的 PLE/EP/QSA 参数作为所有模型的默认参数。

## Flash Next 推荐启动

完整的构建、固定模型及MTP下载、1615面板、API Key、参数说明和实测性能复核见 [Flash Next TP8完整使用说明](flash-next-tp8.md)。

按 8×Tesla T10 16 GiB、128 GiB 主机内存准备。上下文为单路总长 262144，允许并发4混合长度；KV为活动请求共享池。主机内存128 GiB以上可增加容器内存余量；增加内存上限不会自动提高GPU KV容量。

准备现有主模型，以及已验证的 W4A16 MTP 草稿。下面使用主模型目录下的 `runtime/mtp-int4-g32`；若草稿放在其他目录，修改对应模型路径。主模型原文件只读挂载。

```bash
export MODEL_ROOT=/srv/vllm/models
export DATA_ROOT=/srv/vllm-sm75
export VLLM_API_KEY='replace-with-your-own-long-random-key'
export NVAPI_LIB=/usr/lib/x86_64-linux-gnu/libnvidia-api.so.1
mkdir -p "$DATA_ROOT/console" "$DATA_ROOT/cache" "$DATA_ROOT/home" "$DATA_ROOT/workspace"

docker run -d --name vllm-sm75-ultra \
  --gpus all --memory 112g --memory-swap 112g --shm-size 16g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -p 8000:8000 -p 1615:1615 \
  -e VLLM_API_KEY -e OMP_NUM_THREADS=1 \
  -e POWER_MODE=pstate -e PSTATE_IDLE_TIMEOUT=86400 \
  -e PSTATE_GPUS=0,1,2,3,4,5,6,7 -e VLLM_MONITOR=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_P2P_DISABLE=0 -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN \
  -e VLLM_SM75_QWEN38_HC_GEMV=1 \
  -e VLLM_SM75_QSA_MIXED_PREFILL=1 -e VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET=0 \
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=16 \
  -e VLLM_SM75_QSA_FLASHINFER_PREFILL=1 -e VLLM_SM75_QSA_SHARED_TARGET_SCRATCH=1 \
  -e VLLM_SM75_QSA_CUBLAS_INDEXER=1 -e VLLM_SM75_QSA_SELECTED_PAGES_PREFILL=1 \
  -e VLLM_SM75_QSA_FP8_MAIN_KV=1 -e VLLM_ENFORCE_STRICT_TOOL_CALLING=1 \
  -v "$MODEL_ROOT:/models:ro" -v "$DATA_ROOT/console:/data" \
  -v "$DATA_ROOT/cache:/data/cache" -v "$DATA_ROOT/home:/dsh/home" \
  -v "$DATA_ROOT/workspace:/dsh/workspace" \
  -v "$NVAPI_LIB:/usr/local/nvidia/lib64/libnvidia-api.so.1:ro" \
  vllm-sm75:v0.1.7-ultra-beta2 \
  /models/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --served-model-name Flash-Next-TP8 \
  --dtype float16 --hf-overrides '{"dtype":"float16"}' \
  --tensor-parallel-size 8 --enable-expert-parallel --enable-ep-weight-filter \
  --language-model-only --max-model-len 262144 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 2415919104 --block-size 16 \
  --enable-prefix-caching --engram-config '{"cpu_offload":true}' \
  --gdn-prefill-backend flashqla_sm75 --generation-config vllm \
  --enable-prompt-tokens-details \
  --compilation-config '{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --scheduler-cls vllm.v1.core.sched.scheduler_sm75.SM75Scheduler \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"model":"/models/Qwen3.8-Flash-Next-W4A16-FP8PLE/runtime/mtp-int4-g32"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template /opt/vllm-sm75/templates/qwen38-next-high-compat.jinja \
  --host 0.0.0.0 --port 8000
```

Ultra 自动启动1615面板，并把命令中的模型参数作为可编辑的运行配置。管理页在 `http://<服务器IP>:1615`；API地址为 `http://<服务器IP>:8000/v1`。已登录管理员可在设置或“API 地址 / Key”中查看、复制Key。

标准运行使用同一组模型参数和内核环境变量，继续使用 `vllm-sm75:v0.1.7-ultra-beta2`，增加 `-e SM75_EDITION=standard`，保留8000端口及模型/缓存挂载；1615、`/dsh/home`和`/dsh/workspace`用于Ultra。标准运行的编译缓存可持久挂载到 `/root/.cache`，电源管理可通过 `POWER_MODE=pstate` 和既有 P-State 参数启用。

也可在镜像后使用 `--profile flash-next-tp8-256k-mtp3`；用 `MODEL`、`DRAFT_MODEL`指定容器内权重目录，后续明确传入的CLI参数覆盖模板的对应选项。Ultra面板中选择该模型，再选同名模板并载入，得到相同推理参数。模板中的离线和Next内核设置只应用到模型进程。

## 27B 使用

27B继续使用原来的模型、草稿、API名称和推理参数。默认API模型名保留 `VLLM-Qwen3.8-27B`。FP8和AWQ分别保留自身的显存、并发和批处理配置，不互相套用。

完整Docker命令可继续逐项传入原参数；使用模板时，镜像后的参数例如：

```bash
# 27B FP8 + MTP5
--profile qwen38-27b-fp8-mtp5

# 27B AWQ + DFlash2，MODEL / DRAFT_MODEL 为容器内路径
--profile qwen38-27b-awq-dflash2
```

主模型与草稿需只读挂载到对应路径。FP8默认模型目录为 `/models/Qwen3.8-27B-FP8`；AWQ为 `/models/Qwen3.8-27B-W4A16-AWQ`；DFlash2为 `/models/Qwen3.8-27B-DFlash2`。MTP使用主模型保留的MTP权重。Ultra模型库中选择对应主模型，再载入FP8/AWQ及普通/MTP/DFlash2模板即可；草稿路径可在加速设置中调整。

## 路径与持久化

| 容器路径 | 内容 |
|---|---|
| `/models` | 现有模型与草稿，只读 |
| `/data` | Ultra配置、登录信息、API Key、运行记录 |
| `/data/cache` | Ultra编译缓存，按权重格式隔离 |
| `/root/.cache` | 标准CLI默认缓存位置，可持久化 |
| `/dsh/home`、`/dsh/workspace` | Ultra工作台持久数据 |
| `/opt/vllm-sm75/profiles` | 两种运行方式共用的内置模型模板 |
| `/opt/vllm-sm75/templates` | 镜像内置的Next对话兼容模板 |
| `/opt/vllm-sm75/next-manifest.json` | 已安装Next兼容与优化文件的校验清单 |

首次启动需要生成匹配当前GPU/参数的编译缓存。保留挂载后可复用；测试吞吐与首次模型加载、编译耗时分开记录。

## 复核吞吐

`tools/benchmark_v017.py`沿用本次实测的冷KV及精确token-ID计时方法。通过`VLLM_BASE_URL`、`VLLM_MODEL`、`VLLM_API_KEY`指定自己的空闲测试服务；输出目录由`VLLM_BENCH_ROOT`设置。脚本不包含服务器地址或凭据。

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000
export VLLM_MODEL=Flash-Next-TP8
export VLLM_BENCH_ROOT=./benchmark-results
python3 tools/benchmark_v017.py --label next-check --cases 4k,8k --repeats 3 --output 512
```

对照必须使用同一模型、配置和输入；脚本会核对usage、token IDs、请求计数、缓存命中和抢占。该脚本的Decode首末token计时与控制台完整流阶段口径分别记录，不能直接混为一个指标。

## 27B DSpark 与 CPU KV

```bash
export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
export MODEL_ROOT=/srv/models
export SM75_PROFILE=qwen38-27b-fp8-dspark5
export MODEL=/models/Qwen3.8-27B-FP8
export DRAFT_MODEL=/models/Qwen3.8-27B-DSpark
export CPU_KV_GIB=16 SHM_SIZE=32g
EDITION=ultra bash docker/run.sh
```

NVFP4 改用 `qwen38-27b-nvfp4-dspark5-256k` 模板和对应 `MODEL` 路径。主模型 KV 为 FP8，草稿 KV 为 auto；模板关闭自适应验证。`CPU_KV_GIB` 为整组 TP 合计，`SHM_SIZE` 必须为 CPU KV 和进程通信保留空间。

仅 API 运行可以直接使用同一个 Ultra 镜像：

```bash
docker run -d --name vllm-sm75-api --gpus all --shm-size 32g \
  -p 8000:8000 -e SM75_EDITION=standard -e SM75_PROFILE \
  -e MODEL -e DRAFT_MODEL -e CPU_KV_GIB=16 -e VLLM_API_KEY \
  -v "$MODEL_ROOT:/models:ro" \
  vllm-sm75:v0.1.7-ultra-beta2
```

P-State 设置、NVAPI 挂载和原有参数见前面的电源及启动说明。BenchLocal 单镜像部署见 [模型测试](benchlocal.md)。
