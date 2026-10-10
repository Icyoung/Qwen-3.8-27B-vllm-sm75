# Flash Next TP8 完整使用说明

适用版本：`vllm-sm75:v0.1.7-ultra-beta2`。目标模型：`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`。本说明更新至2026-10-04的8×Tesla T10实测配置，包含构建、模型准备、1615面板及API启动、参数、工具调用和性能复核。

推荐组合是 **TP8＋EP、QSA开启、FP8 KV、MTP3、单路256K、并发4混合长度**。基础运行与Ultra使用相同推理核心；下面以带完整面板的Ultra为主。

本次无MTP与混合QSA优化、两个同镜像模板及复测口径见[性能更新](flash-next-performance.md)。

## 1. 实测性能与适用硬件

| 模式 | 输入＋输出 tokens | Prefill tok/s | Decode tok/s | 取样 |
|---|---|---:|---:|---|
| 无MTP | 8,192＋512 | 3,225.82 | 49.93 | 2次完整预热，3次正式取中位数 |
| 无MTP | 32,768＋512 | 2,835.11 | 49.97 | 1次正式 |
| 无MTP | 261,632＋512 | 2,218.50 | 51.05 | 1次正式，输入输出合计262,144 |
| MTP3 | 8,192＋512 | 3,127.89 | 69.86 | 2次完整预热，3次正式取中位数 |
| MTP3 | 32,768＋512 | 2,726.28 | 65.58 | 1次正式 |
| MTP3 | 261,632＋512 | 2,108.20 | 65.08 | 1次正式，输入输出合计262,144 |

这是冷前缀缓存、固定工程文本任务的结果。MTP接受率随内容变化，不能保证任意聊天都达到同一Decode数值。32K/256K数据各为单点；最新混合A/B见[性能更新](flash-next-performance.md)，10月3日的128K和整合对照保留在[整合验收](validation/v0.1.7-unified.md)。此前约44 tok/s来自2026-09-24 v0.1.6关闭QSA/跳过indexer的无MTP配置（FP8 KV、block32、热前缀）；不能当成当前QSA配置的无MTP结果。

| 项目 | 复现条件 |
|---|---|
| GPU | 8×Tesla T10，每卡16 GiB，SM75；八卡独占 |
| CPU / RAM | 实测AMD EPYC 7B13、约125 GiB可见内存；按128 GiB硬件配置 |
| 驱动 | 实测580.178.04；此处记录测试版本，不作为最低版本声明 |
| 宿主 | Linux x86_64、Docker、NVIDIA Container Toolkit、Bash |
| 容器预算 | `--memory 112g --memory-swap 112g --shm-size 16g`；保留宿主余量 |
| GPU KV | 每卡2,304 MiB，`2415919104` bytes，FP8 E4M3 |
| 测得KV容量 | MTP3为302,704 tokens；无MTP为339,110 tokens；所有活动请求共享 |
| 测得显存 | MTP3最高15,438 MiB/卡、最少空闲493 MiB；无MTP混合后段采样最高14,678 MiB/卡、最少空闲1,253 MiB |

**256K是单请求输入、对话模板、思考和输出的总长度。** 并发4表示可以同时处理4个请求，验收负载为1K/8K/16K/32K混合、各输出256。暖内核冷前缀A/B中，无MTP总端到端吞吐23.21→39.82 tok/s，MTP3为22.97→39.06 tok/s；不代表4路各自占满256K。长请求要给输出预留空间，例如261,632输入＋512输出。

128 GiB以上主机可以按实际空闲内存提高容器上限，并继续给宿主和其他服务留余量。增加RAM不会自动增加GPU KV容量。实测cgroup内存峰值触及112 GiB，包含可回收文件页，OOM kill为0；这不等于匿名内存必须占满112 GiB。

## 2. 保存、构建与使用唯一正式镜像

本项目统一使用`vllm-sm75:v0.1.7-ultra-beta2`。它包含Next、27B FP8/AWQ、MTP/DFlash2及完整控制台，运行时无需保留opt1、opt2、RC或其他Next专用镜像。

使用本次beta源码：

```bash
git clone --branch v0.1.7-beta2 --single-branch https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
export SOURCE_ROOT="$PWD"
```

已有正式Ultra时，可从本次源码一次重建，复用已核验的CUDA/Torch/FlashQLA/Node/Harness依赖；无需重编基础CUDA环境。下面在同时包含`docker/`、`ultra/`、`vllm/`的源码根目录执行：

```bash
cd "$SOURCE_ROOT"
docker image inspect vllm-sm75:v0.1.7-ultra-beta2 --format '{{.Id}}'
docker build -f docker/Dockerfile.ultra \
  --build-arg BASE_IMAGE=vllm-sm75:v0.1.7-ultra-beta2 \
  -t vllm-sm75:v0.1.7-ultra-beta2 .
```

该入口重新安装统一推理适配、Next优化和面板源码，并执行安装校验，最终只产生一个正式标签。基础镜像是构建输入；构建完成的镜像包含运行所需文件，不在启动时查找opt2或其他镜像。

构建不会重启现有容器；容器仍指向创建时的image ID。更新运行实例时需在无请求窗口重建容器，并沿用原模型、配置、缓存、工作台、端口和凭据挂载。

保存和迁移只需要这一个镜像：

```bash
docker save -o vllm-sm75-v0.1.7-ultra-beta2.tar vllm-sm75:v0.1.7-ultra-beta2
# 在另一台兼容主机上导入：
docker load -i vllm-sm75-v0.1.7-ultra-beta2.tar
```

镜像备份文件较大，需要单独预留磁盘。源码、模型和持久化配置分别保存。首次完全从官方基础环境构建仍可参考[完整构建文档](../docker/BUILD.md)；本节为复用已有正式Ultra的单镜像重建入口。

## 3. 主模型与MTP权重

使用[模型仓库](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE/tree/ef554143369a706525336f6b42a09094835dc077)的固定修订`ef554143369a706525336f6b42a09094835dc077`。已核对该修订的27个主模型/MTP权重文件，以及7个配置、索引和tokenizer文件，34项SHA256均与本轮实测材料相同。

主模型是W4A16；FP8 PLE指PLE表。MTP使用仓库已经提供的独立INT4草稿，**不需要重新量化**。应具备以下目录：

```text
/data/models/Qwen3.8-Flash-Next-W4A16-FP8PLE/
├── config.json
├── tokenizer.json / tokenizer_config.json / chat_template.jinja
├── model.safetensors.index.json
├── 主模型25个权重文件，包含FP8 PLE分片
└── runtime/mtp-int4-g32/
    ├── config.json
    ├── model.safetensors.index.json
    ├── mtp-dense.safetensors
    └── mtp-routed-experts-int4.safetensors
```

主模型权重124,781,270,509 bytes，MTP权重4,140,094,152 bytes，合计约120 GiB，另需下载临时空间。已有完整模型先校验即可。

```bash
export IMAGE=vllm-sm75:v0.1.7-ultra-beta2
export MODEL_ROOT=/data/models
export MODEL_NAME=Qwen3.8-Flash-Next-W4A16-FP8PLE
export DATA_ROOT=/data/vllm-sm-next/flash-next-v017-20261003
mkdir -p "$DATA_ROOT/download-cache"

# 仅在模型尚未完整准备时执行。该命令包含runtime/mtp-int4-g32。
docker run --rm --entrypoint hf \
  -v "$MODEL_ROOT:/models" \
  -v "$DATA_ROOT/download-cache:/root/.cache/huggingface" \
  "$IMAGE" download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --revision ef554143369a706525336f6b42a09094835dc077 \
  --local-dir "/models/$MODEL_NAME"

# 校验与实测相关的全部权重、配置、索引及tokenizer，耗时取决于磁盘速度。
( cd "$MODEL_ROOT/$MODEL_NAME" && \
  sha256sum -c "$SOURCE_ROOT/docs/models/flash-next-tested.sha256" )
```

需要HF镜像站时，在上述下载容器参数中增加`-e HF_ENDPOINT=https://hf-mirror.com`，仍固定同一修订并运行相同校验。模型仓库自带的旧启动脚本和运行时补丁无需执行；本镜像已安装本次验收的实现。

## 4. 推荐启动：完整参数、API和1615面板

下面的命令已包含MTP3、QSA、B9对应的PLE放置、FP8 KV、工具解析和面板端口。先确认八卡没有其他推理或计算任务；查看`nvidia-smi`和`docker ps`，仅处理实际占卡的服务。

准备独立持久化目录和API Key。首次生成后保留此文件，重启会继续使用同一个Key。

```bash
export IMAGE=vllm-sm75:v0.1.7-ultra-beta2
export MODEL_ROOT=/data/models
export DATA_ROOT=/data/vllm-sm-next/flash-next-v017-20261003
export NVAPI_LIB=/usr/lib/x86_64-linux-gnu/libnvidia-api.so.1
# NVAPI_LIB须改为宿主实际路径；部分发行版为/usr/lib64/libnvidia-api.so.1。
test -f "$NVAPI_LIB"
mkdir -p "$DATA_ROOT/console" "$DATA_ROOT/cache" "$DATA_ROOT/home" \
  "$DATA_ROOT/workspace" "$DATA_ROOT/benchmark"
install -d -m 700 "$DATA_ROOT/secrets"
if [ ! -s "$DATA_ROOT/secrets/api.env" ]; then
  ( umask 077; printf 'VLLM_API_KEY=%s\n' "$(openssl rand -hex 32)" > "$DATA_ROOT/secrets/api.env" )
fi

docker run -d --name flash-next-tp8 \
  --gpus all --memory 112g --memory-swap 112g --shm-size 16g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -p 8000:8000 -p 1615:1615 \
  --env-file "$DATA_ROOT/secrets/api.env" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e SM75_CONSOLE_ROOT=/data -e OMP_NUM_THREADS=1 \
  -e POWER_MODE=pstate -e PSTATE_IDLE_TIMEOUT=86400 \
  -e PSTATE_GPUS=0,1,2,3,4,5,6,7 -e VLLM_MONITOR=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_P2P_DISABLE=0 -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN \
  -e VLLM_SM75_QWEN38_HC_GEMV=1 \
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=16 \
  -e VLLM_SM75_QSA_FLASHINFER_PREFILL=1 -e VLLM_SM75_QSA_SHARED_TARGET_SCRATCH=1 \
  -e VLLM_SM75_QSA_CUBLAS_INDEXER=1 -e VLLM_SM75_QSA_SELECTED_PAGES_PREFILL=1 \
  -e VLLM_SM75_QSA_FP8_MAIN_KV=1 -e VLLM_ENFORCE_STRICT_TOOL_CALLING=1 \
  -e VLLM_SM75_QSA_MIXED_PREFILL=1 -e VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET=0 \
  -e VLLM_CACHE_ROOT=/data/cache/awq/vllm \
  -e TRITON_CACHE_DIR=/data/cache/awq/triton \
  -e TORCH_EXTENSIONS_DIR=/data/cache/shared/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/data/cache/awq/inductor \
  -e CUDA_CACHE_PATH=/data/cache/shared/cuda \
  -e FLASHINFER_WORKSPACE_BASE=/data/cache/shared/flashinfer-home \
  -v "$MODEL_ROOT:/models:ro" -v "$DATA_ROOT/console:/data" \
  -v "$DATA_ROOT/cache:/data/cache" -v "$DATA_ROOT/home:/dsh/home" \
  -v "$DATA_ROOT/workspace:/dsh/workspace" \
  -v "$NVAPI_LIB:/usr/local/nvidia/lib64/libnvidia-api.so.1:ro" \
  "$IMAGE" \
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

Ultra会自动启动1615面板，将上述模型参数导入“Docker 启动参数”运行配置，再自动加载模型。无需再运行第二条面板启动命令。可在面板编辑、保存；下次容器启动时，明确写在Docker命令和环境中的值会重新导入，长期修改也应同步到部署命令。

`PSTATE_IDLE_TIMEOUT=86400`沿用测速配置，避免测试间隔频繁降频；部署后可按需要在电源设置中调小。P-State需要挂载与宿主驱动匹配的NVAPI；缺库处理见[驱动库准备](../docker/BUILD.md#linux-驱动库准备)。该方式不卸载模型，不启用模型自动休眠。

### 面板登录、API地址与Key

- 管理面板：`http://<服务器IP>:1615`。
- OpenAI兼容API Base URL：`http://<服务器IP>:8000/v1`。
- 模型名称：`Flash-Next-TP8`。

首次获取面板登录token：

```bash
docker exec flash-next-tp8 node /opt/sm75-workbench/console/auth-cli.mjs show
```

登录token与API Key独立。登录后可在面板“API 地址 / Key”或设置中查看、复制API Key，日常无需命令行读取。此示例用`secrets/api.env`固定API Key；若在面板重置Key，也要更新部署用文件，防止容器重启导入旧值。保留`console`目录即可保留登录和运行配置。

### 使用同源模板或仅运行API

上述完整参数也可由内置模板展开：保留Docker资源、端口、Key、缓存和目录挂载，在镜像之前增加：

```bash
-e MODEL=/models/Qwen3.8-Flash-Next-W4A16-FP8PLE \
-e DRAFT_MODEL=/models/Qwen3.8-Flash-Next-W4A16-FP8PLE/runtime/mtp-int4-g32
```

将镜像之后的模型路径和整段vLLM参数替换成：

```bash
--profile flash-next-tp8-256k-mtp3
```

该模板和Ultra中“Flash Next · TP8 · QSA · FP8 KV · MTP3 · 256K”使用同一份JSON。后接明确CLI参数会覆盖模板同名项。面板操作为：登记`/models/...`主模型 → 选择该模板 → 核对MTP草稿路径 → 保存并启动。完整参数命令已启动模型时，无需再启动另一个配置。

只要推理API时，仍使用`vllm-sm75:v0.1.7-ultra-beta2`，在镜像名称前增加`-e SM75_EDITION=standard`；保留相同模型参数、内核环境、API Key、GPU、缓存、P-State和NVAPI设置，去掉1615端口及`/dsh/home`、`/dsh/workspace`挂载。该模式直接启动vLLM，使用同一份推理代码和模板，无需第二个基础版镜像。

## 5. 参数说明与权重分布

| 参数/设置 | 本次值 | 作用 |
|---|---|---|
| TP / EP / weight filter | 8 / 开启 / 开启 | 八卡分布及按EP加载专家权重 |
| 模型dtype / HF覆盖 | `float16` / `{"dtype":"float16"}` | 使用已验证的SM75 FP16执行路径 |
| `engram-config` | `{"cpu_offload":true}` | PLE embedding表在CPU，key/value投影留GPU，保留B9收益 |
| HC GEMV | `VLLM_SM75_QWEN38_HC_GEMV=1` | 保留H1的小批量HC投影路径 |
| QSA主KV | `fp8_e4m3`＋`VLLM_SM75_QSA_FP8_MAIN_KV=1` | 主KV使用FP8；indexer缓存仍为FP16 |
| QSA Prefill | `FLASHINFER_PREFILL=1`、`SELECTED_PAGES_PREFILL=1` | 使用适配后的长Prefill选页执行路径 |
| QSA indexer / scratch | `CUBLAS_INDEXER=1`、`SHARED_TARGET_SCRATCH=1`、logits上限16 MiB | 索引计算及临时显存控制；MTP缓存独立 |
| MTP | `method=mtp`、`num_speculative_tokens=3` | 实测选择的MTP3；草稿路径指向`runtime/mtp-int4-g32` |
| `max-model-len` | `262144` | 单请求总长度256K |
| `max-num-seqs` | `4` | 同时最多4个请求 |
| `max-num-batched-tokens` | `4096` | 每轮Prefill批预算，影响预填充与Decode调度 |
| `kv-cache-memory-bytes` | `2415919104` | 每卡2,304 MiB固定KV预算，不是八卡总和 |
| `gpu-memory-utilization` | `0.92` | 运行显存规划；固定KV字节已显式给定 |
| CUDA Graph | `mode=0`、`PIECEWISE`、`[1,2,4,8,16]` | 保留实测图路径；不改成eager或另一套捕获配置 |
| GDN Prefill | `flashqla_sm75` | SM75线性注意力Prefill后端 |
| Scheduler | `SM75Scheduler` | 本项目调度与投机支持 |
| prefix cache | 开启 | 生产请求可复用前缀；复测用独立salt保证冷缓存 |
| 工具 / 推理解析 | `qwen3_xml` / `qwen3` | OpenAI格式工具调用和思考内容解析 |

表中QSA缩写均指启动命令中的完整`VLLM_SM75_QSA_*`变量。模型按原生QSA配置运行；本配置没有关闭QSA或跳过indexer，也没有另加全局Attention后端覆盖。

当前TP8的实际放置如下：

- **VRAM**：各卡对应的专家分片、attention/shared expert/HC/lm head等运行权重、PLE key/value投影、MTP、KV、workspace和CUDA Graph。
- **RAM**：完整FP8 PLE embedding表按rank分布的份额、CPU运行开销和缓存页。
- **磁盘**：模型文件和持久编译缓存。此TP8配置未启用冷专家RAM搬运，也未把PLE表改成NVMe分页。

不要叠加旧的`--cpu-offload-params ple`；它可能把投影一并卸载。也不要把TP4冷专家缓存配置套入此模板。先按上述固定参数复现，再逐项调整；直接提高batch、MTP数量或KV预算可能改变图形状、接受率及显存余量。

## 6. 启动检查与调用

首次启动会加载模型并编译、捕获CUDA Graph，以健康检查和实际请求成功为准。

```bash
docker logs --tail 120 flash-next-tp8
curl --fail http://127.0.0.1:8000/health
set -a
. "$DATA_ROOT/secrets/api.env"
set +a
curl --fail http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
docker exec flash-next-tp8 nvidia-smi
```

`/v1/models`应返回`Flash-Next-TP8`及262144上限；启动日志应确认8个rank、MTP3、QSA和FP8 KV，KV容量约302704 tokens。容量不同先核对镜像、草稿、dtype及固定KV参数。请求过程中检查八卡频率与负载，空闲P8截图不能用来判断推理性能。

### 普通聊天

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Flash-Next-TP8","messages":[{"role":"user","content":"请用三句话说明什么是前缀缓存。"}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
```

### 工具调用

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Flash-Next-TP8","messages":[{"role":"user","content":"Multiply 17 and 23 using the multiply tool."}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false},"tool_choice":"required","tools":[{"type":"function","function":{"name":"multiply","description":"Multiply two integers.","strict":true,"parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false}}}]}'
```

应得到`message.tool_calls`，函数名`multiply`，参数为`{"a":17,"b":23}`。实际工具由客户端执行，再把结果作为`role=tool`、携带对应`tool_call_id`发回，继续生成最终答复。需要模型自行判断是否调用时使用`tool_choice=auto`。镜像内置对话模板兼容`high`思考级别；启用思考会增加输出工作量，测速需与表中`enable_thinking=false`保持一致。

## 7. 复现Prefill和Decode测试

新增的`tools/benchmark_flash_next.py`沿用最终实测的输入构造及原token-ID计时函数：先做**2次8K输入＋512输出预热**，再做**7次8K正式请求**，最后各做1次128K/256K边界请求。温度0、seed42、自然EOS，每次独立cache salt。保留每次输入hash、usage、token事件、MTP计数、接受长度、缓存命中和抢占信息。

使用镜像内的Python和tokenizer依赖运行测速客户端，无需在宿主安装Torch。这个客户端容器不分配GPU，不加载推理模型；模型仍由`flash-next-tp8`提供。

```bash
export RUN_LABEL=next-repro-01
docker run --rm --network host --entrypoint python3 \
  --env-file "$DATA_ROOT/secrets/api.env" \
  -e VLLM_BASE_URL=http://127.0.0.1:8000 -e VLLM_MODEL=Flash-Next-TP8 \
  -e VLLM_BENCH_ROOT=/bench \
  -v "$SOURCE_ROOT/tools:/tools:ro" -v "$MODEL_ROOT:/models:ro" \
  -v "$DATA_ROOT/benchmark:/bench" \
  "$IMAGE" /tools/benchmark_flash_next.py \
  --model-path /models/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --label "$RUN_LABEL" --sizes 8192,130560,261632 \
  --warmups 2 --repeats 7 --long-repeats 1 --output 512 --mixed
```

最后的`--mixed`另发4个并行请求：输入1,024/8,192/16,384/32,768，各输出256。该组记录每路TTFT/Decode和总端到端吞吐；不把相互重叠的全局Prefill计时分摊成每个请求的纯Prefill时间。

先快速复核8K，可将`--sizes`改为`8192`并移除`--mixed`，其余不变。再次运行换一个`RUN_LABEL`，脚本拒绝覆盖已有结果。测试时保持没有其他客户端请求；若自然提前EOS、未产生MTP草稿计数、缓存非零、token计数不符或发生抢占，脚本会报错，不能把该轮当作通过。

结果保存到：

```text
$DATA_ROOT/benchmark/results/
├── next-repro-01-inputs.json   # 输入token IDs及hash
├── next-repro-01.jsonl         # 单路请求逐条原始结果，含预热标记
└── next-repro-01-job.json      # 阶段进度、正式中位数、混合负载和通过状态
```

### 计时口径

| 指标 | 计算方式 |
|---|---|
| 本文表格Prefill | **完整输入tokens ÷ 服务端Prefill阶段时间** |
| 本文表格Decode | `(输出tokens－首个流事件的tokens) ÷ (末token事件时间－首token事件时间)` |
| 客户端首响应口径 | 完整输入tokens ÷ 客户端TTFT；另存`client_prefill_tok_s` |
| 客户端完整生成阶段 | 全部输出tokens ÷ `(流结束时间－首token事件时间)` |
| 并发4总吞吐 | 四路总输出tokens ÷ 整组端到端耗时 |

Prefill分子包含完整输入，**命中缓存的token仍算有效输入**，分母只取相应阶段；不除以整次生成耗时。冷前缀测试保证命中为0，是为比较模型本身的预填充速度。生产中的缓存、排队、思考、并发和网络都会改变观测速率。MTP一轮可返回多个token，不能将流事件数量当作token数。

表中数据来自10月4日最终镜像的实际GPU复测。可移植测速入口另做离线语法/参数检查；完整原始请求由本轮测试脚本保存。

## 8. 性能不符时按什么检查

| 表现 | 优先核对 |
|---|---|
| 8K Prefill明显低于约3,100 | 是否相同输入/计时口径；GPU频率、其他请求；QSA Prefill/cuBLAS/选页变量；batch4096；首次编译是否结束 |
| Decode停留在无MTP水平 | 草稿路径、MTP3是否生效、草稿/接受计数；是否打开了思考；是否沿用旧PLE整体卸载参数 |
| 同配置Decode有波动 | 输出内容和MTP接受长度、GPU时钟、两次完整预热；按7次中位数比较 |
| 128K/256K比8K慢 | 长上下文索引、读KV与选择页成本增加；对照本表相应长度，不用8K值作目标 |
| OOM或无法提供256K | 是否确为8×16 GiB；主KV为FP8；每卡KV字节数；MTP草稿是否INT4；有无叠加其他卸载/图参数或其他GPU任务 |
| 只有面板、模型没就绪 | 模型加载日志、挂载路径、MTP文件、NVAPI库；Docker容器运行不等于模型已经加载成功 |
| API返回401 | 客户端是否使用API Key；面板登录token不是API Key；检查面板显示的当前Key与部署文件 |
| 工具结果只有普通文本 | `qwen3_xml`解析器、自动工具选择、镜像内对话模板及请求中的`tools`/`tool_choice` |

性能比较期间保持同一镜像、主模型/MTP权重、参数和输入。变更batch、MTP数量、图模式、KV预算时单独标记结果，避免把不同配置的最佳点拼成一套性能承诺。

## 9. 路径与日常使用

| 宿主路径 | 容器路径 | 用途 |
|---|---|---|
| `$MODEL_ROOT` | `/models`，只读 | 主模型、MTP草稿 |
| `$DATA_ROOT/console` | `/data` | 管理员登录、API Key、模型和运行配置、使用记录 |
| `$DATA_ROOT/cache` | `/data/cache` | 持久编译缓存；本配置使用`awq/`及`shared/`子目录 |
| `$DATA_ROOT/home` | `/dsh/home` | Ultra工作台用户数据 |
| `$DATA_ROOT/workspace` | `/dsh/workspace` | Ultra工作台文件 |
| `$DATA_ROOT/secrets/api.env` | Docker读取环境文件 | 固定API Key，不放入发布包 |
| `$DATA_ROOT/benchmark` | 测速客户端的`/bench` | 复测进度与结果 |
| 镜像内置 | `/opt/vllm-sm75/profiles` | 共用模型模板 |
| 镜像内置 | `/opt/vllm-sm75/templates` | Next对话模板 |
| 镜像内置 | `/opt/vllm-sm75/next-manifest.json` | Next安装校验清单 |

日常从面板查看状态、复制API Key、切换或编辑配置。仅需重启同一容器时，确认运行中/等待请求均为0后执行`docker restart flash-next-tp8`；缓存和配置由挂载保留。修改Docker端口、挂载或镜像需要重建容器，先保留原容器及数据作回退，再使用相同持久目录启动新容器；同一组GPU和同一份控制台目录只运行一个实例。

## 10. no-MTP与混合并发更新

分组QSA覆盖混合批次中的Prefill请求，保留单路和Decode执行路径。真正无MTP启动使用`flash-next-tp8-256k-nomtp`模板，MTP3使用`flash-next-tp8-256k-mtp3`模板，两者均在同一正式镜像完成GPU实测。机制、参数及验证边界见[性能更新](flash-next-performance.md)。
