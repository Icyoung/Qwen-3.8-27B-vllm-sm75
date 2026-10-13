# Qwen-3.8-27B-vllm-sm75

面向四张 Tesla T10 16GiB / SM75 的推理 infra。基于 [fishensw/VLLM-SM75](https://github.com/fishensw/VLLM-SM75/tree/ecee5c70ee4e8cbd915edc49340d07fa5f5545ad)，保留 OpenAI API、SM75 GDN、FP8/AWQ、FP8 KV、MTP/DFlash2、自动休眠和引擎监控。仅提供标准版，已移除 Ultra 控制台、Harness、Node/npm 依赖及对应构建/启动入口。Apache-2.0 和第三方许可证保留。

**当前是源代码候选：本 fork 未构建镜像、启动模型或测量 P2P/吞吐。** 上游历史报告保留在 `docs/validation` 和旧版本发布说明，不能作为此 fork 的验收结果。版本 `0.1.6-icy.1`；[改动与验证](docs/releases/v0.1.6-icy.1.md)。

## 一键启动

在已安装 Docker、NVIDIA 驱动和 Container Toolkit 的 Linux x86_64 主机上：

```bash
bash start.sh
# 只查看计划，不访问 Docker/GPU，不生成 key 或下载模型：
bash start.sh --dry-run
```

入口检查 SM75 GPU 和 Docker NVIDIA runtime；自动生成并复用私有 API key，复用已有完整镜像，镜像缺失时按资源门槛构建；随后自动下载缺失的 ModelScope FP8 权重并等待认证后的模型列表可用。它不执行生成请求。默认容器名 `qwen38-27b-sm75`，API 地址 `http://127.0.0.1:8000/v1`。key 位于 `$HOME/.config/qwen-3.8-27b-vllm-sm75/api-key`，权限 600，不在输出中打印；已有 `VLLM_API_KEY` 则优先使用。

NUC9 当前未安装 Docker，不能在现状下直接运行。其 8GB RAM 不满足此入口的源码构建门槛（8GiB 可用 RAM、200GiB 可用空间）；应在其他机器构建并导入完整镜像后复用。脚本不自动安装驱动/Toolkit，也不停止或替换已有不同配置的容器。

`--no-wait` 只提交启动，不能表示就绪；`--wait-timeout 3600` 指定下载/加载阶段等待秒数。重复执行复用配置相同的运行容器；旧容器已停止或配置不同会报错并保留它。`--rebuild` 仅在无同名容器时构建。高级环境配置见[配置说明](docs/configuration-icy.md)。本轮仅做模拟检查，没有实际构建或启动。

## 构建和默认模型

需要 Linux x86_64、Docker/BuildKit；运行需要 NVIDIA 驱动和 Container Toolkit。构建使用固定摘要的官方 vLLM 0.30.0 / CUDA 12.9 基座及 SM75 扩展，默认单任务。完整构建建议使用独立构建机，NUC9 的 8GB RAM 不代表已经验证可构建或加载。

```bash
git clone https://github.com/Icyoung/Qwen-3.8-27B-vllm-sm75.git
cd Qwen-3.8-27B-vllm-sm75
MAX_JOBS=1 bash docker/build.sh

# 可选：只从 ModelScope 下载，不使用 GPU，也不启动模型。
bash docker/download.sh

# 运行时也会自动下载缺失文件，并复用同一个 ModelScope 缓存。
export VLLM_API_KEY='replace-with-your-api-key'
bash docker/run.sh
```

默认模型是官方 [Qwen/Qwen3.8-27B-FP8](https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8)，默认来源 `MODEL_PROVIDER=modelscope`。已只读核实该仓库存在，配置为 FP8、128×128 block；尚未下载其权重。SM75 使用适配后的 Marlin/FP16 计算路径，FP8 checkpoint 不等于 Turing 有原生 FP8 Tensor Core。

下载缓存默认 `$HOME/.cache/vllm-sm75/models/modelscope`；Hugging Face 使用独立子目录。模型、revision、缓存目录和镜像可通过环境变量覆盖；具体见 [配置](docs/configuration-icy.md)。未固定 `MODEL_REVISION` 时使用平台当前默认 revision，应在正式部署前记录实际版本。

## 四卡与小内存默认值

TP4，单请求，32K 上下文目标，prefill chunk 512，FP16 激活、FP8 KV、lazy safetensors 加载；CPU KV offload 默认关闭。容器内存上限 6GiB、禁用额外容器 swap、共享内存上限 512MiB。它们是 NUC9 的候选约束，加载/repack 峰值与 32K 容量尚未验证。MTP/DFlash2 按 `VARIANT` 显式开启。

默认 `POWER_MODE=sleep`，空闲 30 分钟使用引擎原有 `exit` 休眠；可设置 `AUTO_SLEEP_IDLE_TIMEOUT=1` 表示一分钟。`pstate` 仍可选，需要匹配驱动的 NVAPI 库。没有移植临时 llama.cpp 服务，也没有改变驱动、GPU 功耗上限或矿工设置。

## 通信

默认 `COMM_BACKEND=nccl`，保持 FP16 reduction；FP8 权重和 FP8 通信分开配置。默认 `P2P_MODE=auto`，不再强制 NCCL 的 `SYS` 距离策略；NCCL 根据拓扑决定是否使用 P2P。`P2P_MODE=off` 同时禁止 NCCL P2P 并约束 Firefly 使用 SHM。

Firefly FP8 通信是可选的有损路径。其 P2P 选择已改成所有 rank 一致投票，且跳过 payload 检查时仍检查所有 peer，避免部分互通拓扑选错后端。缓冲分块默认由 128MiB 调整为 16MiB，节省小块 prefill 的暂存空间；不同分块可能改变 FP8 舍入。

P2P 的能力、正确性和性能是三件事。本轮没有 GPU payload 测试，也不承诺 Thunderbolt/PEX 拓扑能用 P2P。现有问题和后续优化优先级见 [通信与性能审查](docs/optimization-review.md)。

## 本地验证

```bash
python3 -m unittest discover -s tests -p 'test_release_scripts.py' -v
python3 -m unittest discover -s tests -p 'test_p2p_policy.py' -v
python3 -m unittest discover -s tests -p 'test_build_contract.py' -v
python3 tools/check-release.py
```

这些检查模拟 Docker/驱动能力或解析源码，不启动真实容器/推理。完整构建与 GPU 集成仍待另行授权执行。标准版引擎监控仍在 `:8000/monitor`，沿用引擎鉴权。
