# Qwen-3.8-27B-vllm-sm75

四张 Tesla T10 16GiB / SM75 的原生推理部署，使用独立 Python 3.12 `venv`，不安装或依赖 Docker。基于 [fishensw/VLLM-SM75](https://github.com/fishensw/VLLM-SM75/tree/ecee5c70ee4e8cbd915edc49340d07fa5f5545ad)，仅标准版，Ultra/Harness 已移除。默认从 ModelScope 下载官方 `Qwen/Qwen3.8-27B-FP8`；SM75 使用 Marlin 的 FP8 权重、FP16 激活路径，GDN prefill 使用 FlashQLA-SM75。

**0.1.6-icy.2 是原生源码候选。尚未真实安装、启动模型或测量 GPU 性能；CPU 模拟检查不代表全模型验收。** 上游历史报告保留，不能当成本 fork 的运行结果。

## 零环境部署

支持 Ubuntu 22.04/24.04/26.04 x86_64，联网及 root/sudo 权限。系统需有足够磁盘，默认检查 60GiB 空间；实际安装/加载峰值尚待验证。

```bash
# 已下载仓库后，直接执行：
bash start.sh
# 只查看计划，不安装软件、不访问 GPU、不生成凭据：
bash start.sh --dry-run
# 仅准备环境，不启动模型：
bash start.sh --setup-only
```

脚本按需安装系统 Python/venv 和编译工具，正常的 NVIDIA 驱动直接复用。缺驱动时按 Ubuntu GPGPU 流程安装，随后退出 75，提示手动重启并再次运行同一命令；不会自动重启、升级已有正常驱动或自动修复未知驱动/内核冲突。

其余组件全部安装在项目专用目录，默认 `$HOME/.local/share/qwen-3.8-27b-vllm-sm75`：

- `bootstrap`：只负责安装工具；`venv`：独立 Python 3.12 推理环境。
- `cuda`：独立 CUDA 12.9/GCC13 编译环境，不修改全局 CUDA 软链。
- `cache`：ModelScope/HF、vLLM、Triton 与扩展缓存。
- `private`：权限 600 的 API key；`run/server.json` 为权限 600 的服务配置。

可用 `QWEN_SM75_HOME=/path/to/private-home bash start.sh` 指定新目录。已有但不属于本项目的 Python/CUDA 环境不会被覆盖；同一安装合同的失败步骤可以续装，合同改变时使用新目录。`PYTHONPATH/PYTHONHOME` 和其他环境的 CUDA 库路径不会传入推理服务。

官方 vLLM CUDA 12.9 wheel 和工具下载带固定 SHA-256，核心依赖版本见 [native-lock](runtime/native-lock.json) 和 [constraints](runtime/constraints.txt)。仅编译 SM75 FlashQLA，默认 `MAX_JOBS=1`，无需完整构建镜像。

## 服务与默认配置

启动原生 systemd 服务 `qwen38-27b-sm75`：普通用户使用 user service，root 使用 system service。API 默认 `http://127.0.0.1:8000/v1`。启动等待带认证的模型列表，不执行生成请求。默认 TP4、seq1、32K 目标、prefill chunk512、FP16 激活、FP8 KV、关闭 CPU KV offload，service 内存上限 6GiB、swap 上限0。32K 与内存预算未实测。

空闲 30 分钟采用引擎 exit 休眠，新请求重建。默认通信 NCCL FP16 reduction；Firefly FP8 通信独立 opt-in，有损且未在本 fork 验收。P2P auto 按完整 rank 检查，off 同时约束 NCCL/Firefly。

```bash
# 普通用户；root 去掉 --user
journalctl --user -u qwen38-27b-sm75 -f
systemctl --user stop qwen38-27b-sm75
```

不会自动 enable 开机启动或修改 linger，也不会停止其他模型。重复执行复用相同配置的运行服务，配置不同的活动服务保留并报错。更多见 [原生部署](docs/native-deployment.md)、[配置](docs/configuration-icy.md)和 [优化审查](docs/optimization-review.md)。Apache-2.0 及第三方许可证保留。
