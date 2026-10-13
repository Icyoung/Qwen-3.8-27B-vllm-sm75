# 原生部署

入口 `start.sh` → `bootstrap.sh` → `tools/install-native.py` → `tools/start.py`。不依赖 Docker。支持 Ubuntu x86_64，其他发行版可手动准备系统依赖，但自动 bootstrap 当前不支持。

首次运行按需补齐 apt 系统工具；驱动正常则保留。缺驱动用 `ubuntu-drivers install --gpgpu`，之后退出75等待用户手动重启。重启要选正确系统盘，再运行同一命令。Ubuntu 的 Secure Boot/模块签名和非标准内核问题需按实际驱动流程处理，脚本不会自动关闭固件安全设置。

私有 bootstrap venv 安装 uv0.13.0，管理 Python3.12；推理 venv 与编译器前缀分开。vLLM0.30.0 cu129 wheel 固定官方发布 SHA；PyTorch2.13.0+cu129、Triton3.7.1、Transformers5.15.1、FlashInfer0.6.18。CUDA12/13 共享库冲突按原补丁修复并核对 wheel RECORD。编译器为 NVIDIA CUDA12.9 与 GCC13，micromamba 下载固定 SHA，不使用全局 conda 环境。

网络中断后保留下载和 `install-progress.json`，同源码合同从失败步骤继续。已有未受管环境、不同合同或缺失的已完成环境不会被静默覆盖；指定新 `QWEN_SM75_HOME` 是更新路径，原目录保留用于回退。软件安装与模型质量验收是不同步骤。

原生 systemd unit 仅启动本项目服务；使用隔离模式 `python -I`，API key 从权限600的配置加载，主进程命令行不含 key。父子进程在同一 service cgroup，MemoryMax默认6G、MemorySwapMax0。没有自动 restart/enable/linger。非root需正常的用户 systemd session；若没有用户总线，先正常登录或用root独立服务部署，不自动操作别人的会话。

日志：`journalctl --user -u qwen38-27b-sm75 -f`；停止：`systemctl --user stop qwen38-27b-sm75`。root 不加 `--user`。使用 `--no-wait` 时仅提交启动；默认最多等待3600秒确认带认证的模型列表。

本轮没有实际 apt 安装、wheel 下载/安装、CUDA 编译或 GPU 推理，不能声称 NUC9 已部署成功。只做 CPU 策略/隔离测试、dry-run 和源码检查。

来源：[vLLM 原生安装](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)、[官方 cu129 wheel 发布](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)、[NVIDIA CUDA 包](https://anaconda.org/nvidia/cuda-nvcc)、[Ubuntu 驱动安装](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)。
