# LMCache 分层缓存（v0.1.7 beta，实验功能）

本候选接入 LMCache MP：GPU 保留当前计算需要的 KV，CPU 保存可复用前缀，专用磁盘保存更大的缓存集合。重复长提示、公共系统提示或恢复会话时，可从较低层把匹配的 KV 搬回 GPU，减少重复预填充。磁盘层的收益与传输开销取决于命中率、CPU/PCIe 和存储速度；首次请求仍需正常计算。

CPU 与磁盘都不参与本实现的 GPU 注意力计算，容量不能直接加到单个活动请求的 GPU KV 上限。YaRN 负责位置长度扩展，LMCache 负责缓存复用，两者均不代表小显存已经可以运行 1M 上下文。

完整Ultra候选已通过四T10的八项整模型CPU/磁盘/重启回读复验，详情见[分阶段验收](validation/v0.1.6-lmcache.md)。范围为target-only的8K合成文本，不扩展为1M或投机组合的保证。

## 固定运行环境

- vLLM 0.30.0 / Python 3.12 / Linux x86_64 / torch 2.13.0+cu129。
- LMCache 0.5.5 cu129，固定 wheel SHA256；CuPy 使用 cuda12x 14.2.0，清理不同 CuPy 包共享同一命名空间的冲突。
- 必须带本候选回移的 [PR #4731](https://github.com/LMCache/LMCache/pull/4731) packed4D 修复。此 PR 在 2026-09-23 仍未合并；本项目回移新规则并适配 vLLM 0.30 的逻辑轴约定，不改变旧的 5D/MLA/Mamba 规则。
- 实测原版 0.5.5 在 Qwen3.8-27B、TP4、FP8 KV 上漏传逻辑页内容：首次回答正常，从 CPU 恢复后乱码。原样回移 PR 后还会因逻辑轴/物理布局混淆被校验拦截；本地适配先核对真实规格，再做无复制视图转换。安装器校验原始/修补文件 SHA，受管启动探针拒绝未修补或未知版本，不能用普通 `pip install lmcache` 替代。

补丁、来源、Apache 许可证与精确哈希位于 [patches](../ultra/source/lmcache/patches/README.md)，依赖在 [install-lock.json](../ultra/source/lmcache/install-lock.json)。安装器不解析升级 torch、vLLM、CUDA 或 FlashInfer；现有依赖不满足固定要求会先报错。旧基座自带的 LMCache 0.5.4 先按安装 RECORD 检查原目标文件，再升级；未知版本、被修改或未登记的目标文件不会被直接覆盖。

## 安装入口

本项目固定的cu129扩展默认不额外安装、不启用。官方基座可能自带未适配的LMCache；本轮完整标准版自带0.5.5，但其CUDA13依赖声明与固定cu129栈不符，不能直接启用，必须先安装下面的固定扩展。标准版与 Ultra 的公共构建入口都接受 `INSTALL_LMCACHE=1`；默认值为 `0`。该选项覆盖标准 full/fast 和 Ultra full/ui 构建路径，复用同一个安装器与启动探针；现有全量构建保护仍生效。安装不等于启用，运行时还须显式选择 LMCache。

```bash
EDITION=standard INSTALL_LMCACHE=1 bash docker/build.sh
# 若使用 Ultra：在已经验证的标准基础镜像上构建
EDITION=ultra INSTALL_LMCACHE=1 bash docker/build.sh
```

已有**经过验证的 v0.1.6 镜像**可以使用单独扩展层（从仓库根目录构建）。该层安装 LMCache 并附带标准版手动启动助手；Ultra 控制台的新功能仍需本候选的 console 源码，不能仅给旧 UI 镜像加此层：

```bash
docker build -f docker/Dockerfile.lmcache \
  --build-arg BASE_IMAGE=<已固定的v0.1.6镜像或摘要> \
  -t local/vllm-sm75:v0.1.6-lmcache .
```

直接检查独立测试容器的安装计划：

```bash
python3 /opt/sm75-workbench/lmcache/install.py
# 仅在明确要安装扩展的候选环境执行：
python3 /opt/sm75-workbench/lmcache/install.py --install
```

标准扩展镜像的安装器路径为 `/opt/vllm-sm75/lmcache/install.py`。完整镜像构建和安装器在已有测试环境内运行是不同的验收项；见后附记录。

## 控制台配置

运行配置 → 编辑 → **CPU KV 与分层缓存** → 缓存方案选择 **LMCache**。选择“默认”显示引擎原生 CPU KV 选项，选择“LMCache”显示 CPU 与磁盘分层配置；端口和磁盘门槛在“磁盘与服务设置”中展开。

| 配置项 | 含义 |
| --- | --- |
| CPU 缓存总容量 | 整个 MP 服务的 GiB，总量只设一次，不乘 TP 卡数 |
| Chunk size | 必填；按实际模型、TP、KV 精度及混合层布局的启动日志设置 |
| 磁盘缓存目录 | 容器内专用绝对目录，建议挂载独立持久卷；留空只用 CPU |
| 启动所需剩余空间 | 启动前可用空间门槛；不是磁盘容量限额 |
| RPC / HTTP 端口 | 默认 5555 / 5556，均只监听容器内 localhost，不应对外发布 |

选择 LMCache 后点击顶部“保存配置”；验证成功时替换已知原生 CPU KV，无需逐项点击应用。自定义连接器由用户自行移除。配置保留原参数和草稿，遇到不兼容组合会报错，不静默删除。当前实验范围拒绝 DFlash/MTP、释放显存的休眠、自动休眠和 `expandable_segments`；使用 P-State 电源模式。

保存、模板、导入导出均保留 LMCache 配置。保存不会重启模型，下次启动才生效。启动时检查版本/补丁、目录、剩余空间、端口，然后等待真实健康接口就绪再启动模型；依赖检查失败会保留当前模型。停止模型会连同缓存服务停止，释放 CUDA IPC 引用；CPU L1 随服务退出丢失，已完成的磁盘缓存保留。

受管启动支持 Ultra 单容器与 Linux native。普通独立 Docker profile 不会自动在容器外启动一个无法共享 CUDA 内存的服务，需按下文在同一容器内配置。

### Chunk size 不能照抄

本次 Qwen3.8-27B、TP4、FP8 KV、关闭投机的实际统一块为 **1568 tokens**，一个逻辑块包含 49 个 32-token 内核页。官方单卡例子的 784、此前带草稿运行的 1664 均不可替代这个配置。

换模型、TP、KV 精度或注意力后端后，应重新核对 `Setting attention block size to N tokens` 及所有组的共同粒度；chunk 必须是共同粒度的倍数。预填充批次至少为 N，设为 `2N-1`（本例 3135）便于逐块快照并让解码与预填充并行调度。vLLM/连接器仍执行自身的容量与粒度检查，工作台不绕过它们。

## 两版相同的 Docker 参数

以下变量可用于标准版与 Ultra。Ultra 在管理服务启动时导入选中的运行配置，之后面板保存的修改生效；下次管理服务/容器启动才会再次导入 Docker 环境。只设置 `SM75_DEFAULT_PROFILE` 也会选择对应配置。标准版最后附加的 CLI 仍可覆盖普通引擎参数，但 LMCache 不兼容参数会被启动助手拒绝。

| 参数 | 默认值与单位 |
| --- | --- |
| `LMCACHE_ENABLED` | `0`，显式 `1` 选择 LMCache |
| `LMCACHE_CPU_GIB` | `8`，整个缓存服务的 CPU GiB，不乘 TP |
| `LMCACHE_CHUNK_SIZE` | 必填正整数，无通用默认值 |
| `LMCACHE_DISK_ROOT` | 可选宿主机专用持久目录，启动脚本挂载它 |
| `LMCACHE_DISK_PATH` | 容器目录；有 `DISK_ROOT` 时默认 `/lmcache`，否则空值只启 CPU |
| `LMCACHE_MIN_FREE_DISK_GIB` | `4`，启动空闲空间门槛，非容量限额 |
| `LMCACHE_PORT` / `LMCACHE_HTTP_PORT` | `5555` / `5556`，仅容器内 localhost，不发布到宿主 |

原生 `CPU_KV_GIB` 支持非负十进制 GiB（例如 `10.25`），与面板一样四舍五入到安全整数字节数；`0` / `0.0` 关闭。正数舍入不足 1 字节、超过安全整数范围或非法格式会拒绝。

只有标准脚本自身默认的原生 CPU KV 8 GiB 会在选择 LMCache 后让位。显式 `CPU_KV_GIB>0`、自定义 `KV_TRANSFER_CONFIG`、投机或休眠组合都会报错。Ultra 保存配置的原生 CPU KV 需要显式 `CPU_KV_GIB=0` 关闭；已有自定义连接器要明确用 `KV_TRANSFER_CONFIG=''` 移除，CPU 便捷值不会代为删除。已存投机参数可明确设置 `SPECULATIVE_CONFIG=''` 关闭。选择 `POWER_MODE=pstate`，移除 `--enable-sleep-mode` 等 allocator 参数；启动导入不会静默删掉它们。

下面的共同参数仅适用于前文已验证的 Qwen3.8-27B、TP4、FP8 KV、target-only 布局，换模型必须重新核对 chunk。示例不执行任何实际构建或部署。

```bash
export VLLM_API_KEY='替换为自己的模型 API Key'
export MODEL_ROOT=/srv/models
export MODEL=/models/Qwen3.8-27B-FP8
export POWER_MODE=pstate
export PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=8192
export MAX_NUM_BATCHED_TOKENS=3135
export CPU_KV_GIB=0
export SPECULATIVE_CONFIG=''
export LMCACHE_ENABLED=1
export LMCACHE_CPU_GIB=8
export LMCACHE_CHUNK_SIZE=1568
# CPU＋磁盘；只用 CPU 时不要设置下面两项
export LMCACHE_DISK_ROOT=/srv/vllm-lmcache
export LMCACHE_DISK_PATH=/lmcache
```

标准版使用已安装固定 LMCache 扩展的镜像：

```bash
EDITION=standard VLLM_SM75_CACHE_ROOT=/srv/vllm-cache \
  bash docker/run.sh
```

Ultra 使用相同环境和自己的持久数据目录；已有多个运行配置时指定需要导入的配置 ID：

```bash
EDITION=ultra ULTRA_DATA_ROOT=/srv/vllm-ultra \
  SM75_DEFAULT_PROFILE=你的运行配置ID bash docker/run.sh
```

新建 Ultra 数据目录时不设置 `SM75_DEFAULT_PROFILE`，由 `MODEL` 创建可见配置。保存了自定义连接器的旧配置需先明确处理冲突，失败时面板保持可用，旧配置不会自动启动。API Key 同名环境会导入 Ultra API 接入设置，复制/更换由面板统一处理。

标准助手在同一容器内依次执行：检查配置、共享探针验证 ABI/packed4D/真实 CLI、检查专用磁盘与端口、启动 MP 并等待 `/healthcheck` 返回 `status: healthy`、启动 P-State 与 vLLM。依赖缺失、端口占用、就绪超时、进程提前退出或收到停止信号都会失败并清理自己启动的进程；退出顺序始终先模型、后缓存。`LMCACHE_ENABLED=0` 沿用原标准启动路径。

不使用公共脚本时，标准扩展镜像也提供 `/opt/vllm-sm75/lmcache-entrypoint.sh` 作为 Docker entrypoint；传入正常的 `MODEL ...vllm 参数`，提供以上环境、P-State 驱动库和必要卷即可。助手自动添加同 Ultra 的 `LMCacheMPConnector`、`--enable-prefix-caching`、`--mamba-cache-mode align`，不需要另起外部缓存容器。fast 镜像仍须先完成其现有 SM75 overlay 编译/验收流程。

磁盘模式要求标准助手能读取本地模型配置/权重元数据，建议使用 `/models` 只读挂载。标准与 Ultra 都按模型身份、完整引擎参数、有效 chunk size、布局环境和安装源码指纹生成子目录；标准使用独立 `standard-v1-` 命名空间，不尝试复用 Ultra 的旧目录。更换布局后旧文件保留，需用户自行管理磁盘配额和清理。停止不会删除用户磁盘缓存；进行中的异步写入不保证完成。

本轮 v0.1.7 beta 新入口仅通过 CPU/模拟进程、打包与环境传递检查，未运行新镜像构建或真实 GPU；历史 GPU 验收仍以 v0.1.6 的独立记录和限定组合为准。

## 磁盘与兼容边界

本候选选择 Python `fs`，避开 0.5.5 `fs_native` 对带 salt 的混合组缓存键的兼容问题。默认存储策略保留 CPU 副本并异步写磁盘，磁盘可跨正常进程重启复用；并未承诺掉电持久性，进行中的写入也可能尚未落完。

`fs` 没有硬容量上限；需要硬限制时，应使用带文件系统配额的专用卷。目录不是编译缓存目录，也不由“清编译缓存”操作清除。普通文件读取可能来自操作系统页缓存，所以磁盘层命中不代表测到了物理盘吞吐量。

只验收文本路径；Qwen 混合 GDN 页不启用 CacheBlend 或 CacheGen，不据此宣称图片/视频 KV 缓存、跨主机共享、分布式缓存或投机组合可用。GDN 的冷算与恢复不保证逐 token 恒等，验收同时检查任务答案、命中来源和必要的字节传输完整性。

- [本轮测试与已知限制](validation/v0.1.6-lmcache.md)
- [官方 vLLM 兼容说明](https://docs.lmcache.ai/getting_started/compatibility.html)
- [官方 Qwen 混合模型说明](https://docs.lmcache.ai/recipes/qwen3_5.html)

构建环境没有 GPU 时，Dockerfile 使用 `--build-no-gpu`，仍加载两个原生扩展检查 ABI，但将需要 `libcuda.so.1` 的 MP 连接器导入留给 GPU 运行时。两版启动 LMCache 前都必须通过 `lmcache_probe.py` 的完整检查；镜像安装成功不等于分层缓存已经通过 GPU 验收。
