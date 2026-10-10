# BenchLocal 模型测试

Ultra 内置九类题包及所有执行依赖，统一使用 `vllm-sm75:v0.1.7-ultra-beta2`，无需另外构建或下载 BenchLocal 执行镜像。

## 使用

进入“模型测试 → BenchLocal”，默认选择六项，也可以选择全部九项或单独勾选。运行中的模型名称、API 地址和 Key 自动使用当前配置。

| 类目 | 题数 | 内容 | 默认选择 |
|---|---:|---|---|
| 工具调用 | 15 | 工具选择、参数与调用格式 | 是 |
| 信息提取 | 15 | 从原文提取指定字段 | 是 |
| 指令遵循 | 15 | 格式、约束和执行要求 | 是 |
| 指令权限 | 15 | 区分可信指令与不可信内容 | 是 |
| 推理数学 | 15 | 数学与逻辑推理 | 是 |
| 结构化输出 | 15 | JSON 结构与字段约束 | 是 |
| 代码修复 | 15 | 修改程序并运行验证 | 否 |
| 命令行任务 | 40 | Shell 与文件操作任务 | 否 |
| Agent 任务 | 20 | 多步工具和浏览器任务 | 否 |

六项共 90 题，九项共 165 题。测试前可调整思考开关 / 深度、温度、Top P / Top K、输出上限及请求超时；超时默认 300 秒。不适用的思考参数会先提示。

结果按测试类目展示，每排 20 题，CLI-40 分两排；色块保留题号和耗时，点击可查看分数、判定、错误原因及原始记录。保留原题包的部分得分和汇总规则，执行错误单独标明。

雷达图使用本次所选类目：六项显示六轴，九项显示九轴。结果可导出 JSON / CSV；JSON 可导入与其他模型或参数方案对比，记录会保留题包版本和请求参数。

## 单镜像运行方式

默认六项直接由控制台调度。代码修复、CLI、Agent 会执行模型生成的代码，启动脚本使用 **同一个 Ultra 镜像**创建三个 CPU 执行容器，分别限制 2 CPU、4 GiB 内存和 512 个进程。执行容器不使用 GPU，不挂载模型、会话、宿主目录或 Docker 套接字，也不对外发布端口。

```bash
export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
export MODEL_ROOT=/srv/models
EDITION=ultra bash docker/run.sh
```

这一个启动入口会准备测试环境，并将主容器连接到内部测试网络。执行依赖共用同一镜像层，不会为三个容器各复制一份完整镜像。

首次凭据自动生成到 `$ULTRA_DATA_ROOT/benchlocal.env`，权限为 600。它只用于控制台与执行器之间认证，不是模型 API Key。

已有容器或 Unraid 手动部署：

```bash
export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
export IMAGE=vllm-sm75:v0.1.7-ultra-beta2
bash docker/benchlocal.sh start
# 创建主容器时加入 --env-file "$ULTRA_DATA_ROOT/benchlocal.env"
CONTAINER_NAME=vllm-sm75-ultra bash docker/benchlocal.sh connect
bash docker/benchlocal.sh status
```

已有凭据可通过 `BENCHLOCAL_ENV_FILE` 指定原文件。更新执行容器前检查是否有评测正在进行，忙碌时拒绝替换。只使用默认六项时可设置 `BENCHLOCAL=0`，无需启动执行容器。

## 固定依赖

题包使用源码中 `ultra/source/benchlocal-native/packs-pins.json` 固定版本，保留原评分逻辑与许可证。编译环境使用 Go 1.21.13、Rust 1.78.0；Hermes 固定提交 `ea74f61d983ebdfd6a863c45761d1b38081f1d08`，使用独立 Python 环境。浏览器及 agent-browser 的地址和 SHA256 见 `ultra/source/benchlocal-verifiers/assets.lock.json`。

测试记录存放于 `/data/benchlocal`。升级时保留原 `/data` 挂载即可保留历史报告。
