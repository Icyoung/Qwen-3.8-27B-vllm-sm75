# BenchLocal 模型测试

Ultra 内置九类题包及所有执行依赖，统一使用 `vllm-sm75:v0.1.7-ultra-beta2`，无需另外构建或下载 BenchLocal 执行镜像。

## 使用

进入“模型测试 → BenchLocal”，默认选择六项，也可以选择全部九项或单独勾选。运行中的模型名称、API 地址和 Key 自动使用当前配置。

| 类目 | 题数 | 内容 | 默认选择 |
|---|---:|---|---|
| 工具调用 | 15 | 工具选择、参数与调用格式 | 是 |
| 信息提取 | 15 | 从原文提取指定字段 | 是 |
| 指令遵循 | 15 | 格式、约束和执行要求 | 是 |
| 指令权限 | 15 | 区分可信指令与不可信内容 | 否 |
| 推理数学 | 15 | 数学与逻辑推理 | 是 |
| 结构化输出 | 15 | JSON 结构与字段约束 | 否 |
| 代码修复 | 15 | 修改程序并运行验证 | 是 |
| Agent 任务 | 20 | 多步工具和浏览器任务 | 是 |
| 命令行任务 | 40 | Shell 与文件操作任务 | 否 |

六项共 95 题，九项共 165 题。测试前可调整思考开关 / 深度、温度、Top P / Top K、输出上限及请求超时；超时默认 300 秒。不适用的思考参数会先提示。

结果按测试类目展示，每排 20 题，CLI-40 最后执行并分为同组两排；色块保留题号和耗时，点击可查看分数、判定、错误原因及原始记录。保留原题包的部分得分和汇总规则，执行错误单独标明。

雷达图使用本次所选类目：六项显示六轴，九项显示九轴。结果可导出 JSON / CSV；JSON 可导入与其他模型或参数方案对比，记录会保留题包版本和请求参数。

## 单镜像按需执行

Ultra 包含全部九项依赖。工具调用、信息提取、指令遵循、指令权限、推理数学、结构化输出直接由控制台调度。代码修复、Agent、命令行使用同一个 Ultra 镜像按需运行，每次最多一个临时执行容器，每类结束或取消后自动删除。

默认六项为工具调用、信息提取、指令遵循、推理数学、代码修复、Agent，共 95 题；全部九项共 165 题，命令行任务最后执行。

```bash
export ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra
export MODEL_ROOT=/srv/models
EDITION=ultra bash docker/run.sh
```

启动入口由宿主管理员安装受限启动器，并给主 Ultra 挂载只读控制通道。主容器保留原安全选项，没有 Docker socket 或宿主 SSH 凭据。执行环境固定镜像与入口，使用独立 PID/IPC/网络、只读根目录、临时工作目录、2 CPU、4 GiB 内存及 512 个进程上限；不挂载模型、用户会话、真实 API Key 或 GPU。模型请求经过按题授权代理转发，环境准备时间独立于题目计时。

执行器保留默认 proc 遮罩。浏览器使用非 root Chromium 沙箱；仅临时执行容器的 seccomp 在固定 Moby 基线上允许 clone/setns/unshare/chroot，不授予 SYS_ADMIN 或 SYS_CHROOT，不使用 systempaths=unconfined。agent-browser 在容器中自动附加的关闭沙箱参数由浏览器入口移除。

已有 Ultra 或 Unraid 手动部署，先安装启动器，再把输出的目录作为只读路径映射到 `/run/sm75-benchlocal`：

```bash
export IMAGE=vllm-sm75:v0.1.7-ultra-beta2
export CONTAINER_NAME=vllm-sm75-ultra
bash docker/benchlocal.sh install
bash docker/benchlocal.sh path
```

Unraid 的路径项目应保留为可编辑字段，容器路径为 `/run/sm75-benchlocal`、访问模式为 `Read Only`。宿主路径需与上面 `path` 输出一致。主容器需要重建一次以加载映射；原数据目录及图标保持原值。

Unraid 安装器写入专用启动项；普通 systemd 主机使用独立服务。后台进程只接受指定主容器的连接，固定创建自己的评测容器，不能提交其他镜像、入口、挂载或权限。控制连接断开、租约超时或启动器重启都会清理所属任务。

```bash
CONTAINER_NAME=vllm-sm75-ultra bash docker/benchlocal.sh status
CONTAINER_NAME=vllm-sm75-ultra bash docker/benchlocal.sh stop
CONTAINER_NAME=vllm-sm75-ultra bash docker/benchlocal.sh ensure
```

使用 `BENCHLOCAL=0` 可跳过宿主启动器安装，此时三个执行类不可用，其余纯校验类仍可运行。重建或升级时启动器与主 Ultra 必须使用同一固定镜像；更换镜像后重新运行安装入口。
