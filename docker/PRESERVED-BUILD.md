# 完整 Ultra 的保留构建

本入口以已核验、由官方 vLLM v0.30.0-cu129 派生的完整 Ultra 为固定 donor，迁入统一源码后折叠有效文件层；保留所有运行依赖和官方原生库。当前受控实现适用于已验收的本地 Btrfs Docker 存储驱动，其他驱动明确拒绝，不修改主机Docker配置。

```bash
BUILD_MODE=preserve \
PRESERVE_BASE_IMAGE=sha256:bb9f2af2b99be6eff3cbda9cad67d09a9d1218f2549af017f3c6e0d3e0aafe9f \
PRESERVE_WORK_ROOT=/mnt/cache/appdata/vllm-sm75/preserve-build-new \
IMAGE=local/vllm-sm75:v0.1.7-ultra-preserve-test \
bash docker/build.sh
```

工作目录必须新建或为空；IMAGE必须为新的local/候选标签。需至少约100GB构建磁盘余量。准备和打包只用CPU、最多4GiB内存/2CPU；镜像文件系统以只读方式挂载，helper没有Docker socket、模型或用户数据。导入监控宿主可用内存、dockerd RSS和剩余空间，只中止本次导入，不重启Docker服务。

官方来源标签和当前完整runtime配置保留。完整inspect数据不落盘；仅将经过固定环境名称白名单与凭据检查的必要OCI配置写入私有构建产物。未知donor、字段、环境名或疑似凭据直接拒绝，升级donor应先更新保留审计。

输出包括便于迁移的Docker归档、层摘要和候选镜像。只构建并导入，不更新正式标签、不启动模型、不发布GitHub。设置PRESERVE_IMPORT=0可仅生成归档。准备镜像暂留作文件对照；完成回归后按本次清单清理。

迁入的Flash Next TP4模板为`flash-next-tp4-32k`：HC GEMV=0、PCIe-IPC=1，保留已有QSA indexer=0、FP8 KV、32K/seq1、512MiB每卡KV及原冷专家/DiskPLE算法。需只读挂载原FP8 PLE页库到`/ple-store`，MODEL指向对应主模型；SM75_PLE_MODEL随模板MODEL展开。仅在显式TP4_NEXT_RUNTIME=1且实际Flash Next/TP4/SM75构造时激活。普通27B和TP8未启用该适配。

新镜像须完成源码/二进制/权限/OCI对照、CPU/控制台/DSH/BenchLocal回归、离线空JIT缓存及有条件的GPU性能验证；镜像大小变小不代表已通过这些检查。正式数据和容器在验收前保持原样。
