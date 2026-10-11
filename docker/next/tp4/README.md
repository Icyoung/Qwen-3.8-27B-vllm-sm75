# 已验证的 Flash Next TP4 适配

来源为本机独立镜像99cbdb70dec7。runtime目录保留13个原实现文件的字节内容；sm75_next_tp4.py仅负责在真实Flash Next、TP4/PP1、EP、FP16、SM75、无投机场景下启用这些钩子。普通模型不会加载它们。

PCIe IPC是对FlashInfer0.6.18的受审回填：上游0.7.0.post1，wheel SHA256与6个新增文件/3个追加文件的base和最终SHA均记录在manifest.json。保留源文件版权和Apache-2.0声明；没有升级整包或替换attention二进制。

同源模板为flash-next-tp4-32k：HC GEMV=0、PCIe-IPC=1；保留QSA indexer=0、FP8 KV、32K/seq1、512MiB每卡KV、28常驻层/20冷层/256热槽及原DiskPLE策略。模型和FP8 PLE页库须只读挂载，路径通过MODEL、SM75_PLE_MODEL、SM75_PLE_STORE指定。

安装器先检查所有源文件和目标base hash，再写入；未知本地修改拒绝覆盖。禁止把该适配全局导入27B/TP8。该迁移没有开展新的缓存算法、算子或权重优化。
