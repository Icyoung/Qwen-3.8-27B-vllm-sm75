# SM75 infra 优化审查（2026-10-13）

基线：上游 commit `ecee5c70ee4e8cbd915edc49340d07fa5f5545ad`。仅源代码审查与 CPU 验证，没有硬件试跑或新性能数据。

## 本轮已改

1. 移除 Ultra/Harness/Node 构建及运行分支，标准 API 只保留推理所需路径。LMCache 安装器作为独立可选 extension 保留。
2. 官方 ModelScope 27B FP8 为默认模型，下载与推理复用持久化缓存，HF 缓存分开。新增不使用 GPU 的 download-only 入口。
3. NUC9 候选默认关闭 8GiB CPU KV，采用 512-token chunk、seq1、lazy safetensors 和受限容器内存，避免未配置就分配超过整机内存的 CPU KV。实际 repack 峰值仍待测。
4. `_can_p2p` 的 skip-check 分支检查所有 peer，不在第一个 peer 后提前返回。Firefly 使用 CPU group 的 MIN 投票，使所有 rank 选择相同 P2P/SHM 后端；扩展加载失败也由整组一致回退。驱动返回 true 仍不代表 payload 已正确。
5. 移除强制 NCCL `SYS` 距离默认值；提供 auto/off/on 配置。off 同时约束 NCCL 与 Firefly，避免两个通信实现的开关不一致。
6. 默认保持 NCCL FP16 reduction。FP8 权重不要求使用 FP8 通信；Firefly 单独 opt-in。其 buffer 分块由 128MiB 调整为 16MiB；TP4 时每 rank 的 IPC 数据槽从约 256MiB 降为约 32MiB，FP16 work 暂存从约 128MiB 降为约 16MiB，metadata 另计。按代码尺寸推导，不是显存实测。

## 后续优化优先级

| 优先级 | 项目 | 需要的证据 |
|---|---|---|
| 1 | 拓扑与 P2P 数据正确性 | 每个方向的 capability、IPC payload 校验、实际设备 UUID，覆盖部分互通与重排 |
| 2 | 通信路径按消息大小选择 | 同一 prompt 下 NCCL/SHM/P2P 的 4KiB–32MiB 延迟与带宽；decode 小消息保持低固定开销 |
| 3 | 同一 PEX switch 内优先通信 | 验证 CPU/PCIe affinity 和跨 Thunderbolt 上行瓶颈，再评估分层 reduce；不能只看 CUDA 能枚举四卡 |
| 4 | FP16/FP8 reduction 的质量差异 | 同 checkpoint 的 logits 与任务验收；压缩通信的 amax/同步开销需要计入 |
| 5 | 模型 lazy load/repack | cgroup anon/file/peak 与每卡显存，避免四 worker 同时读取/展开导致峰值 |
| 6 | GDN/attention 与 overlap | 真实冷 prefill 组件时间，先排除 prefix-cache 命中造成的假提速 |

没有直接删掉两卡/四卡的拓扑门槛，或把 driver capability 结果当成通过数据校验。本轮也没有修改自旋 kernel 的协议：初始化后单 rank 的 CUDA IPC 失败仍可能使其他 rank 等待 collective，后续应独立完善一致清理与超时机制，再做 GPU 故障注入。CPU 投票解决后端选择不一致，不能代替完整 IPC 初始化验收。
