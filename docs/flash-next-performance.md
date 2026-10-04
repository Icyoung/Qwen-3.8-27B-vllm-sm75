# Flash Next：无MTP与混合负载优化

2026-10-04。本次补齐真正无MTP启动测试，并扩展SM75 TP8的QSA混合批次执行路径。使用同一个`vllm-sm75:v0.1.7-ultra-beta`镜像，模型、QSA、FP8 KV、256K和并发4均保留。

## 运行方式

沿用[完整使用说明](flash-next-tp8.md)的GPU、内存、模型、缓存、API Key和1615/8000端口配置，只选择对应模型模板：

```bash
# MTP3
--profile flash-next-tp8-256k-mtp3

# 真正无MTP：启动时不加载草稿
--profile flash-next-tp8-256k-nomtp
```

两者模型路径均可由`MODEL`指定；MTP3另可用`DRAFT_MODEL`指定草稿。Ultra面板可加载同名模板。仅API部署使用同一镜像并设置`SM75_EDITION=standard`。

完整CLI的无MTP方式是移除`--speculative-config`，将`compilation-config`的`cudagraph_capture_sizes`设为`[1,2,4]`；不是只在运行中关闭草稿计算。其他TP8、QSA、FP8 KV、GPU KV字节预算、256K、seq4和batch4096保持相同。

## 新优化参数

| 参数 | 默认推荐 | 作用 |
|---|---|---|
| `VLLM_SM75_QSA_MIXED_PREFILL` | `1` | 按请求分别使用已有QSA indexer和attention快路径，避免整个混合批次统一退回较慢路径 |
| `VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET` | `0` | 维持原4096预算，优先混合总吞吐 |
| 上一参数的可选值 | `1024` | 有Decode且混有待Prefill请求时缩小单轮工作量，减少流输出长停顿；会有总吞吐代价 |

1024设置保留单路4096预算，不修改KV容量，也不卸载专家。它只对Next TP8启用，其他模型/TP回退。两个环境开关均可通过完整Docker参数或面板高级环境设置配置；从Next切换到27B时清除Next专用开关。

## 选择语义与兼容边界

分组处理复用原有的QSA选择预算和每请求页表，不降低top-k，不删除QSA/indexer，不修改模型权重。MTP草稿、多卡数量或head布局不匹配、CUDA Graph捕获等情况保留原路径。

GPU数值验证覆盖FP16/FP8缓存、非对齐长度、262144边界、物理页重排及decode/prefill混合。选中的索引集合一致；TopK返回顺序不保证位级一致。Attention最大绝对差约3.05e-5，最大相对L2约2.85e-4。整模型还分别测试无MTP、MTP3、冷/缓存混合检索和工具调用。

## 测速口径

单路Prefill按完整输入tokens除以服务端Prefill阶段时间；Decode按流式token IDs计时。并发混合组的总端到端吞吐，是所有输出tokens除以整组耗时，另记录每请求TTFT、Decode阶段速率及最长token间隔。

同一组1K/8K/16K/32K各256输出，总输出仅1024。因此即使Decode内核很快，长Prefill也会拉低整组输出吞吐；优化需要改善实际短请求等待，而不能只换统计公式。

历史44.55/43.72 tok/s来自旧v0.1.6关闭QSA的无MTP配置，不是最新版的确认值。真实无MTP与MTP热关闭分别记录；本次无MTP已由启动日志`speculative_config=None`、monitor的`spec_configured=false`和零草稿计数共同核验。

最终镜像实测如下。两次8K/512预热；8K正式3次取中位，32K与256K边界各1次；温度0、seed42、每次独立冷前缀，各输出512。长文本单点和MTP输出/接受率存在波动，不承诺所有内容都达到相同数值。

| 模式 | 输入tokens | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| 无MTP | 8,192 | 3225.82 | 49.93 |
| 无MTP | 32,768 | 2835.11 | 49.97 |
| 无MTP | 261,632 | 2218.50 | 51.05 |
| MTP3 | 8,192 | 3127.89 | 69.86 |
| MTP3 | 32,768 | 2726.28 | 65.58 |
| MTP3 | 261,632 | 2108.20 | 65.08 |

同一已加载进程，分别预热原路径和分组路径的混合形状，再各测3组冷前缀取中位。四请求1K/8K/16K/32K、各输出256：

| 模式 | 原路径总端到端 tok/s | 分组路径 tok/s | 变化 | 整组耗时 |
|---|---:|---:|---:|---:|
| 无MTP | 23.21 | 39.82 | +71.56% | 44.12 → 25.72秒 |
| MTP3 | 22.97 | 39.06 | +70.01% | 44.57 → 26.22秒 |

单路无MTP约50是本次真实启动确认值，不能把旧关闭QSA的约44当成同条件基线来宣称增益。新增优化的确认收益是上述混合路径A/B。MTP3并不保证混合组优于无MTP；该组仍包含长Prefill等待。

最终两种模式各8组冷/缓存混合检索全部严格匹配，工具调用通过；289项控制台、90项运行检查、12项请求计时检查通过。本轮未重新执行完整27B六模式GPU矩阵，其模型实现及推理二进制保持，新增调度分支对非Next TP8回退。此前矩阵保留在[10月3日整合验收](validation/v0.1.7-unified.md)。

## 无MTP复核

在无MTP模板服务已经就绪、API Key已设置后，沿用完整说明中的客户端容器命令，将测速脚本参数设为：

```bash
--model-path /models/Qwen3.8-Flash-Next-W4A16-FP8PLE \
--label next-nomtp-check --expect-mtp off \
--sizes 8192,32768,261632 --warmups 2 --repeats 3 --long-repeats 1 --output 512 --mixed
```

该选项验证服务确实未配置MTP，不能用热关闭代替。MTP3使用默认`--expect-mtp on`。缓存测试应按实际命中记录核对，不把一次未命中自动认定为错误。
