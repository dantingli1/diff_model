# GLGait 固定长度 seqL 兼容设计

## 问题

`fixed_ordered` 训练批次的所有序列帧数一致，因此 `CollateFn` 用 `seqL=None` 表示无需按长度拆包。`Baseline_trans` 在将帧数裁到 3 的倍数后，无条件写入 `seqL[0]`，与该约定冲突。

## 方案对比

1. 在模型中增加 `seqL is not None` 判断：保持固定采样和全局数据协议，改动最小，采用。
2. 把 YAML 改为 `unfixed_ordered`：可生成长度数组，但改变训练采样分布，不采用。
3. 修改 `CollateFn`，让 fixed 也生成长度数组：影响所有模型和批次布局，不采用。

## 设计

仅将：

```python
seqL[0] = sils.size()[2]
```

改为：

```python
if seqL is not None:
    seqL[0] = sils.size()[2]
```

`seqL=None` 时由现有 `PackSequenceWrapper` 对完整固定长度张量池化；非空时保持原来的可变长度更新逻辑。

## 验证

用原仓库真实 `Baseline_trans` 构造 7 元组、`seqL=None` 的最小前向：修改前应在原行失败，修改后应输出 `(1, 256, 16)` embedding。随后把同一文件同步到 out124，复跑相同前向并核对两端 SHA-256。
