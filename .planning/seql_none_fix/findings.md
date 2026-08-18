# 调研发现

- Gait3D 与 GREW 原生训练 YAML 均为 `fixed_ordered`。
- 原仓库 `CollateFn` 对 fixed 采样返回 `seqL=None`。
- `Baseline_trans.forward` 无条件执行 `seqL[0] = sils.size()[2]`，导致第一批前向失败。
- `PackSequenceWrapper` 已支持 `seqL=None`，会直接对固定长度批次做时序池化。
- 当前 OpenGait 迁移版已使用非空判断，但其输入是 5 元组；原仓库是 7 元组，不能整文件覆盖。
- `unfixed_ordered` 虽会生成非空 `seqL`，但会改为每序列随机 20–40 帧；原模型随后把拼接总帧数广播写入全部长度，并可能在 3 帧分组时破坏样本边界，因此只能绕过异常，不能作为正确修复。
