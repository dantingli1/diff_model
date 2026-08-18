# GLGait 固定长度 seqL 兼容修复

## 目标
修复 `Baseline_trans` 在 `fixed_ordered` 训练时对 `seqL=None` 的非法写入，并同步本地与 out124。

## 阶段
- [x] 核实原生 YAML、CollateFn 与报错链路
- [x] 确认最小修复设计
- [x] 运行失败复现
- [x] 实施单行兼容修复并同步 out124
- [x] 完成本地/远端回归和 daily 记录

## 约束
- 不改变 `fixed_ordered` 采样语义。
- 不修改全局 `CollateFn`。
- 不直接覆盖不同输入协议的整份模型文件。
- 仅在 `seqL` 非空时更新裁剪后的帧数。

## 错误记录
- 首次写入计划文件时父目录不存在；创建 `.planning/seql_none_fix` 与 `docs/superpowers` 后继续。
