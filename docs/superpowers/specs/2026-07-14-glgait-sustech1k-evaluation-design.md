# GLGait SUSTech1K 评估兼容设计

## 背景与根因

GLGait 仓库携带的旧版 OpenGait 评估器只支持 CASIA-B、OUMVLP 和 CASIA-E。当前配置把 `test_dataset_name` 设为 `SUSTech1K`，因此特征提取完成后会在 `evaluate_indoor_dataset` 的数据集白名单检查处抛出 `KeyError`。模型前向、checkpoint 和训练类中心不属于本问题范围。

## 修改范围

同步修改以下两个项目：

- 本机 `/mnt/home/xiaoziqiang/GLGait`
- 远端 `out124:/data0/xiaoziqiang/GLGait`

只修改 `opengait/evaluation/evaluator.py`、新增针对性测试，并更新各自 `.conversation/daily.md`。两边 YAML 含不同的环境路径，保持原样；模型及 checkpoint 不修改。

## 评估协议

采用当前 OpenGait 已有的 SUSTech1K 官方兼容逻辑：

- gallery：序列类型包含 `00-nm`。
- probe：分别统计 Normal、Bag、Clothing、Carrying、Umberalla、Uniform、Occlusion、Night 和 Overall。
- Normal 使用 `01-nm`；其他条件按 `bg`、`cl`、`cr`、`ub`、`uf`、`oc`、`nt` 子串匹配，以覆盖 `01-cr-bg-nt` 等复合协变量。
- Overall 统计序列类型包含 `01`、`02`、`03` 或 `04` 的 probe。
- 评估 12 个视角，排除相同视角的 probe-gallery 组合。
- 计算并记录 Rank-1 至 Rank-5；保持上游兼容的指标键名及 `Umberalla` 拼写。

## 代码设计

在现有 `single_view_gallery_evaluation` 中增加 SUSTech1K 的 probe/gallery 字典、子串筛选分支和 `num_rank=5` 分支；在 `evaluate_indoor_dataset` 白名单中加入 `SUSTech1K`。其余数据集继续走原有分支，不新增抽象、不重构公共评估流程。

## 测试与验收

1. 先新增失败测试，证明旧代码对 `SUSTech1K` 抛出不支持异常。
2. 使用合成的多身份、双视角数据覆盖 `00-nm`、`01-nm` 和复合协变量；距离计算边界替换为确定性的 CPU 实现，验证公开评估入口、条件分类和 Rank-1～Rank-5 日志。
3. 修改后运行针对性测试和现有测试，并执行 Python 编译检查。
4. 对比两边评估器和测试文件 SHA-256，确认同步一致。
5. 在远端 gait0 环境重新运行最小复现，确认不再出现 SUSTech1K 不支持异常；不在本任务中启动完整长时间测试作业。

## 风险控制

本次不改变 embedding、距离度量、训练参数或采样方式，因此不会改变模型输出。主要风险是协议筛选错误，使用当前 OpenGait 参考实现和远端真实目录分布进行双重校验。
