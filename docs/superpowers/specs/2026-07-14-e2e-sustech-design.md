# GLGait SUSTech1K E2E 适配设计

## 目标

在原生 GLGait 中增加从 RGB 预测剪影并完成身份识别的端到端模型。训练协议参考 OpenGait 的 `E2ESetV2_ptdiff_v3`，识别主干保持 GLGait，不改变现有 silhouette-only 模型。

## 输入与数据流

SUSTech1K 训练选择四路数据：`04 Camera-Ratios-HW`、`05 Camera-RGB_raw`、`07 Camera-Sils_raw`、`17 Camera-SkeletonMapRaw`。raw-sil 与 RGB 均为 `128×128`，用于像素级监督；模型将预测 mask 可微对齐为 `64×44` 后送入 GLGait。测试仅加载 ratio 与 RGB，不依赖真值剪影或骨架图。

RGB 数据实际为 `[T,H,W,C]`。`BaseRgbTransform` 仅在最后一维为 3 时转置为 `[T,C,H,W]`，已有 NCHW 输入保持不变。

## 模型结构

新增 `E2EGLGait`，继承现有 `Baseline_trans`：

1. 双解码轻量 U-Net 共享 encoder，分别输出概率分支 P 和阈值分支 T。
2. P 经 sigmoid 后映射至 `[0.2,0.7]`，T 经 sigmoid 得到阈值图，再以渐进式斜率执行可微二值化。
3. 前 10000 iter 使用 raw-sil 经内部对齐后的真值训练识别头；之后使用预测 mask。20000 iter 前识别损失不回传分割前端，之后只向 T 分支回传 0.9 倍梯度。
4. P 使用内缩高斯软标签，T 使用 skeleton 引导的结构目标；两者采用 `WeightedSmoothL1Loss`。
5. 识别输出继续使用 GLGait 原有 CTL 和 Softmax 契约，包括 BNNecks 类别中心 `bnn`。

内部对齐使用 ratio、预测 mask 的纵向包围盒与横向质心生成 ROI，通过 `torchvision.ops.roi_align` 输出 `64×44`，梯度保持连续。

## 配置与评估

新增 `configs/GLGait/E2EGLGait_SUSTech1K.yaml`。训练输入四路、评估输入两路，batch 使用单卡 `[4,4]`。总迭代数采用参考方案的 180000，保留 SUSTech1K 九条件评估函数。

## 范围控制

保留参考方案中影响训练语义的 P/T、DB、warm-up、梯度路由、内部对齐和加权监督。暂不移植定期 OpenCV 图片导出、encoder/decoder 动态冻结和消融开关；这些不是形成可训练 E2E 链路的必要条件。

## 验证

先写失败测试，再实现：

- NHWC RGB 转换为 NCHW，NCHW 输入保持不变；
- 配置的训练/评估模态和损失契约；
- 双解码器输出尺寸；
- P/T→DB→ROI 对齐链路可反向传播；
- GLGait 七元组识别适配与 `64×44` 输入；
- 真实 SUSTech1K 样例训练/评估数据加载；
- 原有 GLGait 与 SUSTech1K 评估测试无回归。
