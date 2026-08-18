# GLGait 固定长度 seqL 兼容修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让原仓库 `Baseline_trans` 在 `fixed_ordered` 返回 `seqL=None` 时完成正常前向，并把同一修复同步到 out124。

**Architecture:** 保留 `CollateFn`、YAML 和 7 元组输入协议不变，只在模型写回裁剪帧数时判断 `seqL` 是否存在。新增一个真实模型最小前向回归测试，本地通过后同步模型与测试到远端，并刷新归档包。

**Tech Stack:** Python 3.10、PyTorch、unittest、OpenGait/GLGait、scp、SHA-256

## Global Constraints

- 所有 Python 验证使用 gait0 环境。
- 不改变 `fixed_ordered`、`CollateFn` 或 YAML 采样字段。
- 原仓库输入保持 7 元组，不能复制 OpenGait 迁移版的 5 元组模型文件。
- 仅修改 `opengait/modeling/models/baseline_trans.py` 中的 `seqL` 写入条件。
- 本地与 out124 文件 SHA-256 必须一致。
- 不提交 Git，不重启长期训练任务。

---

### Task 1: 回归测试与单行修复

**Files:**
- Create: `tests/test_baseline_trans_fixed.py`
- Modify: `opengait/modeling/models/baseline_trans.py:36`
- Modify: `.conversation/daily.md`

**Interfaces:**
- Consumes: `Baseline_trans.forward(inputs)` 的原仓库 7 元组输入，其中最后一项允许为 `None`。
- Produces: `inference_feat.embeddings`，固定长度输入时形状为 `(1, 256, 16)`。

- [x] **Step 1: 添加失败回归测试**

```python
import sys
import unittest
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "opengait"))


class DummyBackbone(torch.nn.Module):
    def forward(self, inputs):
        n, _, frames, _, _ = inputs.shape
        return torch.rand(n, 512, frames, 4, 4, device=inputs.device)


class BaselineTransFixedTest(unittest.TestCase):
    def test_fixed_ordered_forward_accepts_none_sequence_lengths(self):
        from modeling.models.baseline_trans import Baseline_trans

        cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs/GLGait/GLGait_Gait3D.yaml").read_text()
        )
        model = Baseline_trans.__new__(Baseline_trans)
        torch.nn.Module.__init__(model)
        model.build_network(cfg["model_cfg"])
        model.Backbone = DummyBackbone()
        model.eval()

        inputs = (
            [torch.rand(1, 30, 64, 44)],
            torch.tensor([0]),
            None,
            None,
            None,
            None,
            None,
        )
        with torch.no_grad():
            retval = model(inputs)

        self.assertEqual(
            tuple(retval["inference_feat"]["embeddings"].shape),
            (1, 256, 16),
        )


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 确认测试因 `seqL=None` 失败**

Run: `MPLCONFIGDIR=/tmp PYTHONDONTWRITEBYTECODE=1 /mnt/home/xiaoziqiang/.conda/envs/gait0/bin/python -m unittest tests.test_baseline_trans_fixed -v`

Expected: `TypeError: 'NoneType' object does not support item assignment`，位置为 `baseline_trans.py` 的 `seqL[0]`。

- [x] **Step 3: 实施最小修复**

```python
        if seqL is not None:
            seqL[0] = sils.size()[2]
```

- [x] **Step 4: 本地验证转绿**

Run: `MPLCONFIGDIR=/tmp PYTHONDONTWRITEBYTECODE=1 /mnt/home/xiaoziqiang/.conda/envs/gait0/bin/python -m unittest tests.test_baseline_trans_fixed -v`

Expected: `Ran 1 test` 和 `OK`。

### Task 2: 远端同步与归档

**Files:**
- Modify: `/data0/xiaoziqiang/GLGait/opengait/modeling/models/baseline_trans.py`
- Create: `/data0/xiaoziqiang/GLGait/tests/test_baseline_trans_fixed.py`
- Modify: `/data0/xiaoziqiang/GLGait/.conversation/daily.md`
- Replace: `/data0/xiaoziqiang/GLGait.tar.gz`

**Interfaces:**
- Consumes: Task 1 通过验证的本地模型与测试文件。
- Produces: 与本地模型哈希一致、可通过同一回归测试的 out124 项目副本和归档。

- [x] **Step 1: scp 同步模型、测试和 daily**

使用 `scp` 分别覆盖远端对应路径；不覆盖其他文件。

- [x] **Step 2: 运行远端定向回归**

Run: `ssh out124 env -C /data0/xiaoziqiang/GLGait MPLCONFIGDIR=/tmp PYTHONDONTWRITEBYTECODE=1 /data0/xiaoziqiang/.conda/envs/gait0/bin/python -m unittest tests.test_baseline_trans_fixed -v`

Expected: `Ran 1 test` 和 `OK`。

- [x] **Step 3: 核对模型 SHA-256**

本地和 `/data0/xiaoziqiang/GLGait/opengait/modeling/models/baseline_trans.py` 的 SHA-256 必须相同。

- [x] **Step 4: 重建并上传完整归档**

重新生成 `/tmp/GLGait.tar.gz`，上传覆盖 `/data0/xiaoziqiang/GLGait.tar.gz`，并核对两端归档 SHA-256。
