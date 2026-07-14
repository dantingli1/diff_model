# GLGait SUSTech1K 评估兼容实施计划

> **面向执行代理：** 必须使用 `subagent-driven-development`（推荐）或 `executing-plans` 子技能逐任务实施本计划；所有步骤使用复选框跟踪。

**目标：** 让本机和 out124 的 GLGait 按当前 OpenGait 的 SUSTech1K 官方协议完成 Rank-1～Rank-5 评估，同时保持现有模型、配置及其他数据集评估行为不变。

**架构：** 在公共 `evaluate_indoor_dataset` 入口中放行 `SUSTech1K`，并在 `single_view_gallery_evaluation` 开头增加隔离的 SUSTech1K 分支；原有 CASIA-B、OUMVLP、CASIA-E 代码保持原样。先在本机完成红—绿测试和提交，再把同一评估器及新增测试同步到 out124。

**技术栈：** Python 3.10、NumPy、PyTorch、`unittest`、OpenGait 评估接口、gait0 环境。

## 全局约束

- 本机项目固定为 `/mnt/home/xiaoziqiang/GLGait`，远端项目固定为 `out124:/data0/xiaoziqiang/GLGait`。
- 两边仅同步 `opengait/evaluation/evaluator.py` 和新增的 `tests/test_sustech_evaluator.py`；各自 YAML、模型、checkpoint 及其他工作区改动不得覆盖。
- gallery 使用 `00-nm`；probe 使用 Normal、Bag、Clothing、Carrying、Umberalla、Uniform、Occlusion、Night、Overall，复合协变量按子串匹配。
- 评估排除相同视角，计算并记录 Rank-1～Rank-5，保留当前 OpenGait 的 `Umberalla` 键名。
- 所有 Python 命令使用 gait0；不启动完整长时间测试作业。
- 任何文件修改完成后，分别更新对应项目的 `.conversation/daily.md`，单条记录不超过 200 字。

## 文件结构

- 修改：`opengait/evaluation/evaluator.py`——公开评估入口及 SUSTech1K 协议实现。
- 新建：`tests/test_sustech_evaluator.py`——SUSTech1K 协议与既有 CASIA-B 指标键回归测试。
- 修改：`.conversation/daily.md`——两边项目的变更记录。

---

### 任务 1：在本机以测试驱动补齐 SUSTech1K 评估

**文件：**

- 新建：`/mnt/home/xiaoziqiang/GLGait/tests/test_sustech_evaluator.py`
- 修改：`/mnt/home/xiaoziqiang/GLGait/opengait/evaluation/evaluator.py:71-132`

**接口：**

- 输入：`evaluate_indoor_dataset(data: dict, dataset: str, metric: str = 'euc', cross_view_gallery: bool = False)`。
- `data` 必须包含形状为 `[N, C, P]` 的 `embeddings`，以及等长的 `labels`、`types`、`views`。
- 输出：SUSTech1K 返回 `scalar/test_accuracy/<条件>@R1` 指标，并通过消息管理器记录各条件 Rank-1～Rank-5；其他数据集输出保持原状。

- [ ] **步骤 1：写入失败测试**

将以下完整内容写入 `tests/test_sustech_evaluator.py`：

```python
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "opengait"))


class RecordingMessageManager:
    def __init__(self):
        self.messages = []

    def log_info(self, message):
        self.messages.append(str(message))


def cpu_dist(x, y, metric="euc"):
    if metric != "euc":
        raise AssertionError(f"测试仅覆盖欧氏距离，实际为 {metric}")
    x = torch.from_numpy(x).float().reshape(x.shape[0], -1)
    y = torch.from_numpy(y).float().reshape(y.shape[0], -1)
    return torch.cdist(x, y, p=2)


def make_data(sequence_types):
    features = []
    labels = []
    types = []
    views = []

    def append(label, sequence_type, view, value):
        features.append(value)
        labels.append(label)
        types.append(sequence_type)
        views.append(view)

    for view in ("000", "090"):
        for identity in range(6):
            append(f"{identity:04d}", "00-nm", view, identity * 10.0)
        for sequence_type in sequence_types:
            append("0000", sequence_type, view, 0.1)

    return {
        "embeddings": np.asarray(features, dtype=np.float32).reshape(-1, 1, 1),
        "labels": np.asarray(labels),
        "types": np.asarray(types),
        "views": np.asarray(views),
    }


class SustechEvaluatorTest(unittest.TestCase):
    def test_sustech1k_supports_official_conditions_and_rank5(self):
        from evaluation import evaluator

        data = make_data(
            ["01-nm", "01-cr-bg-nt", "01-cl", "01-ub", "01-uf", "01-oc"]
        )
        message_manager = RecordingMessageManager()

        with patch.object(evaluator, "cuda_dist", side_effect=cpu_dist), patch.object(
            evaluator, "get_msg_mgr", return_value=message_manager
        ):
            try:
                result = evaluator.evaluate_indoor_dataset(data, "SUSTech1K")
            except KeyError as error:
                self.fail(f"SUSTech1K 仍未被评估入口支持：{error}")

        conditions = (
            "Normal",
            "Bag",
            "Clothing",
            "Carrying",
            "Umberalla",
            "Uniform",
            "Occlusion",
            "Night",
            "Overall",
        )
        for condition in conditions:
            self.assertAlmostEqual(
                result[f"scalar/test_accuracy/{condition}@R1"], 100.0
            )
        self.assertIn("Overall@R5: 100.00%", "\n".join(message_manager.messages))

    def test_casia_b_metric_keys_remain_unchanged(self):
        from evaluation import evaluator

        data = make_data(
            ["nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06",
             "bg-01", "bg-02", "cl-01", "cl-02"]
        )
        message_manager = RecordingMessageManager()

        with patch.object(evaluator, "cuda_dist", side_effect=cpu_dist), patch.object(
            evaluator, "get_msg_mgr", return_value=message_manager
        ):
            result = evaluator.evaluate_indoor_dataset(data, "CASIA-B")

        self.assertEqual(
            set(result),
            {
                "scalar/test_accuracy/NM",
                "scalar/test_accuracy/BG",
                "scalar/test_accuracy/CL",
            },
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行目标测试并确认红灯原因正确**

运行：

```bash
cd /mnt/home/xiaoziqiang/GLGait
/mnt/home/xiaoziqiang/.conda/envs/gait0/bin/python -m unittest tests.test_sustech_evaluator.SustechEvaluatorTest.test_sustech1k_supports_official_conditions_and_rank5 -v
```

预期：测试以 `FAIL` 结束，失败信息包含 `SUSTech1K 仍未被评估入口支持` 和原始 `KeyError`；这证明测试命中了当前缺失功能。

- [ ] **步骤 3：写入最小生产实现**

在 `single_view_gallery_evaluation` 的第一行函数体、现有 CASIA-B 字典之前插入以下分支，分支返回后保留原函数其余内容不变：

```python
    if dataset == 'SUSTech1K':
        probe_seq_dict = {
            'Normal': ['01-nm'],
            'Bag': ['bg'],
            'Clothing': ['cl'],
            'Carrying': ['cr'],
            'Umberalla': ['ub'],
            'Uniform': ['uf'],
            'Occlusion': ['oc'],
            'Night': ['nt'],
            'Overall': ['01', '02', '03', '04'],
        }
        gallery_seq = ['00-nm']
        msg_mgr = get_msg_mgr()
        acc = {}
        view_list = sorted(np.unique(view))
        view_num = len(view_list)
        num_rank = 5

        for type_, probe_seq in probe_seq_dict.items():
            acc[type_] = np.zeros((view_num, view_num, num_rank)) - 1.
            for v1, probe_view in enumerate(view_list):
                pseq_mask = np.any(np.asarray([
                    np.char.find(seq_type, probe) >= 0 for probe in probe_seq
                ]), axis=0) & np.isin(view, probe_view)
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                for v2, gallery_view in enumerate(view_list):
                    gseq_mask = np.any(np.asarray([
                        np.char.find(seq_type, gallery) >= 0
                        for gallery in gallery_seq
                    ]), axis=0) & np.isin(view, [gallery_view])
                    gallery_y = label[gseq_mask]
                    gallery_x = feature[gseq_mask, :]
                    dist = cuda_dist(probe_x, gallery_x, metric)
                    idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                    acc[type_][v1, v2, :] = np.round(
                        np.sum(
                            np.cumsum(
                                np.reshape(probe_y, [-1, 1])
                                == gallery_y[idx[:, :num_rank]],
                                1,
                            )
                            > 0,
                            0,
                        )
                        * 100
                        / dist.shape[0],
                        2,
                    )

        result_dict = {}
        msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
        for rank in range(num_rank):
            out_str = ""
            for type_ in probe_seq_dict:
                sub_acc = de_diag(acc[type_][:, :, rank], each_angle=True)
                if rank == 0:
                    msg_mgr.log_info(f'{type_}@R{rank + 1}: {sub_acc}')
                    result_dict[f'scalar/test_accuracy/{type_}@R{rank + 1}'] = np.mean(sub_acc)
                out_str += f'{type_}@R{rank + 1}: {np.mean(sub_acc):.2f}%\t'
            msg_mgr.log_info(out_str)
        return result_dict
```

把 `evaluate_indoor_dataset` 中的白名单改为：

```python
    if dataset not in ('CASIA-B', 'OUMVLP', 'CASIA-E', 'SUSTech1K'):
```

- [ ] **步骤 4：运行本机测试和编译检查并确认绿灯**

依次运行：

```bash
cd /mnt/home/xiaoziqiang/GLGait
/mnt/home/xiaoziqiang/.conda/envs/gait0/bin/python -m unittest tests.test_sustech_evaluator -v
/mnt/home/xiaoziqiang/.conda/envs/gait0/bin/python -m unittest discover -s tests -v
/mnt/home/xiaoziqiang/.conda/envs/gait0/bin/python -m py_compile opengait/evaluation/evaluator.py tests/test_sustech_evaluator.py
```

预期：SUSTech1K 两项测试通过，完整 `tests` 目录零失败，编译命令退出码为 0 且无输出。

- [ ] **步骤 5：提交本机生产代码与测试**

```bash
cd /mnt/home/xiaoziqiang/GLGait
git add opengait/evaluation/evaluator.py tests/test_sustech_evaluator.py
git commit -m "feat: 支持 SUSTech1K 官方评估协议"
```

预期：提交只包含上述两个文件，不包含已有的 `baseline_trans.py`、YAML、`.conversation` 或其他未跟踪内容。

---

### 任务 2：同步 out124、验证一致性并记录修改

**文件：**

- 修改：`out124:/data0/xiaoziqiang/GLGait/opengait/evaluation/evaluator.py`
- 新建：`out124:/data0/xiaoziqiang/GLGait/tests/test_sustech_evaluator.py`
- 修改：`/mnt/home/xiaoziqiang/GLGait/.conversation/daily.md`
- 修改：`out124:/data0/xiaoziqiang/GLGait/.conversation/daily.md`

**接口：**

- 消费任务 1 已通过测试的评估器和测试文件。
- 产出两边内容一致、远端 gait0 可执行的 SUSTech1K 评估支持。

- [ ] **步骤 1：同步前再次确认目标文件没有并发改动**

```bash
sha256sum /mnt/home/xiaoziqiang/GLGait/opengait/evaluation/evaluator.py
ssh out124 "sha256sum /data0/xiaoziqiang/GLGait/opengait/evaluation/evaluator.py && git -C /data0/xiaoziqiang/GLGait diff -- opengait/evaluation/evaluator.py"
```

预期：远端评估器仍是修改前哈希 `09750328dbe2d5174c07911707f19e736a87400df2103cd84bd4eaf8681aa790`，且 `git diff` 无输出；若不满足则停止覆盖并重新比较差异。

- [ ] **步骤 2：只同步评估器和新增测试**

```bash
scp /mnt/home/xiaoziqiang/GLGait/opengait/evaluation/evaluator.py out124:/data0/xiaoziqiang/GLGait/opengait/evaluation/evaluator.py
scp /mnt/home/xiaoziqiang/GLGait/tests/test_sustech_evaluator.py out124:/data0/xiaoziqiang/GLGait/tests/test_sustech_evaluator.py
```

预期：两个 `scp` 命令均以退出码 0 完成；远端 YAML 和模型文件时间戳不发生变化。

- [ ] **步骤 3：在远端 gait0 运行针对性测试、全部测试和编译检查**

```bash
ssh out124 "cd /data0/xiaoziqiang/GLGait && /data0/xiaoziqiang/.conda/envs/gait0/bin/python -m unittest tests.test_sustech_evaluator -v"
ssh out124 "cd /data0/xiaoziqiang/GLGait && /data0/xiaoziqiang/.conda/envs/gait0/bin/python -m unittest discover -s tests -v"
ssh out124 "cd /data0/xiaoziqiang/GLGait && /data0/xiaoziqiang/.conda/envs/gait0/bin/python -m py_compile opengait/evaluation/evaluator.py tests/test_sustech_evaluator.py"
```

预期：针对性测试两项通过，远端完整测试零失败，编译命令退出码为 0。

- [ ] **步骤 4：分别更新两边修改记录**

在两边 `.conversation/daily.md` 各追加以下一条，时间使用执行时的 Asia/Shanghai 实际时间：

```markdown
- 2026-07-14 CST：本机与out124同步补齐SUSTech1K官方评估：00-nm为gallery，复合协变量按子串统计，输出Rank-1～5；新增协议及CASIA-B回归测试，gait0测试和编译通过，模型、checkpoint及YAML未改。
```

- [ ] **步骤 5：执行最终一致性与范围核验**

```bash
sha256sum /mnt/home/xiaoziqiang/GLGait/opengait/evaluation/evaluator.py /mnt/home/xiaoziqiang/GLGait/tests/test_sustech_evaluator.py
ssh out124 "sha256sum /data0/xiaoziqiang/GLGait/opengait/evaluation/evaluator.py /data0/xiaoziqiang/GLGait/tests/test_sustech_evaluator.py"
git -C /mnt/home/xiaoziqiang/GLGait show --stat --oneline HEAD
git -C /mnt/home/xiaoziqiang/GLGait status --short
ssh out124 "git -C /data0/xiaoziqiang/GLGait status --short"
```

预期：两边对应文件 SHA-256 逐项相同；本机最新实现提交只含评估器和新增测试；两边原有无关改动仍存在且未被清理或纳入实现提交。
