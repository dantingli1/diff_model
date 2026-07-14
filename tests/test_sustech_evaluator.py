import sys
import unittest
import warnings
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
            [
                "nm-01",
                "nm-02",
                "nm-03",
                "nm-04",
                "nm-05",
                "nm-06",
                "bg-01",
                "bg-02",
                "cl-01",
                "cl-02",
            ]
        )
        message_manager = RecordingMessageManager()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Conversion of an array with ndim > 0 to a scalar is deprecated",
                category=DeprecationWarning,
            )
            with patch.object(
                evaluator, "cuda_dist", side_effect=cpu_dist
            ), patch.object(evaluator, "get_msg_mgr", return_value=message_manager):
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
