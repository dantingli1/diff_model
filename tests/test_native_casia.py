import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_ROOT = PROJECT_ROOT / "opengait"
if str(OPENGAIT_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENGAIT_ROOT))


class NativeCASIAConfigTest(unittest.TestCase):
    @staticmethod
    def load_config():
        config_path = (
            PROJECT_ROOT / "configs" / "GLGait" / "GLGait_CASIA-B.yaml"
        )
        if not config_path.exists():
            raise AssertionError(f"缺少原生 CASIA 配置：{config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def test_uses_aligned_silhouette_with_absolute_paths(self):
        config = self.load_config()
        data_cfg = config["data_cfg"]

        self.assertEqual(data_cfg["dataset_name"], "CASIA-B")
        self.assertEqual(
            data_cfg["dataset_root"],
            "/mnt/home/xiaoziqiang/OpenGait/datasets/CASIA-B/CASIA-B-START",
        )
        self.assertEqual(
            data_cfg["dataset_partition"],
            "/mnt/home/xiaoziqiang/OpenGait/datasets/CASIA-B/CASIA-B.json",
        )
        self.assertEqual(
            [index for index, enabled in enumerate(data_cfg["data_in_use"]) if enabled],
            [0],
        )
        self.assertEqual(
            [item["type"] for item in config["trainer_cfg"]["transform"]],
            ["BaseSilTransform"],
        )
        self.assertEqual(
            [item["type"] for item in config["evaluator_cfg"]["transform"]],
            ["BaseSilTransform"],
        )

    def test_uses_native_ordered_single_gpu_schedule_from_scratch(self):
        config = self.load_config()
        trainer_cfg = config["trainer_cfg"]

        self.assertEqual(trainer_cfg["restore_hint"], 0)
        self.assertEqual(config["evaluator_cfg"]["restore_hint"], 0)
        self.assertEqual(trainer_cfg["sampler"]["sample_type"], "fixed_ordered")
        self.assertEqual(trainer_cfg["sampler"]["frames_num_fixed"], 30)
        self.assertEqual(trainer_cfg["sampler"]["batch_size"], [8, 4])
        self.assertFalse(trainer_cfg["sync_BN"])
        self.assertEqual(trainer_cfg["total_iter"], 80000)
        self.assertEqual(
            config["scheduler_cfg"]["milestones"],
            [20000, 40000, 50000, 70000],
        )

    def test_uses_native_glgait_head_and_ctl(self):
        config = self.load_config()
        model_cfg = config["model_cfg"]

        self.assertEqual(model_cfg["model"], "Baseline_trans")
        self.assertEqual(model_cfg["backbone_cfg"]["type"], "GLGait")
        self.assertEqual(model_cfg["backbone_cfg"]["channels"], [64, 128, 256, 512])
        self.assertEqual(model_cfg["SeparateFCs"]["parts_num"], 16)
        self.assertEqual(model_cfg["SeparateBNNecks"]["class_num"], 74)
        self.assertEqual(model_cfg["SeparateBNNecks"]["parts_num"], 16)
        self.assertEqual(
            [item["log_prefix"] for item in config["loss_cfg"]],
            ["ctl", "softmax"],
        )
        self.assertEqual(config["loss_cfg"][0]["start"], 30000)

    def test_real_aligned_silhouette_keeps_64_by_44_shape(self):
        from data.transform import get_transform

        config = self.load_config()
        root = Path(config["data_cfg"]["dataset_root"])
        sequence_dir = next(path for path in root.glob("*/*/*") if path.is_dir())
        files = sorted(sequence_dir.glob("*.pkl"))
        selected = [
            path
            for path, enabled in zip(files, config["data_cfg"]["data_in_use"])
            if enabled
        ]
        self.assertEqual([path.name.split("-", 1)[0] for path in selected], ["1"])

        transform = get_transform(config["trainer_cfg"]["transform"])[0]
        silhouette = transform(pickle.loads(selected[0].read_bytes()))

        self.assertEqual(tuple(silhouette.shape[1:]), (64, 44))
        self.assertTrue(np.isfinite(silhouette).all())
        self.assertGreaterEqual(float(silhouette.min()), 0.0)
        self.assertLessEqual(float(silhouette.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
