import pickle
import sys
import unittest
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_ROOT = PROJECT_ROOT / "opengait"
if str(OPENGAIT_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENGAIT_ROOT))


class CASIARatioCompatibilityTest(unittest.TestCase):
    def test_scalar_width_height_ratio_is_converted_to_hw_pair(self):
        from modeling.models.e2e_glgait import E2EGLGait

        ratios = torch.tensor([[0.5, 0.75], [0.4, 0.6]])

        try:
            actual = E2EGLGait._prepare_ratios_hw(ratios, batch=2, frames=2)
        except ValueError as error:
            self.fail(f"标量 W/H 应被接受，实际错误：{error}")

        expected = torch.tensor(
            [[1.0, 0.5], [1.0, 0.75], [1.0, 0.4], [1.0, 0.6]]
        )
        torch.testing.assert_close(actual, expected)

    def test_existing_hw_pairs_are_preserved(self):
        from modeling.models.e2e_glgait import E2EGLGait

        ratios_hw = torch.tensor([[[128, 64], [128, 80]]])

        actual = E2EGLGait._prepare_ratios_hw(ratios_hw, batch=1, frames=2)

        torch.testing.assert_close(actual, ratios_hw.reshape(2, 2).float())


class CASIAConfigTest(unittest.TestCase):
    @staticmethod
    def load_config():
        config_path = (
            PROJECT_ROOT / "configs" / "GLGait" / "E2EGLGait_CASIA-B.yaml"
        )
        if not config_path.exists():
            raise AssertionError(f"缺少 CASIA 配置：{config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def test_modalities_and_absolute_dataset_paths(self):
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
            [1, 2, 3, 6],
        )
        self.assertEqual(
            [
                index
                for index, enabled in enumerate(data_cfg["eval_data_in_use"])
                if enabled
            ],
            [1, 2],
        )

    def test_training_uses_native_ordered_sampler_from_scratch(self):
        config = self.load_config()
        trainer_cfg = config["trainer_cfg"]

        self.assertEqual(trainer_cfg["restore_hint"], 0)
        self.assertEqual(trainer_cfg["sampler"]["sample_type"], "fixed_ordered")
        self.assertEqual(trainer_cfg["sampler"]["frames_num_fixed"], 30)
        self.assertEqual(trainer_cfg["sampler"]["batch_size"], [4, 4])
        self.assertFalse(trainer_cfg["sync_BN"])

    def test_glgait_casia_classifier_uses_74_classes_and_16_parts(self):
        config = self.load_config()
        recognition_cfg = config["model_cfg"]["GLGait"]

        self.assertEqual(
            recognition_cfg["SeparateBNNecks"]["class_num"],
            74,
        )
        self.assertEqual(recognition_cfg["SeparateFCs"]["parts_num"], 16)
        self.assertEqual(recognition_cfg["SeparateBNNecks"]["parts_num"], 16)

    def test_real_casia_sequence_matches_modalities_and_transforms(self):
        from data.transform import get_transform

        config = self.load_config()
        data_cfg = config["data_cfg"]
        root = Path(data_cfg["dataset_root"])
        sequence_dir = next(path for path in root.glob("*/*/*") if path.is_dir())
        files = sorted(sequence_dir.glob("*.pkl"))
        selected = [
            path
            for path, enabled in zip(files, data_cfg["data_in_use"])
            if enabled
        ]
        self.assertEqual(
            [path.name.split("-", 1)[0] for path in selected],
            ["2", "3", "4", "7"],
        )

        transforms = get_transform(config["trainer_cfg"]["transform"])
        arrays = [pickle.loads(path.read_bytes()) for path in selected]
        ratios, rgbs, sils, skeleton = [
            transform(array) for transform, array in zip(transforms, arrays)
        ]

        self.assertEqual(ratios.ndim, 1)
        self.assertEqual(tuple(rgbs.shape[1:]), (3, 128, 128))
        self.assertEqual(tuple(sils.shape[1:]), (128, 128))
        self.assertEqual(tuple(skeleton.shape[1:]), (2, 128, 128))
        self.assertEqual(len(config["evaluator_cfg"]["transform"]), 2)


if __name__ == "__main__":
    unittest.main()
