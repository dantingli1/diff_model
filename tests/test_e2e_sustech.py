import sys
import pickle
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "opengait"))


class RgbTransformTest(unittest.TestCase):
    def test_nhwc_sequence_is_transposed_to_nchw(self):
        from data.transform import BaseRgbTransform

        sequence = np.zeros((2, 8, 6, 3), dtype=np.uint8)
        transformed = BaseRgbTransform()(sequence)

        self.assertEqual(transformed.shape, (2, 3, 8, 6))

    def test_nchw_sequence_keeps_layout(self):
        from data.transform import BaseRgbTransform

        sequence = np.zeros((2, 3, 8, 6), dtype=np.uint8)
        transformed = BaseRgbTransform()(sequence)

        self.assertEqual(transformed.shape, (2, 3, 8, 6))


class FrontendComponentsTest(unittest.TestCase):
    def test_dual_decoder_preserves_spatial_size(self):
        from modeling.backbones.u_net_dual_decoder import U_NetDualDecoder

        model = U_NetDualDecoder(
            in_channels=3,
            p_out_channels=1,
            t_out_channels=1,
        ).eval()
        with torch.no_grad():
            p_logits, t_logits = model(torch.randn(2, 3, 32, 24))

        self.assertEqual(tuple(p_logits.shape), (2, 1, 32, 24))
        self.assertEqual(tuple(t_logits.shape), (2, 1, 32, 24))

    def test_weighted_smooth_l1_emphasizes_weighted_pixels(self):
        from modeling.losses.weighted_smooth_l1 import WeightedSmoothL1Loss

        logits = torch.zeros(1, 1, 2, 2)
        labels = torch.zeros_like(logits)
        labels[..., 0, 0] = 1.0
        weights = torch.ones_like(logits)
        weights[..., 0, 0] = 10.0
        loss_fn = WeightedSmoothL1Loss(beta=0.1)

        plain_loss, _ = loss_fn(logits, labels)
        weighted_loss, info = loss_fn(logits, labels, weights=weights)

        self.assertGreater(weighted_loss.item(), plain_loss.item())
        self.assertIn("weight_mean", info)


class E2EMathTest(unittest.TestCase):
    @staticmethod
    def build_math_model():
        from modeling.models.e2e_glgait import E2EGLGait

        model = E2EGLGait.__new__(E2EGLGait)
        torch.nn.Module.__init__(model)
        model.iteration = 20000
        model._build_e2e_frontend({
            "seg_in_channels": 3,
            "db_k": 30,
            "p_db_range": [0.2, 0.7],
            "rec_grad_start_iters": 20000,
            "rec_grad_flow": "t_only",
            "rec_grad_beta": 0.9,
            "rec_target_h": 64,
            "rec_target_w": 44,
        })
        model.train()
        return model

    def test_eval_preprocessing_uses_evaluator_transform_and_restores_engine_cfg(self):
        from modeling.base_model import BaseModel

        model = self.build_math_model()
        trainer_cfg = {"transform": ["train_0", "train_1", "train_2", "train_3"]}
        evaluator_cfg = {"transform": ["eval_0", "eval_1"]}
        model.cfgs = {"evaluator_cfg": evaluator_cfg}
        model.engine_cfg = trainer_cfg
        model.eval()

        def return_selected_transforms(current_model, _inputs):
            return current_model.engine_cfg["transform"]

        with patch.object(
            BaseModel,
            "inputs_pretreament",
            return_selected_transforms,
        ):
            selected_transforms = model.inputs_pretreament(None)

        self.assertEqual(selected_transforms, ["eval_0", "eval_1"])
        self.assertIs(model.engine_cfg, trainer_cfg)

    def test_eval_preprocessing_restores_engine_cfg_after_error(self):
        from modeling.base_model import BaseModel

        model = self.build_math_model()
        trainer_cfg = {"transform": ["train_0", "train_1", "train_2", "train_3"]}
        evaluator_cfg = {"transform": ["eval_0", "eval_1"]}
        model.cfgs = {"evaluator_cfg": evaluator_cfg}
        model.engine_cfg = trainer_cfg
        model.eval()

        def raise_after_selecting_evaluator(current_model, _inputs):
            self.assertIs(current_model.engine_cfg, evaluator_cfg)
            raise RuntimeError("preprocessing failed")

        with patch.object(
            BaseModel,
            "inputs_pretreament",
            raise_after_selecting_evaluator,
        ):
            with self.assertRaisesRegex(RuntimeError, "preprocessing failed"):
                model.inputs_pretreament(None)

        self.assertIs(model.engine_cfg, trainer_cfg)

    def test_t_only_db_alignment_routes_recognition_gradient_to_t(self):
        model = self.build_math_model()
        p_logits = torch.randn(2, 1, 32, 24, requires_grad=True)
        t_logits = torch.randn(2, 1, 32, 24, requires_grad=True)
        prob_map, thresh_map, pred_masks = model._decode_seg_output(
            (p_logits, t_logits)
        )
        rec_mask = model._build_rec_mask(
            prob_map,
            thresh_map,
            region_mask=torch.ones_like(prob_map),
        )
        ratios_hw = torch.tensor([[32.0, 24.0], [32.0, 24.0]])
        aligned = model._align_for_recognition(
            rec_mask,
            ref_masks=pred_masks,
            ratios_hw=ratios_hw,
        )

        aligned.sum().backward()

        self.assertEqual(tuple(aligned.shape), (2, 1, 64, 44))
        self.assertTrue(t_logits.grad is not None and t_logits.grad.abs().sum() > 0)
        self.assertTrue(
            p_logits.grad is None or torch.count_nonzero(p_logits.grad) == 0
        )

    def test_p_and_t_targets_match_raw_silhouette_shape(self):
        model = self.build_math_model()
        sils = torch.zeros(2, 1, 32, 24)
        sils[:, :, 4:28, 6:18] = 1.0
        skeleton = torch.zeros(2, 2, 32, 24)
        skeleton[:, :, 8:24, 10:14] = 1.0

        p_target = model._build_p_target(sils)
        t_target, t_weights = model._build_t_target(sils, skeleton)

        self.assertEqual(tuple(p_target.shape), tuple(sils.shape))
        self.assertEqual(tuple(t_target.shape), tuple(sils.shape))
        self.assertEqual(tuple(t_weights.shape), tuple(sils.shape))
        self.assertGreaterEqual(p_target.min().item(), 0.0)
        self.assertLessEqual(p_target.max().item(), 1.0)
        self.assertGreaterEqual(t_weights.min().item(), 0.0)
        self.assertLessEqual(t_weights.max().item(), 1.0)


class DummySegNet(torch.nn.Module):
    def forward(self, inputs):
        shape = (inputs.shape[0], 1, inputs.shape[2], inputs.shape[3])
        return torch.zeros(shape), torch.zeros(shape)


class DummyRecognitionBackbone(torch.nn.Module):
    def forward(self, inputs):
        batch, _, frames, _, _ = inputs.shape
        return torch.rand(batch, 8, frames, 4, 4)


class E2EAdapterTest(unittest.TestCase):
    @staticmethod
    def build_adapter_model(training):
        model = E2EMathTest.build_math_model()
        model.SegNet = DummySegNet()
        model.iteration = 0
        model.train(training)

        def fake_recognition(self, masks, labels, seq_lens):
            self.received_recognition_masks = masks
            return {
                "training_feat": {
                    "ctl": {"embeddings": masks, "labels": labels},
                    "softmax": {"logits": masks, "labels": labels},
                },
                "visual_summary": {},
                "inference_feat": {"embeddings": masks.mean(dim=(-1, -2))},
            }

        model._forward_recognition = types.MethodType(fake_recognition, model)
        return model

    @staticmethod
    def make_inputs(include_targets):
        batch, frames, height, width = 2, 3, 32, 24
        ratios = torch.tensor([[[height, width]] * frames] * batch).float()
        rgbs = torch.randn(batch, frames, 3, height, width)
        modalities = [ratios, rgbs]
        if include_targets:
            sils = torch.zeros(batch, frames, height, width)
            sils[:, :, 4:28, 6:18] = 1.0
            skeleton = torch.zeros(batch, frames, 2, height, width)
            skeleton[:, :, :, 8:24, 10:14] = 1.0
            modalities.extend((sils, skeleton))
        return (
            modalities,
            torch.tensor([0, 1]),
            None,
            None,
            None,
            None,
            None,
        )

    def test_training_forward_adds_e2e_losses_and_aligned_masks(self):
        model = self.build_adapter_model(training=True)

        output = model(self.make_inputs(include_targets=True))

        self.assertEqual(
            tuple(model.received_recognition_masks.shape),
            (2, 3, 64, 44),
        )
        self.assertIn("ctl", output["training_feat"])
        self.assertIn("softmax", output["training_feat"])
        self.assertIn("p_soft", output["training_feat"])
        self.assertIn("l1_t_structure", output["training_feat"])

    def test_inference_forward_requires_only_ratio_and_rgb(self):
        model = self.build_adapter_model(training=False)

        with torch.no_grad():
            output = model(self.make_inputs(include_targets=False))

        self.assertEqual(
            tuple(model.received_recognition_masks.shape),
            (2, 3, 64, 44),
        )
        self.assertEqual(
            tuple(output["inference_feat"]["embeddings"].shape),
            (2, 3),
        )

    def test_recognition_adapter_builds_glgait_seven_tuple(self):
        model = self.build_adapter_model(training=False)
        masks = torch.rand(2, 3, 64, 44)
        labels = torch.tensor([0, 1])

        recognition_inputs = model._recognition_inputs(masks, labels, None)

        self.assertEqual(len(recognition_inputs), 7)
        self.assertIs(recognition_inputs[0][0], masks)

    def test_e2e_output_is_consumed_by_real_glgait_recognizer(self):
        from modeling.models.e2e_glgait import E2EGLGait

        model = E2EGLGait.__new__(E2EGLGait)
        torch.nn.Module.__init__(model)
        model.iteration = 0
        model.build_network({
            "seg_in_channels": 3,
            "GLGait": {
                "backbone_cfg": {
                    "type": "GLGait",
                    "block": "BasicBlock",
                    "channels": [8, 8, 8, 8],
                    "layers": [1, 1, 1, 1],
                    "strides": [1, 1, 1, 1],
                    "maxpool": False,
                },
                "SeparateFCs": {
                    "in_channels": 8,
                    "out_channels": 16,
                    "parts_num": 16,
                },
                "SeparateBNNecks": {
                    "class_num": 250,
                    "in_channels": 16,
                    "parts_num": 16,
                },
                "bin_num": [16],
            },
        })
        model.Backbone = DummyRecognitionBackbone()
        model.SegNet = DummySegNet()
        model.eval()
        inputs = (
            [
                torch.tensor([[[32, 24], [32, 24], [32, 24]]]).float(),
                torch.rand(1, 3, 3, 32, 24),
            ],
            torch.tensor([0]),
            None,
            None,
            None,
            None,
            None,
        )

        with torch.no_grad():
            output = model(inputs)

        self.assertEqual(
            tuple(output["inference_feat"]["embeddings"].shape),
            (1, 16, 16),
        )


class E2EConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = (
            PROJECT_ROOT / "configs" / "GLGait" / "E2EGLGait_SUSTech1K.yaml"
        )
        with config_path.open("r", encoding="utf-8") as file:
            cls.config = yaml.safe_load(file)

    def test_training_and_evaluation_use_expected_modalities(self):
        data_cfg = self.config["data_cfg"]

        self.assertTrue(
            data_cfg["dataset_partition"].endswith(
                "/GLGait/datasets/SUSTech1K/SUSTech1K.json"
            )
        )
        self.assertEqual(
            [index for index, enabled in enumerate(data_cfg["data_in_use"]) if enabled],
            [4, 5, 7, 17],
        )
        self.assertEqual(
            [
                index
                for index, enabled in enumerate(data_cfg["eval_data_in_use"])
                if enabled
            ],
            [4, 5],
        )
        self.assertEqual(len(self.config["trainer_cfg"]["transform"]), 4)
        self.assertEqual(len(self.config["evaluator_cfg"]["transform"]), 2)

    def test_model_losses_and_single_card_sampler_are_consistent(self):
        self.assertEqual(self.config["model_cfg"]["model"], "E2EGLGait")
        self.assertEqual(self.config["model_cfg"]["rec_target_h"], 64)
        self.assertEqual(self.config["model_cfg"]["rec_target_w"], 44)
        self.assertEqual(
            {loss["log_prefix"] for loss in self.config["loss_cfg"]},
            {"ctl", "softmax", "p_soft", "l1_t_structure"},
        )
        self.assertEqual(self.config["trainer_cfg"]["sampler"]["batch_size"], [4, 4])
        self.assertFalse(self.config["trainer_cfg"]["sync_BN"])

    def test_evaluation_data_cfg_replaces_training_modalities(self):
        from modeling.models.e2e_glgait import E2EGLGait

        eval_cfg = E2EGLGait._build_eval_data_cfg(self.config["data_cfg"])

        self.assertEqual(
            eval_cfg["data_in_use"],
            self.config["data_cfg"]["eval_data_in_use"],
        )
        self.assertEqual(
            self.config["data_cfg"]["data_in_use"][7],
            True,
        )

    def test_real_e2e_sequence_matches_modalities_and_transforms(self):
        from data.transform import get_transform

        data_cfg = self.config["data_cfg"]
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
            ["04", "05", "07", "17"],
        )

        transforms = get_transform(self.config["trainer_cfg"]["transform"])
        arrays = [pickle.loads(path.read_bytes()) for path in selected]
        ratios, rgbs, sils, skeleton = [
            transform(array) for transform, array in zip(transforms, arrays)
        ]

        self.assertEqual(tuple(ratios.shape[1:]), (2,))
        self.assertEqual(tuple(rgbs.shape[1:]), (3, 128, 128))
        self.assertEqual(tuple(sils.shape[1:]), (128, 128))
        self.assertEqual(tuple(skeleton.shape[1:]), (2, 128, 128))

if __name__ == "__main__":
    unittest.main()
