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

    def test_training_forward_passes_current_iteration_to_ctl(self):
        from modeling.models.baseline_trans import Baseline_trans

        cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs/GLGait/GLGait_Gait3D.yaml").read_text()
        )
        model = Baseline_trans.__new__(Baseline_trans)
        torch.nn.Module.__init__(model)
        model.build_network(cfg["model_cfg"])
        model.Backbone = DummyBackbone()
        model.iteration = 54321
        model.train()

        inputs = (
            [torch.rand(2, 30, 64, 44)],
            torch.tensor([0, 1]),
            None,
            None,
            None,
            None,
            None,
        )
        retval = model(inputs)
        ctl_features = retval["training_feat"]["ctl"]

        self.assertIn("iteration", ctl_features)
        self.assertEqual(tuple(ctl_features["iteration"].shape), (1,))
        self.assertEqual(ctl_features["iteration"].dtype, torch.long)
        self.assertEqual(ctl_features["iteration"].item(), 54321)


if __name__ == "__main__":
    unittest.main()
