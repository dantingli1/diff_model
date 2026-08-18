import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "opengait"))


class CTLIterationTest(unittest.TestCase):
    def run_ctl(self, iteration):
        from modeling.losses import base
        from modeling.losses.triplet import CTL

        self.assertIn("iteration", inspect.signature(CTL.forward).parameters)
        loss_fn = CTL(margin=0.2, start=30000)
        embeddings = torch.tensor([[[0.0]], [[0.0]], [[10.0]], [[10.0]]])
        labels = torch.tensor([0, 0, 1, 1])
        centers = torch.tensor([[[20.0]], [[20.0]], [[-10.0]], [[-10.0]]])

        with patch.object(base, "ddp_all_gather", side_effect=lambda tensor: tensor), patch.object(
            torch.distributed,
            "get_world_size",
            return_value=1,
        ):
            loss, _ = loss_fn(
                embeddings=embeddings,
                labels=labels,
                bnn=centers,
                iteration=torch.tensor([iteration], dtype=torch.long),
            )
        return loss

    def test_resumed_iteration_after_start_uses_center_positive(self):
        loss = self.run_ctl(iteration=50000)

        self.assertGreater(loss.item(), 0.0)

    def test_iteration_before_start_ignores_center_positive(self):
        loss = self.run_ctl(iteration=29999)

        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
