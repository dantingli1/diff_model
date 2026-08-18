import torch
import torch.nn.functional as F

from .base import BaseLoss


class WeightedSmoothL1Loss(BaseLoss):
    """支持像素权重图的 Smooth L1。"""

    def __init__(self, loss_term_weight=1.0, beta=1.0):
        super().__init__(loss_term_weight)
        self.beta = beta

    def forward(self, logits, labels, weights=None):
        per_pixel = F.smooth_l1_loss(
            logits.float(),
            labels.float(),
            beta=self.beta,
            reduction="none",
        )
        if weights is None:
            loss = per_pixel.mean()
            weight_mean = torch.ones((), device=loss.device, dtype=loss.dtype)
        else:
            weights = weights.float().to(per_pixel.device)
            if weights.shape != per_pixel.shape:
                weights = weights.expand_as(per_pixel)
            weight_mean = weights.mean()
            normalized = weights / weight_mean.clamp_min(1e-6)
            loss = (per_pixel * normalized).mean()

        self.info.update({
            "loss": loss.detach().clone(),
            "weight_mean": weight_mean.detach().clone(),
        })
        return loss, self.info
