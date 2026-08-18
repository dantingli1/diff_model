import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from .baseline_trans import Baseline_trans
from ..backbones.u_net_dual_decoder import U_NetDualDecoder


class E2EGLGait(Baseline_trans):
    """由 RGB 预测剪影并交给 GLGait 识别的端到端模型。"""

    @staticmethod
    def _build_eval_data_cfg(data_cfg):
        eval_cfg = dict(data_cfg)
        eval_data_in_use = data_cfg.get("eval_data_in_use")
        if eval_data_in_use is not None:
            eval_cfg["data_in_use"] = list(eval_data_in_use)
        return eval_cfg

    def get_loader(self, data_cfg, train=True):
        if not train:
            data_cfg = self._build_eval_data_cfg(data_cfg)
        return super().get_loader(data_cfg, train)

    def inputs_pretreament(self, inputs):
        if self.training:
            return super().inputs_pretreament(inputs)

        engine_cfg = self.engine_cfg
        self.engine_cfg = self.cfgs["evaluator_cfg"]
        try:
            return super().inputs_pretreament(inputs)
        finally:
            self.engine_cfg = engine_cfg

    def build_network(self, model_cfg):
        Baseline_trans.build_network(self, model_cfg["GLGait"])
        self._build_e2e_frontend(model_cfg)

    def _build_e2e_frontend(self, model_cfg):
        self.SegNet = U_NetDualDecoder(
            in_channels=int(model_cfg.get("seg_in_channels", 3)),
            p_out_channels=1,
            t_out_channels=1,
        )
        self.db_k = float(model_cfg.get("db_k", 30.0))
        self.db_k_start_iters = int(model_cfg.get("db_k_start_iters", 0))
        self.db_k_step_iters = int(model_cfg.get("db_k_step_iters", 0))
        self.db_k_step = float(model_cfg.get("db_k_step", 0.0))
        self.db_k_max = float(model_cfg.get("db_k_max", 0.0))
        p_db_range = model_cfg.get("p_db_range", [0.2, 0.7])
        self.p_db_low = float(p_db_range[0])
        self.p_db_high = float(p_db_range[1])

        self.freeze_rec_iters = int(model_cfg.get("freeze_rec_iters", 10000))
        self.rec_grad_start_iters = int(
            model_cfg.get("rec_grad_start_iters", 20000)
        )
        self.rec_grad_flow = str(
            model_cfg.get("rec_grad_flow", "t_only")
        ).lower()
        self.rec_grad_beta = float(model_cfg.get("rec_grad_beta", 0.9))
        self.rec_grad_inner_range = tuple(
            model_cfg.get("rec_grad_inner_range", [0.0, 0.7])
        )
        self.rec_grad_outer_range = tuple(
            model_cfg.get("rec_grad_outer_range", [0.3, 0.7])
        )

        self.rec_target_h = int(model_cfg.get("rec_target_h", 64))
        self.rec_target_w = int(model_cfg.get("rec_target_w", 44))
        self.align_row_min_pixels = int(model_cfg.get("align_row_min_pixels", 2))
        self.align_min_valid_rows_ratio = float(
            model_cfg.get("align_min_valid_rows_ratio", 0.30)
        )
        self.align_min_bbox_ratio = float(
            model_cfg.get("align_min_bbox_ratio", 0.60)
        )
        self.align_safe_top_ratio = float(
            model_cfg.get("align_safe_top_ratio", 0.10)
        )
        self.align_safe_bottom_ratio = float(
            model_cfg.get("align_safe_bottom_ratio", 0.95)
        )

        self.p_blur_ks = int(model_cfg.get("p_blur_ks", 7))
        self.p_blur_sigma = float(model_cfg.get("p_blur_sigma", 2.0))
        self.t_default = float(model_cfg.get("t_default", 0.5))
        self.t_inner_base = float(model_cfg.get("t_inner_base", 0.45))
        self.t_inner_floor = float(model_cfg.get("t_inner_floor", 0.30))
        self.t_inner_sk_tau = float(model_cfg.get("t_inner_sk_tau", 0.70))
        self.t_outer_floor = float(model_cfg.get("t_outer_floor", 0.30))
        self.t_outer_bg_tau = float(model_cfg.get("t_outer_bg_tau", 0.15))
        self.t_outer_sk_tau = float(model_cfg.get("t_outer_sk_tau", 0.70))

    def _current_iter(self):
        return int(getattr(self, "iteration", 0))

    def _current_db_k(self):
        value = self.db_k
        if self.db_k_step_iters > 0 and self._current_iter() >= self.db_k_start_iters:
            elapsed = self._current_iter() - self.db_k_start_iters
            value += (elapsed // self.db_k_step_iters) * self.db_k_step
        if self.db_k_max > 0:
            value = min(value, self.db_k_max)
        return float(value)

    def _map_p_for_db(self, prob_map):
        return self.p_db_low + (self.p_db_high - self.p_db_low) * prob_map

    def _decode_seg_output(self, seg_output):
        if not isinstance(seg_output, (tuple, list)) or len(seg_output) != 2:
            raise ValueError("双解码器必须返回 P logits 与 T logits。")
        p_logits, t_logits = seg_output
        prob_map = torch.sigmoid(p_logits[:, :1])
        thresh_map = torch.sigmoid(t_logits[:, :1])
        pred_masks = torch.sigmoid(
            self._current_db_k() * (self._map_p_for_db(prob_map) - thresh_map)
        )
        return prob_map, thresh_map, pred_masks

    @staticmethod
    def _scale_gradient(tensor, beta, region_mask=None):
        if beta <= 0:
            return tensor.detach()
        if beta >= 1 and region_mask is None:
            return tensor
        if region_mask is None:
            region_mask = torch.ones_like(tensor)
        return tensor.detach() + beta * region_mask * (tensor - tensor.detach())

    def _build_rec_mask(self, prob_map, thresh_map, region_mask=None):
        p_for_db = self._map_p_for_db(prob_map)
        beta = self.rec_grad_beta if self._current_iter() >= self.rec_grad_start_iters else 0.0
        if not self.training:
            rec_p, rec_t = p_for_db, thresh_map
        elif self.rec_grad_flow == "p_only":
            rec_p = self._scale_gradient(p_for_db, beta, region_mask)
            rec_t = thresh_map.detach()
        elif self.rec_grad_flow == "t_only":
            rec_p = p_for_db.detach()
            rec_t = self._scale_gradient(thresh_map, beta, region_mask)
        elif self.rec_grad_flow == "p_t":
            rec_p = self._scale_gradient(p_for_db, beta, region_mask)
            rec_t = self._scale_gradient(thresh_map, beta, region_mask)
        else:
            raise ValueError("rec_grad_flow 仅支持 p_only、t_only 或 p_t。")
        return torch.sigmoid(self._current_db_k() * (rec_p - rec_t))

    def _build_p_target(self, gt_sils):
        if self.p_blur_ks <= 1:
            return gt_sils.float()
        kernel_size = self.p_blur_ks + (1 - self.p_blur_ks % 2)
        radius = kernel_size // 2
        coords = torch.arange(
            kernel_size,
            device=gt_sils.device,
            dtype=gt_sils.dtype,
        ) - radius
        kernel_1d = torch.exp(-(coords ** 2) / (2 * self.p_blur_sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
        kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(
            1, 1, kernel_size, kernel_size
        )
        blurred = F.conv2d(gt_sils.float(), kernel, padding=radius)
        return torch.clamp(blurred * gt_sils.float(), 0.0, 1.0)

    def _build_t_target(self, gt_sils, gt_skeleton):
        skeleton = torch.clamp(gt_skeleton.max(dim=1, keepdim=True)[0], 0.0, 1.0)
        sil_mask = gt_sils > 0.5
        inner_support = skeleton * sil_mask.float()
        outer_support = skeleton * (~sil_mask).float()
        inner_target = self.t_inner_base - (
            self.t_inner_base - self.t_inner_floor
        ) * inner_support
        target = torch.full_like(gt_sils, self.t_default)
        target = torch.where(sil_mask, inner_target, target)

        inner_weight = 0.2 + 0.8 * torch.clamp(
            inner_support / max(self.t_inner_sk_tau, 1e-6), 0.0, 1.0
        )
        outer_confidence = torch.clamp(
            (outer_support - self.t_outer_bg_tau)
            / max(self.t_outer_sk_tau - self.t_outer_bg_tau, 1e-6),
            0.0,
            1.0,
        )
        outer_weight = self.t_outer_floor + (1.0 - self.t_outer_floor) * (
            1.0 - outer_confidence
        )
        weights = torch.where(sil_mask, inner_weight, outer_weight)
        return torch.clamp(target, 0.0, 1.0), torch.clamp(weights, 0.0, 1.0)

    def _align_for_recognition(self, tensor, ref_masks, ratios_hw):
        batch, _, height, width = ref_masks.shape
        device = ref_masks.device
        binary = (ref_masks.max(dim=1, keepdim=True)[0] > 0.5).float()

        row_mass = binary.sum(dim=3).squeeze(1)
        row_index = torch.arange(height, device=device).unsqueeze(0).expand_as(row_mass)
        valid_rows = row_mass >= float(self.align_row_min_pixels)
        has_rows = valid_rows.any(dim=1)
        top = torch.where(valid_rows, row_index, height).min(dim=1)[0].float()
        bottom = torch.where(valid_rows, row_index, -1).max(dim=1)[0].float()
        top = torch.where(has_rows, top, torch.zeros_like(top))
        bottom = torch.where(
            has_rows,
            bottom,
            torch.full_like(bottom, float(height - 1)),
        )
        trusted = (
            has_rows
            & (valid_rows.sum(dim=1) >= height * self.align_min_valid_rows_ratio)
            & ((bottom - top + 1) >= height * self.align_min_bbox_ratio)
        )
        top = torch.where(
            trusted,
            top,
            torch.full_like(top, (height - 1) * self.align_safe_top_ratio),
        )
        bottom = torch.where(
            trusted,
            bottom,
            torch.full_like(bottom, (height - 1) * self.align_safe_bottom_ratio),
        )

        col_mass = binary.sum(dim=2).squeeze(1)
        col_cumsum = col_mass.cumsum(dim=1)
        col_index = torch.arange(width, device=device).unsqueeze(0).expand_as(col_mass)
        after_center = col_cumsum > col_cumsum[:, -1:] / 2.0
        center = torch.where(after_center, col_index, width).min(dim=1)[0].float()
        center = torch.where(
            has_rows,
            center,
            torch.full_like(center, (width - 1) / 2.0),
        )

        ratio_wh = ratios_hw[:, 1].float() / ratios_hw[:, 0].float().clamp_min(1.0)
        top = torch.clamp(top - 1.0, 0.0, float(height))
        bottom = torch.clamp(bottom + 1.0, 1.0, float(height))
        box_height = (bottom - top).clamp_min(1.0)
        box_width = box_height * float(width) / float(height)

        pad_width = self.rec_target_w // 2
        padded = F.pad(tensor, (pad_width, pad_width, 0, 0))
        center = center + float(pad_width)
        side_padding = torch.clamp(
            (self.rec_target_w - self.rec_target_h * ratio_wh) / 2.0,
            min=0.0,
        )
        input_scale = ratio_wh * self.rec_target_h / float(width)
        side_padding = side_padding / input_scale.clamp_min(1e-5)
        left = center - box_width / 2.0 - side_padding
        right = center + box_width / 2.0 + side_padding

        max_width = float(width + 2 * pad_width)
        boxes = torch.stack(
            (
                left.clamp(0.0, max_width),
                top,
                right.clamp(0.0, max_width),
                bottom,
            ),
            dim=1,
        ).to(dtype=tensor.dtype)
        batch_index = torch.arange(
            batch,
            device=device,
            dtype=tensor.dtype,
        ).unsqueeze(1)
        rois = torch.cat((batch_index, boxes), dim=1)
        return roi_align(
            padded,
            rois,
            output_size=(self.rec_target_h, self.rec_target_w),
            spatial_scale=1.0,
            sampling_ratio=-1,
        )

    @staticmethod
    def _normalize_map(tensor):
        tensor = tensor.float()
        if tensor.detach().max() > 1.5:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _prepare_ratios_hw(ratios_hw, batch, frames):
        total = batch * frames
        if ratios_hw.numel() == total * 2:
            return ratios_hw.reshape(total, 2).float()
        if ratios_hw.numel() == total:
            ratio_wh = ratios_hw.reshape(total).float()
            return torch.stack((torch.ones_like(ratio_wh), ratio_wh), dim=1)
        raise ValueError("ratio 必须为 [N,S] 或 [N,S,2]。")

    def _build_rec_region(self, gt_sils, gt_skeleton):
        skeleton = self._normalize_map(
            gt_skeleton.max(dim=1, keepdim=True)[0]
        )
        sil_mask = gt_sils > 0.5
        inner_low, inner_high = self.rec_grad_inner_range
        outer_low, outer_high = self.rec_grad_outer_range
        inner = sil_mask & (skeleton >= inner_low) & (skeleton < inner_high)
        outer = (~sil_mask) & (skeleton >= outer_low) & (skeleton < outer_high)
        return (inner | outer).float()

    @staticmethod
    def _unpack_inputs(inputs):
        if len(inputs) != 7:
            raise ValueError("GLGait E2E 输入必须是七元组。")
        return inputs[0], inputs[1], inputs[-1]

    @staticmethod
    def _recognition_inputs(masks, labels, seq_lens):
        return [
            [masks],
            labels,
            None,
            None,
            None,
            None,
            seq_lens,
        ]

    def _forward_recognition(self, masks, labels, seq_lens):
        return Baseline_trans.forward(
            self,
            self._recognition_inputs(masks, labels, seq_lens),
        )

    def _forward_inference(self, inputs):
        modalities, labels, seq_lens = self._unpack_inputs(inputs)
        if len(modalities) != 2:
            raise ValueError("GLGait E2E 测试仅需要 ratio 与 RGB。")
        ratios_hw, rgbs = modalities
        batch, frames, channels, height, width = rgbs.shape
        flat_rgbs = rgbs.reshape(batch * frames, channels, height, width)
        flat_ratios = self._prepare_ratios_hw(ratios_hw, batch, frames).to(
            rgbs.device
        )
        prob_map, thresh_map, pred_masks = self._decode_seg_output(
            self.SegNet(flat_rgbs)
        )
        rec_mask = self._build_rec_mask(prob_map, thresh_map)
        aligned = self._align_for_recognition(
            rec_mask,
            ref_masks=pred_masks,
            ratios_hw=flat_ratios,
        )
        return self._forward_recognition(
            aligned.reshape(
                batch,
                frames,
                self.rec_target_h,
                self.rec_target_w,
            ),
            labels,
            seq_lens,
        )

    def forward(self, inputs):
        if not self.training:
            return self._forward_inference(inputs)

        modalities, labels, seq_lens = self._unpack_inputs(inputs)
        if len(modalities) != 4:
            raise ValueError(
                "GLGait E2E 训练需要 ratio、RGB、raw-sil 与 skeleton。"
            )
        ratios_hw, rgbs, gt_sils, gt_skeleton = modalities
        batch, frames, channels, height, width = rgbs.shape
        flat_rgbs = rgbs.reshape(batch * frames, channels, height, width)
        flat_sils = self._normalize_map(
            gt_sils.reshape(batch * frames, 1, height, width)
        )
        flat_skeleton = self._normalize_map(
            gt_skeleton.reshape(batch * frames, 2, height, width)
        )
        flat_ratios = self._prepare_ratios_hw(ratios_hw, batch, frames).to(
            rgbs.device
        )

        prob_map, thresh_map, pred_masks = self._decode_seg_output(
            self.SegNet(flat_rgbs)
        )
        p_target = self._build_p_target(flat_sils)
        t_target, t_weights = self._build_t_target(flat_sils, flat_skeleton)
        region_mask = self._build_rec_region(flat_sils, flat_skeleton)

        if self._current_iter() < self.freeze_rec_iters:
            rec_source = flat_sils.detach()
            ref_masks = flat_sils
        else:
            rec_source = self._build_rec_mask(
                prob_map,
                thresh_map,
                region_mask=region_mask,
            )
            ref_masks = pred_masks
        aligned = self._align_for_recognition(
            rec_source,
            ref_masks=ref_masks,
            ratios_hw=flat_ratios,
        )
        output = self._forward_recognition(
            aligned.reshape(
                batch,
                frames,
                self.rec_target_h,
                self.rec_target_w,
            ),
            labels,
            seq_lens,
        )

        use_structural_weights = self._current_iter() >= self.rec_grad_start_iters
        p_weights = (
            t_weights
            if use_structural_weights and self.rec_grad_flow in ("p_only", "p_t")
            else torch.ones_like(t_weights)
        )
        t_loss_weights = (
            t_weights if use_structural_weights else torch.ones_like(t_weights)
        )
        output["training_feat"]["p_soft"] = {
            "logits": prob_map,
            "labels": p_target,
            "weights": p_weights,
        }
        output["training_feat"]["l1_t_structure"] = {
            "logits": thresh_map,
            "labels": t_target,
            "weights": t_loss_weights,
        }
        output["visual_summary"].update({
            "image/e2e_prob": prob_map,
            "image/e2e_thresh": thresh_map,
            "image/e2e_mask": pred_masks,
            "image/e2e_aligned": aligned,
        })
        return output
