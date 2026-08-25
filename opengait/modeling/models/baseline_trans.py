import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateBNNecks, SeparateFCs, SetBlockWrapper


class TemporalDifference(nn.Module):
    """时序差分模块：计算 t+2 帧与 t 帧的差分特征，保持时序长度不变。

    差分能突出步态中的运动信息，与静态外观特征互补。
    """

    def __init__(self, channel_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channel_dim, eps=eps)

    def forward(self, x, seqL=None):
        if seqL is None:
            return self._forward_chunk(x)

        seq_lens = seqL[0].data.cpu().numpy().tolist()
        start = 0
        chunks = []
        for curr_seq_len in seq_lens:
            chunks.append(self._forward_chunk(x.narrow(2, start, curr_seq_len)))
            start += curr_seq_len
        return torch.cat(chunks, dim=2)

    def _forward_chunk(self, x):
        if x.size(2) <= 2:
            return torch.zeros_like(x)

        # 间隔 2 帧差分：diff(t) = x(t+2) - x(t)
        diff = x[:, :, 2:, :, :] - x[:, :, :-2, :, :]  # [n, c, s-2, h, w]

        # 前后各补 1 帧零，保证差分序列长度与输入一致
        pad_front = torch.zeros_like(x[:, :, :1, :, :])
        pad_back = torch.zeros_like(x[:, :, :1, :, :])
        diff = torch.cat([pad_front, diff, pad_back], dim=2)

        diff = diff.permute(0, 2, 3, 4, 1).contiguous()
        diff = self.norm(diff)
        diff = diff.permute(0, 4, 1, 2, 3).contiguous()

        # 缩放差分值，抑制幅值过大带来的训练不稳定
        return diff * 0.25


class MultiTemporalDiffAdaptive(nn.Module):
    """多路时序差分：同时计算 diff_k (k 来自 diff_ks)，softmax 自适应加权融合。

    输入/输出形状: (n, c, s, h, w)，与原特征同维度，便于做残差连接。
    diff_ks : list[int]   差分间隔列表，如 [1, 2, 3]，可任意组合
    """

    def __init__(self, channel_dim, diff_ks=(1, 2, 3), eps=1e-5):
        super().__init__()
        assert all(isinstance(k, int) and k >= 1 for k in diff_ks), \
            f"diff_ks 必须是 >=1 的 int 列表，得到: {diff_ks}"
        self.diff_ks = list(diff_ks)
        self.norm = nn.LayerNorm(channel_dim, eps=eps)
        # 一组可学习权重，按 softmax 在 diff_ks 维度归一化
        self.diff_weight = nn.Parameter(torch.zeros(len(self.diff_ks)))

    def _calc_diff(self, x, k):
        n, c, s, h, w = x.shape
        if s <= k:
            return torch.zeros_like(x)
        d = x[:, :, k:, :, :] - x[:, :, :-k, :, :]
        pad = torch.zeros_like(x[:, :, :k, :, :])
        d = torch.cat([pad, d], dim=2)
        return self._norm_scale(d)

    def _norm_scale(self, diff):
        diff = diff.permute(0, 2, 3, 4, 1).contiguous()
        diff = self.norm(diff)
        diff = diff.permute(0, 4, 1, 2, 3).contiguous()
        return diff * 0.25

    def forward(self, x, seqL=None):
        if seqL is None:
            return self._forward_chunk(x)
        seq_lens = seqL[0].data.cpu().numpy().tolist()
        start, chunks = 0, []
        for L in seq_lens:
            chunks.append(self._forward_chunk(x.narrow(2, start, L)))
            start += L
        return torch.cat(chunks, dim=2)

    def _forward_chunk(self, x):
        diffs = [self._calc_diff(x, k) for k in self.diff_ks]
        w = F.softmax(self.diff_weight, dim=0)
        return sum(w[i] * d for i, d in enumerate(diffs))


class NextFramePredictionLoss(nn.Module):
    """
    TDV (Temporal Difference in Vision, Daithankar et al. 2026) 风格的
    下一帧预测辅助损失。

    核心约束:
        z_{t+1} ≈ z_t + MotionEncoder(z_{t+1} - z_t)

    - 输入: backbone 输出的 5D 特征 (n, c, s, h, w)
    - motion 输入: 相邻帧特征差 (n, c, s-1, h, w)
    - target: 真下一帧表示 (detached 防 trivial collapse)
    - 输出: scalar MSE loss

    Args:
        channel_dim: 特征通道数
        use_motion_proj: True=1x1x1 Conv 学 motion; False=直接用特征差
    """

    def __init__(self, channel_dim, use_motion_proj=True):
        super().__init__()
        if use_motion_proj:
            self.motion_proj = nn.Conv3d(
                channel_dim, channel_dim, kernel_size=1, bias=False)
        else:
            self.motion_proj = None

    def forward(self, features, seqL=None):
        if seqL is None:
            return self._loss_chunk(features)
        seq_lens = seqL[0].data.cpu().numpy().tolist()
        start, losses = 0, []
        for L in seq_lens:
            if L >= 2:
                losses.append(self._loss_chunk(features.narrow(2, start, L)))
            start += L
        if not losses:
            return features.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _loss_chunk(self, x):
        # x: (n, c, s, h, w)
        if x.size(2) < 2:
            return x.new_tensor(0.0)
        # 帧间差分作为 motion 编码输入
        feat_diff = x[:, :, 1:] - x[:, :, :-1]            # (n, c, s-1, h, w)
        if self.motion_proj is not None:
            motion = self.motion_proj(feat_diff)
        else:
            motion = feat_diff
        # 预测下一帧表示
        z_pred = x[:, :, :-1] + motion                       # (n, c, s-1, h, w)
        # target detached: 防止 loss 推动所有表示坍缩到同一点
        z_target = x[:, :, 1:].detach()
        return F.mse_loss(z_pred, z_target)


class DifferentialAttention(nn.Module):
    """差分注意力：将 Q/K 各拆分为两半，计算 attn1 - lambda * attn2。

    用于抑制注意力中的冗余/噪声成分，突出真正的时序依赖。
    """

    def __init__(self, in_channels, num_heads=4, lambda_init=0.5):
        super().__init__()
        self.num_heads = self._resolve_num_heads(in_channels, num_heads)
        self.head_dim = in_channels // self.num_heads
        self.half_head_dim = self.head_dim // 2
        self.scale = 1.0 / math.sqrt(self.half_head_dim)
        self.lambda_init = lambda_init

        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.o_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.num_heads, self.half_head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.num_heads, self.half_head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.num_heads, self.half_head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.num_heads, self.half_head_dim))

    @staticmethod
    def _resolve_num_heads(in_channels, preferred_heads):
        # 需要 in_channels 能被 head 数整除，且每头维度为偶数（要再拆两半）
        for heads in range(min(preferred_heads, in_channels), 0, -1):
            if in_channels % heads == 0 and (in_channels // heads) % 2 == 0:
                return heads
        raise ValueError(
            "DifferentialAttention requires an even per-head dimension, "
            "but got in_channels={}".format(in_channels)
        )

    def _lambda(self):
        lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1))
        lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1))
        # clamp 防止差分系数过大导致梯度爆炸
        return torch.clamp(lambda_1 - lambda_2 + self.lambda_init, min=-5.0, max=5.0)

    def _forward_chunk(self, x):
        n, c, s, h, w = x.size()
        tokens = rearrange(x, 'n c s h w -> (n h w) s c')

        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        q1, q2 = q.split(self.half_head_dim, dim=-1)
        k1, k2 = k.split(self.half_head_dim, dim=-1)

        attn1 = F.softmax(torch.matmul(q1, k1.transpose(-1, -2)) * self.scale, dim=-1)
        attn2 = F.softmax(torch.matmul(q2, k2.transpose(-1, -2)) * self.scale, dim=-1)

        diff_attn = attn1 - self._lambda().view(1, self.num_heads, 1, 1) * attn2
        out = torch.matmul(diff_attn, v)
        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.o_proj(out)

        return rearrange(out, '(n h w) s c -> n c s h w', n=n, h=h, w=w)

    def forward(self, x, seqL=None):
        if seqL is None:
            return self._forward_chunk(x)

        seq_lens = seqL[0].data.cpu().numpy().tolist()
        start = 0
        chunks = []
        for curr_seq_len in seq_lens:
            chunks.append(self._forward_chunk(x.narrow(2, start, curr_seq_len)))
            start += curr_seq_len
        return torch.cat(chunks, dim=2)


class Baseline_trans(BaseModel):
    """多路时序差分 + 差分残差 + 差分注意力融合基线模型。

    流程：
        sils -> Backbone -> [n, c, s, h, w]
             -> MultiTemporalDiffAdaptive 生成多路加权差分特征
             -> 差分残差:  outs = outs + diff_feat
             -> (可选) DifferentialAttention 时序注意力增强
             -> TP(max) -> HPP -> SeparateFCs -> SeparateBNNecks
    """

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        out_channels = model_cfg['SeparateFCs']['in_channels']
        # 多路时序差分：diff_ks 在 yaml 中可配，默认 [1, 2, 3]
        diff_ks = model_cfg.get('diff_ks', [1, 2, 3])
        self.MultiDiff = MultiTemporalDiffAdaptive(
            channel_dim=out_channels,
            diff_ks=diff_ks,
        )
        # ★ 融合方式可选: "residual"（差分残差, 默认） 或 "concat"（拼接后 1x1 Conv）
        self.diff_fuse = model_cfg.get('diff_fuse', 'residual')
        assert self.diff_fuse in ('residual', 'concat'), \
            f"diff_fuse 必须是 'residual' 或 'concat', got '{self.diff_fuse}'"
        if self.diff_fuse == 'concat':
            # 拼接模式: 2C -> C 的 1x1 Conv 降维
            self.FuseConv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=False)
            self.FuseBN = nn.BatchNorm3d(out_channels)
        # 差分注意力（可通过 model_cfg['use_attn'] = false 关闭）
        self.use_attn = model_cfg.get('use_attn', True)
        if self.use_attn:
            self.DiffAttn = DifferentialAttention(in_channels=out_channels)

        # ★ TDV 下一帧预测损失 (yaml 控制开关 + 权重)
        self.use_tdv = model_cfg.get('use_tdv', False)
        self.tdv_weight = model_cfg.get('tdv_weight', 0.1)
        if self.use_tdv:
            self.NextFrameLoss = NextFramePredictionLoss(
                channel_dim=out_channels,
                use_motion_proj=model_cfg.get('tdv_use_motion_proj', True),
            )

    def forward(self, inputs):
        ipts, labs, _, _, _, _, seqL = inputs
        sils = ipts[0]

        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        if sils.size(2) == 1:
            sils = torch.cat((sils, sils, sils), dim=2)
        if sils.size(2) == 2:
            sils = torch.cat((sils, sils[:, :, -1, :, :].unsqueeze(2)), dim=2)
        # 截断到 3 的整数倍，保证差分与下采样对齐
        if sils.size(2) % 3 != 0:
            num = sils.size(2) // 3
            sils = sils[:, :, :num * 3, :, :]

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # 1) 多路时序差分
        diff_feat = self.MultiDiff(outs, seqL)         # [n, c, s, h, w]

        # 2) ★ 按 diff_fuse 选择融合方式：残差 or 拼接
        if self.diff_fuse == 'residual':
            outs = outs + diff_feat                    # 差分残差（默认）
        else:  # 'concat'
            outs = torch.cat([outs, diff_feat], dim=1) # [n, 2c, s, h, w]
            outs = self.FuseConv(outs)                 # [n, c, s, h, w]
            outs = self.FuseBN(outs)

        # 3) 差分注意力增强（可选）
        if self.use_attn:
            outs = self.DiffAttn(outs, seqL)          # [n, c, s, h, w]

        # ★ TDV 下一帧预测损失: outs 仍然是 (n, c, s, h, w) 5D 特征
        if self.training and self.use_tdv:
            tdv_loss = self.NextFrameLoss(outs, seqL) * self.tdv_weight
        else:
            tdv_loss = None

        if seqL is not None:
            seqL[0] = sils.size(2)

        outs_tp, _ = self.TP(outs, seqL, options={"dim": 2})  # [n, c, h, w]
        feat = self.HPP(outs_tp)                              # [n, c, p]

        embed_1 = self.FCs(feat)                              # [n, c, p]
        embed = embed_1

        n, _, s, h, w = sils.size()
        bnn = self.BNNecks.fc_bin[:, :, labs].permute(2, 1, 0).contiguous().float()  # [n, c, p]

        if self.training:
            _, logits = self.BNNecks(embed_1)  # [n, c, p]
            retval = {
                'training_feat': {
                    'ctl': {'embeddings': embed_1, 'labels': labs, 'bnn': bnn,
                            'iteration': embed_1.new_tensor([self.iteration], dtype=torch.long)},
                    'softmax': {'logits': logits, 'labels': labs},
                },
                'visual_summary': {
                    'image/sils': sils.reshape(n * s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }
            if tdv_loss is not None:
                retval['training_feat']['tdv_loss'] = tdv_loss
        else:
            retval = {
                'visual_summary': {
                    'image/sils': sils.view(n * s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }
        return retval
