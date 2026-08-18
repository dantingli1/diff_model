"""
GL_prediff
==========
在 GLGait 主干最开头（conv1+bn1+relu+maxpool 之后、layer1 之前）
插入：
  1) 多路时序差分 (MultiTemporalDiffAdaptive) —— 差分间隔 k 在 yaml 中可配
  2) 差分注意力     (DifferentialAttention)     —— 在 yaml 中可关闭

所有开关、间隔、注意力头数、lambda 初值均由 yaml 控制，无需改代码。
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..base_model import BaseModel
from ..modules import (
    SetBlockWrapper,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateFCs,
    SeparateBNNecks,
)


# =====================================================================
# 1) Conv3D 工具（独立自包含，不依赖 GLGait.py）
# =====================================================================
class _Conv3DNoSpatial(nn.Conv3d):
    """3D conv with kernel (3,1,1) — 仅时序"""

    def __init__(self, in_planes, out_planes, stride=1, padding=1, group=1):
        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            groups=group,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, 1, 1


class _Conv3DSimple(nn.Conv3d):
    """3D conv with kernel (3,3,3) — 时空同时"""

    def __init__(self, in_planes, out_planes, stride=1, padding=1, group=1):
        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=padding,
            groups=group,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride


class _Conv3DNoTemporal(nn.Conv3d):
    """3D conv with kernel (1,3,3) — 仅空间"""

    def __init__(self, in_planes, out_planes, stride=1, padding=1, group=1):
        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            groups=group,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride


class _BasicConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size,
            stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        return self.conv(x)


# =====================================================================
# 2) 多路时序差分  (diff_ks 可在 yaml 中任意组合 [1]/[2]/[3]/[1,2,3]...)
# =====================================================================
class MultiTemporalDiffAdaptive(nn.Module):
    """
    每条 k 对应一路差分:  diff[t] = x[t+k] - x[t]
    多路差分经 softmax 自适应加权后输出单条融合特征。
    输入/输出形状: (n, c, s, h, w)
    """
    def __init__(self, channel_dim, diff_ks=(1, 2, 3), eps=1e-5):
        super().__init__()
        assert all(isinstance(k, int) and k >= 1 for k in diff_ks), \
            f"diff_ks 必须是 >=1 的 int 列表，得到: {diff_ks}"
        self.diff_ks = list(diff_ks)
        self.norm = nn.LayerNorm(channel_dim, eps=eps)
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


# =====================================================================
# 3) 差分注意力
# =====================================================================
class DifferentialAttention(nn.Module):
    """
    Diff-Transformer 风格的差分自注意力:
        attn = softmax(q1 k1^T) - λ · softmax(q2 k2^T)
    λ 由可学习参数动态生成，clamp 在 [-5,5] 防梯度爆炸。
    输入: (n, c, s, h, w)  → 沿 s 做 self-attention，n*h*w 视为 batch
    输出: 同形状
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
        for h in range(min(preferred_heads, in_channels), 0, -1):
            if in_channels % h == 0 and (in_channels // h) % 2 == 0:
                return h
        raise ValueError(
            f"DifferentialAttention 要求 head_dim 为偶数，但 in_channels={in_channels}")

    def _lambda(self):
        lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1))
        lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1))
        lambda_val = lambda_1 - lambda_2 + self.lambda_init
        return torch.clamp(lambda_val, min=-5.0, max=5.0)

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
        start, chunks = 0, []
        for L in seq_lens:
            chunks.append(self._forward_chunk(x.narrow(2, start, L)))
            start += L
        return torch.cat(chunks, dim=2)


# =====================================================================
# 4) 3D 残差 block（P3D 风格）
# =====================================================================
class _PreDiffBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            _Conv3DNoTemporal(inplanes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            _Conv3DNoSpatial(planes, planes),
            nn.BatchNorm3d(planes),
        )
        self.conv3 = nn.Sequential(
            _Conv3DNoTemporal(planes, planes),
            nn.BatchNorm3d(planes),
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out = self.relu1(out1 + out2)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu3(out)


# =====================================================================
# 5) 主干：GL_prediff
#    conv1+bn1+relu+maxpool 之后、layer1 之前 插入差分增强头
# =====================================================================
class GL_prediff(nn.Module):
    """
    差分增强主干的行为 (yaml 中 use_temporal_diff / use_diff_attention / diff_fuse 决定):

        use_T  use_A   diff_fuse    行为
        ----   ----    ---------    ----
        False  False   —            原版 GLGait，不引入差分
        True   False   —            差分直接残差: x = x + diff_feat
        True   True    residual     diff → DiffAttn(C) → Conv(C→C) → 残差（默认）
        True   True    concat       cat(C,C) → DiffAttn(2C) → Conv(2C→C) → 残差
        False  True    —            (DiffAttn 没有差分可融合，跳过)

    diff_fuse='residual': 差分注意力只看差分特征，参数量更少，专注于运动信息
    diff_fuse='concat':   原特征与差分拼接后一起做注意力，两者在注意力层交互
    """
    def __init__(self,
                 channels=(64, 128, 256, 512),
                 in_channel=1,
                 layers=(1, 4, 4, 1),
                 strides=(1, 2, 2, 1),
                 maxpool=False,

                 # ★ yaml 透传 (默认开差分+差分注意力)
                 use_temporal_diff=True,
                 use_diff_attention=True,
                 diff_ks=(1, 2, 3),
                 diff_attn_heads=4,
                 lambda_init=0.5,
                 diff_fuse='residual'):   # ★ 新增: 'residual' 或 'concat'

        super().__init__()
        self.maxpool_flag = maxpool
        self.inplanes = channels[0]

        # ★ 融合方式: 'residual'(默认, 差分自注意力后残差) 或 'concat'(拼接后差分注意力)
        self.diff_fuse = diff_fuse
        assert self.diff_fuse in ('residual', 'concat'), \
            f"diff_fuse 必须是 'residual' 或 'concat', got '{self.diff_fuse}'"

        # 2D 入口
        self.conv1 = _BasicConv2d(in_channel, self.inplanes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ★ 差分增强头
        self.use_temporal_diff = use_temporal_diff
        self.use_diff_attention = use_diff_attention
        self.diff_ks = list(diff_ks)
        self.diff_in_c = channels[0]

        if self.use_temporal_diff:
            self.MultiDiff = MultiTemporalDiffAdaptive(
                channel_dim=self.diff_in_c,
                diff_ks=self.diff_ks,
            )
        else:
            self.MultiDiff = None

        if self.use_temporal_diff and self.use_diff_attention:
            if self.diff_fuse == 'concat':
                # ★ 模式 1: 拼接 → 2C, 原特征和差分在注意力层交互
                cat_c = self.diff_in_c * 2
                self.DiffAttn = DifferentialAttention(
                    in_channels=cat_c,
                    num_heads=diff_attn_heads,
                    lambda_init=lambda_init,
                )
                self.FuseConv = nn.Conv3d(cat_c, self.diff_in_c, kernel_size=1, bias=False)
            else:  # 'residual'
                # ★ 模式 2: 差分自注意力(C) → 1x1 Conv(C→C) → 残差, 参数量更少
                self.DiffAttn = DifferentialAttention(
                    in_channels=self.diff_in_c,
                    num_heads=diff_attn_heads,
                    lambda_init=lambda_init,
                )
                self.FuseConv = nn.Conv3d(self.diff_in_c, self.diff_in_c, kernel_size=1, bias=False)
            self.FuseBN = nn.BatchNorm3d(self.diff_in_c)
        else:
            self.DiffAttn = None
            self.FuseConv = None
            self.FuseBN = None

        # 3D 主干
        self.layer1 = self._make_layer(channels[0], layers[0], strides[0])
        self.layer2 = self._make_layer(channels[1], layers[1], strides[1])
        self.layer3 = self._make_layer(channels[2], layers[2], strides[2])
        self.layer4 = self._make_layer(channels[3], layers[3], strides[3])

        self._initialize_weights()

    # -------- 3D 主干层构造 --------
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            ds_stride = _Conv3DSimple.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1,
                          stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        layers = [_PreDiffBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_PreDiffBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    # -------- 前向 --------
    def forward(self, x, n=None, s=30):
        # ----- 2D 入口 -----
        x = self.conv1(x)                        # (n*s, 1, 64, 44) -> (n*s, C, 64, 44)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)                  # (n*s, C, 32, 22)

        # ----- reshape 成 5D -----
        bs = x.shape[0] // s
        x = x.view(bs, x.shape[0] // bs, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (n, C, s, h, w)

        # ----- ★ 差分增强头（按开关分支） -----
        if self.use_temporal_diff:
            diff_feat = self.MultiDiff(x)        # (n, C, s, h, w)

            if self.use_diff_attention:
                if self.diff_fuse == 'concat':
                    # ★ 模式 1: 原特征 + 差分 → 拼接 → 差分注意力 → 1x1 Conv → 残差
                    cat = torch.cat([x, diff_feat], dim=1)            # (n, 2C, s, h, w)
                    attn_out = self.DiffAttn(cat, seqL=None)           # (n, 2C, s, h, w)
                    fused = self.FuseConv(attn_out)                     # (n, C, s, h, w)
                else:  # 'residual'
                    # ★ 模式 2: 差分自注意力 → 1x1 Conv → 残差（不 cat 原特征）
                    attn_out = self.DiffAttn(diff_feat, seqL=None)   # (n, C, s, h, w)
                    fused = self.FuseConv(attn_out)                     # (n, C, s, h, w)
                fused = self.FuseBN(fused)
                x = x + fused                                           # ★ 始终是残差
            else:
                # 只用 TemporalDifference：差分直接残差相加
                x = x + diff_feat

        # ----- 3D 主干 -----
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ----- 还原为 4D (n*s, C, h, w) -----
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        return x

    # -------- 初始化 --------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# =====================================================================
# 6) 模型入口：BaseModel
# =====================================================================
class GL_prediff_Model(BaseModel):
    def build_network(self, model_cfg):
        # backbone 参数直接从 cfg 展开（不走 backbones 包查找）
        backbone_cfg = dict(model_cfg['backbone_cfg'])
        backbone_cfg.pop('type', None)
        backbone = GL_prediff(**backbone_cfg)
        self.Backbone = SetBlockWrapper(backbone)

        # head
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, labs, _, _, _, _, seqL = inputs
        sils = ipts[0]

        # ---- 长度规整 ----
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        if sils.size()[2] == 1:
            sils = torch.cat((sils, sils, sils), dim=2)
        if sils.size()[2] == 2:
            sils = torch.cat((sils, sils[:, :, -1:, :, :]), dim=2)
        if sils.size()[2] % 3 != 0:
            num = sils.size()[2] // 3
            sils = sils[:, :, :num * 3, :, :]

        del ipts
        outs = self.Backbone(sils)                 # [n, c, s, h, w]

        if seqL is not None:
            seqL[0] = sils.size()[2]

        outs_tp, _ = self.TP(outs, seqL, options={"dim": 2})  # [n, c, h, w]
        feat = self.HPP(outs_tp)                              # [n, c, p]
        embed_1 = self.FCs(feat)                              # [n, c, p]
        embed = embed_1

        n, _, s, h, w = sils.size()
        bnn = self.BNNecks.fc_bin[:, :, labs].permute(2, 1, 0).contiguous().float()

        if self.training:
            embed_2, logits = self.BNNecks(embed_1)
            retval = {
                'training_feat': {
                    'ctl': {
                        'embeddings': embed_1,
                        'labels': labs,
                        'bnn': bnn,
                        'iteration': embed_1.new_tensor([self.iteration], dtype=torch.long),
                    },
                    'softmax': {'logits': logits, 'labels': labs},
                },
                'visual_summary': {
                    'image/sils': sils.reshape(n * s, 1, h, w),
                },
                'inference_feat': {'embeddings': embed},
            }
        else:
            retval = {
                'visual_summary': {
                    'image/sils': sils.view(n * s, 1, h, w),
                },
                'inference_feat': {'embeddings': embed},
            }
        return retval



# """
# GL_prediff
# ==========
# 在 GLGait 主干最开头（conv1+bn1+relu+maxpool 之后、layer1 之前）
# 插入：
#   1) 多路时序差分 (MultiTemporalDiffAdaptive) —— 差分间隔 k 在 yaml 中可配
#   2) 差分注意力     (DifferentialAttention)     —— 在 yaml 中可关闭

# 所有开关、间隔、注意力头数、lambda 初值均由 yaml 控制，无需改代码。
# """
# import math
# from typing import Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from torch import Tensor

# from ..base_model import BaseModel
# from ..modules import (
#     SetBlockWrapper,
#     HorizontalPoolingPyramid,
#     PackSequenceWrapper,
#     SeparateFCs,
#     SeparateBNNecks,
# )


# # =====================================================================
# # 1) Conv3D 工具（独立自包含，不依赖 GLGait.py）
# # =====================================================================
# class _Conv3DNoSpatial(nn.Conv3d):
#     """3D conv with kernel (3,1,1) — 仅时序"""

#     def __init__(self, in_planes, out_planes, stride=1, padding=1, group=1):
#         super().__init__(
#             in_channels=in_planes,
#             out_channels=out_planes,
#             kernel_size=(3, 1, 1),
#             stride=(stride, 1, 1),
#             padding=(padding, 0, 0),
#             groups=group,
#             bias=False,
#         )

#     @staticmethod
#     def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
#         return stride, 1, 1


# class _Conv3DSimple(nn.Conv3d):
#     """3D conv with kernel (3,3,3) — 时空同时"""

#     def __init__(self, in_planes, out_planes, stride=1, padding=1, group=1):
#         super().__init__(
#             in_channels=in_planes,
#             out_channels=out_planes,
#             kernel_size=(3, 3, 3),
#             stride=(1, stride, stride),
#             padding=padding,
#             groups=group,
#             bias=False,
#         )

#     @staticmethod
#     def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
#         return 1, stride, stride


# class _Conv3DNoTemporal(nn.Conv3d):
#     """3D conv with kernel (1,3,3) — 仅空间"""

#     def __init__(self, in_planes, out_planes, stride=1, padding=1, group=1):
#         super().__init__(
#             in_channels=in_planes,
#             out_channels=out_planes,
#             kernel_size=(1, 3, 3),
#             stride=(1, stride, stride),
#             padding=(0, padding, padding),
#             groups=group,
#             bias=False,
#         )

#     @staticmethod
#     def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
#         return 1, stride, stride


# class _BasicConv2d(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size, stride, padding, **kwargs):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_c, out_c, kernel_size,
#             stride=stride, padding=padding, bias=False, **kwargs)

#     def forward(self, x):
#         return self.conv(x)


# # =====================================================================
# # 2) 多路时序差分  (diff_ks 可在 yaml 中任意组合 [1]/[2]/[3]/[1,2,3]...)
# # =====================================================================
# class MultiTemporalDiffAdaptive(nn.Module):
#     """
#     每条 k 对应一路差分:  diff[t] = x[t+k] - x[t]
#     多路差分经 softmax 自适应加权后输出单条融合特征。
#     输入/输出形状: (n, c, s, h, w)
#     """
#     def __init__(self, channel_dim, diff_ks=(1, 2, 3), eps=1e-5):
#         super().__init__()
#         assert all(isinstance(k, int) and k >= 1 for k in diff_ks), \
#             f"diff_ks 必须是 >=1 的 int 列表，得到: {diff_ks}"
#         self.diff_ks = list(diff_ks)
#         self.norm = nn.LayerNorm(channel_dim, eps=eps)
#         self.diff_weight = nn.Parameter(torch.zeros(len(self.diff_ks)))

#     def _calc_diff(self, x, k):
#         n, c, s, h, w = x.shape
#         if s <= k:
#             return torch.zeros_like(x)
#         d = x[:, :, k:, :, :] - x[:, :, :-k, :, :]
#         pad = torch.zeros_like(x[:, :, :k, :, :])
#         d = torch.cat([pad, d], dim=2)
#         return self._norm_scale(d)

#     def _norm_scale(self, diff):
#         diff = diff.permute(0, 2, 3, 4, 1).contiguous()
#         diff = self.norm(diff)
#         diff = diff.permute(0, 4, 1, 2, 3).contiguous()
#         return diff * 0.25

#     def forward(self, x, seqL=None):
#         if seqL is None:
#             return self._forward_chunk(x)
#         seq_lens = seqL[0].data.cpu().numpy().tolist()
#         start, chunks = 0, []
#         for L in seq_lens:
#             chunks.append(self._forward_chunk(x.narrow(2, start, L)))
#             start += L
#         return torch.cat(chunks, dim=2)

#     def _forward_chunk(self, x):
#         diffs = [self._calc_diff(x, k) for k in self.diff_ks]
#         w = F.softmax(self.diff_weight, dim=0)
#         return sum(w[i] * d for i, d in enumerate(diffs))


# # =====================================================================
# # 3) 差分注意力
# # =====================================================================
# class DifferentialAttention(nn.Module):
#     """
#     Diff-Transformer 风格的差分自注意力:
#         attn = softmax(q1 k1^T) - λ · softmax(q2 k2^T)
#     λ 由可学习参数动态生成，clamp 在 [-5,5] 防梯度爆炸。
#     输入: (n, c, s, h, w)  → 沿 s 做 self-attention，n*h*w 视为 batch
#     输出: 同形状
#     """
#     def __init__(self, in_channels, num_heads=4, lambda_init=0.5):
#         super().__init__()
#         self.num_heads = self._resolve_num_heads(in_channels, num_heads)
#         self.head_dim = in_channels // self.num_heads
#         self.half_head_dim = self.head_dim // 2
#         self.scale = 1.0 / math.sqrt(self.half_head_dim)
#         self.lambda_init = lambda_init

#         self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
#         self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
#         self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
#         self.o_proj = nn.Linear(in_channels, in_channels, bias=False)

#         self.lambda_q1 = nn.Parameter(torch.zeros(self.num_heads, self.half_head_dim))
#         self.lambda_k1 = nn.Parameter(torch.zeros(self.num_heads, self.half_head_dim))
#         self.lambda_q2 = nn.Parameter(torch.zeros(self.num_heads, self.half_head_dim))
#         self.lambda_k2 = nn.Parameter(torch.zeros(self.num_heads, self.half_head_dim))

#     @staticmethod
#     def _resolve_num_heads(in_channels, preferred_heads):
#         for h in range(min(preferred_heads, in_channels), 0, -1):
#             if in_channels % h == 0 and (in_channels // h) % 2 == 0:
#                 return h
#         raise ValueError(
#             f"DifferentialAttention 要求 head_dim 为偶数，但 in_channels={in_channels}")

#     def _lambda(self):
#         lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1))
#         lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1))
#         lambda_val = lambda_1 - lambda_2 + self.lambda_init
#         return torch.clamp(lambda_val, min=-5.0, max=5.0)

#     def _forward_chunk(self, x):
#         n, c, s, h, w = x.size()
#         tokens = rearrange(x, 'n c s h w -> (n h w) s c')

#         q = self.q_proj(tokens)
#         k = self.k_proj(tokens)
#         v = self.v_proj(tokens)
#         q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
#         k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
#         v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

#         q1, q2 = q.split(self.half_head_dim, dim=-1)
#         k1, k2 = k.split(self.half_head_dim, dim=-1)

#         attn1 = F.softmax(torch.matmul(q1, k1.transpose(-1, -2)) * self.scale, dim=-1)
#         attn2 = F.softmax(torch.matmul(q2, k2.transpose(-1, -2)) * self.scale, dim=-1)

#         diff_attn = attn1 - self._lambda().view(1, self.num_heads, 1, 1) * attn2
#         out = torch.matmul(diff_attn, v)
#         out = rearrange(out, 'b h s d -> b s (h d)')
#         out = self.o_proj(out)

#         return rearrange(out, '(n h w) s c -> n c s h w', n=n, h=h, w=w)

#     def forward(self, x, seqL=None):
#         if seqL is None:
#             return self._forward_chunk(x)
#         seq_lens = seqL[0].data.cpu().numpy().tolist()
#         start, chunks = 0, []
#         for L in seq_lens:
#             chunks.append(self._forward_chunk(x.narrow(2, start, L)))
#             start += L
#         return torch.cat(chunks, dim=2)


# # =====================================================================
# # 4) 3D 残差 block（P3D 风格）
# # =====================================================================
# class _PreDiffBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             _Conv3DNoTemporal(inplanes, planes, stride),
#             nn.BatchNorm3d(planes),
#             nn.ReLU(inplace=True),
#         )
#         self.conv2 = nn.Sequential(
#             _Conv3DNoSpatial(planes, planes),
#             nn.BatchNorm3d(planes),
#         )
#         self.conv3 = nn.Sequential(
#             _Conv3DNoTemporal(planes, planes),
#             nn.BatchNorm3d(planes),
#         )
#         self.relu1 = nn.ReLU(inplace=True)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out1 = self.conv1(x)
#         out2 = self.conv2(out1)
#         out = self.relu1(out1 + out2)
#         out = self.conv3(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         return self.relu3(out)


# # =====================================================================
# # 5) 主干：GL_prediff
# #    conv1+bn1+relu+maxpool 之后、layer1 之前 插入差分增强头
# # =====================================================================
# class GL_prediff(nn.Module):
#     """
#     差分增强主干的 4 种行为 (yaml 中 use_temporal_diff / use_diff_attention 决定):

#         use_T  use_A   行为
#         ----   ----   ----
#         False  False   原版 GLGait，不引入差分
#         True   False   TemporalDifference 差分残差
#         True   True    差分 -> DifferentialAttention -> 1x1 Conv3d 降维 -> 残差
#         False  True    (DiffAttn 没有差分可融合，跳过)
#     """
#     def __init__(self,
#                  channels=(64, 128, 256, 512),
#                  in_channel=1,
#                  layers=(1, 4, 4, 1),
#                  strides=(1, 2, 2, 1),
#                  maxpool=False,

#                  # ★ yaml 透传 (默认开差分+差分注意力)
#                  use_temporal_diff=True,
#                  use_diff_attention=True,
#                  diff_ks=(1, 2, 3),
#                  diff_attn_heads=4,
#                  lambda_init=0.5):

#         super().__init__()
#         self.maxpool_flag = maxpool
#         self.inplanes = channels[0]

#         # 2D 入口
#         self.conv1 = _BasicConv2d(in_channel, self.inplanes, 3, 1, 1)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # ★ 差分增强头
#         self.use_temporal_diff = use_temporal_diff
#         self.use_diff_attention = use_diff_attention
#         self.diff_ks = list(diff_ks)
#         self.diff_in_c = channels[0]

#         if self.use_temporal_diff:
#             self.MultiDiff = MultiTemporalDiffAdaptive(
#                 channel_dim=self.diff_in_c,
#                 diff_ks=self.diff_ks,
#             )
#         else:
#             self.MultiDiff = None

#         if self.use_temporal_diff and self.use_diff_attention:
#             cat_c = self.diff_in_c * 2                     # 原 C + 差分融合 C
#             self.DiffAttn = DifferentialAttention(
#                 in_channels=cat_c,
#                 num_heads=diff_attn_heads,
#                 lambda_init=lambda_init,
#             )
#             self.FuseConv = nn.Conv3d(cat_c, self.diff_in_c, kernel_size=1, bias=False)
#             self.FuseBN = nn.BatchNorm3d(self.diff_in_c)
#         else:
#             self.DiffAttn = None
#             self.FuseConv = None
#             self.FuseBN = None

#         # 3D 主干
#         self.layer1 = self._make_layer(channels[0], layers[0], strides[0])
#         self.layer2 = self._make_layer(channels[1], layers[1], strides[1])
#         self.layer3 = self._make_layer(channels[2], layers[2], strides[2])
#         self.layer4 = self._make_layer(channels[3], layers[3], strides[3])

#         self._initialize_weights()

#     # -------- 3D 主干层构造 --------
#     def _make_layer(self, planes, blocks, stride):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
#             ds_stride = _Conv3DSimple.get_downsample_stride(stride)
#             downsample = nn.Sequential(
#                 nn.Conv3d(self.inplanes, planes, kernel_size=1,
#                           stride=ds_stride, bias=False),
#                 nn.BatchNorm3d(planes),
#             )
#         layers = [_PreDiffBlock(self.inplanes, planes, stride, downsample)]
#         self.inplanes = planes
#         for _ in range(1, blocks):
#             layers.append(_PreDiffBlock(self.inplanes, planes))
#         return nn.Sequential(*layers)

#     # -------- 前向 --------
#     def forward(self, x, n=None, s=30):
#         # ----- 2D 入口 -----
#         x = self.conv1(x)                        # (n*s, 1, 64, 44) -> (n*s, C, 64, 44)
#         x = self.bn1(x)
#         x = self.relu(x)
#         if self.maxpool_flag:
#             x = self.maxpool(x)                  # (n*s, C, 32, 22)

#         # ----- reshape 成 5D -----
#         bs = x.shape[0] // s
#         x = x.view(bs, x.shape[0] // bs, x.shape[1], x.shape[2], x.shape[3])
#         x = x.permute(0, 2, 1, 3, 4).contiguous()  # (n, C, s, h, w)

#         # ----- ★ 差分增强头（按开关分支） -----
#         if self.use_temporal_diff:
#             diff_feat = self.MultiDiff(x)        # (n, C, s, h, w)

#             if self.use_diff_attention:
#                 cat = torch.cat([x, diff_feat], dim=1)            # (n, 2C, s, h, w)
#                 attn_out = self.DiffAttn(cat, seqL=None)           # (n, 2C, s, h, w)
#                 fused = self.FuseConv(attn_out)                     # (n, C, s, h, w)
#                 fused = self.FuseBN(fused)
#                 x = x + fused                                       # 残差
#             else:
#                 # 只用 TemporalDifference：差分直接残差相加
#                 x = x + diff_feat

#         # ----- 3D 主干 -----
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         # ----- 还原为 4D (n*s, C, h, w) -----
#         x = x.permute(0, 2, 1, 3, 4).contiguous()
#         x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
#         return x

#     # -------- 初始化 --------
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)


# # =====================================================================
# # 6) 模型入口：BaseModel
# # =====================================================================
# class GL_prediff_Model(BaseModel):
#     def build_network(self, model_cfg):
#         # backbone 参数直接从 cfg 展开（不走 backbones 包查找）
#         backbone_cfg = dict(model_cfg['backbone_cfg'])
#         backbone_cfg.pop('type', None)
#         backbone = GL_prediff(**backbone_cfg)
#         self.Backbone = SetBlockWrapper(backbone)

#         # head
#         self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
#         self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
#         self.TP = PackSequenceWrapper(torch.max)
#         self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

#     def forward(self, inputs):
#         ipts, labs, _, _, _, _, seqL = inputs
#         sils = ipts[0]

#         # ---- 长度规整 ----
#         if len(sils.size()) == 4:
#             sils = sils.unsqueeze(1)
#         if sils.size()[2] == 1:
#             sils = torch.cat((sils, sils, sils), dim=2)
#         if sils.size()[2] == 2:
#             sils = torch.cat((sils, sils[:, :, -1:, :, :]), dim=2)
#         if sils.size()[2] % 3 != 0:
#             num = sils.size()[2] // 3
#             sils = sils[:, :, :num * 3, :, :]

#         del ipts
#         outs = self.Backbone(sils)                 # [n, c, s, h, w]

#         if seqL is not None:
#             seqL[0] = sils.size()[2]

#         outs_tp, _ = self.TP(outs, seqL, options={"dim": 2})  # [n, c, h, w]
#         feat = self.HPP(outs_tp)                              # [n, c, p]
#         embed_1 = self.FCs(feat)                              # [n, c, p]
#         embed = embed_1

#         n, _, s, h, w = sils.size()
#         bnn = self.BNNecks.fc_bin[:, :, labs].permute(2, 1, 0).contiguous().float()

#         if self.training:
#             embed_2, logits = self.BNNecks(embed_1)
#             retval = {
#                 'training_feat': {
#                     'ctl': {
#                         'embeddings': embed_1,
#                         'labels': labs,
#                         'bnn': bnn,
#                         'iteration': embed_1.new_tensor([self.iteration], dtype=torch.long),
#                     },
#                     'softmax': {'logits': logits, 'labels': labs},
#                 },
#                 'visual_summary': {
#                     'image/sils': sils.reshape(n * s, 1, h, w),
#                 },
#                 'inference_feat': {'embeddings': embed},
#             }
#         else:
#             retval = {
#                 'visual_summary': {
#                     'image/sils': sils.view(n * s, 1, h, w),
#                 },
#                 'inference_feat': {'embeddings': embed},
#             }
#         return retval