# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from ..base_model import BaseModel
# from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks


# class MultiTemporalDiffAdaptive(nn.Module):
#     """
#     同时计算 diff1(t+1-t), diff2(t+2-t), diff3(t+3-t)三路差分
#     自适应学习权重，输出2路加权差分特征
#     x: [n, c, s, h, w]
#     return: diff_out1, diff_out2  shape [n,c,s,h,w]
#     """
#     def __init__(self, channel_dim, eps=1e-5):
#         super().__init__()
#         self.c = channel_dim
#         self.eps = eps
#         self.norm = nn.LayerNorm(channel_dim, eps=eps)
#         # [output_diff_num=2, total_diff_kind=3]
#         self.diff_weight = nn.Parameter(torch.randn(2, 3) * 0.01)

#     def _calc_diff1(self, x):
#         n, c, s, h, w = x.shape
#         if s <= 1:
#             return torch.zeros_like(x)
#         d = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
#         pad = torch.zeros_like(x[:, :, :1, :, :])
#         d = torch.cat([pad, d], dim=2)
#         return self._norm_scale(d)

#     def _calc_diff2(self, x):
#         n, c, s, h, w = x.shape
#         if s <= 2:
#             return torch.zeros_like(x)
#         d = x[:, :, 2:, :, :] - x[:, :, :-2, :, :]
#         pad = torch.zeros_like(x[:, :, :2, :, :])
#         d = torch.cat([pad, d], dim=2)
#         return self._norm_scale(d)

#     def _calc_diff3(self, x):
#         n, c, s, h, w = x.shape
#         if s <= 3:
#             return torch.zeros_like(x)
#         d = x[:, :, 3:, :, :] - x[:, :, :-3, :, :]
#         pad = torch.zeros_like(x[:, :, :3, :, :])
#         d = torch.cat([pad, d], dim=2)
#         return self._norm_scale(d)

#     def _norm_scale(self, diff):
#         diff = diff.permute(0, 2, 3, 4, 1).contiguous()
#         diff = self.norm(diff)
#         diff = diff.permute(0, 4, 1, 2, 3).contiguous()
#         diff = diff * 0.25
#         return diff

#     def forward(self, x, seqL=None):
#         if seqL is None:
#             return self._forward_chunk(x)
#         seq_lens = seqL[0].data.cpu().numpy().tolist()
#         start = 0
#         out1_list, out2_list = [], []
#         for curr_seq_len in seq_lens:
#             chunk = x.narrow(2, start, curr_seq_len)
#             o1, o2 = self._forward_chunk(chunk)
#             out1_list.append(o1)
#             out2_list.append(o2)
#             start += curr_seq_len
#         diff_out1 = torch.cat(out1_list, dim=2)
#         diff_out2 = torch.cat(out2_list, dim=2)
#         return diff_out1, diff_out2

#     def _forward_chunk(self, x):
#         d1 = self._calc_diff1(x)
#         d2 = self._calc_diff2(x)
#         d3 = self._calc_diff3(x)
#         diffs = torch.stack([d1, d2, d3], dim=0)  # [3, n, c, s, h, w]

#         w = F.softmax(self.diff_weight, dim=-1)  # [2,3]
#         diff_out1 = w[0, 0] * d1 + w[0, 1] * d2 + w[0, 2] * d3
#         diff_out2 = w[1, 0] * d1 + w[1, 1] * d2 + w[1, 2] * d3
#         return diff_out1, diff_out2


# class DifferentialAttention(nn.Module):
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
#         for heads in range(min(preferred_heads, in_channels), 0, -1):
#             if in_channels % heads == 0 and (in_channels // heads) % 2 == 0:
#                 return heads
#         raise ValueError(
#             "DifferentialAttention requires an even per‑head dimension, "
#             f"but got in_channels={in_channels}"
#         )

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
#         start = 0
#         chunks = []
#         for curr_seq_len in seq_lens:
#             chunk = x.narrow(2, start, curr_seq_len)
#             chunks.append(self._forward_chunk(chunk))
#             start += curr_seq_len
#         return torch.cat(chunks, dim=2)


# class baseline_diff(BaseModel):
#     def build_network(self, model_cfg):
#         self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
#         self.Backbone = SetBlockWrapper(self.Backbone)
#         self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
#         self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
#         self.TP = PackSequenceWrapper(torch.max)
#         self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

#         out_channels = model_cfg['SeparateFCs']['in_channels']
#         self.MultiDiff = MultiTemporalDiffAdaptive(channel_dim=out_channels)

#         # concat: original + diff1 + diff2 → 3*C，送入差分注意力
#         cat_channels = out_channels * 3
#         self.DiffAttn = DifferentialAttention(in_channels=cat_channels, num_heads=4)
#         # 注意力输出降维回到原始通道数
#         self.FuseConv = nn.Conv3d(cat_channels, out_channels, kernel_size=1, bias=False)

#     def forward(self, inputs):
#         ipts, labs, _, _, _, _, seqL = inputs
#         sils = ipts[0]

#         if len(sils.size()) == 4:
#             sils = sils.unsqueeze(1)
#         if sils.size()[2] == 1:
#             sils = torch.cat((sils, sils, sils), dim=2)
#         if sils.size()[2] == 2:
#             sils = torch.cat((sils, sils[:, :, -1, :, :].unsqueeze(2)), dim=2)
#         if sils.size()[2] % 3 != 0:
#             num = sils.size()[2] // 3
#             sils = sils[:, :, :num * 3, :, :]

#         del ipts
#         outs = self.Backbone(sils)  # [n, c, s, h, w]

#         # 多路自适应差分
#         diff_feat1, diff_feat2 = self.MultiDiff(outs, seqL)
#         cat_feat = torch.cat([outs, diff_feat1, diff_feat2], dim=1)  # [n,3c,s,h,w]

#         # 差分注意力 + 1×1 Conv3d融合降维
#         attn_out = self.DiffAttn(cat_feat, seqL)
#         outs = self.FuseConv(attn_out)  # [n,c,s,h,w]

#         if seqL is not None:
#             seqL[0] = sils.size()[2]

#         outs_tp, indice = self.TP(outs, seqL, options={"dim": 2})  # [n, c, h, w]
#         feat = self.HPP(outs_tp)  # [n, c, p]

#         embed_1 = self.FCs(feat)  # [n, c, p]
#         embed = embed_1

#         n, _, s, h, w = sils.size()
#         bnn = self.BNNecks.fc_bin[:, :, labs].permute(2, 1, 0).contiguous().float()  # [n,c,p]

#         if self.training:
#             embed_2, logits = self.BNNecks(embed_1)
#             retval = {
#                 'training_feat': {
#                     'ctl': {'embeddings': embed_1, 'labels': labs, 'bnn': bnn,
#                             'iteration': embed_1.new_tensor([self.iteration], dtype=torch.long)},
#                     'softmax': {'logits': logits, 'labels': labs, },
#                 },
#                 'visual_summary': {
#                     'image/sils': sils.reshape(n * s, 1, h, w)
#                 },
#                 'inference_feat': {
#                     'embeddings': embed
#                 }
#             }
#         else:
#             retval = {
#                 'visual_summary': {
#                     'image/sils': sils.view(n * s, 1, h, w)
#                 },
#                 'inference_feat': {
#                     'embeddings': embed
#                 }
#             }
#         return retval