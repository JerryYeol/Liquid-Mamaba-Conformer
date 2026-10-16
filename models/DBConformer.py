'''
=================================================
coding:utf-8
@Time:      2025/4/13 17:06
@File:      DBConformer.py
@Author:    Ziwei Wang
@Function:  Original model code of "DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding"

Extended (liquid + mamba variant):
  * temporal branch: MHA -> liquid-gated attention (LGA, qkv_coef)
  * temporal branch: learnable PE optional swapped for ContinuousTimeEmbedder(t)
  * spatial branch:  S-Conformer transformer stack -> bidirectional CMamba
                     (selective SSM with parallel scan, vendored under cmamba/)
  * dual-branch concat -> closed-form continuous-time (CfC) block -> classifier
=================================================
'''

import math
import os
import sys
import types

from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor
from einops import rearrange
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# The liquid attention package lives next to the repo root; make it importable
# no matter which directory the launcher was started from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from liquid_gated_attention.liquid_mixer import LiquidMixer
    _LIQUID_AVAILABLE = True
except Exception as _lga_err:  # keep the baseline model usable if it is missing
    LiquidMixer = None
    _LIQUID_AVAILABLE = False
    print('[DBConformer] liquid_gated_attention not importable (%s); '
          'falling back to standard MHA.' % _lga_err)

# CMamba core, vendored under ``cmamba/``: this is the spatial branch's
# backbone. It is a copy rather than an import of ``CMamba-main`` because that
# directory cannot go on ``sys.path`` -- see ``cmamba/__init__.py``.
from cmamba import CMambaEncoder


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Stem(nn.Module):
    def __init__(self, data_name, in_planes, out_planes = 64, kernel_size=63, patch_size=125, radix=2):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.data_name = data_name

        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups = radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size, 1, groups=self.out_planes, padding=kernel_size // 2, bias=False,),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))
            kernel_size //= 2

        self.interFre = InterFre()

        self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(0.5)  #

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]
        out = self.interFre(out)
        if self.data_name != 'MI1-7':  # MI1-7 is the Blankertz2007 dataset in our paper
            out = out[:, :, :-1]  # Example: 14001 has 1001 time points, we exclude the final point
        out = self.downSampling(out)
        out = self.dp(out)
        return out


class PatchEmbeddingTemporal(nn.Module):
    def __init__(self, data_name, in_planes, out_planes, kernel_size, radix, patch_size, time_points, num_classes):
        '''
        Outputs patch embeddings of shape (B, P, D)
        '''
        super().__init__()
        self.data_name = data_name
        self.stem = Stem(
            data_name=self.data_name,
            in_planes=in_planes * radix,
            out_planes=out_planes,
            kernel_size=kernel_size,
            patch_size=patch_size,
            radix=radix
        )
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, T)
        out = self.stem(x)         # (B, D, P)
        out = out.permute(0, 2, 1) # → (B, P, D)
        return out


class DSCA(nn.Module):
    """
    Dynamic Channel Sparse Attention (动态通道稀疏注意力)
    核心：根据输入特征自适应预测稀疏度，结合STE直传估计实现可训练的硬稀疏注意力
    输入输出尺寸完全一致，即插即用
    Args:
        dim: 输入/输出通道数
        num_heads: 注意力头数
        reduction: 频域SE模块的通道压缩比
        ste_mask_tau: STE直传估计的温度系数
        eps: 数值稳定项
    """
    def __init__(self, dim, num_heads=4, reduction=4, ste_mask_tau=0.15, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ste_mask_tau = ste_mask_tau
        self.eps = eps
        hidden_dim = max(dim // 8, 16)

        # ====================== 动态稀疏门控模块（对应上方结构图）======================
        self.k_dconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # ====================== 动态通道稀疏注意力模块（对应下方结构图）======================
        # QKV投影 + 深度卷积增强
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

        # 可学习参数：温度系数、头权重
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.head_weights = nn.Parameter(torch.ones(self.num_heads))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # ====================== 局部特征分支（频域引导增强）======================
        self.local_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.local_pw = nn.Conv2d(dim, dim, 1)
        self.local_act = nn.GELU()
        # 频域引导的SE模块（FGSE）
        hidden = max(dim // reduction, 1)
        self.se_reduce = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.se_expand = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

        # ====================== 局部-全局融合 ======================
        self.fusion = nn.Conv2d(dim * 2, dim, 1)

    def _predict_k(self, x):
        """
        动态稀疏门控：预测每个样本的注意力稀疏度k
        对应上方结构图：DConv → MLP → Sigmoid → GAP → 输出稀疏度因子
        """
        gate = self.k_mlp(self.k_dconv(x))
        return F.adaptive_avg_pool2d(gate, 1).mean(dim=1)  # [B, 1, 1, 1] → [B]

    def _build_hard_mask(self, attn, k_values):
        """
        构建硬稀疏掩码：保留每个查询的top-k个键值对
        Args:
            attn: 注意力矩阵 [B, H, N, N]
            k_values: 每个样本的保留通道数 [B]
        Returns:
            hard_mask: 二值稀疏掩码 [B, H, N, N]
        """
        b = attn.shape[0]
        k_max = min(int(k_values.max().item()), self.head_dim, attn.shape[-1])
        # 取top-k个注意力权重的索引
        top_k_indices = torch.topk(attn, k=k_max, dim=-1).indices
        # 生成有效掩码（处理不同样本k值不同的情况）
        arange_k = torch.arange(k_max, device=attn.device).view(1, 1, 1, 1, -1)
        valid = arange_k < k_values.view(b, 1, 1, 1, 1)
        # 构建二值掩码
        sparse_mask = torch.zeros_like(attn.unsqueeze(2))
        fill_values = torch.ones_like(top_k_indices, dtype=attn.dtype).unsqueeze(2)
        fill_values = fill_values.masked_fill(~valid, 0.0)
        sparse_mask.scatter_(-1, top_k_indices.unsqueeze(2), fill_values)
        return sparse_mask.squeeze(2)

    def _continuous_threshold(self, attn, k_ratio):
        """计算连续阈值，用于生成软掩码"""
        attn_mean = attn.mean(dim=-1, keepdim=True)
        attn_std = attn.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return attn_mean + (1.0 - k_ratio) * attn_std

    def _ste_dcsa(self, x, sparsity_factor):
        """
        动态通道稀疏注意力核心计算（结合STE直传估计）
        对应下方结构图的完整流程：QKV生成 → 注意力矩阵 → 动态稀疏掩码 → 加权求和
        """
        b, _, h, w = x.shape
        # 稀疏度因子归一化到[eps, 1-eps]
        sparsity_factor = sparsity_factor.clamp(min=self.eps, max=1.0 - self.eps).view(-1)
        # 计算每个样本保留的通道数k
        k_values_float = 1.0 + (self.head_dim - 1.0) * sparsity_factor
        k_values = torch.floor(k_values_float).long().clamp_(min=1, max=self.head_dim)

        # QKV生成 + 深度卷积增强
        q = self.q_dw(self.q_proj(x))
        k = self.k_dw(self.k_proj(x))
        v = self.v_dw(self.v_proj(x))
        # 维度重排为多头格式 [B, H, D, N]
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        # 归一化 + 注意力矩阵计算
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        soft_prob = F.softmax(attn, dim=-1)

        # ====================== STE直传估计（核心创新）======================
        k_ratio = (k_values_float / self.head_dim).view(b, 1, 1, 1)
        # 硬掩码（前向传播生效，反向传播梯度为0）
        hard_mask = self._build_hard_mask(attn, k_values)
        # 软掩码（反向传播提供梯度）
        threshold = self._continuous_threshold(attn, k_ratio)
        soft_mask = torch.sigmoid((attn - threshold) / max(self.ste_mask_tau, self.eps))
        # STE直传：前向用硬掩码，反向用软掩码梯度
        mask = hard_mask + soft_mask - soft_mask.detach()

        # 稀疏注意力加权
        prob = soft_prob * mask
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # 归一化保证权重和为1
        out = prob @ v

        # 多头加权 + 维度还原
        out = out * self.head_weights.view(1, self.num_heads, 1, 1)
        out = out.reshape(b, self.num_heads * self.head_dim, h, w)
        return self.project_out(out)

    def _fgse(self, x):
        """
        频域引导的SE模块（Frequency-Guided Squeeze-Excitation）
        核心：利用FFT幅值谱生成空间权重，增强高频细节特征
        """
        _, _, h, w = x.shape
        with torch.no_grad():  # 频域计算无梯度，提升效率
            # 实值FFT → 幅值谱 → 逆FFT得到空间显著性图
            fft_map = torch.fft.rfft2(x, norm='ortho')
            fft_map_abs = torch.abs(fft_map)
            spatial_map = torch.fft.irfft2(fft_map_abs, s=(h, w), norm='ortho')
            spatial_weights = torch.sigmoid(spatial_map.mean(dim=1, keepdim=True))
        # 空间加权 + 通道SE
        squeezed = F.adaptive_avg_pool2d(x * spatial_weights, 1)
        excited = torch.sigmoid(self.se_expand(F.relu(self.se_reduce(squeezed), inplace=True)))
        return x * excited.expand_as(x)

    def forward(self, x):
        # The spatial branch hands this over as (B, C, D) -- C electrodes, D
        # embedding dim -- the same 3-D shape the SEBlock it replaces took, so
        # adapt here and keep the module itself 2-D.  The view puts the
        # embedding dim on the feature-channel axis and the electrodes on the
        # token grid, which lands "channel" in the module's name on the
        # electrode axis -- that is the axis the sparse attention keeps top-k
        # of, and the axis the depthwise convs and the FFT then run along.
        if x.dim() == 3:
            out = self._forward4d(x.permute(0, 2, 1).contiguous().unsqueeze(-1))
            return out.squeeze(-1).permute(0, 2, 1).contiguous()  # → (B, C, D)
        return self._forward4d(x)

    def _forward4d(self, x):
        # 步骤1：动态预测稀疏度因子
        sparsity_factor = self._predict_k(x)
        # 步骤2：全局动态稀疏注意力分支
        global_out = self._ste_dcsa(x, sparsity_factor)
        # 步骤3：局部频域增强分支
        local_out = self.local_dw(x)
        local_out = self.local_act(local_out)
        local_out = self._fgse(local_out)  # 频域引导增强
        local_out = self.local_pw(local_out)
        # 步骤4：局部-全局特征融合
        return self.fusion(torch.cat([local_out, global_out], dim=1))


class PatchEmbeddingSpatial(nn.Module):
    def __init__(self, spa_dim, emb_size=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),  # Output: (B*C, spa_dim, T')
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),  # Output: (B*C, spa_dim, 1)
            nn.Flatten(),             # → (B*C, 16)
            nn.Linear(spa_dim, emb_size)   # → (B*C, emb_size)
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        x = x.unsqueeze(2)         # (B, C, 1, T)
        x = x.reshape(B * C, 1, T)  # → (B*C, 1, T)
        x = self.encoder(x)        # → (B*C, emb_size)
        x = x.view(B, C, -1)       # → (B, C, emb_size)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out

class Gate_FC(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__()
        self.fc = nn.Linear(emb_size * 2, emb_size)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


# ======================================================================
#  Continuous-time positional encoding
# ======================================================================

class ContinuousTimeEmbedder(nn.Module):
    """Sinusoidal encoding of real-valued timestamps t (B, P) or (B, P, 1).

    Unlike a learnable (1, P, D) table this carries the *actual* time of each
    token, so it stays meaningful when the patches are not uniformly spaced.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        pe = torch.zeros(t.size(0), t.size(1), self.d_model, device=t.device, dtype=t.dtype)
        pe[..., 0::2] = torch.sin(t * self.div_term)
        pe[..., 1::2] = torch.cos(t * self.div_term)
        return pe


# ======================================================================
#  Liquid-gated attention in the temporal branch
# ======================================================================

class LiquidTemporalEncoder(nn.Module):
    """Temporal branch encoder using liquid-gated attention (LGA).

    Wraps ``liquid_gated_attention.liquid_mixer.LiquidMixer``. The liquid gate
    derives a per-head, per-timestep scalar from the input:

        theta = sigmoid(W x)          (AdaLerp: interpolated with the previous step)
        f     = delta_t * theta       (time-aware, normalised over the sequence)
        S     = cumsum(f)             (the "liquid" state along time)
        q_coef = 1/exp(S),  k_coef = 1-exp(-f),  v_coef = exp(S)

    which modulates Q, K, V before the (linear, softmax-free) attention, i.e.
    an input-dependent exponential decay over the 8 temporal patches -- the
    liquid time constant tracking ERD/ERS evolution.

    ``delta_t`` note: with the default norm_gate=True the pre-activation is
    divided by its sum over the sequence, so a *constant* delta_t cancels out
    and only the per-token *pattern* of delta_t matters. Uniform patches
    therefore make delta_t a no-op; pass non-uniform ``token_times`` (or set
    norm_gate=False) to make the physical time step play a role.
    """

    def __init__(self, depth, emb_size, n_heads, token_times, first_dt='copy',
                 norm_gate=True, use_output_gate=True, adaptive_mu=True, mu_val=0.5,
                 head_decoupled=True, input_aware_theta=True, time_aware_gate=True,
                 channel_mixer_type='swiglu', norm_type='rms_norm', norm_pos='pre_norm',
                 multiple_of=16, causal=False, scan=False):
        super().__init__()
        if emb_size % n_heads != 0:
            raise ValueError(f'emb_size ({emb_size}) must be divisible by n_heads ({n_heads})')
        cfg = types.SimpleNamespace(
            depth=depth, d_model=emb_size, n_heads=n_heads, seq_len=len(token_times),
            causal=causal, scan=scan, temporal_mixer_type='liquid',
            use_output_gate=use_output_gate, norm_gate=norm_gate, adaptive_mu=adaptive_mu,
            mu_val=mu_val, head_decoupled=head_decoupled,
            input_aware_theta=input_aware_theta, time_aware_gate=time_aware_gate,
            channel_mixer_type=channel_mixer_type, norm_type=norm_type,
            norm_pos=norm_pos, multiple_of=multiple_of,
        )
        self.mixer = LiquidMixer(cfg)
        self.first_dt = first_dt
        self.register_buffer('token_times', torch.as_tensor(token_times, dtype=torch.float32))

    def delta_t(self, batch_size, n_tokens, device):
        """Per-token time step, shaped (B, 1, N) for TemporalMixer's broadcast."""
        t = self.token_times[:n_tokens].to(device)
        dt = torch.zeros_like(t)
        dt[1:] = t[1:] - t[:-1]
        if self.first_dt == 'copy' and n_tokens > 1:
            dt[0] = dt[1]   # otherwise dt[0] == 0 zeroes the first token's key
        return dt.view(1, 1, -1).expand(batch_size, 1, -1)

    def forward(self, x):  # x: (B, N, D)
        dt = self.delta_t(x.size(0), x.size(1), x.device)
        return self.mixer(x, dt, key_padding_mask=None)


# ======================================================================
#  Closed-form continuous-time block
# ======================================================================

class CFCCell(nn.Module):
    """Closed-form continuous-time (CfC) cell (Hasani et al., 2022).

        h(t) = sigma(-f(x, h) * t) * g(x, h) + (1 - sigma(-f(x, h) * t)) * h

    ``f`` is a positive time constant (softplus), ``g`` the candidate state.
    """

    def __init__(self, input_size, hidden_size, backbone_units=None, dropout=0.0):
        super().__init__()
        backbone_units = backbone_units or max(hidden_size, 32)
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, backbone_units),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(backbone_units, backbone_units),
            nn.LeakyReLU(0.1),
        )
        self.ff_g = nn.Linear(backbone_units, hidden_size)   # candidate state g
        self.ff_f = nn.Linear(backbone_units, hidden_size)   # time constant  f

    def forward(self, x, h, ts):
        z = self.backbone(torch.cat([x, h], dim=-1))
        g = torch.tanh(self.ff_g(z))
        f = F.softplus(self.ff_f(z))
        gate = torch.sigmoid(-f * ts)
        return gate * g + (1.0 - gate) * h


class CFCBlock(nn.Module):
    """CfC block placed between the fused dual-branch features and the head.

    The fused vector is not a sequence, so it drives ``n_steps`` closed-form
    steps from a learnable initial state with a learnable step size -- i.e. a
    continuous-time gated mixing of the two branch outputs.
    """

    def __init__(self, dim, n_steps=1, delta_t=1.0, learnable_dt=True,
                 backbone_units=None, dropout=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.cell = CFCCell(dim, dim, backbone_units=backbone_units, dropout=dropout)
        self.h0 = nn.Parameter(torch.zeros(1, dim))
        if learnable_dt:
            self.log_dt = nn.Parameter(torch.tensor(math.log(delta_t), dtype=torch.float32))
        else:
            self.register_buffer('log_dt', torch.tensor(math.log(delta_t), dtype=torch.float32))

    def forward(self, x):
        ts = torch.exp(self.log_dt)
        h = self.h0.to(x.dtype).expand(x.size(0), -1)
        for _ in range(self.n_steps):
            h = self.cell(x, h, ts)
        return h


# ======================================================================
#  Model
# ======================================================================

def spatial_mamba_config(emb_size, depth, dropout):
    """Hyper-parameters for the CMamba encoder that backs the spatial branch.

    ``CMambaEncoder`` reads its settings off an attribute bag instead of taking
    arguments (same shape as ``LiquidTemporalEncoder``'s ``cfg`` above), so they
    are collected here. Everything is sized for this branch rather than for
    CMamba's long-term-forecasting defaults (d_model=512, d_ff=2048), which
    assume sequences three orders of magnitude longer than ``C`` electrodes.
    """
    return types.SimpleNamespace(
        d_model=emb_size,
        # 2x expansion is the canonical Mamba ratio. CMamba's 4x is tuned for
        # d_model=512; at emb_size=40 it buys nothing and only costs parameters.
        d_ff=2 * emb_size,
        # Mamba's rule of thumb for the low-rank delta projection.
        dt_rank=max(1, math.ceil(emb_size / 16)),
        d_state=16,
        dt_scale=1.0,
        dt_init='random',
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init_floor=1e-4,
        bias=True,
        # The parallel scan, i.e. CMamba's own contribution over Mamba. It pads
        # the token axis to a power of two internally and slices back, so 22
        # electrodes are handled by scanning 32 -- the padded slots scan to zero
        # and are discarded.
        pscan=True,
        e_layers=depth,
        dropout=dropout,
    )


class BiCMambaEncoder(nn.Module):
    """Bidirectional CMamba over the electrode axis.

        fwd = cmamba(x)                        # electrodes in montage order
        bwd = cmamba(x.flip(1)).flip(1)        # electrodes reversed
        out = g * fwd + (1 - g) * bwd          # g = sigmoid(Linear([fwd; bwd]))

    The selective scan is order-sensitive, but the electrode graph is not:
    functional connectivity in MI is symmetric (the C3/C4 contralateral pair),
    so a single unidirectional sweep imposes a direction on a relation that has
    none. Running both orders and letting a gate choose per electrode, per
    channel, is the symmetric information flow.

    The two directions have independent weights -- tying them would make ``out``
    invariant to the flip and defeat the point. The gate is zero-initialised, so
    training starts from the plain average of the two sweeps and learns the
    weighting from there.
    """

    def __init__(self, configs):
        super().__init__()
        self.fwd_encoder = CMambaEncoder(configs)
        self.bwd_encoder = CMambaEncoder(configs)
        self.gate = nn.Linear(2 * configs.d_model, configs.d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):  # (B, C, D) -> (B, C, D)
        fwd = self.fwd_encoder(x)
        bwd = self.bwd_encoder(x.flip(1)).flip(1)
        g = torch.sigmoid(self.gate(torch.cat([fwd, bwd], dim=-1)))
        return g * fwd + (1.0 - g) * bwd


class DBConformer(nn.Module):
    def __init__(self, args, emb_size=40, tem_depth=5, chn_depth=5, chn=22, n_classes=2):
        super().__init__()

        def opt(name, default):
            return getattr(args, name, default)

        self.embedding = PatchEmbeddingTemporal(
            data_name=args.data_name,
            in_planes=args.chn,  # number of channels
            out_planes=emb_size,  # Default 40
            kernel_size=63,
            radix=1,
            patch_size=args.patch_size,  # needs to be divisible by the number of time points
            time_points=args.time_sample_num,  # number of time points
            num_classes=args.class_num  # number of classes
        )
        self.channel_embedding = PatchEmbeddingSpatial(spa_dim=args.spa_dim, emb_size=emb_size)  # Default 16
        # Stem.forward drops the last time point before average pooling (except
        # for MI1-7), so the real patch count is floor((time_sample_num - 1) /
        # patch_size).  Plain integer division only agrees when patch_size does
        # not divide time_sample_num -- 4000 // 125 is 32 where the Stem really
        # produces 31 -- so mirror the Stem here and any patch_size is safe.
        if args.data_name == 'MI1-7':
            self.P = args.time_sample_num // args.patch_size
        else:
            self.P = (args.time_sample_num - 1) // args.patch_size
        self.C = args.chn  # number of channels
        self.D = emb_size
        self.gate_flag = args.gate_flag  # Default False, due to the reduced performance
        self.posemb_flag = args.posemb_flag  # Default True
        self.branch = args.branch  # Default 'all', options=[all, temporal]
        self.chn_atten_flag = args.chn_atten_flag  # Default True

        # ---------------- temporal token times (uniform by default) --------
        patch_seconds = args.patch_size / float(args.sample_rate)
        default_times = [i * patch_seconds for i in range(self.P)]
        token_times = opt('temporal_token_times', None) or default_times
        assert len(token_times) == self.P, \
            f'expected {self.P} temporal token times, got {len(token_times)}'

        # ---------------- positional encodings -----------------------------
        self.ctpe = None
        if args.posemb_flag:
            if opt('ctpe_flag', False):
                self.ctpe = ContinuousTimeEmbedder(emb_size)
                times = torch.as_tensor(token_times, dtype=torch.float32).view(1, -1)
                self.register_buffer('pos_times', times)
            else:
                self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        # ---------------- attention ----------------------------------------
        n_heads = opt('lga_heads', 4)
        self.use_lga = opt('lga_flag', True) and _LIQUID_AVAILABLE
        if self.use_lga:
            self.temporal_transformer = LiquidTemporalEncoder(
                depth=tem_depth,
                emb_size=emb_size,
                n_heads=n_heads,
                token_times=token_times,
                first_dt=opt('lga_first_dt', 'copy'),
                norm_gate=opt('lga_norm_gate', True),
                use_output_gate=opt('lga_output_gate', True),
                adaptive_mu=opt('lga_adaptive_mu', True),
                mu_val=opt('lga_mu', 0.5),
                head_decoupled=opt('lga_head_decoupled', True),
                input_aware_theta=opt('lga_input_aware_theta', True),
                time_aware_gate=opt('lga_time_aware_gate', True),
            )
        else:
            self.temporal_transformer = TransformerEncoder(tem_depth, emb_size)
        # Spatial branch: bidirectional CMamba over the C electrode tokens,
        # standing in for the original S-Conformer transformer stack. Drop-in at
        # the same contract -- (B, C, D) in, (B, C, D) out -- so the fusion and
        # head below are untouched.
        self.spatial_transformer = BiCMambaEncoder(
            spatial_mamba_config(emb_size, chn_depth, args.dropoutRate))
        # Dynamic sparse channel attention on the spatial branch, in the slot
        # the unused SEBlock used to sit in (defined above).  Drop-in at the
        # same contract as that block -- (B, C, D) in, (B, C, D) out -- so it
        # reweights the electrodes ahead of the branch's BiCMamba and leaves
        # everything downstream untouched.
        # Gated by chn_atten_flag, the flag that already controls this branch's
        # channel attention, and default on exactly like it.
        # 4 heads over the embedding dim, as the LGA attention above uses;
        # fall back to the largest head count that divides emb_size so an odd
        # emb_size degrades instead of failing inside the reshape.
        dsca_heads = opt('dsca_heads', None) or max(h for h in (1, 2, 4) if emb_size % h == 0)
        self.dsca = DSCA(emb_size, num_heads=dsca_heads) if args.chn_atten_flag else None

        # ---------------- fusion / head ------------------------------------
        fuse_gated = args.gate_flag or self.branch == 'temporal' or self.branch == 'spatial'
        if fuse_gated:
            fuse_dim = emb_size
            # Only the branch == 'all' path in forward reads this gate, so it
            # is built only when that path can actually be taken -- otherwise
            # it is a parameter group that never receives a gradient.
            if self.gate_flag and self.branch == 'all':
                self.gate_fc = Gate_FC(emb_size)  # dual-branch weighting aggregation
        else:
            fuse_dim = emb_size * 2
            if args.chn_atten_flag:
                self.spatial_attn_pool = nn.Sequential(
                    nn.Linear(emb_size, emb_size),  # D → D
                    nn.Tanh(),
                    nn.Linear(emb_size, 1),  # D → 1 (score per channel)
                )

        self.cfc = None
        if opt('cfc_flag', True):
            self.cfc = CFCBlock(
                fuse_dim,
                n_steps=opt('cfc_steps', 1),
                delta_t=opt('cfc_delta_t', 1.0),
                learnable_dt=opt('cfc_learnable_dt', True),
                backbone_units=opt('cfc_units', None),
                dropout=opt('cfc_dropout', 0.0),
            )
        self.classifier = ClassificationHead(fuse_dim, n_classes)

    def forward(self, x):  # x: (B, 1, C, T)
        if x.dim() == 4:
            x = x.squeeze(1)  # → (B, C, T)

        x_embed = self.embedding(x)  # → (B, P, D)
        x_embed_spatial = self.channel_embedding(x)  # (B, C, D)

        if self.posemb_flag:
            if self.ctpe is not None:
                x_embed = x_embed + self.ctpe(self.pos_times)  # continuous-time PE
            else:
                x_embed = x_embed + self.pos_embedding_temporal  # temporal positional encoding
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial  # spatial positional encoding

        # Temporal Transformer (attention over the P time patches)
        x_temporal = self.temporal_transformer(x_embed)  # (B, P, D)
        # Dynamic sparse channel attention over the electrodes, ahead of the
        # spatial branch's BiCMamba rather than after it: (B, C, D) in, (B, C,
        # D) out, so the transformer attends over already reweighted channels
        # (and reads the spatial PE, which is added above).
        if self.dsca is not None:
            x_embed_spatial = self.dsca(x_embed_spatial)  # (B, C, D) -> (B, C, D)
        # Spatial Transformer (attention over the C electrodes as tokens)
        x_spatial = self.spatial_transformer(x_embed_spatial)  # (B, C, D)

        if self.branch == 'temporal':
            x_fused = x_temporal.mean(dim=1)
        elif self.branch == 'spatial':  # Using S-Conformer-only doesn't work
            x_fused = x_spatial.mean(dim=1)
        elif self.branch == 'all':
            if self.gate_flag:
                # gated-fusion
                gate = torch.sigmoid(
                    self.gate_fc(torch.cat([x_temporal.mean(dim=1), x_spatial.mean(dim=1)], dim=-1)))  # shape: (B, D)
                x_fused = gate * x_spatial.mean(dim=1) + (1 - gate) * x_temporal.mean(dim=1)
            else:
                if self.chn_atten_flag:
                    # Attention Scores
                    x_t = x_temporal.mean(dim=1)
                    attn_scores = self.spatial_attn_pool(x_spatial)  # (B, C, 1)
                    attn_weights = torch.softmax(attn_scores, dim=1)  # (B, C, 1)
                    x_s = torch.sum(attn_weights * x_spatial, dim=1)  # (B, D)
                    x_fused = torch.cat([x_t, x_s], dim=-1)  # → (B, 2*D)
                else:
                    # Mean pooling
                    x_fused = torch.cat([
                        x_temporal.mean(dim=1),
                        x_spatial.mean(dim=1)
                    ], dim=-1)  # → (B, 2*D)

        if self.cfc is not None:
            x_fused = self.cfc(x_fused)

        _, out = self.classifier(x_fused)  # out: (B, n_classes)
        return x_fused, out
