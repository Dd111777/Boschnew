# -*- coding: utf-8 -*-
"""
phys_model.py
独立模型定义：
- PhysicsSeqPredictor: Recipe(8) + time → (2,T) 物理时序
- TemporalRegressor: 形貌网 (B,K,T) —— 输入为 [broadcast(static_8) + time_embed(32) + phys(2)] 投影到 d_model 后做 TransformerEncoder，再回归到 K
"""

import math
import torch
import torch.nn as nn

# ---------------- 公共：正余弦位置编码 ----------------
def build_sincos_pos(d_model, T):
    pe = torch.zeros(1, T, d_model)
    position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(pe, requires_grad=False)

# ---------------- 物理网 ----------------
class PhysicsSeqPredictor(nn.Module):
    def __init__(self, d_model=140, nhead=7, num_layers=4, dim_ff=256, dropout=0.1, T=10,
                 in_dim=7, out_dim=2):
        super().__init__()
        self.T = T
        self.out_dim = out_dim
        self.time_mlp = nn.Sequential(nn.Linear(1,16), nn.GELU(), nn.Linear(16,32))
        self.input_proj = nn.Linear(in_dim + 32, d_model)
        self.pos = build_sincos_pos(d_model, T)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_ff, dropout=dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, static_8, time_values):
        B = static_8.size(0)
        if time_values.dim()==1: time_values=time_values.unsqueeze(0).expand(B,-1)
        T = time_values.size(1)
        if T != self.T: raise ValueError(f"T mismatch: {T} vs {self.T}")

        t_embed = self.time_mlp(time_values.unsqueeze(-1))     # (B,T,32)
        s = static_8.unsqueeze(1).expand(B,T,-1)               # (B,T,8)
        x = torch.cat([s,t_embed], dim=-1)                     # (B,T,40)
        x = self.input_proj(x) + self.pos                      # (B,T,d)
        x = self.encoder(x); x = self.norm(x)
        y = self.head(x)                                       # (B,T,2)
        return y.transpose(1,2)                                # (B,2,T)

# ---------------- 物理网 Baseline：MLP ----------------
# phys_model.py

class PhysicsMLPBaseline(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.1, T=10, in_dim=7, out_dim=7):
        super().__init__()
        self.T = T
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 32),
        )

        d_in = in_dim + 32
        layers = []
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_dim

        # ✅ 最后一层直接输出 out_dim（不要先输出 2 再接别的层）
        layers.append(nn.Linear(d_in, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, static_x, time_values):
        B = static_x.size(0)
        if time_values.dim() == 1:
            time_values = time_values.unsqueeze(0).expand(B, -1)
        T = time_values.size(1)
        if T != self.T:
            raise ValueError(f"T mismatch: {T} vs {self.T}")

        t_embed = self.time_mlp(time_values.unsqueeze(-1))   # (B,T,32)
        s = static_x.unsqueeze(1).expand(B, T, self.in_dim)  # (B,T,in_dim)
        x = torch.cat([s, t_embed], dim=-1)                  # (B,T,in_dim+32)

        y = self.mlp(x)                                      # (B,T,out_dim)
        return y.transpose(1, 2)                             # (B,out_dim,T)


# ---------------- 物理网 Baseline：GRU ----------------
class PhysicsGRUBaseline(nn.Module):
    """使用 GRU 建模时间相关性的基线模型。

    接口保持与 PhysicsSeqPredictor 一致：
      forward(static_7, time_values) -> (B, 2, T)
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 1, T: int = 10, dropout: float = 0.0, out_dim: int = 2):
        super().__init__()
        self.T = T
        self.out_dim = out_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 32),
        )
        in_dim = 7 + 32
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, static_7, time_values):
        B = static_7.size(0)
        if time_values.dim() == 1:
            time_values = time_values.unsqueeze(0).expand(B, -1)
        T = time_values.size(1)
        if T != self.T:
            raise ValueError(f"T mismatch: {T} vs {self.T}")
        t_embed = self.time_mlp(time_values.unsqueeze(-1))  # (B,T,32)
        s = static_7.unsqueeze(1).expand(B, T, -1)          # (B,T,7)
        x = torch.cat([s, t_embed], dim=-1)                 # (B,T,7+32)
        out, _ = self.gru(x)                                # (B,T,H)
        y = self.head(out)                                  # (B,T,2)
        return y.transpose(1, 2)                            # (B,2,T)
class TemporalRegressorMLP(nn.Module):
    """
    时间步独立 MLP 基线:
      每个时间步的特征 [static(7), phys(2), t(1)] -> K
      不显式建模时间相关性
    """
    def __init__(self, K: int, hidden_dim: int = 128, num_layers: int = 2, T: int = 10):
        super().__init__()
        self.K = K
        self.T = T
        in_feat = 7 + 2 + 1
        layers = []
        dim_in = in_feat
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.GELU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, K))   # 每个时间步直接输出 K 维
        self.mlp = nn.Sequential(*layers)

    def forward(self, static_8: torch.Tensor, phys_2T: torch.Tensor, tvals: torch.Tensor):
        B = static_8.size(0)
        T = phys_2T.size(-1)
        assert T == self.T, f"T mismatch: got {T}, expect {self.T}"

        phys_bt2 = phys_2T.transpose(1, 2).contiguous()     # (B,T,2)
        static_bt7 = static_8.unsqueeze(1).expand(B, T, 7)  # (B,T,7)

        if tvals.dim() == 1:
            t_bt1 = tvals.unsqueeze(0).expand(B, T).unsqueeze(-1)  # (B,T,1)
        else:
            t_bt1 = tvals.unsqueeze(-1)                             # (B,T,1)
        t_bt1 = t_bt1 / float(T)

        x = torch.cat([static_bt7, phys_bt2, t_bt1], dim=-1)  # (B,T,10)
        y = self.mlp(x)                                       # (B,T,K)
        return y.permute(0, 2, 1).contiguous()                # (B,K,T)
class TemporalRegressorGRU(nn.Module):
    """
    GRU 基线：
      用 GRU 建模时间相关性，再用逐 family 线性头输出 (B,K,T)
    """
    def __init__(self,
                 K: int,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 T: int = 10):
        super().__init__()
        self.K = K
        self.T = T
        in_feat = 7 + 2 + 1

        self.input_proj = nn.Linear(in_feat, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(K)])

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        for h in self.heads:
            nn.init.trunc_normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

    def forward(self, static_8: torch.Tensor, phys_2T: torch.Tensor, tvals: torch.Tensor):
        B = static_8.size(0)
        T = phys_2T.size(-1)
        assert T == self.T, f"T mismatch: got {T}, expect {self.T}"

        phys_bt2 = phys_2T.transpose(1, 2).contiguous()     # (B,T,2)
        static_bt7 = static_8.unsqueeze(1).expand(B, T, 7)  # (B,T,7)

        if tvals.dim() == 1:
            t_bt1 = tvals.unsqueeze(0).expand(B, T).unsqueeze(-1)  # (B,T,1)
        else:
            t_bt1 = tvals.unsqueeze(-1)
        t_bt1 = t_bt1 / float(T)

        x = torch.cat([static_bt7, phys_bt2, t_bt1], dim=-1)  # (B,T,10)
        x = self.input_proj(x)
        h_seq, _ = self.gru(x)                               # (B,T,H)

        outs = []
        for k in range(self.K):
            yk = self.heads[k](h_seq).squeeze(-1)            # (B,T)
            outs.append(yk.unsqueeze(1))
        return torch.cat(outs, dim=1)                        # (B,K,T)

class SinusoidalPositionalEncoding(nn.Module):
    """标准正弦位置编码（不可学习），支持 batch_first=True 的 (B, T, C) 输入。"""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, C)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C)
        """
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)  # (1,T,C) + (B,T,C)


class TemporalRegressor(nn.Module):
    """
    共享时序编码器 + 逐 family 独立输出头
    输入：
      - static_8: (B, 8)
      - phys_2T : (B, 2, T)   # [F_Flux, Ion_Flux]
      - tvals   : (B, T)      # 时间刻度 1..T
    输出：
      - y: (B, K, T)          # K=len(FAMILIES)
    """
    def __init__(self,
                 K: int,
                 d_model: int = 140,
                 nhead: int = 7,
                 num_layers: int = 4,
                 dim_ff: int = 256,
                 dropout: float = 0.1,
                 T: int = 10):
        super().__init__()
        self.K = K
        self.T = T
        self.d_model = d_model

        # 每个时间步的原始特征： 7(静态广播) + 2(物理) + 1(time_scalar) = 10
        in_feat = 7 + 2 + 1
        self.proj_in = nn.Linear(in_feat, d_model)

        # 位置编码（正弦）
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=T+7)

        # Transformer 编码器
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 逐 family 独立输出头：把每个时间步的 d_model → 1
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(K)])

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)
        for h in self.heads:
            nn.init.trunc_normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

    def forward(self, static_8: torch.Tensor, phys_2T: torch.Tensor, tvals: torch.Tensor):
        """
        static_8: (B,8)
        phys_2T : (B,2,T)
        tvals   : (B,T)
        return  : (B,K,T)
        """
        B = static_8.size(0)
        T = phys_2T.size(-1)
        assert T == self.T, f"T mismatch: got {T}, expect {self.T}"

        # 构造每个时间步的输入特征
        # phys → (B,T,2)
        phys_bt2 = phys_2T.transpose(1, 2).contiguous()  # (B,T,2)
        # static 广播 → (B,T,7)
        static_bt8 = static_8.unsqueeze(1).expand(B, T, 7)
        # tvals → (B,T,1)（可做简单归一化）
        # Handle both (T,) and (B,T) shapes for tvals
        if tvals.dim() == 1:
            tvals_bt1 = tvals.unsqueeze(0).expand(B, T).unsqueeze(-1)  # (B,T,1)
        else:
            tvals_bt1 = tvals.unsqueeze(-1)  # (B,T,1)
        tvals_bt1 = tvals_bt1 / float(T)  # 缩放到 0..1

        x = torch.cat([static_bt8, phys_bt2, tvals_bt1], dim=-1)  # (B,T,10)
        x = self.proj_in(x)                                       # (B,T,d_model)
        x = self.pos(x)                                           # 位置编码
        x = self.encoder(x)                                       # (B,T,d_model)

        # 逐 family 输出
        outs = []
        for k in range(self.K):
            yk = self.heads[k](x).squeeze(-1)   # (B,T)
            outs.append(yk.unsqueeze(1))        # (B,1,T)
        y = torch.cat(outs, dim=1)              # (B,K,T)
        return y