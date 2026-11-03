import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ===================== Dataset =====================
class EyeFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, x1_eye, x1_angle, y):
        self.x1_eye = x1_eye
        self.x1_angle = x1_angle
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x1_eye)

    def __getitem__(self, idx):
        return (self.x1_eye[idx], self.x1_angle[idx]), self.y[idx]


# ===================== 基础模块 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # x:(B,L,C)
        L = x.size(1)
        x = x + self.pe[:L, :].unsqueeze(0).to(x.device)
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ===================== Transformer Blocks =====================
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mlp = MLP(d_model, hidden_dim=4*d_model, dropout=dropout)

    def forward(self, x):
        ln_x = self.ln(x)
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.mlp(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.ln_q  = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.mlp   = MLP(d_model, hidden_dim=4*d_model, dropout=dropout)

    def forward(self, q, k, v):
        qn = self.ln_q(q)
        kn = self.ln_kv(k)
        vn = self.ln_kv(v)
        attn_out, _ = self.attn(qn, kn, vn, need_weights=False)
        q = q + self.drop(attn_out)
        q = q + self.mlp(q)
        return q


class RollingTransformer(nn.Module):
    """相邻 trial 的交叉注意力：输入已做过 self-attn"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(d_model, nhead, dropout)

    def forward(self, trial_i_feat, trial_j_feat):
        # (B,L,C) x 2 -> (B,L,C)
        return self.cross_attn(trial_i_feat, trial_j_feat, trial_j_feat)


# ===================== 单一路径（模态）编码器 =====================
class SinglePathProcessor(nn.Module):
    """
    返回 (B, 39, d_model)，不做最终聚合。
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, target_length,
                 shared_pos=None, shared_self_stack: nn.ModuleList=None, shared_rolling: RollingTransformer=None):
        super().__init__()
        self.target_length = target_length
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = shared_pos if shared_pos is not None else PositionalEncoding(d_model, dropout)

        if shared_self_stack is not None:
            self.self_attn_layers = shared_self_stack
        else:
            self.self_attn_layers = nn.ModuleList([SelfAttentionBlock(d_model, nhead, dropout)
                                                   for _ in range(num_layers)])

        self.rolling_transformer = shared_rolling if shared_rolling is not None \
                                   else RollingTransformer(d_model, nhead, dropout)

        self.pooling = nn.AdaptiveAvgPool1d(1)

    def _pool_trials_to_fixed_len(self, x, lengths=None):
        L = self.target_length
        if isinstance(x, list):
            B = len(x); num_trials = len(x[0]); C = x[0][0].size(-1)
            out = []
            for b in range(B):
                per_trials = []
                for t in range(num_trials):
                    seg = x[b][t]                       # (Ti,C)
                    seg = seg.transpose(0,1).unsqueeze(0)  # (1,C,Ti)
                    seg = F.adaptive_avg_pool1d(seg, L)    # (1,C,L)
                    seg = seg.squeeze(0).transpose(0,1)    # (L,C)
                    per_trials.append(seg)
                out.append(torch.stack(per_trials, dim=0))   # (40,L,C)
            return torch.stack(out, dim=0)                    # (B,40,L,C)
        elif isinstance(x, torch.Tensor):
            B, num_trials, T_max, C = x.shape
            pooled = []
            for b in range(B):
                per_trials = []
                for t in range(num_trials):
                    if lengths is not None:
                        Ti = int(lengths[b, t].item())
                        seg = x[b, t, :Ti, :]
                    else:
                        seg = x[b, t, :, :]
                    seg = seg.transpose(0,1).unsqueeze(0)
                    seg = F.adaptive_avg_pool1d(seg, L)
                    seg = seg.squeeze(0).transpose(0,1)
                    per_trials.append(seg)
                pooled.append(torch.stack(per_trials, dim=0))
            return torch.stack(pooled, dim=0)
        else:
            raise TypeError("x 必须是 list 或 Tensor")

    def forward(self, x, lengths=None):
        """
        x: list[list[Tensor(Ti,C)]] 或 Tensor(B,40,T,C)
        return: pairs (B,39,d_model)
        """
        x_pooled = self._pool_trials_to_fixed_len(x, lengths)      # (B,40,L,Cin)
        B, num_trials, L, Cin = x_pooled.shape

        x_proj = self.input_proj(x_pooled.view(B*num_trials, L, Cin))  # (B*40,L,d_model)
        x_pos  = self.pos_encoding(x_proj)

        x_attn = x_pos
        for blk in self.self_attn_layers:
            x_attn = blk(x_attn)

        x_attn = x_attn.view(B, num_trials, L, self.d_model)       # (B,40,L,C)

        # rolling 相邻 pair
        pair_vecs = []
        for i in range(num_trials - 1):
            ti = x_attn[:, i,   :, :]   # (B,L,C)
            tj = x_attn[:, i+1, :, :]
            co = self.rolling_transformer(ti, tj)                  # (B,L,C)
            vec = self.pooling(co.transpose(1,2)).squeeze(-1)      # (B,C)
            pair_vecs.append(vec)
        pairs = torch.stack(pair_vecs, dim=1)                       # (B,39,C)
        return pairs


# ===================== 主网络 =====================
class RollingTransformerModel(nn.Module):
    """
    两模态 -> (各自编码 -> 39 pair 表示) -> 跨模态交叉注意力(可双向) -> [不在39上做自注意力聚合] -> 展平 -> MLP(7)
    """
    def __init__(self,
                 input_dim_eye=10,
                 input_dim_angle=10,
                 d_model=512,
                 nhead=8,
                 num_layers=2,
                 dropout=0.1,
                 mlp_hidden=256,
                 out_dim=7,
                 target_length=1024,
                 share_backbone=True,          # 是否共享 self-attn 与 rolling
                 bidirectional_fusion=True,    # 是否做双向融合并 concat
                 num_trials=40):               # 试次数（决定 pair 数 = num_trials-1）
        super().__init__()
        self.d_model = d_model
        self.target_length = target_length
        self.share_backbone = share_backbone
        self.bidirectional_fusion = bidirectional_fusion
        self.num_pairs = max(0, num_trials - 1)

        # 可共享的模块
        shared_pos = PositionalEncoding(d_model, dropout) if share_backbone else None
        shared_stack = nn.ModuleList([SelfAttentionBlock(d_model, nhead, dropout)
                                      for _ in range(num_layers)]) if share_backbone else None
        shared_roll = RollingTransformer(d_model, nhead, dropout) if share_backbone else None

        # 两个模态编码器
        self.eye_processor = SinglePathProcessor(
            input_dim_eye, d_model, nhead, num_layers, dropout, target_length,
            shared_pos=shared_pos, shared_self_stack=shared_stack, shared_rolling=shared_roll
        )
        self.angle_processor = SinglePathProcessor(
            input_dim_angle, d_model, nhead, num_layers, dropout, target_length,
            shared_pos=shared_pos, shared_self_stack=shared_stack, shared_rolling=shared_roll
        )

        # 跨模态交叉注意力（在 39 个 pair 维度上做）
        self.fuse_ln_q  = nn.LayerNorm(d_model)
        self.fuse_ln_kv = nn.LayerNorm(d_model)
        self.fuse_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        if bidirectional_fusion:
            self.fuse_attn_rev = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # —— 取消在 39 个 token 上的自注意力聚合 —— #
        # 改为直接展平 tokens 进入 MLP
        flat_in_dim = self.num_pairs * (2 * d_model if bidirectional_fusion else d_model)

        self.output_mlp = nn.Sequential(
            nn.LayerNorm(flat_in_dim),
            nn.Linear(flat_in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, out_dim)
        )

    def forward(self, x, lengths=None):
        """
        x: (xs_eye, xs_angle)，各为 list[list[Tensor(Ti,C)]] 或 Tensor(B,40,T,C)
        return: (B,7)
        """
        xs_eye, xs_angle = x

        eye_pairs   = self.eye_processor(xs_eye,   lengths)  # (B,Pe,C) 期望 Pe=self.num_pairs
        angle_pairs = self.angle_processor(xs_angle, lengths) # (B,Pe,C)

        # 运行时检查：确保 pair 数与设定一致（避免维度不匹配）
        assert eye_pairs.size(1) == self.num_pairs and angle_pairs.size(1) == self.num_pairs, \
            f"Pair 数不一致：eye={eye_pairs.size(1)}, angle={angle_pairs.size(1)}, 期望 {self.num_pairs}。请保证每个样本有 {self.num_pairs+1} 个 trial。"

        # eye <- angle
        q = self.fuse_ln_q(angle_pairs)  # (B,Pe,C)
        k = self.fuse_ln_kv(eye_pairs)
        v = self.fuse_ln_kv(eye_pairs)
        fuse_e2a, _ = self.fuse_attn(q, k, v, need_weights=False)  # (B,Pe,C)

        if self.bidirectional_fusion:
            # angle <- eye
            q2 = self.fuse_ln_q(eye_pairs)
            k2 = self.fuse_ln_kv(angle_pairs)
            v2 = self.fuse_ln_kv(angle_pairs)
            fuse_a2e, _ = self.fuse_attn_rev(q2, k2, v2, need_weights=False)  # (B,Pe,C)

            # 在特征维拼接，再展平
            fused = torch.cat([fuse_e2a, fuse_a2e], dim=-1)  # (B,Pe,2C)
        else:
            fused = fuse_e2a  # (B,Pe,C)

        # —— 不做 39-token 自注意力；直接展平 —— #
        fused_flat = fused.reshape(fused.size(0), -1)  # (B, Pe*(C 或 2C))

        return self.output_mlp(fused_flat)  # (B,7)


# ===================== MoCA 总分 =====================
def compute_moca_total(moca_subscores: torch.Tensor):
    return moca_subscores.sum(dim=1)
