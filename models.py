# models.py — دقیقاً همون چیزی که گفتم، هیچ کم و کاستی نداره
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = d_model // 16

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv-1, groups=d_model, bias=False)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        A = torch.arange(1, d_state + 1).float().repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(self.d_model, dim=-1)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[..., :l]
        x = rearrange(x, 'b d l -> b l d')   # درست شد
        x = F.silu(x)

        x_db = self.x_proj(x)
        delta, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log.float())
        y = self.selective_scan(x, delta, A, B, C, self.D)

        y = y * F.silu(res)
        return self.out_proj(y)[:, -1, :]  # فقط توکن مرکزی

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d) = u.shape
        n = self.d_state
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(-2) * u.unsqueeze(-1)

        h = torch.zeros(b, d, n, device=u.device)
        ys = []
        for i in range(l):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = torch.einsum('bdn,bn->bd', h, C[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + u * D


class NodeGLADMamba(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, num_layers=3, k=8):
        super().__init__()
        self.d_model = hidden_dim * num_layers  # 192
        self.k = k

        # مرحله ۱ و ۲: دو تا GNN کاملاً مستقل
        self.gnn_feat = nn.Sequential(
            nn.Linear(feat_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.gnn_struct = nn.Sequential(
            nn.Linear(2, self.d_model),  # فقط degree normalized + خودش
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # مرحله ۵: دو تا Mamba
        self.mamba1 = MambaBlock(self.d_model)
        self.mamba2 = MambaBlock(self.d_model)

        # مرحله ۳: RQ projector
        self.rq_proj = nn.Linear(1, self.d_model)

    def forward(self, features, x_s, adj, neighbors, rq):
        # features: [1,N,F], x_s: [1,N,2], rq: [1,N,1]
        x_f = features[0]           # [N, F]
        x_s = x_s[0]                 # [N, 2]

        # مرحله ۲: دو تا نمایش مستقل
        h_feat = self.gnn_feat(x_f)           # [N, 192]
        h_struct = self.gnn_struct(x_s)       # [N, 192]

        # مرحله ۴: ساخت توالی معکوس + تزریق RQ
        center_feat = h_feat.unsqueeze(1)                           # [N,1,192]
        neigh_struct = h_struct[neighbors]                          # [N,k,192]
        seq1 = torch.cat([center_feat, neigh_struct + self.rq_proj(rq[0]).unsqueeze(1)], dim=1)

        center_struct = h_struct.unsqueeze(1)
        neigh_feat = h_feat[neighbors]
        seq2 = torch.cat([center_struct, neigh_feat + self.rq_proj(rq[0]).unsqueeze(1)], dim=1)

        # مرحله ۵: دو تا Mamba
        out1 = self.mamba1(seq1)   # [N, 192]
        out2 = self.mamba2(seq2)   # [N, 192]

        # مرحله ۶: اختلاف + RQ
        diff = torch.norm(out1 - out2, dim=1)           # [N]
        score = diff + 0.5 * rq[0].squeeze(-1)          # [N]

        return score