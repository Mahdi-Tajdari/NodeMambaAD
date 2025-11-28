# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv

class NodeGLADMamba(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, k=16):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        
        # سه لایه GCN
        self.gnn_feat = nn.ModuleList([
            GCNConv(feat_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        self.gnn_struct = nn.ModuleList([
            GCNConv(20, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        
        # Mamba با d_state=16
        self.mamba1 = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=4)
        self.mamba2 = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=4)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.4)  # بالاتر برای ضد-overfit
        self.rq_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, x_struct, edge_index, neighbors, rq):
        h = x
        for conv in self.gnn_feat:
            h = F.silu(conv(h, edge_index))
            h = self.dropout(h)
        h_feat = self.norm(h)
        
        h = x_struct
        for conv in self.gnn_struct:
            h = F.silu(conv(h, edge_index))
            h = self.dropout(h)
        h_struct = self.norm(h)
        
        seq1 = torch.cat([h_feat.unsqueeze(1), h_struct[neighbors]], dim=1)
        seq2 = torch.cat([h_struct.unsqueeze(1), h_feat[neighbors]], dim=1)
        
        out1 = self.mamba1(seq1)[:, 0, :]
        out2 = self.mamba2(seq2.flip(1))[:, 0, :]

        diff = F.mse_loss(out1, out2, reduction='none').mean(dim=1)
        diff = torch.clamp(diff, min=0.0, max=5.0)  # سخت‌تر برای stability

        rq_score = rq.squeeze()
        rq_score = torch.sigmoid(rq_score / (rq_score.mean() + 1e-8))
        rq_score = torch.clamp(rq_score, min=0.0, max=2.0)  # کمتر max

        score = diff + torch.sigmoid(self.rq_weight) * rq_score
        return score