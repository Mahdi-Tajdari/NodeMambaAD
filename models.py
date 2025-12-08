# models.py
# بدون تغییر — forward از args.k استفاده نمی‌کنه، مستقیم از neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv

class NodeGLADMamba(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, k=32):
        super().__init__()
        self.k = k  # فقط برای info، forward از neighbors.shape استفاده می‌کنه
        self.hidden_dim = hidden_dim
        
        # دو تا GNN واقعی
        self.gnn_feat = nn.ModuleList([GCNConv(feat_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim)])
        self.gnn_struct = nn.ModuleList([GCNConv(12, hidden_dim), GCNConv(hidden_dim, hidden_dim)])
        
        # دو تا Mamba رسمی با selective (paper-inspired)
        self.mamba1 = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)

        self.norm = nn.LayerNorm(hidden_dim)
        self.rq_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, x_struct, edge_index, neighbors, rq):
        # x: [N,F], x_struct: [N,12], edge_index: [2,E], neighbors: [N,k], rq: [N,1]
        
        h = x
        for conv in self.gnn_feat:
            h = self.norm(F.elu(conv(h, edge_index)))
        
        h_feat = h
        
        h = x_struct
        for conv in self.gnn_struct:
            h = self.norm(F.elu(conv(h, edge_index)))
        
        h_struct = h
        
        # ساخت توالی — طول = 1 + neighbors.shape[1]
        seq1 = torch.cat([h_feat.unsqueeze(1), h_struct[neighbors]], dim=1)   # [N, 1+k, hidden_dim]
        seq2 = torch.cat([h_struct.unsqueeze(1), h_feat[neighbors]], dim=1)
        
        # Mamba — pooling برای کل توالی
        m1_out = self.mamba1(seq1)  # [N, 1+k, hidden]
        out1 = m1_out.mean(dim=1)  # mean pooling
        
        m2_out = self.mamba2(seq2.flip(1))  # [N, 1+k, hidden]
        out2 = m2_out.mean(dim=1)  # mean pooling

        return out1, out2  # برای contrastive loss