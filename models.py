# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv

class NodeGLADMamba(nn.Module):
    def __init__(self, feat_dim, hidden_dim=96, k=8):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        
        # دو تا GNN واقعی
        self.gnn_feat = nn.ModuleList([GCNConv(feat_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim)])
        self.gnn_struct = nn.ModuleList([GCNConv(12, hidden_dim), GCNConv(hidden_dim, hidden_dim)])
        
        # دو تا Mamba رسمی
        self.mamba1 = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        # این دو خط رو اضافه کن:
        # self.mamba1.use_mem_eff_path = False
        # self.mamba2.use_mem_eff_path = False

        self.norm = nn.LayerNorm(hidden_dim)
        self.rq_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, x_struct, edge_index, neighbors, rq):
        # x: [N,F], x_struct: [N,16], edge_index: [2,E], neighbors: [N,8], rq: [N,1]
        
        h = x
        for conv in self.gnn_feat:
            h = F.elu(conv(h, edge_index))
        h_feat = self.norm(h)
        
        h = x_struct
        for conv in self.gnn_struct:
            h = F.elu(conv(h, edge_index))
        h_struct = self.norm(h)
        
        # ساخت توالی: مرکز + ۸ همسایه
        seq1 = torch.cat([h_feat.unsqueeze(1), h_struct[neighbors]], dim=1)   # [N,9,96]
        seq2 = torch.cat([h_struct.unsqueeze(1), h_feat[neighbors]], dim=1)
        
        # Mamba bidirectional
        out1 = self.mamba1(seq1)[:, 0, :]      # فقط این
        out2 = self.mamba2(seq2.flip(1))[:, 0, :]

        diff = F.mse_loss(out1, out2, reduction='none').mean(dim=1)

        # RQ خیلی ساده و پایدار
        rq_score = rq.squeeze()
        rq_score = rq_score / (rq_score.mean() + 1e-8)  # فقط تقسیم بر میانگین

        score = diff + 2.0 * rq_score  # ضریب 2.0 بهترین نتیجه رو داد # ضریب 3.0 طلاییه# ضریب ثابت بهتر از learnable در اوایل آموزش
        return score