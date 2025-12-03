# models.py - گام 1.5: optimize base Mamba (بدون GCN/RQ)
from utils import perturb_walks
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class NodeMambaAD(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_walks=8, walk_length=6):  # بزرگ‌تر
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = num_walks * (walk_length + 1)  # 8*7=56
        
        # Embedding ساده (linear)
        self.embed = nn.Linear(feat_dim, hidden_dim)
        
        # Mamba bidirectional (بهتر tune شده)
        self.mamba_fwd = Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=4)
        self.mamba_bwd = Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=4)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)  # کمتر
        self.contrast_lambda = 0.5

    def forward(self, x, edge_index, walks, rq):  # rq نادیده
        h = F.silu(self.embed(x))
        h = self.norm(h)
        
        flat_walks = walks.view(walks.size(0), -1)
        seq = h[flat_walks]
        
        seq_fwd = torch.cat([torch.zeros(seq.size(0), 1, self.hidden_dim, device=seq.device), seq], dim=1)
        
        out_fwd = self.mamba_fwd(seq_fwd)[:, 0, :]
        
        seq_bwd = seq_fwd.flip(1)
        out_bwd = self.mamba_bwd(seq_bwd)[:, 0, :]
        
        neg_walks = perturb_walks(walks)
        flat_neg = neg_walks.view(neg_walks.size(0), -1)
        seq_neg = h[flat_neg]
        seq_neg_fwd = torch.cat([torch.zeros(seq.size(0), 1, self.hidden_dim, device=seq.device), seq_neg], dim=1)
        out_neg = self.mamba_fwd(seq_neg_fwd)[:, 0, :]
        
        cos_pos = F.cosine_similarity(out_fwd, out_bwd, dim=1)
        cos_neg = F.cosine_similarity(out_fwd, out_neg, dim=1)
        logits = torch.cat([cos_pos.unsqueeze(1), cos_neg.unsqueeze(1)], dim=1)
        labels = torch.zeros(out_fwd.size(0), dtype=torch.long, device=out_fwd.device)
        contrast_loss = F.cross_entropy(logits, labels)
        
        recon_err = F.mse_loss(out_fwd, out_bwd, reduction='none').mean(dim=1)
        score = recon_err.clamp(0, 5)
        
        gamma = 3.0  # بالاتر برای focus بیشتر
        focal_loss = - ((1 - torch.sigmoid(score)) ** gamma * F.logsigmoid(score)).mean()
        
        loss = self.contrast_lambda * contrast_loss + (1 - self.contrast_lambda) * focal_loss
        
        if self.training:
            return loss, score
        else:
            return None, score