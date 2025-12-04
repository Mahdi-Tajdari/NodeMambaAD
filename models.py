# models.py - نسخه با GCN + focal original_loss for focus on hard anomalies + reconstruction
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv

class NodeGLADMamba(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, k=32):
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
        
        # Mamba با d_state=32
        self.mamba1 = Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=4)
        self.mamba2 = Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=4)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.rq_weight = nn.Parameter(torch.tensor(0.5))
        self.contrast_lambda = 0.5

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
        
        # Anonymize target
        seq1 = torch.cat([torch.zeros_like(h_feat.unsqueeze(1)), h_struct[neighbors]], dim=1)
        seq2 = torch.cat([torch.zeros_like(h_struct.unsqueeze(1)), h_feat[neighbors]], dim=1)
        
        out1 = self.mamba1(seq1)[:, 0, :]
        out2 = self.mamba2(seq2.flip(1))[:, 0, :]
        
        # Negative samples (lighter: only one)
        neg_indices = torch.randperm(neighbors.size(0), device=neighbors.device)
        neg_neighbors = neighbors[neg_indices]
        seq1_neg = torch.cat([torch.zeros_like(h_feat.unsqueeze(1)), h_struct[neg_neighbors]], dim=1)
        out1_neg = self.mamba1(seq1_neg)[:, 0, :]
        
        # Positive cos
        cos_pos = F.cosine_similarity(out1, out2, dim=1)
        # Negative
        cos_neg = F.cosine_similarity(out1, out1_neg, dim=1)
        
        # Contrastive loss
        logits_pos = cos_pos.unsqueeze(1)
        logits_neg = cos_neg.unsqueeze(1)
        labels = torch.zeros(out1.size(0), dtype=torch.long, device=out1.device)
        contrast_loss = F.cross_entropy(torch.cat([logits_pos, logits_neg], dim=1), labels)
        
        # Score
        diff = F.mse_loss(out1, out2, reduction='none').mean(dim=1)
        rq_score = rq.squeeze()
        rq_score = torch.sigmoid(rq_score / (rq_score.mean() + 1e-8))
        rq_score = torch.clamp(rq_score, min=0.0, max=2.0)
        score = torch.clamp(diff, min=0.0, max=5.0) + torch.sigmoid(self.rq_weight) * rq_score
        
        # Focal-like original_loss for focus on high score (anomalies)
        gamma = 2.0
        original_loss = - ( (1 - torch.sigmoid(score)) ** gamma * F.logsigmoid(score) ).mean()
        
        loss = self.contrast_lambda * contrast_loss + (1 - self.contrast_lambda) * original_loss
        
        if self.training:
            return loss, score
        else:
            return None, score

class NodeGLADMambaRecon(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, k=32):
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
        
        # Mamba با d_state=32
        self.mamba1 = Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=4)
        self.mamba2 = Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=4)
        
        # Decoders for reconstruction
        self.mamba_decode_feat = Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=4)
        self.mamba_decode_struct = Mamba(d_model=hidden_dim, d_state=32, d_conv=4, expand=4)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.rq_weight = nn.Parameter(torch.tensor(0.5))
        self.contrast_lambda = 0.5
        self.lambda_recon = 0.3  # tunable

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
        
        # Sequences for feat and struct
        seq_feat = torch.cat([h_feat.unsqueeze(1), h_feat[neighbors]], dim=1)  # [N, 1+k, hidden]
        seq_struct = torch.cat([h_struct.unsqueeze(1), h_struct[neighbors]], dim=1)  # [N, 1+k, hidden]
        
        # Anonymize target (like before, but using seq_struct for seq1, etc.)
        seq1 = torch.cat([torch.zeros_like(h_feat.unsqueeze(1)), h_struct[neighbors]], dim=1)
        seq2 = torch.cat([torch.zeros_like(h_struct.unsqueeze(1)), h_feat[neighbors]], dim=1)
        
        out1 = self.mamba1(seq1)[:, 0, :]
        out2 = self.mamba2(seq2.flip(1))[:, 0, :]
        
        # Negative samples (lighter: only one)
        neg_indices = torch.randperm(neighbors.size(0), device=neighbors.device)
        neg_neighbors = neighbors[neg_indices]
        seq1_neg = torch.cat([torch.zeros_like(h_feat.unsqueeze(1)), h_struct[neg_neighbors]], dim=1)
        out1_neg = self.mamba1(seq1_neg)[:, 0, :]
        
        # Positive cos
        cos_pos = F.cosine_similarity(out1, out2, dim=1)
        # Negative
        cos_neg = F.cosine_similarity(out1, out1_neg, dim=1)
        
        # Contrastive loss
        logits_pos = cos_pos.unsqueeze(1)
        logits_neg = cos_neg.unsqueeze(1)
        labels = torch.zeros(out1.size(0), dtype=torch.long, device=out1.device)
        contrast_loss = F.cross_entropy(torch.cat([logits_pos, logits_neg], dim=1), labels)
        
        # Reconstruction
        # Approximate recon from out2 for seq_feat, and out1 for seq_struct
        recon_seq_feat = self.mamba_decode_feat(out2.unsqueeze(1).repeat(1, self.k + 1, 1))  # [N, 1+k, hidden]
        recon_seq_struct = self.mamba_decode_struct(out1.unsqueeze(1).repeat(1, self.k + 1, 1))  # [N, 1+k, hidden]
        
        recon_loss = F.mse_loss(recon_seq_feat, seq_feat) + F.mse_loss(recon_seq_struct, seq_struct)
        
        # Score
        diff = F.mse_loss(out1, out2, reduction='none').mean(dim=1)
        rq_score = rq.squeeze()
        rq_score = torch.sigmoid(rq_score / (rq_score.mean() + 1e-8))
        rq_score = torch.clamp(rq_score, min=0.0, max=2.0)
        recon_error = (recon_seq_feat - seq_feat).pow(2).mean(dim=(1,2)) + (recon_seq_struct - seq_struct).pow(2).mean(dim=(1,2))
        score = torch.clamp(diff + recon_error, min=0.0, max=5.0) + torch.sigmoid(self.rq_weight) * rq_score
        
        # Focal-like original_loss for focus on high score (anomalies)
        gamma = 2.0
        original_loss = - ( (1 - torch.sigmoid(score)) ** gamma * F.logsigmoid(score) ).mean()
        
        loss = self.lambda_recon * recon_loss + self.contrast_lambda * contrast_loss + (1 - self.contrast_lambda - self.lambda_recon) * original_loss
        
        if self.training:
            return loss, score
        else:
            return None, score
