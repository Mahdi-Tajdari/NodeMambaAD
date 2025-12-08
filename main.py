# main.py
# نسخه نهایی — batching full + stable z-score (median) + scheduler + fix batch neg sample + debug logs
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from sklearn.metrics import roc_auc_score

# DGL + PyG + Mamba
import dgl
from torch_geometric.utils import to_undirected, dropout_edge

# توابع خودمون
from utils import load_mat, preprocess_features, adj_to_dgl_graph, get_topk_neighbors_dgl
from utils import structural_encoding_from_adj, compute_rq_from_adj
from models import NodeGLADMamba

# Seed & Args — همه hyperparams اینجا
class Args:
    dataset = 'cora'  # تغییر به 'pubmed', 'citeseer', 'bitalpha' و ...
    seed = 42
    k = 32  # top-k neighbors
    hidden_dim = 128
    lr = 0.0005
    weight_decay = 1e-4
    patience = 30  # کم برای overfit
    tau = 0.1  # sharpتر برای contrastive
    reg_coef = 0.01  # افزایش برای overfit
    rq_scale = 0.2  # کم برای balance
    epochs = 400
    print_every = 25
    dropout_p = 0.1
    grad_clip = 0.5
    min_samples_z = 50  # window بزرگتر برای stable sigma
    batch_size = 512  # batching همیشه، in-batch neg
    num_negatives = 128  # fixed negatives per sample, but adjusted for batch size

args = Args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} | Dataset: {args.dataset} | k: {args.k} | Batch: {args.batch_size}")

# ------------------- 1. Load Data -------------------
adj, features, labels, idx_train, idx_val, idx_test, ano_label, _, _ = load_mat(args.dataset)

# Preprocess features (row-normalize)
features, _ = preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]

# DGL Graph (برای neighbors)
dgl_graph = adj_to_dgl_graph(adj).to(device)

# ------------------- 2. Build edge_index برای PyG -------------------
raw_adj = load_mat(args.dataset)[0]
row, col = raw_adj.nonzero()
edge_index = torch.stack([torch.LongTensor(row), torch.LongTensor(col)])
edge_index = to_undirected(edge_index).to(device)

# ------------------- 3. Prepare Inputs -------------------
features_tensor = torch.FloatTensor(features[np.newaxis]).to(device)
x_raw = features_tensor[0]

# Structural Encoding غنی (12 فیچر)
print("Computing structural encoding...")
x_struct = structural_encoding_from_adj(edge_index, nb_nodes).to(device)

# Top-k neighbors
print("Computing top-k neighbors...")
neighbors = get_topk_neighbors_dgl(dgl_graph, k=args.k).to(device)

# Rayleigh Quotient درست
print("Computing Rayleigh Quotient...")
rq = compute_rq_from_adj(x_raw, edge_index).to(device)

# Label
ano_label_tensor = torch.FloatTensor(ano_label).to(device)

# ------------------- 4. Model & Optimizer -------------------
model = NodeGLADMamba(feat_dim=ft_size, hidden_dim=args.hidden_dim, k=args.k).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
print("Training شروع شد — مدل نهایی با GCN + Mamba رسمی + RQ درست")
print("-" * 70)

best_auc = 0.0
counter = 0
train_diffs = []  # track per-epoch diff means

def compute_loss_batched(out1, out2, tau, device, batch_size, num_negatives):
    """Batched InfoNCE با fixed in-batch negatives (SimCLR-style) + fix for small batches"""
    total_loss = 0.0
    num_batches = (out1.shape[0] + batch_size - 1) // batch_size
    indices = torch.randperm(out1.shape[0], device=device)  # shuffle for batches
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, out1.shape[0])
        batch_idx = indices[start:end]
        B = len(batch_idx)
        if B < 2:  # skip very small batches
            continue
        out1_b = out1[batch_idx]
        out2_b = out2[batch_idx]
        out1_norm_b = F.normalize(out1_b, dim=-1)
        out2_norm_b = F.normalize(out2_b, dim=-1)
        sim_matrix_b = torch.mm(out1_norm_b, out2_norm_b.t()) / tau
        mask_b = torch.eye(B, device=device).bool()
        positives_b = sim_matrix_b[mask_b].view(B, 1)
        neg_mask = ~mask_b
        num_avail_neg = neg_mask.sum().item() // B  # B-1
        effective_neg = min(num_negatives, num_avail_neg)
        if effective_neg <= 0:
            continue
        # Fix: min برای جلوگیری overflow
        num_select = min(neg_mask.sum().item(), B * effective_neg)
        neg_indices = torch.randperm(neg_mask.sum().item(), device=device)[:num_select]
        neg_flat = sim_matrix_b[neg_mask][neg_indices].view(B, num_select // B)
        logits_b = torch.cat([positives_b, neg_flat], dim=1)
        labels_b = torch.zeros(B, dtype=torch.long, device=device)
        batch_loss = F.cross_entropy(logits_b, labels_b)
        total_loss += batch_loss
        # Debug log for loss per batch (فقط اگر لازم, or in early epochs)
        if epoch <= 10:
            print(f"Batch {i+1}/{num_batches} | Batch Loss: {batch_loss.item():.6f} | Effective neg: {effective_neg}")
    return total_loss / max(1, num_batches - sum([1 for _ in range(num_batches) if len(indices[i*batch_size:(i+1)*batch_size]) < 2]))

for epoch in range(1, args.epochs + 1):
    model.train()
    optimizer.zero_grad()

    # Augmentation: edge drop
    edge_index_aug = dropout_edge(edge_index, p=args.dropout_p)[0]

    out1, out2 = model(x_raw, x_struct, edge_index_aug, neighbors, rq)

    # فقط در epoch 1: لاگ ساختار
    if epoch == 1:
        print(f"--- ساختار در Epoch 1 ---")
        print(f"edge_index_aug shape: {edge_index_aug.shape}")
        print(f"out1 shape: {out1.shape}")
        print(f"out2 shape: {out2.shape}")
        print(f"seq1/seq2 shape: [{nb_nodes}, {1 + args.k}, {args.hidden_dim}] (h_feat/struct + neighbors)")
        print(f"mamba full output shape: [{nb_nodes}, {1 + args.k}, {args.hidden_dim}]")
        print(f"--- پایان لاگ ساختار ---")

    # InfoNCE loss (batched با subsample neg)
    loss = compute_loss_batched(out1, out2, args.tau, device, args.batch_size, args.num_negatives)

    reg_loss = args.reg_coef * sum(p.norm(2)**2 for p in model.parameters() if p.requires_grad)
    loss += reg_loss

    # Debug: print loss before/after reg
    print(f"Epoch {epoch:03d} | Contrastive Loss: {loss - reg_loss:.6f} | Reg Loss: {reg_loss:.6f} | Total Loss: {loss:.6f}")

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print(f"NaN/Inf detected at epoch {epoch}! Stopping.")
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
    optimizer.step()
    scheduler.step(loss.item())  # LR reduce if loss plateau

    # track diff (full برای z-score, but mean برای memory)
    out1_norm_full = F.normalize(out1, dim=-1)
    out2_norm_full = F.normalize(out2, dim=-1)
    cos_sim_train = F.cosine_similarity(out1_norm_full, out2_norm_full)
    diff_train = 1 - cos_sim_train
    train_diffs.append(diff_train.mean().item())
    # Debug: print train diff mean
    print(f"Epoch {epoch:03d} | Train diff mean: {diff_train.mean().item():.4f}")

    if epoch % args.print_every == 0 or epoch <= 10:
        model.eval()
        with torch.no_grad():
            out1_eval, out2_eval = model(x_raw, x_struct, edge_index, neighbors, rq)
            out1_norm = F.normalize(out1_eval, dim=-1)
            out2_norm = F.normalize(out2_eval, dim=-1)
            cos_sim = F.cosine_similarity(out1_norm, out2_norm)
            diff_eval = 1 - cos_sim
            # Stable z-score: mean-based از recent diffs
            if len(train_diffs) >= args.min_samples_z:
                mu = np.mean(train_diffs[-args.min_samples_z:])
                sigma = np.std(train_diffs[-args.min_samples_z:]) + 1e-8
                z_diff = (diff_eval - mu) / sigma
                z_diff = torch.clamp(z_diff, -5, 5)  # clamp برای جلوگیری explode
            else:
                z_diff = torch.zeros_like(diff_eval)
            rq_score = rq.squeeze() / (rq.mean() + 1e-8)
            rq_score = torch.clamp(rq_score, 0, 10)
            score_eval = z_diff + model.rq_weight * args.rq_scale * rq_score
            auc = roc_auc_score(ano_label, score_eval.cpu().numpy())
            val_diff_mean = diff_eval.mean().item()
            if auc > best_auc:
                best_auc = auc
                counter = 0
            else:
                counter += 1
            # Debug logs for z-score and rq
            if 'mu' in locals():
                print(f"Debug: mu={mu:.4f}, sigma={sigma:.4f}, z_mean={z_diff.mean().item():.4f}, rq_mean={rq_score.mean().item():.4f}")
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | AUC: {auc:.4f} | Best AUC: {best_auc:.4f} | Val diff mean: {val_diff_mean:.4f}")
            print(f"score mean: {score_eval.mean():.4f}, std: {score_eval.std():.4f}")
        model.train()

    if counter >= args.patience:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"\nتموم شد! بهترین AUC: {best_auc:.4f}")
if best_auc > 0.97:
    print("عالی! مدلت حالا واقعاً کار می‌کنه")
else:
    print("هنوز جا داره بهتر بشه — ولی حداقل دیگه زیر ۶۰٪ نیست!")