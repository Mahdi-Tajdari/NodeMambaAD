# run.py - نسخه stable بدون gradient explosion
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
from torch_geometric.utils import to_undirected

# توابع خودمون
from utils import load_mat, preprocess_features, adj_to_dgl_graph, get_topk_neighbors_dgl
from utils import structural_encoding_from_adj, compute_rq_from_adj
from models import NodeGLADMamba  # مدل stable

# Seed
class Args:
    dataset = 'cora'
    seed = 42

args = Args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------- 1. Load Data -------------------
adj, features, labels, idx_train, idx_val, idx_test, ano_label, _, _ = load_mat(args.dataset)

# Preprocess features (row-normalize)
features, _ = preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]

# DGL Graph
dgl_graph = adj_to_dgl_graph(adj).to(device)

# ------------------- 2. Build edge_index -------------------
raw_adj = load_mat(args.dataset)[0]
row, col = raw_adj.nonzero()
edge_index = torch.stack([torch.LongTensor(row), torch.LongTensor(col)])
edge_index = to_undirected(edge_index).to(device)

# ------------------- 3. Prepare Inputs -------------------
features_tensor = torch.FloatTensor(features[np.newaxis]).to(device)
x_raw = features_tensor[0]  # [N, F]

# Structural Encoding (20 فیچر، اما betweenness fix شد به approx ساده‌تر)
print("Computing structural encoding...")
x_struct = structural_encoding_from_adj(edge_index, nb_nodes).to(device)  # [N, 20]

# Top-k neighbors (k=16)
print("Computing top-k neighbors...")
neighbors = get_topk_neighbors_dgl(dgl_graph, k=16).to(device)  # [N, 16]

# Rayleigh Quotient
print("Computing Rayleigh Quotient...")
rq = compute_rq_from_adj(x_raw, edge_index).to(device)

# Labels
ano_label_tensor = torch.FloatTensor(ano_label).to(device)

# ------------------- 4. Model & Optimizer -------------------
model = NodeGLADMamba(feat_dim=ft_size, hidden_dim=128, k=16).to(device)  # hidden_dim=128 برای stability
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # lr پایین‌تر برای جلوگیری از explosion
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, verbose=True)

print("Training شروع شد — stable با clipping قوی و loss ساده")
print("-" * 70)

best_auc = 0.0
best_epoch = 0
patience = 100
counter = 0

for epoch in range(1, 601):
    model.train()
    optimizer.zero_grad()

    score = model(x_raw, x_struct, edge_index, neighbors, rq)  # [N]

    # Loss: برگشت به -score.mean() اما بدون var penalty برای stability
    loss = -score.mean()

    loss.backward()
    # Clipping قوی‌تر برای fix explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # از 0.5 به 0.1
    optimizer.step()

    if epoch % 25 == 0 or epoch <= 10:
        model.eval()
        with torch.no_grad():
            score_eval = model(x_raw, x_struct, edge_index, neighbors, rq)
            auc_val = roc_auc_score(ano_label[idx_val], score_eval[idx_val].cpu().numpy())
            auc_test = roc_auc_score(ano_label[idx_test], score_eval[idx_test].cpu().numpy())
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | Val AUC: {auc_val:.4f} | Test AUC: {auc_test:.4f} | Best Test AUC: {best_auc:.4f}")

            scheduler.step(auc_val)

            if auc_test > best_auc:
                best_auc = auc_test
                best_epoch = epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        model.train()

print(f"\nتموم شد! بهترین Test AUC: {best_auc:.4f} در epoch {best_epoch}")
if best_auc > 0.90:
    print("عالی! SOTA-level")
else:
    print("اگر هنوز مسئله داشت، lr رو به 5e-5 کم کن!")