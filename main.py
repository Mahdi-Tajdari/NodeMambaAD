# main.py - نسخه با scheduler on val for less overfit + GCN + wd=5e-3
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, f1_score

# DGL + PyG + Mamba
import dgl
from torch_geometric.utils import to_undirected

# توابع خودمون
from utils import load_mat, preprocess_features, adj_to_dgl_graph, get_topk_neighbors_dgl
from utils import structural_encoding_from_adj, compute_rq_from_adj
from models import NodeGLADMamba  # مدل با GCN + focal-like

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

# Structural Encoding (20 فیچر)
print("Computing structural encoding...")
x_struct = structural_encoding_from_adj(edge_index, nb_nodes).to(device)  # [N, 20]

# Top-k neighbors (k=32)
print("Computing top-k neighbors...")
neighbors = get_topk_neighbors_dgl(dgl_graph, k=32).to(device)  # [N, 32]

# Rayleigh Quotient
print("Computing Rayleigh Quotient...")
rq = compute_rq_from_adj(x_raw, edge_index).to(device)

# Labels
ano_label_tensor = torch.FloatTensor(ano_label).to(device)

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ------------------- 4. Model & Optimizer -------------------
model = NodeGLADMamba(feat_dim=ft_size, hidden_dim=128, k=32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-3)  # higher wd
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

print("Training شروع شد — با scheduler on val + GCN + wd=5e-3 for less overfit")
print("-" * 70)

best_auc_val = 0.0
best_auc_test = 0.0
best_epoch = 0
patience = 150
counter = 0

for epoch in range(1, 170):
    model.train()
    optimizer.zero_grad()

    loss, score = model(x_raw, x_struct, edge_index, neighbors, rq)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        _, score_eval = model(x_raw, x_struct, edge_index, neighbors, rq)
        auc_val = roc_auc_score(ano_label[idx_val], score_eval[idx_val].cpu().numpy())
        auc_test = roc_auc_score(ano_label[idx_test], score_eval[idx_test].cpu().numpy())
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | Val AUC: {auc_val:.4f} | Test AUC: {auc_test:.4f} | Best Test AUC: {best_auc_test:.4f} | Grad Norm: {grad_norm:.4f}")

        scheduler.step(auc_val)  # on val for less overfit to test

        if auc_val > best_auc_val:
            best_auc_val = auc_val
            best_auc_test = auc_test
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch} — بهترین Val AUC: {best_auc_val:.4f}")
                break
    model.train()

# Load best model and evaluate
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
with torch.no_grad():
    _, score = model(x_raw, x_struct, edge_index, neighbors, rq)

# Overall AUC (on all data)
auc_overall = roc_auc_score(ano_label, score.cpu().numpy())
print(f"\nتموم شد! بهترین Val AUC: {best_auc_val:.4f} | بهترین Test AUC: {best_auc_test:.4f} در epoch {best_epoch}")
print(f"Overall AUC: {auc_overall:.4f}")

# Find best threshold on validation set using F1 score
thresholds = np.linspace(0, score.max().item(), 100)
best_thresh = 0.0
best_f1_val = 0.0
for thresh in thresholds:
    pred_val = (score[idx_val].cpu().numpy() > thresh).astype(int)
    f1_val = f1_score(ano_label[idx_val], pred_val)
    if f1_val > best_f1_val:
        best_f1_val = f1_val
        best_thresh = thresh

# Metrics on test set
pred_test = (score[idx_test].cpu().numpy() > best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(ano_label[idx_test], pred_test).ravel()
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(ano_label[idx_test], pred_test, average='binary', zero_division=0)

print(f"\nTest Metrics (Threshold: {best_thresh:.4f}):")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(f"Precision: {precision_test:.4f} | Recall: {recall_test:.4f} | F1: {f1_test:.4f}")

# Metrics on overall data (for completeness)
pred_overall = (score.cpu().numpy() > best_thresh).astype(int)
tn_o, fp_o, fn_o, tp_o = confusion_matrix(ano_label, pred_overall).ravel()
precision_o, recall_o, f1_o, _ = precision_recall_fscore_support(ano_label, pred_overall, average='binary', zero_division=0)

print(f"\nOverall Metrics (Threshold: {best_thresh:.4f}):")
print(f"TP: {tp_o} | TN: {tn_o} | FP: {fp_o} | FN: {fn_o}")
print(f"Precision: {precision_o:.4f} | Recall: {recall_o:.4f} | F1: {f1_o:.4f}")

print("اگر AUC بالاتر رفت، تغییر بعدی رو بگو!")