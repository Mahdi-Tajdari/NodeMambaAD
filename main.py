# main.py - نسخه با batching + relabel edge_index + local neighbors for BitcoinOTC + fixed ano_label
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, f1_score
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.utils import subgraph  # for relabel

# DGL + PyG + Mamba
import dgl

# توابع خودمون
from utils import load_mat, preprocess_features, adj_to_dgl_graph, get_topk_neighbors_dgl
from utils import structural_encoding_from_adj, compute_rq_from_adj
from models import NodeGLADMambaRecon  # مدل جدید با reconstruction

# Seed
class Args:
    dataset = 'bitcoinotc'
    seed = 42
    batch_size = 4096  # adjust if needed

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
edge_index = torch.stack([torch.LongTensor(row), torch.LongTensor(col)]).to(device)  # directed for BitcoinOTC

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
model = NodeGLADMambaRecon(feat_dim=ft_size, hidden_dim=64, k=32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
scaler = GradScaler()

print("Training شروع شد — با batching + relabel + local neighbors + fixed ano_label")
print("-" * 70)

best_auc_val = 0.0
best_auc_test = 0.0
best_epoch = 0
patience = 150
counter = 0

node_indices = torch.arange(nb_nodes, device=device)

for epoch in range(1, 230):
    model.train()
    total_loss = 0.0
    score_train = torch.zeros(nb_nodes, device=device)
    
    perm = torch.randperm(nb_nodes, device=device)
    for i in range(0, nb_nodes, args.batch_size):
        optimizer.zero_grad()
        batch_idx = perm[i:i + args.batch_size].sort()[0]  # sort for subgraph
        
        with autocast():
            # Relabel edge_index for batch
            batch_edge_index, _ = subgraph(batch_idx, edge_index, relabel_nodes=True, num_nodes=nb_nodes)
            
            # Local map for neighbors
            global_to_local = torch.full((nb_nodes,), -1, dtype=torch.long, device=device)
            global_to_local[batch_idx] = torch.arange(len(batch_idx), device=device)
            
            batch_neighbors_global = neighbors[batch_idx]
            batch_neighbors = global_to_local[batch_neighbors_global]
            mask_outside = batch_neighbors == -1
            if mask_outside.any():
                # Pad with self (local index)
                self_indices = torch.arange(len(batch_idx), device=device).unsqueeze(1).repeat(1, model.k)
                batch_neighbors[mask_outside] = self_indices[mask_outside]
            
            batch_x = x_raw[batch_idx]
            batch_struct = x_struct[batch_idx]
            batch_rq = rq[batch_idx]
            
            loss, batch_score = model(batch_x, batch_struct, batch_edge_index, batch_neighbors, batch_rq)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * len(batch_idx)
        score_train[batch_idx] = batch_score.detach()
    
    avg_loss = total_loss / nb_nodes

    model.eval()
    score_eval = torch.zeros(nb_nodes, device=device)
    with torch.no_grad():
        with autocast():
            for i in range(0, nb_nodes, args.batch_size):
                batch_idx = node_indices[i:i + args.batch_size].sort()[0]
                
                batch_edge_index, _ = subgraph(batch_idx, edge_index, relabel_nodes=True, num_nodes=nb_nodes)
                
                global_to_local = torch.full((nb_nodes,), -1, dtype=torch.long, device=device)
                global_to_local[batch_idx] = torch.arange(len(batch_idx), device=device)
                
                batch_neighbors_global = neighbors[batch_idx]
                batch_neighbors = global_to_local[batch_neighbors_global]
                mask_outside = batch_neighbors == -1
                if mask_outside.any():
                    self_indices = torch.arange(len(batch_idx), device=device).unsqueeze(1).repeat(1, model.k)
                    batch_neighbors[mask_outside] = self_indices[mask_outside]
                
                batch_x = x_raw[batch_idx]
                batch_struct = x_struct[batch_idx]
                batch_rq = rq[batch_idx]
                
                _, batch_score = model(batch_x, batch_struct, batch_edge_index, batch_neighbors, batch_rq)
                score_eval[batch_idx] = batch_score
        
        auc_val = roc_auc_score(ano_label_tensor[idx_val].cpu().numpy(), score_eval[idx_val].cpu().numpy())
        auc_test = roc_auc_score(ano_label_tensor[idx_test].cpu().numpy(), score_eval[idx_test].cpu().numpy())
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | Val AUC: {auc_val:.4f} | Test AUC: {auc_test:.4f} | Best Test AUC: {best_auc_test:.4f}")

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

# Load best model and evaluate
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
score = torch.zeros(nb_nodes, device=device)
with torch.no_grad():
    with autocast():
        for i in range(0, nb_nodes, args.batch_size):
            batch_idx = node_indices[i:i + args.batch_size].sort()[0]
            
            batch_edge_index, _ = subgraph(batch_idx, edge_index, relabel_nodes=True, num_nodes=nb_nodes)
            
            global_to_local = torch.full((nb_nodes,), -1, dtype=torch.long, device=device)
            global_to_local[batch_idx] = torch.arange(len(batch_idx), device=device)
            
            batch_neighbors_global = neighbors[batch_idx]
            batch_neighbors = global_to_local[batch_neighbors_global]
            mask_outside = batch_neighbors == -1
            if mask_outside.any():
                self_indices = torch.arange(len(batch_idx), device=device).unsqueeze(1).repeat(1, model.k)
                batch_neighbors[mask_outside] = self_indices[mask_outside]
            
            batch_x = x_raw[batch_idx]
            batch_struct = x_struct[batch_idx]
            batch_rq = rq[batch_idx]
            
            _, batch_score = model(batch_x, batch_struct, batch_edge_index, batch_neighbors, batch_rq)
            score[batch_idx] = batch_score

# Overall AUC (on all data)
auc_overall = roc_auc_score(ano_label_tensor.cpu().numpy(), score.cpu().numpy())
print(f"\nتموم شد! بهترین Val AUC: {best_auc_val:.4f} | بهترین Test AUC: {best_auc_test:.4f} در epoch {best_epoch}")
print(f"Overall AUC: {auc_overall:.4f}")

# Find best threshold on validation set using F1 score
thresholds = np.linspace(0, score.max().item(), 100)
best_thresh = 0.0
best_f1_val = 0.0
for thresh in thresholds:
    pred_val = (score[idx_val].cpu().numpy() > thresh).astype(int)
    f1_val = f1_score(ano_label_tensor[idx_val].cpu().numpy(), pred_val)
    if f1_val > best_f1_val:
        best_f1_val = f1_val
        best_thresh = thresh

# Metrics on test set
pred_test = (score[idx_test].cpu().numpy() > best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(ano_label_tensor[idx_test].cpu().numpy(), pred_test).ravel()
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(ano_label_tensor[idx_test].cpu().numpy(), pred_test, average='binary', zero_division=0)

print(f"\nTest Metrics (Threshold: {best_thresh:.4f}):")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(f"Precision: {precision_test:.4f} | Recall: {recall_test:.4f} | F1: {f1_test:.4f}")

# Metrics on overall data (for completeness)
pred_overall = (score.cpu().numpy() > best_thresh).astype(int)
tn_o, fp_o, fn_o, tp_o = confusion_matrix(ano_label_tensor.cpu().numpy(), pred_overall).ravel()
precision_o, recall_o, f1_o, _ = precision_recall_fscore_support(ano_label_tensor.cpu().numpy(), pred_overall, average='binary', zero_division=0)

print(f"\nOverall Metrics (Threshold: {best_thresh:.4f}):")
print(f"TP: {tp_o} | TN: {tn_o} | FP: {fp_o} | FN: {fn_o}")
print(f"Precision: {precision_o:.4f} | Recall: {recall_o:.4f} | F1: {f1_o:.4f}")

print("اگر AUC بالاتر رفت، تغییر بعدی رو بگو!")
