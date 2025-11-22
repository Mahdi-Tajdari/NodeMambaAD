# run.py
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
# from model import Model # Your DGL-compatible model goes here
from utils import load_mat, preprocess_features, adj_to_dgl_graph, normalize_adj
import random
import os
import dgl

# --- (Argument parsing and seed setting logic remains unchanged) ---

# --- (Using placeholder logic for args/seeds for demonstration) ---
class Args:
    dataset = 'cora'
    seed = 1
args = Args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
# ------------------------------------------------------------------

# 1. Load data (Exact step 1)
adj, features, labels, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

# 2. Preprocess features (Exact step 2)
# Applies Row-Normalization and returns features (dense) and features_tuple (discarded here)
features, _ = preprocess_features(features) 

# 3. Create DGL Graph (Exact step 3)
dgl_graph = adj_to_dgl_graph(adj) 

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

# 4. Normalize Adjacency and add Self-Loops (Exact step 4)
adj = normalize_adj(adj)
# A_hat = D^-1/2 * (A + I) * D^-1/2 (The dot product performs A * D^-1/2 * D^-1/2, then the second line adds I)
adj = (adj + sp.eye(adj.shape[0])).todense()

# 5. Final conversion to PyTorch Tensors (with added Batch Dimension)
features = torch.FloatTensor(features[np.newaxis]) # Shape [1, N, F]
adj = torch.FloatTensor(adj[np.newaxis])         # Shape [1, N, N]
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# ------------------------------------------------------------------
# Data is now 100% prepared according to the reference paper.
# The tensors 'features', 'adj', and the object 'dgl_graph' are ready for the model.
# ------------------------------------------------------------------

print("Data Loaded and Preprocessed exactly like the reference project.")
print(f"Final Feature Tensor Shape (features): {features.shape}")
print(f"Final Adjacency Tensor Shape (adj): {adj.shape}")
print(f"Number of nodes (nb_nodes): {nb_nodes}")
# print("You can now initialize your DGL-compatible Model and run the training loop.")