import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import Model
from utils import *

import random
import os
import argparse
from torch_geometric.utils import to_dense_adj # برای استفاده در صورت نیاز


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='Dataset to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use. Set to -1 for CPU.') # پیش فرض را 0 می گذاریم
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of encoder layers.')
parser.add_argument('--num_heads', type=int, default=1, help='Number of attention heads.')
args = parser.parse_args()


os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# تنظیم دستگاه برای Colab GPU
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu))
    print(f"Using GPU: {device}")
else:
    device = torch.device('cpu')
    print("Using CPU")
    
    
# --- فاز 1: بارگذاری و آماده سازی داده ها ---
print(f"Loading dataset: {args.dataset}...")
adj, features, ano_label = load_mat(args.dataset)


features_dense, _ = preprocess_features(features)

dgl_graph = adj_to_dgl_graph(adj)
dgl_graph = dgl_graph.to(device)

nb_nodes = features_dense.shape[0]
ft_size = features_dense.shape[1]

adj_tensor_no_loop = dgl_graph.adjacency_matrix().to_dense().clone().detach().to(device)

adj_tensor_with_loop = adj_tensor_no_loop + torch.eye(adj_tensor_no_loop.size(0)).to(device)
adj_tensor_with_loop = adj_tensor_with_loop.to(device) 

adj_normalized = normalize_adj(adj)
adj_normalized_with_loop = (adj_normalized + sp.eye(adj_normalized.shape[0])).todense()

# انتقال تنسورها به GPU
features_torch = torch.FloatTensor(features_dense[np.newaxis]).to(device)
adj_torch = torch.FloatTensor(adj_normalized_with_loop[np.newaxis]).to(device)
ano_labels_torch = torch.FloatTensor(ano_label).to(device)


# --- فاز 2: تعریف و تست مدل (Feedforward Test) ---

print("\n--- Model Setup and Feedforward Test ---")

model = Model(
    ft_size=ft_size,
    hidden_dim=args.hidden_dim,
    num_layers=args.num_layers,
    num_heads=args.num_heads
).to(device) # اطمینان از انتقال مدل به GPU

print(f"Model initialized: GATMambaAutoencoder (Heads={args.num_heads}, Hidden={args.hidden_dim})")
print(f"Input Feature Shape (Batch): {features_torch.shape}")
print(f"Device Check (Features): {features_torch.device}")


# اجرای Feedforward
model.eval()
with torch.no_grad():
    try:
        reconstruction, embedding = model(features_torch, adj_torch)

        print("\n--- Feedforward Results ---")
        print(f"Reconstruction Shape: {reconstruction.shape}")
        print(f"Embedding Shape: {embedding.shape}")
        
        # بررسی صحت ابعاد
        expected_embedding_dim = args.hidden_dim * args.num_heads
        if reconstruction.shape == features_torch.squeeze(0).shape and \
           embedding.shape[0] == nb_nodes and \
           embedding.shape[1] == expected_embedding_dim:
            print("✅ Tensor shapes match the expected Autoencoder output.")
            print("✅ GPU/Device check passed.")
        else:
            print("❌ WARNING: Output tensor shapes do not match expectations.")

    except Exception as e:
        print(f"\n❌ FATAL ERROR during Feedforward: {e}")
        print("Check if all dependencies are correctly installed for your CUDA version.")


# --- فاز 3: تعریف تابع زیان، بهینه سازی و آموزش ---
# (این بخش باید در مرحله بعدی اضافه شود)
