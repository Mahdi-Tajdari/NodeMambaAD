import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import GATMamba
from utils import *

import random
import os
import argparse
from torch_geometric.utils import to_dense_adj, from_scipy_sparse_matrix # [NEW]: کتابخانه لازم برای تبدیل فرمت
from torch_geometric.data import Data # [NEW]: کتابخانه لازم برای ساخت شیء Data (اختیاری)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='Dataset to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use. Set to -1 for CPU.')
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

# تنظیم دستگاه برای GPU
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu))
    print(f"Using GPU: {device}")
else:
    device = torch.device('cpu')
    print("Using CPU")
    
    
# --- فاز 1: بارگذاری و آماده سازی داده ها (استاندارد PyG) ---
print(f"Loading dataset: {args.dataset}...")
adj, features, ano_label = load_mat(args.dataset)


features_dense, _ = preprocess_features(features)

# [NEW]: بارگذاری و تبدیل استاندارد به فرمت PyTorch Geometric
nb_nodes = features_dense.shape[0]
ft_size = features_dense.shape[1]

# ویژگی گره: [N, D]
x_pyg = torch.FloatTensor(features_dense).to(device)

# ماتریس مجاورت به edge_index: [2, E]
# ما از adj اصلی (Sparse) استفاده می‌کنیم تا به فرمت PyG تبدیل شود.
edge_index_pyg, _ = from_scipy_sparse_matrix(adj)
edge_index_pyg = edge_index_pyg.to(device)

# بردار Batch: [N] (برای یک گراف تکی، همه گره‌ها به گراف 0 تعلق دارند)
batch_pyg = torch.zeros(nb_nodes, dtype=torch.long).to(device)

# [NEW]: edge_attr: برای Cora ویژگی یال نداریم، پس None ارسال می‌کنیم.
edge_attr_pyg = None 

# --- فاز 2: تعریف و تست مدل (Feedforward Test) ---

print("\n--- Model Setup and Feedforward Test ---")

# [NEW]: استفاده از نام پارامترهای صحیح در GATMamba
model = GATMamba(
    D_NODE_FEAT=ft_size,             # ابعاد ویژگی گره
    D_EDGE_FEAT=0,                   # [NEW]: ابعاد ویژگی یال را 0 می‌گذاریم چون Cora ویژگی یال ندارد
    uni_hidden=args.hidden_dim,      # ابعاد مخفی
    num_model_layers=args.num_layers, # تعداد لایه‌ها
    num_heads=args.num_heads
).to(device)


print(f"Model initialized: GATMamba (Heads={args.num_heads}, Hidden={args.hidden_dim}, Layers={args.num_layers})")
print(f"Input Feature Shape (x): {x_pyg.shape}")
print(f"Edge Index Shape: {edge_index_pyg.shape}")
print(f"Batch Shape: {batch_pyg.shape}")
print(f"Device Check: {x_pyg.device}")


# اجرای Feedforward
model.eval()
with torch.no_grad():
    try:
        # [NEW]: فراخوانی مدل با ورودی‌های استاندارد PyG
        # x_in, edge_index_in, batch_in, edge_attr_in
        prediction, embedding = model(x_pyg, edge_index_pyg, batch_pyg, edge_attr_pyg)

        print("\n--- Feedforward Results ---")
        print(f"Output Prediction Shape (Graph): {prediction.shape}")
        print(f"Output Embedding Shape (Pooled): {embedding.shape}")
        
        # بررسی صحت ابعاد
        expected_embedding_dim = args.hidden_dim * args.num_heads
        if prediction.shape[0] == 1 and \
           embedding.shape[0] == 1 and \
           embedding.shape[1] == expected_embedding_dim:
            print("✅ Tensor shapes match the expected Graph-Level output.")
            print("✅ GPU/Device check passed.")
        else:
            print("❌ WARNING: Output tensor shapes do not match expectations.")
            print(f"  Expected Pooled Embedding Dim: 1 x {expected_embedding_dim}")

    except Exception as e:
        print(f"\n❌ FATAL ERROR during Feedforward: {e}")
        print("Check model implementation and dependencies.")


# --- فاز 3: تعریف تابع زیان، بهینه سازی و آموزش ---
# (این بخش باید در مرحله بعدی اضافه شود)
