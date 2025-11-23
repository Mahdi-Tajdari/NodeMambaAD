# utils.py
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl

# توابع دقیقا مشابه کد مرجع
def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset_name):
    """Load .mat dataset."""
    data = sio.loadmat("./{}.mat".format(dataset_name))
    
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    
    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)
    
    ano_labels = np.squeeze(np.array(label))
    
    # [NOTE: Keeping simplified train/val/test split for this utility function]
    num_node = adj.shape[0]
    idx_train = np.arange(int(num_node * 0.3))
    idx_val = np.arange(int(num_node * 0.1))
    idx_test = np.arange(num_node - int(num_node * 0.4))
    
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None
    
    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format. (FIXED FOR NETWORKX VERSION ERROR)"""
    # FIX: Changed from 'from_scipy_sparse_matrix' to 'from_scipy_sparse_array'
    nx_graph = nx.from_scipy_sparse_array(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph
# فقط این ۳ تا تابع رو به انتهای utils.py اضافه کن

# فقط این ۳ تا تابع رو در utils.py کپی کن (جایگزین قبلی‌ها کن)

# utils.py — این تابع رو دقیقاً جایگزین کن (فقط همین یکی!)
def get_topk_neighbors_dgl(g, k=8):
    """سازگار با DGL قدیمی + بدون .device + بدون .number_of_nodes() مشکل"""
    num_nodes = g.number_of_nodes()  # درست برای DGL قدیمی
    adj = g.adjacency_matrix(transpose=False).coalesce()
    src, dst = adj.indices()[0], adj.indices()[1]  # src: منبع، dst: مقصد

    neighbors = []
    for i in range(num_nodes):
        neigh = dst[src == i]  # همسایه‌های نود i
        
        if len(neigh) == 0:
            neigh = torch.tensor([i], dtype=torch.long)
        elif len(neigh) > k:
            # بدون استفاده از g.device — خود تنسور می‌دونه کجاست
            perm = torch.randperm(len(neigh))[:k]
            neigh = neigh[perm]
        else:
            # پد با خود نود
            pad = torch.full((k - len(neigh),), i, dtype=torch.long, device=neigh.device if len(neigh) > 0 else torch.device('cpu'))
            neigh = torch.cat([neigh, pad], dim=0)
        
        neighbors.append(neigh)
    
    return torch.stack(neighbors)  # [N, k]

def structural_encoding_from_adj(adj_dense):
    """از adj [1,N,N] → [1,N,2]"""
    A = adj_dense[0]  # [N,N]
    deg = A.sum(dim=1)
    deg_norm = deg / (deg.max() + 1e-8)
    deg_feat = torch.stack([deg_norm, deg_norm], dim=1)  # [N,2]
    return deg_feat.unsqueeze(0)  # [1,N,2]

def compute_rq_from_adj(features, adj_dense):
    """Rayleigh Quotient ساده و سریع"""
    x = features[0]  # [N,F]
    A = adj_dense[0]  # [N,N]
    D = torch.diag(A.sum(dim=1))
    L = D - A
    xLx = torch.sum(x * torch.matmul(L, x), dim=1)
    xx = torch.sum(x ** 2, dim=1) + 1e-8
    rq = xLx / xx
    return rq.unsqueeze(0).unsqueeze(-1)  # [1,N,1]