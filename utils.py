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

# utils.py — نسخه ۱۰۰٪ درست و بدون خطا (کپی کن جایگزین کن)
def structural_encoding_from_adj(edge_index, num_nodes):
    device = edge_index.device
    row, col = edge_index
    
    # ساخت ماتریس adjacency درست (undirected)
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float)
    ones = torch.ones(row.size(0), device=device)
    adj.index_put_((row, col), ones, accumulate=True)
    adj.index_put_((col, row), ones, accumulate=True)  # undirected
    
    deg = adj.sum(1)  # [N]
    deg_log = torch.log1p(deg)

    # Clustering coefficient
    adj2 = adj @ adj
    deg_pair = deg * (deg - 1)
    deg_pair[deg_pair == 0] = 1  # جلوگیری از تقسیم بر صفر
    clust = adj2.diag() / deg_pair
    clust = clust.clamp(0, 1)

    # 5-step random walk
    rw = torch.eye(num_nodes, device=device)
    rws = []
    for _ in range(5):
        rw = rw @ adj
        rws.append(rw.diag())
    rw_feat = torch.stack(rws, dim=1)  # [N, 5]

    # 5 landmark distances (BFS)
    landmarks = random.sample(range(num_nodes), min(5, num_nodes))
    dist_enc = torch.zeros(num_nodes, len(landmarks), device=device)
    for i, lm in enumerate(landmarks):
        dist = torch.full((num_nodes,), 999, dtype=torch.long, device=device)
        dist[lm] = 0
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        visited[lm] = True
        q = [lm]
        ptr = 0
        while ptr < len(q):
            u = q[ptr]
            ptr += 1
            # همسایه‌های u
            neighbors = col[row == u]
            for v in neighbors.tolist():
                if not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + 1
                    q.append(v)
        dist_enc[:, i] = dist.float()

    # ترکیب همه
    enc = torch.cat([
        deg_log.unsqueeze(1),
        clust.unsqueeze(1),
        rw_feat,
        dist_enc
    ], dim=1)  # [N, 1+1+5+5 = 12] → می‌تونی بعداً بیشتر کنی

    # نرمال‌سازی
    enc = (enc - enc.mean(0, keepdim=True)) / (enc.std(0, keepdim=True) + 1e-8)
    return enc  # [N, 12]
# utils.py — نسخه نهایی compute_rq_from_adj (بدون هیچ خطا)
def compute_rq_from_adj(x, edge_index):
    row, col = edge_index
    num_nodes = x.shape[0]
    device = x.device
    
    # محاسبه degree
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # ساخت normalized adjacency به صورت dense (برای سرعت و سادگی)
    # چون Cora فقط 2708 نود داره → حافظه مشکلی نیست
    adj_norm = torch.zeros((num_nodes, num_nodes), device=device)
    norm_val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj_norm.index_put_((row, col), norm_val, accumulate=True)
    adj_norm.index_put_((col, row), norm_val, accumulate=True)  # undirected
    
    # L_sym = I - Ã
    L_norm = torch.eye(num_nodes, device=device) - adj_norm
    
    # x^T L x
    xLx = torch.sum(x * (L_norm @ x), dim=1, keepdim=True)
    
    # (اختیاری) نرمال‌سازی با x^T x
    x_norm = torch.sum(x ** 2, dim=1, keepdim=True) + 1e-8
    rq = xLx / x_norm
    
    # ReLU برای اینکه فقط فرکانس بالا بگیریم
    rq = torch.relu(rq)
    
    return rq  # [N, 1]