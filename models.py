import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)

    def forward(self, data):
        x = self.gcn(data.x, data.edge_index)
        return F.relu(x)