from typing import Any, Dict, Optional
import torch
from torch.nn import (
    Linear,
    ReLU,
    Sequential,
    Dropout
)
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from mamba_ssm import Mamba

class GATMambaBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        att_type: str = 'mamba',
        d_state: int = 16,
        d_conv: int = 4,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
                
        if self.att_type == 'mamba':
            self.self_attn = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=1
            )
            
        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        hs = []
        if self.conv is not None:  
            h_local = self.conv(x, edge_index, edge_attr=edge_attr, **kwargs) 
            h_local = F.dropout(h_local, p=self.dropout, training=self.training) 
            h_local = self.norm1(h_local) if self.norm1 is not None else h_local
            hs.append(h_local)
        
        if self.att_type == 'mamba':
            h_global, mask = to_dense_batch(x, batch)
            h_global = self.self_attn(h_global)[mask]
            h_global = F.dropout(h_global, p=self.dropout, training=self.training)
            h_global = self.norm2(h_global) if self.norm2 is not None else h_global
            hs.append(h_global)
        
        out = sum(hs) + x
        out = out + self.mlp(out)
        out = self.norm3(out) if self.norm3 is not None else out
        
        return out

class GATMambaAutoencoder(torch.nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, gnn_dropout=0.1, mlp_dropout=0.3, num_heads=1):
        super(GATMambaAutoencoder, self).__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_transform = Linear(in_features, hidden_dim * num_heads)

        self.encoder_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv_gat = GATConv(hidden_dim * num_heads, hidden_dim, heads = num_heads, dropout = gnn_dropout)
            layer = GATMambaBlock(
                channels=hidden_dim * num_heads,
                conv=conv_gat,
                heads=num_heads,
                attn_dropout=mlp_dropout,
                dropout=mlp_dropout,
                att_type='mamba'
            )
            self.encoder_layers.append(layer)
        
        self.decoder = Sequential(
            Linear(hidden_dim * num_heads, hidden_dim),
            ReLU(),
            Linear(hidden_dim, in_features)
        )

    def forward(self, features, edge_index, edge_attr=None):
        
        x = self.input_transform(features)

        for layer in self.encoder_layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
        
        H = x 

        x_reconstruction = self.decoder(H)
        
        return x_reconstruction, H

class Model(torch.nn.Module):
    def __init__(self, ft_size, hidden_dim=64, num_layers=2, num_heads=1):
        super().__init__()
        self.autoencoder = GATMambaAutoencoder(
            in_features=ft_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
    
    def forward(self, features, adj_tensor, edge_index=None, edge_attr=None):
        
        # اطمینان از اینکه همه تنسورها روی دستگاه صحیح هستند
        current_device = features.device
        
        x = features.squeeze(0)
        
        if edge_index is None:
            # انتقال adj به دستگاه ویژگی ها قبل از تبدیل
            adj = adj_tensor.squeeze(0).to(current_device)
            
            # استخراج edge_index
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        
        # اطمینان از اینکه edge_index روی دستگاه صحیح است (اگر از قبل نبود)
        edge_index = edge_index.to(current_device)
        
        reconstruction, embedding = self.autoencoder(x, edge_index, edge_attr)
        
        return reconstruction, embedding
