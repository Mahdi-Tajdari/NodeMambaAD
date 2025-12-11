from typing import Any, Dict, Optional
import torch
from torch.nn import (
    Linear,
    ReLU,
    Sequential,
)
from torch_geometric.nn import GATConv, global_mean_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from mamba_ssm import Mamba

# [NEW]: تابع sinusoidal_positional_embedding حذف شد (PE فضایی)

# Copied from https://github.com/bowang-lab/Graph-Mamba/blob/main/notebooks/mamba.ipynb
class GATMambaBlock(torch.nn.Module):

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        att_type: str = 'transformer',
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
        
        if self.att_type == 'transformer':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                dropout=attn_dropout,
                batch_first=True,
            )
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

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        if self.att_type == 'transformer' and hasattr(self.attn, '_reset_parameters'):
             self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()
    
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        # [NEW]: edge_attr را به عنوان آرگومان صریح می‌آوریم.
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        
        # [NEW]: تنظیم آرگومان‌های conv: جلوگیری از ارسال edge_attr=None به GATConv
        conv_kwargs = {}
        if edge_attr is not None and edge_attr.numel() > 0:
            conv_kwargs['edge_attr'] = edge_attr
            
        # Algorithm 2 lines 4-5 (Local Message Passing)
        if self.conv is not None:  
            # h = self.conv(x, edge_index, **kwargs)  <- کد قدیمی
            h = self.conv(x, edge_index, **conv_kwargs) # [NEW]: ارسال مشروط edge_attr
            h = F.dropout(h, p=self.dropout, training=self.training) 
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)
        
        # Algorithm 2 lines 6-7 (Global Attention/Mamba)
        if self.att_type == 'mamba':
            h, mask = to_dense_batch(x, batch)
            h = self.self_attn(h)[mask]
        elif self.att_type == 'transformer': 
            h, mask = to_dense_batch(x, batch)
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, is_causal=False)
            h = h[mask]
            
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        # Algorithm 2 lines 8-9 (Combination and MLP)
        out = sum(hs)  # Combine local and global representations
        out = out + self.mlp(out) # MLP and Residual Connection
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')

class GATMamba(torch.nn.Module):
    # [NEW]: D_NODE_FEAT و D_EDGE_FEAT برای تعریف ابعاد ورودی گراف عمومی
    def __init__(self, D_NODE_FEAT=64, D_EDGE_FEAT=4, uni_hidden=64, gnn_dropout=0.1, mlp_dropout=0.3, num_heads=1, num_model_layers=1):
        super(GATMamba, self).__init__()
        
        self.num_heads_gnn = num_heads
        # [NEW]: PE فضایی حذف شد.
        self.sin_pe_dim = 0 
        
        self.num_uni_features = D_NODE_FEAT
        self.edge_embedding_size = 16
        
        # [NEW]: فرض بر ویژگی یال پیوسته
        self.num_continuous_edge_features = D_EDGE_FEAT 
        self.graph_dim_foundation = uni_hidden
        
        # [NEW]: edge_embedding (برای یال‌های دسته‌ای) حذف شد.
        
        # [NEW]: L_edge برای ویژگی‌های پیوسته یال (حتی اگر D_EDGE_FEAT=0 باشد)
        self.edge_linear_transform = Linear(self.num_continuous_edge_features, self.edge_embedding_size) 
        
        # [NEW]: L_node برای ویژگی‌های گره 
        self.uni_feature_linear_transform = Linear(self.num_uni_features, self.graph_dim_foundation) 
        
        hidden = self.graph_dim_foundation + self.sin_pe_dim

        self.layers = torch.nn.ModuleList()
        for _ in range(num_model_layers):
            self.conv_gat = GATConv(self.graph_dim_foundation + self.sin_pe_dim, hidden, heads = self.num_heads_gnn, dropout = gnn_dropout)
            self.layer = GATMambaBlock(hidden*self.num_heads_gnn, self.conv_gat, attn_dropout=mlp_dropout, dropout = mlp_dropout, att_type='mamba')
            self.layers.append(self.layer)

        self.mlp = Sequential(
            Linear(hidden*self.num_heads_gnn, hidden*self.num_heads_gnn // 2),
            ReLU(),
            Linear(hidden*self.num_heads_gnn // 2, hidden*self.num_heads_gnn // 4),
            ReLU(),
            Dropout(mlp_dropout),
            Linear(hidden*self.num_heads_gnn // 4, 1),
)
    # [NEW]: تابع forward برای پذیرش ورودی‌های استاندارد PyG
    def forward(self, x_in: Tensor, edge_index_in: Adj, batch_in: Optional[Tensor] = None, edge_attr_in: Optional[Tensor] = None):
        
        x = x_in
        edge_index = edge_index_in
        edge_attr = edge_attr_in
        
        # [NEW]: ساخت batch برای یک گراف تکی در صورت لزوم
        if batch_in is None:
             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
             batch = batch_in
             
        # Node feature transformation 
        x_foundation = x
        x_foundation = self.uni_feature_linear_transform(x_foundation) # Algorithm 1 line 7
        
        # [NEW]: x فقط شامل ویژگی‌های خطی شده است.
        x = x_foundation 

        # Edge feature transformation
        e = None
        # [NEW]: اگر edge_attr وجود دارد، آن را تبدیل می‌کنیم.
        if edge_attr is not None and edge_attr.numel() > 0 and self.num_continuous_edge_features > 0:
            e = self.edge_linear_transform(edge_attr)
        # [NEW]: اگر D_EDGE_FEAT = 0 باشد، e همان None می‌ماند.

        for layer in self.layers: # Algorithm 1 lines 11-13
            # [NEW]: e به عنوان edge_attr ارسال می‌شود. اگر None باشد، GATMambaBlock آن را به GATConv ارسال نمی‌کند.
            x = layer(x, edge_index, batch, edge_attr=e)
            x = F.relu(x)

        x = global_mean_pool(x, batch) # Algorithm 1 line 14
        prediction = self.mlp(x) # Algorithm 1 line 16
        
        return prediction, x
