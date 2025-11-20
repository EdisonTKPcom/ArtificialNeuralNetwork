"""
Graph Attention Network (GAT) - Veličković et al., 2018.

Uses attention mechanism to weight neighbor contributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_model import BaseModel


class GATLayer(nn.Module):
    """
    Single Graph Attention Layer.
    
    Computes attention coefficients for each edge and aggregates
    neighbor features weighted by attention.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.empty(size=(num_heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Updated features (num_nodes, out_features * num_heads) if concat
                    or (num_nodes, out_features) if not concat
        """
        num_nodes = x.size(0)
        
        # Apply linear transformation for each head
        # (num_nodes, in_features) @ (num_heads, in_features, out_features)
        # -> (num_heads, num_nodes, out_features)
        h = torch.stack([torch.matmul(x, self.W[i]) for i in range(self.num_heads)])
        
        # Compute attention coefficients
        # Concatenate h_i and h_j for all edges
        h_i = h.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # (num_heads, num_nodes, num_nodes, out_features)
        h_j = h.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # (num_heads, num_nodes, num_nodes, out_features)
        
        # Concatenate along feature dimension
        h_cat = torch.cat([h_i, h_j], dim=3)  # (num_heads, num_nodes, num_nodes, 2*out_features)
        
        # Compute attention logits
        e = torch.matmul(h_cat, self.a).squeeze(3)  # (num_heads, num_nodes, num_nodes)
        e = self.leakyrelu(e)
        
        # Mask attention for non-existent edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
        
        # Softmax to get attention weights
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        
        # Apply attention to aggregate neighbor features
        h_prime = torch.matmul(attention, h)  # (num_heads, num_nodes, out_features)
        
        if self.concat:
            # Concatenate heads
            return h_prime.transpose(0, 1).reshape(num_nodes, -1)
        else:
            # Average heads
            return h_prime.mean(dim=0)


class GraphAttentionNetwork(BaseModel):
    """
    Graph Attention Network for node classification.
    
    Uses multi-head attention to learn importance of neighbors.
    Attention weights are computed dynamically based on node features.
    
    Good for: node classification with heterogeneous neighborhoods.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension per attention head
        output_dim: Output dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        output_dim: int = 10,
        num_heads: int = 8,
        dropout: float = 0.6
    ):
        super().__init__()
        
        self.dropout = dropout
        
        # Multi-head attention layers (concatenate outputs)
        self.gat1 = GATLayer(
            input_dim,
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # Final layer (average outputs)
        self.gat2 = GATLayer(
            hidden_dim * num_heads,
            output_dim,
            num_heads=1,
            dropout=dropout,
            concat=False
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Node predictions (num_nodes, output_dim)
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, adj)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, adj)
        
        return x
