"""
Graph Convolutional Network (GCN) - Kipf & Welling, 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_model import BaseModel


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer.
    
    H' = σ(D^{-1/2} Â D^{-1/2} H W)
    where Â = A + I (adjacency with self-loops)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_features)
            adj_norm: Normalized adjacency matrix
            
        Returns:
            output: Updated features (num_nodes, out_features)
        """
        # Aggregate neighbors
        support = self.linear(x)
        output = torch.matmul(adj_norm, support)
        return output


class GraphConvolutionalNetwork(BaseModel):
    """
    Graph Convolutional Network for node classification.
    
    Applies graph convolutions to propagate information across the graph.
    Each layer aggregates features from neighbors weighted by edge structure.
    
    Good for: semi-supervised node classification, graph-level prediction.
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Hidden dimension
        output_dim: Output dimension (number of classes)
        num_layers: Number of GCN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 10,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        self.layers.append(GCNLayer(hidden_dim, output_dim))
        
    def normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix: D^{-1/2} (A + I) D^{-1/2}
        
        Args:
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            adj_norm: Normalized adjacency matrix
        """
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Compute degree matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize: D^{-1/2} A D^{-1/2}
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        return adj_norm
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Node predictions (num_nodes, output_dim)
        """
        # Normalize adjacency
        adj_norm = self.normalize_adjacency(adj)
        
        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj_norm)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation/dropout)
        x = self.layers[-1](x, adj_norm)
        
        return x
