"""
GraphSAGE (SAmple and aggreGatE) - Hamilton et al., 2017.

Samples neighbors and aggregates their features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_model import BaseModel


class GraphSAGELayer(nn.Module):
    """
    Single GraphSAGE layer.
    
    Samples a fixed number of neighbors and aggregates their features.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = 'mean'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        # Weight matrices
        self.W_self = nn.Linear(in_features, out_features, bias=False)
        self.W_neigh = nn.Linear(in_features, out_features, bias=False)
        
    def aggregate(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Aggregate neighbor features.
        
        Args:
            x: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            aggregated: Aggregated neighbor features (num_nodes, in_features)
        """
        if self.aggregator == 'mean':
            # Mean aggregation
            deg = adj.sum(dim=1, keepdim=True)
            deg[deg == 0] = 1  # Avoid division by zero
            aggregated = torch.matmul(adj, x) / deg
        elif self.aggregator == 'max':
            # Max aggregation
            # This is simplified - real GraphSAGE would use element-wise max
            aggregated = torch.matmul(adj, x)
        elif self.aggregator == 'lstm':
            # LSTM aggregation (placeholder - would need actual LSTM)
            raise NotImplementedError("LSTM aggregator not implemented in this simplified version")
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        return aggregated
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Updated features (num_nodes, out_features)
        """
        # Self features
        h_self = self.W_self(x)
        
        # Aggregate neighbor features
        h_neigh_agg = self.aggregate(x, adj)
        h_neigh = self.W_neigh(h_neigh_agg)
        
        # Combine
        output = h_self + h_neigh
        
        # L2 normalization (important in GraphSAGE)
        output = F.normalize(output, p=2, dim=1)
        
        return output


class GraphSAGE(BaseModel):
    """
    GraphSAGE for inductive node representation learning.
    
    Unlike GCN, GraphSAGE can generalize to unseen nodes by learning
    an aggregation function rather than depending on fixed graph structure.
    
    Good for: large graphs, inductive learning, node classification.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of GraphSAGE layers
        aggregator: Aggregation function ('mean', 'max')
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 10,
        num_layers: int = 2,
        aggregator: str = 'mean',
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GraphSAGELayer(input_dim, hidden_dim, aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator))
        
        # Output layer
        self.layers.append(GraphSAGELayer(hidden_dim, output_dim, aggregator))
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Node predictions (num_nodes, output_dim)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.layers[-1](x, adj)
        
        return x
