"""
Base Graph Neural Network implementation.

GNNs aggregate information from neighbors to update node representations.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class GraphNeuralNetwork(BaseModel):
    """
    Basic Graph Neural Network with message passing.
    
    Aggregates neighbor features to update node representations.
    
    Good for: node classification, graph classification, link prediction.
    
    Args:
        input_dim: Dimension of node features
        hidden_dim: Dimension of hidden representations
        output_dim: Dimension of output (e.g., number of classes)
        num_layers: Number of GNN layers
        dropout: Dropout probability
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
        
        # Input layer
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def aggregate_neighbors(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Aggregate features from neighbors.
        
        Args:
            x: Node features (num_nodes, feature_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)
            normalize: Whether to normalize by degree
            
        Returns:
            aggregated: Aggregated features (num_nodes, feature_dim)
        """
        if normalize:
            # Add self-loops
            adj = adj + torch.eye(adj.size(0), device=adj.device)
            
            # Degree normalization: D^{-1/2} A D^{-1/2}
            deg = adj.sum(dim=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
            aggregated = torch.matmul(norm, x)
        else:
            aggregated = torch.matmul(adj, x)
        
        return aggregated
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Node predictions (num_nodes, output_dim)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = self.aggregate_neighbors(x, adj)
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.aggregate_neighbors(x, adj)
        x = self.layers[-1](x)
        
        return x
