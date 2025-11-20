"""
Graph Neural Networks for learning on graph-structured data.
"""

from .gnn_base import GraphNeuralNetwork
from .gcn import GraphConvolutionalNetwork
from .gat import GraphAttentionNetwork
from .graphsage import GraphSAGE

__all__ = [
    'GraphNeuralNetwork',
    'GNN',
    'GraphConvolutionalNetwork',
    'GCN',
    'GraphAttentionNetwork',
    'GAT',
    'GraphSAGE',
]

# Aliases for convenience
GNN = GraphNeuralNetwork
GCN = GraphConvolutionalNetwork
GAT = GraphAttentionNetwork
