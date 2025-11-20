"""
Artificial Neural Network Architectures Lab

A comprehensive collection of neural network architectures implemented in PyTorch.

Architecture Families:
- Feedforward: Perceptron, MLP, RBF, ELM
- Convolutional: CNN, LeNet, AlexNet, VGG, Inception, ResNet, DenseNet, MobileNet
- Recurrent: RNN, LSTM, GRU, Bidirectional, Seq2Seq
- Transformer: Attention, Transformer, BERT-like, GPT-like, ViT
- Autoencoder: Basic, Denoising, Sparse, Variational (VAE)
- Generative: GAN, DCGAN, WGAN, Diffusion
- Graph: GNN, GCN, GAT, GraphSAGE
- Competitive: SOM, LVQ
- Energy-based: Hopfield, RBM, DBN
- Hybrid: Mixture of Experts, and scaffolds for NTM, DNC, SNN
"""

__version__ = "0.1.0"
__author__ = "ANN Lab Contributors"

# Core utilities
from . import core

# Architecture families
from . import feedforward
from . import conv
from . import recurrent
from . import transformer
from . import autoencoder
from . import generative
from . import graph
from . import competitive
from . import energy
from . import hybrid

__all__ = [
    "core",
    "feedforward",
    "conv",
    "recurrent",
    "transformer",
    "autoencoder",
    "generative",
    "graph",
    "competitive",
    "energy",
    "hybrid",
]
