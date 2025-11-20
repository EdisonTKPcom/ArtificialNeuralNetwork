"""
Autoencoder architectures for representation learning.

Autoencoders learn compressed representations through reconstruction.
"""

from .autoencoder_basic import BasicAutoencoder, ConvolutionalAutoencoder
from .denoising_autoencoder import DenoisingAutoencoder
from .sparse_autoencoder import SparseAutoencoder
from .variational_autoencoder import VariationalAutoencoder

__all__ = [
    'BasicAutoencoder',
    'ConvolutionalAutoencoder',
    'DenoisingAutoencoder',
    'SparseAutoencoder',
    'VariationalAutoencoder',
]
