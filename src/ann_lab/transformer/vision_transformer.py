"""
Vision Transformer (ViT) for image classification.

Applies transformers to image patches, treating images as sequences of patches.
"""

import torch
import torch.nn as nn
from typing import Optional
from ..core.base_model import BaseModel
from .transformer_encoder_decoder import TransformerEncoderLayer


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv2d with kernel=patch_size and stride=patch_size
        # effectively splits image into patches and projects them
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (batch, channels, height, width)
            
        Returns:
            patches: Embedded patches (batch, num_patches, embed_dim)
        """
        # x: (batch, channels, img_size, img_size)
        x = self.projection(x)  # (batch, embed_dim, num_patches_h, num_patches_w)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        return x


class VisionTransformer(BaseModel):
    """
    Vision Transformer (ViT) for image classification.
    
    Splits images into patches, linearly embeds them, and processes
    with standard transformer encoder layers.
    
    Based on "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).
    
    Good for: image classification, especially with large datasets.
    
    Args:
        img_size: Input image size (square)
        patch_size: Size of image patches
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ff_dim: Feedforward hidden dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable [CLS] token (used for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable positional embeddings
        # +1 for [CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (batch, channels, height, width)
            
        Returns:
            logits: Classification logits (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Use [CLS] token for classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Extract attention maps from all layers (for visualization).
        
        Args:
            x: Image tensor (batch, channels, height, width)
            
        Returns:
            attention_maps: List of attention maps from each layer
        """
        batch_size = x.size(0)
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        attention_maps = []
        for layer in self.encoder_layers:
            # This is simplified - would need to modify layer to return attention weights
            x = layer(x)
            # Placeholder for actual attention extraction
            attention_maps.append(None)
        
        return attention_maps


class HybridViT(BaseModel):
    """
    Hybrid Vision Transformer with CNN backbone.
    
    Uses a CNN to extract features before applying transformer.
    Can be more sample-efficient than pure ViT on smaller datasets.
    
    Args:
        backbone: CNN backbone (e.g., small ResNet)
        feature_dim: Output dimension of CNN backbone
        num_classes: Number of output classes
        embed_dim: Transformer embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        ff_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.backbone = backbone
        
        # Project CNN features to transformer dimension
        self.feature_proj = nn.Linear(feature_dim, embed_dim)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (will be initialized based on feature map size)
        self.pos_embed = None
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (batch, channels, height, width)
            
        Returns:
            logits: Classification logits (batch, num_classes)
        """
        # Extract CNN features
        features = self.backbone(x)  # (batch, feature_dim, h, w)
        
        # Flatten and project
        batch_size = features.size(0)
        features = features.flatten(2).transpose(1, 2)  # (batch, h*w, feature_dim)
        features = self.feature_proj(features)  # (batch, h*w, embed_dim)
        
        # Initialize positional embeddings if needed
        if self.pos_embed is None or self.pos_embed.size(1) != features.size(1) + 1:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, features.size(1) + 1, features.size(2))
            ).to(features.device)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, features], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits
