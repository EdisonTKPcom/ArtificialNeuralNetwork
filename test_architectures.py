#!/usr/bin/env python
"""
Quick test to verify all major architectures can be imported and instantiated.
"""

import torch
from ann_lab.feedforward import MLPClassifier
from ann_lab.conv import SimpleCNN, LeNet, ResNet18
from ann_lab.recurrent import LSTMClassifier, GRUClassifier
from ann_lab.transformer import BERTLikeEncoder, GPTLikeDecoder
from ann_lab.autoencoder import VariationalAutoencoder
from ann_lab.generative import DCGAN
from ann_lab.graph import GCN, GAT

print('\n' + '='*60)
print('ARTIFICIAL NEURAL NETWORK ARCHITECTURES LAB')
print('='*60)
print()
print('Architecture Examples (with parameter counts):')
print()

print('📊 Feedforward:')
mlp = MLPClassifier(784, [256, 128], 10)
print(f'  • MLP (784→256→128→10): {mlp.num_parameters():,} params')
print()

print('🖼️  Convolutional:')
cnn = SimpleCNN(1, 10)
print(f'  • SimpleCNN: {cnn.num_parameters():,} params')
lenet = LeNet(1, 10)
print(f'  • LeNet: {lenet.num_parameters():,} params')
resnet18 = ResNet18(1, 10)
print(f'  • ResNet-18: {resnet18.num_parameters():,} params')
print()

print('🔄 Recurrent:')
lstm = LSTMClassifier(100, 128, 2, 10)
print(f'  • LSTM (2 layers): {lstm.num_parameters():,} params')
gru = GRUClassifier(100, 128, 2, 10)
print(f'  • GRU (2 layers): {gru.num_parameters():,} params')
print()

print('🤖 Transformer:')
bert = BERTLikeEncoder(5000, 256, 4, 6, 1024, num_classes=2)
print(f'  • BERT-like (6 layers): {bert.num_parameters():,} params')
gpt = GPTLikeDecoder(5000, 256, 4, 6, 1024)
print(f'  • GPT-like (6 layers): {gpt.num_parameters():,} params')
print()

print('🎨 Generative:')
vae = VariationalAutoencoder(784, 20, [256, 128])
print(f'  • VAE (latent=20): {vae.num_parameters():,} params')
dcgan = DCGAN(100, 1, 64)
print(f'  • DCGAN: {dcgan.num_parameters():,} params')
print()

print('🕸️  Graph:')
gcn = GCN(128, 64, 7, 2)
print(f'  • GCN (2 layers): {gcn.num_parameters():,} params')
gat = GAT(128, 8, 7, 8)
print(f'  • GAT (8 heads): {gat.num_parameters():,} params')
print()

print('='*60)
print('✅ All architectures functional and ready to use!')
print('='*60)
print()
