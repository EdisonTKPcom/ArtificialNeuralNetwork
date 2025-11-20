"""
Generative Adversarial Networks and other generative models.
"""

from .gan_basic import BasicGAN, Generator, Discriminator
from .dcgan import DCGAN, DCGenerator, DCDiscriminator
from .wgan import WGAN, WGANDiscriminator
from .diffusion_toy import SimpleDiffusion

__all__ = [
    'BasicGAN',
    'Generator',
    'Discriminator',
    'DCGAN',
    'DCGenerator',
    'DCDiscriminator',
    'WGAN',
    'WGANDiscriminator',
    'SimpleDiffusion',
]
