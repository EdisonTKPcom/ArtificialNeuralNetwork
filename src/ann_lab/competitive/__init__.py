"""
Competitive learning networks (unsupervised).
"""

from .som import SelfOrganizingMap
from .lvq import LearningVectorQuantization

__all__ = [
    'SelfOrganizingMap',
    'LearningVectorQuantization',
]
