from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .entropy_regularization import SoftLabelLoss, SoftEntropy

__all__ = [
    'TripletLoss',
    'SoftLabelLoss',
    'SoftEntropy',
    'SoftTripletLoss'
]
