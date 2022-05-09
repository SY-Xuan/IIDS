from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .label_smoothing import LabelSmoothing
from .entropy_regularization import SoftLabelLoss, SoftEntropy

__all__ = [
    'TripletLoss',
    'LabelSmoothing',
    'SoftLabelLoss',
    'SoftEntropy',
    'SoftTripletLoss'
]
