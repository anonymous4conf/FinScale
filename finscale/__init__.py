"""
FinScale: Adaptive Multi-Modal Test-Time Scaling for Financial Time-Series Reasoning

A theoretical framework for adaptive multi-modal test-time scaling that dynamically 
allocates computational resources based on market conditions and data informativeness.
"""

from .model import FinScale
from .data import FinMultiTimeDataset, ModalityEncoder
from .allocation import EntropyAllocator, HierarchicalProcessor
from .transfer import CrossDomainTransfer
from .regime import MarketRegimeDetector
from .utils import compute_mutual_information, estimate_entropy

__version__ = "1.0.0"
__author__ = "FinScale Team"

__all__ = [
    "FinScale",
    "FinMultiTimeDataset", 
    "ModalityEncoder",
    "EntropyAllocator",
    "HierarchicalProcessor",
    "CrossDomainTransfer",
    "MarketRegimeDetector",
    "compute_mutual_information",
    "estimate_entropy"
] 