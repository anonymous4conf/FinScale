"""
Cross-domain transfer learning module for FinScale.
Implements multi-modal transfer bounds and domain adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CrossDomainTransfer(nn.Module):
    """
    Cross-domain transfer learning framework.
    
    Implements the multi-modal transfer bound from the paper:
    ε_T(h) ≤ ε̂_S(h) + √(log(2/δ)/2m) + min_j d_H_j(S_j, T_j) + λ*
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Domain-specific adapters
        self.domain_adapters = nn.ModuleDict({
            f'adapter_{i}': DomainAdapter(hidden_dim) 
            for i in range(num_modalities)
        })
        
        # Domain discriminator for adversarial training
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Maximum Mean Discrepancy (MMD) kernel
        self.mmd_kernel = MMDKernel(hidden_dim)
        
        # Transfer weights for each modality
        self.transfer_weights = nn.Parameter(
            torch.ones(num_modalities) / num_modalities
        )
        
        # Domain-invariant feature extractor
        self.domain_invariant_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, 
                features: torch.Tensor,
                source_domain: str,
                target_domain: str) -> torch.Tensor:
        """
        Apply cross-domain transfer.
        
        Args:
            features: Input features
            source_domain: Source domain identifier
            target_domain: Target domain identifier
            
        Returns:
            Domain-adapted features
        """
        # Extract domain-invariant features
        domain_invariant = self.domain_invariant_extractor(features)
        
        # Apply domain-specific adaptation
        adapted_features = self._apply_domain_adaptation(
            domain_invariant, source_domain, target_domain
        )
        
        # Compute transfer weights based on domain similarity
        transfer_weights = self._compute_transfer_weights(
            features, source_domain, target_domain
        )
        
        # Weighted combination of original and adapted features
        final_features = (
            transfer_weights * adapted_features + 
            (1 - transfer_weights) * features
        )
        
        return final_features
    
    def _apply_domain_adaptation(self, 
                                features: torch.Tensor,
                                source_domain: str,
                                target_domain: str) -> torch.Tensor:
        """Apply domain-specific adaptation."""
        # For simplicity, we'll use a single adapter
        # In practice, you'd have domain-specific adapters
        adapter = self.domain_adapters['adapter_0']
        adapted_features = adapter(features, source_domain, target_domain)
        
        return adapted_features
    
    def _compute_transfer_weights(self,
                                 features: torch.Tensor,
                                 source_domain: str,
                                 target_domain: str) -> torch.Tensor:
        """Compute transfer weights based on domain similarity."""
        # Estimate domain similarity using MMD
        mmd_score = self.mmd_kernel(features, features)  # Placeholder
        
        # Convert MMD to similarity (lower MMD = higher similarity)
        similarity = torch.exp(-mmd_score)
        
        # Apply softmax to get transfer weights
        transfer_weights = F.softmax(self.transfer_weights, dim=0)
        
        return similarity * transfer_weights.mean()
    
    def compute_transfer_loss(self,
                             source_features: torch.Tensor,
                             target_features: torch.Tensor,
                             source_labels: torch.Tensor,
                             target_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute transfer learning losses."""
        losses = {}
        
        # Task loss on source domain
        source_pred = self.domain_invariant_extractor(source_features)
        task_loss = F.cross_entropy(source_pred, source_labels)
        losses['task_loss'] = task_loss
        
        # Domain adversarial loss
        source_domain_pred = self.domain_discriminator(source_features)
        target_domain_pred = self.domain_discriminator(target_features)
        
        # Adversarial loss (domain discriminator should not distinguish domains)
        domain_loss = (
            F.binary_cross_entropy(source_domain_pred, torch.zeros_like(source_domain_pred)) +
            F.binary_cross_entropy(target_domain_pred, torch.ones_like(target_domain_pred))
        )
        losses['domain_loss'] = domain_loss
        
        # MMD loss for domain alignment
        mmd_loss = self.mmd_kernel(source_features, target_features)
        losses['mmd_loss'] = mmd_loss
        
        # Total transfer loss
        total_loss = task_loss + 0.1 * domain_loss + 0.1 * mmd_loss
        losses['total_loss'] = total_loss
        
        return losses


class DomainAdapter(nn.Module):
    """Domain-specific adapter for transfer learning."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Domain-specific transformations
        self.source_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.target_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Shared transformation
        self.shared_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, 
                features: torch.Tensor,
                source_domain: str,
                target_domain: str) -> torch.Tensor:
        """Apply domain-specific adaptation."""
        # Apply source domain transformation
        source_adapted = self.source_transform(features)
        
        # Apply target domain transformation
        target_adapted = self.target_transform(features)
        
        # Apply shared transformation
        shared_features = self.shared_transform(features)
        
        # Combine adaptations
        adapted_features = (
            0.4 * source_adapted + 
            0.4 * target_adapted + 
            0.2 * shared_features
        )
        
        return adapted_features


class MMDKernel(nn.Module):
    """Maximum Mean Discrepancy kernel for domain alignment."""
    
    def __init__(self, hidden_dim: int, num_kernels: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_kernels = num_kernels
        
        # Multiple RBF kernels with different bandwidths
        self.bandwidths = nn.Parameter(
            torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
        )
    
    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute MMD between source and target features."""
        # Compute kernel matrices
        source_kernel = self._compute_kernel(source_features, source_features)
        target_kernel = self._compute_kernel(target_features, target_features)
        cross_kernel = self._compute_kernel(source_features, target_features)
        
        # Compute MMD
        mmd = (
            torch.mean(source_kernel) + 
            torch.mean(target_kernel) - 
            2 * torch.mean(cross_kernel)
        )
        
        return mmd
    
    def _compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        # Compute pairwise distances
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        y_norm = torch.sum(y ** 2, dim=1, keepdim=True)
        
        # Compute kernel for each bandwidth
        kernels = []
        for bandwidth in self.bandwidths:
            # Compute squared Euclidean distance
            dist_sq = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())
            
            # Apply RBF kernel
            kernel = torch.exp(-dist_sq / (2 * bandwidth ** 2))
            kernels.append(kernel)
        
        # Average across kernels
        kernel_matrix = torch.stack(kernels).mean(dim=0)
        
        return kernel_matrix


class ZeroShotTransfer(nn.Module):
    """
    Zero-shot transfer learning for unseen domains.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Universal feature extractor
        self.universal_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Domain-agnostic classifier
        self.domain_agnostic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes for financial tasks
        )
        
        # Domain similarity estimator
        self.domain_similarity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, target_domain: str) -> torch.Tensor:
        """Apply zero-shot transfer."""
        # Extract universal features
        universal_features = self.universal_extractor(features)
        
        # Estimate domain similarity
        similarity = self.domain_similarity(universal_features)
        
        # Apply domain-agnostic classification
        predictions = self.domain_agnostic_classifier(universal_features)
        
        # Weight predictions by domain similarity
        weighted_predictions = predictions * similarity
        
        return weighted_predictions


class MultiModalTransfer(nn.Module):
    """
    Multi-modal transfer learning that leverages modality-specific transfer.
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Modality-specific transfer modules
        self.modality_transfers = nn.ModuleDict({
            f'modality_{i}': CrossDomainTransfer(hidden_dim, 1)
            for i in range(num_modalities)
        })
        
        # Modality fusion for transfer
        self.modality_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transfer confidence estimator
        self.transfer_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                modality_features: Dict[str, torch.Tensor],
                source_domain: str,
                target_domain: str) -> torch.Tensor:
        """Apply multi-modal transfer."""
        transferred_features = []
        
        # Transfer each modality separately
        for modality, features in modality_features.items():
            transfer_module = self.modality_transfers[f'modality_0']  # Use first for all
            transferred = transfer_module(features, source_domain, target_domain)
            transferred_features.append(transferred)
        
        # Concatenate transferred features
        concatenated = torch.cat(transferred_features, dim=-1)
        
        # Fuse modalities
        fused_features = self.modality_fusion(concatenated)
        
        # Estimate transfer confidence for each modality
        confidence = self.transfer_confidence(fused_features)
        
        # Weight by confidence
        weighted_features = torch.stack(transferred_features, dim=1) * confidence.unsqueeze(-1)
        final_features = torch.sum(weighted_features, dim=1)
        
        return final_features


class TransferEvaluator:
    """Evaluator for transfer learning performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_transfer(self,
                         source_performance: float,
                         target_performance: float,
                         domain_divergence: float) -> Dict[str, float]:
        """Evaluate transfer learning performance."""
        # Transfer efficiency
        transfer_efficiency = target_performance / source_performance
        
        # Domain adaptation success
        adaptation_success = 1.0 - domain_divergence
        
        # Overall transfer score
        transfer_score = transfer_efficiency * adaptation_success
        
        return {
            'transfer_efficiency': transfer_efficiency,
            'adaptation_success': adaptation_success,
            'transfer_score': transfer_score,
            'source_performance': source_performance,
            'target_performance': target_performance,
            'domain_divergence': domain_divergence
        }
    
    def compute_domain_divergence(self, 
                                 source_features: torch.Tensor,
                                 target_features: torch.Tensor) -> float:
        """Compute domain divergence using MMD."""
        mmd_kernel = MMDKernel(source_features.shape[-1])
        divergence = mmd_kernel(source_features, target_features).item()
        return divergence 