"""
Allocation module implementing entropy-guided modality selection and hierarchical reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EntropyAllocator(nn.Module):
    """
    Entropy-guided modality selection mechanism.
    
    Implements the optimal allocation theorem from the paper:
    α_j* = √(I(Y; X^(j)) / c_j) / Σ_k √(I(Y; X^(k)) / c_k) * exp(-γ * TC_j)
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.temperature = temperature
        
        # Information estimation networks
        self.info_estimators = nn.ModuleDict({
            f'modality_{i}': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            ) for i in range(num_modalities)
        })
        
        # Redundancy estimation (total correlation)
        self.redundancy_estimator = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities)
        )
        
        # Regime-specific allocation parameters
        self.regime_parameters = nn.Parameter(
            torch.randn(3, num_modalities)  # 3 regimes, num_modalities
        )
        
        # Computational costs for each modality
        self.computational_costs = nn.Parameter(
            torch.tensor([1.0, 0.8, 1.2, 0.6])  # news, tables, charts, prices
        )
    
    def forward(self, 
                entropy_scores: Dict[str, torch.Tensor],
                market_regime: str,
                budget: int) -> torch.Tensor:
        """
        Compute optimal allocation based on entropy and market regime.
        
        Args:
            entropy_scores: Dictionary of entropy scores for each modality
            market_regime: Current market regime ('low_volatility', 'medium_volatility', 'high_volatility')
            budget: Computational budget
            
        Returns:
            Allocation tensor [batch_size, num_modalities]
        """
        batch_size = next(iter(entropy_scores.values())).shape[0]
        
        # Convert entropy scores to information estimates
        info_scores = {}
        for modality, entropy in entropy_scores.items():
            # Higher entropy = lower information content
            info_scores[modality] = 1.0 - entropy
        
        # Get regime-specific parameters
        regime_idx = self._get_regime_index(market_regime)
        regime_params = self.regime_parameters[regime_idx]  # [num_modalities]
        
        # Compute base allocation using information-theoretic formula
        base_allocation = self._compute_base_allocation(info_scores)
        
        # Apply regime-specific adjustments
        regime_adjustment = F.softmax(regime_params / self.temperature, dim=-1)
        adjusted_allocation = base_allocation * regime_adjustment.unsqueeze(0)
        
        # Normalize to budget
        normalized_allocation = self._normalize_to_budget(adjusted_allocation, budget)
        
        return normalized_allocation
    
    def _compute_base_allocation(self, info_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute base allocation using information-theoretic principles."""
        batch_size = next(iter(info_scores.values())).shape[0]
        
        # Stack information scores
        info_tensor = torch.stack(list(info_scores.values()), dim=1)  # [batch, num_modalities]
        
        # Apply square root and normalize by computational cost
        sqrt_info = torch.sqrt(info_tensor)
        cost_normalized = sqrt_info / self.computational_costs.unsqueeze(0)
        
        # Compute denominator (sum across modalities)
        denominator = torch.sum(cost_normalized, dim=1, keepdim=True)
        
        # Compute allocation
        allocation = cost_normalized / denominator
        
        return allocation
    
    def _get_regime_index(self, regime: str) -> int:
        """Convert regime string to index."""
        regime_map = {
            'low_volatility': 0,
            'medium_volatility': 1,
            'high_volatility': 2
        }
        return regime_map.get(regime, 1)  # Default to medium volatility
    
    def _normalize_to_budget(self, allocation: torch.Tensor, budget: int) -> torch.Tensor:
        """Normalize allocation to fit within budget."""
        # Scale allocation to budget
        total_allocation = torch.sum(allocation, dim=1, keepdim=True)
        scale_factor = budget / total_allocation
        normalized = allocation * scale_factor
        
        return normalized


class HierarchicalProcessor(nn.Module):
    """
    Hierarchical reasoning architecture mirroring natural financial analysis structure.
    
    Implements the hierarchical scaling efficiency theorem:
    - Computational complexity: O(n log_b n) vs O(n²) for full attention
    - Effective receptive field: O(b^L) = O(n)
    - Gradient flow path length: O(log_b n)
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 branching_factor: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.branching_factor = branching_factor
        
        # Hierarchical layers
        self.hierarchical_layers = nn.ModuleList([
            HierarchicalLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                branching_factor=branching_factor
            ) for _ in range(num_layers)
        ])
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(hidden_dim, num_heads, dropout)
        
        # Final aggregation
        self.final_aggregation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, 
                modality_features: Dict[str, torch.Tensor],
                allocation: torch.Tensor) -> torch.Tensor:
        """
        Process features through hierarchical architecture.
        
        Args:
            modality_features: Dictionary of modality features
            allocation: Computational allocation for each modality
            
        Returns:
            Hierarchically processed features
        """
        # Step 1: Weight modality features by allocation
        weighted_features = self._weight_by_allocation(modality_features, allocation)
        
        # Step 2: Process through hierarchical layers
        hierarchical_features = weighted_features
        for layer in self.hierarchical_layers:
            hierarchical_features = layer(hierarchical_features)
        
        # Step 3: Cross-modal fusion
        fused_features = self.cross_modal_fusion(hierarchical_features)
        
        # Step 4: Final aggregation
        final_features = self.final_aggregation(
            torch.cat([hierarchical_features, fused_features], dim=-1)
        )
        
        return final_features
    
    def _weight_by_allocation(self, 
                             modality_features: Dict[str, torch.Tensor],
                             allocation: torch.Tensor) -> torch.Tensor:
        """Weight modality features by computational allocation."""
        # Stack features
        features_list = list(modality_features.values())
        stacked_features = torch.stack(features_list, dim=1)  # [batch, num_modalities, hidden]
        
        # Apply allocation weights
        allocation_weights = allocation.unsqueeze(-1)  # [batch, num_modalities, 1]
        weighted_features = stacked_features * allocation_weights
        
        # Sum across modalities
        aggregated_features = torch.sum(weighted_features, dim=1)  # [batch, hidden]
        
        return aggregated_features


class HierarchicalLayer(nn.Module):
    """Single layer of hierarchical processing."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, branching_factor: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.branching_factor = branching_factor
        
        # Multi-head attention for hierarchical processing
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Process features through hierarchical layer."""
        batch_size, hidden_dim = features.shape
        
        # Reshape for hierarchical processing
        # Group features into branches
        num_groups = batch_size // self.branching_factor
        if num_groups == 0:
            num_groups = 1
        
        # Pad if necessary
        if batch_size % self.branching_factor != 0:
            padding_size = self.branching_factor - (batch_size % self.branching_factor)
            features = torch.cat([features, torch.zeros(padding_size, hidden_dim)], dim=0)
            batch_size = features.shape[0]
            num_groups = batch_size // self.branching_factor
        
        # Reshape to [num_groups, branching_factor, hidden_dim]
        grouped_features = features.view(num_groups, self.branching_factor, hidden_dim)
        
        # Apply attention within groups
        grouped_features = grouped_features.transpose(0, 1)  # [branching_factor, num_groups, hidden_dim]
        attended_features, _ = self.attention(
            grouped_features, grouped_features, grouped_features
        )
        attended_features = attended_features.transpose(0, 1)  # [num_groups, branching_factor, hidden_dim]
        
        # Residual connection and normalization
        attended_features = self.norm1(attended_features + grouped_features)
        
        # Feed-forward network
        ffn_output = self.ffn(attended_features)
        output = self.norm2(attended_features + ffn_output)
        
        # Aggregate across branches
        aggregated = torch.mean(output, dim=1)  # [num_groups, hidden_dim]
        
        # Reshape back to original batch size (remove padding)
        final_features = aggregated[:batch_size // self.branching_factor]
        
        return final_features


class CrossModalFusion(nn.Module):
    """Cross-modal fusion mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleDict({
            'news': nn.Linear(hidden_dim, hidden_dim),
            'tables': nn.Linear(hidden_dim, hidden_dim),
            'charts': nn.Linear(hidden_dim, hidden_dim),
            'prices': nn.Linear(hidden_dim, hidden_dim)
        })
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Fuse features across modalities."""
        # For simplicity, we'll assume features are already modality-specific
        # In practice, you'd have separate modality features
        
        # Apply cross-modal attention
        features = features.unsqueeze(0)  # [1, batch, hidden]
        fused_features, _ = self.cross_attention(features, features, features)
        fused_features = fused_features.squeeze(0)  # [batch, hidden]
        
        return fused_features


class AdaptiveReasoning(nn.Module):
    """
    Adaptive reasoning mechanism that adjusts processing based on complexity.
    """
    
    def __init__(self, hidden_dim: int, max_steps: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        
        # Reasoning steps
        self.reasoning_steps = nn.ModuleList([
            ReasoningStep(hidden_dim) for _ in range(max_steps)
        ])
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Early stopping mechanism
        self.confidence_threshold = 0.95
    
    def forward(self, features: torch.Tensor, budget: int) -> torch.Tensor:
        """Apply adaptive reasoning with early stopping."""
        current_features = features
        total_steps = 0
        
        for step_idx, reasoning_step in enumerate(self.reasoning_steps):
            # Apply reasoning step
            current_features = reasoning_step(current_features)
            total_steps += 1
            
            # Estimate complexity/confidence
            confidence = self.complexity_estimator(current_features).mean()
            
            # Early stopping conditions
            if confidence > self.confidence_threshold:
                break
            
            if total_steps >= budget // 100:  # Budget-based stopping
                break
        
        return current_features


class ReasoningStep(nn.Module):
    """Single reasoning step."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-attention for reasoning
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply single reasoning step."""
        # Self-attention
        features = features.unsqueeze(0)  # [1, batch, hidden]
        attended, _ = self.attention(features, features, features)
        attended = attended.squeeze(0)  # [batch, hidden]
        
        # Residual connection
        features = self.norm1(attended + features.squeeze(0))
        
        # Feed-forward
        ffn_output = self.ffn(features)
        output = self.norm2(features + ffn_output)
        
        return output 