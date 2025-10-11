"""
Main FinScale model implementing adaptive multi-modal test-time scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging

from .allocation import EntropyAllocator, HierarchicalProcessor
from .data import ModalityEncoder
from .transfer import CrossDomainTransfer
from .regime import MarketRegimeDetector
from .utils import compute_mutual_information, estimate_entropy

logger = logging.getLogger(__name__)

@dataclass
class FinScaleConfig:
    """Configuration for FinScale model."""
    budget: int = 8000  # Computational budget in tokens
    modalities: List[str] = None  # List of modalities to use
    hierarchical_layers: int = 4  # Number of hierarchical layers
    hidden_dim: int = 768  # Hidden dimension for encoders
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout rate
    max_seq_length: int = 512  # Maximum sequence length
    temperature: float = 1.0  # Temperature for allocation softmax
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['news', 'tables', 'charts', 'prices']

class FinScale(nn.Module):
    """
    FinScale: Adaptive Multi-Modal Test-Time Scaling for Financial Time-Series Reasoning.
    
    This model implements the theoretical framework described in the paper, combining:
    1. Entropy-guided modality selection
    2. Hierarchical reasoning architecture  
    3. Cross-domain transfer learning
    4. Market regime adaptation
    """
    
    def __init__(self, config: FinScaleConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.modality_encoders = nn.ModuleDict({
            modality: ModalityEncoder(modality, config.hidden_dim, config.max_seq_length)
            for modality in config.modalities
        })
        
        # Entropy-guided allocation mechanism
        self.entropy_allocator = EntropyAllocator(
            hidden_dim=config.hidden_dim,
            num_modalities=len(config.modalities),
            temperature=config.temperature
        )
        
        # Hierarchical reasoning processor
        self.hierarchical_processor = HierarchicalProcessor(
            hidden_dim=config.hidden_dim,
            num_layers=config.hierarchical_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Cross-domain transfer module
        self.transfer_module = CrossDomainTransfer(
            hidden_dim=config.hidden_dim,
            num_modalities=len(config.modalities)
        )
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(
            hidden_dim=config.hidden_dim,
            num_regimes=3  # low, medium, high volatility
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            'return_prediction': nn.Linear(config.hidden_dim, 2),  # binary classification
            'volatility_regime': nn.Linear(config.hidden_dim, 3),   # low/medium/high
            'earnings_surprise': nn.Linear(config.hidden_dim, 3)    # beat/meet/miss
        })
        
        # Budget forcing mechanism
        self.budget_forcing = BudgetForcing(config.budget)
        
        self.to(self.device)
        
    def forward(self, 
                data: Dict[str, torch.Tensor],
                task: str = 'return_prediction',
                market_regime: Optional[str] = None,
                budget: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive multi-modal allocation.
        
        Args:
            data: Dictionary containing modality data
            task: Target task ('return_prediction', 'volatility_regime', 'earnings_surprise')
            market_regime: Market regime if known, otherwise auto-detected
            budget: Computational budget override
            
        Returns:
            Dictionary containing predictions and allocation information
        """
        if budget is None:
            budget = self.config.budget
            
        # Step 1: Encode modalities and estimate information content
        modality_features = {}
        entropy_scores = {}
        
        for modality in self.config.modalities:
            if modality in data:
                features = self.modality_encoders[modality](data[modality])
                modality_features[modality] = features
                
                # Estimate entropy/information content
                entropy_scores[modality] = estimate_entropy(features)
        
        # Step 2: Detect market regime if not provided
        if market_regime is None:
            regime_features = self._aggregate_modality_features(modality_features)
            regime_probs = self.regime_detector(regime_features)
            market_regime = self._get_regime_from_probs(regime_probs)
        
        # Step 3: Compute optimal allocation based on entropy and regime
        allocation = self.entropy_allocator(
            entropy_scores=entropy_scores,
            market_regime=market_regime,
            budget=budget
        )
        
        # Step 4: Apply budget forcing
        allocation = self.budget_forcing(allocation, budget)
        
        # Step 5: Process through hierarchical architecture
        hierarchical_output = self.hierarchical_processor(
            modality_features=modality_features,
            allocation=allocation
        )
        
        # Step 6: Apply cross-domain transfer if needed
        if hasattr(self, 'source_domain') and hasattr(self, 'target_domain'):
            hierarchical_output = self.transfer_module(
                features=hierarchical_output,
                source_domain=self.source_domain,
                target_domain=self.target_domain
            )
        
        # Step 7: Task-specific prediction
        logits = self.task_heads[task](hierarchical_output)
        probs = F.softmax(logits, dim=-1)
        
        return {
            'predictions': probs,
            'logits': logits,
            'allocation': allocation,
            'market_regime': market_regime,
            'entropy_scores': entropy_scores,
            'hierarchical_features': hierarchical_output
        }
    
    def _aggregate_modality_features(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate modality features for regime detection."""
        features_list = list(modality_features.values())
        return torch.mean(torch.stack(features_list), dim=0)
    
    def _get_regime_from_probs(self, regime_probs: torch.Tensor) -> str:
        """Convert regime probabilities to regime string."""
        regime_idx = torch.argmax(regime_probs, dim=-1).item()
        regimes = ['low_volatility', 'medium_volatility', 'high_volatility']
        return regimes[regime_idx]
    
    def train_step(self, batch: Dict[str, torch.Tensor], task: str) -> Dict[str, float]:
        """Single training step."""
        self.train()
        
        # Forward pass
        outputs = self.forward(batch, task=task)
        
        # Compute loss
        targets = batch[f'{task}_labels']
        loss = F.cross_entropy(outputs['logits'], targets)
        
        # Add allocation regularization
        allocation_penalty = self._compute_allocation_penalty(outputs['allocation'])
        total_loss = loss + 0.1 * allocation_penalty
        
        return {
            'loss': total_loss.item(),
            'task_loss': loss.item(),
            'allocation_penalty': allocation_penalty.item()
        }
    
    def _compute_allocation_penalty(self, allocation: torch.Tensor) -> torch.Tensor:
        """Compute penalty for extreme allocations."""
        # Encourage more balanced allocation
        mean_allocation = torch.mean(allocation, dim=-1, keepdim=True)
        penalty = torch.mean((allocation - mean_allocation) ** 2)
        return penalty
    
    def predict(self, 
                data: Dict[str, torch.Tensor],
                task: str = 'return_prediction',
                market_regime: Optional[str] = None) -> Dict[str, Any]:
        """Make predictions with adaptive allocation."""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(data, task=task, market_regime=market_regime)
            
            # Convert to numpy for easier handling
            predictions = outputs['predictions'].cpu().numpy()
            allocation = outputs['allocation'].cpu().numpy()
            
            return {
                'predictions': predictions,
                'allocation': allocation,
                'market_regime': outputs['market_regime'],
                'entropy_scores': {k: v.cpu().numpy() for k, v in outputs['entropy_scores'].items()}
            }
    
    def set_domains(self, source_domain: str, target_domain: str):
        """Set source and target domains for transfer learning."""
        self.source_domain = source_domain
        self.target_domain = target_domain
    
    def get_allocation_analysis(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze allocation patterns for interpretability."""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(data)
            
            # Compute mutual information between modalities
            modality_features = {}
            for modality in self.config.modalities:
                if modality in data:
                    features = self.modality_encoders[modality](data[modality])
                    modality_features[modality] = features
            
            mutual_info = {}
            modalities = list(modality_features.keys())
            for i, mod1 in enumerate(modalities):
                for j, mod2 in enumerate(modalities):
                    if i < j:
                        mi = compute_mutual_information(
                            modality_features[mod1],
                            modality_features[mod2]
                        )
                        mutual_info[f'{mod1}_{mod2}'] = mi.item()
            
            return {
                'allocation': outputs['allocation'].cpu().numpy(),
                'entropy_scores': {k: v.cpu().numpy() for k, v in outputs['entropy_scores'].items()},
                'mutual_information': mutual_info,
                'market_regime': outputs['market_regime']
            }


class BudgetForcing(nn.Module):
    """Budget forcing mechanism to control computational allocation."""
    
    def __init__(self, max_budget: int):
        super().__init__()
        self.max_budget = max_budget
    
    def forward(self, allocation: torch.Tensor, budget: int) -> torch.Tensor:
        """Apply budget forcing to allocation."""
        # Normalize allocation to sum to budget
        total_allocated = torch.sum(allocation)
        if total_allocated > budget:
            # Scale down allocation to fit budget
            scale_factor = budget / total_allocated
            allocation = allocation * scale_factor
        
        return allocation 