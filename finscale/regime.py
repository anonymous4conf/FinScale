"""
Market regime detection and adaptation module for FinScale.
Implements regime-adaptive allocation under uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MarketRegimeDetector(nn.Module):
    """
    Market regime detector implementing the regime-adaptive allocation theorem.
    
    Under regime uncertainty, the optimal allocation is:
    α*_t = Σ_k P(R_t = k | D_1:t) * α*_k
    """
    
    def __init__(self, hidden_dim: int, num_regimes: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        
        # Regime detection network
        self.regime_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Regime-specific parameters
        self.regime_parameters = nn.Parameter(
            torch.randn(num_regimes, 4)  # 4 modalities
        )
        
        # Temporal dynamics (Markov transition matrix)
        self.transition_matrix = nn.Parameter(
            torch.softmax(torch.randn(num_regimes, num_regimes), dim=-1)
        )
        
        # Regime-specific allocation strategies
        self.regime_allocators = nn.ModuleList([
            RegimeSpecificAllocator(hidden_dim, 4)  # 4 modalities
            for _ in range(num_regimes)
        ])
        
        # Historical regime memory
        self.regime_memory = []
        self.max_memory = 100
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Detect current market regime.
        
        Args:
            features: Input features for regime detection
            
        Returns:
            Regime probabilities [batch_size, num_regimes]
        """
        # Current regime probabilities
        current_probs = self.regime_detector(features)
        
        # Incorporate temporal dynamics if we have history
        if len(self.regime_memory) > 0:
            # Get previous regime
            prev_regime = self.regime_memory[-1]
            
            # Apply transition dynamics
            transition_probs = self.transition_matrix[prev_regime]
            current_probs = current_probs * transition_probs.unsqueeze(0)
            current_probs = F.softmax(current_probs, dim=-1)
        
        # Update memory
        if len(self.regime_memory) >= self.max_memory:
            self.regime_memory.pop(0)
        
        # Store current regime (most likely)
        current_regime = torch.argmax(current_probs, dim=-1).item()
        self.regime_memory.append(current_regime)
        
        return current_probs
    
    def get_regime_allocation(self, 
                             features: torch.Tensor,
                             regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Get regime-adaptive allocation.
        
        Args:
            features: Input features
            regime_probs: Regime probabilities [batch_size, num_regimes]
            
        Returns:
            Adaptive allocation [batch_size, num_modalities]
        """
        batch_size = features.shape[0]
        
        # Get allocation for each regime
        regime_allocations = []
        for i in range(self.num_regimes):
            allocator = self.regime_allocators[i]
            regime_allocation = allocator(features)
            regime_allocations.append(regime_allocation)
        
        # Stack allocations
        regime_allocations = torch.stack(regime_allocations, dim=1)  # [batch, regimes, modalities]
        
        # Weight by regime probabilities
        regime_probs = regime_probs.unsqueeze(-1)  # [batch, regimes, 1]
        weighted_allocation = regime_allocations * regime_probs
        
        # Sum across regimes
        final_allocation = torch.sum(weighted_allocation, dim=1)  # [batch, modalities]
        
        return final_allocation


class RegimeSpecificAllocator(nn.Module):
    """Allocator specific to a particular market regime."""
    
    def __init__(self, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Regime-specific allocation network
        self.allocation_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Regime-specific feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute regime-specific allocation."""
        # Extract regime-specific features
        regime_features = self.feature_extractor(features)
        
        # Compute allocation
        allocation = self.allocation_network(regime_features)
        
        return allocation


class VolatilityRegimeDetector(nn.Module):
    """
    Specialized detector for volatility regimes (low, medium, high).
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Volatility-specific features
        self.volatility_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 volatility levels
        )
        
        # Volatility thresholds
        self.low_threshold = 0.15
        self.high_threshold = 0.35
        
        # Regime-specific allocation strategies
        self.low_vol_allocation = nn.Parameter(torch.tensor([0.2, 0.4, 0.2, 0.2]))  # news, tables, charts, prices
        self.medium_vol_allocation = nn.Parameter(torch.tensor([0.3, 0.2, 0.2, 0.3]))
        self.high_vol_allocation = nn.Parameter(torch.tensor([0.4, 0.1, 0.2, 0.3]))
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Detect volatility regime.
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (regime_probs, regime_name)
        """
        # Extract volatility features
        volatility_features = self.volatility_extractor(features)
        regime_probs = F.softmax(volatility_features, dim=-1)
        
        # Determine regime
        regime_idx = torch.argmax(regime_probs, dim=-1).item()
        regime_names = ['low_volatility', 'medium_volatility', 'high_volatility']
        regime_name = regime_names[regime_idx]
        
        return regime_probs, regime_name
    
    def get_volatility_allocation(self, regime_name: str) -> torch.Tensor:
        """Get allocation for specific volatility regime."""
        if regime_name == 'low_volatility':
            return self.low_vol_allocation
        elif regime_name == 'medium_volatility':
            return self.medium_vol_allocation
        elif regime_name == 'high_volatility':
            return self.high_vol_allocation
        else:
            return self.medium_vol_allocation  # Default


class MarketSentimentDetector(nn.Module):
    """
    Market sentiment detector for regime identification.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Sentiment analysis network
        self.sentiment_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # bullish, neutral, bearish
        )
        
        # Sentiment-based allocation
        self.bullish_allocation = nn.Parameter(torch.tensor([0.25, 0.35, 0.2, 0.2]))
        self.neutral_allocation = nn.Parameter(torch.tensor([0.3, 0.25, 0.2, 0.25]))
        self.bearish_allocation = nn.Parameter(torch.tensor([0.4, 0.15, 0.2, 0.25]))
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Detect market sentiment."""
        sentiment_features = self.sentiment_analyzer(features)
        sentiment_probs = F.softmax(sentiment_features, dim=-1)
        
        # Determine sentiment
        sentiment_idx = torch.argmax(sentiment_probs, dim=-1).item()
        sentiment_names = ['bullish', 'neutral', 'bearish']
        sentiment_name = sentiment_names[sentiment_idx]
        
        return sentiment_probs, sentiment_name
    
    def get_sentiment_allocation(self, sentiment_name: str) -> torch.Tensor:
        """Get allocation for specific sentiment."""
        if sentiment_name == 'bullish':
            return self.bullish_allocation
        elif sentiment_name == 'neutral':
            return self.neutral_allocation
        elif sentiment_name == 'bearish':
            return self.bearish_allocation
        else:
            return self.neutral_allocation  # Default


class RegimeTransitionModel(nn.Module):
    """
    Model for predicting regime transitions.
    """
    
    def __init__(self, hidden_dim: int, num_regimes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        
        # Transition prediction network
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_regimes * num_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Historical regime sequence
        self.regime_history = []
        self.max_history = 50
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict regime transition probabilities."""
        # Predict transition matrix
        transition_logits = self.transition_predictor(features)
        transition_probs = transition_logits.view(-1, self.num_regimes, self.num_regimes)
        
        return transition_probs
    
    def update_history(self, regime: int):
        """Update regime history."""
        self.regime_history.append(regime)
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)
    
    def get_transition_probability(self, from_regime: int, to_regime: int) -> float:
        """Get transition probability between regimes."""
        if len(self.regime_history) < 2:
            return 1.0 / self.num_regimes  # Uniform if no history
        
        # Count transitions
        transitions = 0
        total_from = 0
        
        for i in range(len(self.regime_history) - 1):
            if self.regime_history[i] == from_regime:
                total_from += 1
                if self.regime_history[i + 1] == to_regime:
                    transitions += 1
        
        if total_from == 0:
            return 1.0 / self.num_regimes
        
        return transitions / total_from


class AdaptiveRegimeAllocator(nn.Module):
    """
    Adaptive allocator that adjusts based on regime uncertainty.
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Regime detectors
        self.volatility_detector = VolatilityRegimeDetector(hidden_dim)
        self.sentiment_detector = MarketSentimentDetector(hidden_dim)
        
        # Adaptive fusion
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute adaptive allocation based on regime detection."""
        # Detect volatility regime
        vol_probs, vol_regime = self.volatility_detector(features)
        vol_allocation = self.volatility_detector.get_volatility_allocation(vol_regime)
        
        # Detect sentiment regime
        sent_probs, sent_regime = self.sentiment_detector(features)
        sent_allocation = self.sentiment_detector.get_sentiment_allocation(sent_regime)
        
        # Combine features for adaptive fusion
        combined_features = torch.cat([features, features], dim=-1)  # Placeholder
        adaptive_allocation = self.adaptive_fusion(combined_features)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(features)
        
        # Weight allocations by uncertainty
        if uncertainty > 0.5:  # High uncertainty
            # Use more conservative allocation
            final_allocation = 0.5 * vol_allocation + 0.5 * sent_allocation
        else:
            # Use adaptive allocation
            final_allocation = adaptive_allocation
        
        return {
            'allocation': final_allocation,
            'volatility_regime': vol_regime,
            'sentiment_regime': sent_regime,
            'uncertainty': uncertainty,
            'volatility_probs': vol_probs,
            'sentiment_probs': sent_probs
        } 