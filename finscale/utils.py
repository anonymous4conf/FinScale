"""
Utility functions for FinScale framework.
Includes entropy estimation, mutual information computation, and other helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
import logging

logger = logging.getLogger(__name__)

def estimate_entropy(features: torch.Tensor, method: str = 'histogram') -> torch.Tensor:
    """
    Estimate entropy of features using various methods.
    
    Args:
        features: Input features [batch_size, feature_dim]
        method: Entropy estimation method ('histogram', 'kde', 'neural')
        
    Returns:
        Entropy estimates [batch_size]
    """
    if method == 'histogram':
        return _histogram_entropy(features)
    elif method == 'kde':
        return _kde_entropy(features)
    elif method == 'neural':
        return _neural_entropy(features)
    else:
        raise ValueError(f"Unknown entropy method: {method}")


def _histogram_entropy(features: torch.Tensor) -> torch.Tensor:
    """Estimate entropy using histogram method."""
    batch_size = features.shape[0]
    
    # Discretize features into bins
    num_bins = 20
    min_val = torch.min(features)
    max_val = torch.max(features)
    
    # Create bins
    bins = torch.linspace(min_val, max_val, num_bins + 1)
    
    # Count samples in each bin
    histograms = []
    for i in range(batch_size):
        hist, _ = torch.histogram(features[i], bins=bins)
        hist = hist.float() / torch.sum(hist)  # Normalize
        histograms.append(hist)
    
    histograms = torch.stack(histograms)
    
    # Compute entropy
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    histograms = histograms + epsilon
    
    entropy = -torch.sum(histograms * torch.log(histograms), dim=1)
    
    return entropy


def _kde_entropy(features: torch.Tensor) -> torch.Tensor:
    """Estimate entropy using kernel density estimation."""
    # This is a simplified KDE implementation
    # In practice, you'd use a more sophisticated KDE library
    
    batch_size = features.shape[0]
    feature_dim = features.shape[1]
    
    # Use nearest neighbor distances as proxy for density
    distances = torch.cdist(features, features)
    
    # Get k-th nearest neighbor distance for each point
    k = min(5, batch_size - 1)
    knn_distances, _ = torch.topk(distances, k=k+1, dim=1, largest=False)
    knn_distances = knn_distances[:, 1:]  # Exclude self
    
    # Estimate density using k-th nearest neighbor
    density_estimate = 1.0 / (knn_distances[:, -1] + 1e-10)
    
    # Estimate entropy as negative log of density
    entropy_estimate = -torch.log(density_estimate + 1e-10)
    
    return entropy_estimate


def _neural_entropy(features: torch.Tensor) -> torch.Tensor:
    """Estimate entropy using neural network."""
    # This is a placeholder for neural entropy estimation
    # In practice, you'd implement a more sophisticated neural entropy estimator
    
    batch_size = features.shape[0]
    
    # Simple entropy estimator based on feature variance
    feature_variance = torch.var(features, dim=1)
    entropy_estimate = torch.log(feature_variance + 1e-10)
    
    return entropy_estimate


def compute_mutual_information(x: torch.Tensor, y: torch.Tensor, method: str = 'neural') -> torch.Tensor:
    """
    Compute mutual information between two sets of features.
    
    Args:
        x: First set of features [batch_size, feature_dim]
        y: Second set of features [batch_size, feature_dim]
        method: MI estimation method ('neural', 'kde', 'histogram')
        
    Returns:
        Mutual information estimate
    """
    if method == 'neural':
        return _neural_mutual_information(x, y)
    elif method == 'kde':
        return _kde_mutual_information(x, y)
    elif method == 'histogram':
        return _histogram_mutual_information(x, y)
    else:
        raise ValueError(f"Unknown MI method: {method}")


def _neural_mutual_information(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute mutual information using neural network."""
    # This is a simplified neural MI estimator
    # In practice, you'd use a more sophisticated implementation
    
    # Concatenate features
    combined = torch.cat([x, y], dim=1)
    
    # Simple MI estimator based on correlation
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    
    correlation = torch.sum(x_norm * y_norm, dim=1)
    mi_estimate = torch.abs(correlation)
    
    return torch.mean(mi_estimate)


def _kde_mutual_information(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute mutual information using KDE."""
    # Simplified KDE-based MI estimation
    batch_size = x.shape[0]
    
    # Compute distances
    x_distances = torch.cdist(x, x)
    y_distances = torch.cdist(y, y)
    
    # Use nearest neighbor distances
    k = min(5, batch_size - 1)
    
    # Get k-th nearest neighbor for x and y
    x_knn_dist, _ = torch.topk(x_distances, k=k+1, dim=1, largest=False)
    y_knn_dist, _ = torch.topk(y_distances, k=k+1, dim=1, largest=False)
    
    x_knn_dist = x_knn_dist[:, 1:]  # Exclude self
    y_knn_dist = y_knn_dist[:, 1:]  # Exclude self
    
    # Estimate MI using k-nearest neighbor method
    # This is a simplified version
    mi_estimate = torch.mean(torch.log(x_knn_dist[:, -1] / y_knn_dist[:, -1] + 1e-10))
    
    return mi_estimate


def _histogram_mutual_information(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute mutual information using histogram method."""
    # Convert to numpy for sklearn
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    # Discretize for histogram method
    num_bins = 20
    
    # Discretize x and y
    x_binned = np.digitize(x_np, bins=np.linspace(x_np.min(), x_np.max(), num_bins))
    y_binned = np.digitize(y_np, bins=np.linspace(y_np.min(), y_np.max(), num_bins))
    
    # Compute MI for each sample
    mi_scores = []
    for i in range(x_np.shape[0]):
        mi_score = mutual_info_score(x_binned[i], y_binned[i])
        mi_scores.append(mi_score)
    
    return torch.tensor(np.mean(mi_scores), dtype=torch.float32)


def compute_total_correlation(features: torch.Tensor) -> torch.Tensor:
    """
    Compute total correlation (multi-information) among features.
    
    Args:
        features: Input features [batch_size, num_features]
        
    Returns:
        Total correlation estimate
    """
    batch_size, num_features = features.shape
    
    # Compute individual entropies
    individual_entropies = []
    for i in range(num_features):
        feature_entropy = estimate_entropy(features[:, i:i+1])
        individual_entropies.append(feature_entropy)
    
    individual_entropies = torch.stack(individual_entropies, dim=1)
    
    # Compute joint entropy
    joint_entropy = estimate_entropy(features)
    
    # Total correlation = sum of individual entropies - joint entropy
    total_correlation = torch.sum(individual_entropies, dim=1) - joint_entropy
    
    return total_correlation


def compute_information_gain(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute information gain for feature selection.
    
    Args:
        features: Input features [batch_size, feature_dim]
        labels: Target labels [batch_size]
        
    Returns:
        Information gain for each feature
    """
    batch_size, feature_dim = features.shape
    
    # Compute entropy of labels
    label_entropy = estimate_entropy(labels.unsqueeze(1))
    
    # Compute conditional entropy for each feature
    conditional_entropies = []
    for i in range(feature_dim):
        feature_values = features[:, i:i+1]
        conditional_entropy = estimate_entropy(feature_values)
        conditional_entropies.append(conditional_entropy)
    
    conditional_entropies = torch.stack(conditional_entropies, dim=1)
    
    # Information gain = label entropy - conditional entropy
    information_gain = label_entropy.unsqueeze(1) - conditional_entropies
    
    return information_gain


def compute_optimal_allocation(info_scores: torch.Tensor, 
                             computational_costs: torch.Tensor,
                             budget: int) -> torch.Tensor:
    """
    Compute optimal allocation using information-theoretic principles.
    
    Implements the optimal allocation theorem:
    α_j* = √(I(Y; X^(j)) / c_j) / Σ_k √(I(Y; X^(k)) / c_k)
    
    Args:
        info_scores: Information scores for each modality [batch_size, num_modalities]
        computational_costs: Computational costs for each modality [num_modalities]
        budget: Total computational budget
        
    Returns:
        Optimal allocation [batch_size, num_modalities]
    """
    batch_size, num_modalities = info_scores.shape
    
    # Apply square root and normalize by computational cost
    sqrt_info = torch.sqrt(info_scores + 1e-10)
    cost_normalized = sqrt_info / computational_costs.unsqueeze(0)
    
    # Compute denominator (sum across modalities)
    denominator = torch.sum(cost_normalized, dim=1, keepdim=True)
    
    # Compute allocation
    allocation = cost_normalized / (denominator + 1e-10)
    
    # Scale to budget
    allocation = allocation * budget
    
    return allocation


def compute_regime_penalty(allocation: torch.Tensor, 
                          regime_probs: torch.Tensor,
                          regime_allocations: torch.Tensor) -> torch.Tensor:
    """
    Compute penalty for regime-adaptive allocation.
    
    Args:
        allocation: Current allocation [batch_size, num_modalities]
        regime_probs: Regime probabilities [batch_size, num_regimes]
        regime_allocations: Optimal allocations for each regime [num_regimes, num_modalities]
        
    Returns:
        Regime penalty
    """
    batch_size, num_modalities = allocation.shape
    num_regimes = regime_probs.shape[1]
    
    # Compute expected allocation under current regime
    expected_allocation = torch.matmul(regime_probs, regime_allocations)
    
    # Compute penalty as squared difference
    penalty = torch.mean((allocation - expected_allocation) ** 2)
    
    return penalty


def compute_efficiency_score(accuracy: float, 
                           computational_cost: float,
                           budget: float) -> float:
    """
    Compute efficiency score based on accuracy and computational cost.
    
    Args:
        accuracy: Prediction accuracy
        computational_cost: Computational cost used
        budget: Total budget available
        
    Returns:
        Efficiency score
    """
    # Efficiency = accuracy / (normalized cost)
    normalized_cost = computational_cost / budget
    efficiency = accuracy / (normalized_cost + 1e-10)
    
    return efficiency


def compute_sharpe_ratio(returns: torch.Tensor, risk_free_rate: float = 0.02) -> float:
    """
    Compute Sharpe ratio for financial returns.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    sharpe_ratio = torch.mean(excess_returns) / (torch.std(excess_returns) + 1e-10)
    
    return sharpe_ratio.item()


def compute_calibration_error(predictions: torch.Tensor, 
                            targets: torch.Tensor,
                            num_bins: int = 10) -> float:
    """
    Compute calibration error for probabilistic predictions.
    
    Args:
        predictions: Predicted probabilities [batch_size, num_classes]
        targets: True labels [batch_size]
        num_bins: Number of bins for calibration
        
    Returns:
        Calibration error
    """
    batch_size, num_classes = predictions.shape
    
    # Get predicted probabilities for true class
    pred_probs = predictions[torch.arange(batch_size), targets]
    
    # Create bins
    bin_edges = torch.linspace(0, 1, num_bins + 1)
    bin_indices = torch.bucketize(pred_probs, bin_edges) - 1
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    
    # Compute accuracy and confidence for each bin
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        if torch.sum(mask) > 0:
            bin_acc = torch.mean((pred_probs[mask] > 0.5).float())
            bin_conf = torch.mean(pred_probs[mask])
            bin_accuracies.append(bin_acc.item())
            bin_confidences.append(bin_conf.item())
    
    # Compute calibration error
    if len(bin_accuracies) > 0:
        calibration_error = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
    else:
        calibration_error = 0.0
    
    return calibration_error


def compute_attention_weights(query: torch.Tensor, 
                            key: torch.Tensor,
                            value: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention weights for interpretability.
    
    Args:
        query: Query tensor [batch_size, seq_len, hidden_dim]
        key: Key tensor [batch_size, seq_len, hidden_dim]
        value: Value tensor [batch_size, seq_len, hidden_dim]
        mask: Attention mask [batch_size, seq_len]
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1)))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output, attention_weights


def visualize_allocation(allocation: torch.Tensor, 
                        modality_names: List[str],
                        save_path: Optional[str] = None) -> None:
    """
    Visualize computational allocation across modalities.
    
    Args:
        allocation: Allocation tensor [batch_size, num_modalities]
        modality_names: Names of modalities
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Compute average allocation
    avg_allocation = torch.mean(allocation, dim=0).cpu().numpy()
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modality_names, avg_allocation)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_allocation):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title('Computational Allocation Across Modalities')
    plt.ylabel('Allocation Fraction')
    plt.ylim(0, max(avg_allocation) * 1.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def log_performance_metrics(metrics: Dict[str, float], 
                          step: int,
                          logger: logging.Logger) -> None:
    """
    Log performance metrics.
    
    Args:
        metrics: Dictionary of metrics
        step: Current step
        logger: Logger instance
    """
    log_str = f"Step {step}: "
    for metric_name, metric_value in metrics.items():
        log_str += f"{metric_name}={metric_value:.4f} "
    
    logger.info(log_str) 