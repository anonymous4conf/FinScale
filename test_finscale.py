"""
Test script for FinScale framework.
"""

import torch
import numpy as np
import logging
from unittest.mock import Mock, patch

from finscale import FinScale, FinScaleConfig
from finscale.utils import estimate_entropy, compute_mutual_information

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_initialization():
    """Test FinScale model initialization."""
    logger.info("Testing model initialization...")
    
    config = FinScaleConfig(
        budget=8000,
        modalities=['news', 'tables', 'charts', 'prices'],
        hierarchical_layers=4,
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        max_seq_length=512,
        temperature=1.0
    )
    
    model = FinScale(config)
    
    # Check that model components are initialized
    assert hasattr(model, 'modality_encoders')
    assert hasattr(model, 'entropy_allocator')
    assert hasattr(model, 'hierarchical_processor')
    assert hasattr(model, 'transfer_module')
    assert hasattr(model, 'regime_detector')
    assert hasattr(model, 'task_heads')
    
    logger.info("✓ Model initialization successful")

def test_synthetic_data_creation():
    """Test synthetic data creation."""
    logger.info("Testing synthetic data creation...")
    
    # Create synthetic data
    batch_size = 4
    
    # News data
    news_data = [
        f"Company {i} reports strong earnings in Q{np.random.randint(1, 5)}" 
        for i in range(batch_size)
    ]
    
    # Table data
    table_data = torch.randn(batch_size, 8, 8)
    
    # Chart data
    chart_data = torch.randn(batch_size, 3, 224, 224)
    
    # Price data
    price_data = torch.randn(batch_size, 100, 10)
    
    # Labels
    return_prediction_labels = torch.randint(0, 2, (batch_size,))
    volatility_regime_labels = torch.randint(0, 3, (batch_size,))
    earnings_surprise_labels = torch.randint(0, 3, (batch_size,))
    
    data = {
        'news': news_data,
        'tables': table_data,
        'charts': chart_data,
        'prices': price_data,
        'return_prediction_labels': return_prediction_labels,
        'volatility_regime_labels': volatility_regime_labels,
        'earnings_surprise_labels': earnings_surprise_labels
    }
    
    # Check data shapes
    assert len(data['news']) == batch_size
    assert data['tables'].shape == (batch_size, 8, 8)
    assert data['charts'].shape == (batch_size, 3, 224, 224)
    assert data['prices'].shape == (batch_size, 100, 10)
    assert data['return_prediction_labels'].shape == (batch_size,)
    
    logger.info("✓ Synthetic data creation successful")

def test_entropy_estimation():
    """Test entropy estimation functions."""
    logger.info("Testing entropy estimation...")
    
    # Create test features
    features = torch.randn(10, 768)
    
    # Test histogram entropy
    entropy_hist = estimate_entropy(features, method='histogram')
    assert entropy_hist.shape == (10,)
    assert torch.all(entropy_hist >= 0)
    
    # Test neural entropy
    entropy_neural = estimate_entropy(features, method='neural')
    assert entropy_neural.shape == (10,)
    
    logger.info("✓ Entropy estimation successful")

def test_mutual_information():
    """Test mutual information computation."""
    logger.info("Testing mutual information computation...")
    
    # Create test features
    x = torch.randn(10, 768)
    y = torch.randn(10, 768)
    
    # Test neural mutual information
    mi_neural = compute_mutual_information(x, y, method='neural')
    assert isinstance(mi_neural, torch.Tensor)
    assert mi_neural >= 0
    
    logger.info("✓ Mutual information computation successful")

@patch('finscale.data.NewsEncoder')
@patch('finscale.data.TableEncoder')
@patch('finscale.data.ChartEncoder')
@patch('finscale.data.PriceEncoder')
def test_model_forward_pass(mock_news_encoder, mock_table_encoder, mock_chart_encoder, mock_price_encoder):
    """Test model forward pass with mocked encoders."""
    logger.info("Testing model forward pass...")
    
    # Mock encoder outputs
    mock_news_encoder.return_value.return_value = torch.randn(4, 768)
    mock_table_encoder.return_value.return_value = torch.randn(4, 768)
    mock_chart_encoder.return_value.return_value = torch.randn(4, 768)
    mock_price_encoder.return_value.return_value = torch.randn(4, 768)
    
    # Create model
    config = FinScaleConfig(
        budget=8000,
        modalities=['news', 'tables', 'charts', 'prices'],
        hierarchical_layers=2,  # Reduced for testing
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        max_seq_length=512,
        temperature=1.0
    )
    
    model = FinScale(config)
    
    # Create test data
    batch_size = 4
    
    data = {
        'news': [f"Company {i} reports earnings" for i in range(batch_size)],
        'tables': torch.randn(batch_size, 8, 8),
        'charts': torch.randn(batch_size, 3, 224, 224),
        'prices': torch.randn(batch_size, 100, 10),
        'return_prediction_labels': torch.randint(0, 2, (batch_size,)),
        'volatility_regime_labels': torch.randint(0, 3, (batch_size,)),
        'earnings_surprise_labels': torch.randint(0, 3, (batch_size,))
    }
    
    # Move data to device
    data = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
            for k, v in data.items()}
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model.forward(data, task='return_prediction')
    
    # Check outputs
    assert 'predictions' in outputs
    assert 'logits' in outputs
    assert 'allocation' in outputs
    assert 'market_regime' in outputs
    assert 'entropy_scores' in outputs
    
    assert outputs['predictions'].shape == (batch_size, 2)  # Binary classification
    assert outputs['allocation'].shape == (batch_size, 4)  # 4 modalities
    
    logger.info("✓ Model forward pass successful")

def test_allocation_analysis():
    """Test allocation analysis functionality."""
    logger.info("Testing allocation analysis...")
    
    config = FinScaleConfig(
        budget=8000,
        modalities=['news', 'tables', 'charts', 'prices'],
        hierarchical_layers=2,
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        max_seq_length=512,
        temperature=1.0
    )
    
    model = FinScale(config)
    
    # Create test data
    batch_size = 4
    
    data = {
        'news': [f"Company {i} reports earnings" for i in range(batch_size)],
        'tables': torch.randn(batch_size, 8, 8),
        'charts': torch.randn(batch_size, 3, 224, 224),
        'prices': torch.randn(batch_size, 100, 10),
        'return_prediction_labels': torch.randint(0, 2, (batch_size,)),
        'volatility_regime_labels': torch.randint(0, 3, (batch_size,)),
        'earnings_surprise_labels': torch.randint(0, 3, (batch_size,))
    }
    
    # Move data to device
    data = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
            for k, v in data.items()}
    
    # Get allocation analysis
    model.eval()
    with torch.no_grad():
        analysis = model.get_allocation_analysis(data)
    
    # Check analysis outputs
    assert 'allocation' in analysis
    assert 'entropy_scores' in analysis
    assert 'mutual_information' in analysis
    assert 'market_regime' in analysis
    
    logger.info("✓ Allocation analysis successful")

def test_regime_adaptation():
    """Test regime adaptation capabilities."""
    logger.info("Testing regime adaptation...")
    
    config = FinScaleConfig(
        budget=8000,
        modalities=['news', 'tables', 'charts', 'prices'],
        hierarchical_layers=2,
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        max_seq_length=512,
        temperature=1.0
    )
    
    model = FinScale(config)
    
    # Create test data
    batch_size = 4
    
    data = {
        'news': [f"Company {i} reports earnings" for i in range(batch_size)],
        'tables': torch.randn(batch_size, 8, 8),
        'charts': torch.randn(batch_size, 3, 224, 224),
        'prices': torch.randn(batch_size, 100, 10),
        'return_prediction_labels': torch.randint(0, 2, (batch_size,)),
        'volatility_regime_labels': torch.randint(0, 3, (batch_size,)),
        'earnings_surprise_labels': torch.randint(0, 3, (batch_size,))
    }
    
    # Move data to device
    data = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
            for k, v in data.items()}
    
    # Test different regimes
    regimes = ['low_volatility', 'medium_volatility', 'high_volatility']
    
    model.eval()
    with torch.no_grad():
        for regime in regimes:
            predictions = model.predict(
                data=data,
                task='return_prediction',
                market_regime=regime
            )
            
            assert predictions['market_regime'] == regime
            assert predictions['allocation'].shape == (batch_size, 4)
    
    logger.info("✓ Regime adaptation successful")

def test_cross_domain_transfer():
    """Test cross-domain transfer capabilities."""
    logger.info("Testing cross-domain transfer...")
    
    config = FinScaleConfig(
        budget=8000,
        modalities=['news', 'tables', 'charts', 'prices'],
        hierarchical_layers=2,
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        max_seq_length=512,
        temperature=1.0
    )
    
    model = FinScale(config)
    
    # Set domains
    model.set_domains(source_domain='US_market', target_domain='Chinese_market')
    
    # Create test data
    batch_size = 4
    
    data = {
        'news': [f"Company {i} reports earnings" for i in range(batch_size)],
        'tables': torch.randn(batch_size, 8, 8),
        'charts': torch.randn(batch_size, 3, 224, 224),
        'prices': torch.randn(batch_size, 100, 10),
        'return_prediction_labels': torch.randint(0, 2, (batch_size,)),
        'volatility_regime_labels': torch.randint(0, 3, (batch_size,)),
        'earnings_surprise_labels': torch.randint(0, 3, (batch_size,))
    }
    
    # Move data to device
    data = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
            for k, v in data.items()}
    
    # Test transfer
    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            data=data,
            task='return_prediction'
        )
        
        assert 'predictions' in predictions
        assert 'allocation' in predictions
        assert predictions['predictions'].shape == (batch_size, 2)
    
    logger.info("✓ Cross-domain transfer successful")

def run_all_tests():
    """Run all tests."""
    logger.info("Starting FinScale tests...")
    
    try:
        test_model_initialization()
        test_synthetic_data_creation()
        test_entropy_estimation()
        test_mutual_information()
        test_model_forward_pass()
        test_allocation_analysis()
        test_regime_adaptation()
        test_cross_domain_transfer()
        
        logger.info("✓ All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests() 