"""
Basic usage example for FinScale framework.
Demonstrates training and prediction with adaptive multi-modal allocation.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm

from finscale import FinScale, FinScaleConfig
from finscale.data import FinMultiTimeDataset, create_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_data(num_samples: int = 1000):
    """Create synthetic data for demonstration."""
    # Generate synthetic multi-modal data
    batch_size = 32
    
    # News data (text)
    news_data = [
        f"Company {i} reports strong earnings in Q{np.random.randint(1, 5)}" 
        for i in range(num_samples)
    ]
    
    # Table data (financial ratios)
    table_data = torch.randn(num_samples, 8, 8)  # 8 quarters, 8 features
    
    # Chart data (images)
    chart_data = torch.randn(num_samples, 3, 224, 224)  # RGB images
    
    # Price data (time series)
    price_data = torch.randn(num_samples, 100, 10)  # 100 time steps, 10 features
    
    # Labels
    return_prediction_labels = torch.randint(0, 2, (num_samples,))
    volatility_regime_labels = torch.randint(0, 3, (num_samples,))
    earnings_surprise_labels = torch.randint(0, 3, (num_samples,))
    
    return {
        'news': news_data,
        'tables': table_data,
        'charts': chart_data,
        'prices': price_data,
        'return_prediction_labels': return_prediction_labels,
        'volatility_regime_labels': volatility_regime_labels,
        'earnings_surprise_labels': earnings_surprise_labels
    }

def train_finscale():
    """Train FinScale model."""
    logger.info("Initializing FinScale model...")
    
    # Create configuration
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
    
    # Initialize model
    model = FinScale(config)
    
    # Create synthetic data
    logger.info("Creating synthetic data...")
    train_data = create_synthetic_data(1000)
    val_data = create_synthetic_data(200)
    
    # Create data loaders
    train_loader = create_dataloader_from_dict(train_data, batch_size=32)
    val_loader = create_dataloader_from_dict(val_data, batch_size=32)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    logger.info("Starting training...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move data to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model.forward(batch, task='return_prediction')
            
            # Compute loss
            targets = batch['return_prediction_labels']
            loss = torch.nn.functional.cross_entropy(outputs['logits'], targets)
            
            # Add allocation penalty
            allocation_penalty = torch.mean((outputs['allocation'] - 0.25) ** 2)
            total_loss_batch = loss + 0.1 * allocation_penalty
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            # Compute accuracy
            predictions = torch.argmax(outputs['predictions'], dim=-1)
            accuracy = (predictions == targets).float().mean()
            
            total_loss += total_loss_batch.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'acc': f"{accuracy.item():.4f}",
                'alloc_penalty': f"{allocation_penalty.item():.4f}"
            })
        
        # Validation
        model.eval()
        val_accuracy = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model.forward(batch, task='return_prediction')
                predictions = torch.argmax(outputs['predictions'], dim=-1)
                targets = batch['return_prediction_labels']
                
                accuracy = (predictions == targets).float().mean()
                val_accuracy += accuracy.item()
                val_batches += 1
        
        # Log metrics
        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_accuracy / num_batches
        avg_val_acc = val_accuracy / val_batches
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                   f"Train Acc={avg_train_acc:.4f}, Val Acc={avg_val_acc:.4f}")
        
        scheduler.step()
    
    logger.info("Training completed!")
    return model

def predict_with_finscale(model):
    """Demonstrate prediction with FinScale."""
    logger.info("Running prediction example...")
    
    # Create test data
    test_data = create_synthetic_data(50)
    test_loader = create_dataloader_from_dict(test_data, batch_size=16)
    
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Make predictions
            predictions = model.predict(
                data=batch,
                task='return_prediction',
                market_regime='high_volatility'
            )
            
            # Print results
            logger.info(f"Predictions shape: {predictions['predictions'].shape}")
            logger.info(f"Allocation: {predictions['allocation'].mean(axis=0)}")
            logger.info(f"Market regime: {predictions['market_regime']}")
            
            # Analyze allocation patterns
            allocation_analysis = model.get_allocation_analysis(batch)
            logger.info(f"Allocation analysis: {allocation_analysis}")
            
            break

def create_dataloader_from_dict(data_dict, batch_size=32):
    """Create DataLoader from dictionary data."""
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, data_dict):
            self.data = data_dict
            self.length = len(data_dict['return_prediction_labels'])
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            return {
                'news': self.data['news'][idx],
                'tables': self.data['tables'][idx],
                'charts': self.data['charts'][idx],
                'prices': self.data['prices'][idx],
                'return_prediction_labels': self.data['return_prediction_labels'][idx],
                'volatility_regime_labels': self.data['volatility_regime_labels'][idx],
                'earnings_surprise_labels': self.data['earnings_surprise_labels'][idx]
            }
    
    dataset = SyntheticDataset(data_dict)
    
    def collate_fn(batch):
        collated = {}
        
        # Group by modality
        for modality in ['news', 'tables', 'charts', 'prices']:
            modality_data = [item[modality] for item in batch]
            if modality == 'news':
                collated[modality] = modality_data
            else:
                collated[modality] = torch.stack(modality_data)
        
        # Add labels
        for task in ['return_prediction', 'volatility_regime', 'earnings_surprise']:
            label_key = f'{task}_labels'
            labels = [item[label_key] for item in batch]
            collated[label_key] = torch.stack(labels)
        
        return collated
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def demonstrate_regime_adaptation(model):
    """Demonstrate regime adaptation capabilities."""
    logger.info("Demonstrating regime adaptation...")
    
    # Create test data
    test_data = create_synthetic_data(10)
    test_loader = create_dataloader_from_dict(test_data, batch_size=10)
    
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Test different market regimes
            regimes = ['low_volatility', 'medium_volatility', 'high_volatility']
            
            for regime in regimes:
                predictions = model.predict(
                    data=batch,
                    task='return_prediction',
                    market_regime=regime
                )
                
                logger.info(f"Regime: {regime}")
                logger.info(f"  Allocation: {predictions['allocation'].mean(axis=0)}")
                logger.info(f"  Average entropy scores: {np.mean(list(predictions['entropy_scores'].values())):.4f}")
            
            break

def demonstrate_cross_domain_transfer(model):
    """Demonstrate cross-domain transfer capabilities."""
    logger.info("Demonstrating cross-domain transfer...")
    
    # Set source and target domains
    model.set_domains(source_domain='US_market', target_domain='Chinese_market')
    
    # Create test data
    test_data = create_synthetic_data(10)
    test_loader = create_dataloader_from_dict(test_data, batch_size=10)
    
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Make predictions with transfer
            predictions = model.predict(
                data=batch,
                task='return_prediction'
            )
            
            logger.info("Cross-domain transfer results:")
            logger.info(f"  Predictions shape: {predictions['predictions'].shape}")
            logger.info(f"  Allocation: {predictions['allocation'].mean(axis=0)}")
            
            break

if __name__ == "__main__":
    logger.info("Starting FinScale basic usage example...")
    
    # Train model
    model = train_finscale()
    
    # Demonstrate prediction
    predict_with_finscale(model)
    
    # Demonstrate regime adaptation
    demonstrate_regime_adaptation(model)
    
    # Demonstrate cross-domain transfer
    demonstrate_cross_domain_transfer(model)
    
    logger.info("Basic usage example completed!") 