"""
Cross-market transfer learning example for FinScale.
Demonstrates zero-shot transfer between different financial markets.
"""

import torch
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from finscale import FinScale, FinScaleConfig
from finscale.transfer import CrossDomainTransfer, TransferEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_market_specific_data(market_name: str, num_samples: int = 1000):
    """Create market-specific synthetic data."""
    np.random.seed(hash(market_name) % 2**32)  # Different seed for each market
    
    # Market-specific characteristics
    market_params = {
        'US_market': {
            'volatility': 0.15,
            'correlation': 0.3,
            'trend_strength': 0.4
        },
        'Chinese_market': {
            'volatility': 0.25,
            'correlation': 0.5,
            'trend_strength': 0.6
        },
        'European_market': {
            'volatility': 0.20,
            'correlation': 0.4,
            'trend_strength': 0.5
        },
        'Japanese_market': {
            'volatility': 0.18,
            'correlation': 0.35,
            'trend_strength': 0.45
        }
    }
    
    params = market_params.get(market_name, market_params['US_market'])
    
    # Generate synthetic data with market-specific characteristics
    batch_size = 32
    
    # News data with market-specific language patterns
    news_data = [
        f"{market_name} company {i} reports {'strong' if np.random.random() > 0.5 else 'weak'} earnings" 
        for i in range(num_samples)
    ]
    
    # Table data with market-specific financial ratios
    base_ratios = np.random.normal(0, 1, (num_samples, 8, 8))
    market_adjustment = np.random.normal(0, params['correlation'], (num_samples, 8, 8))
    table_data = torch.tensor(base_ratios + market_adjustment, dtype=torch.float32)
    
    # Chart data with market-specific patterns
    chart_data = torch.randn(num_samples, 3, 224, 224) * params['volatility']
    
    # Price data with market-specific volatility
    price_data = torch.randn(num_samples, 100, 10) * params['volatility']
    
    # Labels with market-specific patterns
    base_prob = 0.5 + params['trend_strength'] * (np.random.random(num_samples) - 0.5)
    return_prediction_labels = torch.tensor((base_prob > 0.5).astype(int))
    
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

def train_source_model(market_name: str):
    """Train FinScale model on source market."""
    logger.info(f"Training source model on {market_name}...")
    
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
    
    # Create source market data
    train_data = create_market_specific_data(market_name, 1000)
    val_data = create_market_specific_data(market_name, 200)
    
    # Create data loaders
    train_loader = create_dataloader_from_dict(train_data, batch_size=32)
    val_loader = create_dataloader_from_dict(val_data, batch_size=32)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0.0
    
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            predictions = torch.argmax(outputs['predictions'], dim=-1)
            accuracy = (predictions == targets).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy.item():.4f}"
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
        
        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_accuracy / num_batches
        avg_val_acc = val_accuracy / val_batches
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                   f"Train Acc={avg_train_acc:.4f}, Val Acc={avg_val_acc:.4f}")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
        
        scheduler.step()
    
    logger.info(f"Source model training completed! Best val acc: {best_val_acc:.4f}")
    return model

def evaluate_zero_shot_transfer(source_model, target_market: str):
    """Evaluate zero-shot transfer to target market."""
    logger.info(f"Evaluating zero-shot transfer to {target_market}...")
    
    # Create target market data
    test_data = create_market_specific_data(target_market, 500)
    test_loader = create_dataloader_from_dict(test_data, batch_size=32)
    
    # Set domains for transfer
    source_market = 'US_market'  # Assume US as source
    source_model.set_domains(source_domain=source_market, target_domain=target_market)
    
    source_model.eval()
    
    total_accuracy = 0.0
    total_predictions = []
    total_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(source_model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Make predictions with transfer
            predictions = source_model.predict(
                data=batch,
                task='return_prediction'
            )
            
            # Convert predictions to labels
            pred_labels = np.argmax(predictions['predictions'], axis=1)
            true_labels = batch['return_prediction_labels'].cpu().numpy()
            
            # Compute accuracy
            accuracy = np.mean(pred_labels == true_labels)
            total_accuracy += accuracy
            num_batches += 1
            
            total_predictions.extend(pred_labels)
            total_targets.extend(true_labels)
    
    avg_accuracy = total_accuracy / num_batches
    
    logger.info(f"Zero-shot transfer accuracy to {target_market}: {avg_accuracy:.4f}")
    
    return avg_accuracy, total_predictions, total_targets

def evaluate_fine_tuned_transfer(source_model, target_market: str, fine_tune_samples: int = 100):
    """Evaluate fine-tuned transfer with limited target data."""
    logger.info(f"Evaluating fine-tuned transfer to {target_market} with {fine_tune_samples} samples...")
    
    # Create fine-tuning data
    fine_tune_data = create_market_specific_data(target_market, fine_tune_samples)
    test_data = create_market_specific_data(target_market, 500)
    
    fine_tune_loader = create_dataloader_from_dict(fine_tune_data, batch_size=16)
    test_loader = create_dataloader_from_dict(test_data, batch_size=32)
    
    # Fine-tune the model
    model = source_model  # Use the same model for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Fine-tuning loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in fine_tune_loader:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model.forward(batch, task='return_prediction')
            
            # Compute loss
            targets = batch['return_prediction_labels']
            loss = torch.nn.functional.cross_entropy(outputs['logits'], targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Fine-tuning epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    # Evaluate fine-tuned model
    model.eval()
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            predictions = model.predict(
                data=batch,
                task='return_prediction'
            )
            
            pred_labels = np.argmax(predictions['predictions'], axis=1)
            true_labels = batch['return_prediction_labels'].cpu().numpy()
            
            accuracy = np.mean(pred_labels == true_labels)
            total_accuracy += accuracy
            num_batches += 1
    
    avg_accuracy = total_accuracy / num_batches
    
    logger.info(f"Fine-tuned transfer accuracy to {target_market}: {avg_accuracy:.4f}")
    
    return avg_accuracy

def compare_transfer_methods():
    """Compare different transfer learning methods."""
    logger.info("Comparing transfer learning methods...")
    
    # Train source model
    source_model = train_source_model('US_market')
    
    # Target markets
    target_markets = ['Chinese_market', 'European_market', 'Japanese_market']
    
    results = {}
    
    for target_market in target_markets:
        logger.info(f"\nEvaluating transfer to {target_market}...")
        
        # Zero-shot transfer
        zero_shot_acc, _, _ = evaluate_zero_shot_transfer(source_model, target_market)
        
        # Fine-tuned transfer with different sample sizes
        fine_tune_10_acc = evaluate_fine_tuned_transfer(source_model, target_market, 10)
        fine_tune_50_acc = evaluate_fine_tuned_transfer(source_model, target_market, 50)
        fine_tune_100_acc = evaluate_fine_tuned_transfer(source_model, target_market, 100)
        
        results[target_market] = {
            'zero_shot': zero_shot_acc,
            'fine_tune_10': fine_tune_10_acc,
            'fine_tune_50': fine_tune_50_acc,
            'fine_tune_100': fine_tune_100_acc
        }
    
    return results

def visualize_transfer_results(results):
    """Visualize transfer learning results."""
    logger.info("Creating transfer learning visualization...")
    
    markets = list(results.keys())
    methods = ['zero_shot', 'fine_tune_10', 'fine_tune_50', 'fine_tune_100']
    
    # Create data for plotting
    data = []
    for market in markets:
        for method in methods:
            data.append({
                'Market': market,
                'Method': method,
                'Accuracy': results[market][method]
            })
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    plt.subplot(2, 2, 1)
    accuracies = [[results[market][method] for method in methods] for market in markets]
    x = np.arange(len(markets))
    width = 0.2
    
    for i, method in enumerate(methods):
        plt.bar(x + i * width, [acc[i] for acc in accuracies], width, label=method)
    
    plt.xlabel('Target Market')
    plt.ylabel('Accuracy')
    plt.title('Transfer Learning Performance')
    plt.xticks(x + width * 1.5, markets)
    plt.legend()
    plt.ylim(0, 1)
    
    # Heatmap
    plt.subplot(2, 2, 2)
    heatmap_data = np.array([[results[market][method] for method in methods] for market in markets])
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', 
                xticklabels=methods, yticklabels=markets, cmap='YlOrRd')
    plt.title('Transfer Learning Accuracy Heatmap')
    
    # Line plot showing improvement with fine-tuning
    plt.subplot(2, 2, 3)
    sample_sizes = [0, 10, 50, 100]  # 0 for zero-shot
    for market in markets:
        accuracies = [results[market]['zero_shot']] + [results[market][f'fine_tune_{s}'] for s in [10, 50, 100]]
        plt.plot(sample_sizes, accuracies, marker='o', label=market)
    
    plt.xlabel('Fine-tuning Samples')
    plt.ylabel('Accuracy')
    plt.title('Transfer Learning Improvement')
    plt.legend()
    plt.grid(True)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    avg_accuracies = []
    for method in methods:
        avg_acc = np.mean([results[market][method] for market in markets])
        avg_accuracies.append(avg_acc)
    
    plt.bar(methods, avg_accuracies)
    plt.xlabel('Transfer Method')
    plt.ylabel('Average Accuracy')
    plt.title('Average Performance Across Markets')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Transfer learning visualization saved as 'transfer_learning_results.png'")

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
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

if __name__ == "__main__":
    logger.info("Starting cross-market transfer learning example...")
    
    # Compare transfer methods
    results = compare_transfer_methods()
    
    # Visualize results
    visualize_transfer_results(results)
    
    # Print summary
    logger.info("\nTransfer Learning Summary:")
    for market, accuracies in results.items():
        logger.info(f"\n{market}:")
        for method, acc in accuracies.items():
            logger.info(f"  {method}: {acc:.4f}")
    
    logger.info("Cross-market transfer learning example completed!") 