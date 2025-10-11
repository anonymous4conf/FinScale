"""
Data processing module for FinScale framework.
Handles multi-modal financial data encoding and dataset management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import logging

logger = logging.getLogger(__name__)

class ModalityEncoder(nn.Module):
    """Encoder for different data modalities in financial markets."""
    
    def __init__(self, modality: str, hidden_dim: int, max_seq_length: int):
        super().__init__()
        self.modality = modality
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        if modality == 'news':
            self.encoder = NewsEncoder(hidden_dim, max_seq_length)
        elif modality == 'tables':
            self.encoder = TableEncoder(hidden_dim, max_seq_length)
        elif modality == 'charts':
            self.encoder = ChartEncoder(hidden_dim)
        elif modality == 'prices':
            self.encoder = PriceEncoder(hidden_dim, max_seq_length)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Encode modality data."""
        return self.encoder(data)


class NewsEncoder(nn.Module):
    """Encoder for financial news text."""
    
    def __init__(self, hidden_dim: int, max_seq_length: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # Use FinBERT for financial text
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.bert = AutoModel.from_pretrained('ProsusAI/finbert')
        
        # Project to target hidden dimension
        self.projection = nn.Linear(768, hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode news texts."""
        # Tokenize and encode
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Project to target dimension
        features = self.projection(embeddings)
        features = self.dropout(features)
        
        return features


class TableEncoder(nn.Module):
    """Encoder for financial tables (balance sheets, income statements)."""
    
    def __init__(self, hidden_dim: int, max_seq_length: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # Table structure encoder
        self.table_encoder = nn.Sequential(
            nn.Linear(50, 256),  # Assume 50 financial ratios/features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim)
        )
        
        # Temporal attention for quarterly data
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
    
    def forward(self, table_data: torch.Tensor) -> torch.Tensor:
        """Encode financial table data."""
        # table_data shape: [batch_size, num_quarters, num_features]
        batch_size, num_quarters, num_features = table_data.shape
        
        # Encode each quarter's features
        encoded_quarters = []
        for i in range(num_quarters):
            quarter_features = self.table_encoder(table_data[:, i, :])
            encoded_quarters.append(quarter_features)
        
        # Stack quarters
        encoded_quarters = torch.stack(encoded_quarters, dim=1)  # [batch, quarters, hidden]
        
        # Apply temporal attention
        encoded_quarters = encoded_quarters.transpose(0, 1)  # [quarters, batch, hidden]
        attended, _ = self.temporal_attention(
            encoded_quarters, encoded_quarters, encoded_quarters
        )
        attended = attended.transpose(0, 1)  # [batch, quarters, hidden]
        
        # Aggregate across quarters
        features = torch.mean(attended, dim=1)  # [batch, hidden]
        
        return features


class ChartEncoder(nn.Module):
    """Encoder for technical chart images."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Use pre-trained ResNet for chart images
        self.resnet = AutoModelForImageClassification.from_pretrained('microsoft/resnet-50')
        self.image_processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
        
        # Project to target dimension
        self.projection = nn.Linear(1000, hidden_dim)  # ResNet output dim
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode chart images."""
        # Preprocess images
        inputs = self.image_processor(images, return_tensors='pt')
        
        # Get ResNet features
        with torch.no_grad():
            outputs = self.resnet(**inputs)
            features = outputs.logits
        
        # Project to target dimension
        features = self.projection(features)
        features = self.dropout(features)
        
        return features


class PriceEncoder(nn.Module):
    """Encoder for price time series data."""
    
    def __init__(self, hidden_dim: int, max_seq_length: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=10,  # OHLCV + 5 technical indicators
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Final projection
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, price_data: torch.Tensor) -> torch.Tensor:
        """Encode price time series."""
        # price_data shape: [batch_size, seq_length, features]
        batch_size, seq_length, features = price_data.shape
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(price_data)
        # lstm_out: [batch, seq, hidden_dim]
        
        # Apply attention
        lstm_out = lstm_out.transpose(0, 1)  # [seq, batch, hidden]
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = attended.transpose(0, 1)  # [batch, seq, hidden]
        
        # Global average pooling
        features = torch.mean(attended, dim=1)  # [batch, hidden]
        
        # Final projection
        features = self.projection(features)
        features = self.dropout(features)
        
        return features


class FinMultiTimeDataset(Dataset):
    """Dataset for FinMultiTime multi-modal financial data."""
    
    def __init__(self, 
                 data_path: str,
                 modalities: List[str],
                 split: str = 'train',
                 max_samples: Optional[int] = None):
        """
        Initialize FinMultiTime dataset.
        
        Args:
            data_path: Path to FinMultiTime data
            modalities: List of modalities to load
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
        """
        self.data_path = data_path
        self.modalities = modalities
        self.split = split
        
        # Load data
        self.data = self._load_data()
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load multi-modal data from disk."""
        data = []
        
        # Load metadata
        metadata_path = os.path.join(self.data_path, f'{self.split}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        for item in metadata:
            sample = {'metadata': item}
            
            # Load modality data
            for modality in self.modalities:
                modality_path = os.path.join(
                    self.data_path, 
                    self.split, 
                    modality, 
                    f"{item['stock_id']}_{item['date']}.pkl"
                )
                
                if os.path.exists(modality_path):
                    sample[modality] = pd.read_pickle(modality_path)
            
            data.append(sample)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Process each modality
        processed = {}
        
        for modality in self.modalities:
            if modality in sample:
                processed[modality] = self._process_modality(
                    modality, sample[modality]
                )
        
        # Add labels
        processed['return_prediction_labels'] = torch.tensor(
            sample['metadata']['return_label'], dtype=torch.long
        )
        processed['volatility_regime_labels'] = torch.tensor(
            sample['metadata']['volatility_label'], dtype=torch.long
        )
        processed['earnings_surprise_labels'] = torch.tensor(
            sample['metadata']['earnings_label'], dtype=torch.long
        )
        
        return processed
    
    def _process_modality(self, modality: str, data: Any) -> torch.Tensor:
        """Process modality-specific data."""
        if modality == 'news':
            return self._process_news(data)
        elif modality == 'tables':
            return self._process_tables(data)
        elif modality == 'charts':
            return self._process_charts(data)
        elif modality == 'prices':
            return self._process_prices(data)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def _process_news(self, data: pd.DataFrame) -> torch.Tensor:
        """Process news data."""
        # Extract text and sentiment
        texts = data['text'].tolist()
        sentiments = data['sentiment'].values
        
        # Combine text and sentiment
        processed_texts = []
        for text, sentiment in zip(texts, sentiments):
            processed_text = f"[SENTIMENT: {sentiment}] {text}"
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def _process_tables(self, data: pd.DataFrame) -> torch.Tensor:
        """Process financial table data."""
        # Extract financial ratios and metrics
        features = data[['revenue', 'earnings', 'debt_ratio', 'pe_ratio', 
                       'roe', 'roa', 'current_ratio', 'quick_ratio']].values
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _process_charts(self, data: str) -> List[Image.Image]:
        """Process chart image data."""
        # Load chart images
        chart_path = os.path.join(self.data_path, 'charts', data)
        image = Image.open(chart_path).convert('RGB')
        
        return [image]
    
    def _process_prices(self, data: pd.DataFrame) -> torch.Tensor:
        """Process price time series data."""
        # Extract OHLCV + technical indicators
        features = data[['open', 'high', 'low', 'close', 'volume',
                       'sma_20', 'sma_50', 'rsi', 'macd', 'bollinger_upper']].values
        
        return torch.tensor(features, dtype=torch.float32)


def create_dataloader(dataset: FinMultiTimeDataset, 
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """Create DataLoader for FinMultiTime dataset."""
    
    def collate_fn(batch):
        """Custom collate function for multi-modal data."""
        collated = {}
        
        # Group by modality
        for modality in dataset.modalities:
            modality_data = [item[modality] for item in batch if modality in item]
            if modality_data:
                collated[modality] = modality_data
        
        # Add labels
        for task in ['return_prediction', 'volatility_regime', 'earnings_surprise']:
            label_key = f'{task}_labels'
            labels = [item[label_key] for item in batch]
            collated[label_key] = torch.stack(labels)
        
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    ) 