#!/usr/bin/env python3
"""
Debug script to check 6-month data split
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
from models.trainer import ModelTrainer

def debug_6months_split():
    """Debug the 6-month data split"""
    print("=== 6-Month Data Split Debug ===\n")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Define training period (6 months)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    # Prepare data
    features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
    
    print(f"Total dataset: {len(dataset)} samples")
    print(f"Flare positive rate: {dataset['flare6h_label'].mean():.3f}")
    print(f"Kp positive rate: {dataset['kp12h_label'].mean():.3f}")
    
    # Sort by timestamp
    dataset = dataset.sort_values('timestamp')
    
    # Find positive samples
    positive_samples = dataset[dataset['flare6h_label'] == 1]
    negative_samples = dataset[dataset['flare6h_label'] == 0]
    
    print(f"\nPositive samples: {len(positive_samples)}")
    print(f"Negative samples: {len(negative_samples)}")
    
    # Check the split logic
    if len(positive_samples) > 0:
        # Split positive samples
        pos_split_idx = int(len(positive_samples) * 0.8)
        train_pos = positive_samples.iloc[:pos_split_idx]
        test_pos = positive_samples.iloc[pos_split_idx:]
        
        # Split negative samples
        neg_split_idx = int(len(negative_samples) * 0.8)
        train_neg = negative_samples.iloc[:neg_split_idx]
        test_neg = negative_samples.iloc[neg_split_idx:]
        
        # Combine
        train_data = pd.concat([train_pos, train_neg]).sort_values('timestamp')
        test_data = pd.concat([test_pos, test_neg]).sort_values('timestamp')
    else:
        # Fallback to time-based split
        split_idx = int(len(dataset) * 0.8)
        train_data = dataset.iloc[:split_idx]
        test_data = dataset.iloc[split_idx:]
    
    print(f"\n=== Split Results ===")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Train flare positive rate: {train_data['flare6h_label'].mean():.3f}")
    print(f"Test flare positive rate: {test_data['flare6h_label'].mean():.3f}")
    
    # Check if we have positive samples in training
    train_positives = (train_data['flare6h_label'] == 1).sum()
    test_positives = (test_data['flare6h_label'] == 1).sum()
    
    print(f"\n=== Positive Sample Distribution ===")
    print(f"Positive samples in train: {train_positives}")
    print(f"Positive samples in test: {test_positives}")
    print(f"Total positive samples: {train_positives + test_positives}")
    
    if train_positives == 0:
        print("❌ PROBLEM: No positive samples in training set!")
        print("This will cause the model to predict all zeros.")
    else:
        print("✅ Good: Positive samples in both train and test sets")
    
    # Show some sample timestamps
    print(f"\n=== Sample Timestamps ===")
    print(f"Train start: {train_data['timestamp'].min()}")
    print(f"Train end: {train_data['timestamp'].max()}")
    print(f"Test start: {test_data['timestamp'].min()}")
    print(f"Test end: {test_data['timestamp'].max()}")
    
    return train_data, test_data

if __name__ == "__main__":
    debug_6months_split()
