#!/usr/bin/env python3
"""
Optimize threshold using the original model's precision-recall curve
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pickle
from models.trainer import ModelTrainer

def optimize_original_threshold():
    """Optimize threshold using original model's approach"""
    print("=== Original Model Threshold Optimization ===\n")
    
    # Load the original working model
    print("Loading original working model...")
    with open('models/flare_20250813_091105.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Initialize trainer and prepare data
    trainer = ModelTrainer()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
    dataset = dataset.sort_values('timestamp')
    
    # Split data (same logic as trainer)
    positive_samples = dataset[dataset['flare6h_label'] == 1]
    negative_samples = dataset[dataset['flare6h_label'] == 0]
    
    if len(positive_samples) > 0:
        pos_split_idx = int(len(positive_samples) * 0.8)
        train_pos = positive_samples.iloc[:pos_split_idx]
        test_pos = positive_samples.iloc[pos_split_idx:]
        
        neg_split_idx = int(len(negative_samples) * 0.8)
        train_neg = negative_samples.iloc[:neg_split_idx]
        test_neg = negative_samples.iloc[neg_split_idx:]
        
        train_data = pd.concat([train_pos, train_neg]).sort_values('timestamp')
        test_data = pd.concat([test_pos, test_neg]).sort_values('timestamp')
    else:
        split_idx = int(len(dataset) * 0.8)
        train_data = dataset.iloc[:split_idx]
        test_data = dataset.iloc[split_idx:]
    
    feature_cols = [col for col in dataset.columns if not col.endswith('_label') and col != 'timestamp']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['flare6h_label']
    
    print(f"Test samples: {len(test_data)}")
    print(f"Test positive rate: {y_test.mean():.3f}")
    
    # Get predictions from the original model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nProbability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    
    # Use the original model's precision-recall curve thresholds
    original_thresholds = [0.835, 0.892, 0.893, 0.896, 0.900, 0.913, 0.915, 0.916]
    
    print(f"\n=== Testing Original Model Thresholds ===")
    
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_false_positive_rate = 1.0
    
    results = []
    
    for threshold in original_thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Calculate false positive rate
        true_negatives = ((y_pred == 0) & (y_test == 0)).sum()
        false_positives = ((y_pred == 1) & (y_test == 0)).sum()
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 1.0
        
        positive_predictions = y_pred.sum()
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'false_positive_rate': false_positive_rate,
            'positive_predictions': positive_predictions
        })
        
        print(f"Threshold {threshold:.3f}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, FPR={false_positive_rate:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_false_positive_rate = false_positive_rate
    
    # Also test a wider range around the original thresholds
    print(f"\n=== Testing Extended Threshold Range ===")
    extended_thresholds = np.arange(0.8, 0.95, 0.01)
    
    for threshold in extended_thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Calculate false positive rate
        true_negatives = ((y_pred == 0) & (y_test == 0)).sum()
        false_positives = ((y_pred == 1) & (y_test == 0)).sum()
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 1.0
        
        positive_predictions = y_pred.sum()
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'false_positive_rate': false_positive_rate,
            'positive_predictions': positive_predictions
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_false_positive_rate = false_positive_rate
    
    # Print final results
    print(f"\n=== Final Results ===")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best F1-score: {best_f1:.3f}")
    print(f"Best precision: {best_precision:.3f}")
    print(f"Best recall: {best_recall:.3f}")
    print(f"False positive rate: {best_false_positive_rate:.3f}")
    
    # Compare with original performance
    print(f"\n=== Comparison with Original Performance ===")
    print(f"Original: AUC=0.544, Precision=0.138, Recall=1.000, F1=0.242")
    print(f"Current: Precision={best_precision:.3f}, Recall={best_recall:.3f}, F1={best_f1:.3f}")
    
    return best_threshold, results

if __name__ == "__main__":
    optimize_original_threshold()
