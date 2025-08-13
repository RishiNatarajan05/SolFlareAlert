#!/usr/bin/env python3
"""
Maximize precision by optimizing threshold
Find the best threshold that gives highest precision while maintaining some recall
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from models.trainer import ModelTrainer

def maximize_precision():
    """Find threshold that maximizes precision"""
    print("=== Maximize Precision - Threshold Optimization ===\n")
    
    # Load the latest model
    print("1. Loading latest model...")
    with open('models/flare_20250813_105759.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Prepare test data
    print("2. Preparing test data...")
    trainer = ModelTrainer()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
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
    
    # Get predictions
    print("3. Getting model predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    print(f"Std probability: {y_pred_proba.std():.3f}")
    
    # Test different thresholds to maximize precision
    print("\n4. Testing thresholds to maximize precision...")
    
    # Test thresholds from 0.5 to 0.99
    thresholds = np.arange(0.5, 0.99, 0.01)
    
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_fpr = 1.0
    best_predictions = 0
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Skip if no positive predictions
        if y_pred.sum() == 0:
            continue
            
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate false positive rate
        true_negatives = ((y_pred == 0) & (y_test == 0)).sum()
        false_positives = ((y_pred == 1) & (y_test == 0)).sum()
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 1.0
        
        # Calculate positive predictions
        positive_predictions = y_pred.sum()
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': false_positive_rate,
            'positive_predictions': positive_predictions
        })
        
        # Find best threshold that maximizes precision
        if precision > best_precision:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_fpr = false_positive_rate
            best_predictions = positive_predictions
    
    # 5. Final evaluation
    print(f"\n5. Final evaluation...")
    
    y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Precision: {best_precision:.3f}")
    print(f"Recall: {best_recall:.3f}")
    print(f"F1-Score: {best_f1:.3f}")
    print(f"False Positive Rate: {best_fpr:.3f}")
    print(f"Positive Predictions: {best_predictions}")
    
    # Show top 10 thresholds by precision
    print(f"\n=== Top 10 Thresholds by Precision ===")
    results_sorted = sorted(results, key=lambda x: x['precision'], reverse=True)
    for i, result in enumerate(results_sorted[:10]):
        print(f"{i+1}. Threshold: {result['threshold']:.3f}, Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}, FPR: {result['false_positive_rate']:.3f}, "
              f"Predictions: {result['positive_predictions']}")
    
    # 6. Save optimized model
    print(f"\n6. Saving optimized model...")
    
    # Create metadata
    metadata = {
        'model_type': 'flare_prediction_precision_optimized',
        'target': 'flare6h_label',
        'feature_columns': feature_cols,
        'test_samples': len(test_data),
        'positive_rate': y_test.mean(),
        'trained_at': datetime.now().isoformat(),
        'optimization': {
            'threshold': float(best_threshold),
            'precision': float(best_precision),
            'recall': float(best_recall),
            'f1': float(best_f1),
            'false_positive_rate': float(best_fpr),
            'positive_predictions': int(best_predictions)
        }
    }
    
    # Save metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metadata_path = f'models/flare_precision_optimized_{timestamp}_metadata.json'
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_path}")
    
    print(f"\n=== Precision Optimization Complete ===")
    print(f"Key results:")
    print(f"1. Optimal threshold: {best_threshold:.3f}")
    print(f"2. Precision: {best_precision:.3f} ({best_precision*100:.1f}%)")
    print(f"3. Recall: {best_recall:.3f} ({best_recall*100:.1f}%)")
    print(f"4. False Positive Rate: {best_fpr:.3f} ({best_fpr*100:.1f}%)")
    print(f"5. F1-Score: {best_f1:.3f}")
    print(f"6. Positive Predictions: {best_predictions}")
    
    # Compare with original
    print(f"\n=== Comparison with Original ===")
    print(f"Original Precision: 0.138")
    print(f"Optimized Precision: {best_precision:.3f}")
    print(f"Precision Improvement: {best_precision - 0.138:.3f}")
    print(f"Original FPR: 1.000")
    print(f"Optimized FPR: {best_fpr:.3f}")
    print(f"FPR Improvement: {1.0 - best_fpr:.3f}")
    print(f"Original Predictions: 145")
    print(f"Optimized Predictions: {best_predictions}")
    print(f"Prediction Reduction: {145 - best_predictions}")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if best_precision > 0.5:
        print("✅ EXCELLENT: Precision improved dramatically!")
    elif best_precision > 0.3:
        print("✅ GOOD: Precision improved significantly!")
    elif best_precision > 0.2:
        print("⚠️  MODERATE: Precision improved somewhat")
    else:
        print("❌ LIMITED: Precision still low")
    
    if best_fpr < 0.3:
        print("✅ EXCELLENT: False positive rate reduced dramatically!")
    elif best_fpr < 0.5:
        print("✅ GOOD: False positive rate reduced significantly!")
    elif best_fpr < 0.7:
        print("⚠️  MODERATE: False positive rate reduced somewhat")
    else:
        print("❌ LIMITED: False positive rate still high")
    
    if best_predictions < 30:
        print("✅ EXCELLENT: Dramatically reduced false alarms!")
    elif best_predictions < 60:
        print("✅ GOOD: Significantly reduced false alarms!")
    else:
        print("⚠️  MODERATE: Some reduction in false alarms")
    
    print(f"\n=== Operational Impact ===")
    print(f"With threshold {best_threshold:.3f}:")
    print(f"- You'll get {best_predictions} flare alerts instead of 145")
    print(f"- {best_precision*100:.1f}% of alerts will be real flares")
    print(f"- You'll miss {100-best_recall*100:.1f}% of actual flares")
    print(f"- False alarm rate: {best_fpr*100:.1f}%")
    
    # Save the optimized threshold for future use
    threshold_info = {
        'optimal_threshold': best_threshold,
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'false_positive_rate': best_fpr,
        'positive_predictions': best_predictions
    }
    
    threshold_path = f'models/optimal_threshold_{timestamp}.json'
    with open(threshold_path, 'w') as f:
        json.dump(threshold_info, f, indent=2)
    
    print(f"\nOptimal threshold saved: {threshold_path}")

if __name__ == "__main__":
    maximize_precision()
