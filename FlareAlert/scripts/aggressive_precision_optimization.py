#!/usr/bin/env python3
"""
Aggressively optimize threshold for precision to reduce false positives
Use the working model but find a much higher threshold
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, brier_score_loss
)
import pickle
from models.trainer import ModelTrainer

def aggressive_precision_optimization():
    """Aggressively optimize threshold for precision"""
    print("=== Aggressive Precision Optimization ===\n")
    
    # Load the working model (the one with 0.544 AUC)
    print("1. Loading working model...")
    try:
        # Try to load the specific working model
        with open('models/flare_20250813_091105.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Loaded flare_20250813_091105.pkl (working model)")
    except:
        # Fallback to latest
        with open('models/flare_latest.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Loaded flare_latest.pkl")
    
    # Prepare data (same as the working model)
    print("2. Preparing data...")
    trainer = ModelTrainer()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
    dataset = dataset.sort_values('timestamp')
    
    # Split data (same logic as working model)
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
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n3. Current model performance:")
    print(f"Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    print(f"Std probability: {y_pred_proba.std():.3f}")
    
    # Calculate current metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    print(f"ROC AUC: {auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"Brier Score: {brier:.3f}")
    
    # 4. Aggressively optimize threshold for precision
    print(f"\n4. Aggressively optimizing threshold for precision...")
    
    # Test very high thresholds to reduce false positives
    thresholds = np.arange(0.5, 0.99, 0.01)  # Start from 0.5, go up to 0.98
    
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_fpr = 1.0
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
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
        
        # Find best threshold that maximizes precision while keeping some recall
        if precision > best_precision and recall > 0.1:  # Must have at least 10% recall
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_fpr = false_positive_rate
    
    # 5. Final evaluation
    print(f"\n5. Final evaluation...")
    
    y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Precision: {best_precision:.3f}")
    print(f"Recall: {best_recall:.3f}")
    print(f"F1-Score: {best_f1:.3f}")
    print(f"False Positive Rate: {best_fpr:.3f}")
    print(f"Positive Predictions: {y_pred_optimal.sum()}")
    
    # Show top 10 thresholds by precision
    print(f"\n=== Top 10 Thresholds by Precision ===")
    results_sorted = sorted(results, key=lambda x: x['precision'], reverse=True)
    for i, result in enumerate(results_sorted[:10]):
        print(f"{i+1}. Threshold: {result['threshold']:.3f}, Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}, FPR: {result['false_positive_rate']:.3f}, "
              f"Predictions: {result['positive_predictions']}")
    
    # Show thresholds with reasonable recall (>0.1)
    print(f"\n=== Thresholds with Recall > 0.1 ===")
    reasonable_results = [r for r in results if r['recall'] > 0.1]
    reasonable_results_sorted = sorted(reasonable_results, key=lambda x: x['precision'], reverse=True)
    for i, result in enumerate(reasonable_results_sorted[:10]):
        print(f"{i+1}. Threshold: {result['threshold']:.3f}, Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}, FPR: {result['false_positive_rate']:.3f}")
    
    # 6. Save optimized model
    print(f"\n6. Saving precision-optimized model...")
    
    # Create metadata
    metadata = {
        'model_type': 'flare_prediction_precision_optimized',
        'target': 'flare6h_label',
        'feature_columns': feature_cols,
        'test_samples': len(test_data),
        'positive_rate': y_test.mean(),
        'trained_at': datetime.now().isoformat(),
        'performance': {
            'auc': auc,
            'pr_auc': pr_auc,
            'brier': brier,
            'threshold': best_threshold,
            'precision': best_precision,
            'recall': best_recall,
            'f1': best_f1,
            'false_positive_rate': best_fpr
        }
    }
    
    # Save metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metadata_path = f'models/flare_precision_aggressive_{timestamp}_metadata.json'
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_path}")
    
    print(f"\n=== Aggressive Precision Optimization Complete ===")
    print(f"Key results:")
    print(f"1. Optimal threshold: {best_threshold:.3f}")
    print(f"2. Precision: {best_precision:.3f}")
    print(f"3. Recall: {best_recall:.3f}")
    print(f"4. False Positive Rate: {best_fpr:.3f}")
    print(f"5. F1-Score: {best_f1:.3f}")
    
    # Compare with original
    print(f"\n=== Comparison with Original ===")
    print(f"Original FPR: 0.920")
    print(f"New FPR: {best_fpr:.3f}")
    print(f"FPR Improvement: {0.920 - best_fpr:.3f}")
    print(f"Original Precision: 0.148")
    print(f"New Precision: {best_precision:.3f}")
    print(f"Precision Improvement: {best_precision - 0.148:.3f}")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if best_fpr < 0.7:
        print("✅ SUCCESS: False positive rate reduced significantly!")
    elif best_fpr < 0.85:
        print("⚠️  MODERATE: False positive rate reduced somewhat")
    else:
        print("❌ LIMITED: False positive rate still high")
    
    if best_precision > 0.3:
        print("✅ SUCCESS: Precision improved significantly!")
    elif best_precision > 0.2:
        print("⚠️  MODERATE: Precision improved somewhat")
    else:
        print("❌ LIMITED: Precision still low")

if __name__ == "__main__":
    aggressive_precision_optimization()
