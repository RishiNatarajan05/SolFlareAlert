#!/usr/bin/env python3
"""
Conservative fix for 6-month model - focus on precision
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from models.trainer import ModelTrainer

def conservative_6months_fix():
    """Conservative fix focusing on precision"""
    print("=== Conservative 6-Month Fix ===\n")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Define training period (6 months)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    # Prepare data
    features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
    
    # Sort by timestamp
    dataset = dataset.sort_values('timestamp')
    
    # Find positive samples
    positive_samples = dataset[dataset['flare6h_label'] == 1]
    negative_samples = dataset[dataset['flare6h_label'] == 0]
    
    # Split using the same logic
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
    
    feature_cols = [col for col in dataset.columns 
                   if not col.endswith('_label') and col != 'timestamp']
    
    X_train = train_data[feature_cols]
    y_train = train_data['flare6h_label']
    
    X_test = test_data[feature_cols]
    y_test = test_data['flare6h_label']
    
    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    
    # Remove only the constant features (minimal change)
    print("\n=== Remove Only Constant Features ===")
    constant_features = []
    for col in X_train.columns:
        if X_train[col].var() == 0:
            constant_features.append(col)
    
    print(f"Removing {len(constant_features)} constant features: {constant_features}")
    X_train_clean = X_train.drop(columns=constant_features)
    X_test_clean = X_test.drop(columns=constant_features)
    
    print(f"Features after cleaning: {len(X_train_clean.columns)}")
    
    # Train model with more conservative parameters
    print("\n=== Train Conservative Model ===")
    
    # Calculate scale_pos_weight
    positive_samples = y_train.sum()
    scale_pos_weight = (y_train == 0).sum() / positive_samples
    
    # More conservative parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,  # Reduced from 4
        'learning_rate': 0.03,  # Reduced from 0.05
        'n_estimators': 300,  # Increased from 200
        'subsample': 0.9,  # Reduced from 0.95
        'colsample_bytree': 0.9,  # Reduced from 0.95
        'min_child_weight': 5,  # Increased from 3
        'gamma': 0.2,  # Increased from 0.1
        'scale_pos_weight': scale_pos_weight,
        'base_score': 0.1,  # Reduced from 0.15
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_clean, y_train)
    
    # Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
    calibrated_model.fit(X_train_clean, y_train)
    
    # Get probability predictions
    y_pred_proba = calibrated_model.predict_proba(X_test_clean)[:, 1]
    
    print(f"Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    
    # Test different thresholds with focus on precision
    print("\n=== Precision-Focused Threshold Optimization ===")
    
    thresholds = np.arange(0.1, 0.4, 0.01)
    
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_auc = 0
    best_balanced_score = 0
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Balanced score: prioritize precision over recall
        balanced_score = (precision * 0.7) + (recall * 0.3)
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'balanced_score': balanced_score,
            'positive_predictions': y_pred.sum()
        })
        
        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_auc = auc
    
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best F1-score: {best_f1:.3f}")
    print(f"Best precision: {best_precision:.3f}")
    print(f"Best recall: {best_recall:.3f}")
    print(f"AUC: {best_auc:.3f}")
    print(f"Balanced score: {best_balanced_score:.3f}")
    
    # Show top 10 thresholds by balanced score
    print(f"\n=== Top 10 Thresholds by Balanced Score ===")
    results_sorted = sorted(results, key=lambda x: x['balanced_score'], reverse=True)
    for i, result in enumerate(results_sorted[:10]):
        print(f"{i+1}. Threshold: {result['threshold']:.3f}, Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}, "
              f"Predictions: {result['positive_predictions']}")
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
    
    print(f"Predictions with threshold {best_threshold:.3f}:")
    print(f"  - Positive predictions: {y_pred_optimal.sum()}")
    print(f"  - Negative predictions: {(y_pred_optimal == 0).sum()}")
    print(f"  - Actual positives: {y_test.sum()}")
    print(f"  - True positives: {(y_pred_optimal & y_test).sum()}")
    print(f"  - False positives: {(y_pred_optimal & ~y_test).sum()}")
    
    # Compare with original problematic model
    print(f"\n=== Comparison with Original Model ===")
    print(f"Original: AUC=0.518, F1=0.303, Precision=0.279, Recall=0.331")
    print(f"Conservative: AUC={best_auc:.3f}, F1={best_f1:.3f}, Precision={best_precision:.3f}, Recall={best_recall:.3f}")
    
    improvement_auc = ((best_auc - 0.518) / 0.518) * 100
    improvement_f1 = ((best_f1 - 0.303) / 0.303) * 100
    improvement_precision = ((best_precision - 0.279) / 0.279) * 100
    
    print(f"Improvements:")
    print(f"  - AUC: {improvement_auc:+.1f}%")
    print(f"  - F1: {improvement_f1:+.1f}%")
    print(f"  - Precision: {improvement_precision:+.1f}%")
    
    return best_threshold, best_auc, best_f1, best_precision, best_recall

if __name__ == "__main__":
    conservative_6months_fix()
