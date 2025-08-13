#!/usr/bin/env python3
"""
Compare model performance across different time periods
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

def train_period_model(days, period_name):
    """Train model for a specific time period"""
    print(f"\n=== {period_name} Model Training ===")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Define training period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    print(f"Total days: {days}")
    
    # Prepare data
    features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
    
    if len(dataset) == 0:
        print(f"No data available for {period_name}")
        return None
    
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
    
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Flare positive rate: {dataset['flare6h_label'].mean():.3f}")
    
    # Remove constant features
    constant_features = []
    for col in X_train.columns:
        if X_train[col].var() == 0:
            constant_features.append(col)
    
    if constant_features:
        print(f"Removing {len(constant_features)} constant features: {constant_features}")
        X_train_clean = X_train.drop(columns=constant_features)
        X_test_clean = X_test.drop(columns=constant_features)
    else:
        X_train_clean = X_train
        X_test_clean = X_test
    
    # Train model with conservative parameters
    positive_samples = y_train.sum()
    scale_pos_weight = (y_train == 0).sum() / positive_samples
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.03,
        'n_estimators': 300,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 5,
        'gamma': 0.2,
        'scale_pos_weight': scale_pos_weight,
        'base_score': 0.1,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_clean, y_train)
    
    # Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
    calibrated_model.fit(X_train_clean, y_train)
    
    # Get probability predictions
    y_pred_proba = calibrated_model.predict_proba(X_test_clean)[:, 1]
    
    # Optimize threshold
    thresholds = np.arange(0.1, 0.4, 0.01)
    
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_auc = 0
    best_balanced_score = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Balanced score: prioritize precision over recall
        balanced_score = (precision * 0.7) + (recall * 0.3)
        
        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_auc = auc
    
    # Final evaluation
    y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
    
    results = {
        'period': period_name,
        'days': days,
        'total_samples': len(dataset),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'positive_rate': dataset['flare6h_label'].mean(),
        'threshold': best_threshold,
        'auc': best_auc,
        'f1': best_f1,
        'precision': best_precision,
        'recall': best_recall,
        'balanced_score': best_balanced_score,
        'positive_predictions': y_pred_optimal.sum(),
        'true_positives': (y_pred_optimal & y_test).sum(),
        'false_positives': (y_pred_optimal & ~y_test).sum(),
        'actual_positives': y_test.sum()
    }
    
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"AUC: {best_auc:.3f}")
    print(f"F1: {best_f1:.3f}")
    print(f"Precision: {best_precision:.3f}")
    print(f"Recall: {best_recall:.3f}")
    print(f"Positive predictions: {y_pred_optimal.sum()}")
    print(f"True positives: {(y_pred_optimal & y_test).sum()}")
    print(f"False positives: {(y_pred_optimal & ~y_test).sum()}")
    
    return results

def compare_time_periods():
    """Compare models across different time periods"""
    print("=== Time Period Comparison ===\n")
    
    # Define time periods to test
    periods = [
        (30, "1 Month"),
        (90, "3 Months"),
        (180, "6 Months")
    ]
    
    results = []
    
    for days, period_name in periods:
        result = train_period_model(days, period_name)
        if result:
            results.append(result)
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    if not results:
        print("No results to compare")
        return
    
    # Print header
    print(f"{'Period':<12} {'Days':<6} {'Samples':<8} {'AUC':<6} {'F1':<6} {'Precision':<10} {'Recall':<8} {'TP':<4} {'FP':<4}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['period']:<12} {result['days']:<6} {result['total_samples']:<8} "
              f"{result['auc']:<6.3f} {result['f1']:<6.3f} {result['precision']:<10.3f} "
              f"{result['recall']:<8.3f} {result['true_positives']:<4} {result['false_positives']:<4}")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    for i, result in enumerate(results):
        print(f"\n{result['period']} Model:")
        print(f"  - Total samples: {result['total_samples']:,}")
        print(f"  - Positive rate: {result['positive_rate']:.3f}")
        print(f"  - Optimal threshold: {result['threshold']:.3f}")
        print(f"  - AUC: {result['auc']:.3f}")
        print(f"  - F1-score: {result['f1']:.3f}")
        print(f"  - Precision: {result['precision']:.3f}")
        print(f"  - Recall: {result['recall']:.3f}")
        print(f"  - True positives: {result['true_positives']}")
        print(f"  - False positives: {result['false_positives']}")
        print(f"  - False positive ratio: {result['false_positives']/max(result['true_positives'], 1):.2f}")
    
    # Find best model by different metrics
    print("\n" + "="*80)
    print("BEST MODELS BY METRIC")
    print("="*80)
    
    if results:
        best_auc = max(results, key=lambda x: x['auc'])
        best_f1 = max(results, key=lambda x: x['f1'])
        best_precision = max(results, key=lambda x: x['precision'])
        best_balanced = max(results, key=lambda x: x['balanced_score'])
        
        print(f"Best AUC: {best_auc['period']} (AUC: {best_auc['auc']:.3f})")
        print(f"Best F1: {best_f1['period']} (F1: {best_f1['f1']:.3f})")
        print(f"Best Precision: {best_precision['period']} (Precision: {best_precision['precision']:.3f})")
        print(f"Best Balanced Score: {best_balanced['period']} (Score: {best_balanced['balanced_score']:.3f})")
    
    return results

if __name__ == "__main__":
    compare_time_periods()
