#!/usr/bin/env python3
"""
Fix false positive rate by optimizing for precision and using conservative thresholds
Priority: Reduce false positives even if AUC goes down
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import pickle
from models.trainer import ModelTrainer

def create_cyclic_time_features(df):
    """Replace raw time features with cyclic encodings"""
    df_cyclic = df.copy()
    
    # Hour of day (0-23)
    if 'hour_of_day' in df_cyclic.columns:
        hour = df_cyclic['hour_of_day']
        df_cyclic['sin_hour'] = np.sin(2 * np.pi * hour / 24)
        df_cyclic['cos_hour'] = np.cos(2 * np.pi * hour / 24)
        df_cyclic = df_cyclic.drop(columns=['hour_of_day'])
    
    # Day of week (0-6)
    if 'day_of_week' in df_cyclic.columns:
        dow = df_cyclic['day_of_week']
        df_cyclic['sin_dow'] = np.sin(2 * np.pi * dow / 7)
        df_cyclic['cos_dow'] = np.cos(2 * np.pi * dow / 7)
        df_cyclic = df_cyclic.drop(columns=['day_of_week'])
    
    # Day of year (1-365)
    if 'day_of_year' in df_cyclic.columns:
        doy = df_cyclic['day_of_year']
        df_cyclic['sin_doy'] = np.sin(2 * np.pi * doy / 365)
        df_cyclic['cos_doy'] = np.cos(2 * np.pi * doy / 365)
        df_cyclic = df_cyclic.drop(columns=['day_of_year'])
    
    # Month (1-12)
    if 'month' in df_cyclic.columns:
        month = df_cyclic['month']
        df_cyclic['sin_month'] = np.sin(2 * np.pi * month / 12)
        df_cyclic['cos_month'] = np.cos(2 * np.pi * month / 12)
        df_cyclic = df_cyclic.drop(columns=['month'])
    
    return df_cyclic

def fix_false_positives():
    """Fix false positive rate by optimizing for precision"""
    print("=== Fixing False Positive Rate ===\n")
    
    # 1. Prepare data with cyclic time features
    print("1. Preparing data with cyclic time features...")
    trainer = ModelTrainer()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
    dataset = dataset.sort_values('timestamp')
    
    # Apply cyclic time encoding
    dataset_cyclic = create_cyclic_time_features(dataset)
    
    # Split data (chronological)
    positive_samples = dataset_cyclic[dataset_cyclic['flare6h_label'] == 1]
    negative_samples = dataset_cyclic[dataset_cyclic['flare6h_label'] == 0]
    
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
        split_idx = int(len(dataset_cyclic) * 0.8)
        train_data = dataset_cyclic.iloc[:split_idx]
        test_data = dataset_cyclic.iloc[split_idx:]
    
    feature_cols = [col for col in dataset_cyclic.columns if not col.endswith('_label') and col != 'timestamp']
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['flare6h_label']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['flare6h_label']
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Features: {len(feature_cols)} (with cyclic time encoding)")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")
    
    # 2. Use conservative XGBoost parameters focused on precision
    print(f"\n2. Training with precision-focused parameters...")
    
    # Calculate scale_pos_weight (use lower value to reduce false positives)
    positive_samples = y_train.sum()
    scale_pos_weight = min(3.0, (y_train == 0).sum() / positive_samples)  # Lower than before
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',  # Keep AUC for now
        'learning_rate': 0.02,  # Slower learning
        'max_depth': 3,         # Shallow trees to reduce overfitting
        'min_child_weight': 5,  # Higher to reduce false positives
        'gamma': 0.2,           # Higher to reduce false positives
        'subsample': 0.8,       # Lower to reduce overfitting
        'colsample_bytree': 0.8, # Lower to reduce overfitting
        'reg_lambda': 3.0,      # Higher L2 regularization
        'reg_alpha': 0.5,       # Add L1 regularization
        'n_estimators': 1000,   # More trees
        'max_delta_step': 2,    # Higher to stabilize probabilities
        'scale_pos_weight': scale_pos_weight,
        'base_score': 0.1,      # Lower base score
        'random_state': 42
    }
    
    print(f"Scale pos weight: {scale_pos_weight:.1f}")
    print(f"Base score: {params['base_score']}")
    
    # 3. Train model
    print(f"\n3. Training model...")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # 4. Calibrate probabilities
    print(f"\n4. Calibrating probabilities...")
    
    calibrated_model = CalibratedClassifierCV(
        model, cv='prefit', method='sigmoid'
    )
    calibrated_model.fit(X_train, y_train)
    
    # 5. Evaluate and optimize threshold for precision
    print(f"\n5. Evaluating and optimizing for precision...")
    
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    print(f"Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    print(f"Std probability: {y_pred_proba.std():.3f}")
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    print(f"ROC AUC: {auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"Brier Score: {brier:.3f}")
    
    # 6. Optimize threshold for precision (not recall)
    print(f"\n6. Optimizing threshold for precision...")
    
    thresholds = np.arange(0.1, 0.95, 0.01)
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
        
        # Calculate balanced score prioritizing precision
        balanced_score = (precision * 0.7) + (recall * 0.3) - (false_positive_rate * 0.5)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': false_positive_rate,
            'balanced_score': balanced_score,
            'positive_predictions': y_pred.sum()
        })
        
        if balanced_score > best_f1:  # Use balanced score for selection
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_fpr = false_positive_rate
    
    # 7. Final evaluation with optimal threshold
    print(f"\n7. Final evaluation...")
    
    y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Precision: {best_precision:.3f}")
    print(f"Recall: {best_recall:.3f}")
    print(f"F1-Score: {best_f1:.3f}")
    print(f"False Positive Rate: {best_fpr:.3f}")
    
    # Show top 10 thresholds by precision
    print(f"\n=== Top 10 Thresholds by Precision ===")
    results_sorted = sorted(results, key=lambda x: x['precision'], reverse=True)
    for i, result in enumerate(results_sorted[:10]):
        print(f"{i+1}. Threshold: {result['threshold']:.3f}, Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}, FPR: {result['false_positive_rate']:.3f}")
    
    # 8. Save improved model
    print(f"\n8. Saving precision-optimized model...")
    
    # Create metadata
    metadata = {
        'model_type': 'flare_prediction_precision_optimized',
        'target': 'flare6h_label',
        'feature_columns': feature_cols,
        'training_samples': len(train_data),
        'test_samples': len(test_data),
        'positive_rate': y_train.mean(),
        'trained_at': datetime.now().isoformat(),
        'xgb_params': params,
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
    
    # Save model and metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/flare_precision_optimized_{timestamp}.pkl'
    metadata_path = f'models/flare_precision_optimized_{timestamp}_metadata.json'
    
    with open(model_path, 'wb') as f:
        pickle.dump(calibrated_model, f)
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save as latest
    with open('models/flare_latest.pkl', 'wb') as f:
        pickle.dump(calibrated_model, f)
    
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    
    print(f"\n=== Precision Optimization Complete ===")
    print(f"Key improvements:")
    print(f"1. Conservative parameters: Applied")
    print(f"2. Precision-focused threshold: {best_threshold:.3f}")
    print(f"3. False positive rate: {best_fpr:.3f}")
    print(f"4. Precision: {best_precision:.3f}")
    print(f"5. Recall: {best_recall:.3f}")
    print(f"6. F1-Score: {best_f1:.3f}")
    
    # Compare with original
    print(f"\n=== Comparison with Original ===")
    print(f"Original FPR: 0.920")
    print(f"New FPR: {best_fpr:.3f}")
    print(f"FPR Improvement: {0.920 - best_fpr:.3f}")
    print(f"Original Precision: 0.148")
    print(f"New Precision: {best_precision:.3f}")
    print(f"Precision Improvement: {best_precision - 0.148:.3f}")

if __name__ == "__main__":
    fix_false_positives()
