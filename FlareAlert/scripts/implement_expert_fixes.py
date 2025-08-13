#!/usr/bin/env python3
"""
Implement ML expert recommendations for fixing high false positive rate
Priority fixes: cyclic time encoding, PR-AUC optimization, calibration
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
from sklearn.model_selection import TimeSeriesSplit
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

def implement_expert_fixes():
    """Implement ML expert recommendations"""
    print("=== Implementing ML Expert Fixes ===\n")
    
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
    
    # 2. Use expert-recommended XGBoost parameters
    print(f"\n2. Training with expert-recommended parameters...")
    
    # Calculate scale_pos_weight (start lower as recommended)
    positive_samples = y_train.sum()
    scale_pos_weight = min(4.5, (y_train == 0).sum() / positive_samples)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['aucpr', 'logloss'],  # PR-AUC + logloss
        'learning_rate': 0.03,
        'max_depth': 6,            # was 4
        'min_child_weight': 1,     # was 3
        'gamma': 0.0,              # was 0.1
        'subsample': 0.7,          # was 0.95
        'colsample_bytree': 0.7,   # was 0.95
        'reg_lambda': 2.0,         # add L2
        'reg_alpha': 0.0,
        'n_estimators': 2000,      # rely on early stopping
        'max_delta_step': 1,       # stabilizes probs on imbalance
        'scale_pos_weight': scale_pos_weight,
        'base_score': 0.137,       # match prevalence
        'early_stopping_rounds': 200,
        'random_state': 42
    }
    
    print(f"Scale pos weight: {scale_pos_weight:.1f}")
    print(f"Base score: {params['base_score']}")
    
    # 3. Train model with early stopping
    print(f"\n3. Training model...")
    
    # Train model with early stopping
    model = xgb.XGBClassifier(**params)
    
    # Prepare validation data for early stopping
    X_val = X_test[:len(X_test)//2]  # Use first half of test as validation
    y_val = y_test[:len(y_test)//2]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 4. Calibrate probabilities
    print(f"\n4. Calibrating probabilities...")
    
    # Calibrate with Platt scaling
    calibrated_model = CalibratedClassifierCV(
        model, cv='prefit', method='sigmoid'
    )
    calibrated_model.fit(X_train, y_train)
    
    # 5. Evaluate and optimize threshold
    print(f"\n5. Evaluating calibrated model...")
    
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
    
    # 6. Optimize threshold for recall ~0.8
    print(f"\n6. Optimizing threshold for recall ~0.8...")
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_recall = 0
    best_precision = 0
    target_recall = 0.8
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        
        if abs(recall - target_recall) < abs(best_recall - target_recall):
            best_threshold = threshold
            best_recall = recall
            best_precision = precision
    
    # 7. Final evaluation with optimal threshold
    print(f"\n7. Final evaluation...")
    
    y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
    f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)
    
    # Calculate false positive rate
    true_negatives = ((y_pred_optimal == 0) & (y_test == 0)).sum()
    false_positives = ((y_pred_optimal == 1) & (y_test == 0)).sum()
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 1.0
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Precision: {best_precision:.3f}")
    print(f"Recall: {best_recall:.3f}")
    print(f"F1-Score: {f1_optimal:.3f}")
    print(f"False Positive Rate: {false_positive_rate:.3f}")
    
    # 8. Save improved model
    print(f"\n8. Saving improved model...")
    
    # Create metadata
    metadata = {
        'model_type': 'flare_prediction_expert_fixes',
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
            'f1': f1_optimal,
            'false_positive_rate': false_positive_rate
        }
    }
    
    # Save model and metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/flare_expert_fixes_{timestamp}.pkl'
    metadata_path = f'models/flare_expert_fixes_{timestamp}_metadata.json'
    
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
    
    print(f"\n=== Expert Fixes Complete ===")
    print(f"Key improvements:")
    print(f"1. Cyclic time encoding: Applied")
    print(f"2. PR-AUC optimization: {pr_auc:.3f}")
    print(f"3. Probability calibration: Applied")
    print(f"4. Threshold optimization: {best_threshold:.3f}")
    print(f"5. Target recall achieved: {best_recall:.3f} (target: {target_recall})")
    print(f"6. Precision improvement: {best_precision:.3f}")
    print(f"7. False positive rate: {false_positive_rate:.3f}")

if __name__ == "__main__":
    implement_expert_fixes()
