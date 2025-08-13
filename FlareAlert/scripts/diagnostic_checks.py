#!/usr/bin/env python3
"""
Quick diagnostic checks for ML expert consultation
Based on expert recommendations for probability collapse and leakage detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import pickle
from models.trainer import ModelTrainer

def diagnostic_checks():
    """Run all diagnostic checks recommended by ML expert"""
    print("=== ML Expert Diagnostic Checks ===\n")
    
    # Load current model and data
    print("1. Loading current model and data...")
    with open('models/flare_latest.pkl', 'rb') as f:
        model = pickle.load(f)
    
    trainer = ModelTrainer()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
    dataset = dataset.sort_values('timestamp')
    
    # Split data (same as training)
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
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['flare6h_label']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['flare6h_label']
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n2. Probability Distribution Analysis:")
    print(f"Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    print(f"Std probability: {y_pred_proba.std():.3f}")
    
    # Check for probability collapse
    print(f"\n3. Probability Collapse Check:")
    print(f"Probabilities in 0.89-0.92 range: {((y_pred_proba >= 0.89) & (y_pred_proba <= 0.92)).sum()}/{len(y_pred_proba)}")
    print(f"Probabilities > 0.9: {(y_pred_proba > 0.9).sum()}/{len(y_pred_proba)}")
    print(f"Probabilities < 0.1: {(y_pred_proba < 0.1).sum()}/{len(y_pred_proba)}")
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    print(f"\n4. Model Performance Metrics:")
    print(f"ROC AUC: {auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"Brier Score: {brier:.3f}")
    
    # Calibration analysis
    print(f"\n5. Calibration Analysis:")
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    print(f"Calibration curve - True probs: {prob_true}")
    print(f"Calibration curve - Pred probs: {prob_pred}")
    
    # Permutation importance on test set
    print(f"\n6. Permutation Importance (Test Set):")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    feature_importance = dict(zip(feature_cols, perm_importance.importances_mean))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    print("Top 10 features by permutation importance:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Leakage check: shift features by +6h
    print(f"\n7. Leakage Check (Shift features by +6h):")
    shifted_dataset = dataset.copy()
    shifted_dataset['timestamp'] = shifted_dataset['timestamp'] + timedelta(hours=6)
    
    # Re-prepare data with shifted timestamps
    shifted_features_df, shifted_labels_df, shifted_dataset = trainer.prepare_training_data(
        start_time + timedelta(hours=6), end_time + timedelta(hours=6)
    )
    
    if len(shifted_dataset) > 0:
        shifted_dataset = shifted_dataset.sort_values('timestamp')
        shifted_feature_cols = [col for col in shifted_dataset.columns if not col.endswith('_label') and col != 'timestamp']
        X_shifted = shifted_dataset[shifted_feature_cols].fillna(0)
        y_shifted = shifted_dataset['flare6h_label']
        
        if len(X_shifted) > 0 and len(y_shifted) > 0:
            y_pred_shifted = model.predict_proba(X_shifted)[:, 1]
            auc_shifted = roc_auc_score(y_shifted, y_pred_shifted)
            print(f"Original AUC: {auc:.3f}")
            print(f"Shifted AUC: {auc_shifted:.3f}")
            print(f"AUC change: {auc_shifted - auc:.3f}")
            print(f"Leakage detected: {'YES' if auc_shifted > 0.6 else 'NO'}")
        else:
            print("Not enough shifted data for comparison")
    else:
        print("No shifted data available")
    
    # Day_of_year importance check
    print(f"\n8. Day_of_year Importance Check:")
    if 'day_of_year' in feature_cols:
        print(f"Day_of_year permutation importance: {feature_importance.get('day_of_year', 0):.4f}")
        print(f"Day_of_year rank: {list(feature_importance.keys()).index('day_of_year') + 1 if 'day_of_year' in feature_importance else 'N/A'}")
        
        # Check if removing day_of_year affects performance
        X_test_no_doy = X_test.drop(columns=['day_of_year'])
        model_no_doy = pickle.load(open('models/flare_latest.pkl', 'rb'))
        try:
            y_pred_no_doy = model_no_doy.predict_proba(X_test_no_doy)[:, 1]
            auc_no_doy = roc_auc_score(y_test, y_pred_no_doy)
            print(f"AUC without day_of_year: {auc_no_doy:.3f}")
            print(f"AUC change: {auc_no_doy - auc:.3f}")
        except:
            print("Could not test without day_of_year (model expects this feature)")
    else:
        print("Day_of_year not in features")
    
    print(f"\n=== Diagnostic Checks Complete ===")
    print(f"Key Issues Found:")
    print(f"1. Probability collapse: {'YES' if y_pred_proba.std() < 0.1 else 'NO'}")
    print(f"2. Overconfidence: {'YES' if y_pred_proba.mean() > 0.8 else 'NO'}")
    print(f"3. Poor calibration: {'YES' if abs(prob_true.mean() - prob_pred.mean()) > 0.2 else 'NO'}")
    print(f"4. Low PR-AUC: {'YES' if pr_auc < 0.3 else 'NO'}")

if __name__ == "__main__":
    diagnostic_checks()
