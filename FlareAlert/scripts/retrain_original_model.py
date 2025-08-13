#!/usr/bin/env python3
"""
Retrain model with original parameters to get back to AUC 0.544 performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from models.trainer import ModelTrainer

def retrain_original_model():
    """Retrain model with original parameters"""
    print("=== Retraining with Original Parameters ===\n")
    
    # Initialize trainer
    trainer = ModelTrainer()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    # Prepare data
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
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['flare6h_label']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['flare6h_label']
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")
    
    # Use original parameters (from working model)
    positive_samples = y_train.sum()
    scale_pos_weight = (y_train == 0).sum() / positive_samples
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.95,
        'colsample_bytree': 0.95,
        'min_child_weight': 3,
        'gamma': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'base_score': 0.15,
        'random_state': 42
    }
    
    print(f"\n=== Training Model with Original Parameters ===")
    print(f"Scale pos weight: {scale_pos_weight:.1f}")
    
    # Train model WITHOUT calibration (to avoid overconfidence)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n=== Model Performance ===")
    print(f"AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    
    # Test different thresholds
    print(f"\n=== Threshold Optimization ===")
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba > threshold).astype(int)
        
        f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)
        precision_thresh = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall_thresh = recall_score(y_test, y_pred_thresh, zero_division=0)
        
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = threshold
            best_precision = precision_thresh
            best_recall = recall_thresh
    
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best F1-score: {best_f1:.3f}")
    print(f"Best precision: {best_precision:.3f}")
    print(f"Best recall: {best_recall:.3f}")
    
    # Save model
    import pickle
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/flare_original_{timestamp}.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n=== Model Saved ===")
    print(f"Model saved to: {model_path}")
    
    return model, auc, best_threshold

if __name__ == "__main__":
    retrain_original_model()
