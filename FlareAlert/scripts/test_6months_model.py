#!/usr/bin/env python3
"""
Test script to debug 6-month model predictions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from models.trainer import ModelTrainer

def test_6months_model():
    """Test the 6-month model to see why it's predicting all zeros"""
    print("=== 6-Month Model Test ===\n")
    
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
    print(f"Test positive rate: {y_test.mean():.3f}")
    
    # Test 1: Basic XGBoost without calibration
    print("\n=== Test 1: Basic XGBoost ===")
    
    # Calculate scale_pos_weight
    positive_samples = (y_train == 1).sum()
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
    
    model1 = xgb.XGBClassifier(**params)
    model1.fit(X_train, y_train)
    
    y_pred_proba1 = model1.predict_proba(X_test)[:, 1]
    y_pred1 = model1.predict(X_test)
    
    auc1 = roc_auc_score(y_test, y_pred_proba1)
    positive_predictions1 = (y_pred1 == 1).sum()
    
    print(f"Test 1 AUC: {auc1:.3f}")
    print(f"Test 1 positive predictions: {positive_predictions1}")
    print(f"Test 1 prediction distribution: {np.bincount(y_pred1)}")
    
    # Test 2: With calibration
    print("\n=== Test 2: With Calibration ===")
    
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(model1, cv=3, method='isotonic')
    calibrated_model.fit(X_train, y_train)
    
    y_pred_proba2 = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred2 = (y_pred_proba2 > 0.5).astype(int)
    
    auc2 = roc_auc_score(y_test, y_pred_proba2)
    positive_predictions2 = (y_pred2 == 1).sum()
    
    print(f"Test 2 AUC: {auc2:.3f}")
    print(f"Test 2 positive predictions: {positive_predictions2}")
    print(f"Test 2 prediction distribution: {np.bincount(y_pred2)}")
    
    # Test 3: Check probability distribution
    print("\n=== Test 3: Probability Distribution ===")
    print(f"Min probability: {y_pred_proba2.min():.3f}")
    print(f"Max probability: {y_pred_proba2.max():.3f}")
    print(f"Mean probability: {y_pred_proba2.mean():.3f}")
    print(f"Std probability: {y_pred_proba2.std():.3f}")
    
    # Check how many predictions are above different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for threshold in thresholds:
        above_threshold = (y_pred_proba2 > threshold).sum()
        print(f"Predictions > {threshold}: {above_threshold}")
    
    return auc1, auc2, positive_predictions1, positive_predictions2

if __name__ == "__main__":
    test_6months_model()
