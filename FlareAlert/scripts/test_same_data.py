#!/usr/bin/env python3
"""
Test script to train models on exact same data with exact same parameters
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

def test_same_data():
    """Test training on exact same data with exact same parameters"""
    print("=== Same Data Test ===\n")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Define training period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    # Prepare data using the same logic
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
    print(f"Positive rate: {y_train.mean():.3f}")
    
    # Test 1: Use exact same parameters as trainer.py
    print("\n=== Test 1: Exact Same Parameters as Trainer ===")
    
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
    
    print(f"Parameters: {params}")
    
    model1 = xgb.XGBClassifier(**params)
    model1.fit(X_train, y_train)
    
    y_pred_proba1 = model1.predict_proba(X_test)[:, 1]
    auc1 = roc_auc_score(y_test, y_pred_proba1)
    
    print(f"Test 1 AUC: {auc1:.3f}")
    
    # Test 2: Use parameters from fine-tuning script
    print("\n=== Test 2: Fine-tuning Script Parameters ===")
    
    params2 = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.95,
        'colsample_bytree': 0.95,
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42
    }
    
    print(f"Parameters: {params2}")
    
    model2 = xgb.XGBClassifier(**params2)
    model2.fit(X_train, y_train)
    
    y_pred_proba2 = model2.predict_proba(X_test)[:, 1]
    auc2 = roc_auc_score(y_test, y_pred_proba2)
    
    print(f"Test 2 AUC: {auc2:.3f}")
    
    # Test 3: Use trainer.py's train_flare_model method
    print("\n=== Test 3: Using Trainer's train_flare_model Method ===")
    
    model3, metadata3 = trainer.train_flare_model(train_data)
    
    # Evaluate using the same test data
    y_pred_proba3 = model3.predict_proba(X_test)[:, 1]
    auc3 = roc_auc_score(y_test, y_pred_proba3)
    
    print(f"Test 3 AUC: {auc3:.3f}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Test 1 (Exact params): {auc1:.3f}")
    print(f"Test 2 (Fine-tune params): {auc2:.3f}")
    print(f"Test 3 (Trainer method): {auc3:.3f}")
    
    return auc1, auc2, auc3

if __name__ == "__main__":
    test_same_data()
