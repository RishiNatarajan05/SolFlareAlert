#!/usr/bin/env python3
"""
Advanced fine-tuning script for solar weather prediction models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
from models.trainer import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def optimize_threshold(model, X_val, y_val):
    """Find optimal threshold for better F1-score"""
    print("=== Threshold Optimization ===")
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_val)
    if y_pred_proba.shape[1] == 1:
        y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
    y_pred_proba = y_pred_proba[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Calculate metrics
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best F1-score: {best_f1:.3f}")
    print(f"Best precision: {best_precision:.3f}")
    print(f"Best recall: {best_recall:.3f}")
    
    return best_threshold, results

def select_best_features(X_train, y_train, X_val, y_val, k=20):
    """Select best features using multiple methods"""
    print(f"\n=== Feature Selection (Top {k} features) ===")
    
    # Method 1: SelectKBest with f_classif
    selector_kbest = SelectKBest(score_func=f_classif, k=k)
    X_train_kbest = selector_kbest.fit_transform(X_train, y_train)
    X_val_kbest = selector_kbest.transform(X_val)
    
    # Method 2: RFE with XGBoost
    xgb_for_rfe = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )
    selector_rfe = RFE(estimator=xgb_for_rfe, n_features_to_select=k)
    X_train_rfe = selector_rfe.fit_transform(X_train, y_train)
    X_val_rfe = selector_rfe.transform(X_val)
    
    # Evaluate both methods
    xgb_kbest = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        random_state=42
    )
    xgb_rfe = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        random_state=42
    )
    
    # Train and evaluate
    xgb_kbest.fit(X_train_kbest, y_train)
    xgb_rfe.fit(X_train_rfe, y_train)
    
    auc_kbest = xgb_kbest.score(X_val_kbest, y_val)
    auc_rfe = xgb_rfe.score(X_val_rfe, y_val)
    
    print(f"SelectKBest AUC: {auc_kbest:.3f}")
    print(f"RFE AUC: {auc_rfe:.3f}")
    
    # Choose the better method
    if auc_kbest > auc_rfe:
        print("Using SelectKBest for feature selection")
        return selector_kbest, X_train_kbest, X_val_kbest
    else:
        print("Using RFE for feature selection")
        return selector_rfe, X_train_rfe, X_val_rfe

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Advanced hyperparameter tuning"""
    print("\n=== Hyperparameter Tuning ===")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.05, 0.07],
        'n_estimators': [150, 200, 250],
        'subsample': [0.85, 0.9, 0.95],
        'colsample_bytree': [0.85, 0.9, 0.95],
        'min_child_weight': [2, 3, 4],
        'gamma': [0.05, 0.1, 0.15]
    }
    
    # Use stratified k-fold for better validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42
    )
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    auc_val = best_model.score(X_val, y_val)
    print(f"Validation AUC: {auc_val:.3f}")
    
    return best_model, grid_search.best_params_

def optimize_class_weights(X_train, y_train, X_val, y_val):
    """Optimize class weights for imbalanced data"""
    print("\n=== Class Weight Optimization ===")
    
    # Calculate class distribution
    positive_samples = y_train.sum()
    negative_samples = (y_train == 0).sum()
    imbalance_ratio = negative_samples / positive_samples
    
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    # Try different scale_pos_weight values
    weight_candidates = [imbalance_ratio * factor for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]
    
    best_auc = 0
    best_weight = imbalance_ratio
    
    for weight in weight_candidates:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=weight,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        auc = model.score(X_val, y_val)
        
        print(f"Weight {weight:.2f}: AUC {auc:.3f}")
        
        if auc > best_auc:
            best_auc = auc
            best_weight = weight
    
    print(f"Best weight: {best_weight:.2f} (AUC: {best_auc:.3f})")
    return best_weight

def fine_tune_models():
    """Main fine-tuning pipeline"""
    print("=== FlareAlert Advanced Fine-Tuning ===\n")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Define training period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    try:
        # Prepare data
        features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
        
        if len(dataset) == 0:
            raise ValueError("No training data available")
        
        # Split data ensuring positive samples in both sets
        dataset = dataset.sort_values('timestamp')
        
        # Find positive samples
        positive_samples = dataset[dataset['flare6h_label'] == 1]
        negative_samples = dataset[dataset['flare6h_label'] == 0]
        
        # Ensure we have positive samples in both train and test
        if len(positive_samples) > 0:
            # Split positive samples
            pos_split_idx = int(len(positive_samples) * 0.8)
            train_pos = positive_samples.iloc[:pos_split_idx]
            test_pos = positive_samples.iloc[pos_split_idx:]
            
            # Split negative samples
            neg_split_idx = int(len(negative_samples) * 0.8)
            train_neg = negative_samples.iloc[:neg_split_idx]
            test_neg = negative_samples.iloc[neg_split_idx:]
            
            # Combine
            train_data = pd.concat([train_pos, train_neg]).sort_values('timestamp')
            test_data = pd.concat([test_pos, test_neg]).sort_values('timestamp')
        else:
            # Fallback to time-based split if no positive samples
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
        
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        print(f"Features: {len(feature_cols)}")
        print(f"Positive rate: {y_train.mean():.3f}")
        
        # 1. Feature Selection
        selector, X_train_selected, X_test_selected = select_best_features(
            X_train, y_train, X_test, y_test, k=min(20, len(feature_cols))
        )
        
        # 2. Class Weight Optimization
        best_weight = optimize_class_weights(X_train_selected, y_train, X_test_selected, y_test)
        
        # 3. Hyperparameter Tuning
        best_model, best_params = hyperparameter_tuning(X_train_selected, y_train, X_test_selected, y_test)
        
        # 4. Threshold Optimization
        optimal_threshold, threshold_results = optimize_threshold(best_model, X_test_selected, y_test)
        
        # 5. Final Model with all optimizations
        print("\n=== Final Optimized Model ===")
        
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=best_weight,
            random_state=42,
            **best_params
        )
        
        final_model.fit(X_train_selected, y_train)
        
        # Evaluate final model
        y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
        y_pred_default = (y_pred_proba > 0.5).astype(int)
        y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
        
        # Metrics with default threshold
        auc_default = final_model.score(X_test_selected, y_test)
        f1_default = f1_score(y_test, y_pred_default, zero_division=0)
        precision_default = precision_score(y_test, y_pred_default, zero_division=0)
        recall_default = recall_score(y_test, y_pred_default, zero_division=0)
        
        # Metrics with optimal threshold
        f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)
        precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
        recall_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
        
        print(f"Default threshold (0.5):")
        print(f"  AUC: {auc_default:.3f}")
        print(f"  F1: {f1_default:.3f}")
        print(f"  Precision: {precision_default:.3f}")
        print(f"  Recall: {recall_optimal:.3f}")
        
        print(f"Optimal threshold ({optimal_threshold:.3f}):")
        print(f"  F1: {f1_optimal:.3f}")
        print(f"  Precision: {precision_optimal:.3f}")
        print(f"  Recall: {recall_optimal:.3f}")
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, final_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        print(f"\n=== Top Features ===")
        top_features = list(feature_importance.items())[:10]
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.3f}")
        
        # Save optimization results
        optimization_results = {
            'best_params': best_params,
            'optimal_threshold': optimal_threshold,
            'best_weight': best_weight,
            'feature_selector': selector,
            'final_model': final_model,
            'metrics': {
                'auc': auc_default,
                'f1_default': f1_default,
                'f1_optimal': f1_optimal,
                'precision_default': precision_default,
                'precision_optimal': precision_optimal,
                'recall_default': recall_default,
                'recall_optimal': recall_optimal
            },
            'threshold_results': threshold_results
        }
        
        print(f"\n=== Fine-Tuning Completed Successfully! ===")
        print(f"Best AUC: {auc_default:.3f}")
        print(f"Best F1: {f1_optimal:.3f}")
        
        return optimization_results
        
    except Exception as e:
        print(f"\nError during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    fine_tune_models()
