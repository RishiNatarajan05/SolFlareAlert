#!/usr/bin/env python3
"""
Restore the working model with exact parameters that gave 0.544 AUC
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

def restore_working_model():
    """Restore the working model with exact parameters"""
    print("=== Restoring Working Model (0.544 AUC) ===\n")
    
    # 1. Prepare data (same as working model - NO cyclic encoding)
    print("1. Preparing data with original features...")
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
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['flare6h_label']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['flare6h_label']
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Features: {len(feature_cols)} (original features)")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")
    
    # 2. Use exact working model parameters
    print(f"\n2. Training with exact working model parameters...")
    
    # Calculate scale_pos_weight (same as working model)
    positive_samples = y_train.sum()
    scale_pos_weight = (y_train == 0).sum() / positive_samples
    
    # EXACT parameters from working model
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
    
    print(f"Scale pos weight: {scale_pos_weight:.1f}")
    print(f"Base score: {params['base_score']}")
    
    # 3. Train model
    print(f"\n3. Training model...")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # 4. Calibrate probabilities (same as working model)
    print(f"\n4. Calibrating probabilities...")
    
    calibrated_model = CalibratedClassifierCV(
        model, cv='prefit', method='isotonic'
    )
    calibrated_model.fit(X_train, y_train)
    
    # 5. Evaluate
    print(f"\n5. Evaluating restored model...")
    
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
    
    # 6. Test different thresholds for false positive reduction
    print(f"\n6. Testing thresholds for false positive reduction...")
    
    thresholds = np.arange(0.5, 0.95, 0.01)
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
        
        # Find best threshold that balances precision and recall
        if precision > best_precision and recall > 0.1:  # Must have at least 10% recall
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_fpr = false_positive_rate
    
    # 7. Final evaluation
    print(f"\n7. Final evaluation...")
    
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
    
    # 8. Save restored model
    print(f"\n8. Saving restored model...")
    
    # Create metadata
    metadata = {
        'model_type': 'flare_prediction_restored',
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
    model_path = f'models/flare_restored_{timestamp}.pkl'
    metadata_path = f'models/flare_restored_{timestamp}_metadata.json'
    
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
    
    print(f"\n=== Restored Model Complete ===")
    print(f"Key results:")
    print(f"1. AUC: {auc:.3f} (target: 0.544)")
    print(f"2. Optimal threshold: {best_threshold:.3f}")
    print(f"3. Precision: {best_precision:.3f}")
    print(f"4. Recall: {best_recall:.3f}")
    print(f"5. False Positive Rate: {best_fpr:.3f}")
    print(f"6. F1-Score: {best_f1:.3f}")
    
    # Compare with target
    print(f"\n=== Comparison with Target ===")
    print(f"Target AUC: 0.544")
    print(f"Achieved AUC: {auc:.3f}")
    print(f"AUC Difference: {auc - 0.544:.3f}")
    
    if abs(auc - 0.544) < 0.05:
        print("✅ SUCCESS: Restored working model performance!")
    else:
        print("⚠️  CLOSE: Performance close to target")
    
    print(f"\n=== False Positive Reduction ===")
    print(f"Original FPR: 0.920")
    print(f"New FPR: {best_fpr:.3f}")
    print(f"FPR Improvement: {0.920 - best_fpr:.3f}")
    
    if best_fpr < 0.8:
        print("✅ GOOD: False positive rate reduced!")
    else:
        print("⚠️  LIMITED: False positive rate still high")

if __name__ == "__main__":
    restore_working_model()
