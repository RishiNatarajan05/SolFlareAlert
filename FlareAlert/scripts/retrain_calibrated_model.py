#!/usr/bin/env python3
"""
Retrain model with better calibration parameters
Implement ML expert recommendations for fixing overconfidence
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

def retrain_calibrated_model():
    """Retrain model with better calibration parameters"""
    print("=== Retraining with Better Calibration ===\n")
    
    # 1. Prepare data (same as before)
    print("1. Preparing data...")
    trainer = ModelTrainer()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
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
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")
    
    # 2. Calculate new scale_pos_weight (reduced from 6.2 to 3.5)
    print(f"\n2. Calculating new parameters...")
    
    # Use more conservative scale_pos_weight
    scale_pos_weight = 3.5  # Reduced from 6.2
    
    # 3. New calibrated parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.95,
        'colsample_bytree': 0.95,
        'min_child_weight': 3,
        'gamma': 0.05,  # Reduced from 0.1
        'scale_pos_weight': scale_pos_weight,  # Reduced from 6.2
        'base_score': 0.137,  # Set to prevalence
        'max_delta_step': 3,  # NEW: Stabilizes logistic updates
        'reg_lambda': 2.0,  # NEW: L2 regularization
        'reg_alpha': 0.0,  # NEW: L1 regularization (keep at 0)
        'random_state': 42
    }
    
    print(f"Scale pos weight: {scale_pos_weight:.1f} (reduced from 6.2)")
    print(f"Base score: {params['base_score']} (set to prevalence)")
    print(f"Max delta step: {params['max_delta_step']} (NEW)")
    print(f"Reg lambda: {params['reg_lambda']} (NEW)")
    print(f"Gamma: {params['gamma']} (reduced from 0.1)")
    
    # 4. Train model with new parameters
    print(f"\n3. Training model with calibrated parameters...")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # 5. Evaluate raw model
    print(f"\n4. Evaluating raw model...")
    
    y_pred_proba_raw = model.predict_proba(X_test)[:, 1]
    
    print(f"Raw probability range: {y_pred_proba_raw.min():.3f} - {y_pred_proba_raw.max():.3f}")
    print(f"Raw mean probability: {y_pred_proba_raw.mean():.3f}")
    print(f"Raw std probability: {y_pred_proba_raw.std():.3f}")
    
    # Calculate metrics
    auc_raw = roc_auc_score(y_test, y_pred_proba_raw)
    pr_auc_raw = average_precision_score(y_test, y_pred_proba_raw)
    brier_raw = brier_score_loss(y_test, y_pred_proba_raw)
    
    print(f"Raw ROC AUC: {auc_raw:.3f}")
    print(f"Raw PR AUC: {pr_auc_raw:.3f}")
    print(f"Raw Brier Score: {brier_raw:.3f}")
    
    # 6. Calibrate probabilities
    print(f"\n5. Calibrating probabilities...")
    
    calibrated_model = CalibratedClassifierCV(
        model, cv='prefit', method='isotonic'
    )
    calibrated_model.fit(X_train, y_train)
    
    # 7. Evaluate calibrated model
    print(f"\n6. Evaluating calibrated model...")
    
    y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    
    print(f"Calibrated probability range: {y_pred_proba_cal.min():.3f} - {y_pred_proba_cal.max():.3f}")
    print(f"Calibrated mean probability: {y_pred_proba_cal.mean():.3f}")
    print(f"Calibrated std probability: {y_pred_proba_cal.std():.3f}")
    
    # Calculate metrics
    auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
    pr_auc_cal = average_precision_score(y_test, y_pred_proba_cal)
    brier_cal = brier_score_loss(y_test, y_pred_proba_cal)
    
    print(f"Calibrated ROC AUC: {auc_cal:.3f}")
    print(f"Calibrated PR AUC: {pr_auc_cal:.3f}")
    print(f"Calibrated Brier Score: {brier_cal:.3f}")
    
    # 8. Test thresholds for precision optimization
    print(f"\n7. Testing thresholds for precision optimization...")
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_fpr = 1.0
    best_predictions = 0
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba_cal > threshold).astype(int)
        
        # Skip if no positive predictions
        if y_pred.sum() == 0:
            continue
            
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
        
        # Find best threshold that maximizes precision while keeping some recall
        if precision > best_precision and recall > 0.1:  # Must have at least 10% recall
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_fpr = false_positive_rate
            best_predictions = positive_predictions
    
    # 9. Final evaluation
    print(f"\n8. Final evaluation...")
    
    y_pred_optimal = (y_pred_proba_cal > best_threshold).astype(int)
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Precision: {best_precision:.3f}")
    print(f"Recall: {best_recall:.3f}")
    print(f"F1-Score: {best_f1:.3f}")
    print(f"False Positive Rate: {best_fpr:.3f}")
    print(f"Positive Predictions: {best_predictions}")
    
    # Show top 10 thresholds by precision
    print(f"\n=== Top 10 Thresholds by Precision ===")
    results_sorted = sorted(results, key=lambda x: x['precision'], reverse=True)
    for i, result in enumerate(results_sorted[:10]):
        print(f"{i+1}. Threshold: {result['threshold']:.3f}, Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}, FPR: {result['false_positive_rate']:.3f}, "
              f"Predictions: {result['positive_predictions']}")
    
    # 10. Save calibrated model
    print(f"\n9. Saving calibrated model...")
    
    # Create metadata
    metadata = {
        'model_type': 'flare_prediction_calibrated',
        'target': 'flare6h_label',
        'feature_columns': feature_cols,
        'training_samples': len(train_data),
        'test_samples': len(test_data),
        'positive_rate': y_train.mean(),
        'trained_at': datetime.now().isoformat(),
        'xgb_params': params,
        'performance': {
            'raw_auc': float(auc_raw),
            'raw_pr_auc': float(pr_auc_raw),
            'raw_brier': float(brier_raw),
            'calibrated_auc': float(auc_cal),
            'calibrated_pr_auc': float(pr_auc_cal),
            'calibrated_brier': float(brier_cal),
            'threshold': float(best_threshold),
            'precision': float(best_precision),
            'recall': float(best_recall),
            'f1': float(best_f1),
            'false_positive_rate': float(best_fpr),
            'positive_predictions': int(best_predictions)
        }
    }
    
    # Save model and metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/flare_calibrated_{timestamp}.pkl'
    metadata_path = f'models/flare_calibrated_{timestamp}_metadata.json'
    
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
    
    print(f"\n=== Calibrated Model Complete ===")
    print(f"Key results:")
    print(f"1. Raw AUC: {auc_raw:.3f}")
    print(f"2. Calibrated AUC: {auc_cal:.3f}")
    print(f"3. Optimal threshold: {best_threshold:.3f}")
    print(f"4. Precision: {best_precision:.3f}")
    print(f"5. Recall: {best_recall:.3f}")
    print(f"6. False Positive Rate: {best_fpr:.3f}")
    print(f"7. F1-Score: {best_f1:.3f}")
    print(f"8. Positive Predictions: {best_predictions}")
    
    # Compare with previous model
    print(f"\n=== Comparison with Previous Model ===")
    print(f"Previous AUC: 0.636")
    print(f"New AUC: {auc_cal:.3f}")
    print(f"AUC Difference: {auc_cal - 0.636:.3f}")
    print(f"Previous Precision: 0.179")
    print(f"New Precision: {best_precision:.3f}")
    print(f"Precision Difference: {best_precision - 0.179:.3f}")
    print(f"Previous FPR: 0.736")
    print(f"New FPR: {best_fpr:.3f}")
    print(f"FPR Difference: {best_fpr - 0.736:.3f}")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if auc_cal > 0.636:
        print("✅ EXCELLENT: AUC improved!")
    elif auc_cal > 0.6:
        print("✅ GOOD: AUC maintained!")
    else:
        print("⚠️  AUC decreased")
    
    if best_precision > 0.179:
        print("✅ EXCELLENT: Precision improved!")
    elif best_precision > 0.15:
        print("✅ GOOD: Precision maintained!")
    else:
        print("⚠️  Precision decreased")
    
    if best_fpr < 0.736:
        print("✅ EXCELLENT: False positive rate reduced!")
    elif best_fpr < 0.8:
        print("✅ GOOD: False positive rate maintained!")
    else:
        print("⚠️  False positive rate increased")
    
    print(f"\n=== Calibration Improvements ===")
    print(f"1. Reduced scale_pos_weight: 6.2 → {scale_pos_weight}")
    print(f"2. Added max_delta_step: {params['max_delta_step']}")
    print(f"3. Added reg_lambda: {params['reg_lambda']}")
    print(f"4. Reduced gamma: 0.1 → {params['gamma']}")
    print(f"5. Set base_score to prevalence: {params['base_score']}")

if __name__ == "__main__":
    retrain_calibrated_model()
