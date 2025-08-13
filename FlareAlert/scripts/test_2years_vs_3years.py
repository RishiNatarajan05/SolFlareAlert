#!/usr/bin/env python3
"""
Compare 2 years vs 3 years of training data to see if 2 years helps maintain precision
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, brier_score_loss
)
from models.trainer import ModelTrainer
import json

def test_training_periods():
    """Test different training periods to find optimal balance"""
    print("=== Testing 2 Years vs 3 Years Training Periods ===\n")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Test periods: 1 year, 2 years, 3 years
    periods = [
        (365, "1 Year"),
        (730, "2 Years"), 
        (1095, "3 Years")
    ]
    
    results = {}
    
    for days, period_name in periods:
        print(f"\n{'='*50}")
        print(f"Testing {period_name} ({days} days)")
        print(f"{'='*50}")
        
        # Define training period
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        print(f"Total days: {days}")
        
        try:
            # Prepare data
            features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
            
            if len(dataset) == 0:
                print(f"No data available for {period_name}")
                continue
            
            # Sort by timestamp
            dataset = dataset.sort_values('timestamp')
            
            print(f"Total samples: {len(dataset)}")
            print(f"Positive samples: {dataset['flare6h_label'].sum()}")
            print(f"Positive rate: {dataset['flare6h_label'].mean():.3f}")
            
            # Split data ensuring positive samples in both sets
            positive_samples = dataset[dataset['flare6h_label'] == 1]
            negative_samples = dataset[dataset['flare6h_label'] == 0]
            
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
            
            print(f"Training samples: {len(train_data)}")
            print(f"Test samples: {len(test_data)}")
            
            # Prepare features
            feature_cols = [col for col in dataset.columns 
                          if not col.endswith('_label') and col != 'timestamp']
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['flare6h_label']
            X_test = test_data[feature_cols].fillna(0)
            y_test = test_data['flare6h_label']
            
            # Train simple model for comparison (XGBoost)
            import xgboost as xgb
            from sklearn.calibration import CalibratedClassifierCV
            
            print("\nTraining XGBoost model...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=10,
                random_state=42,
                eval_metric='logloss'
            )
            
            # Calibrate the model
            calibrated_model = CalibratedClassifierCV(
                xgb_model, 
                method='isotonic', 
                cv=3
            )
            
            calibrated_model.fit(X_train, y_train)
            
            # Get predictions
            y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            brier = brier_score_loss(y_test, y_pred_proba)
            
            print(f"ROC AUC: {auc:.3f}")
            print(f"PR AUC: {pr_auc:.3f}")
            print(f"Brier Score: {brier:.3f}")
            
            # Test different thresholds
            print("\nTesting different thresholds...")
            thresholds = np.arange(0.1, 0.9, 0.05)
            threshold_results = []
            
            for threshold in thresholds:
                y_pred = (y_pred_proba > threshold).astype(int)
                
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Calculate false positive rate
                fp = ((y_pred == 1) & (y_test == 0)).sum()
                tn = ((y_pred == 0) & (y_test == 0)).sum()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                threshold_results.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'fpr': fpr,
                    'positive_predictions': y_pred.sum()
                })
            
            # Find best threshold for different objectives
            best_f1_idx = np.argmax([r['f1'] for r in threshold_results])
            best_precision_idx = np.argmax([r['precision'] for r in threshold_results if r['recall'] > 0.1])
            
            best_f1 = threshold_results[best_f1_idx]
            best_precision = threshold_results[best_precision_idx]
            
            print(f"\nBest F1 Score:")
            print(f"  Threshold: {best_f1['threshold']:.3f}")
            print(f"  F1: {best_f1['f1']:.3f}")
            print(f"  Precision: {best_f1['precision']:.3f}")
            print(f"  Recall: {best_f1['recall']:.3f}")
            print(f"  FPR: {best_f1['fpr']:.3f}")
            
            print(f"\nBest Precision (with >10% recall):")
            print(f"  Threshold: {best_precision['threshold']:.3f}")
            print(f"  F1: {best_precision['f1']:.3f}")
            print(f"  Precision: {best_precision['precision']:.3f}")
            print(f"  Recall: {best_precision['recall']:.3f}")
            print(f"  FPR: {best_precision['fpr']:.3f}")
            
            # Store results
            results[period_name] = {
                'days': days,
                'total_samples': len(dataset),
                'positive_samples': dataset['flare6h_label'].sum(),
                'positive_rate': dataset['flare6h_label'].mean(),
                'auc': auc,
                'pr_auc': pr_auc,
                'brier': brier,
                'best_f1': best_f1,
                'best_precision': best_precision,
                'all_thresholds': threshold_results
            }
            
        except Exception as e:
            print(f"Error testing {period_name}: {e}")
            continue
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    comparison_data = []
    for period_name, result in results.items():
        comparison_data.append({
            'Period': period_name,
            'Days': result['days'],
            'Samples': result['total_samples'],
            'Positive Rate': f"{result['positive_rate']:.3f}",
            'AUC': f"{result['auc']:.3f}",
            'PR-AUC': f"{result['pr_auc']:.3f}",
            'Best F1': f"{result['best_f1']['f1']:.3f}",
            'Best Precision': f"{result['best_precision']['precision']:.3f}",
            'Best Recall': f"{result['best_precision']['recall']:.3f}",
            'Best FPR': f"{result['best_precision']['fpr']:.3f}"
        })
    
    # Print comparison table
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Save detailed results
    with open('models/training_period_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: models/training_period_comparison.json")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        # Find best period for precision
        best_precision_period = max(results.items(), 
                                  key=lambda x: x[1]['best_precision']['precision'])
        
        # Find best period for F1
        best_f1_period = max(results.items(), 
                           key=lambda x: x[1]['best_f1']['f1'])
        
        print(f"Best for Precision: {best_precision_period[0]}")
        print(f"  Precision: {best_precision_period[1]['best_precision']['precision']:.3f}")
        print(f"  Recall: {best_precision_period[1]['best_precision']['recall']:.3f}")
        
        print(f"\nBest for F1 Score: {best_f1_period[0]}")
        print(f"  F1: {best_f1_period[1]['best_f1']['f1']:.3f}")
        print(f"  Precision: {best_f1_period[1]['best_f1']['precision']:.3f}")
        print(f"  Recall: {best_f1_period[1]['best_f1']['recall']:.3f}")
        
        # Check if 2 years is optimal
        if '2 Years' in results:
            two_year = results['2 Years']
            three_year = results.get('3 Years', {})
            
            if three_year:
                precision_improvement = (two_year['best_precision']['precision'] - 
                                       three_year['best_precision']['precision'])
                
                if precision_improvement > 0.01:  # 1% improvement
                    print(f"\n✅ 2 Years shows {precision_improvement:.3f} better precision than 3 Years")
                    print("   Consider using 2 years for better precision!")
                else:
                    print(f"\n❌ 2 Years doesn't significantly improve precision")
                    print("   Stick with 3 years for more data")

if __name__ == "__main__":
    test_training_periods()
