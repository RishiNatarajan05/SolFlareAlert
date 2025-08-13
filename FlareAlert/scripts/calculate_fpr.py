#!/usr/bin/env python3
"""
Calculate false positive rate for the latest model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from models.trainer import ModelTrainer

def calculate_fpr():
    """Calculate false positive rate for latest model"""
    print("=== False Positive Rate Analysis ===\n")
    
    # Load the latest model
    print("1. Loading latest model...")
    with open('models/flare_20250813_105759.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Prepare test data
    print("2. Preparing test data...")
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
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['flare6h_label']
    
    print(f"Test samples: {len(test_data)}")
    print(f"Test positive rate: {y_test.mean():.3f}")
    
    # Get predictions
    print("3. Getting model predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print(f"Probability range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    print(f"Std probability: {y_pred_proba.std():.3f}")
    
    # Calculate metrics
    print("\n4. Calculating metrics...")
    
    # Basic metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Calculate false positive rate
    true_negatives = ((y_pred == 0) & (y_test == 0)).sum()
    false_positives = ((y_pred == 1) & (y_test == 0)).sum()
    true_positives = ((y_pred == 1) & (y_test == 1)).sum()
    false_negatives = ((y_pred == 0) & (y_test == 1)).sum()
    
    total_negatives = (y_test == 0).sum()
    total_positives = (y_test == 1).sum()
    
    false_positive_rate = false_positives / total_negatives if total_negatives > 0 else 0
    true_positive_rate = true_positives / total_positives if total_positives > 0 else 0
    
    print(f"\n=== Confusion Matrix ===")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"True Positives: {true_positives}")
    print(f"False Negatives: {false_negatives}")
    
    print(f"\n=== Rates ===")
    print(f"False Positive Rate: {false_positive_rate:.3f} ({false_positive_rate*100:.1f}%)")
    print(f"True Positive Rate (Recall): {true_positive_rate:.3f} ({true_positive_rate*100:.1f}%)")
    
    print(f"\n=== Summary ===")
    print(f"Total test samples: {len(y_test)}")
    print(f"Actual positives: {total_positives}")
    print(f"Actual negatives: {total_negatives}")
    print(f"Predicted positives: {y_pred.sum()}")
    print(f"Predicted negatives: {(y_pred == 0).sum()}")
    
    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / len(y_test)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Operational impact
    print(f"\n=== Operational Impact ===")
    print(f"With current model:")
    print(f"- You'll get {y_pred.sum()} flare alerts")
    print(f"- {precision*100:.1f}% of alerts will be real flares")
    print(f"- {false_positive_rate*100:.1f}% of quiet periods will trigger false alarms")
    print(f"- You'll miss {false_negatives} actual flares out of {total_positives}")
    
    if false_positive_rate > 0.8:
        print(f"\n⚠️  HIGH FALSE POSITIVE RATE: {false_positive_rate*100:.1f}%")
        print("This means most quiet periods will trigger false alarms!")
    elif false_positive_rate > 0.5:
        print(f"\n⚠️  MODERATE FALSE POSITIVE RATE: {false_positive_rate*100:.1f}%")
        print("Many quiet periods will trigger false alarms.")
    else:
        print(f"\n✅ ACCEPTABLE FALSE POSITIVE RATE: {false_positive_rate*100:.1f}%")

if __name__ == "__main__":
    calculate_fpr()
