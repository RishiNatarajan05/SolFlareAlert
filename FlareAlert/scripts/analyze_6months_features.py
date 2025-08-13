#!/usr/bin/env python3
"""
Analyze 6-month model features to identify discrimination issues
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from models.trainer import ModelTrainer

def analyze_6months_features():
    """Analyze feature quality for 6-month model"""
    print("=== 6-Month Feature Analysis ===\n")
    
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
    
    # 1. Mutual Information Analysis
    print("\n=== Mutual Information Analysis ===")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("Top 15 features by mutual information:")
    for i, row in mi_df.head(15).iterrows():
        print(f"  {row['feature']}: {row['mutual_info']:.4f}")
    
    # 2. F-statistic Analysis
    print("\n=== F-Statistic Analysis ===")
    f_scores, p_values = f_classif(X_train, y_train)
    f_df = pd.DataFrame({
        'feature': feature_cols,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values('f_score', ascending=False)
    
    print("Top 15 features by F-score:")
    for i, row in f_df.head(15).iterrows():
        print(f"  {row['feature']}: F={row['f_score']:.2f}, p={row['p_value']:.4f}")
    
    # 3. Feature Distribution Analysis
    print("\n=== Feature Distribution Analysis ===")
    
    # Check for features with low variance
    feature_vars = X_train.var()
    low_var_features = feature_vars[feature_vars < 0.01]
    
    print(f"Features with low variance (< 0.01): {len(low_var_features)}")
    for feature in low_var_features.index:
        print(f"  {feature}: var={feature_vars[feature]:.6f}")
    
    # 4. Correlation Analysis
    print("\n=== Correlation Analysis ===")
    
    # Check for highly correlated features
    corr_matrix = X_train.corr().abs()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    print(f"Highly correlated feature pairs (r > 0.95): {len(high_corr_pairs)}")
    for feat1, feat2, corr in high_corr_pairs[:10]:  # Show top 10
        print(f"  {feat1} <-> {feat2}: r={corr:.3f}")
    
    # 5. Feature Importance from Random Forest
    print("\n=== Random Forest Feature Importance ===")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 features by Random Forest importance:")
    for i, row in rf_importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 6. Summary of problematic features
    print("\n=== Feature Quality Summary ===")
    
    # Combine all analyses
    feature_analysis = pd.DataFrame({
        'feature': feature_cols,
        'mutual_info': mi_scores,
        'f_score': f_scores,
        'rf_importance': rf.feature_importances_,
        'variance': feature_vars
    })
    
    # Identify poor features
    poor_features = feature_analysis[
        (feature_analysis['mutual_info'] < 0.001) &
        (feature_analysis['f_score'] < 1.0) &
        (feature_analysis['rf_importance'] < 0.01) &
        (feature_analysis['variance'] < 0.01)
    ]
    
    print(f"Potentially poor features: {len(poor_features)}")
    for _, row in poor_features.iterrows():
        print(f"  {row['feature']}: MI={row['mutual_info']:.4f}, F={row['f_score']:.2f}, RF={row['rf_importance']:.4f}, Var={row['variance']:.6f}")
    
    return feature_analysis, mi_df, f_df, rf_importance

if __name__ == "__main__":
    analyze_6months_features()
