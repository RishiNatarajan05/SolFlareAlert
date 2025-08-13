#!/usr/bin/env python3
"""
Test script for feature engineering and label generation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
from data.features import FeatureEngineer
from data.labels import LabelGenerator
import pandas as pd

def test_feature_engineering():
    """Test feature engineering"""
    print("=== Testing Feature Engineering ===")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Test with current time
    current_time = datetime.now()
    
    # Get features
    features = fe.get_all_features(current_time)
    
    print(f"Generated {len(features)} features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    return features

def test_label_generation():
    """Test label generation"""
    print("\n=== Testing Label Generation ===")
    
    # Initialize label generator
    lg = LabelGenerator()
    
    # Test with a recent time window
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)  # Last 7 days
    
    print(f"Generating labels from {start_time} to {end_time}")
    
    # Generate labels
    labels = lg.generate_all_labels(start_time, end_time)
    
    print(f"Generated {len(labels)} label samples")
    print(f"Columns: {list(labels.columns)}")
    
    # Get statistics
    stats = lg.get_label_statistics(labels)
    
    print("\nLabel Statistics:")
    for target, target_stats in stats.items():
        print(f"\n{target}:")
        for key, value in target_stats.items():
            print(f"  {key}: {value}")
    
    # Validate labels
    validation = lg.validate_labels(labels)
    print(f"\nLabel Validation: {'PASS' if validation['is_valid'] else 'FAIL'}")
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    return labels

def test_integration():
    """Test feature + label integration"""
    print("\n=== Testing Feature + Label Integration ===")
    
    fe = FeatureEngineer()
    lg = LabelGenerator()
    
    # Use a recent time window
    end_time = datetime.now()
    start_time = end_time - timedelta(days=3)  # Last 3 days
    
    # Generate labels
    labels = lg.generate_all_labels(start_time, end_time)
    
    # Generate features for each timestamp
    feature_data = []
    
    for _, row in labels.iterrows():
        timestamp = row['timestamp']
        features = fe.get_all_features(timestamp)
        features['timestamp'] = timestamp
        features['flare6h_label'] = row['flare6h_label']
        features['kp12h_label'] = row['kp12h_label']
        feature_data.append(features)
    
    # Create combined dataset
    dataset = pd.DataFrame(feature_data)
    
    print(f"Created dataset with {len(dataset)} samples and {len(dataset.columns)} columns")
    print(f"Feature columns: {[col for col in dataset.columns if not col.endswith('_label') and col != 'timestamp']}")
    print(f"Label columns: {[col for col in dataset.columns if col.endswith('_label')]}")
    
    # Show sample data
    print("\nSample data:")
    print(dataset.head())
    
    return dataset

if __name__ == "__main__":
    print("FlareAlert Feature Engineering & Label Generation Test\n")
    
    try:
        # Test feature engineering
        features = test_feature_engineering()
        
        # Test label generation
        labels = test_label_generation()
        
        # Test integration
        dataset = test_integration()
        
        print("\n=== All Tests Completed Successfully! ===")
        print("Ready for machine learning model training!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

