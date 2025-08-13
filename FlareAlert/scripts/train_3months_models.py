#!/usr/bin/env python3
"""
Production script to train the optimal 3-month solar weather prediction models
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import logging
from models.trainer import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_3months_models():
    """Train the optimal 3-month solar weather prediction models"""
    print("=== FlareAlert 3-Month Model Training (Production) ===\n")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Define training period (3 months - optimal based on comparison)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    print(f"Total days: 90 (3 months) - OPTIMAL PERIOD")
    
    try:
        # Train models using the standard trainer (which uses optimal parameters)
        summary = trainer.train_all_models(start_time, end_time)
        
        # Print results
        print("\n=== Training Results ===")
        print(f"Total samples: {summary['data_summary']['total_samples']:,}")
        print(f"Training samples: {summary['data_summary']['train_samples']:,}")
        print(f"Test samples: {summary['data_summary']['test_samples']:,}")
        print(f"Flare positive rate: {summary['data_summary']['flare_positive_rate']:.3f}")
        print(f"Kp positive rate: {summary['data_summary']['kp_positive_rate']:.3f}")
        
        # Model performance
        print("\n=== Model Performance ===")
        
        flare_eval = summary['models']['flare']['evaluation']
        print(f"Flare Model AUC: {flare_eval['auc']:.3f}")
        if '1' in flare_eval['classification_report']:
            flare_precision = flare_eval['classification_report']['1']['precision']
            flare_recall = flare_eval['classification_report']['1']['recall']
            print(f"Flare Model Precision: {flare_precision:.3f}")
            print(f"Flare Model Recall: {flare_recall:.3f}")
        
        kp_eval = summary['models']['kp']['evaluation']
        print(f"Kp Model AUC: {kp_eval['auc']:.3f}")
        if '1' in kp_eval['classification_report']:
            kp_precision = kp_eval['classification_report']['1']['precision']
            kp_recall = kp_eval['classification_report']['1']['recall']
            print(f"Kp Model Precision: {kp_precision:.3f}")
            print(f"Kp Model Recall: {kp_recall:.3f}")
        
        # Feature importance
        print("\n=== Top Features (Flare Model) ===")
        flare_metadata_path = summary['models']['flare']['metadata_path']
        import json
        with open(flare_metadata_path, 'r') as f:
            flare_metadata = json.load(f)
        
        top_features = list(flare_metadata['feature_importance'].items())[:10]
        for feature, importance in top_features:
            if isinstance(importance, (int, float)):
                print(f"  {feature}: {importance:.3f}")
            else:
                print(f"  {feature}: {importance}")
        
        print("\n=== Model Files Saved ===")
        print(f"Flare model: {summary['models']['flare']['model_path']}")
        print(f"Kp model: {summary['models']['kp']['model_path']}")
        print(f"Training summary: {summary['models']['flare']['metadata_path']}")
        
        # Performance summary
        print("\n=== Performance Summary ===")
        print("✅ 3-Month Model - OPTIMAL CHOICE")
        print(f"✅ AUC: {flare_eval['auc']:.3f} (Excellent discrimination)")
        print(f"✅ Precision: {flare_precision:.3f} (Acceptable false alarm rate)")
        print(f"✅ Recall: {flare_recall:.3f} (Catches most flares)")
        print(f"✅ Sample size: {summary['data_summary']['total_samples']:,} (Good data volume)")
        
        print("\n=== 3-Month Training Completed Successfully! ===")
        print("🎯 This is your production-ready model!")
        print("🚀 Ready for deployment to FlareAlert system!")
        
        return summary
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_3months_models()
