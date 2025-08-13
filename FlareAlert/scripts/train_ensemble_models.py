#!/usr/bin/env python3
"""
Script to train ensemble solar weather prediction models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import logging
from models.trainer import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_ensemble_models():
    """Train the ensemble solar weather prediction models"""
    print("=== FlareAlert Ensemble Model Training ===\n")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Define training period (last 30 days - proven to work well)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    try:
        # Train ensemble models
        summary = trainer.train_all_models(start_time, end_time)
        
        # Print results
        print("\n=== Training Results ===")
        print(f"Total samples: {summary['data_summary']['total_samples']}")
        print(f"Training samples: {summary['data_summary']['train_samples']}")
        print(f"Test samples: {summary['data_summary']['test_samples']}")
        print(f"Flare positive rate: {summary['data_summary']['flare_positive_rate']:.3f}")
        print(f"Kp positive rate: {summary['data_summary']['kp_positive_rate']:.3f}")
        
        # Model performance
        print("\n=== Ensemble Model Performance ===")
        
        flare_eval = summary['models']['flare']['evaluation']
        print(f"Flare Ensemble AUC: {flare_eval['auc']:.3f}")
        if '1' in flare_eval['classification_report']:
            flare_precision = flare_eval['classification_report']['1']['precision']
            flare_recall = flare_eval['classification_report']['1']['recall']
            flare_f1 = flare_eval['classification_report']['1']['f1-score']
            print(f"Flare Ensemble Precision: {flare_precision:.3f}")
            print(f"Flare Ensemble Recall: {flare_recall:.3f}")
            print(f"Flare Ensemble F1-Score: {flare_f1:.3f}")
        
        kp_eval = summary['models']['kp']['evaluation']
        print(f"Kp Ensemble AUC: {kp_eval['auc']:.3f}")
        if '1' in kp_eval['classification_report']:
            kp_precision = kp_eval['classification_report']['1']['precision']
            kp_recall = kp_eval['classification_report']['1']['recall']
            kp_f1 = kp_eval['classification_report']['1']['f1-score']
            print(f"Kp Ensemble Precision: {kp_precision:.3f}")
            print(f"Kp Ensemble Recall: {kp_recall:.3f}")
            print(f"Kp Ensemble F1-Score: {kp_f1:.3f}")
        
        # Feature importance
        print("\n=== Top Features (Flare Ensemble) ===")
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
        
        # Cross-validation scores
        if 'cv_auc_mean' in flare_metadata:
            print(f"\nFlare Ensemble CV AUC: {flare_metadata['cv_auc_mean']:.3f} ± {flare_metadata['cv_auc_std']:.3f}")
        
        # Ensemble details
        if 'ensemble_params' in flare_metadata:
            print(f"\nEnsemble Configuration:")
            print(f"  Voting: {flare_metadata['ensemble_params']['voting']}")
            print(f"  Models: {', '.join(flare_metadata['ensemble_params']['models'])}")
            print(f"  Weights: {flare_metadata['ensemble_params']['weights']}")
        
        print("\n=== Model Files Saved ===")
        print(f"Flare ensemble: {summary['models']['flare']['model_path']}")
        print(f"Kp ensemble: {summary['models']['kp']['model_path']}")
        print(f"Training summary: {summary['models']['flare']['metadata_path']}")
        
        print("\n=== Ensemble Training Completed Successfully! ===")
        print("Ensemble models are ready for deployment!")
        
        return summary
        
    except Exception as e:
        print(f"\nError during ensemble training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_ensemble_models()
