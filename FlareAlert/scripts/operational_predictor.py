#!/usr/bin/env python3
"""
Operational Solar Flare Predictor using Hazard Ensemble Model
Real-time prediction system with operational smoothing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data.features import FeatureEngineer
from hazard_ensemble_model import HazardEnsembleModel, apply_operational_smoothing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OperationalPredictor:
    """Operational solar flare prediction system"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.alert_history = []  # Track recent alerts for smoothing
        self.last_alert_time = None
        self.consecutive_alerts = 0
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained hazard ensemble model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            
            # Load metadata if available
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Model metadata loaded")
            else:
                self.metadata = {}
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def get_current_features(self, timestamp: datetime = None) -> Dict:
        """Get current features for prediction"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get all features
        features = self.feature_engineer.get_all_features(timestamp)
        features['timestamp'] = timestamp
        
        return features
    
    def predict_current_risk(self, timestamp: datetime = None) -> Dict:
        """Make current risk prediction"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get current features
        features = self.get_current_features(timestamp)
        
        # Prepare features for prediction
        feature_df = pd.DataFrame([features])
        X = feature_df[self.model.feature_cols]
        
        # Make prediction
        risk_probability = self.model.predict(X)[0]
        risk_level = self.model.predict_with_threshold(X)[0]
        
        # Determine risk category
        if risk_probability >= 0.7:
            risk_category = "HIGH"
        elif risk_probability >= 0.4:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"
        
        return {
            'timestamp': timestamp,
            'risk_probability': risk_probability,
            'risk_level': risk_level,
            'risk_category': risk_category,
            'features': features
        }
    
    def should_issue_alert(self, prediction: Dict, 
                          hysteresis_hours: int = 2, 
                          cooldown_hours: int = 6) -> bool:
        """Determine if an alert should be issued based on operational rules"""
        
        timestamp = prediction['timestamp']
        risk_level = prediction['risk_level']
        
        # If no risk detected, no alert
        if risk_level == 0:
            self.consecutive_alerts = 0
            return False
        
        # Hysteresis: require consecutive hours over threshold
        self.consecutive_alerts += 1
        if self.consecutive_alerts < hysteresis_hours:
            return False
        
        # Cooldown: suppress repeat alerts
        if self.last_alert_time is not None:
            hours_since_last = (timestamp - self.last_alert_time).total_seconds() / 3600
            if hours_since_last < cooldown_hours:
                return False
        
        # Issue alert
        self.last_alert_time = timestamp
        return True
    
    def generate_alert_message(self, prediction: Dict) -> str:
        """Generate human-readable alert message"""
        timestamp = prediction['timestamp']
        risk_prob = prediction['risk_probability']
        risk_cat = prediction['risk_category']
        
        # Get key features for context
        features = prediction['features']
        
        # Recent flare activity
        recent_flares = features.get('flare_count_6h', 0)
        m_plus_flares = features.get('m_plus_count_6h', 0)
        hours_since_mx = features.get('hours_since_last_mx', 72)
        
        # CME activity
        recent_cmes = features.get('cme_count_24h', 0)
        max_cme_speed = features.get('max_cme_speed', 0)
        
        # Geomagnetic activity
        current_kp = features.get('current_kp', 0)
        
        message = f"""
🚨 SOLAR FLARE ALERT 🚨
Time: {timestamp.strftime('%Y-%m-%d %H:%M UTC')}
Risk Level: {risk_cat} ({risk_prob:.1%})

RECENT ACTIVITY:
• Flares (6h): {recent_flares} total, {m_plus_flares} M/X class
• Hours since last M/X flare: {hours_since_mx:.1f}
• CMEs (24h): {recent_cmes}, Max speed: {max_cme_speed:.0f} km/s
• Current Kp index: {current_kp:.1f}

PREDICTION:
• Probability of M1.0+ flare in next 6 hours: {risk_prob:.1%}
• Risk Category: {risk_cat}

RECOMMENDATIONS:
• Monitor solar activity closely
• Check for active regions on solar disk
• Prepare for potential geomagnetic impacts
        """.strip()
        
        return message
    
    def run_continuous_monitoring(self, check_interval_minutes: int = 60):
        """Run continuous monitoring with periodic predictions"""
        print("=== Starting Continuous Solar Flare Monitoring ===")
        print(f"Check interval: {check_interval_minutes} minutes")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Get current prediction
                prediction = self.predict_current_risk()
                
                # Check if alert should be issued
                should_alert = self.should_issue_alert(prediction)
                
                # Display current status
                timestamp = prediction['timestamp']
                risk_prob = prediction['risk_probability']
                risk_cat = prediction['risk_category']
                
                status_msg = f"[{timestamp.strftime('%H:%M:%S')}] Risk: {risk_cat} ({risk_prob:.1%})"
                
                if should_alert:
                    print("🚨 " + status_msg + " - ALERT ISSUED!")
                    alert_message = self.generate_alert_message(prediction)
                    print(alert_message)
                    print("-" * 50)
                else:
                    print("📊 " + status_msg)
                
                # Wait for next check
                import time
                time.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n=== Monitoring stopped ===")
    
    def analyze_historical_performance(self, start_time: datetime, end_time: datetime) -> Dict:
        """Analyze model performance on historical data"""
        print(f"=== Analyzing Historical Performance ===")
        print(f"Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Generate timestamps for analysis
        timestamps = []
        current = start_time
        while current <= end_time:
            timestamps.append(current)
            current += timedelta(hours=1)
        
        predictions = []
        alerts_issued = 0
        
        for timestamp in timestamps:
            try:
                prediction = self.predict_current_risk(timestamp)
                should_alert = self.should_issue_alert(prediction)
                
                predictions.append({
                    'timestamp': timestamp,
                    'risk_probability': prediction['risk_probability'],
                    'risk_level': prediction['risk_level'],
                    'alert_issued': should_alert
                })
                
                if should_alert:
                    alerts_issued += 1
                    
            except Exception as e:
                print(f"Error predicting for {timestamp}: {e}")
                continue
        
        # Calculate statistics
        risk_probs = [p['risk_probability'] for p in predictions]
        alerts = [p['alert_issued'] for p in predictions]
        
        stats = {
            'total_predictions': len(predictions),
            'alerts_issued': alerts_issued,
            'alert_rate': alerts_issued / len(predictions) if predictions else 0,
            'avg_risk_probability': np.mean(risk_probs),
            'max_risk_probability': np.max(risk_probs),
            'min_risk_probability': np.min(risk_probs),
            'high_risk_periods': sum(1 for p in risk_probs if p >= 0.7),
            'medium_risk_periods': sum(1 for p in risk_probs if 0.4 <= p < 0.7),
            'low_risk_periods': sum(1 for p in risk_probs if p < 0.4)
        }
        
        print(f"\nPerformance Summary:")
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Alerts issued: {stats['alerts_issued']}")
        print(f"Alert rate: {stats['alert_rate']:.2%}")
        print(f"Average risk probability: {stats['avg_risk_probability']:.3f}")
        print(f"High risk periods: {stats['high_risk_periods']}")
        print(f"Medium risk periods: {stats['medium_risk_periods']}")
        print(f"Low risk periods: {stats['low_risk_periods']}")
        
        return stats

def main():
    """Main function for operational prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Operational Solar Flare Predictor')
    parser.add_argument('--model', type=str, help='Path to trained model file')
    parser.add_argument('--mode', choices=['single', 'continuous', 'analyze'], 
                       default='single', help='Prediction mode')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Check interval in minutes (for continuous mode)')
    parser.add_argument('--start-date', type=str, help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for analysis (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = OperationalPredictor()
    
    if args.model:
        predictor.load_model(args.model)
    else:
        # Try to find the latest model
        import glob
        model_files = glob.glob('models/hazard_ensemble_*.pkl')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            predictor.load_model(latest_model)
        else:
            print("No model found. Please specify a model path or train a model first.")
            return
    
    if args.mode == 'single':
        # Single prediction
        prediction = predictor.predict_current_risk()
        print("=== Current Solar Flare Risk Assessment ===")
        print(f"Time: {prediction['timestamp']}")
        print(f"Risk Probability: {prediction['risk_probability']:.1%}")
        print(f"Risk Category: {prediction['risk_category']}")
        print(f"Alert Level: {'YES' if prediction['risk_level'] else 'NO'}")
        
        if prediction['risk_level']:
            print("\n" + predictor.generate_alert_message(prediction))
    
    elif args.mode == 'continuous':
        # Continuous monitoring
        predictor.run_continuous_monitoring(args.interval)
    
    elif args.mode == 'analyze':
        # Historical analysis
        if not args.start_date or not args.end_date:
            print("Please provide --start-date and --end-date for analysis mode")
            return
        
        start_time = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_time = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        stats = predictor.analyze_historical_performance(start_time, end_time)

if __name__ == "__main__":
    main()
