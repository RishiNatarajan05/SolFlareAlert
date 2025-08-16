#!/usr/bin/env python3
"""
Hazard Model + Calibrated XGBoost Ensemble for Solar Flare Prediction
Implements the recommended approach: discrete-time hazard baseline + calibrated tree model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, 
    average_precision_score, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from models.trainer import ModelTrainer
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HazardEnsembleModel:
    """Hazard model + calibrated XGBoost ensemble for solar flare prediction"""
    
    def __init__(self):
        self.hazard_model = None
        self.xgb_model = None
        self.calibrated_xgb = None
        self.scaler = StandardScaler()
        self.ensemble_weight = 1.0  # 100% hazard, 0% XGBoost (temporarily disable)
        self.optimal_threshold = 0.5
        self.feature_cols = None
        
    def create_cyclic_features(self, df):
        """Create cyclic time encodings"""
        # Create time features from timestamp if they don't exist
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.weekday
            df['month'] = df['timestamp'].dt.month
            df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Hour of day (0-23)
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        # Day of week (0-6)
        if 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (1-12)
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Drop raw time features (keep cyclic ones)
        df = df.drop(['hour_of_day', 'day_of_week', 'month', 'day_of_year'], axis=1, errors='ignore')
        
        return df
    
    def prepare_hazard_features(self, df):
        """Prepare features specifically for hazard model"""
        # Select features that work well for linear models
        hazard_features = [
            # Recent activity (most important for hazard)
            'flare_count_6h', 'flare_count_24h', 'm_plus_count_6h', 'm_plus_count_24h',
            'hours_since_last_mx', 'max_flare_class', 'max_flare_value',
            
            # CME activity
            'cme_count_24h', 'max_cme_speed', 'hours_since_last_cme',
            
            # Geomagnetic activity
            'current_kp', 'max_kp_24h', 'kp_storm_count_24h',
            
            # Cyclic time features
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
            'month_sin', 'month_cos',
            
            # Lag features
            'flare_count_7d', 'm_plus_count_7d', 'flare_activity_ratio', 'm_plus_activity_ratio'
        ]
        
        # Keep only features that exist in the dataset
        available_features = [f for f in hazard_features if f in df.columns]
        
        return df[available_features]
    
    def prepare_xgb_features(self, df):
        """Prepare features for XGBoost (can use all features)"""
        # Remove timestamp and label columns
        feature_cols = [col for col in df.columns 
                       if not col.endswith('_label') and col != 'timestamp']
        return df[feature_cols]
    
    def train_hazard_model(self, X_train, y_train, X_val, y_val):
        """Train discrete-time hazard model (logistic regression)"""
        print("=== Training Hazard Model ===")
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        
        # Scale features for linear model
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train logistic regression with L2 regularization
        self.hazard_model = LogisticRegression(
            C=1.0,  # L2 regularization strength
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        self.hazard_model.fit(X_train_scaled, y_train)
        
        # Evaluate hazard model
        y_pred_hazard = self.hazard_model.predict_proba(X_val_scaled)[:, 1]
        auc_hazard = roc_auc_score(y_val, y_pred_hazard)
        pr_auc_hazard = average_precision_score(y_val, y_pred_hazard)
        
        print(f"Hazard model AUC: {auc_hazard:.3f}")
        print(f"Hazard model PR-AUC: {pr_auc_hazard:.3f}")
        
        return y_pred_hazard
    
    def train_xgb_model(self, X_train, y_train, X_val, y_val):
        """Train several XGBoost variants and pick the best on validation"""
        print("=== Training Calibrated XGBoost Model ===")

        # Safety: fill missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)

        # Compute class weight
        positive_samples = int(y_train.sum())
        negative_samples = int((y_train == 0).sum())
        scale_pos_weight = (negative_samples / max(positive_samples, 1))

        # Candidate parameter sets (shallow, conservative)
        candidates = [
            {
                'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 400,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'min_child_weight': 1, 'gamma': 0.0,
                'scale_pos_weight': scale_pos_weight, 'max_delta_step': 1,
                'random_state': 42
            },
            {
                'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 500,
                'subsample': 0.9, 'colsample_bytree': 0.9,
                'min_child_weight': 2, 'gamma': 0.1,
                'scale_pos_weight': scale_pos_weight, 'max_delta_step': 2,
                'random_state': 42
            },
            {
                'max_depth': 2, 'learning_rate': 0.1, 'n_estimators': 500,
                'subsample': 0.9, 'colsample_bytree': 0.9,
                'min_child_weight': 1, 'gamma': 0.0,
                'scale_pos_weight': scale_pos_weight, 'max_delta_step': 1,
                'random_state': 42
            },
        ]

        best_auc = -1.0
        best_model = None
        best_proba = None
        best_desc = ""

        for idx, params in enumerate(candidates, start=1):
            # Uncalibrated model
            model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', **params)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_proba = proba
                best_desc = f"xgb{idx}-raw"

            # Platt (sigmoid) calibration
            try:
                calib = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                calib.fit(X_train, y_train)
                proba_c = calib.predict_proba(X_val)[:, 1]
                auc_c = roc_auc_score(y_val, proba_c)
                if auc_c > best_auc:
                    best_auc = auc_c
                    best_model = calib
                    best_proba = proba_c
                    best_desc = f"xgb{idx}-sigmoid"
            except Exception:
                pass

        self.xgb_model = best_model
        self.calibrated_xgb = best_model  # unify usage in predict()
        print(f"Calibrated XGBoost AUC: {best_auc:.3f} ({best_desc})")
        return best_proba
    
    def train_ensemble(self, dataset):
        """Train the complete ensemble"""
        print("=== Training Hazard + XGBoost Ensemble ===")
        
        # Prepare features and target
        feature_cols = [col for col in dataset.columns 
                       if not col.endswith('_label') and col != 'timestamp']
        self.feature_cols = feature_cols
        
        # Create cyclic time features
        dataset = self.create_cyclic_features(dataset.copy())
        
        # Prepare features for each model
        X_hazard = self.prepare_hazard_features(dataset)
        X_xgb = self.prepare_xgb_features(dataset)
        y = dataset['flare6h_label']
        
        # Handle missing values
        X_hazard = X_hazard.fillna(0)
        X_xgb = X_xgb.fillna(0)
        
        # Time-series split (preserve temporal order)
        # Ensure positive samples in both train and validation sets
        positive_samples = dataset[dataset['flare6h_label'] == 1]
        negative_samples = dataset[dataset['flare6h_label'] == 0]
        
        if len(positive_samples) > 0:
            # Split positive samples
            pos_split_idx = int(len(positive_samples) * 0.8)
            train_pos = positive_samples.iloc[:pos_split_idx]
            val_pos = positive_samples.iloc[pos_split_idx:]
            
            # Split negative samples
            neg_split_idx = int(len(negative_samples) * 0.8)
            train_neg = negative_samples.iloc[:neg_split_idx]
            val_neg = negative_samples.iloc[neg_split_idx:]
            
            # Combine and sort by timestamp
            train_data = pd.concat([train_pos, train_neg]).sort_values('timestamp')
            val_data = pd.concat([val_pos, val_neg]).sort_values('timestamp')
        else:
            # Fallback to time-based split if no positive samples
            split_idx = int(len(dataset) * 0.8)
            train_data = dataset.iloc[:split_idx]
            val_data = dataset.iloc[split_idx:]
        
        # Prepare features for train/validation
        X_hazard_train = self.prepare_hazard_features(self.create_cyclic_features(train_data.copy()))
        X_hazard_val = self.prepare_hazard_features(self.create_cyclic_features(val_data.copy()))
        X_xgb_train = self.prepare_xgb_features(self.create_cyclic_features(train_data.copy()))
        X_xgb_val = self.prepare_xgb_features(self.create_cyclic_features(val_data.copy()))
        y_train = train_data['flare6h_label']
        y_val = val_data['flare6h_label']
        
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Positive rate: {y_train.mean():.3f}")
        
        # Train both models
        y_pred_hazard = self.train_hazard_model(X_hazard_train, y_train, X_hazard_val, y_val)
        y_pred_xgb = self.train_xgb_model(X_xgb_train, y_train, X_xgb_val, y_val)

        # Evaluate hazard-only baseline
        auc_hazard_only = roc_auc_score(y_val, y_pred_hazard)

        # Try a few hazard weights; balanced approach
        weight_grid = [0.6, 0.7, 0.8]
        best_w = 1.0
        best_pred = y_pred_hazard
        best_auc = auc_hazard_only
        for w in weight_grid:
            pred = (w * y_pred_hazard + (1 - w) * y_pred_xgb)
            auc = roc_auc_score(y_val, pred)
            if auc > best_auc:
                best_auc = auc
                best_w = w
                best_pred = pred
        self.ensemble_weight = best_w

        # Evaluate ensemble with best weight
        auc_ensemble = roc_auc_score(y_val, best_pred)
        pr_auc_ensemble = average_precision_score(y_val, best_pred)
        
        print(f"\n=== Ensemble Results ===")
        print(f"Ensemble AUC: {auc_ensemble:.3f}")
        print(f"Ensemble PR-AUC: {pr_auc_ensemble:.3f}")
        
        # Optimize threshold for operational use
        self.optimize_threshold(y_val, best_pred)
        
        return {
            'hazard_model': self.hazard_model,
            'xgb_model': self.xgb_model,
            'calibrated_xgb': self.calibrated_xgb,
            'scaler': self.scaler,
            'ensemble_weight': self.ensemble_weight,
            'optimal_threshold': self.optimal_threshold,
            'feature_cols': self.feature_cols,
            'hazard_features': list(X_hazard.columns),
            'metrics': {
                'auc_ensemble': auc_ensemble,
                'pr_auc_ensemble': pr_auc_ensemble,
                'auc_hazard': roc_auc_score(y_val, y_pred_hazard),
                'auc_xgb': roc_auc_score(y_val, y_pred_xgb)
            }
        }
    
    def optimize_threshold(self, y_true, y_pred_proba):
        """Optimize threshold for operational alerts"""
        print("=== Threshold Optimization ===")
        
        # Try different thresholds (more granular for better optimization)
        thresholds = np.arange(0.05, 0.9, 0.01)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate false positive rate
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr
            })
        
        # Find threshold that gives ~50% precision with good recall (target 50%)
        target_precision = 0.50
        candidates = [r for r in results if r['precision'] >= target_precision * 0.95]
        
        if candidates:
            # Among high-precision candidates, pick best F1 then recall
            best_result = max(candidates, key=lambda x: (x['f1'], x['recall']))
        else:
            # Fallback to best precision
            best_result = max(results, key=lambda x: x['precision'])
        
        self.optimal_threshold = best_result['threshold']
        
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")
        print(f"Precision: {best_result['precision']:.3f}")
        print(f"Recall: {best_result['recall']:.3f}")
        print(f"F1-score: {best_result['f1']:.3f}")
        print(f"False Positive Rate: {best_result['fpr']:.3f}")
        
        return best_result
    
    def predict(self, X):
        """Make ensemble predictions"""
        # Prepare features
        X_cyclic = self.create_cyclic_features(X.copy())
        X_hazard = self.prepare_hazard_features(X_cyclic)
        X_xgb = self.prepare_xgb_features(X_cyclic)
        
        # Handle missing values
        X_hazard = X_hazard.fillna(0)
        X_xgb = X_xgb.fillna(0)
        
        # Get predictions from both models
        X_hazard_scaled = self.scaler.transform(X_hazard)
        y_pred_hazard = self.hazard_model.predict_proba(X_hazard_scaled)[:, 1]
        if self.calibrated_xgb is not None:
            y_pred_xgb = self.calibrated_xgb.predict_proba(X_xgb)[:, 1]
        else:
            y_pred_xgb = self.xgb_model.predict_proba(X_xgb)[:, 1]
        
        # Ensemble prediction
        y_pred_ensemble = (self.ensemble_weight * y_pred_hazard + 
                          (1 - self.ensemble_weight) * y_pred_xgb)
        
        return y_pred_ensemble
    
    def predict_with_threshold(self, X):
        """Make binary predictions using optimal threshold"""
        y_pred_proba = self.predict(X)
        return (y_pred_proba > self.optimal_threshold).astype(int)
    
    def predict_proba(self, X):
        """Make probability predictions (compatible with sklearn interface)"""
        y_pred_proba = self.predict(X)
        # Return 2D array with [P(0), P(1)] for sklearn compatibility
        return np.column_stack([1 - y_pred_proba, y_pred_proba])

def apply_operational_smoothing(predictions, timestamps, hysteresis_hours=1, cooldown_hours=3):
    """Apply operational smoothing to reduce false alarms"""
    print("=== Applying Operational Smoothing ===")
    
    smoothed_predictions = predictions.copy()
    
    # Hysteresis: require consecutive hours over threshold
    for i in range(hysteresis_hours, len(predictions)):
        if predictions[i] == 1:
            # Check if we have enough consecutive positive predictions
            consecutive_count = sum(predictions[i-hysteresis_hours+1:i+1])
            if consecutive_count < hysteresis_hours:
                smoothed_predictions[i] = 0
    
    # Cooldown: suppress repeat alerts
    last_alert_time = None
    for i, (pred, timestamp) in enumerate(zip(smoothed_predictions, timestamps)):
        if pred == 1:
            if last_alert_time is not None:
                hours_since_last = (timestamp - last_alert_time).total_seconds() / 3600
                if hours_since_last < cooldown_hours:
                    smoothed_predictions[i] = 0
                else:
                    last_alert_time = timestamp
            else:
                last_alert_time = timestamp
    
    # Calculate smoothing impact
    original_alerts = predictions.sum()
    smoothed_alerts = smoothed_predictions.sum()
    reduction = (original_alerts - smoothed_alerts) / original_alerts * 100 if original_alerts > 0 else 0
    
    print(f"Original alerts: {original_alerts}")
    print(f"Smoothed alerts: {smoothed_alerts}")
    print(f"Reduction: {reduction:.1f}%")
    
    return smoothed_predictions

def train_hazard_ensemble():
    """Main training function"""
    print("=== FlareAlert Hazard Ensemble Training ===\n")
    
    # Initialize trainer and model
    trainer = ModelTrainer()
    ensemble_model = HazardEnsembleModel()
    
    # Define training period (1 year for optimal precision)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)  # 1 year = 365 days
    
    print(f"Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    try:
        # Prepare data
        features_df, labels_df, dataset = trainer.prepare_training_data(start_time, end_time)
        
        if len(dataset) == 0:
            raise ValueError("No training data available")
        
        # Sort by timestamp for time-series validation
        dataset = dataset.sort_values('timestamp')
        
        # Train ensemble
        ensemble_results = ensemble_model.train_ensemble(dataset)
        
        # Test on held-out data with operational smoothing
        print("\n=== Testing with Operational Smoothing ===")
        
        # Use last 20% for final testing
        test_split = int(len(dataset) * 0.8)
        test_data = dataset.iloc[test_split:].copy()
        
        X_test = test_data[ensemble_model.feature_cols]
        y_test = test_data['flare6h_label']
        timestamps = test_data['timestamp']
        
        # Make predictions
        y_pred_proba = ensemble_model.predict(X_test)
        y_pred_binary = ensemble_model.predict_with_threshold(X_test)
        
        # Apply operational smoothing
        y_pred_smoothed = apply_operational_smoothing(y_pred_binary, timestamps)
        
        # Evaluate results
        print("\n=== Final Evaluation ===")
        
        # Without smoothing
        precision_raw = precision_score(y_test, y_pred_binary, zero_division=0)
        recall_raw = recall_score(y_test, y_pred_binary, zero_division=0)
        f1_raw = f1_score(y_test, y_pred_binary, zero_division=0)
        
        # Calculate false positive rate for raw predictions
        fp_raw = ((y_pred_binary == 1) & (y_test == 0)).sum()
        tn_raw = ((y_pred_binary == 0) & (y_test == 0)).sum()
        fpr_raw = fp_raw / (fp_raw + tn_raw) if (fp_raw + tn_raw) > 0 else 0
        
        # With smoothing
        precision_smooth = precision_score(y_test, y_pred_smoothed, zero_division=0)
        recall_smooth = recall_score(y_test, y_pred_smoothed, zero_division=0)
        f1_smooth = f1_score(y_test, y_pred_smoothed, zero_division=0)
        
        # Calculate false positive rate for smoothed predictions
        fp_smooth = ((y_pred_smoothed == 1) & (y_test == 0)).sum()
        tn_smooth = ((y_pred_smoothed == 0) & (y_test == 0)).sum()
        fpr_smooth = fp_smooth / (fp_smooth + tn_smooth) if (fp_smooth + tn_smooth) > 0 else 0
        
        print(f"Raw predictions - Precision: {precision_raw:.3f}, Recall: {recall_raw:.3f}, F1: {f1_raw:.3f}, FPR: {fpr_raw:.3f}")
        print(f"Smoothed predictions - Precision: {precision_smooth:.3f}, Recall: {recall_smooth:.3f}, F1: {f1_smooth:.3f}, FPR: {fpr_smooth:.3f}")
        
        # Print improvement summary
        precision_improvement = precision_smooth - precision_raw
        recall_change = recall_smooth - recall_raw
        f1_change = f1_smooth - f1_raw
        fpr_improvement = fpr_raw - fpr_smooth
        
        print(f"\n=== Smoothing Impact ===")
        print(f"Precision change: {precision_improvement:+.3f}")
        print(f"Recall change: {recall_change:+.3f}")
        print(f"F1 change: {f1_change:+.3f}")
        print(f"FPR improvement: {fpr_improvement:+.3f}")
        
        # Save model and results
        import pickle
        import json
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ensemble model
        model_filename = f'models/hazard_ensemble_{timestamp_str}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(ensemble_model, f)
        
        # Save metadata
        metadata = {
            'model_type': 'hazard_ensemble',
            'trained_at': datetime.now().isoformat(),
            'training_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'ensemble_weight': ensemble_model.ensemble_weight,
            'optimal_threshold': ensemble_model.optimal_threshold,
            'feature_columns': ensemble_model.feature_cols,
            'hazard_features': ensemble_results['hazard_features'],
            'metrics': ensemble_results['metrics'],
            'final_test_metrics': {
                'raw': {'precision': precision_raw, 'recall': recall_raw, 'f1': f1_raw},
                'smoothed': {'precision': precision_smooth, 'recall': recall_smooth, 'f1': f1_smooth}
            }
        }
        
        metadata_filename = f'models/hazard_ensemble_{timestamp_str}_metadata.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n=== Training Completed! ===")
        print(f"Model saved: {model_filename}")
        print(f"Metadata saved: {metadata_filename}")
        print(f"Ensemble AUC: {ensemble_results['metrics']['auc_ensemble']:.3f}")
        print(f"Final F1-score (smoothed): {f1_smooth:.3f}")
        
        # Print feature importance summary
        print(f"\n=== Feature Importance Summary ===")
        print(f"Hazard model features: {len(ensemble_results['hazard_features'])}")
        print(f"XGBoost features: {len(ensemble_model.feature_cols)}")
        print(f"Ensemble weight (Hazard/XGBoost): {ensemble_results['ensemble_weight']:.1f}/{1-ensemble_results['ensemble_weight']:.1f}")
        
        # Print top features if available
        if hasattr(ensemble_model, 'hazard_model') and hasattr(ensemble_model.hazard_model, 'coef_'):
            coefs = ensemble_model.hazard_model.coef_[0]
            feature_names = ensemble_results['hazard_features']
            feature_importance = list(zip(feature_names, coefs))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\nTop 5 Hazard Model Features:")
            for i, (feature, coef) in enumerate(feature_importance[:5]):
                print(f"  {i+1}. {feature}: {coef:.4f}")
        
        return ensemble_model, ensemble_results
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    train_hazard_ensemble()
