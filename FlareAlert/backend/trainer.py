import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json
import os
from typing import Dict, Tuple, Optional, List
import logging

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

from data.features import FeatureEngineer
from data.labels import LabelGenerator
from config import Config

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and calibrate solar weather prediction models"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.feature_engineer = FeatureEngineer()
        self.label_generator = LabelGenerator()
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
    
    def prepare_training_data(self, start_time: datetime, end_time: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data with features and labels
        
        Returns:
            Tuple of (features_df, labels_df)
        """
        logger.info(f"Preparing training data from {start_time} to {end_time}")
        
        # Generate labels
        labels_df = self.label_generator.generate_all_labels(start_time, end_time)
        
        # Generate features for each timestamp
        feature_data = []
        
        for _, row in labels_df.iterrows():
            timestamp = row['timestamp']
            features = self.feature_engineer.get_all_features(timestamp)
            features['timestamp'] = timestamp
            feature_data.append(features)
        
        features_df = pd.DataFrame(feature_data)
        
        # Merge features and labels
        dataset = pd.merge(features_df, labels_df, on='timestamp', how='inner')
        
        logger.info(f"Prepared dataset with {len(dataset)} samples")
        
        return features_df, labels_df, dataset
    
    def train_flare_model(self, dataset: pd.DataFrame) -> Tuple[object, Dict]:
        """Train Hazard Ensemble model for flare prediction"""
        logger.info("Training Hazard Ensemble flare prediction model")
        
        # Import the hazard ensemble model
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
        from hazard_ensemble_model import HazardEnsembleModel
        
        # Create and train the hazard ensemble
        ensemble_model = HazardEnsembleModel()
        ensemble_results = ensemble_model.train_ensemble(dataset)
        
        # Prepare metadata
        metadata = {
            'model_type': 'hazard_ensemble_flare_prediction',
            'target': 'flare6h_label',
            'feature_columns': ensemble_model.feature_cols,
            'training_samples': len(dataset),
            'positive_samples': dataset['flare6h_label'].sum(),
            'negative_samples': (dataset['flare6h_label'] == 0).sum(),
            'positive_rate': dataset['flare6h_label'].mean(),
            'ensemble_weight': ensemble_model.ensemble_weight,
            'optimal_threshold': ensemble_model.optimal_threshold,
            'hazard_features': ensemble_results['hazard_features'],
            'metrics': ensemble_results['metrics'],
            'trained_at': datetime.now().isoformat(),
            'model_architecture': 'hazard_ensemble'
        }
        
        return ensemble_model, metadata
    
    def train_kp_model(self, dataset: pd.DataFrame) -> Tuple[xgb.XGBClassifier, Dict]:
        """Train XGBoost model for Kp prediction"""
        logger.info("Training Kp prediction model")
        
        # Prepare features and target
        feature_cols = [col for col in dataset.columns 
                       if not col.endswith('_label') and col != 'timestamp']
        X = dataset[feature_cols]
        y = dataset['kp12h_label']
        
        # Ensure all features are numeric and handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Converting {col} from object to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill missing values with 0 for all features
        X = X.fillna(0)
        
        # Convert to numeric types
        X = X.astype(float)
        y = y.astype(int)
        
        # Handle class imbalance and no positive samples
        positive_samples = (y == 1).sum()
        if positive_samples == 0:
            logger.warning("No positive samples found - creating dummy model")
            # Create a dummy model that always predicts 0
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy='constant', constant=0)
            dummy_model.fit(X, y)
            
            # Create metadata for dummy model
            metadata = {
                'model_type': 'flare_prediction_dummy',
                'target': 'flare6h_label',
                'feature_columns': feature_cols,
                'training_samples': len(dataset),
                'positive_samples': 0,
                'negative_samples': len(y),
                'positive_rate': 0.0,
                'feature_importance': {col: 0.0 for col in feature_cols},
                'trained_at': datetime.now().isoformat(),
                'xgb_params': None
            }
            return dummy_model, metadata
        
        scale_pos_weight = (y == 0).sum() / positive_samples
        
        # XGBoost parameters - Fine-tuned for optimal performance
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',  # Optimized for AUC
            'max_depth': 4,  # Optimal depth
            'learning_rate': 0.05,  # Stable learning rate
            'n_estimators': 200,  # Sufficient trees
            'subsample': 0.95,  # Fine-tuned: increased from 0.9 to 0.95
            'colsample_bytree': 0.95,  # Fine-tuned: increased from 0.9 to 0.95
            'min_child_weight': 3,  # Regularization
            'gamma': 0.1,  # Regularization
            'scale_pos_weight': scale_pos_weight,
            'base_score': 0.15,  # Good for imbalanced data
            'random_state': 42
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
        calibrated_model.fit(X, y)
        
        # Get feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        # Model metadata
        metadata = {
            'model_type': 'kp_prediction',
            'target': 'kp12h_label',
            'feature_columns': feature_cols,
            'training_samples': len(dataset),
            'positive_samples': y.sum(),
            'negative_samples': (y == 0).sum(),
            'positive_rate': y.mean(),
            'feature_importance': feature_importance,
            'trained_at': datetime.now().isoformat(),
            'xgb_params': params
        }
        
        return calibrated_model, metadata
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict:
        """Evaluate model performance"""
        logger.info(f"Evaluating {model_name}")
        
        # Handle missing values in test data
        X_test_clean = X_test.copy()
        X_test_clean = X_test_clean.fillna(0)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test_clean)
        # Handle models that only predict one class
        if y_pred_proba.shape[1] == 1:
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        y_pred_proba = y_pred_proba[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else 0.0
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        evaluation = {
            'model_name': model_name,
            'auc': auc,
            'classification_report': report,
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist()
            },
            'test_samples': len(y_test),
            'positive_rate': y_test.mean()
        }
        
        return evaluation
    
    def save_model(self, model, metadata: Dict, model_name: str):
        """Save model and metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = os.path.join(self.models_dir, f'{model_name}_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = os.path.join(self.models_dir, f'{model_name}_{timestamp}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save latest version
        latest_model_path = os.path.join(self.models_dir, f'{model_name}_latest.pkl')
        latest_metadata_path = os.path.join(self.models_dir, f'{model_name}_latest_metadata.json')
        
        with open(latest_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved {model_name} model to {model_path}")
        return model_path, metadata_path
    
    def train_ensemble_flare_model(self, dataset: pd.DataFrame) -> Tuple[VotingClassifier, Dict]:
        """Train ensemble model for flare prediction using multiple algorithms"""
        logger.info("Training ensemble flare prediction model")
        
        # Prepare features and target
        feature_cols = [col for col in dataset.columns 
                       if not col.endswith('_label') and col != 'timestamp']
        X = dataset[feature_cols]
        y = dataset['flare6h_label']
        
        # Ensure all features are numeric and handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Converting {col} from object to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill missing values with 0 for all features
        X = X.fillna(0)
        
        # Convert to numeric types
        X = X.astype(float)
        y = y.astype(int)
        
        # Handle class imbalance
        positive_samples = (y == 1).sum()
        if positive_samples == 0:
            logger.warning("No positive samples found - creating dummy ensemble")
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy='constant', constant=0)
            dummy_model.fit(X, y)
            
            metadata = {
                'model_type': 'flare_ensemble_dummy',
                'target': 'flare6h_label',
                'feature_columns': feature_cols,
                'training_samples': len(dataset),
                'positive_samples': 0,
                'negative_samples': len(y),
                'positive_rate': 0.0,
                'feature_importance': {col: 0.0 for col in feature_cols},
                'trained_at': datetime.now().isoformat(),
                'ensemble_params': None
            }
            return dummy_model, metadata
        
        scale_pos_weight = (y == 0).sum() / positive_samples
        
        # Create base models with optimized parameters
        base_models = [
            ('xgb', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=3,
                gamma=0.1,
                scale_pos_weight=scale_pos_weight,
                base_score=0.15,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                random_state=42
            )),
            ('ada', AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            ))
        ]
        
        # Create voting classifier with soft voting
        ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',  # Use probability predictions
            weights=[0.3, 0.2, 0.2, 0.2, 0.1]  # Weight XGBoost higher
        )
        
        # Train ensemble
        ensemble.fit(X, y)
        
        # Get feature importance from XGBoost (most important model)
        xgb_model = ensemble.named_estimators_['xgb']
        feature_importance = dict(zip(feature_cols, xgb_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        # Cross-validation score
        cv_scores = cross_val_score(ensemble, X, y, cv=3, scoring='roc_auc')
        
        # Model metadata
        metadata = {
            'model_type': 'flare_ensemble',
            'target': 'flare6h_label',
            'feature_columns': feature_cols,
            'training_samples': len(dataset),
            'positive_samples': y.sum(),
            'negative_samples': (y == 0).sum(),
            'positive_rate': y.mean(),
            'feature_importance': feature_importance,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'trained_at': datetime.now().isoformat(),
            'ensemble_params': {
                'voting': 'soft',
                'weights': [0.3, 0.2, 0.2, 0.2, 0.1],
                'models': ['xgb', 'rf', 'et', 'gb', 'ada']
            }
        }
        
        return ensemble, metadata
    
    def train_ensemble_kp_model(self, dataset: pd.DataFrame) -> Tuple[VotingClassifier, Dict]:
        """Train ensemble model for Kp prediction using multiple algorithms"""
        logger.info("Training ensemble Kp prediction model")
        
        # Prepare features and target
        feature_cols = [col for col in dataset.columns 
                       if not col.endswith('_label') and col != 'timestamp']
        X = dataset[feature_cols]
        y = dataset['kp12h_label']
        
        # Ensure all features are numeric and handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Converting {col} from object to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill missing values with 0 for all features
        X = X.fillna(0)
        
        # Convert to numeric types
        X = X.astype(float)
        y = y.astype(int)
        
        # Handle class imbalance
        positive_samples = (y == 1).sum()
        if positive_samples == 0:
            logger.warning("No positive samples found - creating dummy ensemble")
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy='constant', constant=0)
            dummy_model.fit(X, y)
            
            metadata = {
                'model_type': 'kp_ensemble_dummy',
                'target': 'kp12h_label',
                'feature_columns': feature_cols,
                'training_samples': len(dataset),
                'positive_samples': 0,
                'negative_samples': len(y),
                'positive_rate': 0.0,
                'feature_importance': {col: 0.0 for col in feature_cols},
                'trained_at': datetime.now().isoformat(),
                'ensemble_params': None
            }
            return dummy_model, metadata
        
        scale_pos_weight = (y == 0).sum() / positive_samples
        
        # Create base models with optimized parameters
        base_models = [
            ('xgb', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=3,
                gamma=0.1,
                scale_pos_weight=scale_pos_weight,
                base_score=0.15,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                random_state=42
            )),
            ('ada', AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            ))
        ]
        
        # Create voting classifier with soft voting
        ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',  # Use probability predictions
            weights=[0.3, 0.2, 0.2, 0.2, 0.1]  # Weight XGBoost higher
        )
        
        # Train ensemble
        ensemble.fit(X, y)
        
        # Get feature importance from XGBoost (most important model)
        xgb_model = ensemble.named_estimators_['xgb']
        feature_importance = dict(zip(feature_cols, xgb_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        # Cross-validation score
        cv_scores = cross_val_score(ensemble, X, y, cv=3, scoring='roc_auc')
        
        # Model metadata
        metadata = {
            'model_type': 'kp_ensemble',
            'target': 'kp12h_label',
            'feature_columns': feature_cols,
            'training_samples': len(dataset),
            'positive_samples': y.sum(),
            'negative_samples': (y == 0).sum(),
            'positive_rate': y.mean(),
            'feature_importance': feature_importance,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'trained_at': datetime.now().isoformat(),
            'ensemble_params': {
                'voting': 'soft',
                'weights': [0.3, 0.2, 0.2, 0.2, 0.1],
                'models': ['xgb', 'rf', 'et', 'gb', 'ada']
            }
        }
        
        return ensemble, metadata
    
    def optimize_threshold(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Optimize prediction threshold for better precision-recall balance"""
        logger.info("Optimizing prediction threshold")
        
        # Get probability predictions
        y_pred_proba = model.predict_proba(X_val)
        if y_pred_proba.shape[1] == 1:
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        y_pred_proba = y_pred_proba[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            report = classification_report(y_val, y_pred, output_dict=True)
            
            if '1' in report:
                f1 = report['1']['f1-score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def train_all_models(self, start_time: datetime, end_time: datetime) -> Dict:
        """Train both flare and Kp models"""
        logger.info("Starting model training pipeline")
        
        # Prepare data
        features_df, labels_df, dataset = self.prepare_training_data(start_time, end_time)
        
        if len(dataset) == 0:
            raise ValueError("No training data available")
        
        # Split data ensuring positive samples in both sets
        dataset = dataset.sort_values('timestamp')
        
        # Find positive samples
        positive_samples = dataset[dataset['flare6h_label'] == 1]
        negative_samples = dataset[dataset['flare6h_label'] == 0]
        
        # Ensure we have positive samples in both train and test
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
        
        feature_cols = [col for col in dataset.columns 
                       if not col.endswith('_label') and col != 'timestamp']
        
        X_train = train_data[feature_cols]
        y_train_flare = train_data['flare6h_label']
        y_train_kp = train_data['kp12h_label']
        
        X_test = test_data[feature_cols]
        y_test_flare = test_data['flare6h_label']
        y_test_kp = test_data['kp12h_label']
        
        # Train single XGBoost models (Strategy 2 - best performer)
        flare_model, flare_metadata = self.train_flare_model(train_data)
        kp_model, kp_metadata = self.train_kp_model(train_data)
        
        # Evaluate models
        flare_eval = self.evaluate_model(flare_model, X_test, y_test_flare, 'flare_model')
        kp_eval = self.evaluate_model(kp_model, X_test, y_test_kp, 'kp_model')
        
        # Save models
        flare_model_path, flare_metadata_path = self.save_model(flare_model, flare_metadata, 'flare')
        kp_model_path, kp_metadata_path = self.save_model(kp_model, kp_metadata, 'kp')
        
        # Training summary
        summary = {
            'training_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'data_summary': {
                'total_samples': len(dataset),
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'flare_positive_rate': dataset['flare6h_label'].mean(),
                'kp_positive_rate': dataset['kp12h_label'].mean()
            },
            'models': {
                'flare': {
                    'model_path': flare_model_path,
                    'metadata_path': flare_metadata_path,
                    'evaluation': flare_eval
                },
                'kp': {
                    'model_path': kp_model_path,
                    'metadata_path': kp_metadata_path,
                    'evaluation': kp_eval
                }
            }
        }
        
        # Save training summary
        summary_path = os.path.join(self.models_dir, f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Model training completed successfully")
        return summary
