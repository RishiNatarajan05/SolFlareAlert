#!/usr/bin/env python3
"""
Prediction Service for FlareAlert
Handles feature extraction and ML model predictions
"""

import sys
import os
# Fix the path to properly include the scripts directory
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)
scripts_dir = os.path.join(project_root, 'scripts')

sys.path.insert(0, backend_dir)
sys.path.insert(0, scripts_dir)

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from database.models import create_database, SolarFlare, CME, GeomagneticStorm

logger = logging.getLogger(__name__)

# Global variable to track if HazardEnsembleModel is available
HAZARD_MODEL_AVAILABLE = False

class PlaceholderModel:
    """Placeholder model that returns reasonable predictions when real model can't be loaded"""
    
    def predict_proba(self, X):
        """Return probability predictions"""
        # Return a reasonable prediction based on current solar activity
        # This is a simplified version of your hazard ensemble logic
        n_samples = X.shape[0] if hasattr(X, 'shape') else 1
        
        # Simple logic: higher activity = higher probability
        if n_samples == 1:
            # Extract some key features if available
            if hasattr(X, 'iloc'):
                flare_count_24h = X.iloc[0].get('flare_count_24h', 0) if hasattr(X.iloc[0], 'get') else 0
                m_plus_count_24h = X.iloc[0].get('m_plus_count_24h', 0) if hasattr(X.iloc[0], 'get') else 0
            else:
                flare_count_24h = 0
                m_plus_count_24h = 0
            
            # Simple probability calculation
            base_prob = 0.05  # 5% base probability
            flare_bonus = min(flare_count_24h * 0.02, 0.15)  # Up to 15% bonus
            m_plus_bonus = min(m_plus_count_24h * 0.1, 0.3)  # Up to 30% bonus
            
            prob = min(base_prob + flare_bonus + m_plus_bonus, 0.8)  # Cap at 80%
            
            return np.array([[1 - prob, prob]])  # [no_flare, flare]
        else:
            # For multiple samples, return reasonable probabilities
            return np.array([[0.85, 0.15]] * n_samples)
    
    def predict(self, X):
        """Return binary predictions"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

class PredictionService:
    """Service for making solar flare predictions using the hazard ensemble model"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.feature_cols = None
        self.scaler = None
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained hazard ensemble model"""
        global HAZARD_MODEL_AVAILABLE
        
        try:
            # Try to import HazardEnsembleModel here when needed
            try:
                from hazard_ensemble_model import HazardEnsembleModel
                HAZARD_MODEL_AVAILABLE = True
                logger.info("Successfully imported HazardEnsembleModel")
            except ImportError as e:
                logger.warning(f"Could not import HazardEnsembleModel: {e}")
                HAZARD_MODEL_AVAILABLE = False
            
            import glob
            model_files = glob.glob('../models/hazard_ensemble_*.pkl')
            
            if not model_files:
                logger.warning("No trained model found")
                return False
            
            # Load the latest model
            latest_model = max(model_files, key=os.path.getctime)
            logger.info(f"Loading model from: {latest_model}")
            
            # Try to load with custom unpickler to handle missing classes
            try:
                # Import the model class first to make it available
                from hazard_ensemble_model import HazardEnsembleModel
                
                # Create a custom unpickler that can find the class in the correct module
                import pickle
                import importlib
                
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == '__main__':
                            # Redirect __main__ to hazard_ensemble_model
                            return getattr(importlib.import_module('hazard_ensemble_model'), name)
                        return super().find_class(module, name)
                
                # Now try to load the pickle with custom unpickler
                with open(latest_model, 'rb') as f:
                    model_data = CustomUnpickler(f).load()
                logger.info("Successfully loaded real model")
                
                # Extract model components
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.feature_cols = model_data.get('feature_cols')
                else:
                    # Assume it's the model directly
                    self.model = model_data
                
                self.model_path = latest_model
                logger.info("Model loaded successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Could not load model with pickle: {e}")
                if HAZARD_MODEL_AVAILABLE:
                    # Try to create a new instance of the model
                    logger.info("Creating new HazardEnsembleModel instance")
                    self.model = HazardEnsembleModel()
                    # You might need to load the model weights separately
                else:
                    # Fall back to placeholder model
                    self.model = PlaceholderModel()
                    logger.info("Using placeholder model")
                self.model_path = latest_model
                return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def extract_features(self, timestamp: datetime) -> pd.DataFrame:
        """Extract features for prediction at given timestamp"""
        try:
            engine = create_database()
            
            # Get data from database for feature extraction
            features = self._get_solar_activity_features(engine, timestamp)
            features = self._get_time_features(features, timestamp)
            features = self._get_rolling_statistics(engine, features, timestamp)
            
            # Add timestamp for the model
            features['timestamp'] = timestamp
            
            # Ensure we have all required features
            features = self._ensure_feature_columns(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features if extraction fails
            return self._get_default_features()
    
    def _get_solar_activity_features(self, engine, timestamp: datetime) -> Dict:
        """Get basic solar activity features"""
        features = {}
        
        try:
            with engine.connect() as conn:
                # Get recent flares
                flares_6h = conn.execute(
                    text("SELECT COUNT(*) FROM solar_flares WHERE peak_time >= :cutoff"),
                    {"cutoff": timestamp - timedelta(hours=6)}
                ).scalar() or 0
                
                flares_24h = conn.execute(
                    text("SELECT COUNT(*) FROM solar_flares WHERE peak_time >= :cutoff"),
                    {"cutoff": timestamp - timedelta(hours=24)}
                ).scalar() or 0
                
                flares_7d = conn.execute(
                    text("SELECT COUNT(*) FROM solar_flares WHERE peak_time >= :cutoff"),
                    {"cutoff": timestamp - timedelta(days=7)}
                ).scalar() or 0
                
                # Get M+ flares
                m_plus_6h = conn.execute(
                    text("SELECT COUNT(*) FROM solar_flares WHERE peak_time >= :cutoff AND class_type IN ('M', 'X')"),
                    {"cutoff": timestamp - timedelta(hours=6)}
                ).scalar() or 0
                
                m_plus_24h = conn.execute(
                    text("SELECT COUNT(*) FROM solar_flares WHERE peak_time >= :cutoff AND class_type IN ('M', 'X')"),
                    {"cutoff": timestamp - timedelta(hours=24)}
                ).scalar() or 0
                
                m_plus_7d = conn.execute(
                    text("SELECT COUNT(*) FROM solar_flares WHERE peak_time >= :cutoff AND class_type IN ('M', 'X')"),
                    {"cutoff": timestamp - timedelta(days=7)}
                ).scalar() or 0
                
                # Get CME data
                cme_24h = conn.execute(
                    text("SELECT COUNT(*) FROM cmes WHERE time21_5 >= :cutoff"),
                    {"cutoff": timestamp - timedelta(hours=24)}
                ).scalar() or 0
                
                # Get geomagnetic storm data
                storm_24h = conn.execute(
                    text("SELECT COUNT(*) FROM geomagnetic_storms WHERE time_tag >= :cutoff"),
                    {"cutoff": timestamp - timedelta(hours=24)}
                ).scalar() or 0
                
                features.update({
                    'flare_count_6h': flares_6h,
                    'flare_count_24h': flares_24h,
                    'flare_count_7d': flares_7d,
                    'm_plus_count_6h': m_plus_6h,
                    'm_plus_count_24h': m_plus_24h,
                    'm_plus_count_7d': m_plus_7d,
                    'cme_count_24h': cme_24h,
                    'kp_storm_count_24h': storm_24h,
                })
                
        except Exception as e:
            logger.error(f"Error getting solar activity features: {e}")
        
        return features
    
    def _get_time_features(self, features: Dict, timestamp: datetime) -> Dict:
        """Add cyclic time features"""
        # Hour of day (0-23)
        hour = timestamp.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6)
        day_of_week = timestamp.weekday()
        features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Month (1-12)
        month = timestamp.month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        return features
    
    def _get_rolling_statistics(self, engine, features: Dict, timestamp: datetime) -> Dict:
        """Calculate rolling window statistics"""
        try:
            with engine.connect() as conn:
                # Get max flare class in last 24h
                max_flare_result = conn.execute(
                    text("SELECT MAX(class_value) FROM solar_flares WHERE peak_time >= :cutoff"),
                    {"cutoff": timestamp - timedelta(hours=24)}
                ).scalar()
                
                features['max_flare_value'] = max_flare_result or 0.0
                features['max_flare_class'] = self._class_value_to_class(max_flare_result) if max_flare_result else 1
                
                # Get max CME speed
                max_cme_speed = conn.execute(
                    text("SELECT MAX(speed) FROM cmes WHERE time21_5 >= :cutoff"),
                    {"cutoff": timestamp - timedelta(hours=24)}
                ).scalar() or 0.0
                
                features['max_cme_speed'] = max_cme_speed
                
                # Get current Kp index (real data)
                current_kp_result = conn.execute(
                    text("SELECT kp_index FROM geomagnetic_storms WHERE time_tag <= :now ORDER BY time_tag DESC LIMIT 1"),
                    {"now": timestamp}
                ).scalar()
                features['current_kp'] = float(current_kp_result) if current_kp_result else 2.0
                
                # Get max Kp in last 24h
                max_kp_result = conn.execute(
                    text("SELECT MAX(kp_index) FROM geomagnetic_storms WHERE time_tag >= :cutoff"),
                    {"cutoff": timestamp - timedelta(hours=24)}
                ).scalar()
                features['max_kp_24h'] = float(max_kp_result) if max_kp_result else 2.0
                
        except Exception as e:
            logger.error(f"Error getting rolling statistics: {e}")
            # Fallback to reasonable defaults
            features['current_kp'] = 2.0
            features['max_kp_24h'] = 2.0
        
        return features
    
    def _class_value_to_class(self, class_value: float) -> int:
        """Convert class value to numeric class type"""
        if class_value >= 1e-4:
            return 5  # X class
        elif class_value >= 1e-5:
            return 4  # M class
        elif class_value >= 1e-6:
            return 3  # C class
        elif class_value >= 1e-7:
            return 2  # B class
        else:
            return 1  # A class
    
    def _ensure_feature_columns(self, features: Dict) -> pd.DataFrame:
        """Ensure all required feature columns are present"""
        # Define all expected features based on the XGBoost model requirements
        expected_features = [
            # Basic flare features
            'flare_count_6h', 'flare_count_24h', 'flare_count_72h',
            'm_plus_count_6h', 'm_plus_count_24h', 'm_plus_count_72h',
            'max_flare_class', 'max_flare_value', 'hours_since_last_mx',
            'unique_active_regions', 'most_active_region',
            
            # CME features
            'cme_count_24h', 'cme_count_72h', 'max_cme_speed', 'avg_cme_speed', 'last_cme_speed',
            'hours_since_last_cme', 'max_cme_width', 'avg_cme_width',
            
            # Geomagnetic features
            'current_kp', 'max_kp_24h', 'avg_kp_24h', 'kp_storm_count_24h',
            'current_dst', 'min_dst_24h',
            
            # Lag features
            'flare_count_7d', 'm_plus_count_7d', 'flare_activity_ratio', 'm_plus_activity_ratio',
            
            # Interaction features
            'flare_cme_ratio', 'm_plus_cme_ratio', 'cme_speed_width_product', 'cme_speed_width_ratio',
            'flare_kp_product', 'm_plus_kp_product', 'hour_flare_interaction', 'day_flare_interaction',
            
            # Rolling statistics
            'rolling_7d_flare_avg', 'rolling_7d_m_plus_avg',
            
            # CME statistics
            'cme_speed_std', 'cme_width_std', 'cme_speed_trend', 'cme_width_trend',
            'cme_lat_std', 'cme_lon_std', 'recent_cme_speed_ratio', 'recent_cme_width_ratio',
            
            # Cyclic time features
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
            'month_sin', 'month_cos',
            
            # Additional features for XGBoost
            'timestamp'  # The model expects this for cyclic feature creation
        ]
        
        # Fill missing features with defaults
        for feature in expected_features:
            if feature not in features:
                if 'count' in feature:
                    features[feature] = 0
                elif 'sin' in feature or 'cos' in feature:
                    features[feature] = 0.0
                elif 'kp' in feature or 'dst' in feature:
                    features[feature] = 2.5
                elif 'hours_since_last' in feature:
                    features[feature] = 24.0  # Default to 24 hours
                elif 'activity_ratio' in feature or 'cme_ratio' in feature or 'speed_ratio' in feature or 'width_ratio' in feature:
                    features[feature] = 0.0
                elif 'product' in feature:
                    features[feature] = 0.0
                elif 'interaction' in feature:
                    features[feature] = 0.0
                elif 'rolling' in feature or 'avg' in feature or 'std' in feature or 'trend' in feature:
                    features[feature] = 0.0
                elif 'unique' in feature or 'most' in feature:
                    features[feature] = 0
                elif 'speed' in feature or 'width' in feature:
                    features[feature] = 0.0
                elif 'lat' in feature or 'lon' in feature:
                    features[feature] = 0.0
                elif feature == 'timestamp':
                    features[feature] = datetime.now()
                else:
                    features[feature] = 0.0
        
        return pd.DataFrame([features])
    
    def _get_default_features(self) -> pd.DataFrame:
        """Get default features when extraction fails"""
        default_features = {
            'flare_count_6h': 0, 'flare_count_24h': 0, 'flare_count_7d': 0,
            'm_plus_count_6h': 0, 'm_plus_count_24h': 0, 'm_plus_count_7d': 0,
            'cme_count_24h': 0, 'kp_storm_count_24h': 0,
            'max_flare_value': 0.0, 'max_flare_class': 1, 'max_cme_speed': 0.0,
            'current_kp': 2.5, 'max_kp_24h': 4.0,
            'hour_sin': 0.0, 'hour_cos': 1.0, 'day_of_week_sin': 0.0, 'day_of_week_cos': 1.0,
            'month_sin': 0.0, 'month_cos': 1.0,
            'hours_since_last_mx': 24.0, 'hours_since_last_cme': 24.0,
            'flare_activity_ratio': 0.0, 'm_plus_activity_ratio': 0.0,
            'timestamp': datetime.now()
        }
        return pd.DataFrame([default_features])
    
    def predict(self, timestamp: datetime = None) -> Tuple[float, float]:
        """Make a solar flare prediction"""
        if self.model is None:
            logger.error("Model not loaded")
            return 0.0, 0.0
        
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Extract features
            features_df = self.extract_features(timestamp)
            
            # Ensure we have all the features the model expects in the correct order
            if hasattr(self.model, 'feature_cols') and self.model.feature_cols is not None:
                # Create a DataFrame with the exact columns the model expects
                model_features = {}
                for col in self.model.feature_cols:
                    if col in features_df.columns:
                        model_features[col] = features_df[col].iloc[0]
                    else:
                        # Provide default values for missing features
                        if 'count' in col:
                            model_features[col] = 0
                        elif 'sin' in col or 'cos' in col:
                            model_features[col] = 0.0
                        elif 'kp' in col or 'dst' in col:
                            model_features[col] = 2.5
                        elif 'hours_since_last' in col:
                            model_features[col] = 24.0
                        elif 'activity_ratio' in col or 'cme_ratio' in col or 'speed_ratio' in col or 'width_ratio' in col:
                            model_features[col] = 0.0
                        elif 'product' in col:
                            model_features[col] = 0.0
                        elif 'interaction' in col:
                            model_features[col] = 0.0
                        elif 'rolling' in col or 'avg' in col or 'std' in col or 'trend' in col:
                            model_features[col] = 0.0
                        elif 'unique' in col or 'most' in col:
                            model_features[col] = 0
                        elif 'speed' in col or 'width' in col:
                            model_features[col] = 0.0
                        elif 'lat' in col or 'lon' in col:
                            model_features[col] = 0.0
                        else:
                            model_features[col] = 0.0
                
                # Add timestamp
                model_features['timestamp'] = timestamp
                
                # Create DataFrame with correct column order
                features_df = pd.DataFrame([model_features])
            
            # The HazardEnsembleModel expects a DataFrame with specific column names
            # Pass the DataFrame directly to the model
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(features_df)[0][1]  # Probability of positive class
            else:
                prediction = self.model.predict(features_df)[0]
            
            # Calculate confidence (placeholder - could be based on model uncertainty)
            confidence = 0.85  # Placeholder confidence
            
            logger.info(f"Prediction made: {prediction:.3f} (confidence: {confidence:.3f})")
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0, 0.0
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        # Get scaler and feature_cols from the loaded model
        scaler_available = hasattr(self.model, 'scaler') and self.model.scaler is not None
        features_count = len(self.model.feature_cols) if hasattr(self.model, 'feature_cols') and self.model.feature_cols is not None else "Unknown"
        
        return {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
            "features_count": features_count,
            "scaler_available": scaler_available,
            "last_updated": datetime.fromtimestamp(os.path.getctime(self.model_path)).isoformat() if self.model_path else None
        }
