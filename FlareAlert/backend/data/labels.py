import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func
from typing import Dict, List, Tuple, Optional
import logging

from database.models import SolarFlare, GeomagneticStorm
from config import Config

logger = logging.getLogger(__name__)

class LabelGenerator:
    """Generate labels for solar weather prediction models"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use the database in the backend directory where ingestion creates it
            db_url = 'sqlite:///backend/flarealert.db'
        else:
            db_url = f'sqlite:///{db_path}'
        
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def generate_flare_labels(self, start_time: datetime, end_time: datetime, 
                        prediction_hours: int = 6) -> pd.DataFrame:
        """
        Generate flare prediction labels
        
        Args:
            start_time: Start of the time window
            end_time: End of the time window
            prediction_hours: Hours ahead to predict (default: 6)
        
        Returns:
            DataFrame with columns: [timestamp, flare6h_label]
        """
        session = self.Session()
        
        try:
            # Generate timestamps for labeling
            timestamps = []
            current = start_time
            while current <= end_time:
                timestamps.append(current)
                current += timedelta(hours=1)  # Hourly intervals
            
            labels = []
            
            for timestamp in timestamps:
                # Look ahead window for flares
                look_ahead_start = timestamp
                look_ahead_end = timestamp + timedelta(hours=prediction_hours)
                
                # Check if any M1.0+ flare occurs in the look-ahead window
                flares = session.query(SolarFlare).filter(
                    SolarFlare.begin_time >= look_ahead_start,
                    SolarFlare.begin_time < look_ahead_end,
                    SolarFlare.class_type.in_(['M', 'X']),
                    SolarFlare.class_value >= 1.0
                ).all()
                
                # Label: 1 if any M1.0+ flare occurs, 0 otherwise
                label = 1 if flares else 0
                
                labels.append({
                    'timestamp': timestamp,
                    'flare6h_label': label
                })
            
            df = pd.DataFrame(labels)
            
            # Debug: Print some statistics
            positive_count = df['flare6h_label'].sum()
            total_count = len(df)
            print(f"Flare labels: {positive_count}/{total_count} positive ({positive_count/total_count*100:.1f}%)")
            
            return df
            
        finally:
            session.close()
    
    def generate_kp_labels(self, start_time: datetime, end_time: datetime, 
                          prediction_hours: int = 12) -> pd.DataFrame:
        """
        Generate Kp index prediction labels
        
        Args:
            start_time: Start of the time window
            end_time: End of the time window
            prediction_hours: Hours ahead to predict (default: 12)
        
        Returns:
            DataFrame with columns: [timestamp, kp12h_label]
        """
        session = self.Session()
        
        try:
            # Generate timestamps for labeling
            timestamps = []
            current = start_time
            while current <= end_time:
                timestamps.append(current)
                current += timedelta(hours=1)  # Hourly intervals
            
            labels = []
            
            for timestamp in timestamps:
                # Look ahead window for geomagnetic storms
                look_ahead_start = timestamp
                look_ahead_end = timestamp + timedelta(hours=prediction_hours)
                
                # Check if any Kp >= 5 storm occurs in the look-ahead window
                storms = session.query(GeomagneticStorm).filter(
                    GeomagneticStorm.time_tag >= look_ahead_start,
                    GeomagneticStorm.time_tag < look_ahead_end,
                    GeomagneticStorm.kp_index >= 5.0
                ).all()
                
                # Label: 1 if any Kp >= 5 storm occurs, 0 otherwise
                label = 1 if storms else 0
                
                labels.append({
                    'timestamp': timestamp,
                    'kp12h_label': label
                })
            
            df = pd.DataFrame(labels)
            
            # Debug: Print some statistics
            positive_count = df['kp12h_label'].sum()
            total_count = len(df)
            print(f"Kp labels: {positive_count}/{total_count} positive ({positive_count/total_count*100:.1f}%)")
            
            return df
            
        finally:
            session.close()
    
    def generate_all_labels(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Generate both flare and Kp labels for the same time window
        
        Returns:
            DataFrame with columns: [timestamp, flare6h_label, kp12h_label]
        """
        # Generate individual labels
        flare_labels = self.generate_flare_labels(start_time, end_time)
        kp_labels = self.generate_kp_labels(start_time, end_time)
        
        # Merge on timestamp
        merged = pd.merge(flare_labels, kp_labels, on='timestamp', how='outer')
        
        # Fill any missing values with 0
        merged = merged.fillna(0)
        
        return merged
    
    def get_label_statistics(self, labels_df: pd.DataFrame) -> Dict:
        """
        Get statistics about the generated labels
        
        Returns:
            Dictionary with label statistics
        """
        stats = {}
        
        if 'flare6h_label' in labels_df.columns:
            flare_labels = labels_df['flare6h_label']
            stats['flare6h'] = {
                'total_samples': len(flare_labels),
                'positive_samples': flare_labels.sum(),
                'negative_samples': (flare_labels == 0).sum(),
                'positive_rate': flare_labels.mean(),
                'class_imbalance': (flare_labels == 0).sum() / max(flare_labels.sum(), 1)
            }
        
        if 'kp12h_label' in labels_df.columns:
            kp_labels = labels_df['kp12h_label']
            stats['kp12h'] = {
                'total_samples': len(kp_labels),
                'positive_samples': kp_labels.sum(),
                'negative_samples': (kp_labels == 0).sum(),
                'positive_rate': kp_labels.mean(),
                'class_imbalance': (kp_labels == 0).sum() / max(kp_labels.sum(), 1)
            }
        
        return stats
    
    def validate_labels(self, labels_df: pd.DataFrame) -> Dict:
        """
        Validate the generated labels for data quality
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': []
        }
        
        # Check for missing timestamps
        if labels_df['timestamp'].isnull().any():
            validation['is_valid'] = False
            validation['issues'].append('Missing timestamps found')
        
        # Check for duplicate timestamps
        if labels_df['timestamp'].duplicated().any():
            validation['is_valid'] = False
            validation['issues'].append('Duplicate timestamps found')
        
        # Check for invalid label values
        for col in ['flare6h_label', 'kp12h_label']:
            if col in labels_df.columns:
                invalid_values = labels_df[~labels_df[col].isin([0, 1])]
                if not invalid_values.empty:
                    validation['is_valid'] = False
                    validation['issues'].append(f'Invalid values in {col}')
        
        # Check for reasonable time ranges
        if len(labels_df) > 0:
            time_range = labels_df['timestamp'].max() - labels_df['timestamp'].min()
            if time_range < timedelta(hours=1):
                validation['issues'].append('Very short time range')
            elif time_range > timedelta(days=365):
                validation['issues'].append('Very long time range')
        
        return validation
