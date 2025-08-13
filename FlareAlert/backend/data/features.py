import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func
from typing import Dict, List, Tuple
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from database.models import SolarFlare, CME, GeomagneticStorm

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for solar weather prediction"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use the database in the current directory
            db_url = 'sqlite:///flarealert.db'
        else:
            db_url = f'sqlite:///{db_path}'
        
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_flare_features(self, timestamp: datetime, hours_back: int = 72) -> Dict:
        """Extract flare-related features for a given timestamp"""
        session = self.Session()
        
        try:
            # Time window
            start_time = timestamp - timedelta(hours=hours_back)
            
            # Get flares in the window
            flares = session.query(SolarFlare).filter(
                SolarFlare.begin_time >= start_time,
                SolarFlare.begin_time < timestamp
            ).all()
            
            if not flares:
                return self._empty_flare_features()
            
            # Convert to DataFrame for easier analysis
            flare_data = []
            for flare in flares:
                flare_data.append({
                    'begin_time': flare.begin_time,
                    'class_type': flare.class_type,
                    'class_value': flare.class_value,
                    'active_region': flare.active_region_num
                })
            
            df = pd.DataFrame(flare_data)
            
            # Calculate features
            features = {}
            
            # Counts by class
            features['flare_count_6h'] = len(df[df['begin_time'] >= timestamp - timedelta(hours=6)])
            features['flare_count_24h'] = len(df[df['begin_time'] >= timestamp - timedelta(hours=24)])
            features['flare_count_72h'] = len(df)
            
            # M-class and above counts
            m_plus_flares = df[df['class_type'].isin(['M', 'X'])]
            features['m_plus_count_6h'] = len(m_plus_flares[m_plus_flares['begin_time'] >= timestamp - timedelta(hours=6)])
            features['m_plus_count_24h'] = len(m_plus_flares[m_plus_flares['begin_time'] >= timestamp - timedelta(hours=24)])
            features['m_plus_count_72h'] = len(m_plus_flares)
            
            # Max flare class (convert to numeric)
            if not df.empty:
                max_flare = df.loc[df['class_value'].idxmax()]
                # Convert flare class to numeric: A=1, B=2, C=3, M=4, X=5
                class_mapping = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
                features['max_flare_class'] = class_mapping.get(max_flare['class_type'], 1)
                features['max_flare_value'] = max_flare['class_value']
            else:
                features['max_flare_class'] = 1  # A class
                features['max_flare_value'] = 0.0
            
            # Time since last M/X flare
            m_x_flares = df[df['class_type'].isin(['M', 'X'])]
            if not m_x_flares.empty:
                last_m_x = m_x_flares['begin_time'].max()
                features['hours_since_last_mx'] = (timestamp - last_m_x).total_seconds() / 3600
            else:
                features['hours_since_last_mx'] = 72.0  # Default to window size
            
            # Active region features
            if not df.empty:
                features['unique_active_regions'] = df['active_region'].nunique()
                features['most_active_region'] = df['active_region'].mode().iloc[0] if not df['active_region'].mode().empty else 0
            else:
                features['unique_active_regions'] = 0
                features['most_active_region'] = 0
            
            return features
            
        finally:
            session.close()
    
    def get_cme_features(self, timestamp: datetime, hours_back: int = 72) -> Dict:
        """Extract CME-related features for a given timestamp"""
        session = self.Session()
        
        try:
            # Time window
            start_time = timestamp - timedelta(hours=hours_back)
            
            # Get CMEs in the window
            cmes = session.query(CME).filter(
                CME.time21_5 >= start_time,
                CME.time21_5 < timestamp
            ).all()
            
            if not cmes:
                return self._empty_cme_features()
            
            # Convert to DataFrame
            cme_data = []
            for cme in cmes:
                cme_data.append({
                    'time': cme.time21_5,
                    'speed': cme.speed,
                    'latitude': cme.latitude,
                    'longitude': cme.longitude,
                    'half_angle': cme.half_angle
                })
            
            df = pd.DataFrame(cme_data)
            
            # Calculate features
            features = {}
            
            # Counts
            features['cme_count_24h'] = len(df[df['time'] >= timestamp - timedelta(hours=24)])
            features['cme_count_72h'] = len(df)
            
            # Speed features
            if not df.empty:
                features['max_cme_speed'] = df['speed'].max()
                features['avg_cme_speed'] = df['speed'].mean()
                features['last_cme_speed'] = df.loc[df['time'].idxmax(), 'speed']
            else:
                features['max_cme_speed'] = 0.0
                features['avg_cme_speed'] = 0.0
                features['last_cme_speed'] = 0.0
            
            # Time since last CME
            if not df.empty:
                last_cme_time = df['time'].max()
                features['hours_since_last_cme'] = (timestamp - last_cme_time).total_seconds() / 3600
            else:
                features['hours_since_last_cme'] = 72.0
            
            # Width features
            if not df.empty:
                features['max_cme_width'] = df['half_angle'].max() * 2  # Convert to full angle
                features['avg_cme_width'] = df['half_angle'].mean() * 2
            else:
                features['max_cme_width'] = 0.0
                features['avg_cme_width'] = 0.0
            
            return features
            
        finally:
            session.close()
    
    def get_storm_features(self, timestamp: datetime, hours_back: int = 72) -> Dict:
        """Extract geomagnetic storm features for a given timestamp"""
        session = self.Session()
        
        try:
            # Time window
            start_time = timestamp - timedelta(hours=hours_back)
            
            # Get storms in the window
            storms = session.query(GeomagneticStorm).filter(
                GeomagneticStorm.time_tag >= start_time,
                GeomagneticStorm.time_tag < timestamp
            ).all()
            
            if not storms:
                return self._empty_storm_features()
            
            # Convert to DataFrame
            storm_data = []
            for storm in storms:
                storm_data.append({
                    'time': storm.time_tag,
                    'kp': storm.kp_index,
                    'dst': storm.dst_index
                })
            
            df = pd.DataFrame(storm_data)
            
            # Calculate features
            features = {}
            
            # Kp features
            if not df.empty:
                features['current_kp'] = df.loc[df['time'].idxmax(), 'kp']
                features['max_kp_24h'] = df[df['time'] >= timestamp - timedelta(hours=24)]['kp'].max()
                features['avg_kp_24h'] = df[df['time'] >= timestamp - timedelta(hours=24)]['kp'].mean()
                features['kp_storm_count_24h'] = len(df[(df['time'] >= timestamp - timedelta(hours=24)) & (df['kp'] >= 5)])
            else:
                features['current_kp'] = 0.0
                features['max_kp_24h'] = 0.0
                features['avg_kp_24h'] = 0.0
                features['kp_storm_count_24h'] = 0
            
            # Dst features
            if not df.empty:
                features['current_dst'] = df.loc[df['time'].idxmax(), 'dst']
                features['min_dst_24h'] = df[df['time'] >= timestamp - timedelta(hours=24)]['dst'].min()
            else:
                features['current_dst'] = 0.0
                features['min_dst_24h'] = 0.0
            
            return features
            
        finally:
            session.close()
    
    def get_all_features(self, timestamp: datetime) -> Dict:
        """Get all features for a given timestamp"""
        features = {}
        
        # Get features from each data type
        features.update(self.get_flare_features(timestamp))
        features.update(self.get_cme_features(timestamp))
        features.update(self.get_storm_features(timestamp))
        
        # Add enhanced time-based features
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['month'] = timestamp.month
        features['day_of_year'] = timestamp.timetuple().tm_yday
        
        # Add lag features (previous activity)
        features.update(self._get_lag_features(timestamp))
        
        # Add interaction features
        features.update(self._get_interaction_features(features))
        
        # Add rolling statistics
        features.update(self._get_rolling_features(timestamp))
        
        # Add enhanced CME features
        features.update(self._get_enhanced_cme_features(timestamp))
        
        return features
    
    def _empty_flare_features(self) -> Dict:
        """Return empty flare features"""
        return {
            'flare_count_6h': 0, 'flare_count_24h': 0, 'flare_count_72h': 0,
            'm_plus_count_6h': 0, 'm_plus_count_24h': 0, 'm_plus_count_72h': 0,
            'max_flare_class': 1, 'max_flare_value': 0.0,  # 1 = A class (numeric)
            'hours_since_last_mx': 72.0,
            'unique_active_regions': 0, 'most_active_region': 0
        }
    
    def _empty_cme_features(self) -> Dict:
        """Return empty CME features"""
        return {
            'cme_count_24h': 0, 'cme_count_72h': 0,
            'max_cme_speed': 0.0, 'avg_cme_speed': 0.0, 'last_cme_speed': 0.0,
            'hours_since_last_cme': 72.0,
            'max_cme_width': 0.0, 'avg_cme_width': 0.0
        }
    
    def _empty_storm_features(self) -> Dict:
        """Return empty storm features"""
        return {
            'current_kp': 0.0, 'max_kp_24h': 0.0, 'avg_kp_24h': 0.0, 'kp_storm_count_24h': 0,
            'current_dst': 0.0, 'min_dst_24h': 0.0
        }
    
    def _get_lag_features(self, timestamp: datetime) -> Dict:
        """Get lag features (previous activity patterns)"""
        session = self.Session()
        
        try:
            # Look back 7 days for lag patterns
            start_time = timestamp - timedelta(days=7)
            
            # Get flares in the lag window
            flares = session.query(SolarFlare).filter(
                SolarFlare.begin_time >= start_time,
                SolarFlare.begin_time < timestamp
            ).all()
            
            features = {}
            
            if flares:
                # Convert to DataFrame
                flare_data = []
                for flare in flares:
                    flare_data.append({
                        'begin_time': flare.begin_time,
                        'class_type': flare.class_type,
                        'class_value': flare.class_value
                    })
                
                df = pd.DataFrame(flare_data)
                
                # Lag features
                features['flare_count_7d'] = len(df)
                features['m_plus_count_7d'] = len(df[df['class_type'].isin(['M', 'X'])])
                
                # Recent activity (last 12h vs previous 6 days)
                recent_12h = df[df['begin_time'] >= timestamp - timedelta(hours=12)]
                previous_6d = df[df['begin_time'] < timestamp - timedelta(hours=12)]
                
                features['flare_activity_ratio'] = len(recent_12h) / max(len(previous_6d), 1)
                features['m_plus_activity_ratio'] = len(recent_12h[recent_12h['class_type'].isin(['M', 'X'])]) / max(len(previous_6d[previous_6d['class_type'].isin(['M', 'X'])]), 1)
            else:
                features['flare_count_7d'] = 0
                features['m_plus_count_7d'] = 0
                features['flare_activity_ratio'] = 0.0
                features['m_plus_activity_ratio'] = 0.0
            
            return features
            
        finally:
            session.close()
    
    def _get_interaction_features(self, base_features: Dict) -> Dict:
        """Get interaction features between different phenomena"""
        features = {}
        
        # Flare-CME interactions
        features['flare_cme_ratio'] = base_features.get('flare_count_24h', 0) / max(base_features.get('cme_count_24h', 1), 1)
        features['m_plus_cme_ratio'] = base_features.get('m_plus_count_24h', 0) / max(base_features.get('cme_count_24h', 1), 1)
        
        # Speed-width interactions
        features['cme_speed_width_product'] = base_features.get('avg_cme_speed', 0) * base_features.get('avg_cme_width', 0)
        features['cme_speed_width_ratio'] = base_features.get('avg_cme_speed', 0) / max(base_features.get('avg_cme_width', 1), 1)
        
        # Flare-Kp interactions
        features['flare_kp_product'] = base_features.get('flare_count_24h', 0) * base_features.get('current_kp', 0)
        features['m_plus_kp_product'] = base_features.get('m_plus_count_24h', 0) * base_features.get('current_kp', 0)
        
        # Time-based interactions
        features['hour_flare_interaction'] = base_features.get('hour_of_day', 0) * base_features.get('flare_count_24h', 0)
        features['day_flare_interaction'] = base_features.get('day_of_week', 0) * base_features.get('flare_count_24h', 0)
        
        return features
    
    def _get_rolling_features(self, timestamp: datetime) -> Dict:
        """Get rolling statistics features"""
        session = self.Session()
        
        try:
            # Look back 14 days for rolling statistics
            start_time = timestamp - timedelta(days=14)
            
            # Get flares for rolling stats
            flares = session.query(SolarFlare).filter(
                SolarFlare.begin_time >= start_time,
                SolarFlare.begin_time < timestamp
            ).all()
            
            features = {}
            
            if flares:
                # Convert to DataFrame
                flare_data = []
                for flare in flares:
                    flare_data.append({
                        'begin_time': flare.begin_time,
                        'class_type': flare.class_type,
                        'class_value': flare.class_value
                    })
                
                df = pd.DataFrame(flare_data)
                df = df.sort_values('begin_time')
                
                # Rolling averages (7-day windows)
                if len(df) >= 7:
                    # 7-day rolling flare count
                    df['date'] = df['begin_time'].dt.date
                    daily_flares = df.groupby('date').size().reset_index(name='flare_count')
                    features['rolling_7d_flare_avg'] = daily_flares['flare_count'].rolling(7, min_periods=1).mean().iloc[-1]
                    
                    # 7-day rolling M+ count
                    m_plus_daily = df[df['class_type'].isin(['M', 'X'])].groupby(df['begin_time'].dt.date).size().reset_index(name='m_plus_count')
                    if not m_plus_daily.empty:
                        features['rolling_7d_m_plus_avg'] = m_plus_daily['m_plus_count'].rolling(7, min_periods=1).mean().iloc[-1]
                    else:
                        features['rolling_7d_m_plus_avg'] = 0.0
                else:
                    features['rolling_7d_flare_avg'] = len(df) / 14.0  # Simple average
                    features['rolling_7d_m_plus_avg'] = len(df[df['class_type'].isin(['M', 'X'])]) / 14.0
            else:
                features['rolling_7d_flare_avg'] = 0.0
                features['rolling_7d_m_plus_avg'] = 0.0
            
            return features
            
        finally:
            session.close()
    
    def _get_enhanced_cme_features(self, timestamp: datetime) -> Dict:
        """Get enhanced CME features with more sophisticated analysis"""
        session = self.Session()
        
        try:
            # Look back 7 days for CME analysis
            start_time = timestamp - timedelta(days=7)
            
            # Get CMEs in the window
            cmes = session.query(CME).filter(
                CME.time21_5 >= start_time,
                CME.time21_5 < timestamp
            ).all()
            
            features = {}
            
            if cmes:
                # Convert to DataFrame
                cme_data = []
                for cme in cmes:
                    cme_data.append({
                        'time': cme.time21_5,
                        'speed': cme.speed,
                        'latitude': cme.latitude,
                        'longitude': cme.longitude,
                        'half_angle': cme.half_angle
                    })
                
                df = pd.DataFrame(cme_data)
                
                # Enhanced features
                features['cme_speed_std'] = df['speed'].std()  # Speed variability
                features['cme_width_std'] = (df['half_angle'] * 2).std()  # Width variability
                features['cme_speed_trend'] = self._calculate_trend(df['speed'])  # Speed trend
                features['cme_width_trend'] = self._calculate_trend(df['half_angle'] * 2)  # Width trend
                
                # Directional features
                features['cme_lat_std'] = df['latitude'].std()  # Latitude variability
                features['cme_lon_std'] = df['longitude'].std()  # Longitude variability
                
                # Recent vs historical
                recent_cmes = df[df['time'] >= timestamp - timedelta(hours=24)]
                if not recent_cmes.empty and len(df) > len(recent_cmes):
                    features['recent_cme_speed_ratio'] = recent_cmes['speed'].mean() / df['speed'].mean()
                    features['recent_cme_width_ratio'] = (recent_cmes['half_angle'] * 2).mean() / (df['half_angle'] * 2).mean()
                else:
                    features['recent_cme_speed_ratio'] = 1.0
                    features['recent_cme_width_ratio'] = 1.0
            else:
                features['cme_speed_std'] = 0.0
                features['cme_width_std'] = 0.0
                features['cme_speed_trend'] = 0.0
                features['cme_width_trend'] = 0.0
                features['cme_lat_std'] = 0.0
                features['cme_lon_std'] = 0.0
                features['recent_cme_speed_ratio'] = 1.0
                features['recent_cme_width_ratio'] = 1.0
            
            return features
            
        finally:
            session.close()
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend of a series"""
        if len(series) < 2:
            return 0.0
        
        try:
            # Simple linear trend (slope)
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            return slope
        except:
            return 0.0
