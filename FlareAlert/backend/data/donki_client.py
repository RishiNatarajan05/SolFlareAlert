import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

class DONKIClient:
    """NASA DONKI API client for fetching space weather data"""
    
    BASE_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_solar_flares(self, start_date: str, end_date: str) -> List[Dict]:
        """Fetch solar flare data from DONKI"""
        url = f"{self.BASE_URL}/FLR"
        params = {
            'startDate': start_date,
            'endDate': end_date
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching solar flares: {e}")
            return []
    
    def get_cmes(self, start_date: str, end_date: str) -> List[Dict]:
        """Fetch CME data from DONKI"""
        url = f"{self.BASE_URL}/CME"
        params = {
            'startDate': start_date,
            'endDate': end_date
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching CMEs: {e}")
            return []
    
    def get_geomagnetic_storms(self, start_date: str, end_date: str) -> List[Dict]:
        """Fetch geomagnetic storm data from DONKI"""
        url = f"{self.BASE_URL}/GST"
        params = {
            'startDate': start_date,
            'endDate': end_date
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching geomagnetic storms: {e}")
            return []
    
    def get_recent_data(self, hours_back: int = 72) -> Dict[str, List]:
        """Get recent data from the last N hours"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours_back)
        
        # Format dates for DONKI API (YYYY-MM-DD)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching data from {start_str} to {end_str}")
        
        return {
            'flares': self.get_solar_flares(start_str, end_str),
            'cmes': self.get_cmes(start_str, end_str),
            'storms': self.get_geomagnetic_storms(start_str, end_str)
        }
