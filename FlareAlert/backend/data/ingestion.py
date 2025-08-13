from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import Base, SolarFlare, CME, GeomagneticStorm, create_database
from data.donki_client import DONKIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.engine = create_database()
        self.Session = sessionmaker(bind=self.engine)
        self.donki_client = DONKIClient()
    
    def parse_flare_data(self, flare_data: dict) -> dict:
        """Parse and normalize flare data"""
        try:
            # Parse flare class (e.g., 'M1.5' -> class_type='M', class_value=1.5)
            class_type_raw = flare_data.get('classType', '')
            class_type = ''
            class_value = 0.0
            
            if class_type_raw:
                # Extract letter and number from class string
                import re
                match = re.match(r'([ABCXM])(\d+\.?\d*)', class_type_raw)
                if match:
                    class_type = match.group(1)  # A, B, C, M, or X
                    class_value = float(match.group(2))  # 1.5, 2.3, etc.
                else:
                    # Fallback: try to extract just the letter
                    class_type = class_type_raw[0] if class_type_raw else ''
                    class_value = 0.0
            
            return {
                'flare_id': flare_data.get('flrID', ''),
                'begin_time': datetime.fromisoformat(flare_data.get('beginTime', '').replace('Z', '+00:00')),
                'peak_time': datetime.fromisoformat(flare_data.get('peakTime', '').replace('Z', '+00:00')),
                'end_time': datetime.fromisoformat(flare_data.get('endTime', '').replace('Z', '+00:00')),
                'class_type': class_type,
                'class_value': class_value,
                'source_location': flare_data.get('sourceLocation', ''),
                'active_region_num': int(flare_data.get('activeRegionNum', 0))
            }
        except Exception as e:
            logger.error(f"Error parsing flare data: {e}")
            return None
    
    def parse_cme_data(self, cme_data: dict) -> dict:
        """Parse and normalize CME data"""
        try:
            # CME data has nested structure with cmeAnalyses array
            cme_analyses = cme_data.get('cmeAnalyses', [])
            if not cme_analyses:
                return None  # Skip CMEs without analysis data
            
            # Use the most accurate analysis (usually the first one)
            analysis = cme_analyses[0]
            
            time_str = analysis.get('time21_5', '')
            if not time_str or time_str.strip() == '':
                return None  # Skip CMEs with empty timestamps
                
            return {
                'cme_id': cme_data.get('activityID', ''),
                'time21_5': datetime.fromisoformat(time_str.replace('Z', '+00:00')),
                'latitude': float(analysis.get('latitude', 0)),
                'longitude': float(analysis.get('longitude', 0)),
                'speed': float(analysis.get('speed', 0)),
                'half_angle': float(analysis.get('halfAngle', 0)),
                'type': analysis.get('type', '')
            }
        except Exception as e:
            logger.debug(f"Skipping CME with invalid data: {e}")
            return None
    
    def parse_storm_data(self, storm_data: dict) -> dict:
        """Parse and normalize geomagnetic storm data"""
        try:
            # Storm data has nested structure with allKpIndex array
            kp_data = storm_data.get('allKpIndex', [])
            if not kp_data:
                return None  # Skip storms without Kp data
            
            # Use the first Kp measurement
            kp_measurement = kp_data[0]
            
            time_str = kp_measurement.get('observedTime', '')
            if not time_str or time_str.strip() == '':
                return None  # Skip storms with empty timestamps
                
            return {
                'storm_id': storm_data.get('gstID', ''),
                'time_tag': datetime.fromisoformat(time_str.replace('Z', '+00:00')),
                'kp_index': float(kp_measurement.get('kpIndex', 0)),
                'dst_index': 0.0  # DST not available in this API
            }
        except Exception as e:
            logger.debug(f"Skipping storm with invalid data: {e}")
            return None
    
    def ingest_data(self, hours_back: int = 72):
        """Main ingestion function"""
        logger.info(f"Starting data ingestion for last {hours_back} hours...")
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Fetch data from DONKI
        data = self.donki_client.get_recent_data(hours_back)
        
        logger.info(f"Fetched {len(data['flares'])} flares, {len(data['cmes'])} CMEs, {len(data['storms'])} storms from API")
        
        session = self.Session()
        
        try:
            flares_added = 0
            cmes_added = 0
            storms_added = 0
            
            # Process flares
            for flare in data['flares']:
                parsed_flare = self.parse_flare_data(flare)
                if parsed_flare:
                    # Check if already exists
                    existing = session.query(SolarFlare).filter_by(flare_id=parsed_flare['flare_id']).first()
                    if not existing:
                        new_flare = SolarFlare(**parsed_flare)
                        session.add(new_flare)
                        flares_added += 1
            
            # Process CMEs
            for cme in data['cmes']:
                parsed_cme = self.parse_cme_data(cme)
                if parsed_cme:
                    existing = session.query(CME).filter_by(cme_id=parsed_cme['cme_id']).first()
                    if not existing:
                        new_cme = CME(**parsed_cme)
                        session.add(new_cme)
                        cmes_added += 1
            
            # Process storms
            for storm in data['storms']:
                parsed_storm = self.parse_storm_data(storm)
                if parsed_storm:
                    existing = session.query(GeomagneticStorm).filter_by(storm_id=parsed_storm['storm_id']).first()
                    if not existing:
                        new_storm = GeomagneticStorm(**parsed_storm)
                        session.add(new_storm)
                        storms_added += 1
            
            session.commit()
            logger.info(f"Data ingestion completed successfully! Added: {flares_added} flares, {cmes_added} CMEs, {storms_added} storms")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error during ingestion: {e}")
        finally:
            session.close()

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.ingest_data()
