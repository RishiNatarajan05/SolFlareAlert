#!/usr/bin/env python3
"""
Script to fetch a year's worth of historical solar weather data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import logging
from data.donki_client import DONKIClient
from data.ingestion import DataIngestion
from database.models import SolarFlare, CME, GeomagneticStorm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_historical_data():
    """Fetch 1 year of historical data for optimal precision"""
    print("=== FlareAlert 1-Year Historical Data Fetch ===\n")
    
    # Initialize clients
    donki_client = DONKIClient()
    ingestion = DataIngestion()
    
    # Calculate date range (1 year back from today)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365)  # 1 year = 365 days
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total days: {(end_date - start_date).days}")
    print(f"Expected precision: ~33% (optimal balance)")
    print()
    
    # Fetch data in chunks to avoid overwhelming the API
    chunk_size = 30  # 30 days per chunk
    current_start = start_date
    
    total_flares = 0
    total_cmes = 0
    total_storms = 0
    
    chunk_count = 0
    
    while current_start < end_date:
        chunk_count += 1
        current_end = min(current_start + timedelta(days=chunk_size), end_date)
        
        print(f"Chunk {chunk_count}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        # Format dates for API
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')
        
        try:
            # Fetch data for this chunk
            flares = donki_client.get_solar_flares(start_str, end_str)
            cmes = donki_client.get_cmes(start_str, end_str)
            storms = donki_client.get_geomagnetic_storms(start_str, end_str)
            
            print(f"  - Flares: {len(flares)}")
            print(f"  - CMEs: {len(cmes)}")
            print(f"  - Storms: {len(storms)}")
            
            # Ingest this chunk
            session = ingestion.Session()
            
            try:
                flares_added = 0
                cmes_added = 0
                storms_added = 0
                
                # Process flares
                for flare in flares:
                    parsed_flare = ingestion.parse_flare_data(flare)
                    if parsed_flare:
                        existing = session.query(SolarFlare).filter_by(flare_id=parsed_flare['flare_id']).first()
                        if not existing:
                            new_flare = SolarFlare(**parsed_flare)
                            session.add(new_flare)
                            flares_added += 1
                
                # Process CMEs
                for cme in cmes:
                    parsed_cme = ingestion.parse_cme_data(cme)
                    if parsed_cme:
                        existing = session.query(CME).filter_by(cme_id=parsed_cme['cme_id']).first()
                        if not existing:
                            new_cme = CME(**parsed_cme)
                            session.add(new_cme)
                            cmes_added += 1
                
                # Process storms
                for storm in storms:
                    parsed_storm = ingestion.parse_storm_data(storm)
                    if parsed_storm:
                        existing = session.query(GeomagneticStorm).filter_by(storm_id=parsed_storm['storm_id']).first()
                        if not existing:
                            new_storm = GeomagneticStorm(**parsed_storm)
                            session.add(new_storm)
                            storms_added += 1
                
                session.commit()
                print(f"  - Added: {flares_added} flares, {cmes_added} CMEs, {storms_added} storms")
                
                total_flares += flares_added
                total_cmes += cmes_added
                total_storms += storms_added
                
            except Exception as e:
                session.rollback()
                print(f"  - Error ingesting chunk: {e}")
            finally:
                session.close()
            
            # Add delay to be respectful to the API
            import time
            time.sleep(1)
            
        except Exception as e:
            print(f"  - Error fetching chunk: {e}")
        
        current_start = current_end
        print()
    
    print("=== 1-Year Historical Data Fetch Complete ===")
    print(f"Total data added:")
    print(f"  - Flares: {total_flares}")
    print(f"  - CMEs: {total_cmes}")
    print(f"  - Storms: {total_storms}")
    print()
    print("Ready to retrain models with optimal precision!")

if __name__ == "__main__":
    fetch_historical_data()
