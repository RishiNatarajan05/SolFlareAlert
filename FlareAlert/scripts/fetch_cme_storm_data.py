#!/usr/bin/env python3
"""
Script to fetch CME and storm data to test the fixed parsing
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

def fetch_cme_storm_data():
    """Fetch CME and storm data to test parsing"""
    print("=== FlareAlert CME/Storm Data Fetch ===\n")
    
    # Initialize clients
    donki_client = DONKIClient()
    ingestion = DataIngestion()
    
    # Fetch last 3 months of data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=90)
    
    print(f"Fetching CME and storm data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    
    # Format dates for API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        # Fetch data
        cmes = donki_client.get_cmes(start_str, end_str)
        storms = donki_client.get_geomagnetic_storms(start_str, end_str)
        
        print(f"Found {len(cmes)} CMEs and {len(storms)} storms")
        print()
        
        # Ingest data
        session = ingestion.Session()
        
        try:
            cmes_added = 0
            storms_added = 0
            
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
            print(f"Added: {cmes_added} CMEs, {storms_added} storms")
            
        except Exception as e:
            session.rollback()
            print(f"Error ingesting data: {e}")
        finally:
            session.close()
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_cme_storm_data()
