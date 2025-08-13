#!/usr/bin/env python3
"""
Simple script to view current data in the database
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database.models import SolarFlare, CME, GeomagneticStorm
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from config import Config

def view_data():
    """View current data in the database"""
    # Use the database in the backend directory (where ingestion creates it)
    engine = create_engine('sqlite:///backend/flarealert.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get counts
        flare_count = session.query(SolarFlare).count()
        cme_count = session.query(CME).count()
        storm_count = session.query(GeomagneticStorm).count()
        
        print("=== FlareAlert Database Status ===")
        print(f"Solar Flares: {flare_count}")
        print(f"CMEs: {cme_count}")
        print(f"Geomagnetic Storms: {storm_count}")
        print()
        
        # Show recent flares
        if flare_count > 0:
            print("=== Recent Solar Flares ===")
            recent_flares = session.query(SolarFlare).order_by(SolarFlare.begin_time.desc()).limit(5).all()
            for flare in recent_flares:
                print(f"• {flare.class_type}{flare.class_value} at {flare.begin_time.strftime('%Y-%m-%d %H:%M')}")
            print()
        
        # Show recent CMEs
        if cme_count > 0:
            print("=== Recent CMEs ===")
            recent_cmes = session.query(CME).order_by(CME.time21_5.desc()).limit(5).all()
            for cme in recent_cmes:
                print(f"• Speed: {cme.speed} km/s at {cme.time21_5.strftime('%Y-%m-%d %H:%M')}")
            print()
        
        # Show recent storms
        if storm_count > 0:
            print("=== Recent Geomagnetic Storms ===")
            recent_storms = session.query(GeomagneticStorm).order_by(GeomagneticStorm.time_tag.desc()).limit(5).all()
            for storm in recent_storms:
                print(f"• Kp: {storm.kp_index} at {storm.time_tag.strftime('%Y-%m-%d %H:%M')}")
            print()
            
    finally:
        session.close()

if __name__ == "__main__":
    view_data()
