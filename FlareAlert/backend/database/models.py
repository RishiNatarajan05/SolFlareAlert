from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from datetime import datetime
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

Base = declarative_base()

class SolarFlare(Base):
    __tablename__ = "solar_flares"
    
    id = Column(Integer, primary_key=True)
    flare_id = Column(String, unique=True)
    begin_time = Column(DateTime)
    peak_time = Column(DateTime)
    end_time = Column(DateTime)
    class_type = Column(String)  # A, B, C, M, X
    class_value = Column(Float)  # 1.0, 2.3, etc.
    source_location = Column(String)
    active_region_num = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class CME(Base):
    __tablename__ = "cmes"
    
    id = Column(Integer, primary_key=True)
    cme_id = Column(String, unique=True)
    time21_5 = Column(DateTime)
    latitude = Column(Float)
    longitude = Column(Float)
    speed = Column(Float)
    half_angle = Column(Float)
    type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class GeomagneticStorm(Base):
    __tablename__ = "geomagnetic_storms"
    
    id = Column(Integer, primary_key=True)
    storm_id = Column(String, unique=True)
    time_tag = Column(DateTime)
    kp_index = Column(Float)
    dst_index = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
def create_database(db_path: str = None):
    if db_path is None:
        db_url = Config.get_database_url()
    else:
        db_url = f'sqlite:///{db_path}'
    
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine
