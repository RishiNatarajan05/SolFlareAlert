#!/usr/bin/env python3
"""
Database initialization script for FlareAlert
Creates all required tables
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database.models import create_database, Base
from config import Config

def init_database():
    """Initialize the database with all required tables"""
    print("=== FlareAlert Database Initialization ===")
    
    try:
        # Create database and tables
        engine = create_database()
        print(f"Database created successfully at: {Config.DATABASE_PATH}")
        print("Tables created:")
        
        # List all tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        for table_name in inspector.get_table_names():
            print(f"  - {table_name}")
        
        print("\nDatabase initialization complete!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    init_database()
