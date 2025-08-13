#!/usr/bin/env python3
"""
Setup script for FlareAlert backend
"""

import os
import sys
from database.models import create_database

def setup_backend():
    """Initialize backend environment"""
    print("Setting up FlareAlert backend...")
    
    # Create database
    print("Creating database...")
    engine = create_database()
    print("Database created successfully!")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Backend setup completed!")

if __name__ == "__main__":
    setup_backend()
