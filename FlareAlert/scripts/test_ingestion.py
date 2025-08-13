#!/usr/bin/env python3
"""
Test script for data ingestion pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from data.ingestion import DataIngestion
from data.donki_client import DONKIClient

def test_donki_client():
    """Test DONKI API client"""
    print("Testing DONKI API client...")
    client = DONKIClient()
    
    # Test with last 24 hours
    data = client.get_recent_data(24)
    
    print(f"Flares fetched: {len(data['flares'])}")
    print(f"CMEs fetched: {len(data['cmes'])}")
    print(f"Storms fetched: {len(data['storms'])}")
    
    if data['flares']:
        print(f"Sample flare: {data['flares'][0]}")
    
    return data

def test_ingestion():
    """Test data ingestion"""
    print("\nTesting data ingestion...")
    ingestion = DataIngestion()
    ingestion.ingest_data(24)  # Last 24 hours
    print("Ingestion test completed!")

if __name__ == "__main__":
    print("=== FlareAlert Data Ingestion Test ===\n")
    
    # Test DONKI client
    data = test_donki_client()
    
    # Test ingestion if we got data
    if any(data.values()):
        test_ingestion()
    else:
        print("No data received from DONKI API. Skipping ingestion test.")
    
    print("\n=== Test completed ===")
