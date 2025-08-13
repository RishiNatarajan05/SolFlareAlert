import os
from typing import Optional

class Config:
    """Configuration management for FlareAlert"""
    
    # Database configuration
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'flarealert.db')
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{DATABASE_PATH}')
    
    # AWS configuration (for future use)
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # S3 configuration (for model storage)
    S3_BUCKET = os.getenv('S3_BUCKET', 'flarealert-models')
    S3_MODEL_PREFIX = os.getenv('S3_MODEL_PREFIX', 'models')
    
    # API configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Data ingestion configuration
    INGESTION_HOURS_BACK = int(os.getenv('INGESTION_HOURS_BACK', '72'))
    INGESTION_INTERVAL_MINUTES = int(os.getenv('INGESTION_INTERVAL_MINUTES', '30'))
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with fallback to SQLite"""
        if cls.DATABASE_URL.startswith('sqlite:///'):
            return cls.DATABASE_URL
        else:
            # For PostgreSQL/MySQL in AWS
            return cls.DATABASE_URL
    
    @classmethod
    def is_aws_environment(cls) -> bool:
        """Check if running in AWS environment"""
        return bool(cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY)
