import os
from typing import Optional

class Config:
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'flarealert.db')
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{DATABASE_PATH}')
    
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    S3_BUCKET = os.getenv('S3_BUCKET', 'flarealert-models')
    S3_MODEL_PREFIX = os.getenv('S3_MODEL_PREFIX', 'models')
    
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    INGESTION_HOURS_BACK = int(os.getenv('INGESTION_HOURS_BACK', '72'))
    INGESTION_INTERVAL_MINUTES = int(os.getenv('INGESTION_INTERVAL_MINUTES', '30'))
    
    @classmethod
    def get_database_url(cls) -> str:
        if cls.DATABASE_URL.startswith('sqlite:///'):
            return cls.DATABASE_URL
        else:
            return cls.DATABASE_URL
    
    @classmethod
    def is_aws_environment(cls) -> bool:
        return bool(cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY)
