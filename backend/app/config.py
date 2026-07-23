"""Centralized configuration management"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')


class Settings:
    """Application settings loaded from environment variables"""
    
    # MongoDB
    MONGO_URL: str = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    DB_NAME: str = os.environ.get('DB_NAME', 'flight_analytics')
    
    # CORS
    CORS_ORIGINS: str = os.environ.get('CORS_ORIGINS', '*')
    
    # App
    APP_TITLE: str = "Aviator AI - Aviation Intelligence Platform"
    APP_VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"
    API_V2_PREFIX: str = "/api/v2"
    
    # ML
    TFIDF_MAX_FEATURES: int = 128
    TFIDF_NGRAM_RANGE: tuple = (2, 5)
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 50
    
    # Caching
    CACHE_TTL: int = 300  # 5 minutes
    
    # Data
    DATA_DIR: Path = ROOT_DIR / 'data'


settings = Settings()
