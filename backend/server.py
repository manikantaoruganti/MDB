"""
Flight Analytics API — Server Entry Point
==========================================
This file is preserved for backward compatibility with existing deployment scripts.
The actual application logic has been refactored into the `app/` package.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8001
    
    OR (new style):
    uvicorn app.main:app --host 0.0.0.0 --port 8001
"""
import sys
from pathlib import Path

# Ensure the backend directory is in the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the app from the new modular architecture
from app.main import app  # noqa: F401

# This preserves: `uvicorn server:app` compatibility
