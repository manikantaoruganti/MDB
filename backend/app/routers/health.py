"""Health check router"""
from fastapi import APIRouter
from app.database import get_database
import time

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    start = time.time()
    try:
        db = get_database()
        await db.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    latency = round((time.time() - start) * 1000, 2)
    
    return {
        "status": "healthy",
        "database": db_status,
        "latency_ms": latency,
        "version": "2.0.0"
    }
