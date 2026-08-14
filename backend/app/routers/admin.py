"""Admin router — preserves existing admin endpoints"""
from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest-data")
async def trigger_data_ingestion():
    """Trigger data ingestion (for admin use)"""
    try:
        from data_ingestion import run_full_ingestion
        result = await run_full_ingestion()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
