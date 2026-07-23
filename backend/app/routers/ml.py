"""Machine Learning router — experimental models and predictive analytics"""
from fastapi import APIRouter, Query, HTTPException
from app.services import ml_service

router = APIRouter(prefix="/ml", tags=["ml"])

@router.get("/clusters")
async def get_clusters(n_clusters: int = Query(5, ge=2, le=20), limit: int = Query(500, ge=100, le=2000)):
    """Get route clusters using KMeans on embeddings"""
    return await ml_service.get_route_clusters(n_clusters, limit)

@router.get("/anomalies")
async def get_anomalies(limit: int = Query(1000, ge=100, le=5000)):
    """Get anomalous routes using Isolation Forest"""
    return await ml_service.get_anomalies(limit)

@router.get("/forecast/{iata}")
async def get_forecast(iata: str):
    """Get demand forecast for a specific airport"""
    result = await ml_service.get_demand_forecast(iata)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
