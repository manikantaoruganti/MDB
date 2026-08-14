"""Intelligence router — provides advanced network analysis and global insights"""
from fastapi import APIRouter, Query, HTTPException
from app.services import intelligence_service

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


@router.get("/airport-connectivity/{iata}")
async def get_airport_connectivity(iata: str):
    """Get network connectivity graph for a specific airport hub"""
    data = await intelligence_service.get_airport_connectivity(iata)
    if not data:
        raise HTTPException(status_code=404, detail="Airport not found")
    return data


@router.get("/airline-comparison")
async def compare_airlines(
    airlines: str = Query(..., description="Comma-separated airline IATA codes")
):
    """Compare multiple airlines across key performance indicators"""
    codes = [c.strip().upper() for c in airlines.split(",")][:5]
    return await intelligence_service.get_airline_comparison(codes)


@router.get("/network-graph")
async def get_network_graph(limit: int = Query(50, ge=10, le=200)):
    """Get the global aviation network graph data"""
    return await intelligence_service.get_network_graph(limit)


@router.get("/global-heatmap")
async def get_global_heatmap():
    """Get airport density and route intensity heatmap data"""
    return await intelligence_service.get_global_heatmap()

@router.get("/country-index")
async def get_country_index(limit: int = Query(50, ge=10, le=200)):
    """Get aviation index by country"""
    return await intelligence_service.get_country_index(limit)
