"""Search router — handles airport and route lookups"""
from fastapi import APIRouter, Query, HTTPException
from app.services import search_service

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/airports")
async def search_airports(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=100)
):
    """Search for airports by name, city, or IATA/ICAO code"""
    return await search_service.search_airports(q, limit)


@router.get("/airport/{iata}")
async def get_airport(iata: str):
    """Get detailed information for a specific airport"""
    airport = await search_service.get_airport_by_iata(iata)
    if not airport:
        raise HTTPException(status_code=404, detail="Airport not found")
    return airport


@router.get("/routes/{airport_id}")
async def get_routes(
    airport_id: str,
    limit: int = Query(50, ge=1, le=500)
):
    """Get all outbound routes for a specific airport"""
    return await search_service.get_routes_by_airport(airport_id, limit)
