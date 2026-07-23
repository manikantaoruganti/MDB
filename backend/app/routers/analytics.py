"""Analytics router — provides high-level aviation data insights"""
from fastapi import APIRouter, Query, HTTPException
from app.services import analytics_service
from app.schemas.analytics import PlatformStats, BusiestAirport, TopAirline, PopularRoute, CountryStat

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/stats", response_model=PlatformStats)
async def get_stats():
    """Get high-level platform statistics"""
    return await analytics_service.get_platform_stats()


@router.get("/busiest-airports", response_model=list[BusiestAirport])
async def get_busiest_airports(limit: int = Query(10, ge=1, le=100)):
    """Get airports with the highest number of outbound routes"""
    return await analytics_service.get_busiest_airports(limit)


@router.get("/top-airlines", response_model=list[TopAirline])
async def get_top_airlines(limit: int = Query(10, ge=1, le=100)):
    """Get airlines with the most active routes"""
    return await analytics_service.get_top_airlines(limit)


@router.get("/popular-routes", response_model=list[PopularRoute])
async def get_popular_routes(limit: int = Query(10, ge=1, le=100)):
    """Get the most common source-destination pairs"""
    return await analytics_service.get_popular_routes(limit)


@router.get("/airports-by-country", response_model=list[CountryStat])
async def get_airports_by_country(limit: int = Query(20, ge=1, le=200)):
    """Get airport distribution across different countries"""
    return await analytics_service.get_airports_by_country(limit)


@router.get("/trends")
async def get_trends():
    """Get equipment and stop distribution trends"""
    return await analytics_service.get_route_trends()
