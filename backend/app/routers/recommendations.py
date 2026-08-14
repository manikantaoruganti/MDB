"""Recommendations router — handles AI-powered route suggestions"""
from fastapi import APIRouter, Query, HTTPException
from app.services import recommendation_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/direct-routes")
async def get_direct_routes(
    source: str = Query(..., description="Source airport IATA code"),
    destination: str = Query(..., description="Destination airport IATA code")
):
    """Find all direct routes between two airports"""
    routes = await recommendation_service.get_direct_routes(source, destination)
    if not routes:
        return []
    return routes


@router.get("/similar-routes")
async def get_similar_routes(
    source: str = Query(..., description="Source airport IATA code"),
    destination: str = Query(..., description="Destination airport IATA code"),
    top_k: int = Query(10, ge=1, le=50)
):
    """Get AI-powered route recommendations using TF-IDF similarity"""
    recommendations = await recommendation_service.get_similar_routes(source, destination, top_k)
    if not recommendations:
        return []
    return recommendations


@router.get("/hybrid")
async def get_hybrid_recommendations(
    source: str = Query(..., description="Source airport IATA code"),
    destination: str = Query(..., description="Destination airport IATA code"),
    top_k: int = Query(10, ge=1, le=50)
):
    """Get hybrid recommendations combining TF-IDF and geographic scoring"""
    return await recommendation_service.get_hybrid_recommendations(source, destination, top_k)
