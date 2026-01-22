from fastapi import FastAPI, APIRouter, Query, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- ENV ----------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ---------- FASTAPI ----------
app = FastAPI(title="Flight Analytics API")

# Health route (for Render)
@app.get("/")
def root():
    return {"status": "ok"}

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MONGO ----------
client = AsyncIOMotorClient(
    MONGO_URL,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
    socketTimeoutMS=5000,
    tls=True
)
db = client[DB_NAME]

# ---------- ROUTER ----------
api_router = APIRouter(prefix="/api")

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- MODELS ----------
class Stats(BaseModel):
    total_airports: int
    total_airlines: int
    total_routes: int
    total_countries: int

class Airport(BaseModel):
    id: int
    name: str
    city: str
    country: str
    iata: Optional[str] = None
    icao: Optional[str] = None

class Route(BaseModel):
    source: str
    dest: str
    airline: Optional[str] = None

# ---------- ANALYTICS ----------
@api_router.get("/analytics/stats", response_model=Stats)
async def get_stats():
    try:
        total_airports = await db.airports.count_documents({})
        total_airlines = await db.airlines.count_documents({})
        total_routes = await db.routes.count_documents({})
        countries = await db.airports.distinct("country")

        return Stats(
            total_airports=total_airports,
            total_airlines=total_airlines,
            total_routes=total_routes,
            total_countries=len(countries)
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")

# ---------- BUSIEST AIRPORTS ----------
@api_router.get("/analytics/busiest-airports")
async def busiest_airports(limit: int = 10):
    pipeline = [
        {"$group": {"_id": "$dest_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {
            "$lookup": {
                "from": "airports",
                "localField": "_id",
                "foreignField": "id",
                "as": "airport"
            }
        },
        {"$unwind": "$airport"},
        {"$project": {"_id": 0, "airport": "$airport.name", "count": 1}}
    ]
    return await db.routes.aggregate(pipeline).to_list(length=limit)

# ---------- SEARCH ----------
@api_router.get("/search/airports")
async def search_airports(q: str, limit: int = 10):
    regex = {"$regex": q, "$options": "i"}
    cursor = db.airports.find(
        {"$or": [{"name": regex}, {"city": regex}, {"iata": regex}]}
    ).limit(limit)
    return await cursor.to_list(length=limit)

# ---------- ROUTES FOR AIRPORT ----------
@api_router.get("/search/routes/{airport_id}")
async def routes_for_airport(airport_id: int, limit: int = 20):
    cursor = db.routes.find(
        {"$or": [{"source_id": airport_id}, {"dest_id": airport_id}]}
    ).limit(limit)
    return await cursor.to_list(length=limit)

# ---------- RECOMMENDATIONS ----------
@api_router.get("/recommendations/direct-routes")
async def direct_routes(source: str, destination: str):
    return await db.routes.find(
        {"source": source, "dest": destination}
    ).to_list(length=50)

# ---------- INCLUDE ROUTER ----------
app.include_router(api_router)
