from fastapi import FastAPI, APIRouter, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel

# ---------- ENV ----------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ---------- FASTAPI ----------
app = FastAPI(title="Flight Analytics API")

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
client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
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

# ---------- ANALYTICS ----------
@api_router.get("/analytics/stats", response_model=Stats)
async def get_stats():
    total_airports = await db.airports.count_documents({})
    total_airlines = await db.airlines.count_documents({})
    total_routes = await db.routes.count_documents({})
    countries = await db.airports.distinct("country")

    return Stats(
        total_airports=total_airports,
        total_airlines=total_airlines,
        total_routes=total_routes,
        total_countries=len(countries),
    )

# ---------- BUSIEST AIRPORTS (FIXED) ----------
@api_router.get("/analytics/busiest-airports")
async def busiest_airports(limit: int = 10):
    pipeline = [
        {"$group": {"_id": "$dest", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$project": {"_id": 0, "airport": "$_id", "routes": "$count"}},
    ]
    return await db.routes.aggregate(pipeline).to_list(length=limit)

# ---------- SEARCH AIRPORTS ----------
@api_router.get("/search/airports")
async def search_airports(q: str, limit: int = 10):
    regex = {"$regex": q, "$options": "i"}
    cursor = db.airports.find(
        {"$or": [{"name": regex}, {"city": regex}, {"iata": regex}]},
        {"_id": 0},
    ).limit(limit)

    return await cursor.to_list(length=limit)

# ---------- ROUTES FOR AIRPORT (FIXED) ----------
@api_router.get("/search/routes/{iata}")
async def routes_for_airport(iata: str, limit: int = 20):
    cursor = db.routes.find(
        {"$or": [{"source": iata.upper()}, {"dest": iata.upper()}]},
        {"_id": 0, "embedding": 0},
    ).limit(limit)

    return await cursor.to_list(length=limit)

# ---------- DIRECT ROUTES ----------
@api_router.get("/recommendations/direct-routes")
async def direct_routes(
    source: str = Query(...),
    destination: str = Query(...)
):
    return await db.routes.find(
        {"source": source.upper(), "dest": destination.upper()},
        {"_id": 0, "embedding": 0},
    ).to_list(length=50)

# ---------- INCLUDE ROUTER ----------
app.include_router(api_router)
