from fastapi import FastAPI, APIRouter, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pathlib import Path
from pydantic import BaseModel
<<<<<<< HEAD
import os
import logging
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
=======
>>>>>>> 1485d35c8503370179b6d6723497b41fcceb3176

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
    return Stats(
        total_airports=await db.airports.count_documents({}),
        total_airlines=await db.airlines.count_documents({}),
        total_routes=await db.routes.count_documents({}),
        total_countries=len(await db.airports.distinct("country")),
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
    return await db.airports.find(
        {"$or": [{"name": regex}, {"city": regex}, {"iata": regex}]},
        {"_id": 0},
    ).limit(limit).to_list(length=limit)

<<<<<<< HEAD
# ---------- ROUTES FOR AIRPORT ----------
@api_router.get("/search/routes/{iata}")
async def routes_for_airport(iata: str, limit: int = 20):
    return await db.routes.find(
        {"$or": [{"source": iata.upper()}, {"dest": iata.upper()}]},
        {"_id": 0, "embedding": 0},
    ).limit(limit).to_list(length=limit)
=======
# ---------- ROUTES FOR AIRPORT (FIXED) ----------
@api_router.get("/search/routes/{iata}")
async def routes_for_airport(iata: str, limit: int = 20):
    cursor = db.routes.find(
        {"$or": [{"source": iata.upper()}, {"dest": iata.upper()}]},
        {"_id": 0, "embedding": 0},
    ).limit(limit)

    return await cursor.to_list(length=limit)
>>>>>>> 1485d35c8503370179b6d6723497b41fcceb3176

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

# ---------- SIMILAR ROUTES (AI) ----------
@api_router.get("/recommendations/similar-routes")
async def similar_routes(
    source: str = Query(...),
    destination: str = Query(...),
    top_k: int = 10
):
    route_text = f"{source.upper()}-{destination.upper()}"

    docs = await db.routes.find(
        {"embedding": {"$exists": True}},
        {"_id": 0}
    ).to_list(length=None)

    if not docs:
        raise HTTPException(status_code=404, detail="No routes with embeddings")

    embeddings = []
    meta = []

    for d in docs:
        try:
            emb = d["embedding"]
            if isinstance(emb, (bytes, bytearray)):
                emb = pickle.loads(emb)

            embeddings.append(emb)
            meta.append({
                "source": d["source"],
                "dest": d["dest"],
                "airline": d.get("airline"),
                "route_text": d.get("route_text"),
            })
        except:
            continue

    if not embeddings:
        raise HTTPException(status_code=404, detail="No valid embeddings")

    embeddings = np.array(embeddings)

    texts = [m["route_text"] for m in meta]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=128,
    )
    vectorizer.fit(texts)

    query_vec = vectorizer.transform([route_text]).toarray()
    scores = cosine_similarity(query_vec, embeddings)[0]

    top_idx = scores.argsort()[-top_k:][::-1]

    return [
        {
            **meta[i],
            "similarity": float(scores[i])
        }
        for i in top_idx
    ]

# ---------- INCLUDE ROUTER ----------
app.include_router(api_router)
