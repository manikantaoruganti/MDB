"""
Data ingestion script for OpenFlights dataset
- Ingests airports, airlines, routes
- Generates TF-IDF embeddings
- Saves TF-IDF vectorizer as .pkl
"""

import asyncio
import os
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# --------------------------------------------------
# PATHS & ENV
# --------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
VECTORIZER_PATH = ROOT_DIR / "tfidf_vectorizer.pkl"

load_dotenv(ROOT_DIR / ".env")

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# MONGODB
# --------------------------------------------------
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

# --------------------------------------------------
# AIRPORTS
# --------------------------------------------------
async def ingest_airports():
    logger.info("Ingesting airports...")

    cols = [
        "id", "name", "city", "country", "iata", "icao",
        "latitude", "longitude", "altitude", "timezone",
        "dst", "tz", "type", "source"
    ]

    df = pd.read_csv(DATA_DIR / "airports.dat", header=None, names=cols, na_values="\\N")
    df = df[["id", "name", "city", "country", "iata", "icao"]]
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    await db.airports.delete_many({})
    await db.airports.insert_many(df.to_dict("records"))

    await db.airports.create_index("id", unique=True)
    await db.airports.create_index("iata")
    await db.airports.create_index("country")

    logger.info(f"Ingested {len(df)} airports")
    return len(df)

# --------------------------------------------------
# AIRLINES
# --------------------------------------------------
async def ingest_airlines():
    logger.info("Ingesting airlines...")

    cols = ["id", "name", "alias", "iata", "icao", "callsign", "country", "active"]
    df = pd.read_csv(DATA_DIR / "airlines.dat", header=None, names=cols, na_values="\\N")

    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df["active"] = df["active"] == "Y"

    await db.airlines.delete_many({})
    await db.airlines.insert_many(df.to_dict("records"))

    await db.airlines.create_index("id", unique=True)
    await db.airlines.create_index("iata")
    await db.airlines.create_index("country")

    logger.info(f"Ingested {len(df)} airlines")
    return len(df)

# --------------------------------------------------
# ROUTES + EMBEDDINGS
# --------------------------------------------------
async def ingest_routes_with_embeddings():
    logger.info("Ingesting routes and computing embeddings...")

    cols = [
        "airline", "airline_id", "source", "source_id",
        "dest", "dest_id", "codeshare", "stops", "equipment"
    ]

    df = pd.read_csv(DATA_DIR / "routes.dat", header=None, names=cols, na_values="\\N")
    df = df.dropna(subset=["source", "dest"])

    df["route_text"] = df["source"].astype(str) + "-" + df["dest"].astype(str)

    # 🔥 TF-IDF VECTORIZATION
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=128
    )

    vectors = vectorizer.fit_transform(df["route_text"]).toarray()

    # 🔥 SAVE VECTORIZER AS .PKL (CRITICAL)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    logger.info(f"Saved TF-IDF vectorizer to {VECTORIZER_PATH}")

    # CLEAN COLLECTION
    await db.routes.delete_many({})

    # REMOVE OLD BAD INDEX IF EXISTS
    indexes = await db.routes.index_information()
    if "id_1" in indexes:
        await db.routes.drop_index("id_1")
        logger.info("Dropped legacy id_1 index")

    records = []
    for i, row in df.iterrows():
        records.append({
            "source": row["source"],
            "dest": row["dest"],
            "airline": row["airline"],
            "route_text": row["route_text"],
            "embedding": vectors[i].tolist()  # ✅ numeric array
        })

        if len(records) == 1000:
            await db.routes.insert_many(records)
            records.clear()

    if records:
        await db.routes.insert_many(records)

    await db.routes.create_index("source")
    await db.routes.create_index("dest")

    logger.info(f"Ingested {len(df)} routes with embeddings")
    return len(df)

# --------------------------------------------------
# FULL PIPELINE
# --------------------------------------------------
async def run_full_ingestion():
    logger.info("Starting full data ingestion...")

    airports = await ingest_airports()
    airlines = await ingest_airlines()
    routes = await ingest_routes_with_embeddings()

    logger.info(
        f"Ingestion complete → Airports: {airports}, "
        f"Airlines: {airlines}, Routes: {routes}"
    )

    return {
        "airports": airports,
        "airlines": airlines,
        "routes": routes
    }

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_full_ingestion())
