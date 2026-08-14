"""Search service — logic for finding airports and routes"""
from app.database import get_database


async def search_airports(query: str, limit: int = 20):
    db = get_database()
    search_filter = {
        "$or": [
            {"name": {"$regex": query, "$options": "i"}},
            {"city": {"$regex": query, "$options": "i"}},
            {"iata": {"$regex": f"^{query}$", "$options": "i"}},
            {"icao": {"$regex": f"^{query}$", "$options": "i"}}
        ]
    }
    return await db.airports.find(search_filter, {"_id": 0}).to_list(length=limit)


async def get_airport_by_iata(iata: str):
    db = get_database()
    return await db.airports.find_one({"iata": iata.upper()}, {"_id": 0})


async def get_routes_by_airport(airport_id: str, limit: int = 50):
    db = get_database()
    # Search by either IATA or numeric ID if applicable
    return await db.routes.find(
        {"$or": [{"source": airport_id.upper()}, {"source_id": airport_id}]},
        {"_id": 0, "embedding": 0}
    ).to_list(length=limit)
