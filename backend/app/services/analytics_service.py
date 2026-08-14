"""Analytics service — business logic for analytics endpoints"""
from app.database import get_database

async def get_platform_stats():
    db = get_database()
    total_airports = await db.airports.count_documents({})
    total_airlines = await db.airlines.count_documents({})
    total_routes = await db.routes.count_documents({})
    countries = await db.airports.distinct('country')
    total_countries = len([c for c in countries if c])
    return {
        "total_airports": total_airports,
        "total_airlines": total_airlines,
        "total_routes": total_routes,
        "total_countries": total_countries,
    }


async def get_busiest_airports(limit=10):
    db = get_database()
    pipeline = [
        {"$group": {"_id": "$dest_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$lookup": {"from": "airports", "localField": "_id", "foreignField": "id", "as": "airport"}},
        {"$unwind": "$airport"},
        {"$project": {
            "_id": 0, "airport_id": "$_id",
            "name": "$airport.name", "city": "$airport.city",
            "country": "$airport.country", "iata": "$airport.iata",
            "latitude": "$airport.latitude", "longitude": "$airport.longitude",
            "routes": "$count"
        }}
    ]
    return await db.routes.aggregate(pipeline).to_list(length=limit)


async def get_top_airlines(limit=10):
    db = get_database()
    pipeline = [
        {"$match": {"airline_id": {"$ne": None}}},
        {"$group": {"_id": "$airline_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$lookup": {"from": "airlines", "localField": "_id", "foreignField": "id", "as": "airline"}},
        {"$unwind": "$airline"},
        {"$project": {
            "_id": 0, "airline_id": "$_id",
            "name": "$airline.name", "iata": "$airline.iata",
            "country": "$airline.country", "routes": "$count"
        }}
    ]
    return await db.routes.aggregate(pipeline).to_list(length=limit)


async def get_popular_routes(limit=10):
    db = get_database()
    pipeline = [
        {"$group": {"_id": {"source_id": "$source_id", "dest_id": "$dest_id"}, "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$lookup": {"from": "airports", "localField": "_id.source_id", "foreignField": "id", "as": "src"}},
        {"$lookup": {"from": "airports", "localField": "_id.dest_id", "foreignField": "id", "as": "dst"}},
        {"$project": {
            "_id": 0,
            "source": {"$arrayElemAt": ["$src.iata", 0]},
            "source_name": {"$arrayElemAt": ["$src.name", 0]},
            "dest": {"$arrayElemAt": ["$dst.iata", 0]},
            "dest_name": {"$arrayElemAt": ["$dst.name", 0]},
            "airlines": "$count"
        }}
    ]
    return await db.routes.aggregate(pipeline).to_list(length=limit)


async def get_airports_by_country(limit=20):
    db = get_database()
    pipeline = [
        {"$match": {"country": {"$ne": None}}},
        {"$group": {"_id": "$country", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$project": {"_id": 0, "country": "$_id", "airports": "$count"}}
    ]
    return await db.airports.aggregate(pipeline).to_list(length=limit)


async def get_route_trends():
    """Get route distribution trends by equipment type and stops"""
    db = get_database()
    equipment_pipeline = [
        {"$match": {"equipment": {"$ne": None}}},
        {"$group": {"_id": "$equipment", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    stops_pipeline = [
        {"$group": {"_id": "$stops", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    equipment = await db.routes.aggregate(equipment_pipeline).to_list(length=15)
    stops = await db.routes.aggregate(stops_pipeline).to_list(length=10)
    return {"equipment_distribution": equipment, "stops_distribution": stops}
