"""Intelligence service — advanced aviation metrics and network analysis"""
import math
from app.database import get_database


async def get_airport_connectivity(iata: str):
    db = get_database()
    iata = iata.upper()
    
    airport = await db.airports.find_one({"iata": iata}, {"_id": 0})
    if not airport:
        return None
    
    outbound = await db.routes.find({"source": iata}, {"_id": 0, "embedding": 0}).to_list(length=500)
    inbound = await db.routes.find({"dest": iata}, {"_id": 0, "embedding": 0}).to_list(length=500)
    
    connected_iatas = {r.get("dest") for r in outbound if r.get("dest")}
    connected_iatas.update({r.get("source") for r in inbound if r.get("source")})
    
    nodes = [{
        "iata": iata, "name": airport.get("name"), 
        "lat": airport.get("latitude"), "lng": airport.get("longitude"), 
        "country": airport.get("country"), "hub": True
    }]
    
    if connected_iatas:
        connected_airports = await db.airports.find(
            {"iata": {"$in": list(connected_iatas)}}, {"_id": 0}
        ).to_list(length=500)
        
        for a in connected_airports:
            nodes.append({
                "iata": a.get("iata"), "name": a.get("name"),
                "lat": a.get("latitude"), "lng": a.get("longitude"),
                "country": a.get("country"), "hub": False
            })
    
    edges = [{"source": iata, "target": r.get("dest"), "airline": r.get("airline")} for r in outbound if r.get("dest")]
    edges.extend([{"source": r.get("source"), "target": iata, "airline": r.get("airline")} for r in inbound if r.get("source")])
    
    return {
        "hub": iata,
        "total_connections": len(connected_iatas),
        "outbound_routes": len(outbound),
        "inbound_routes": len(inbound),
        "nodes": nodes,
        "edges": edges[:200]
    }


async def get_airline_comparison(codes: list):
    db = get_database()
    results = []
    for code in codes:
        airline = await db.airlines.find_one({"$or": [{"iata": code}, {"icao": code}]}, {"_id": 0})
        if not airline: continue
        
        route_count = await db.routes.count_documents({"airline": code})
        
        dest_pipeline = [{"$match": {"airline": code}}, {"$group": {"_id": "$dest"}}, {"$count": "total"}]
        dest_result = await db.routes.aggregate(dest_pipeline).to_list(length=1)
        unique_destinations = dest_result[0]["total"] if dest_result else 0
        
        source_pipeline = [{"$match": {"airline": code}}, {"$group": {"_id": "$source"}}, {"$count": "total"}]
        source_result = await db.routes.aggregate(source_pipeline).to_list(length=1)
        unique_sources = source_result[0]["total"] if source_result else 0
        
        country_pipeline = [
            {"$match": {"airline": code}},
            {"$lookup": {"from": "airports", "localField": "dest", "foreignField": "iata", "as": "dest_airport"}},
            {"$unwind": "$dest_airport"},
            {"$group": {"_id": "$dest_airport.country"}},
            {"$count": "total"}
        ]
        country_result = await db.routes.aggregate(country_pipeline).to_list(length=1)
        countries_served = country_result[0]["total"] if country_result else 0
        
        results.append({
            "code": code,
            "name": airline.get("name", "Unknown"),
            "total_routes": route_count,
            "unique_sources": unique_sources,
            "unique_destinations": unique_destinations,
            "countries_served": countries_served,
            "connectivity_score": round((route_count * 0.4 + unique_destinations * 0.3 + countries_served * 0.3) / 10, 2)
        })
    return sorted(results, key=lambda x: x["connectivity_score"], reverse=True)


async def get_network_graph(limit: int = 50):
    db = get_database()
    pipeline = [
        {"$group": {"_id": "$source", "outbound": {"$sum": 1}}},
        {"$sort": {"outbound": -1}},
        {"$limit": limit},
        {"$lookup": {"from": "airports", "localField": "_id", "foreignField": "iata", "as": "airport"}},
        {"$unwind": "$airport"},
        {"$project": {
            "_id": 0, "iata": "$_id", "name": "$airport.name",
            "lat": "$airport.latitude", "lng": "$airport.longitude", "outbound": 1
        }}
    ]
    nodes = await db.routes.aggregate(pipeline).to_list(length=limit)
    for n in nodes:
        n["lat"] = float(n.get("lat", 0) or 0)
        n["lng"] = float(n.get("lng", 0) or 0)
        
    node_iatas = {n["iata"] for n in nodes}
    
    edge_pipeline = [
        {"$match": {"source": {"$in": list(node_iatas)}, "dest": {"$in": list(node_iatas)}}},
        {"$group": {"_id": {"source": "$source", "dest": "$dest"}, "weight": {"$sum": 1}}},
        {"$project": {"_id": 0, "source": "$_id.source", "target": "$_id.dest", "weight": 1}}
    ]
    edges = await db.routes.aggregate(edge_pipeline).to_list(length=5000)
    return {
        "nodes": nodes, 
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }


async def get_global_heatmap():
    db = get_database()
    airports = await db.airports.find(
        {"latitude": {"$ne": None}, "longitude": {"$ne": None}},
        {"_id": 0, "iata": 1, "name": 1, "country": 1, "latitude": 1, "longitude": 1}
    ).to_list(length=None)
    
    pipeline = [{"$group": {"_id": "$source", "count": {"$sum": 1}}}]
    route_counts = await db.routes.aggregate(pipeline).to_list(length=None)
    route_map = {r["_id"]: r["count"] for r in route_counts}
    
    heatmap = [{
        "iata": a.get("iata", "N/A"),
        "name": a.get("name", "Unknown"),
        "country": a.get("country", "Unknown"),
        "lat": float(a["latitude"]) if a.get("latitude") else 0,
        "lng": float(a["longitude"]) if a.get("longitude") else 0,
        "intensity": route_map.get(a.get("iata"), 0)
    } for a in airports]

    return {
        "data": heatmap,
        "total_points": len(heatmap)
    }

async def get_country_index(limit: int = 50):
    db = get_database()
    pipeline = [
        {"$match": {"country": {"$ne": None, "$ne": ""}}},
        {"$group": {"_id": "$country", "airports": {"$sum": 1}}},
        {"$sort": {"airports": -1}},
        {"$limit": limit}
    ]
    countries = await db.airports.aggregate(pipeline).to_list(length=limit)
    
    results = []
    for c in countries:
        country = c["_id"]
        airports = c["airports"]
        airlines = await db.airlines.count_documents({"country": country})
        
        country_airports = await db.airports.find({"country": country}, {"iata": 1}).to_list(length=None)
        iatas = [a["iata"] for a in country_airports if "iata" in a]
        routes = await db.routes.count_documents({"source": {"$in": iatas}})
        
        idx = (airports * 2 + airlines * 5 + (routes * 0.5)) / 10
        results.append({
            "country": country,
            "airports": airports,
            "airlines": airlines,
            "routes": routes,
            "aviation_index": idx
        })
    results.sort(key=lambda x: x["aviation_index"], reverse=True)
    return results
