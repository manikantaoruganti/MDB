"""Recommendation service — ML-powered route recommendations"""
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from app.database import get_database
from app.config import settings


async def get_similar_routes(source: str, destination: str, top_k: int = 10):
    """Existing TF-IDF similarity — preserved logic"""
    db = get_database()
    route_text = f"{source.upper()}-{destination.upper()}"
    all_routes = await db.routes.find(
        {"embedding": {"$exists": True}}, {"_id": 0}
    ).to_list(length=None)
    if not all_routes:
        return None

    embeddings, route_info = [], []
    for route in all_routes:
        try:
            emb = pickle.loads(route['embedding'])
            embeddings.append(emb)
            route_info.append({
                'route_text': route['route_text'],
                'source': route['source'],
                'dest': route['dest'],
                'airline': route.get('airline', 'Unknown')
            })
        except Exception:
            continue

    if not embeddings:
        return None

    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=settings.TFIDF_NGRAM_RANGE,
        max_features=settings.TFIDF_MAX_FEATURES
    )
    all_texts = [r['route_text'] for r in route_info]
    vectorizer.fit(all_texts)
    query_vector = vectorizer.transform([route_text]).toarray()
    matrix = np.vstack(embeddings)
    sims = cosine_similarity(query_vector, matrix)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    return [{**route_info[i], 'similarity': float(sims[i])} for i in top_idx]


async def get_direct_routes(source: str, destination: str):
    db = get_database()
    return await db.routes.find(
        {"source": source.upper(), "dest": destination.upper()},
        {"_id": 0, "embedding": 0},
    ).to_list(length=50)


async def get_hybrid_recommendations(source: str, destination: str, top_k: int = 10):
    """Hybrid recommendation combining TF-IDF similarity + geographic scoring"""
    db = get_database()
    tfidf_results = await get_similar_routes(source, destination, top_k * 2)
    if not tfidf_results:
        return []

    src_airport = await db.airports.find_one({"iata": source.upper()}, {"_id": 0})
    dst_airport = await db.airports.find_one({"iata": destination.upper()}, {"_id": 0})

    for r in tfidf_results:
        geo_score = 0.0
        r_src = await db.airports.find_one({"iata": r["source"]}, {"_id": 0})
        r_dst = await db.airports.find_one({"iata": r["dest"]}, {"_id": 0})
        if src_airport and r_src and dst_airport and r_dst:
            try:
                src_lat_diff = abs(src_airport.get("latitude", 0) - r_src.get("latitude", 0))
                dst_lat_diff = abs(dst_airport.get("latitude", 0) - r_dst.get("latitude", 0))
                geo_score = max(0, 1 - (src_lat_diff + dst_lat_diff) / 180)
            except Exception:
                pass
        r['geo_score'] = geo_score
        r['hybrid_score'] = round(r['similarity'] * 0.7 + geo_score * 0.3, 4)

    tfidf_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return tfidf_results[:top_k]
