"""Machine Learning service — models and predictive analytics"""
from app.database import get_database
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import random

async def get_route_clusters(n_clusters: int = 5, limit: int = 500):
    """Cluster routes using KMeans on their TF-IDF embeddings"""
    db = get_database()
    
    routes = await db.routes.find(
        {"embedding": {"$exists": True}},
        {"_id": 0, "source": 1, "dest": 1, "embedding": 1}
    ).to_list(length=limit)
    
    if not routes:
        return []
        
    embeddings = []
    valid_routes = []
    
    for r in routes:
        try:
            emb = pickle.loads(r["embedding"])
            embeddings.append(emb)
            valid_routes.append({"source": r["source"], "dest": r["dest"]})
        except Exception:
            continue
            
    if not embeddings:
        return []
        
    X = np.vstack(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Group routes by cluster
    clusters = [{"id": i, "routes": []} for i in range(n_clusters)]
    for idx, label in enumerate(labels):
        clusters[label]["routes"].append(valid_routes[idx])
        
    # Return top 5 routes per cluster for visualization
    for c in clusters:
        c["count"] = len(c["routes"])
        c["routes"] = c["routes"][:5]
        
    return clusters

async def get_anomalies(limit: int = 1000):
    """Detect anomalous routes using Isolation Forest on basic features"""
    db = get_database()
    
    # We will fetch routes and use basic features like distance (if we fetch coords)
    # For speed, we'll use a simplified feature set (e.g., number of airlines on route, direct/stops)
    
    # First get route counts
    pipeline = [
        {"$group": {
            "_id": {"source": "$source", "dest": "$dest"},
            "airline_count": {"$sum": 1},
            "avg_stops": {"$avg": "$stops"}
        }},
        {"$limit": limit}
    ]
    
    route_stats = await db.routes.aggregate(pipeline).to_list(length=limit)
    if not route_stats:
        return []
        
    features = []
    for r in route_stats:
        features.append([r["airline_count"], r.get("avg_stops", 0)])
        
    X = np.array(features)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    predictions = iso_forest.fit_predict(X)
    scores = iso_forest.score_samples(X)
    
    anomalies = []
    for idx, pred in enumerate(predictions):
        if pred == -1: # Anomaly
            r = route_stats[idx]
            anomalies.append({
                "source": r["_id"]["source"],
                "dest": r["_id"]["dest"],
                "airline_count": r["airline_count"],
                "avg_stops": r.get("avg_stops", 0),
                "anomaly_score": float(scores[idx])
            })
            
    # Sort by most anomalous
    anomalies.sort(key=lambda x: x["anomaly_score"])
    return anomalies[:20]

async def get_demand_forecast(iata: str):
    """Real ARIMA forecasting based on airport density and synthetic historical data"""
    db = get_database()
    iata = iata.upper()
    
    outbound = await db.routes.count_documents({"source": iata})
    inbound = await db.routes.count_documents({"dest": iata})
    base_demand = outbound + inbound
    
    if base_demand == 0:
        return {"error": "No data for this airport"}
        
    current_demand = base_demand * 1000
    
    np.random.seed(sum(ord(c) for c in iata))
    
    dates = pd.date_range(start='2021-01-01', periods=36, freq='ME')
    historical_data = []
    
    for i, date in enumerate(dates):
        trend = current_demand * (1 + (i * 0.005)) 
        seasonality = 1.0 + 0.25 * np.sin(date.month / 12.0 * 2 * np.pi)
        noise = np.random.uniform(0.9, 1.1)
        val = int(trend * seasonality * noise)
        historical_data.append(val)
        
    df = pd.DataFrame({'passengers': historical_data}, index=dates)
    
    try:
        model = ARIMA(df['passengers'], order=(1, 1, 1))
        results = model.fit()
        
        forecast_obj = results.get_forecast(steps=12)
        mean_forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)
        
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        forecast_payload = []
        
        for i in range(12):
            forecast_payload.append({
                "month": months[i],
                "passengers": int(mean_forecast.iloc[i]),
                "lower_bound": int(conf_int.iloc[i, 0]),
                "upper_bound": int(conf_int.iloc[i, 1])
            })
            
        return {
            "airport": iata,
            "base_route_count": base_demand,
            "forecast": forecast_payload
        }
        
    except Exception as e:
        return {"error": f"ARIMA Model failed to converge: {str(e)}"}
