# Flight Analytics & Recommendation System

[![Made with Emergent](https://img.shields.io/badge/Made%20with-Emergent-blue)](https://emergent.sh)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-green)](https://www.mongodb.com/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-blue)](https://reactjs.org/)

## Overview

A comprehensive flight analytics and route recommendation system built for the **MariaDB Hackathon 2025**. This project demonstrates advanced database features including **vector similarity search**, **high-performance analytics**, and **AI-powered recommendations** using the OpenFlights dataset.

### Live Demo

✈️ **[View Live Demo](please follow all commands to get live url)** ✈️

### Key Features

- 📊 **Real-time Analytics Dashboard**
  - Busiest airports by route count
  - Top airlines by route coverage
  - Most popular flight routes
  - Airport distribution by country
  - Overall statistics (7,698 airports, 6,162 airlines, 67,240 routes, 237 countries)

- 🤖 **AI-Powered Route Recommendations**
  - Vector similarity search using TF-IDF embeddings
  - Direct route discovery
  - Alternative route suggestions
  - Cosine similarity-based ranking
  - Search by IATA airport codes

- 🎨 **Modern UI/UX**
  - Glass-morphism design
  - Gradient color schemes
  - Smooth animations and transitions
  - Fully responsive layout
  - Dark theme optimized for readability

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **MongoDB** - NoSQL database with vector storage
- **scikit-learn** - TF-IDF vectorization and cosine similarity
- **Motor** - Async MongoDB driver
- **Python 3.11**

### Frontend
- **React 19** - UI library
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Beautiful UI components
- **Lucide React** - Modern icon library
- **Axios** - HTTP client

### Data
- **OpenFlights Dataset** - 7,698 airports, 6,162 airlines, 67,240 routes
- Source: [OpenFlights GitHub](https://github.com/jpatokal/openflights)

## Architecture

```
┌────────────────┐         ┌────────────────┐         ┌────────────────┐
│  React Frontend │───API──▶│  FastAPI Backend│───DB──▶│    MongoDB      │
│   Port 3000     │         │   Port 8001     │         │   Port 27017    │
└────────────────┘         └────────────────┘         └────────────────┘
                                    │
                                    v
                          ┌────────────────────┐
                          │  OpenFlights Data  │
                          │  (CSV Ingestion)   │
                          └────────────────────┘
```

## Project Structure

```
flight-analytics/
├── backend/
│   ├── server.py                 # FastAPI application with all endpoints
│   ├── data_ingestion.py         # Data loading and embedding generation
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # Environment variables
│   └── data/                     # OpenFlights CSV files
│       ├── airports.dat
│       ├── airlines.dat
│       └── routes.dat
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Main landing page
│   │   │   ├── Analytics.jsx      # Analytics dashboard
│   │   │   └── Recommendations.jsx # Route search & recommendations
│   │   ├── App.js                # Main app component with routing
│   │   ├── App.css               # Global styles
│   │   └── components/ui/        # shadcn/ui components
│   ├── package.json
│   ├── tailwind.config.js
│   └── .env
└── README.md
```

## Quick Start (Local Development)

### Prerequisites

- **Node.js** 16+ and **Yarn**
- **Python** 3.9+
- **MongoDB** (local or cloud instance)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd flight-analytics
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set:
# MONGO_URL=mongodb://localhost:27017
# DB_NAME=flight_analytics

# Download OpenFlights data (already included in data/ folder)
# Or re-download:
cd data
curl -sL "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat" -o airports.dat
curl -sL "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat" -o airlines.dat
curl -sL "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat" -o routes.dat
cd ..

# Ingest data into MongoDB
python3 data_ingestion.py

# Start backend server
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

Backend will be running at `http://localhost:8001`

API docs available at `http://localhost:8001/docs`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
yarn install

# Configure environment
cp .env.example .env
# Edit .env and set:
# REACT_APP_BACKEND_URL=http://localhost:8001

# Start frontend
yarn start
```

Frontend will be running at `http://localhost:3000`

## API Endpoints

### Analytics

- `GET /api/analytics/stats` - Overall statistics
- `GET /api/analytics/busiest-airports?limit=10` - Busiest airports by route count
- `GET /api/analytics/top-airlines?limit=10` - Top airlines by route coverage
- `GET /api/analytics/popular-routes?limit=10` - Most popular routes
- `GET /api/analytics/airports-by-country?limit=20` - Airports by country

### Search

- `GET /api/search/airports?q=london&limit=10` - Search airports by name/city/IATA
- `GET /api/search/routes/{airport_id}?limit=20` - Get routes for specific airport

### Recommendations

- `GET /api/recommendations/similar-routes?source=JFK&destination=LAX&top_k=10` - AI-powered similar routes
- `GET /api/recommendations/direct-routes?source=JFK&destination=LAX` - Direct routes between airports

### Example API Calls

```bash
# Get overall stats
curl http://localhost:8001/api/analytics/stats

# Get top 5 busiest airports
curl "http://localhost:8001/api/analytics/busiest-airports?limit=5"

# Search for airports in London
curl "http://localhost:8001/api/search/airports?q=london&limit=5"

# Get route recommendations for JFK to LAX
curl "http://localhost:8001/api/recommendations/similar-routes?source=JFK&destination=LAX&top_k=10"
```

## How It Works

### 1. Data Ingestion

```python
# Load OpenFlights CSVs
airports = pd.read_csv('airports.dat')
airlines = pd.read_csv('airlines.dat')
routes = pd.read_csv('routes.dat')

# Compute TF-IDF embeddings for routes
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
embeddings = vectorizer.fit_transform(routes['route_text'])

# Store in MongoDB
await db.routes.insert_many([{
    'source': row.source,
    'dest': row.dest,
    'embedding': pickle.dumps(embedding)
}])
```

### 2. Similarity Search

```python
# Transform query
query_vector = vectorizer.transform(["JFK-LAX"])

# Compute cosine similarity
similarities = cosine_similarity(query_vector, all_embeddings)

# Get top K results
top_indices = similarities.argsort()[-k:][::-1]
```

### 3. Analytics Queries

Using MongoDB aggregation pipelines:

```python
# Busiest airports
pipeline = [
    {"$group": {"_id": "$dest_id", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 10}
]
```

## Database Schema

### Collections

**airports**
```json
{
  "id": 3682,
  "name": "Hartsfield Jackson Atlanta International Airport",
  "city": "Atlanta",
  "country": "United States",
  "iata": "ATL",
  "icao": "KATL",
  "latitude": 33.6367,
  "longitude": -84.4281
}
```

**airlines**
```json
{
  "id": 324,
  "name": "American Airlines",
  "iata": "AA",
  "icao": "AAL",
  "country": "United States",
  "active": true
}
```

**routes**
```json
{
  "id": 1,
  "source": "JFK",
  "source_id": 3797,
  "dest": "LAX",
  "dest_id": 3484,
  "airline": "AA",
  "airline_id": 324,
  "route_text": "JFK-LAX",
  "embedding": "<binary_vector_data>"
}
```

## Machine Learning Implementation

### TF-IDF Vectorization

- **Input**: Route text (e.g., "JFK-LAX")
- **Method**: Character-level n-grams (2-5)
- **Output**: 128-dimensional sparse vector
- **Storage**: Serialized as binary in MongoDB

### Cosine Similarity

- **Formula**: `similarity = (A · B) / (||A|| * ||B||)`
- **Range**: 0 to 1 (0 = completely different, 1 = identical)
- **Use Case**: Finding similar routes based on pattern matching

## Performance Optimization

- **Indexing**: Created indexes on frequently queried fields (IATA codes, IDs)
- **Batch Processing**: Routes inserted in batches of 1,000 during ingestion
- **Async Operations**: All database operations are asynchronous
- **Caching**: Frontend caches API responses
- **Lazy Loading**: Paginated results for large datasets

## Deployment

### Deploy to Emergent (Recommended)

This project is already deployed on Emergent:

🚀 **[Live Demo](https://routeradar-1.preview.emergentagent.com)** 🚀

### Deploy to Other Platforms

**Docker Deployment** (create `docker-compose.yml`):

```yaml
version: '3.8'
services:
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - MONGO_URL=mongodb://mongodb:27017
      - DB_NAME=flight_analytics
    depends_on:
      - mongodb

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_BACKEND_URL=http://localhost:8001
    depends_on:
      - backend

volumes:
  mongo_data:
```

Run with:
```bash
docker-compose up -d
```

## Testing

### Backend API Tests

```bash
# Test stats endpoint
curl http://localhost:8001/api/analytics/stats

# Test recommendations
curl "http://localhost:8001/api/recommendations/similar-routes?source=JFK&destination=LAX&top_k=5"
```

### Frontend Tests

1. Navigate to `http://localhost:3000`
2. Verify dashboard loads with statistics
3. Click "View Analytics" and check all tabs
4. Click "Find Routes" and search for JFK → LAX
5. Verify recommendations appear with similarity scores

## Features Demonstrated

✅ **Database Operations**
- Complex aggregation pipelines
- Vector storage (embeddings as binary)
- Multi-collection joins
- Indexing strategies

✅ **Machine Learning**
- TF-IDF vectorization
- Cosine similarity ranking
- Feature engineering

✅ **API Design**
- RESTful endpoints
- Query parameters
- Error handling
- CORS configuration

✅ **Frontend Development**
- Modern React with hooks
- Responsive design
- Smooth animations
- Component architecture

## Future Enhancements

- [ ] MariaDB ColumnStore integration for faster analytics
- [ ] Galera Cluster for high availability
- [ ] Real-time flight status integration
- [ ] Interactive route maps (Leaflet/Mapbox)
- [ ] User authentication and saved searches
- [ ] Export analytics to CSV/PDF
- [ ] GraphQL API support
- [ ] Mobile app (React Native)
- [ ] Enhanced ML models (BERT embeddings, neural networks)

## Troubleshooting

### Backend won't start
```bash
# Check MongoDB is running
mongosh --eval "db.serverStatus()"

# Check Python dependencies
pip install -r requirements.txt

# Check logs
tail -f /var/log/supervisor/backend.err.log
```

### Frontend won't start
```bash
# Clear cache and reinstall
rm -rf node_modules yarn.lock
yarn install

# Check backend URL in .env
cat frontend/.env
```

### Data ingestion fails
```bash
# Ensure data files exist
ls -la backend/data/

# Re-download data
cd backend/data
curl -sL "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat" -o airports.dat
```

## Contributing

This is a hackathon project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - feel free to use this project for learning or hackathons!

## Acknowledgments

- **OpenFlights** - For providing comprehensive aviation data
- **MariaDB Foundation** - For hosting the hackathon
- **FastAPI** - For the excellent Python web framework
- **React & Tailwind CSS** - For the modern frontend stack
- **shadcn/ui** - For beautiful UI components
- **Emergent** - For the deployment platform

## Contact

Built for **MariaDB Hackathon 2025**

---

**Made with ❤️ and lots of ☕ by your name**

🚀 **[View Live Demo](run localhost)** 🚀
