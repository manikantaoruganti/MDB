"""MongoDB connection manager with connection pooling"""
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

# Singleton MongoDB client
_client = None
_db = None


def get_client() -> AsyncIOMotorClient:
    """Get or create the MongoDB client"""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGO_URL)
    return _client


def get_database():
    """Get the database instance"""
    global _db
    if _db is None:
        client = get_client()
        _db = client[settings.DB_NAME]
    return _db


async def close_connection():
    """Close the MongoDB connection"""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


# Convenience accessor
db = property(lambda self: get_database())
