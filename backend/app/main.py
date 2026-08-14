"""Main application factory — the new modular entry point"""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import close_connection
from app.routers import analytics, search, recommendations, admin, intelligence, health, ml
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)
    
    # --- Register routers under /api (backward compatible) ---
    from fastapi import APIRouter
    api_router = APIRouter(prefix="/api")
    api_router.include_router(analytics.router)
    api_router.include_router(search.router)
    api_router.include_router(recommendations.router)
    api_router.include_router(admin.router)
    app.include_router(api_router)
    
    # --- Register v2 routers (new intelligence endpoints) ---
    v2_router = APIRouter(prefix="/api/v2")
    v2_router.include_router(intelligence.router)
    v2_router.include_router(ml.router)
    app.include_router(v2_router)
    
    # --- Health check at root level ---
    app.include_router(health.router)
    
    # --- CORS middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=settings.CORS_ORIGINS.split(','),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # --- Root endpoints (backward compatible) ---
    @app.get("/")
    async def root():
        return {"message": "Flight Analytics API running"}
    
    @api_router.get("/")
    async def api_root():
        return {"message": "Flight Analytics API - Ready", "version": settings.APP_VERSION}
    
    # --- Lifecycle events ---
    @app.on_event("shutdown")
    async def shutdown():
        await close_connection()
    
    logger.info(f"🛫 {settings.APP_TITLE} v{settings.APP_VERSION} initialized")
    return app


# Create app instance
app = create_app()
