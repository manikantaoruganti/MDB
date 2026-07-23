"""Analytics schemas — preserved from original server.py and expanded"""
from pydantic import BaseModel
from typing import Optional, Any


class PlatformStats(BaseModel):
    total_airports: int
    total_airlines: int
    total_routes: int
    total_countries: int


class BusiestAirport(BaseModel):
    airport_id: Optional[int] = None
    name: str
    city: Optional[str] = None
    country: Optional[str] = None
    iata: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    routes: int


class TopAirline(BaseModel):
    airline_id: Optional[int] = None
    name: str
    iata: Optional[str] = None
    country: Optional[str] = None
    routes: int


class PopularRoute(BaseModel):
    source: Optional[str] = None
    source_name: Optional[str] = None
    dest: Optional[str] = None
    dest_name: Optional[str] = None
    airlines: int


class CountryStat(BaseModel):
    country: str
    airports: int

class RouteTrends(BaseModel):
    equipment_distribution: Any
    stops_distribution: Any
