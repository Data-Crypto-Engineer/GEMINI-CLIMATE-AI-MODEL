import os
import math
from typing import Dict, Any, List, Optional

# Geographic bounds of Pakistan
PAKISTAN_BOUNDS = {
    "min_lat": 23.5,
    "max_lat": 37.1,
    "min_lon": 60.8,
    "max_lon": 77.8,
}

# Major Administrative Regions
PAKISTAN_REGIONS = [
    "Punjab",
    "Sindh",
    "Khyber Pakhtunkhwa",
    "Balochistan",
    "Gilgit-Baltistan",
    "Azad Jammu & Kashmir",
    "Islamabad Capital Territory"
]

# Risk Categories and Visual Design tokens
RISK_CATEGORIES = ["LOW", "MODERATE", "HIGH", "EXTREME"]

RISK_THRESHOLDS = {
    "LOW": (0, 25),
    "MODERATE": (26, 55),
    "HIGH": (56, 80),
    "EXTREME": (81, 100),
}

RISK_COLORS = {
    "LOW": "#10b981",       # Emerald Green
    "MODERATE": "#f59e0b",  # Amber Yellow
    "HIGH": "#f97316",      # Bright Orange
    "EXTREME": "#ef4444",   # Crimson Red
    "INSUFFICIENT": "#6b7280" # Slate Gray
}

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate if geographic coordinates fall within Pakistan territory."""
    try:
        lat = float(lat)
        lon = float(lon)
        return (
            PAKISTAN_BOUNDS["min_lat"] <= lat <= PAKISTAN_BOUNDS["max_lat"]
            and PAKISTAN_BOUNDS["min_lon"] <= lon <= PAKISTAN_BOUNDS["max_lon"]
        )
    except (ValueError, TypeError):
        return False

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth in kilometers."""
    r = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(r * c, 2)

def classify_risk_score(score: Optional[float]) -> str:
    """Classify numerical risk score (0-100) into defined categories."""
    if score is None or math.isnan(score):
        return "INSUFFICIENT"
    score = float(score)
    if score <= 25:
        return "LOW"
    elif score <= 55:
        return "MODERATE"
    elif score <= 80:
        return "HIGH"
    else:
        return "EXTREME"

def get_risk_color(category: str) -> str:
    """Get standard UI color hex for risk category."""
    return RISK_COLORS.get(category.upper(), "#6b7280")

def format_unit(value: Any, unit: str, decimal_places: int = 1) -> str:
    """Safely format numbers with appropriate climate metric units."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"N/A {unit}"
    try:
        val_float = float(value)
        return f"{val_float:.{decimal_places}f} {unit}"
    except (ValueError, TypeError):
        return f"{value} {unit}"

def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely cast value to float or return fallback default."""
    if value is None:
        return default
    try:
        f = float(value)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default

def get_project_root() -> str:
    """Get absolute path to climate_pakistan_ai root directory."""
    return os.path.dirname(os.path.abspath(__file__))
