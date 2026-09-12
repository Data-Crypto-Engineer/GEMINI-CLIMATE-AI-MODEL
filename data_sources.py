"""
Data Sources Module for Climate Pakistan AI
Strict Extensibility: One function per data source.
Registered in DATA_SOURCE_REGISTRY for zero-friction expansion.

Dependency Direction:
utilities.py -> data_sources.py
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable

# Only import utilities (conforming to strict one-way dependency)
from utilities import get_project_root, validate_coordinates

def _get_default_path(filename: str) -> str:
    """Resolve default sample path inside project root."""
    root = get_project_root()
    return os.path.join(root, "data", "sample", filename)

# -------------------------------------------------------------------------
# Source 1: Pakistan District Geospatial & Reference Source
# -------------------------------------------------------------------------
def load_district_reference_source(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads baseline geographic boundaries, administrative metadata,
    coordinates, and terrain stats for Pakistan districts.
    
    Expected Inputs:
        filepath (str, optional): Path to CSV. Defaults to sample districts.
    Expected Outputs:
        pd.DataFrame with columns:
        ['district_id', 'district', 'province', 'latitude', 'longitude',
         'elevation_m', 'slope_deg', 'urban_density', 'population_est']
    """
    path = filepath or _get_default_path("pakistan_districts.csv")
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Districts file not found at: {path}")
        df = pd.read_csv(path)
        # Basic sanity filtering
        df['valid_coords'] = df.apply(
            lambda r: validate_coordinates(r['latitude'], r['longitude']), axis=1
        )
        return df
    except Exception as e:
        print(f"[DATA_SOURCE WARNING] load_district_reference_source error: {e}")
        return pd.DataFrame(columns=[
            'district_id', 'district', 'province', 'latitude', 'longitude',
            'elevation_m', 'slope_deg', 'urban_density', 'population_est'
        ])

# -------------------------------------------------------------------------
# Source 2: Weather Observations Source (PMD / Synoptic Stations / ERA5)
# -------------------------------------------------------------------------
def load_weather_source(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads atmospheric weather observations (temperature, humidity, wind, pressure).
    
    Expected Inputs:
        filepath (str, optional): Path to CSV. Defaults to sample weather.
    Expected Outputs:
        pd.DataFrame with columns:
        ['district_id', 'timestamp', 'temp_c', 'temp_max_c', 'temp_min_c',
         'humidity_pct', 'wind_speed_kmh', 'wind_dir_deg', 'pressure_hpa',
         'consecutive_hot_days', 'historical_normal_temp_c']
    """
    path = filepath or _get_default_path("weather_sample.csv")
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weather data file not found at: {path}")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[DATA_SOURCE WARNING] load_weather_source error: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------------
# Source 3: Rainfall & Precipitation Source (NASA GPM IMERG / PMD Radar)
# -------------------------------------------------------------------------
def load_rainfall_source(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads cumulative rainfall, rainfall anomaly, and precipitation forecasts.
    
    Expected Inputs:
        filepath (str, optional): Path to CSV. Defaults to sample rainfall.
    Expected Outputs:
        pd.DataFrame with columns:
        ['district_id', 'timestamp', 'rainfall_current_mm', 'rainfall_24h_mm',
         'rainfall_7d_mm', 'rainfall_30d_mm', 'rainfall_anomaly_pct',
         'rainfall_forecast_48h_mm', 'rainfall_intensity_mmh']
    """
    path = filepath or _get_default_path("rainfall_sample.csv")
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rainfall data file not found at: {path}")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[DATA_SOURCE WARNING] load_rainfall_source error: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------------
# Source 4: River & Barrage Hydrological Source (IRSA / FFD / WAPDA)
# -------------------------------------------------------------------------
def load_river_source(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads Indus basin river flow levels, discharge rates, barrage stages,
    and rates of change.
    
    Expected Inputs:
        filepath (str, optional): Path to CSV. Defaults to sample river gauges.
    Expected Outputs:
        pd.DataFrame with columns:
        ['district_id', 'associated_river', 'gauge_station', 'river_level_m',
         'river_discharge_cusecs', 'normal_discharge_cusecs',
         'flood_warning_threshold_cusecs', 'rate_of_change_cusecs_hr', 'flood_stage']
    """
    path = filepath or _get_default_path("river_gauges_sample.csv")
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"River gauge data file not found at: {path}")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[DATA_SOURCE WARNING] load_river_source error: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------------
# Source 5: Satellite Earth Observation Source (Copernicus / NASA MODIS / Sentinel)
# -------------------------------------------------------------------------
def load_satellite_source(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads satellite-derived environmental indices (NDVI vegetation index,
    soil moisture, land surface temperature, and flood-prone zone index).
    
    Expected Inputs:
        filepath (str, optional): Path to CSV. Defaults to sample satellite.
    Expected Outputs:
        pd.DataFrame with columns:
        ['district_id', 'timestamp', 'ndvi', 'soil_moisture_m3m3',
         'land_surface_temp_c', 'flood_prone_index', 'drought_vulnerability_index']
    """
    path = filepath or _get_default_path("satellite_sample.csv")
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Satellite data file not found at: {path}")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[DATA_SOURCE WARNING] load_satellite_source error: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------------
# Source 6: Digital Elevation & Terrain Slope Source (SRTM / ALOS PALSAR)
# -------------------------------------------------------------------------
def load_elevation_source(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Extracts elevation and terrain slope profiles from district reference data.
    
    Expected Inputs:
        filepath (str, optional): Path to CSV.
    Expected Outputs:
        pd.DataFrame with columns:
        ['district_id', 'elevation_m', 'slope_deg', 'terrain_category']
    """
    districts = load_district_reference_source(filepath)
    if districts.empty:
        return pd.DataFrame()
    
    df = districts[['district_id', 'elevation_m', 'slope_deg']].copy()
    
    # Categorize terrain
    def _categorize(row):
        elev = row['elevation_m']
        slope = row['slope_deg']
        if elev > 1200 or slope > 10:
            return "Mountainous / High Relief"
        elif elev > 400:
            return "Plateau / Foothills"
        elif elev <= 50:
            return "Coastal & Deltaic Lowland"
        else:
            return "Alluvial Floodplain"
            
    df['terrain_category'] = df.apply(_categorize, axis=1)
    return df

# -------------------------------------------------------------------------
# Source 7: Historical Climate Hazard Disaster Catalog (NDMA / EM-DAT)
# -------------------------------------------------------------------------
def load_historical_hazards_source(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads historical record of floods, heatwaves, and droughts across Pakistan.
    
    Expected Inputs:
        filepath (str, optional): Path to CSV. Defaults to sample disasters.
    Expected Outputs:
        pd.DataFrame with columns:
        ['event_id', 'hazard_type', 'location_district', 'province',
         'start_year', 'end_year', 'severity_level', 'fatalities_est',
         'people_affected_est', 'primary_driver', 'description']
    """
    path = filepath or _get_default_path("historical_disasters.csv")
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Historical disasters file not found at: {path}")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[DATA_SOURCE WARNING] load_historical_hazards_source error: {e}")
        return pd.DataFrame()

# =========================================================================
# CENTRAL DATA SOURCE REGISTRY
# Extensibility Rule: Adding a new data source requires adding one function
# above and registering it in this dictionary.
# =========================================================================
DATA_SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "district_reference": {
        "function": load_district_reference_source,
        "name": "Pakistan Survey & Administrative Boundaries",
        "description": "Geospatial coordinates, administrative hierarchy, terrain slope and population",
        "update_cadence": "Static Reference"
    },
    "weather": {
        "function": load_weather_source,
        "name": "PMD Synoptic Surface Observations",
        "description": "Temperature, maximum/minimum, relative humidity, atmospheric pressure, wind",
        "update_cadence": "Hourly / 3-Hourly"
    },
    "rainfall": {
        "function": load_rainfall_source,
        "name": "NASA GPM IMERG & PMD Radar Network",
        "description": "Real-time rainfall, 24-hour and 7-day cumulative precipitation, rainfall anomalies",
        "update_cadence": "30-Minute / Daily"
    },
    "river_gauges": {
        "function": load_river_source,
        "name": "IRSA & Federal Flood Commission River Gauges",
        "description": "Indus, Jhelum, Chenab, Kabul, Ravi river discharges, barrage levels, flood stages",
        "update_cadence": "Daily / Hourly During High Flow"
    },
    "satellite": {
        "function": load_satellite_source,
        "name": "Copernicus & NASA Earth Observations",
        "description": "Sentinel NDVI vegetation health, SMAP soil moisture, MODIS land surface temp",
        "update_cadence": "Daily / 5-Day"
    },
    "elevation": {
        "function": load_elevation_source,
        "name": "SRTM Digital Elevation Model",
        "description": "Topographic elevation in meters and terrain gradient slope",
        "update_cadence": "Static Topography"
    },
    "historical_hazards": {
        "function": load_historical_hazards_source,
        "name": "NDMA Disaster Archive & EM-DAT Pakistan",
        "description": "Historical flood, heatwave, and drought impacts (2010 super floods, 2015 heatwave, 2022 deluge)",
        "update_cadence": "Historical Catalog"
    }
}
