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
    """Resolve default sample path across different project layouts (Streamlit Cloud, local, nested)."""
    root = get_project_root()
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidate_dirs = [
        os.path.join(root, "data", "sample"),
        os.path.join(script_dir, "data", "sample"),
        os.path.join(cwd, "data", "sample"),
        os.path.join(cwd, "climate_pakistan_ai", "data", "sample"),
        os.path.join(root, "sample"),
        os.path.join(cwd, "sample"),
    ]

    for d in candidate_dirs:
        target = os.path.join(d, filename)
        if os.path.exists(target):
            return target

    # Default fallback to primary candidate
    return os.path.join(root, "data", "sample", filename)

# Robust fallback district data if CSV path is not resolved on remote hosting
EMBEDDED_DISTRICTS_DATA = [
    {"district_id": "ISB_01", "district": "Islamabad", "province": "Islamabad Capital Territory", "latitude": 33.6844, "longitude": 73.0479, "elevation_m": 540, "slope_deg": 4.2, "urban_density": 0.72, "population_est": 1200000},
    {"district_id": "LHR_01", "district": "Lahore", "province": "Punjab", "latitude": 31.5204, "longitude": 74.3587, "elevation_m": 217, "slope_deg": 0.8, "urban_density": 0.88, "population_est": 13000000},
    {"district_id": "RAW_01", "district": "Rawalpindi", "province": "Punjab", "latitude": 33.5651, "longitude": 73.0169, "elevation_m": 508, "slope_deg": 3.5, "urban_density": 0.79, "population_est": 3500000},
    {"district_id": "MUL_01", "district": "Multan", "province": "Punjab", "latitude": 30.1575, "longitude": 71.5249, "elevation_m": 122, "slope_deg": 0.4, "urban_density": 0.65, "population_est": 2200000},
    {"district_id": "FAIS_01", "district": "Faisalabad", "province": "Punjab", "latitude": 31.4504, "longitude": 73.1350, "elevation_m": 184, "slope_deg": 0.5, "urban_density": 0.78, "population_est": 3800000},
    {"district_id": "RYK_01", "district": "Rahim Yar Khan", "province": "Punjab", "latitude": 28.4195, "longitude": 70.2957, "elevation_m": 83, "slope_deg": 0.3, "urban_density": 0.45, "population_est": 1800000},
    {"district_id": "DGK_01", "district": "Dera Ghazi Khan", "province": "Punjab", "latitude": 30.0561, "longitude": 70.6348, "elevation_m": 124, "slope_deg": 1.8, "urban_density": 0.42, "population_est": 1200000},
    {"district_id": "SIAL_01", "district": "Sialkot", "province": "Punjab", "latitude": 32.4945, "longitude": 74.5229, "elevation_m": 256, "slope_deg": 0.6, "urban_density": 0.68, "population_est": 1700000},
    {"district_id": "KHI_01", "district": "Karachi", "province": "Sindh", "latitude": 24.8607, "longitude": 67.0011, "elevation_m": 8, "slope_deg": 1.1, "urban_density": 0.92, "population_est": 16000000},
    {"district_id": "HYD_01", "district": "Hyderabad", "province": "Sindh", "latitude": 25.3960, "longitude": 68.3578, "elevation_m": 28, "slope_deg": 0.9, "urban_density": 0.75, "population_est": 2500000},
    {"district_id": "SUK_01", "district": "Sukkur", "province": "Sindh", "latitude": 27.7052, "longitude": 68.8574, "elevation_m": 67, "slope_deg": 0.7, "urban_density": 0.58, "population_est": 600000},
    {"district_id": "JAC_01", "district": "Jacobabad", "province": "Sindh", "latitude": 28.2835, "longitude": 68.4388, "elevation_m": 56, "slope_deg": 0.4, "urban_density": 0.42, "population_est": 400000},
    {"district_id": "THAR_01", "district": "Tharparkar (Mithi)", "province": "Sindh", "latitude": 24.7438, "longitude": 69.7997, "elevation_m": 42, "slope_deg": 1.2, "urban_density": 0.18, "population_est": 300000},
    {"district_id": "BAD_01", "district": "Badin", "province": "Sindh", "latitude": 24.6558, "longitude": 68.8384, "elevation_m": 10, "slope_deg": 0.5, "urban_density": 0.32, "population_est": 450000},
    {"district_id": "DADU_01", "district": "Dadu", "province": "Sindh", "latitude": 26.7329, "longitude": 67.7763, "elevation_m": 34, "slope_deg": 0.8, "urban_density": 0.38, "population_est": 480000},
    {"district_id": "LARK_01", "district": "Larkana", "province": "Sindh", "latitude": 27.5589, "longitude": 68.2120, "elevation_m": 53, "slope_deg": 0.5, "urban_density": 0.52, "population_est": 650000},
    {"district_id": "PESH_01", "district": "Peshawar", "province": "Khyber Pakhtunkhwa", "latitude": 34.0151, "longitude": 71.5249, "elevation_m": 359, "slope_deg": 2.1, "urban_density": 0.81, "population_est": 2300000},
    {"district_id": "NOW_01", "district": "Nowshera", "province": "Khyber Pakhtunkhwa", "latitude": 34.0153, "longitude": 71.9747, "elevation_m": 288, "slope_deg": 2.4, "urban_density": 0.54, "population_est": 420000},
    {"district_id": "SWAT_01", "district": "Swat (Mingora)", "province": "Khyber Pakhtunkhwa", "latitude": 34.7717, "longitude": 72.3602, "elevation_m": 980, "slope_deg": 14.5, "urban_density": 0.41, "population_est": 500000},
    {"district_id": "DIK_01", "district": "D.I. Khan", "province": "Khyber Pakhtunkhwa", "latitude": 31.8314, "longitude": 70.9019, "elevation_m": 173, "slope_deg": 1.2, "urban_density": 0.44, "population_est": 550000},
    {"district_id": "ABB_01", "district": "Abbottabad", "province": "Khyber Pakhtunkhwa", "latitude": 34.1688, "longitude": 73.2215, "elevation_m": 1256, "slope_deg": 12.8, "urban_density": 0.49, "population_est": 380000},
    {"district_id": "CHIT_01", "district": "Chitral", "province": "Khyber Pakhtunkhwa", "latitude": 35.8510, "longitude": 71.7864, "elevation_m": 1500, "slope_deg": 22.0, "urban_density": 0.22, "population_est": 180000},
    {"district_id": "QTA_01", "district": "Quetta", "province": "Balochistan", "latitude": 30.1798, "longitude": 66.9750, "elevation_m": 1680, "slope_deg": 8.5, "urban_density": 0.67, "population_est": 1100000},
    {"district_id": "GWD_01", "district": "Gwadar", "province": "Balochistan", "latitude": 25.1216, "longitude": 62.3254, "elevation_m": 8, "slope_deg": 2.0, "urban_density": 0.35, "population_est": 140000},
    {"district_id": "SIBI_01", "district": "Sibi", "province": "Balochistan", "latitude": 29.5448, "longitude": 67.8764, "elevation_m": 130, "slope_deg": 1.5, "urban_density": 0.31, "population_est": 160000},
    {"district_id": "KHUZ_01", "district": "Khuzdar", "province": "Balochistan", "latitude": 27.8117, "longitude": 66.6177, "elevation_m": 1237, "slope_deg": 5.2, "urban_density": 0.28, "population_est": 280000},
    {"district_id": "TUR_01", "district": "Turbat", "province": "Balochistan", "latitude": 26.0022, "longitude": 63.0499, "elevation_m": 129, "slope_deg": 2.3, "urban_density": 0.32, "population_est": 220000},
    {"district_id": "JAFF_01", "district": "Jaffarabad (Dera Allah Yar)", "province": "Balochistan", "latitude": 28.3735, "longitude": 68.3508, "elevation_m": 52, "slope_deg": 0.3, "urban_density": 0.30, "population_est": 250000},
    {"district_id": "CHAG_01", "district": "Chagai (Nok Kundi)", "province": "Balochistan", "latitude": 28.8278, "longitude": 62.7533, "elevation_m": 680, "slope_deg": 1.9, "urban_density": 0.12, "population_est": 90000},
    {"district_id": "GLG_01", "district": "Gilgit", "province": "Gilgit-Baltistan", "latitude": 35.9221, "longitude": 74.3087, "elevation_m": 1500, "slope_deg": 24.5, "urban_density": 0.38, "population_est": 220000},
    {"district_id": "SKD_01", "district": "Skardu", "province": "Gilgit-Baltistan", "latitude": 35.2974, "longitude": 75.6333, "elevation_m": 2228, "slope_deg": 26.0, "urban_density": 0.31, "population_est": 150000},
    {"district_id": "HUN_01", "district": "Hunza (Karimabad)", "province": "Gilgit-Baltistan", "latitude": 36.3257, "longitude": 74.6599, "elevation_m": 2438, "slope_deg": 28.0, "urban_density": 0.20, "population_est": 60000},
    {"district_id": "MUZ_01", "district": "Muzaffarabad", "province": "Azad Jammu & Kashmir", "latitude": 34.3700, "longitude": 73.4711, "elevation_m": 737, "slope_deg": 16.2, "urban_density": 0.55, "population_est": 300000},
    {"district_id": "MIR_01", "district": "Mirpur", "province": "Azad Jammu & Kashmir", "latitude": 33.1484, "longitude": 73.7519, "elevation_m": 340, "slope_deg": 3.8, "urban_density": 0.59, "population_est": 450000},
    {"district_id": "RAWL_01", "district": "Rawalakot", "province": "Azad Jammu & Kashmir", "latitude": 33.8584, "longitude": 73.7654, "elevation_m": 1615, "slope_deg": 18.0, "urban_density": 0.36, "population_est": 120000},
    {"district_id": "NEEL_01", "district": "Neelum (Athmuqam)", "province": "Azad Jammu & Kashmir", "latitude": 34.5878, "longitude": 73.9056, "elevation_m": 1370, "slope_deg": 25.0, "urban_density": 0.18, "population_est": 95000}
]

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
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty and len(df) > 0:
                df['valid_coords'] = df.apply(
                    lambda r: validate_coordinates(r['latitude'], r['longitude']), axis=1
                )
                return df
        # If file not found or empty, load embedded dataset
        df = pd.DataFrame(EMBEDDED_DISTRICTS_DATA)
        df['valid_coords'] = True
        return df
    except Exception as e:
        print(f"[DATA_SOURCE WARNING] load_district_reference_source error: {e}")
        df = pd.DataFrame(EMBEDDED_DISTRICTS_DATA)
        df['valid_coords'] = True
        return df

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
