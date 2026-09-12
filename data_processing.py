"""
Data Processing Module for Climate Pakistan AI
Implements standard validation, cleaning, unit standardization,
spatial-temporal alignment, and derived feature engineering.

Dependency Direction:
utilities.py -> data_sources.py -> data_processing.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from utilities import validate_coordinates, safe_float
from data_sources import (
    load_district_reference_source,
    load_weather_source,
    load_rainfall_source,
    load_river_source,
    load_satellite_source,
    load_historical_hazards_source,
    DATA_SOURCE_REGISTRY
)

# -------------------------------------------------------------------------
# Step 1: Validation and Cleaning
# -------------------------------------------------------------------------
def validate_environmental_ranges(row: Dict[str, Any]) -> List[str]:
    """
    Validates physical plausibility of environmental measurements.
    Reports warnings for abnormal or out-of-bounds readings without hallucinating values.
    """
    issues = []
    
    # Temperature check (Pakistan record low ~ -24°C in Skardu/Kalam, record high 53.5°C in Mohenjo-daro/Turbat)
    temp = safe_float(row.get('temp_c'))
    if temp is not None and (temp < -30.0 or temp > 55.0):
        issues.append(f"Implausible air temperature: {temp}°C")
        
    temp_max = safe_float(row.get('temp_max_c'))
    if temp_max is not None and temp is not None and temp_max < temp:
        issues.append(f"Inconsistent max temperature ({temp_max}°C) lower than current temp ({temp}°C)")
        
    # Humidity check
    rh = safe_float(row.get('humidity_pct'))
    if rh is not None and (rh < 0.0 or rh > 100.0):
        issues.append(f"Invalid relative humidity: {rh}%")
        
    # Rainfall check
    rain_24h = safe_float(row.get('rainfall_24h_mm'))
    if rain_24h is not None and rain_24h < 0.0:
        issues.append(f"Negative rainfall reading: {rain_24h} mm")
        
    # River discharge
    disc = safe_float(row.get('river_discharge_cusecs'))
    if disc is not None and disc < 0.0:
        issues.append(f"Negative river discharge: {disc} cusecs")
        
    return issues

# -------------------------------------------------------------------------
# Step 2: Derived Feature Engineering
# -------------------------------------------------------------------------
def calculate_heat_index(temp_c: Optional[float], humidity_pct: Optional[float]) -> Optional[float]:
    """
    Calculates the Steadman Heat Index in Celsius using Rothfusz regression.
    Represents perceived temperature taking into account relative humidity.
    """
    if temp_c is None or humidity_pct is None:
        return None
    
    # Heat index only meaningfully applies when temp >= 26.7°C (80°F) and RH >= 40%
    if temp_c < 26.7:
        return round(temp_c, 1)
        
    # Convert C to F
    T = (temp_c * 9.0 / 5.0) + 32.0
    R = humidity_pct
    
    # Simple Rothfusz regression
    HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))
    if HI >= 80.0:
        HI = (-42.379 + 2.04901523 * T + 10.14333127 * R
              - 0.22475541 * T * R - 0.00683783 * (T ** 2)
              - 0.05481717 * (R ** 2) + 0.00122874 * (T ** 2) * R
              + 0.00085282 * T * (R ** 2) - 0.00000199 * (T ** 2) * (R ** 2))
              
    # Convert back to C
    HI_c = (HI - 32.0) * 5.0 / 9.0
    return round(HI_c, 1)

def calculate_drought_proxy(rainfall_anomaly_pct: Optional[float],
                            soil_moisture: Optional[float],
                            temp_anomaly_c: Optional[float]) -> float:
    """
    Calculates a standardized agricultural and meteorological drought index (0 to 100).
    Higher values represent more acute drought conditions.
    """
    score = 20.0 # baseline
    
    # Rainfall deficit contribution (up to 45 pts)
    if rainfall_anomaly_pct is not None:
        if rainfall_anomaly_pct < -80:
            score += 45.0
        elif rainfall_anomaly_pct < -50:
            score += 35.0
        elif rainfall_anomaly_pct < -20:
            score += 20.0
        elif rainfall_anomaly_pct > 20:
            score -= 15.0
            
    # Soil moisture deficit (0.05 is dry desert, 0.45 is saturated) (up to 30 pts)
    if soil_moisture is not None:
        if soil_moisture < 0.10:
            score += 30.0
        elif soil_moisture < 0.20:
            score += 20.0
        elif soil_moisture < 0.30:
            score += 10.0
        else:
            score -= 10.0
            
    # Thermal stress exacerbating evapotranspiration (up to 15 pts)
    if temp_anomaly_c is not None:
        if temp_anomaly_c > 3.0:
            score += 15.0
        elif temp_anomaly_c > 1.5:
            score += 10.0
        elif temp_anomaly_c > 0.0:
            score += 5.0
            
    return max(0.0, min(100.0, round(score, 1)))

# -------------------------------------------------------------------------
# Step 3: End-to-End Alignment and Pipeline Processing
# -------------------------------------------------------------------------
def process_district_features(district_id: str) -> Dict[str, Any]:
    """
    Loads, cleans, aligns, and derives features for a single district.
    Returns structured data dictionary and data quality report.
    """
    # 1. Load data sources
    districts_df = load_district_reference_source()
    weather_df = load_weather_source()
    rainfall_df = load_rainfall_source()
    river_df = load_river_source()
    satellite_df = load_satellite_source()
    historical_df = load_historical_hazards_source()
    
    # 2. Extract district row
    d_row = districts_df[districts_df['district_id'] == district_id]
    if d_row.empty:
        # Try match by district name
        d_row = districts_df[districts_df['district'].str.lower() == district_id.lower()]
        
    if d_row.empty:
        return {
            "status": "error",
            "message": f"District identifier '{district_id}' not recognized in Pakistan reference dataset.",
            "data_quality": {"completeness_pct": 0, "issues": ["Unknown district"]}
        }
        
    d_info = d_row.iloc[0].to_dict()
    did = d_info['district_id']
    
    # 3. Match across source datasets
    w_row = weather_df[weather_df['district_id'] == did]
    w_info = w_row.iloc[0].to_dict() if not w_row.empty else {}
    
    r_row = rainfall_df[rainfall_df['district_id'] == did]
    r_info = r_row.iloc[0].to_dict() if not r_row.empty else {}
    
    riv_row = river_df[river_df['district_id'] == did]
    riv_info = riv_row.iloc[0].to_dict() if not riv_row.empty else {}
    
    sat_row = satellite_df[satellite_df['district_id'] == did]
    sat_info = sat_row.iloc[0].to_dict() if not sat_row.empty else {}
    
    # Filter historical events for this district
    district_name = d_info.get('district', '')
    hist_events = historical_df[
        historical_df['location_district'].str.contains(district_name.split()[0], case=False, na=False)
    ].to_dict('records')
    
    # 4. Merge all attributes into aligned record
    merged: Dict[str, Any] = {
        **d_info,
        **w_info,
        **r_info,
        **riv_info,
        **sat_info,
        "historical_events": hist_events,
        "historical_event_count": len(hist_events)
    }
    
    # 5. Derived calculations
    # Temperature anomaly
    temp = safe_float(merged.get('temp_c'))
    normal_temp = safe_float(merged.get('historical_normal_temp_c'))
    temp_anomaly = None
    if temp is not None and normal_temp is not None:
        temp_anomaly = round(temp - normal_temp, 2)
    merged['temp_anomaly_c'] = temp_anomaly
    
    # Heat index
    humidity = safe_float(merged.get('humidity_pct'))
    merged['heat_index_c'] = calculate_heat_index(temp, humidity)
    
    # Cumulative rainfall ratio (24h to 7d)
    r24 = safe_float(merged.get('rainfall_24h_mm'), 0.0)
    r7d = safe_float(merged.get('rainfall_7d_mm'), 0.0)
    merged['rain_intensity_ratio'] = round(r24 / max(1.0, r7d), 2)
    
    # River discharge saturation ratio
    disc = safe_float(merged.get('river_discharge_cusecs'), 0.0)
    thresh = safe_float(merged.get('flood_warning_threshold_cusecs'), 100000.0)
    merged['river_capacity_pct'] = round((disc / max(1.0, thresh)) * 100.0, 1)
    
    # Drought index proxy
    rain_anom = safe_float(merged.get('rainfall_anomaly_pct'))
    sm = safe_float(merged.get('soil_moisture_m3m3'))
    merged['drought_proxy_index'] = calculate_drought_proxy(rain_anom, sm, temp_anomaly)
    
    # 6. Data quality report
    issues = validate_environmental_ranges(merged)
    expected_fields = [
        'temp_c', 'humidity_pct', 'rainfall_24h_mm', 'river_discharge_cusecs',
        'elevation_m', 'ndvi', 'soil_moisture_m3m3'
    ]
    present_fields = [f for f in expected_fields if merged.get(f) is not None]
    completeness = round((len(present_fields) / len(expected_fields)) * 100.0, 1)
    
    data_quality = {
        "completeness_pct": completeness,
        "missing_fields": [f for f in expected_fields if f not in present_fields],
        "issues": issues,
        "is_sufficient": completeness >= 60.0
    }
    
    return {
        "status": "success",
        "district_id": did,
        "district": d_info.get('district'),
        "province": d_info.get('province'),
        "features": merged,
        "data_quality": data_quality
    }

def get_all_processed_districts() -> List[Dict[str, Any]]:
    """Processes all districts and returns clean list of features."""
    districts_df = load_district_reference_source()
    results = []
    for _, row in districts_df.iterrows():
        res = process_district_features(row['district_id'])
        if res.get('status') == 'success':
            results.append(res)
    return results
