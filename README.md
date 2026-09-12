# Climate Pakistan AI 🇵🇰

**Pakistan-Focused AI Climate-Risk & Early-Warning Intelligence Platform**

Climate Pakistan AI is an open-source, modular climate-risk intelligence and early-warning platform that leverages environmental and geospatial data to identify localized **flood**, **heatwave**, and **drought** hazards across all regions of Pakistan (Punjab, Sindh, Khyber Pakhtunkhwa, Balochistan, Gilgit-Baltistan, Azad Jammu & Kashmir, and Islamabad Capital Territory).

---

## 🏗️ Architecture & Dependency Direction

The project follows a **strict six-file maximum** with a **strict one-way dependency direction**:

```text
utilities.py
      ↓
data_sources.py
      ↓
data_processing.py
      ↓
climate_models.py
      ↓
rag_system.py
      ↓
app.py
```

### Module Responsibilities

1. **`utilities.py`**: Pakistan spatial bounding boxes, coordinate validation, metric unit formatters, risk score classification thresholds, and mathematical conversions.
2. **`data_sources.py`**: Strict extensibility requirement — **one function per data source** registered in `DATA_SOURCE_REGISTRY`. Loads weather, precipitation, river gauges, satellite observations, elevation, and historical disaster archives.
3. **`data_processing.py`**: Data validation, anomaly detection, missingness handling, spatial-temporal alignment across sources, and derived feature engineering (Heat Index, cumulative rain ratios, river rise rates, SPEI drought proxies).
4. **`climate_models.py`**: Open-source machine learning and calibrated physical risk modeling for `predict_flood_risk`, `predict_heatwave_risk`, and `predict_drought_risk`. Assigns `LOW`, `MODERATE`, `HIGH`, `EXTREME` risk categories with contributing factors.
5. **`rag_system.py`**: Retrieval-Augmented Generation (RAG) knowledge base covering NDMA contingency plans, PMD heatwave SOPs, and agricultural drought guides. Connects to Google Gemini API (`gemini-3.8-flash`) for grounded explanations and multi-tier preventive recommendations (Public, Agriculture, Authorities) with deterministic safe fallbacks.
6. **`app.py`**: Full-featured Streamlit web application providing interactive GIS Folium maps, hazard diagnostic dashboards, active early warnings, and an interactive AI Climate Assistant.

---

## 🚀 Quickstart & Local Execution

### 1. Installation
```bash
git clone https://github.com/your-org/climate-pakistan-ai.git
cd climate-pakistan-ai/climate_pakistan_ai
pip install -r requirements.txt
```

### 2. Configure Gemini API Key (Optional for LLM Layer)
In Streamlit Secrets (`.streamlit/secrets.toml`):
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```
Or set the environment variable:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```
*Note: If the key is not set, the application continues functioning with full numerical prediction capabilities and verified offline rule-based explanations.*

### 3. Run Streamlit Application
```bash
streamlit run app.py
```

### 4. Cloudflare Tunnel (Optional for Remote Access)
```bash
cloudflared tunnel --url http://localhost:8501
```

---

## 📊 Registered Data Sources
| Source | Function | Update Cadence | Description |
|---|---|---|---|
| District Boundaries | `load_district_reference_source` | Static | Geographic coordinates, elevation, population |
| Weather Observations | `load_weather_source` | Hourly / 3-Hourly | PMD Synoptic stations, temperature, humidity, pressure |
| Rainfall & Radar | `load_rainfall_source` | 30-min / Daily | NASA GPM IMERG & PMD Radar precipitation telemetry |
| River Gauges | `load_river_source` | Daily / Flood Alerts | IRSA & FFD Indus, Chenab, Kabul, Jhelum river levels |
| Satellite Earth Obs | `load_satellite_source` | 1-5 Days | Sentinel NDVI, SMAP soil moisture, MODIS thermal |
| Topographic Slope | `load_elevation_source` | Static | SRTM digital elevation and slope profiles |
| Historical Disasters | `load_historical_hazards_source` | Historical | NDMA disaster records (2010 super floods, 2022 deluge) |

---

## 🛡️ Governance & Disclaimer
*Climate Pakistan AI is an AI-assisted diagnostic prototype intended to demonstrate predictive climate analytics and preventive decision-support. It is not an official government early-warning announcement and must not replace directives from the Pakistan Meteorological Department (PMD) or the National Disaster Management Authority (NDMA).*
