# Automated Site Analysis API – Deployment & Architecture Documentation  
## ALKF – Automated Geospatial Intelligence System

---

# Executive Summary

The **Automated Site Analysis API (ALKF)** is a production-deployed geospatial intelligence microservice designed to automate professional urban feasibility analysis.

The system provides:

- Walking Accessibility Analysis  
- Driving Distance & Isochrone Mapping  
- Public Transport Network Assessment  
- Land Use & Zoning Context Evaluation  
- 360° View Sector Classification  
- Road Traffic Environmental Noise Modeling  
- Automated PDF Report Generation  

The API is deployed on **Render Cloud (Free Tier Optimized)** using a modular **FastAPI architecture** with dataset preloading, caching, and structured logging.

---

# Project Objectives

## Primary Objectives

1. Automate complex GIS analysis workflows  
2. Convert standalone research scripts into a scalable API service  
3. Optimize heavy geospatial computations for cloud deployment  
4. Deliver professional analytical maps in PNG and PDF formats  
5. Enable on-demand lot-based analysis through HTTP endpoints  

## Secondary Objectives

- Modular code architecture  
- Reduced dataset size for cloud constraints  
- Improved runtime efficiency  
- Production-level logging  
- Report bundling functionality  

---

# System Overview

The system follows a layered architecture:

```
Client → API → Analysis Engine → Data Layer → Rendering Engine → Output
```

## Core Components

- FastAPI Backend  
- Uvicorn ASGI Server  
- GeoPandas  
- OSMnx  
- NetworkX  
- Matplotlib  
- Static GIS Datasets  

---

# High-Level Architecture Diagram

```
User Request (LOT ID)
        │
        ▼
FastAPI Endpoint
        │
        ▼
Input Resolver
        │
        ▼
Coordinate Transformation
        │
        ▼
Spatial Data Extraction
        │
        ▼
Analysis Modules
        │
        ▼
Visualization Engine
        │
        ▼
PNG / PDF Output
```

---

# Deployment Architecture

```
Client Browser / Colab
        │
        ▼
Render Cloud Service
        │
        ▼
Uvicorn ASGI Server
        │
        ▼
FastAPI Application
        │
        ▼
Modular GIS Analysis
```

## Hosting Platform

- Render (Free Plan)  
- Python Runtime  
- Auto-deploy via GitHub  

---

# Repository Structure

```
Automated-Site-Analysis-API/
│
├── app.py
├── render.yaml
├── requirements.txt
├── runtime.txt
│
├── data/
│   ├── BUILDINGS_FINAL.gpkg
│   └── ZONE_REDUCED.gpkg
│
├── modules/
│   ├── walking.py
│   ├── driving.py
│   ├── transport.py
│   ├── context.py
│   ├── view.py
│   └── noise.py
```

---

# Data Layer Design

## Static Datasets

### 1. Building Height Dataset
- Reduced from **342,000+ rows → 42,073 rows**
- Retained columns:
  - `HEIGHT_M`
  - `geometry`

### 2. Zoning Dataset
- Reduced attributes
- Converted to EPSG:3857

---

# Optimization Strategy

- Preloading datasets at startup  
- Removing unnecessary attributes  
- CRS standardization  
- DPI reduction (400 → 200)  
- In-memory caching  

---

# Coordinate Transformation Pipeline

The system converts coordinates:

1. EPSG:2326 → EPSG:4326  
2. EPSG:4326 → EPSG:3857  

Reason:
- OSM uses WGS84  
- Spatial operations optimized in projected CRS  

---

# Module Documentation

## Walking Analysis

**Purpose:** Assess pedestrian accessibility.

**Methods:**
- Extract walkable OSM graph  
- Compute shortest paths using NetworkX  
- Generate service buffers  
- Amenity density overlay  

---

## Driving Distance Analysis

**Purpose:** Evaluate vehicular connectivity.

**Methods:**
- Drive network extraction  
- Travel-time weighting  
- Isochrone buffer generation  
- Connectivity scoring  

---

## Transportation Network Module

**Purpose:** Assess public transport accessibility.

Includes:
- Bus stop extraction  
- Transit density clustering  
- Proximity mapping  

---

## Context & Zoning Module

**Purpose:** Analyze surrounding land use.

Features:
- Zoning overlay  
- Amenity mapping  
- Green/water coverage  
- Density analysis  

---

## 360° View Analysis Engine

### Methodology

1. Divide 360° into equal sectors  
2. Calculate:
   - Green ratio  
   - Water ratio  
   - Building density  
   - Average height  
3. Normalize features  
4. Apply weighted scoring  
5. Merge dominant sectors  

### Scoring Model

```
Green Score  = green_ratio
Water Score  = water_ratio
City Score   = height_norm × density_norm
Open Score   = (1 - density_norm) × (1 - height_norm)
```

---

## Noise Modelling Engine

### Base Formula

```
L = L₀ − 20 log₁₀(r)
```

### Extended Model Includes

- Heavy vehicle adjustment  
- Ground absorption  
- Barrier attenuation  
- Reflection factor  
- Grid-based propagation  

---

# Visualization Engine

Uses:

- Matplotlib  
- GeoPandas layered plotting  
- Styled legends  
- Sector arc rendering  
- Contour mapping  

Output:

- PNG Images  
- Combined PDF Report  

---

# API Endpoint Design

| Endpoint     | Description |
|-------------|------------|
| /walking    | Pedestrian analysis |
| /driving    | Vehicular analysis |
| /transport  | Transit mapping |
| /context    | Zoning context |
| /view       | 360° sector view |
| /noise      | Environmental noise |
| /report     | Combined PDF |

---

# Request Format

```json
{
  "lot_id": "IL 1657"
}
```

---

# Deployment Configuration (render.yaml)

```yaml
services:
  - type: web
    name: automated-site-analysis-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port 10000
    plan: free
    autoDeploy: true
```

---

# Requirements

```
fastapi
uvicorn
geopandas
osmnx
shapely
pyproj
networkx
numpy
pandas
matplotlib
reportlab
```

---

# Web API Testing Guide

## Health Check

Open:

```
https://automated-site-analysis-api.onrender.com
```

Expected:

```json
{
  "status": "Automated Site Analysis API Running"
}
```

---

## Swagger UI

```
https://automated-site-analysis-api.onrender.com/docs
```

Use **Try it out** → Execute.

---

## Google Colab Test Code

```python
import requests
from IPython.display import Image, display

BASE_URL = "https://automated-site-analysis-api.onrender.com"

lot_data = {"lot_id": "IL 1657"}

response = requests.post(BASE_URL + "/view", json=lot_data)

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
    display(Image("output.png"))
else:
    print("Error:", response.status_code)
    print(response.text)
```

---

# Performance Testing

```python
import time
import requests

start = time.time()
r = requests.post(
    "https://automated-site-analysis-api.onrender.com/view",
    json={"lot_id": "IL 1657"}
)
end = time.time()

print("Status:", r.status_code)
print("Time Taken:", round(end-start, 2), "seconds")
```

### Expected Timing (Free Plan)

| Scenario    | Time |
|------------|------|
| Cold Start | 10–15 sec |
| Normal     | 5–10 sec |
| Cached     | < 2 sec |

---

# Production Validation Checklist

- Server starts without error  
- Static datasets load correctly  
- CRS transformation works  
- All endpoints return 200  
- PNG renders correctly  
- PDF downloads correctly  
- No memory crash  
- No timeout  

---

# Future Roadmap

- Authentication layer  
- Rate limiting  
- Batch lot processing  
- SaaS dashboard  
- Background job queue  
- Persistent storage  
- Paid tier scaling  

---

# Conclusion

The **Automated Site Analysis API** represents a transition from standalone GIS research scripts to a scalable, cloud-deployed geospatial intelligence microservice.

It demonstrates:

- Advanced spatial computation  
- Environmental modelling  
- Network-based routing  
- Sector classification algorithms  
- Cloud optimization under resource constraints  
- Production API engineering practices  

---

**Status: Production Deployed (Render Free Tier Optimized)**  
**Version: 1.0**
