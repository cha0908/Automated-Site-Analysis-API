<p align="center">
  <img src="images/alkf-logo.png" width="120">
</p>

<h2 align="center">ALKF+ — Automated Spatial Intelligence System API</h2>

<p align="center">
  Modular geospatial intelligence platform for automated urban feasibility assessment.<br>
  Converts raw geographic lot data, addresses, and coordinates into structured analytical maps and PDF reports via a cloud-deployed API.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-production-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Render-deployed-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/status-production-brightgreen?style=flat-square" />
</p>

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Objectives](#project-objectives)
3. [System Architecture](#system-architecture)
4. [Quick Start](#quick-start)
5. [Search Endpoint](#search-endpoint)
6. [API Reference](#api-reference)
7. [Module Documentation](#module-documentation)
8. [Resolver & Data Pipeline](#resolver--data-pipeline)
9. [Coordinate Transformation Pipeline](#coordinate-transformation-pipeline)
10. [Data Layer Design](#data-layer-design)
11. [Visualisation Engine](#visualisation-engine)
12. [PDF Report Generator](#pdf-report-generator)
13. [Deployment](#deployment)
14. [Performance & Optimisation](#performance--optimisation)
15. [Repository Structure](#repository-structure)
16. [Web API Testing Guide](#web-api-testing-guide)
17. [Production Validation Checklist](#production-validation-checklist)
18. [Troubleshooting](#troubleshooting)
19. [Future Roadmap](#future-roadmap)
20. [Implementation Stages](#implementation-stages)
21. [Changelog](#changelog)

---

## Executive Summary

The **Automated Site Analysis API (ALKF+)** is a production-deployed geospatial intelligence microservice designed to automate professional urban feasibility analysis for the Hong Kong context.

The system accepts a Hong Kong lot ID, street address, building name, or coordinate pair and returns a comprehensive set of spatial analysis outputs — professional PNG maps and a multi-page PDF report — generated entirely on-demand via HTTP endpoints.

**7 Analysis Modules:**

| Module | Output |
|--------|--------|
| `/walking` | Pedestrian isochrone map (5 min & 15 min) |
| `/driving` | Drive-time rings + MTR ingress/egress routes |
| `/transport` | Transit node accessibility and scoring map |
| `/context` | OZP zoning + land-use + amenity context map |
| `/view` | 360° polar view sector classification diagram |
| `/noise` | Road traffic noise propagation heatmap |
| `/report` | 8-page combined PDF report (landscape A4) |

**Platform characteristics:**

- Multi-type input resolver: LOT · STT · GLA · LPP · UN · BUILDINGCSUID · LOTCSUID · PRN · ADDRESS
- Unified `/search` endpoint for lot ID, address, and building name lookup
- LandsD iC1000 official lot boundary API integration
- Government GIS coordinate resolution (EPSG:2326 → EPSG:4326 → EPSG:3857)
- In-memory result caching for sub-second repeat responses
- Render Cloud (Free Tier) deployment with auto-deploy via GitHub
- ReportLab 8-page PDF report generation

---

## Project Objectives

### Primary Objectives

1. Automate complex GIS analysis workflows that previously required manual desktop GIS software
2. Convert standalone geospatial research scripts into a scalable, cloud-deployed API service
3. Optimize heavy geospatial computations for constrained free-tier cloud deployment
4. Deliver professional-grade analytical maps in PNG and PDF formats on demand
5. Enable on-demand analysis for any Hong Kong lot ID, address, or coordinate via HTTP

### Secondary Objectives

- Modular code architecture — each analysis module operates independently
- Reduced dataset size optimised for cloud memory constraints
- Improved runtime efficiency through startup preloading and in-memory caching
- Production-level structured logging with per-request timing
- Unified search and resolution for multiple HK government identifier types
- PDF report bundling of all 7 analyses into a single downloadable document

---

## System Architecture

### End-to-End Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Request                         │
│     LOT / STT / GLA / LPP / BUILDINGCSUID / ADDRESS / etc. │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI  ·  app.py                         │
│   Pydantic LocationRequest validation                       │
│   MD5 cache check: key = MD5(data_type_value_analysis)      │
│   Structured logging with request timing                    │
└──────────────┬─────────────────────────────┬────────────────┘
               │                             │
               ▼                             ▼
┌──────────────────────────┐   ┌─────────────────────────────┐
│      resolver.py         │   │   Static Datasets           │
│                          │   │   (preloaded at startup)    │
│  resolve_location()      │   │                             │
│  ├─ ADDRESS              │   │  ZONE_REDUCED.gpkg          │
│  │  → coords from        │   │  (OZP zones, EPSG:3857)     │
│  │    /search directly   │   │                             │
│  └─ LOT/STT/GLA/etc.     │   │  BUILDINGS_FINAL.gpkg       │
│     → GeoData SearchNum  │   │  (HEIGHT_M > 5m, EPSG:3857) │
│     → EPSG:2326→4326     │   └─────────────────────────────┘
│                          │
│  get_lot_boundary()      │
│  ├─ ADDRESS → None       │
│  └─ Others               │
│     → iC1000 API (GML)   │
│     → point-in-polygon   │
│     → EPSG:3857          │
│     → cached result      │
└──────────────┬───────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Analysis Modules                          │
│                                                             │
│   walking.py    Pedestrian isochrone + amenity overlay      │
│   driving.py    Drive-time rings + MTR ingress/egress       │
│   transport.py  Transit density + route diversity scoring   │
│   context.py    OZP zoning + OSM features + clustering      │
│   view.py       360° sector scoring + polar rendering       │
│   noise.py      Grid-based noise propagation model          │
│                                                             │
│   OSMnx graph extraction  ·  NetworkX Dijkstra routing      │
│   GeoPandas spatial ops   ·  Shapely geometry               │
│   Matplotlib + contextily rendering                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Output Layer                            │
│                                                             │
│   PNG  →  StreamingResponse (image/png)                     │
│   PDF  →  ReportLab SimpleDocTemplate                       │
│           StreamingResponse (application/pdf)               │
│           8-page landscape A4 · 0.3-inch margins           │
└─────────────────────────────────────────────────────────────┘
```

### Request Lifecycle with Cache

```
Client ── POST /walking {data_type, value} ──► FastAPI
                                                   │
                            ┌──────────────────────▼──────────────────────┐
                            │   key = MD5(data_type + value + endpoint)    │
                            └──────────────────────┬──────────────────────┘
                                                   │
                              ┌────────────────────▼────────────────────┐
                              │           CACHE_STORE lookup            │
                              └────────────┬────────────────┬───────────┘
                                    HIT    │                │  MISS
                                           ▼                ▼
                                  Return cached      resolve_location()
                                  PNG buffer         get_lot_boundary()
                                  (< 1 sec)          generate_walking()
                                                           │
                                                           ▼
                                                  Store in CACHE_STORE
                                                  Return PNG buffer
                                                  (5–10 sec first time)
```

### Key Design Decisions

| Decision | Reason |
|----------|--------|
| Datasets preloaded at startup | Eliminates per-request file I/O |
| All datasets in EPSG:3857 | No per-request reprojection cost |
| MD5 in-memory cache | Sub-second repeat responses |
| ADDRESS type bypasses GIS API | Coordinates pre-resolved at `/search` |
| `HEIGHT_M > 5 m` filter at load | Reduces building dataset 342k → 42k rows |
| `OMP_NUM_THREADS=1` | Prevents thread explosion on free-tier server |
| `matplotlib.use("Agg")` | Headless non-interactive rendering backend |
| Temp file for GML parsing | `gpd.read_file()` requires a file path, not a buffer |

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- Git

### Installation

```bash
git clone https://github.com/your-org/automated-site-analysis-api.git
cd automated-site-analysis-api
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 10000 --reload
```

API available at: `http://localhost:10000`

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `10000` |
| `LOG_LEVEL` | Logging verbosity | `info` |
| `CACHE_ENABLED` | In-memory caching | `true` |
| `DPI` | Output image resolution | `200` |

### Your First Request

```bash
# LOT type — server resolves coordinates
curl -X POST http://localhost:10000/walking \
  -H "Content-Type: application/json" \
  -d '{"data_type": "LOT", "value": "IL 1657"}' \
  --output walking.png

# ADDRESS type — pass pre-resolved coordinates from /search
curl -X POST http://localhost:10000/context \
  -H "Content-Type: application/json" \
  -d '{"data_type": "ADDRESS", "value": "129 Repulse Bay Road", "lon": 114.1955, "lat": 22.2407}' \
  --output context.png
```

---

## Search Endpoint

### `GET /search`

Unified lot ID, address, and building name search. Automatically routes to the correct data source and pre-resolves coordinates for ADDRESS-type results.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | required | Search query |
| `limit` | int | `100` | Max results per source |

**Example requests:**

```
GET /search?q=IL+1657          → lot search only
GET /search?q=STTL+467         → lot search only
GET /search?q=NKIL             → lot prefix — returns up to 100 results
GET /search?q=The+Lily         → address / building name search only
GET /search?q=Repulse+Bay      → area / address search
```

**Search routing logic:** Queries beginning with a recognised lot prefix (`IL`, `NKIL`, `KIL`, `STTL`, `STL`, `TML`, `TPTL`, `DD`, `RBL`, `KCTL`, `ML`, `GLA`, `LPP`) are routed to the lot search API only. All other queries are routed to the address and building name search API only. This prevents cross-contamination of noisy results between the two data sources.

**Response schema:**

```json
{
  "count": 2,
  "results": [
    {
      "lot_id":    "IL 1657",
      "name":      "IL 1657",
      "address":   "IL 1657, Mid-Levels, Hong Kong",
      "district":  "Central and Western",
      "ref_id":    "REF123",
      "data_type": "LOT",
      "source":    "lot_search",
      "lon":       null,
      "lat":       null
    },
    {
      "lot_id":    "129 REPULSE BAY ROAD",
      "name":      "The Lily",
      "address":   "129 REPULSE BAY ROAD",
      "district":  "Southern",
      "ref_id":    "",
      "data_type": "ADDRESS",
      "source":    "address_search",
      "lon":       114.1955,
      "lat":       22.2407
    }
  ]
}
```

**Important field notes:**
- `LOT` results: `lon`/`lat` = `null` — the resolver handles coordinate lookup per analysis request
- `ADDRESS` results: `lon`/`lat` pre-resolved — pass these directly to all analysis endpoints
- `data_type` field tells you exactly which type to send in the analysis request body
- `lot_id` / `address` is the value to send as the `value` field in analysis requests

---

## API Reference

**Base URL:** `https://automated-site-analysis-api.onrender.com`

**Authentication:** None required.

### Request Model

```json
{
  "data_type": "LOT",
  "value":     "IL 1657",
  "lon":       null,
  "lat":       null
}
```

**Accepted `data_type` values:**

| `data_type` | Example `value` | Boundary API |
|-------------|----------------|-------------|
| `LOT` | `IL 1657`, `NKIL 6304` | iC1000 `/lot` |
| `STT` | `STT 1234` | iC1000 `/stt` |
| `GLA` | `GLA 567` | iC1000 `/gla` |
| `LPP` | `LPP 890` | iC1000 `/lot` |
| `UN` | `UN 123` | iC1000 `/lot` |
| `BUILDINGCSUID` | `CSUID_xxx` | iC1000 `/lot` |
| `LOTCSUID` | `CSUID_yyy` | iC1000 `/lot` |
| `PRN` | `PRN 456` | iC1000 `/lot` |
| `ADDRESS` | `129 Repulse Bay Road` | None — requires `lon`/`lat` |

### Endpoints

#### `POST /walking`
Returns a pedestrian accessibility isochrone map.
**Response:** `200 OK` — `image/png`

#### `POST /driving`
Returns a drive-time ring map with ingress (red) and egress (green) MTR route lines.
**Response:** `200 OK` — `image/png`

#### `POST /transport`
Returns a public transit node accessibility map with bus and MTR markers scored by route count.
**Response:** `200 OK` — `image/png`

#### `POST /context`
Returns a land-use context map with OZP zoning overlay and amenity distribution.
**Response:** `200 OK` — `image/png`

#### `POST /view`
Returns a 360° polar view sector classification diagram (GREEN / WATER / CITY / OPEN).
**Response:** `200 OK` — `image/png`

#### `POST /noise`
Returns a grid-based road traffic noise propagation heatmap (dB scale).
**Response:** `200 OK` — `image/png`

#### `POST /report`
Generates an 8-page combined PDF report containing all analyses.
**Response:** `200 OK` — `application/pdf`
**Content-Disposition:** `attachment; filename=site_analysis_report.pdf`

#### `GET /`
Health check.
**Response:** `{ "status": "Automated Site Analysis API Running - Multi Identifier Enabled" }`

---

### Full curl Workflow

```bash
BASE="https://automated-site-analysis-api.onrender.com"

# Search first
curl "$BASE/search?q=IL+1657"

# LOT type analysis
LOT='{"data_type": "LOT", "value": "IL 1657"}'
curl -s -X POST $BASE/walking   -H "Content-Type: application/json" -d "$LOT" -o walking.png
curl -s -X POST $BASE/driving   -H "Content-Type: application/json" -d "$LOT" -o driving.png
curl -s -X POST $BASE/transport -H "Content-Type: application/json" -d "$LOT" -o transport.png
curl -s -X POST $BASE/context   -H "Content-Type: application/json" -d "$LOT" -o context.png
curl -s -X POST $BASE/view      -H "Content-Type: application/json" -d "$LOT" -o view.png
curl -s -X POST $BASE/noise     -H "Content-Type: application/json" -d "$LOT" -o noise.png
curl -s -X POST $BASE/report    -H "Content-Type: application/json" -d "$LOT" -o report.pdf

# ADDRESS type — lon/lat required from /search
ADDR='{"data_type": "ADDRESS", "value": "129 Repulse Bay Road", "lon": 114.1955, "lat": 22.2407}'
curl -s -X POST $BASE/walking -H "Content-Type: application/json" -d "$ADDR" -o walking_addr.png
```

### Python Client

```python
import requests

BASE_URL = "https://automated-site-analysis-api.onrender.com"

def search(query: str):
    resp = requests.get(f"{BASE_URL}/search", params={"q": query})
    resp.raise_for_status()
    return resp.json()["results"]

def fetch_analysis(endpoint: str, data_type: str, value: str,
                   lon: float = None, lat: float = None, output_path: str = None):
    payload = {"data_type": data_type, "value": value}
    if lon is not None: payload["lon"] = lon
    if lat is not None: payload["lat"] = lat
    resp = requests.post(f"{BASE_URL}/{endpoint}", json=payload, timeout=60)
    resp.raise_for_status()
    if output_path:
        with open(output_path, "wb") as f:
            f.write(resp.content)
    return resp.content

# LOT type
fetch_analysis("walking", "LOT", "IL 1657", output_path="walking.png")
fetch_analysis("report",  "LOT", "IL 1657", output_path="report.pdf")

# ADDRESS type — get coords from /search first
results = search("The Lily")
r = results[0]
fetch_analysis("context", "ADDRESS", r["address"], lon=r["lon"], lat=r["lat"], output_path="context.png")
```

---

## Module Documentation

### `walking.py` — Pedestrian Accessibility Analysis

Evaluates walkability and proximity to amenities from the site centroid using a real pedestrian road network.

**Method:**
1. Extract walkable OSM graph via osmnx within a configurable radius
2. Snap site centroid to nearest walk network node
3. Run Dijkstra shortest-path routing to all reachable amenity nodes
4. Classify amenities by type — food, health, education, retail, recreation
5. Generate 5-minute and 15-minute isochrone service area buffers
6. Render choropleth map with amenity cluster overlays and legend

| Parameter | Value | Description |
|-----------|-------|-------------|
| Walk speed | 1.39 m/s | Standard pedestrian speed (5 km/h) |
| Graph radius | 1 200 m | osmnx extraction radius |
| Isochrone bands | 5 min · 15 min | Service area thresholds |

---

### `driving.py` — Drive-Time & MTR Route Analysis

Assesses vehicular road network reach and computes real shortest-path routes to the nearest 3 MTR stations in both ingress and egress directions.

**Method:**
1. Extract drive-mode graph from osmnx within `graph_dist` metres
2. Assign `travel_time` edge weights: `length (m) / (35 000 m / 60 min)`
3. Snap site centroid and each station to nearest graph node
4. Dijkstra shortest path: site → station (egress, green) and station → site (ingress, red)
5. Render concentric drive-time rings (dashed gold) with distance/time labels
6. Place single directional arrow at 60% along the longest route segment
7. Place TO/FROM station labels at map edge with 22° rotation collision avoidance

**Drive-time ring configurations:**

| `max_drive_minutes` | Ring Radii (m) | Map Extent | Graph Fetch Dist |
|---------------------|---------------|------------|-----------------|
| 5 | 83 / 250 / 400 | 600 m | 800 m |
| 10 | 250 / 500 / 750 | 875 m | 1 500 m |
| 15 | 375 / 750 / 1 125 | 1 400 m | 2 500 m |

**Site polygon detection — 5-step fallback chain:**

```
Step 1 — Official lot boundary via get_lot_boundary()
Step 2 — OZP zone_data polygon containing the site point
Step 3 — OSM building footprint (progressive radius: 80 → 150 → 250 m)
Step 4 — OSM landuse / amenity polygon within 100 m
Step 5 — 80 m circular buffer (guaranteed fallback, always succeeds)
```

---

### `transport.py` — Public Transit Accessibility

Scores the site's proximity to and density of public transit infrastructure using a composite normalised metric.

**Method:**
1. Query OSM for all bus stops and MTR/rail stations within 800 m
2. Compute transit node density per unit area
3. Score route diversity — number of distinct routes within walking distance
4. Render node markers on map, scaled by route count

```
Transit Score = 0.5 × node_density_norm + 0.5 × route_diversity_norm
```

---

### `context.py` — Land-Use, Zoning & Amenity Context

Provides a spatial overview of the surrounding land-use environment using official OZP zoning data and OpenStreetMap features.

**Layers rendered:**
- OZP zoning overlay from `ZONE_REDUCED.gpkg` — colour-coded by zone type
- Amenity point distribution filtered by site type via `context_rules()`
- Green space polygons — parks and recreational areas
- Building footprint density overlay
- Road network classification

**Zoning-driven context filter:**

| `SITE_TYPE` | OSM Tags Fetched |
|-------------|-----------------|
| `RESIDENTIAL` | school · college · university · park · neighbourhood |
| `COMMERCIAL` | bank · restaurant · market · railway station |
| `INSTITUTIONAL` | school · college · hospital · park |
| `HOTEL` / `MIXED` | all amenity + leisure types |

**Bus stop clustering:** All raw bus stops within 900 m are reduced to 6 representative cluster centres via KMeans (k=6, random_state=0) to reduce visual clutter on the output map.

---

### `view.py` — 360° View Sector Classification Engine

Classifies the visual environment from the site across all compass directions using building height data and OSM green/water features.

**Methodology:**
1. Divide 360° into 36 equal sectors (10° each)
2. For each sector extract: green ratio, water ratio, building count (density), average `HEIGHT_M`
3. Normalise all extracted features to [0, 1] range
4. Apply composite scoring model — each sector receives four scores:

```
Green Score  = green_ratio
Water Score  = water_ratio
City Score   = height_norm × density_norm
Open Score   = (1 − density_norm) × (1 − height_norm)
```

5. Label each sector by its highest-scoring view type
6. Merge adjacent sectors that share the same dominant label

| Label | Winning Condition |
|-------|------------------|
| `GREEN VIEW` | Dominant green_ratio |
| `WATER VIEW` | Dominant water_ratio |
| `CITY VIEW` | High building density + high height |
| `OPEN VIEW` | Low density + low height |

**Input data:** `BUILDINGS_FINAL.gpkg` — preloaded at startup, filtered to `HEIGHT_M > 5 m`.

---

### `noise.py` — Road Traffic Noise Propagation Model

Simulates ambient road noise levels across the surrounding area using a physics-based point-source propagation grid.

**Base propagation formula:**

```
L(r) = L₀ − 20·log₁₀(r)
```

Where `L₀` = source emission level (dB) by road class and `r` = distance from source centre (m).

**Source emission levels:**

| Road Class | L₀ (dB) |
|------------|---------|
| Motorway | 78 |
| Primary | 72 |
| Secondary | 68 |
| Tertiary | 64 |
| Residential | 58 |

**Empirical correction factors applied:**

| Correction | Value |
|------------|-------|
| Heavy vehicle presence (motorway/primary) | +3 dB |
| Building mass barrier attenuation | Variable — reduces propagation across façades |
| Ground absorption — soft surfaces | Up to −3 dB |
| Hard surface reflection — streets/plazas | +1 to +2 dB |

---

## Resolver & Data Pipeline

### `resolver.py` — Multi-Type Input Resolver

**`resolve_location()` — coordinate resolution:**

```
ADDRESS type:
    lon/lat taken directly from /search result and returned immediately.
    No Government GIS API call is made.
    Raises ValueError if lon/lat fields are missing.

All other types (LOT / STT / GLA / LPP / etc.):
    GET https://mapapi.geodata.gov.hk/gs/api/v1.0.0/lus/{data_type}/SearchNumber?text={value}
    → select candidate with highest score from candidates[]
    → transform x, y from EPSG:2326 → EPSG:4326 via pyproj
    → return (lon, lat)
```

**`get_lot_boundary()` — official polygon retrieval:**

```
ADDRESS type → return None immediately (no boundary available)

All other types:
    1. Check _LOT_BOUNDARY_CACHE[(round(lon,5), round(lat,5), data_type)]
       → return cached result if available
    2. Transform site coordinates to EPSG:2326 (HK Grid)
    3. Build ±300 m bounding box in EPSG:2326
    4. GET /iC1000/{lot_type}?bbox={minx},{miny},{maxx},{maxy},EPSG:2326
    5. Write GML response to NamedTemporaryFile (suffix .gml)
    6. gpd.read_file(tmp_path) → parse GeoDataFrame
    7. Delete temp file in finally block (guaranteed cleanup)
    8. Run point-in-polygon test for each returned polygon
    9. If no polygon contains point → use nearest polygon within 0.0005° (~50m)
   10. Reproject result to EPSG:3857
   11. Cache result → return single-row GeoDataFrame
```

**iC1000 API path mapping:**

| `data_type` | API path |
|-------------|----------|
| `GLA` | `/iC1000/gla` |
| `STT` | `/iC1000/stt` |
| All others | `/iC1000/lot` |

Returns `None` gracefully on any API error, non-200 response, empty GML, or no polygon within the distance threshold. All calling modules handle `None` by falling through to their own site polygon fallback chains.

---

## Coordinate Transformation Pipeline

All spatial operations, geometry calculations, map rendering, and dataset queries run in **EPSG:3857** (Web Mercator, metric units). Coordinate conversion follows a fixed three-stage pipeline:

```
Raw input → EPSG:2326 (HK Grid, metres) → EPSG:4326 (WGS84, degrees) → EPSG:3857 (Web Mercator, metres)
```

| Stage | CRS | Used For |
|-------|-----|----------|
| Government API output | EPSG:2326 | Raw lot coordinate from SearchNumber API |
| OSM / osmnx queries | EPSG:4326 | All OpenStreetMap network queries |
| Spatial computation | EPSG:3857 | Distance calc, buffering, rendering, all modules |

All static datasets (`ZONE_REDUCED.gpkg`, `BUILDINGS_FINAL.gpkg`) are pre-projected to EPSG:3857 at load time, eliminating all per-request reprojection overhead.

---

## Data Layer Design

### Static Datasets (preloaded at API startup)

#### Building Height Dataset — `BUILDINGS_FINAL.gpkg`

- Source: Hong Kong government building footprint data
- Original size: 342 000+ rows
- Reduced to: **42 073 rows** (columns stripped to `HEIGHT_M` + `geometry` only)
- Additional runtime filter: `HEIGHT_M > 5 m` applied at startup
- Projection: EPSG:3857
- Used by: `view.py` (height scoring) · `driving.py` (site polygon fallback)

#### Zoning Dataset — `ZONE_REDUCED.gpkg`

- Source: HK OZP (Outline Zoning Plan) spatial data
- Attributes retained: `ZONE_LABEL` · `PLAN_NO` · `geometry`
- Projection: EPSG:3857
- Used by: `context.py` (zoning overlay) · `driving.py` (site polygon fallback) · `resolver.py` (site type classification)

### External Data Sources (fetched per request)

| Source | Used By | Data Fetched |
|--------|---------|-------------|
| osmnx / OpenStreetMap | walking · driving · transport · context | Road networks, amenities, building footprints, land use |
| HK GeoData SearchNumber API | resolver.py | Lot ID → EPSG:2326 coordinates |
| LandsD iC1000 API | resolver.py | Official lot boundary polygons (GML) |
| CartoDB PositronNoLabels | All map modules | Basemap tiles via contextily |

---

## Visualisation Engine

All maps are rendered using **Matplotlib** with **contextily** for basemap tile fetching. Each module builds a layered figure in EPSG:3857 and returns a `BytesIO` PNG buffer.

**Common rendering elements across all modules:**

- CartoDB PositronNoLabels basemap (zoom 14–16 depending on map extent)
- North arrow (upper-right corner)
- Legend with colour-coded layer descriptions
- Site footprint polygon (red fill)
- Title with site identifier and analysis type

**Output specifications:**

| Property | Value |
|----------|-------|
| Figure size | 11 × 11 inches (most modules) |
| Resolution | 120–130 DPI |
| Format | PNG (BytesIO buffer) |
| Backend | Matplotlib Agg (non-interactive) |

---

## PDF Report Generator

The `/report` endpoint generates a full multi-page landscape PDF report using ReportLab's `SimpleDocTemplate`.

**Generation process:**
1. All 7 analysis functions are called sequentially: `walking×2` · `driving` · `transport` · `context` · `view` · `noise`
2. Each function returns a `BytesIO` PNG buffer
3. Each image is scaled to fit the page frame (preserving aspect ratio)
4. Pages are assembled with title, spacer, and image per page
5. PDF is built into a `BytesIO` buffer and returned as a streaming response

**PDF specifications:**

| Property | Value |
|----------|-------|
| Page size | Landscape A4 (297 × 210 mm) |
| Margins | 0.3 inch (all sides) |
| Title font size | 18 pt |
| Cover page | Yes — site identifier + report title |
| Total pages | 8 |

**Report page order:**

| Page | Content |
|------|---------|
| 1 | Cover — site identifier + title |
| 2 | Walking Accessibility (5 min) |
| 3 | Walking Accessibility (15 min) |
| 4 | Driving Distance (15 min) |
| 5 | Transport Network |
| 6 | Context & Zoning |
| 7 | View Analysis |
| 8 | Noise Assessment |

---

## Deployment

### Render Cloud (Primary)

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

**Deployment steps:**
1. Fork this repository
2. Connect to Render via the dashboard — New Web Service → Connect repository
3. Render auto-detects `render.yaml` and configures the service
4. Set environment variables under **Environment → Environment Variables**:

| Variable | Value |
|----------|-------|
| `PYTHON_VERSION` | `3.11.0` |
| `LOG_LEVEL` | `info` |

5. Click **Deploy** — live URL appears in the Render dashboard

### Local Development

```bash
pip install -r requirements.txt

# Development with hot reload
uvicorn app:app --host 0.0.0.0 --port 10000 --reload

# Production-like (no reload)
uvicorn app:app --host 0.0.0.0 --port 10000
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
```

```bash
docker build -t alkf-api .
docker run -p 10000:10000 alkf-api
```

---

## Performance & Optimisation

### Response Time Profile

| Scenario | Response Time |
|----------|--------------|
| Cold start — first request after idle | 10–15 sec |
| Warm request — server active | 5–10 sec |
| Cached request — same input repeated | < 1 sec |
| Full PDF report — all 7 analyses | 45–90 sec |

### Optimisation Details

| Optimisation | Detail |
|--------------|--------|
| Dataset preloading | `.gpkg` files loaded once at startup into memory |
| Building dataset reduction | 342 000+ → 42 000 rows — unused columns stripped |
| Startup-time filtering | `HEIGHT_M > 5 m` applied once at load, not per request |
| CRS pre-projection | All datasets in EPSG:3857 at load — zero per-request reprojection |
| Result caching | `CACHE_STORE` keyed by `MD5(data_type_value_analysis_type)` |
| Lot boundary caching | `_LOT_BOUNDARY_CACHE` keyed by `(round(lon,5), round(lat,5), data_type)` |
| Thread limiting | `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` — free-tier thread control |
| Non-interactive rendering | `matplotlib.use("Agg")` set at module import time |
| DPI | Output at 120–130 DPI — balance between quality and payload size |
| `gc.collect()` | Called post-render in heavy modules to release osmnx graph memory |

---

## Repository Structure

```
automated-site-analysis-api/
│
├── app.py                       # FastAPI entrypoint — all routes, cache, logging, PDF report
├── render.yaml                  # Render cloud deployment configuration
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version pin for Render
│
├── data/
│   ├── BUILDINGS_FINAL.gpkg     # Building footprints — HEIGHT_M > 5m, EPSG:3857 (42k rows)
│   └── ZONE_REDUCED.gpkg        # OZP zoning polygons — ZONE_LABEL, PLAN_NO, EPSG:3857
│
├── images/
│   └── alkf-logo.png            # Logo for README and frontend
│
├── static/
│   └── HK_MTR_logo.png          # MTR logo for driving map station icons
│
└── modules/
    ├── resolver.py              # Multi-type location resolver + LandsD lot boundary fetcher
    ├── walking.py               # Pedestrian isochrone + amenity accessibility map
    ├── driving.py               # Drive-time rings + MTR ingress/egress route map
    ├── transport.py             # Public transit node accessibility + scoring map
    ├── context.py               # OZP zoning + land-use + OSM amenity context map
    ├── view.py                  # 360° polar view sector classification map
    └── noise.py                 # Road traffic noise propagation heatmap
```

---

## Web API Testing Guide

### Health Check

```
GET https://automated-site-analysis-api.onrender.com
```

Expected response:

```json
{ "status": "Automated Site Analysis API Running - Multi Identifier Enabled" }
```

### Swagger UI

```
https://automated-site-analysis-api.onrender.com/docs
```

Use **Try it out** → fill in the request body → **Execute** to test any endpoint interactively.

### Google Colab Test

```python
import requests
from IPython.display import Image, display

BASE_URL = "https://automated-site-analysis-api.onrender.com"

response = requests.post(
    f"{BASE_URL}/view",
    json={"data_type": "LOT", "value": "IL 1657"},
    timeout=60
)

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
    display(Image("output.png"))
else:
    print("Error:", response.status_code, response.text)
```

### Performance Timing Test

```python
import time, requests

start = time.time()
r = requests.post(
    "https://automated-site-analysis-api.onrender.com/walking",
    json={"data_type": "LOT", "value": "IL 1657"},
    timeout=120
)
elapsed = round(time.time() - start, 2)
print(f"Status: {r.status_code} | Time: {elapsed}s | Size: {len(r.content)/1024:.1f} KB")
```

---

## Production Validation Checklist

- [ ] Server starts without error — `uvicorn` logs show startup complete
- [ ] `ZONE_DATA` and `BUILDING_DATA` load without error at startup
- [ ] `HEIGHT_M` column present in `BUILDINGS_FINAL.gpkg`
- [ ] CRS transformation pipeline works: EPSG:2326 → 4326 → 3857
- [ ] `/search` returns results for known lot IDs and addresses
- [ ] All 6 analysis endpoints return HTTP 200 with `image/png` content
- [ ] `/report` returns HTTP 200 with `application/pdf` content
- [ ] PNG renders are visually correct (basemap loads, site polygon visible)
- [ ] PDF downloads correctly and all 8 pages are populated
- [ ] Repeated requests for same lot ID return cached result (< 1 s)
- [ ] No memory crash during full PDF generation (all 7 analyses)
- [ ] CORS headers present in responses (`Access-Control-Allow-Origin: *`)

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `500` on ADDRESS request | Missing `lon`/`lat` in request body | Include coords pre-resolved from `/search` |
| `ValueError: HEIGHT_M column not found` | Wrong dataset version | Verify `BUILDINGS_FINAL.gpkg` has `HEIGHT_M` column in `data/` |
| OSMnx graph empty / timeout | Sparse OSM coverage or network timeout | Retry · check lot coordinates are valid |
| Cold start slow (10–15 s) | Free-tier idle — dataset preload triggered | Expected behaviour on Render free plan |
| PDF generation timeout | All 7 analyses run sequentially | Increase client timeout to 120+ seconds |
| Port conflict on local | Port 10000 already in use | Use `uvicorn app:app --port 8080` |
| CORS errors in browser | CORS middleware not applied | Verify `CORSMiddleware` with `allow_origins=["*"]` in `app.py` |

---

## Future Roadmap

- API key authentication and per-key rate limiting
- Batch multi-lot processing endpoint — array of lot IDs in a single request
- Background job queue (Celery + Redis) for long-running report generation
- Persistent cloud storage for generated PNG/PDF assets (S3 or GCS)
- SaaS dashboard frontend — React + Mapbox GL interactive map interface
- Scalable deployment on Render paid tier or AWS ECS
- `max_drive_minutes` exposed as a request body parameter on `/driving`
- True network isochrones replacing circular drive-time ring approximations
- Walking speed parameter exposed per request
- Async concurrent module execution for faster PDF report generation

---

## Implementation Stages

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Multi-type input resolver (LOT / STT / GLA / LPP / UN / BUILDINGCSUID / LOTCSUID / PRN / ADDRESS) | ✅ Complete |
| 2 | OSM data integration via osmnx | ✅ Complete |
| 3 | Network graph engine — walk + drive Dijkstra routing | ✅ Complete |
| 4 | View sector model — 360° polar classification | ✅ Complete |
| 5 | Noise propagation model — grid-based L₀ formula | ✅ Complete |
| 6 | Visualisation pipeline — matplotlib + contextily | ✅ Complete |
| 7 | Dataset optimisation — 42k buildings, EPSG:3857 pre-projection | ✅ Complete |
| 8 | Modular API refactor — independent module files | ✅ Complete |
| 9 | Render cloud deployment via `render.yaml` | ✅ Complete |
| 10 | In-memory caching + structured logging layer | ✅ Complete |
| 11 | PDF report generator — ReportLab 8-page landscape A4 | ✅ Complete |
| 12 | Unified `/search` endpoint — lot + address + building name | ✅ Complete |
| 13 | ADDRESS type support with pre-resolved coordinates | ✅ Complete |
| 14 | LandsD iC1000 lot boundary API integration (GML → EPSG:3857) | ✅ Complete |
| 15 | Lot boundary in-memory cache `_LOT_BOUNDARY_CACHE` | ✅ Complete |
| 16 | CORS middleware — `allow_origins=["*"]` | ✅ Complete |

---

## Requirements

```
fastapi
uvicorn
geopandas
osmnx
contextily
shapely
pyproj
networkx
numpy
pandas
matplotlib
requests
scikit-learn
reportlab
```

---

## Changelog

### v3.0.0
- **Multi-type input resolver** — LOT, STT, GLA, LPP, UN, BUILDINGCSUID, LOTCSUID, PRN, ADDRESS all supported
- **ADDRESS type** — coordinates pre-resolved at `/search`; passed via `lon`/`lat` fields; resolver bypasses GIS API entirely for this type
- **`/search` endpoint** — unified lot + address + building name search; query prefix auto-detection; returns `data_type`, `source`, `lon`, `lat` per result
- **`get_lot_boundary()`** — LandsD iC1000 API integration; GML via `NamedTemporaryFile`; point-in-polygon with 50 m nearest fallback; result cached by `(lon, lat, data_type)`
- **`LocationRequest` model** — added optional `lon` and `lat` fields for ADDRESS type support
- **`generate_driving()`** — 5-step site polygon fallback chain; progressive OSM building search 80→150→250 m; 3 configurable drive-time ring presets
- **Startup validation** — raises `ValueError` if `HEIGHT_M` column missing from building dataset
- **CORS** — `CORSMiddleware` with `allow_origins=["*"]` added
- **PDF report** — updated to 8 pages (cover + walking×2 + driving + transport + context + view + noise); landscape A4, 0.3 inch margins

### v1.1.0
- PDF report generator — `/report` endpoint, 7-page ReportLab output
- In-memory caching layer — repeated requests return in < 1 s via `CACHE_STORE`
- Structured logging with per-request duration timing

### v1.0.0
- Initial release: 6 analysis modules — walking, driving, transport, context, view, noise
- Render cloud deployment via `render.yaml`
- Building dataset optimised from 342 000+ → 42 000 rows
- All datasets pre-projected to EPSG:3857 at startup

---

**Status: Production Deployed — Render Cloud (Free Tier)**  
**Version: 3.0.0**  
**Live URL: `https://automated-site-analysis-api.onrender.com`**

---

© ALKF – Automated Geospatial Intelligence System
