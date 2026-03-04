<p>
  <img src="images/alkf-logo.png" width="140" align="left">
</p>

<h1 style="margin-top:0;">AUTOMATED SPATIAL INTELLIGENCE SYSTEM API – ALKF+</h1>
<hr>

> Modular geospatial intelligence platform for automated urban feasibility assessment. Converts raw geographic lot data, addresses, and coordinates into structured analytical maps and PDF reports via a cloud-deployed API.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Module Breakdown](#module-breakdown)
4. [API Reference](#api-reference)
5. [Search Endpoint](#search-endpoint)
6. [Deployment](#deployment)
7. [Performance Profile](#performance-profile)
8. [Sample Outputs](#sample-outputs)
9. [Implementation Stages](#implementation-stages)
10. [Future Enhancements](#future-enhancements)
11. [Changelog](#changelog)

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/automated-site-analysis-api.git
cd automated-site-analysis-api

# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn app:app --host 0.0.0.0 --port 10000 --reload
```

The API will be available at `http://localhost:10000`.

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PORT` | Server port | No | `10000` |
| `LOG_LEVEL` | Logging verbosity (`debug`, `info`, `warning`) | No | `info` |
| `CACHE_ENABLED` | Enable in-memory result caching | No | `true` |
| `DPI` | Output image resolution | No | `200` |

### Your First Request

```bash
# By lot number
curl -X POST http://localhost:10000/walking \
  -H "Content-Type: application/json" \
  -d '{"data_type": "LOT", "value": "IL 1657"}' \
  --output walking_analysis.png

# By address (with pre-resolved coordinates from /search)
curl -X POST http://localhost:10000/walking \
  -H "Content-Type: application/json" \
  -d '{"data_type": "ADDRESS", "value": "1 Austin Road West", "lon": 114.1714, "lat": 22.3025}' \
  --output walking_analysis.png
```

---

## System Architecture

### End-to-End Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        User Request                         │
│          LOT / ADDRESS / COORDINATES / BUILDINGCSUID        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Endpoint                         │
│          Request validation (Pydantic LocationRequest)      │
│          Cache check (MD5 key: data_type_value_analysis)    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               modules/resolver.py                           │
│                                                             │
│  resolve_location()                                         │
│  ├── ADDRESS type → use pre-resolved lon/lat directly       │
│  └── All other types → HK GeoData SearchNumber API          │
│      → EPSG:2326 (HK Grid) → EPSG:4326 (WGS84)              │
│                                                             │
│  get_lot_boundary()                                         │
│  ├── ADDRESS type → returns None immediately                │
│  ├── LOT/GLA/STT → LandsD iC1000 API (GML, EPSG:2326)       │
│  │   → bbox query ±300m → point-in-polygon match            │
│  │   → fallback: nearest polygon < 0.0005° (~50m)           │
│  └── Returns single-row GeoDataFrame (EPSG:3857) or None    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Static Dataset Layer (startup preload)         │
│                                                             │
│  ZONE_DATA      ← data/ZONE_REDUCED.gpkg     (EPSG:3857)    │
│  BUILDING_DATA  ← data/BUILDINGS_FINAL.gpkg  (EPSG:3857)    │
│                   filtered: HEIGHT_M > 5 m                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Analysis Modules (parallel capable)            │
│                                                             │
│  modules/walking.py    → Pedestrian isochrone map           │
│  modules/driving.py    → Drive-time ring map + MTR routes   │
│  modules/transport.py  → Transit node accessibility map     │
│  modules/context.py    → Land-use zoning context map        │
│  modules/view.py       → 360° view sector classification    │
│  modules/noise.py      → Road traffic noise heatmap         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Layer                                   │
│                                                             │
│  PNG → StreamingResponse (image/png)                        │
│  PDF → SimpleDocTemplate via reportlab → StreamingResponse  │
│         (application/pdf, attachment)                       │
└─────────────────────────────────────────────────────────────┘
```

### Request Lifecycle (with Cache)

```
Client ──POST /walking {data_type, value}──► FastAPI
                                                │
                             ┌──────────────────▼─────────────────┐
                             │  cache_key = MD5(data_type_value_  │
                             │              analysis_type)        │
                             └──────────────────┬─────────────────┘
                                                │
                          ┌─────────────────────▼──────────────────┐
                          │          CACHE_STORE lookup            │
                          └──────────┬──────────────────┬──────────┘
                               HIT   │                  │  MISS
                                     ▼                  ▼
                            Return cached       resolve_location()
                            PNG buffer          get_lot_boundary()
                            (< 1 sec)           generate_walking()
                                                    │
                                                    ▼
                                           Store in CACHE_STORE
                                           Return PNG buffer
                                           (5–10 sec first time)
```

### Repository Structure

```
Automated-Site-Analysis-API/
│
├── app.py                      # FastAPI entrypoint — routes, cache, logging, PDF report
├── render.yaml                 # Render cloud deployment config
├── requirements.txt
├── runtime.txt
│
├── data/
│   ├── BUILDINGS_FINAL.gpkg    # Building footprints with HEIGHT_M (EPSG:3857, filtered > 5m)
│   └── ZONE_REDUCED.gpkg       # OZP zoning polygons — ZONE_LABEL + PLAN_NO (EPSG:3857)
│
├── images/
│   └── alkf-logo.png
│
├── static/
│   └── HK_MTR_logo.png
│
└── modules/
    ├── resolver.py             # Multi-type location resolver + lot boundary fetcher
    ├── context.py              # Land-use, zoning & amenity context map
    ├── driving.py              # Drive-time ring map with ingress/egress MTR routing
    ├── noise.py                # Road traffic noise propagation heatmap
    ├── transport.py            # Public transit accessibility map
    ├── view.py                 # 360° polar view sector classification
    └── walking.py              # Pedestrian accessibility isochrone map
```

---

## Module Breakdown

### `resolver.py` — Multi-Type Input Resolver

Translates any supported input type into WGS84 coordinates and an official lot boundary polygon.

**Supported input types:**

| `data_type` | Description | Boundary Available |
|-------------|-------------|-------------------|
| `LOT` | Inland lot, NKIL, KIL, etc. | ✅ Via iC1000 API |
| `STT` | Short-term tenancy lot | ✅ Via iC1000 API (stt) |
| `GLA` | Government land allocation | ✅ Via iC1000 API (gla) |
| `LPP` | Licence / permit parcel | ✅ Via iC1000 API |
| `UN` | Utility notation | ✅ Via iC1000 API |
| `BUILDINGCSUID` | Building CSUID | ✅ Via iC1000 API |
| `LOTCSUID` | Lot CSUID | ✅ Via iC1000 API |
| `PRN` | Property reference number | ✅ Via iC1000 API |
| `ADDRESS` | Pre-resolved from `/search` | ❌ Always returns None |

**`resolve_location()` flow:**

```
ADDRESS type:
    lon/lat passed directly from /search result → return immediately
    (no Government API call made)

All other types:
    GET /lus/{data_type}/SearchNumber?text={value}
    → parse candidates[] → select highest score
    → transform EPSG:2326 → EPSG:4326 via pyproj
    → return (lon, lat)
```

**`get_lot_boundary()` flow:**

```
ADDRESS type → return None immediately

All other types:
    1. Check _LOT_BOUNDARY_CACHE[(lon, lat, data_type)]
    2. Transform lon/lat → EPSG:2326 (HK Grid)
    3. Build ±300m bbox in EPSG:2326
    4. GET /iC1000/{lot_type}?bbox={minx},{miny},{maxx},{maxy},EPSG:2326
    5. Write GML response to temp file → gpd.read_file()
    6. Delete temp file (finally block)
    7. Point-in-polygon test for each returned polygon
    8. If no match → use closest polygon within 0.0005° (~50m)
    9. Reproject to EPSG:3857 → cache → return single-row GeoDataFrame
```

**Error handling:** Returns `None` gracefully on any API failure, empty response, or no polygon within distance threshold. Calling modules fall through to OSM/buffer fallback chains.

---

### `walking.py` — Pedestrian Accessibility Analysis

Evaluates walkability and proximity to amenities from the site centroid.

**Method:**
1. Extract walk graph from osmnx within configurable radius
2. Snap site centroid to nearest graph node
3. Run Dijkstra shortest-path routing to all reachable amenity nodes
4. Classify amenities by type (food, health, education, retail, recreation)
5. Generate 5-minute and 15-minute isochrone buffers
6. Render choropleth map with amenity cluster overlays

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Walk speed | 1.39 m/s | Standard pedestrian speed (5 km/h) |
| Graph radius | 1 200 m | osmnx walk network extraction radius |
| Time thresholds | 5 min, 15 min | Isochrone bands |

---

### `driving.py` — Vehicular Connectivity & MTR Route Analysis

Assesses road network reach from the site and computes real ingress/egress routes to the nearest 3 MTR stations.

**Method:**
1. Extract drive graph from osmnx within `graph_dist` (800–2500 m by config)
2. Assign `travel_time` edge weights: `length / (35 km/h in m/min)`
3. Snap site and each MTR station to nearest graph node
4. Compute Dijkstra shortest path: site → station (egress, green) and station → site (ingress, red)
5. Render concentric drive-time rings (dashed gold) at 3 radii per `max_drive_minutes`
6. Place directional arrows at 60% of longest route segment
7. Place TO/FROM labels at map edge with 22° rotation collision avoidance

**Drive-time ring configurations:**

| `max_drive_minutes` | Ring Radii | `map_extent` | `graph_dist` |
|---------------------|-----------|-------------|-------------|
| 5 min | 83 / 250 / 400 m | 600 m | 800 m |
| 10 min | 250 / 500 / 750 m | 875 m | 1 500 m |
| 15 min | 375 / 750 / 1 125 m | 1 400 m | 2 500 m |

**Site polygon detection (5-step fallback chain):**

```
1. Official lot boundary via get_lot_boundary()
2. OZP zone_data polygon (if zone_data provided)
3. OSM building footprint — progressive radius 80 → 150 → 250 m
4. OSM landuse / amenity polygon within 100 m
5. 80 m circular buffer (guaranteed fallback)
```

---

### `transport.py` — Public Transit Accessibility

Scores the site's proximity to and density of public transit infrastructure.

**Method:**
1. Query OSM for bus stops and MTR/rail stations within 800 m
2. Compute transit node density per unit area
3. Score route diversity (number of distinct routes within walking distance)
4. Render node markers scaled by route count

**Scoring:**

```
Transit Score = 0.5 × (node_density_norm) + 0.5 × (route_diversity_norm)
```

---

### `context.py` — Land-Use, Zoning & Amenity Context Map

Provides a spatial overview of the surrounding land-use environment using OZP zoning data and OSM features.

**Layers rendered:**
- OZP zoning overlay (from `ZONE_REDUCED.gpkg`) — colour-coded by zone type
- Amenity point distribution (OSM) — filtered by site type via `context_rules()`
- Green space polygons (parks, recreational areas)
- Building footprint density (light grey overlay)
- Road network classification

**Zoning-driven context filter (`context_rules()`):**

| `SITE_TYPE` | OSM Tags Fetched for Labels |
|-------------|----------------------------|
| `RESIDENTIAL` | school, college, university · park · neighbourhood |
| `COMMERCIAL` | bank, restaurant, market · railway station |
| `INSTITUTIONAL` | school, college, hospital · park |
| `HOTEL` / `MIXED` | all amenity + leisure types |

**Bus stop clustering:** Raw stops within 900 m collapsed to 6 representative points via KMeans (k=6) to reduce visual clutter.

---

### `view.py` — 360° View Sector Classification Engine

Classifies the visual environment from the site across all directions using building height data and OSM green/water features.

**Methodology:**

1. Divide 360° into N equal sectors (default: 36 × 10° sectors)
2. For each sector extract: green ratio, water ratio, building density, average `HEIGHT_M`
3. Normalise all features to [0, 1]
4. Apply composite scoring model:

```
Green Score  = green_ratio
Water Score  = water_ratio
City Score   = height_norm × density_norm
Open Score   = (1 − density_norm) × (1 − height_norm)
```

5. Label each sector by highest-scoring view type:

| Label | Condition |
|-------|-----------|
| `GREEN VIEW` | Dominant green_ratio |
| `WATER VIEW` | Dominant water_ratio |
| `CITY VIEW` | High density + high height |
| `OPEN VIEW` | Low density + low height |

6. Merge adjacent sectors sharing the same dominant type

**Input data:** `BUILDINGS_FINAL.gpkg` — preloaded at startup, filtered to `HEIGHT_M > 5 m`

**Output:** Polar sector diagram with colour-coded view classifications.

---

### `noise.py` — Road Traffic Noise Propagation Model

Simulates noise levels across the surrounding area based on road type and traffic volume using a physics-based propagation grid.

**Base propagation model:**

```
L(r) = L₀ − 20·log₁₀(r)
```

Where `L₀` = source emission level (dB) by road class, `r` = distance from source (m).

**Source emission levels by road class:**

| Road Class | L₀ (dB) |
|------------|---------|
| Motorway | 78 |
| Primary | 72 |
| Secondary | 68 |
| Tertiary | 64 |
| Residential | 58 |

**Correction factors:**

| Factor | Adjustment |
|--------|------------|
| Heavy vehicle (motorway/primary) | +3 dB |
| Building mass barrier attenuation | Variable reduction across façades |
| Ground absorption (soft surfaces) | Up to −3 dB |
| Hard surface reflection (streets/plazas) | +1 to +2 dB |

**Output:** Grid-based noise heatmap (dB scale) with road network overlay.

---

## API Reference

### Base URL

```
https://automated-site-analysis-api.onrender.com
```

### Authentication

No authentication required (current version).

---

### Request Model

All analysis endpoints accept `POST` with `Content-Type: application/json`:

```json
{
  "data_type": "LOT",
  "value":     "IL 1657",
  "lon":       null,
  "lat":       null
}
```

**For `ADDRESS` type, `lon` and `lat` must be pre-resolved from the `/search` endpoint:**

```json
{
  "data_type": "ADDRESS",
  "value":     "129 Repulse Bay Road",
  "lon":       114.1955,
  "lat":       22.2407
}
```

**Accepted `data_type` values:**

| `data_type` | Example `value` | Notes |
|-------------|----------------|-------|
| `LOT` | `IL 1657` | Inland lot |
| `NKIL` / `KIL` | `NKIL 6304` | New Kowloon / Kowloon lot |
| `STT` | `STT 1234` | Short-term tenancy |
| `GLA` | `GLA 567` | Government land allocation |
| `BUILDINGCSUID` | `CSUID_xxx` | Building CSUID |
| `LOTCSUID` | `CSUID_yyy` | Lot CSUID |
| `ADDRESS` | `129 Repulse Bay Road` | Requires `lon` + `lat` from `/search` |

---

### Endpoints

#### `GET /search`

Unified lot ID / address / building name search. Returns candidates for display and pre-resolves coordinates for ADDRESS-type queries.

```
GET /search?q=IL+1657        → lot search only
GET /search?q=STTL+467       → lot search only
GET /search?q=The+Lily       → address/building name search only
GET /search?q=Repulse+Bay    → address/area search
GET /search?q=NKIL           → lot prefix search (up to 100 results)
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | required | Search query |
| `limit` | int | `100` | Max results per source |

**Response:**

```json
{
  "count": 3,
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

**Search logic:**

- Query starting with a known lot prefix (`IL`, `NKIL`, `KIL`, `STTL`, `STL`, `TML`, `TPTL`, `DD`, `RBL`, `KCTL`, `ML`, `GLA`, `LPP`) → **lot search only** (address search skipped to avoid noisy results)
- All other queries → **address/building name search only** via `locationSearch` API
- `ADDRESS` results have `lon`/`lat` pre-resolved; pass these directly to analysis endpoints
- `LOT` results have `lon`/`lat` = null; the resolver handles coordinate lookup per-request

---

#### `POST /walking`

Returns a pedestrian accessibility map (PNG).

**Response:** `200 OK` — `image/png` stream

**Errors:**

| Code | Reason |
|------|--------|
| `500` | OSMnx graph extraction failed / resolver error |

---

#### `POST /driving`

Returns a drive-time ring map with ingress/egress MTR routes (PNG).

**Response:** `200 OK` — `image/png` stream

---

#### `POST /transport`

Returns a public transit accessibility map with scored bus/MTR nodes (PNG).

**Response:** `200 OK` — `image/png` stream

---

#### `POST /context`

Returns a land-use context map with OZP zoning and amenity overlays (PNG).

**Response:** `200 OK` — `image/png` stream

---

#### `POST /view`

Returns a 360° polar view classification diagram (PNG).

**Response:** `200 OK` — `image/png` stream

---

#### `POST /noise`

Returns a grid-based road traffic noise propagation heatmap (PNG).

**Response:** `200 OK` — `image/png` stream

---

#### `POST /report`

Generates a combined multi-page PDF report containing all analyses.

**Response:** `200 OK` — `application/pdf` stream
**Content-Disposition:** `attachment; filename=site_analysis_report.pdf`

**Report page order:**

| Page | Content |
|------|---------|
| 1 | Cover page — site identifier + title |
| 2 | Walking Accessibility (5 min) |
| 3 | Walking Accessibility (15 min) |
| 4 | Driving Distance |
| 5 | Transport Network |
| 6 | Context & Zoning |
| 7 | View Analysis |
| 8 | Noise Assessment |

**PDF generation:** ReportLab `SimpleDocTemplate`, landscape A4, 0.3-inch margins. Each analysis map is scaled to fit the page while preserving aspect ratio.

---

#### `GET /`

Health check endpoint.

```json
{ "status": "Automated Site Analysis API Running - Multi Identifier Enabled" }
```

---

### Full curl Workflow

```bash
BASE="https://automated-site-analysis-api.onrender.com"

# Step 1: Search for a lot or address
curl "$BASE/search?q=IL+1657"

# Step 2: Run individual analyses (LOT type)
LOT='{"data_type": "LOT", "value": "IL 1657"}'
curl -s -X POST $BASE/walking   -H "Content-Type: application/json" -d "$LOT" -o walking.png
curl -s -X POST $BASE/driving   -H "Content-Type: application/json" -d "$LOT" -o driving.png
curl -s -X POST $BASE/transport -H "Content-Type: application/json" -d "$LOT" -o transport.png
curl -s -X POST $BASE/context   -H "Content-Type: application/json" -d "$LOT" -o context.png
curl -s -X POST $BASE/view      -H "Content-Type: application/json" -d "$LOT" -o view.png
curl -s -X POST $BASE/noise     -H "Content-Type: application/json" -d "$LOT" -o noise.png
curl -s -X POST $BASE/report    -H "Content-Type: application/json" -d "$LOT" -o report.pdf

# Step 3: ADDRESS type — use lon/lat from /search result
ADDR='{"data_type": "ADDRESS", "value": "129 Repulse Bay Road", "lon": 114.1955, "lat": 22.2407}'
curl -s -X POST $BASE/walking -H "Content-Type: application/json" -d "$ADDR" -o walking_addr.png
```

---

### Python Client Example

```python
import requests

BASE_URL = "https://automated-site-analysis-api.onrender.com"

def search(query: str):
    resp = requests.get(f"{BASE_URL}/search", params={"q": query})
    resp.raise_for_status()
    return resp.json()["results"]

def fetch_analysis(endpoint: str, data_type: str, value: str,
                   lon: float = None, lat: float = None,
                   output_path: str = None):
    payload = {"data_type": data_type, "value": value}
    if lon is not None: payload["lon"] = lon
    if lat is not None: payload["lat"] = lat

    resp = requests.post(f"{BASE_URL}/{endpoint}", json=payload, timeout=60)
    resp.raise_for_status()
    if output_path:
        with open(output_path, "wb") as f:
            f.write(resp.content)
    return resp.content

# LOT type — coordinates resolved server-side
fetch_analysis("walking", "LOT", "IL 1657", output_path="walking.png")
fetch_analysis("report",  "LOT", "IL 1657", output_path="report.pdf")

# ADDRESS type — get coords from /search first
results = search("The Lily")
addr = results[0]
fetch_analysis("context", "ADDRESS", addr["address"],
               lon=addr["lon"], lat=addr["lat"],
               output_path="context.png")
```

---

## Deployment

### Render (Cloud)

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

**Steps:**
1. Fork this repository
2. Connect to Render via the dashboard
3. Render detects `render.yaml` automatically
4. Set environment variables in Render dashboard → **Environment**:

| Variable | Value |
|----------|-------|
| `PYTHON_VERSION` | `3.11.0` |
| `LOG_LEVEL` | `info` |

---

### Local Development

```bash
pip install -r requirements.txt

# Hot reload
uvicorn app:app --host 0.0.0.0 --port 10000 --reload

# Production-like
uvicorn app:app --host 0.0.0.0 --port 10000
```

---

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

### Troubleshooting

**Cold start is slow (10–15 s)**
Expected on Render free plan. First request after idle triggers `ZONE_DATA` and `BUILDING_DATA` preload. Subsequent requests: 5–10 s. Cached requests: < 1 s.

**OSMnx graph empty / `InsufficientResponseError`**
Limited OSM road coverage in the area, or network timeout. Retry or check the lot location.

**`500 Internal Server Error` — ADDRESS type**
Ensure `lon` and `lat` are passed in the request body. ADDRESS type requires pre-resolved coordinates from `/search` — the resolver does not call the GIS API for this type.

**`ValueError: HEIGHT_M column not found`**
`BUILDINGS_FINAL.gpkg` is missing the `HEIGHT_M` field. Verify the dataset was not accidentally replaced with a version lacking this column.

**Port conflict on local**
```bash
uvicorn app:app --port 8080
```

---

## Performance Profile

| Scenario | Response Time |
|----------|--------------|
| Cold start (first request after idle) | 10–15 sec |
| Normal request (warm server) | 5–10 sec |
| Cached request (same data_type + value repeated) | < 1 sec |
| Full PDF report (all 7 analyses) | 45–90 sec |

### Optimisation Details

- **Dataset reduction** — Buildings reduced from 342 000+ → 42 000 rows; unused attributes stripped; `HEIGHT_M` precomputed
- **Startup filtering** — `BUILDING_DATA` filtered to `HEIGHT_M > 5 m` at load time, not per request
- **CRS standardisation** — All datasets converted to EPSG:3857 at startup; no per-request reprojection
- **In-memory caching** — `CACHE_STORE` keyed by `MD5(data_type_value_analysis_type)`; invalidated on server restart
- **Thread limiting** — `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` prevent thread explosion in headless server
- **Agg backend** — `matplotlib.use("Agg")` at module level; no display required
- **DPI** — Output images at 130–200 DPI for web-optimised payloads

---

## Sample Outputs

> All outputs below generated for lot `IL 1657`.

| Analysis | Output |
|----------|--------|
| Walking Accessibility (5 min) | Pedestrian isochrone map with amenity cluster markers |
| Walking Accessibility (15 min) | Extended isochrone with wider amenity coverage |
| Driving Distance (15 min) | Drive-time rings + ingress/egress MTR route lines |
| Transport Network | Bus stop and MTR node map scored by route count |
| Context & Zoning | OZP zoning overlay with green space and amenity distribution |
| 360° View Classification | Polar sector diagram (GREEN / WATER / CITY / OPEN) |
| Noise Assessment | Grid-based dB heatmap with road type overlay |
| PDF Report | 8-page combined report (cover + 7 analyses) |

---

## Implementation Stages

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Multi-Type Input Resolver (LOT / ADDRESS / CSUID / STT / GLA / PRN) | ✅ Complete |
| 2 | OSM Data Integration | ✅ Complete |
| 3 | Network Graph Engine (walk + drive) | ✅ Complete |
| 4 | View Sector Model (360° polar classification) | ✅ Complete |
| 5 | Noise Propagation Model | ✅ Complete |
| 6 | Visualisation Pipeline | ✅ Complete |
| 7 | Dataset Optimisation (42k buildings, EPSG:3857) | ✅ Complete |
| 8 | Modular API Refactor | ✅ Complete |
| 9 | Render Cloud Deployment | ✅ Complete |
| 10 | Caching & Logging Layer | ✅ Complete |
| 11 | PDF Report Generator (ReportLab, 8-page) | ✅ Complete |
| 12 | Unified `/search` endpoint (lot + address + building name) | ✅ Complete |
| 13 | ADDRESS type support with pre-resolved coordinates | ✅ Complete |
| 14 | LandsD iC1000 lot boundary API integration (GML → EPSG:3857) | ✅ Complete |

---

## Future Enhancements

- API key authentication + rate limiting
- Batch multi-lot processing endpoint
- Background job queue (Celery + Redis) for long-running reports
- Persistent cloud storage for generated assets (S3 / GCS)
- SaaS dashboard frontend (React + Mapbox GL)
- Scalable deployment on Render paid tier or AWS ECS
- `max_drive_minutes` parameter exposed on `/driving` endpoint
- Isochrone-based station filtering for `/driving` (network distance vs Euclidean)

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
- **Multi-type input resolver** — supports LOT, STT, GLA, LPP, UN, BUILDINGCSUID, LOTCSUID, PRN, ADDRESS
- **ADDRESS type** — coordinates pre-resolved in `/search`; passed directly to all analysis endpoints via `lon`/`lat` fields; resolver skips GIS API call entirely
- **`/search` endpoint** — unified lot ID + address + building name search; auto-detects lot vs address query by prefix; returns `data_type`, `source`, `lon`, `lat` per result
- **`get_lot_boundary()`** — LandsD iC1000 API integration; GML response via temp file; point-in-polygon match with 50m fallback; result cached by `(lon, lat, data_type)`
- **`LocationRequest` model** — added optional `lon` and `lat` fields for ADDRESS pre-resolution
- **`generate_driving()`** — 5-step site polygon fallback chain; progressive OSM building radius 80→150→250m; drive-time ring configs for 5/10/15 min horizons
- **Startup validation** — raises `ValueError` on missing `HEIGHT_M` column in building dataset
- **CORS** — `allow_origins=["*"]` middleware added

### v1.1.0
- PDF report generator (`/report` endpoint) — 7-page ReportLab output
- In-memory caching layer — repeated requests return in < 1 s
- Structured logging with request timing

### v1.0.0
- Initial release: walking, driving, transport, context, view, noise modules
- Render cloud deployment via `render.yaml`
- Building dataset optimised from 342k → 42k rows
- CRS standardised to EPSG:3857 across all datasets

---

© ALKF – Automated Geospatial Intelligence System
