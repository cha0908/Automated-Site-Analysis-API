# Technical Architecture & System Design Documentation

> **Module:** `generate_context()`
> **Output:** PNG map (120 dpi) via `BytesIO` buffer
> **Dependencies:** osmnx · geopandas · contextily · matplotlib · shapely · sklearn · numpy

---

## Table of Contents

1. [Module Objective](#1-module-objective)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Function Signature & Parameters](#3-function-signature--parameters)
4. [Global Constants](#4-global-constants)
5. [Data Architecture](#5-data-architecture)
6. [Pipeline Phases](#6-pipeline-phases)
   - 6.1 [Location Resolution](#61-location-resolution)
   - 6.2 [Zoning Classification](#62-zoning-classification)
   - 6.3 [OSM Feature Fetch](#63-osm-feature-fetch)
   - 6.4 [Site Footprint Detection](#64-site-footprint-detection)
   - 6.5 [MTR Station Processing](#65-mtr-station-processing)
   - 6.6 [Bus Stop Clustering](#66-bus-stop-clustering)
   - 6.7 [Place Label Extraction](#67-place-label-extraction)
   - 6.8 [Map Rendering](#68-map-rendering)
   - 6.9 [Output](#69-output)
7. [Zoning & Context Rules](#7-zoning--context-rules)
8. [Label Placement Logic](#8-label-placement-logic)
9. [MTR Station Name Resolution](#9-mtr-station-name-resolution)
10. [Map Layers & Styling](#10-map-layers--styling)
11. [Computational Complexity](#11-computational-complexity)
12. [Assumptions & Limitations](#12-assumptions--limitations)
13. [Extensibility Roadmap](#13-extensibility-roadmap)
14. [Quick Start](#14-quick-start)
15. [Summary](#15-summary)

---

## 1. Module Objective

This module generates an **automated site context map** for any Hong Kong land parcel or address. Given a lot number, address string, or coordinates, it:

- Resolves the location to WGS84 coordinates via the HK GeoData API
- Retrieves the official OZP (Outline Zoning Plan) zoning classification
- Fetches surrounding OSM land use, amenity, transport, and building data
- Applies zoning-specific contextual rules to filter relevant POIs
- Clusters bus stops to reduce visual clutter
- Resolves MTR station names with a fallback chain for unnamed features
- Renders a fully annotated map and returns it as a PNG byte buffer

---

## 2. High-Level System Architecture

```
┌───────────────────────────────────────────────────────┐
│                     Input Layer                       │
│          data_type: LOT / ADDRESS / COORDINATES       │
│          value:     "IL 157" / "1 Austin Rd" / lat,lon│
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│              Universal Location Resolver              │
│    modules.resolver.resolve_location()                │
│    → (lon, lat) in WGS84                              │
│                                                       │
│    modules.resolver.get_lot_boundary()                │
│    → lot boundary GeoDataFrame (EPSG:3857) or None    │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                 Zoning Lookup                         │
│    ZONE_DATA (OZP GeoDataFrame, EPSG:3857)            │
│    Point-in-polygon → ZONE_LABEL + PLAN_NO            │
│    infer_site_type()  → RESIDENTIAL / COMMERCIAL /    │
│                          INSTITUTIONAL / HOTEL / MIXED│
│    context_rules()    → OSM tag filter dict           │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│              OSM Data Acquisition                     │
│                                                       │
│  ox.features_from_point()  ×4 parallel fetches        │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │ Land use /   │ │ MTR Stations │ │  Bus Stops    │  │
│  │ Amenity /    │ │ (r=2000 m)   │ │  (r=900 m)    │  │
│  │ Buildings    │ └──────┬───────┘ └──────┬────────┘  │
│  │ (r=1500 m)   │        │                │           │
│  └──────┬───────┘        │                │           │
└─────────┼────────────────┼────────────────┼───────────┘
          │                │                │
          ▼                ▼                ▼
┌───────────────────────────────────────────────────────┐
│              Feature Processing                       │
│                                                       │
│  Site footprint  ── lot boundary OR nearest polygon   │
│  MTR stations    ── name resolution + fallback chain  │
│  Bus stops       ── KMeans clustering (k=6)           │
│  Place labels    ── zoning-filtered + dedup           │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                Map Rendering                          │
│                                                       │
│  CartoDB Positron basemap (zoom=16)                   │
│  Land use polygons  (residential/industrial/park/     │
│                      school/buildings)                │
│  Transport markers  (MTR + bus stops)                 │
│  Site footprint     (red fill, SITE label)            │
│  Place annotations  (collision-aware label placement) │
│  Info box + legend + title                            │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                  Output                               │
│    BytesIO PNG buffer (120 dpi, 12×12 inches)         │
└───────────────────────────────────────────────────────┘
```

---

## 3. Function Signature & Parameters

```python
def generate_context(
    data_type : str,               # "LOT" | "ADDRESS" | "COORDINATES"
    value     : str,               # e.g. "IL 157" | "1 Austin Road West" | "22.3,114.1"
    ZONE_DATA : gpd.GeoDataFrame   # OZP zoning polygons with ZONE_LABEL and PLAN_NO columns
) -> BytesIO:                      # PNG image buffer
```

**Returns:** `BytesIO` object containing a 120 dpi PNG map. Caller is responsible for saving or serving the buffer.

**Raises:** `ValueError` if no OZP zoning polygon is found for the resolved site point.

---

## 4. Global Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `FETCH_RADIUS` | `1500 m` | OSM feature fetch radius for land use / amenity / buildings |
| `MAP_HALF_SIZE` | `900 m` | Half-width of rendered map viewport (900 m each side = 1.8 km total) |
| `NEAREST_NAMED_STATION_M` | `150 m` | Max distance to borrow a name from a nearby named MTR station |
| `MTR_COLOR` | `#ffd166` | Fill colour for MTR station polygons |

---

## 5. Data Architecture

### 5.1 Coordinate Reference Systems

| Layer | Input CRS | Working CRS | Notes |
|-------|-----------|-------------|-------|
| Resolver output | EPSG:4326 | EPSG:4326 | WGS84 lon/lat |
| ZONE_DATA (OZP) | Any | **EPSG:3857** | Reprojected on entry via `to_crs(3857)` |
| OSM features | EPSG:4326 | **EPSG:3857** | All `ox.features_from_point()` calls reprojected |
| Site point / footprint | EPSG:4326 → 3857 | **EPSG:3857** | All spatial ops in metres |
| Rendered map | EPSG:3857 | EPSG:3857 | contextily basemap in Web Mercator |

### 5.2 Required ZONE_DATA Schema

| Column | Type | Description |
|--------|------|-------------|
| `ZONE_LABEL` | str | OZP zone code, e.g. `R(A)`, `C`, `G/IC`, `OU(Hotel)` |
| `PLAN_NO` | str | OZP plan number, e.g. `S/K1/29` |
| `geometry` | Polygon/MultiPolygon | Zone boundary |

### 5.3 Key Intermediate Objects

| Variable | Type | Description |
|----------|------|-------------|
| `site_point` | `shapely.Point` (EPSG:3857) | Centroid used for all distance queries |
| `site_geom` | `shapely.Polygon` (EPSG:3857) | Site footprint (lot or buffered fallback) |
| `site_gdf` | GeoDataFrame | Single-row GDF wrapping `site_geom` for plotting |
| `polygons` | GeoDataFrame | All OSM polygon features within `FETCH_RADIUS` |
| `stations` | GeoDataFrame | MTR station features within 2000 m, top 2 nearest |
| `bus_stops` | GeoDataFrame | Up to 6 clustered bus stop representatives |
| `labels` | GeoDataFrame | Up to 24 deduplicated POI labels |

---

## 6. Pipeline Phases

### 6.1 Location Resolution

```python
lon, lat  = resolve_location(data_type, value)   # → WGS84 decimal degrees
lot_gdf   = get_lot_boundary(lon, lat, data_type) # → GeoDataFrame (EPSG:3857) or None
```

- If `get_lot_boundary()` returns a GeoDataFrame, the official lot polygon is used as the site footprint and `site_point` is set to its centroid.
- If it returns `None`, a fallback footprint is derived in Phase 6.4.

### 6.2 Zoning Classification

```python
ozp       = ZONE_DATA.to_crs(3857)
primary   = ozp[ozp.contains(site_point)].iloc[0]
zone      = primary["ZONE_LABEL"]
SITE_TYPE = infer_site_type(zone)
```

Point-in-polygon lookup against OZP polygons. Raises `ValueError` if no match found.

**`infer_site_type()` logic:**

| `ZONE_LABEL` Prefix | `SITE_TYPE` |
|---------------------|-------------|
| Starts with `R` | `RESIDENTIAL` |
| Starts with `C` | `COMMERCIAL` |
| Starts with `G` | `INSTITUTIONAL` |
| Contains `HOTEL` or starts with `OU` | `HOTEL` |
| All other | `MIXED` |

### 6.3 OSM Feature Fetch

Four separate `ox.features_from_point()` calls, all reprojected to EPSG:3857:

| Call | Tags | Radius | Used For |
|------|------|--------|----------|
| 1 | `landuse`, `leisure`, `amenity`, `building` | 1500 m | Land use polygons + buildings |
| 2 | `railway: station` | 2000 m | MTR station geometry |
| 3 | `highway: bus_stop` | 900 m | Bus stop points |
| 4 | `LABEL_RULES` (zone-specific) | 800 m | POI place labels |

**Sub-filtering from call 1:**

```python
residential = polygons[ landuse == "residential" ]
industrial  = polygons[ landuse in ["industrial", "commercial"] ]
parks       = polygons[ leisure == "park" ]
schools     = polygons[ amenity in ["school", "college", "university"] ]
buildings   = polygons[ building is not null ]
```

### 6.4 Site Footprint Detection

Only executed when `get_lot_boundary()` returned `None`:

```python
candidates = polygons[
    geometry.geom_type in ["Polygon", "MultiPolygon"] AND
    geometry.distance(site_point) < 40 m
]

if candidates:
    site_geom = candidates.sort_by_area(descending=True).geometry[0]
else:
    site_geom = site_point.buffer(40)   # 40 m circular fallback
```

Priority: nearest large polygon → 40 m buffer last resort.

### 6.5 MTR Station Processing

```python
stations["name"]     = name:en  fillna  name
stations["centroid"] = geometry.centroid
stations["dist"]     = centroid.distance(site_point)
stations             = dropna(name).sort_by_dist().head(2)

# Filter to viewport at render time:
stations_in_view = stations[ dist <= MAP_HALF_SIZE ]
```

Name resolution — see §9 for full fallback chain.

### 6.6 Bus Stop Clustering

Raw bus stops within 900 m are collapsed to 6 representative points using KMeans:

```python
if len(bus_stops) > 6:
    coords  = [[g.x, g.y] for g in bus_stops.geometry]
    labels  = KMeans(n_clusters=6, random_state=0).fit(coords).labels_
    bus_stops = bus_stops.groupby(labels).first()   # one rep per cluster
```

This prevents dense bus corridors from overwhelming the map with overlapping markers.

### 6.7 Place Label Extraction

```python
labels["label"] = name:en  fillna  name
labels = dropna(label).drop_duplicates(label).head(24)
```

Tags fetched are determined by `context_rules(SITE_TYPE)` — see §7.

### 6.8 Map Rendering

Rendering order (z-order low → high):

| Z-order | Layer |
|---------|-------|
| 1 | CartoDB Positron basemap |
| 2 | Residential polygons |
| 2 | Industrial/Commercial polygons |
| 2 | Park polygons |
| 2 | School polygons |
| 2 | Building polygons (light grey, low alpha) |
| 9 | Bus stop markers |
| 10 | MTR station polygons |
| 11 | Site footprint |
| 12 | SITE label text |
| 12 | Place annotation boxes |
| 12 | MTR name labels |

Label placement uses a collision-avoidance loop — see §8.

### 6.9 Output

```python
buffer = BytesIO()
plt.savefig(buffer, format="png", dpi=120)
plt.close(fig)
return buffer
```

The figure is 12 × 12 inches at 120 dpi = 1440 × 1440 px output.
`plt.close(fig)` is called explicitly to free memory.

---

## 7. Zoning & Context Rules

`context_rules(SITE_TYPE)` returns a tag filter dict passed to `ox.features_from_point()` for place label fetching. This ensures only contextually relevant POIs are annotated.

| `SITE_TYPE` | OSM Tags Fetched for Labels |
|-------------|----------------------------|
| `RESIDENTIAL` | `amenity`: school, college, university · `leisure`: park · `place`: neighbourhood |
| `COMMERCIAL` | `amenity`: bank, restaurant, market · `railway`: station |
| `INSTITUTIONAL` | `amenity`: school, college, hospital · `leisure`: park |
| `HOTEL` / `MIXED` | `amenity`: True · `leisure`: True *(all types)* |

---

## 8. Label Placement Logic

Labels are placed with a simple greedy collision-avoidance algorithm:

```python
offsets = [(0,35), (0,-35), (35,0), (-35,0), (25,25), (-25,25)]
placed  = []    # list of already-placed shapely Points

for each label:
    p = feature.representative_point()

    if p.distance(site_point) < 140 m:   skip   # too close to SITE marker
    if any p.distance(placed_p) < 120 m: skip   # too close to existing label

    offset = offsets[i % 6]
    ax.text(p.x + dx, p.y + dy, wrapped_label, ...)
    placed.append(p)
```

**`wrap_label(text, width=18)`** wraps text at 18 characters per line using `textwrap.wrap()`.

Limitations: offsets are cyclic, not optimal. Dense POI clusters will still produce some overlap. See §13 for improvement path.

---

## 9. MTR Station Name Resolution

Each station in `stations_in_view` goes through the following name resolution chain:

```
1. Use name:en column                           if non-null non-empty string
         ↓ else
2. Use name column                              if non-null non-empty string
         ↓ else
3. Search named_stations list for nearest
   station within NEAREST_NAMED_STATION_M=150 m
   and borrow its name                          if found within distance
         ↓ else
4. Render label as "UNNAMED"
```

`named_stations` is built at the start of the MTR label loop as a list of `(centroid, name)` tuples from all stations that passed steps 1–2.

---

## 10. Map Layers & Styling

| Layer | Colour | Alpha | Notes |
|-------|--------|-------|-------|
| Residential polygons | `#f2c6a0` (peach) | 0.75 | `landuse=residential` |
| Industrial / Commercial polygons | `#b39ddb` (lavender) | 0.75 | `landuse=industrial/commercial` |
| Park polygons | `#b7dfb9` (mint green) | 0.90 | `leisure=park` |
| School polygons | `#9ecae1` (sky blue) | 0.90 | `amenity=school/college/university` |
| Building polygons | `#d9d9d9` (light grey) | 0.35 | All buildings; low alpha — context only |
| Bus stop markers | `#0d47a1` (dark blue) | — | `markersize=35` |
| MTR station polygons | `#ffd166` (amber) | 0.90 | `MTR_COLOR` constant |
| Site footprint | `#e53935` (red) | — | `edgecolor=darkred`, `linewidth=2` |
| SITE label | White bold | — | Centred on site footprint |
| Place label boxes | White fill | 0.85 | `boxstyle=round,pad=0.25` |
| MTR name boxes | White fill | 0.85 | `pad=1.0`, offset +120 m vertically |
| Basemap | CartoDB Positron | 0.95 | `zoom=16` |

**Info box** (top-left, `transAxes` coordinates):

```
{data_type}: {value}
OZP Plan: {PLAN_NO}
Zoning: {ZONE_LABEL}
Site Type: {SITE_TYPE}
```

---

## 11. Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Location resolution (API) | O(1) | Single HTTP request |
| OZP point-in-polygon lookup | O(Z) | Z = number of OZP polygons; no spatial index used |
| OSM feature fetch ×4 | O(1) network | Results scale with fetch radius and OSM density |
| Bus stop KMeans (k=6) | O(N × k × iter) | N ≤ few hundred; fast in practice |
| Label placement collision check | O(L²) | L ≤ 24; brute-force distance check |
| Map rendering | O(F) | F = total features rendered |

**Note:** OZP point-in-polygon uses a simple `ozp.contains(site_point)` which is O(Z) with no spatial index. For large OZP datasets, wrapping with `STRtree` would reduce this to O(log Z).

---

## 12. Assumptions & Limitations

| Assumption / Limitation | Implication |
|-------------------------|-------------|
| `resolve_location()` and `get_lot_boundary()` return valid results | If resolver fails, the function raises an unhandled exception |
| ZONE_DATA covers the site location | `ValueError` raised if no polygon found — no graceful fallback |
| OSM data is current and complete | Missing amenities / stations will not be shown; no warning raised |
| MTR station geometry exists in OSM as `railway=station` polygons | Point-geometry stations may not render correctly |
| KMeans clustering assumes > 6 bus stops | If exactly 6 or fewer, all are shown as-is without clustering |
| Label collision avoidance is greedy, not optimal | Dense POI areas will have overlapping labels |
| Map viewport is fixed at ±900 m from site centroid | No zoom or pan; edge lots may be partially clipped |
| `ax.set_position()` called after `tight_layout()` | May cause minor layout inconsistency on some backends |

---

## 13. Extensibility Roadmap

| Priority | Feature | Implementation Path |
|----------|---------|---------------------|
| 1 | STRtree spatial index for OZP lookup | Replace `ozp.contains()` with `STRtree.query()` — O(log Z) |
| 2 | Optimised label placement (force-directed) | Replace greedy offset cycle with repulsion-based placement |
| 3 | Dynamic viewport scaling | Scale `MAP_HALF_SIZE` based on lot area or density |
| 4 | Multi-site batch mode | Accept list of `(data_type, value)` pairs; return list of buffers |
| 5 | Additional transport layers | Minibus, tram, ferry terminal overlays |
| 6 | Height data integration | Colour buildings by `building:levels` if available in OSM |
| 7 | Error handling & fallbacks | Graceful degradation if OSM fetch or resolver fails |
| 8 | Async OSM fetching | Run 4 `ox.features_from_point()` calls concurrently with `asyncio` |
| 9 | Legend auto-hide unused layers | Only show legend entries for layers with at least one feature |
| 10 | Export formats | Add PDF / SVG / GeoTIFF export alongside PNG |

---

## 14. Quick Start

### 14.1 Installation

```bash
pip install osmnx geopandas contextily matplotlib shapely scikit-learn numpy
```

### 14.2 Prepare OZP Data

```python
import geopandas as gpd

ZONE_DATA = gpd.read_file("ozp_zones.geojson")
# Required columns: ZONE_LABEL, PLAN_NO, geometry
```

### 14.3 Generate a Map

```python
from context_map import generate_context

# By lot number
buffer = generate_context("LOT", "IL 157", ZONE_DATA)

# By address
buffer = generate_context("ADDRESS", "1 Austin Road West", ZONE_DATA)

# By coordinates (lat,lon)
buffer = generate_context("COORDINATES", "22.3025,114.1714", ZONE_DATA)
```

### 14.4 Save or Serve the Output

```python
# Save to file
with open("site_context.png", "wb") as f:
    f.write(buffer.getvalue())

# Serve via Flask
from flask import send_file
return send_file(buffer, mimetype="image/png")

# Display in Jupyter
from IPython.display import Image
Image(data=buffer.getvalue())
```

---

## 15. Summary

| Component | Technology |
|-----------|------------|
| Location resolution | `modules.resolver` — HK GeoData API |
| Official lot boundary | `modules.resolver.get_lot_boundary()` |
| OZP zoning lookup | GeoPandas point-in-polygon |
| OSM data acquisition | osmnx `features_from_point()` |
| Bus stop clustering | scikit-learn KMeans (k=6) |
| MTR name resolution | Multi-step fallback chain |
| Map rendering | matplotlib + contextily (CartoDB Positron) |
| Collision-aware labelling | Greedy distance-based placement |
| Output | BytesIO PNG buffer (12×12 in, 120 dpi) |

This module forms a self-contained, API-driven site context mapping engine suitable for integration into planning workflows, EIA preparation pipelines, and web-based property analysis tools in the Hong Kong context.

---

> **Module:** `generate_context()`
> **File:** `context_map.py`
> **Output:** PNG BytesIO buffer
> **Depends on:** `modules.resolver.resolve_location`, `modules.resolver.get_lot_boundary`, OZP GeoDataFrame
