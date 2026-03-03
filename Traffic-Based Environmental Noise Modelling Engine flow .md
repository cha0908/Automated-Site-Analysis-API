# Traffic-Based Environmental Noise Modelling Engine v2.10
## Technical Architecture & System Design Documentation

> **Model Type:** Near-field screening acoustic model
> **Compliance:** Not ISO 9613 compliant · Not CNOSSOS-EU compliant
> **Data Sources:** ATC + LNRS © CSDI HK · Road Network © OpenStreetMap · Basemap © CARTO

---

## Table of Contents

1. [System Objective](#1-system-objective)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Version History](#3-version-history)
4. [Data Architecture](#4-data-architecture)
5. [Data Contracts](#5-data-contracts)
6. [Traffic Assignment](#6-traffic-assignment)
7. [Emission Model](#7-emission-model)
8. [Propagation Engine](#8-propagation-engine)
9. [Visualisation & Export](#9-visualisation--export)
10. [Configuration Reference](#10-configuration-reference)
11. [Computational Complexity](#11-computational-complexity)
12. [Performance Considerations](#12-performance-considerations)
13. [Assumptions](#13-assumptions)
14. [Limitations](#14-limitations)
15. [Extensibility Roadmap](#15-extensibility-roadmap)
16. [Quick Start](#16-quick-start)
17. [Summary](#17-summary)

---

## 1. System Objective

This system computes spatially distributed environmental road traffic noise levels (Leq dB(A)) for urban near-field assessment using:

- Official Annual Traffic Census (ATC) Excel datasets
- Vehicle class composition (light / heavy split)
- Road geometry from OpenStreetMap (OSM)
- Low Noise Road Surface (LNRS) spatial overlays
- Street canyon reflection modelling via building footprints
- Physics-based acoustic propagation on a regular grid
- Road proximity masking to eliminate noise bleed into open areas

The system is designed as a modular, extensible, near-field screening model suitable for concept planning, comparative assessment, and early-stage Environmental Impact Assessment.

---

## 2. High-Level System Architecture

```
┌───────────────────────────────────────────────────────┐
│                   User Input Layer                    │
│         (LOT / Address / Coordinates query)           │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│               Location Resolver                       │
│    HK GeoData API → EPSG:4326 (WGS84)                 │
│    Supports: LOT number · Street address · Lat/Lon    │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│             Spatial Data Acquisition                  │
│                                                       │
│  ┌─────────────────┐  ┌────────────┐  ┌───────────┐  │
│  │  OSM Road Net   │  │ ATC Station│  │ LNRS WFS  │  │
│  │  (osmnx)        │  │ WFS / File │  │ / Fallback│  │
│  └────────┬────────┘  └─────┬──────┘  └─────┬─────┘  │
└───────────┼─────────────────┼───────────────┼─────────┘
            │                 │               │
            ▼                 ▼               ▼
┌───────────────────────────────────────────────────────┐
│            Traffic & Correction Assignment            │
│                                                       │
│  TrafficAssigner → CanyonAssigner → LNRSAssigner      │
│  (ATC snap+fallback) (bldg corridor)  (-3 dB LNRS)   │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│               Emission Engine                         │
│                                                       │
│  L_light  (light vehicles)                            │
│  L_heavy  (heavy vehicles)                            │
│  L_source = energy sum (L_light + L_heavy)            │
│  L_link   = L_source + lnrs_corr + canyon_gain        │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│               Propagation Engine                      │
│                                                       │
│  Grid energy accumulation (NumPy vectorised)          │
│  ── Geometric spreading    20·log₁₀(d+1)              │
│  ── Ground absorption      G × coeff × log₁₀(d+1)    │
│  ── Base reflection        + base_reflection dB       │
│                                                       │
│  Post-processing pipeline (ordered):                  │
│  ① Gaussian smoothing    (smooth_sigma)               │
│  ② Road proximity mask   (road_mask_distance) [FIX R] │
│  ③ Noise floor clamp     (noise_floor_db = 45)        │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│             Visualisation & Export                    │
│                                                       │
│  Noise contourf   (RdYlGn_r, 50–70 dB)    [FIX S]    │
│  Contour labels   (every 10 dB)                       │
│  Building façades (coloured by exposure)              │
│  EPD thresholds   (floating legend box)    [FIX T]    │
│  Basemap overlay  (CartoDB / OSM)                     │
│  PNG export       (200 dpi)                           │
└───────────────────────────────────────────────────────┘
```

---

## 3. Version History

| Version | Key Fixes |
|---------|-----------|
| **v2.10** | **FIX R** — Road proximity mask (STRtree, 80 m default) eliminates noise bleed into parks/water. **FIX S** — Colorbar tightened to 50–70 dB(A) for full gradient utilisation. **FIX T** — EPD threshold lines on colorbar + floating legend box (lower-left). |
| v2.9 | **FIX O** — Canyon reflection bonus computed per road segment from building corridor intersection area. Max +8 dB in dense urban canyons. |
| v2.8 | **FIX N** — Road line sources densified to `densify_spacing` m intervals (default 5 m). Eliminates point-source artefacts in energy field. |
| v2.7 | **FIX M** — HK-calibrated fallback flows: primary 5 500, secondary 4 000, trunk 7 000, motorway 9 000 veh/hr. Default heavy_pct raised to 0.18. |
| v2.6 | **FIX K** — Gaussian smoothing applied before noise floor (not after). **FIX L** — `base_reflection` moved to CONFIG key. |
| v2.5 | **FIX A** — `ground_term_coeff` reduced 5 → 1.5; corrects near-field over-attenuation (max was 55 dB, now 63–70 dB). **FIX B** — LNRS local file fallback (.geojson/.csv/.shp). **FIX C** — Geometry validity check + `make_valid()` repair in `_extract_lines()`. |
| v2.4 | **BUG 1** — ATC ID normalisation (strip prefix + leading zeros). **BUG 2** — LNRS WFS: probe + six typename variants. **BUG 4** — Noise floor 45 dB; sub-floor cells masked to NaN. **BUG 5** — ATC folder case-insensitive resolution (Linux). **BUG 6** — Road-type differentiated fallback flows. **BUG 7** — Contour labels every 10 dB only. |

---

## 4. Data Architecture

### 4.1 Coordinate Reference Systems

| Layer | Input CRS | Working CRS | Notes |
|-------|-----------|-------------|-------|
| Location Resolver output | EPSG:4326 | EPSG:4326 | WGS84 decimal degrees |
| OSM road network | EPSG:4326 | **EPSG:3857** | Reprojected via `to_crs(3857)` |
| ATC station points (WFS) | EPSG:4326 | **EPSG:3857** | Set CRS if absent, then reproject |
| LNRS corridors (WFS/file) | EPSG:4326 | **EPSG:3857** | Assumed 4326 if CRS missing |
| Building footprints | EPSG:4326 | **EPSG:3857** | OSM features |
| Propagation grid | — | **EPSG:3857** | All distance ops in metres |

> All distance-based calculations (attenuation, snapping, masking, canyon) are performed in EPSG:3857 where 1 unit ≈ 1 metre at Hong Kong latitudes (22°N).

### 4.2 Class Responsibility Map

| Class | Phase | Responsibility |
|-------|-------|----------------|
| `LocationResolver` | 1 | LOT/address/coordinate → (lat, lon) via HK GeoData API |
| `ATCLoader` | 1 | Load ATC Excel files from folder or ZIP; parse flows |
| `WFSLoader` | 1 | Fetch ATC station geometry and LNRS from CSDI WFS |
| `TrafficAssigner` | 2 | Snap roads to ATC stations; assign flow/heavy_pct/speed |
| `CanyonAssigner` | 2 | Compute per-road canyon reflection bonus from buildings |
| `LNRSAssigner` | 2 | Apply −3 dB correction to LNRS-intersecting roads |
| `EmissionEngine` | 3 | Compute L_light, L_heavy, L_source, L_link per road |
| `PropagationEngine` | 4 | Grid energy accumulation, smoothing, masking, floor |
| `NoiseVisualizer` | 5 | Render noise map PNG with contours, façades, legend |
| `NoiseModelPipeline` | 6 | Orchestrate all phases; loop over resolved sites |

---

## 5. Data Contracts

### 5.1 ATC Excel File Contract

Files are individual Excel workbooks named by station ID (e.g. `S1001.xlsx`, `ST1002.xls`).
Engines: `xlrd` for `.xls`, `openpyxl` for `.xlsx`.
The loader auto-detects the header row by scanning the first 20 rows for recognisable column name matches.

```
2024.zip
 └── current/
      ├── S1001.xlsx
      ├── S1002.xlsx
      ├── ST1003.xls
      └── ...
```

**Recognised column name variants:**

| Semantic Field | Accepted Column Names |
|----------------|-----------------------|
| AM peak flow | `AM Peak`, `AM_Peak`, `am_peak`, `AMPeak`, `AM`, `Morning Peak`, `Peak AM` |
| PM peak flow | `PM Peak`, `PM_Peak`, `pm_peak`, `PMPeak`, `PM`, `Evening Peak`, `Peak PM` |
| Daily flow | `AADT`, `aadt`, `Annual Average Daily Traffic`, `Daily Flow`, `ADT`, `Total` |
| Heavy vehicle % | `Commercial %`, `Heavy %`, `Heavy_pct`, `HV%`, `CV%`, `Comm %`, `% Heavy` |
| Speed | `Speed`, `speed`, `Speed (km/h)`, `Speed_kmh`, `Spd`, `V (km/h)` |

**Derivation logic:**

```
flow      = max(AM_peak, PM_peak)    if peak data available
          = AADT / 16                if only daily data available
heavy_pct = commercial_pct / 100     if value > 1  (percentage form)
          = commercial_pct           if value ≤ 1  (fraction form)
```

### 5.2 ATC Station ID Normalisation (BUG 1)

ATC file names use prefixed IDs; WFS features use plain numeric IDs.
`_normalise_station_id()` resolves the mismatch on both sides before matching:

```python
def _normalise_station_id(raw):
    s = re.sub(r'^[A-Za-z_\-]+', '', str(raw).strip())  # strip alpha prefix
    return (s.lstrip('0') or '0').lower()               # strip leading zeros
```

| Raw ID | Normalised |
|--------|------------|
| `S1001` | `1001` |
| `ST1001` | `1001` |
| `ATC1001` | `1001` |
| `01001` | `1001` |
| `1001` | `1001` |

### 5.3 ATC Station WFS Contract

```
Endpoint : CSDI HK MapServer WFS
Typename : ATC_STATION_PT
Output   : GeoJSON, EPSG:4326
```

| Required Attribute | Accepted Field Names | Use |
|--------------------|----------------------|-----|
| Station ID | `STATION_NO`, `STATIONNO`, `STATION_ID`, `STN_NO`, `NO`, `ID` | Snapping key |
| Geometry | Point | Spatial snapping |

A `_norm_id` column is added on load by applying `_normalise_station_id()` to the detected ID field.

### 5.4 LNRS Contract

```
Primary:  CSDI HK WFS — six typename variants tried in order:
          noise_lnrs → NOISE_LNRS → td:noise_lnrs →
          epd:noise_lnrs → lnrs → LNRS

Fallback: Local file at CONFIG["lnrs_fallback_path"]
```

**Fallback file formats supported:**

| Extension | Geometry Source | CRS Handling |
|-----------|----------------|--------------|
| `.geojson` / `.gpkg` / `.shp` | Native geometry | Auto-detected; assumed EPSG:4326 if absent |
| `.csv` (WKT) | Column named `geometry`, `geom`, or `wkt` | `shapely.wkt.loads()`; assumed EPSG:4326 |
| `.csv` (lon/lat) | `lon`/`longitude`/`x` + `lat`/`latitude`/`y` | Point features; assumed EPSG:4326 |

**Applied correction:**

```
lnrs_corr = -3 dB(A)    if road intersects any LNRS geometry
           =  0 dB(A)    otherwise
```

Spatial test uses `STRtree.query(road_geom, predicate="intersects")` — O(log F) per road.

---

## 6. Traffic Assignment

### 6.1 ATC Station Snapping

```
SNAP_THRESHOLD = 500 m

for each road segment:
    centroid = road.geometry.centroid
    nearest  = argmin( hypot(centroid - station_coords) )
    if distance(centroid, nearest) <= 500 m:
        use ATC record for this station
    else:
        use road-type fallback
```

**Assignment priority table:**

| Condition | Flow Assigned | heavy_pct | Speed |
|-----------|--------------|-----------|-------|
| ATC match, flow recorded | ATC peak hour flow | ATC commercial % | ATC value or road-type default |
| ATC match, flow missing | Road-type fallback | ATC commercial % | Road-type default |
| No ATC match (> 500 m) | Road-type fallback | CONFIG `default_heavy_pct` = 0.18 | Road-type default |

### 6.2 HK-Calibrated Fallback Flows (FIX M, v2.7)

| OSM `highway` Tag | Fallback Flow (veh/hr) | Default Speed (km/h) |
|-------------------|----------------------|---------------------|
| `motorway` | 9 000 | 110 |
| `trunk` | 7 000 | 80 |
| `primary` | 5 500 | 60 |
| `secondary` | 4 000 | 50 |
| `tertiary` | 2 500 | 40 |
| `residential` | 1 000 | 30 |
| `unclassified` | 1 500 | 40 |
| `living_street` | 300 | 20 |
| `service` | 500 | 20 |
| `default` | 3 500 | 40 |

### 6.3 Canyon Reflection Bonus (FIX O, v2.9)

Models increased noise levels in enclosed street canyons due to multiple façade reflections.

```
corridor     = road_geometry.buffer(canyon_buffer_m)        # default: 25 m each side
covered_area = sum( area(corridor ∩ building_i) )           # all intersecting buildings
canyon_gain  = canyon_max_bonus × min(covered_area / canyon_full_area, 1.0)
             = 8.0 × min(covered_area / 2000 m², 1.0)      dB(A)

Range:  0.0 dB  (open road, no buildings in corridor)
     →  8.0 dB  (fully enclosed urban canyon)
```

Canyon gain is added to `L_link` in the emission stage.

---

## 7. Emission Model

Traffic is separated into light and heavy vehicle streams.
Each stream generates an A-weighted emission level using empirical speed-dependent formulae derived from HK EPD guidance.

### 7.1 Light Vehicle Emission

```
L_light = 27.7  +  10·log₁₀(Q_light)  +  0.02·V       dB(A)
```

### 7.2 Heavy Vehicle Emission

```
L_heavy = 23.1  +  10·log₁₀(Q_heavy)  +  0.08·V       dB(A)
```

**Where:**

| Symbol | Definition |
|--------|------------|
| `Q_light` | `flow × (1 − heavy_pct)`  vehicles/hour |
| `Q_heavy` | `flow × heavy_pct`  vehicles/hour |
| `V` | Road posted speed (km/h) |

Both `Q_light` and `Q_heavy` are clamped to minimum 1.0 to avoid `log(0)`.

### 7.3 Energy Summation

```
L_source = 10·log₁₀( 10^(L_light/10)  +  10^(L_heavy/10) )     dB(A)
```

### 7.4 Final Link Level

```
L_link = L_source  +  lnrs_corr  +  canyon_gain                  dB(A)
```

| Term | Range | Source |
|------|-------|--------|
| `L_source` | Typically 55–75 dB(A) | Emission model |
| `lnrs_corr` | 0 or −3 dB | LNRS spatial overlay |
| `canyon_gain` | 0 to +8 dB | Building corridor area |

---

## 8. Propagation Engine

### 8.1 Grid Setup

```
Domain  = site_polygon.buffer(study_radius)
Grid    = meshgrid( arange(minx, maxx, grid_resolution),
                    arange(miny, maxy, grid_resolution) )
```

**Default parameters:**

```
study_radius    = 150 m   →  domain: 300 × 300 m
grid_resolution =   5 m   →  grid:    60 ×  60 = 3 600 cells
densify_spacing =   5 m   →  line sources densified every 5 m
```

### 8.2 Line Source Densification (FIX N, v2.8)

Each road `LineString` is densified before propagation:

```python
n_pts  = ceil(line.length / densify_spacing) + 1
coords = [ line.interpolate(d) for d in linspace(0, line.length, n_pts) ]
```

This ensures energy is distributed along the full road length rather than only at original OSM node locations, eliminating point-source hotspot artefacts.

### 8.3 Geometry Validation (FIX C, v2.5)

Before coordinate extraction, every road geometry is validated and repaired:

```
if geometry is None or empty           → skip
if not geometry.is_valid               → make_valid(geometry)
                                          fallback: geometry.buffer(0)
if geometry is GeometryCollection      → extract LineString sub-geometries only
if geometry type is not LineString     → skip  (no Point or Polygon sources)
if line.length < 1e-3 m                → skip  (degenerate)
```

### 8.4 Attenuation Formula (FIX A, v2.5)

For each grid cell `(x, y)` and each road segment node pair `(x₁,y₁)→(x₂,y₂)`:

```
d        = perpendicular distance from cell to segment  (metres)

A_div    = 20 × log₁₀(d + 1)                           [geometric spreading, dB]
A_ground = G × ground_term_coeff × log₁₀(d + 1)        [ground absorption, dB]
         = 0.6 × 1.5 × log₁₀(d + 1)

L_cell   = L_link  −  A_div  −  A_ground  +  base_reflection    [dB(A)]
```

**Coefficient correction history:**

| Version | `ground_term_coeff` | Total A at d=10 m | Effect |
|---------|--------------------|-------------------|--------|
| ≤ v2.4 | 5 | 20.8 + 3.1 = **23.9 dB** | Over-attenuated — max only 55 dB |
| ≥ v2.5 | **1.5** | 20.8 + 0.9 = **21.7 dB** | Correct near-field — max 63–70 dB |

### 8.5 Energy Accumulation

```
For each road segment (densified to n_nodes):
    le[x,y] = (1 / n_segs) × sum_i( 10^( L_cell_i[x,y] / 10 ) )

energy[x,y] += le[x,y]               (accumulated over all road segments)

noise[x,y]   = 10 × log₁₀( energy[x,y] + 1e-12 )
```

### 8.6 Post-Processing Pipeline (ordered)

```
Step 1 — Gaussian smoothing
         noise = gaussian_filter(noise, sigma=smooth_sigma)
         Purpose : reduce grid discretisation artefacts
         Applied BEFORE masking so smoothing crosses cell boundaries cleanly

Step 2 — Road proximity mask                                         [FIX R, v2.10]
         min_dist[x,y] = minimum distance from cell to any road segment
         noise[ min_dist > road_mask_distance ] = NaN
         Default : road_mask_distance = 80 m
         Purpose : eliminate noise bleed into parks, hills, open water

Step 3 — Noise floor
         noise[ noise < noise_floor_db ] = NaN
         Default : noise_floor_db = 45.0 dB(A)
         Purpose : suppress low-level background clutter; clean colour palette
```

**Road proximity mask detail (FIX R):**

```python
min_dist = full(X.shape, inf)
for each densified segment (x1,y1) → (x2,y2):
    d = segment_perpendicular_distance(X, Y, x1, y1, x2, y2)
    min_dist = minimum(min_dist, d)

mask = min_dist <= road_mask_distance
noise[~mask] = NaN
```

---

## 9. Visualisation & Export

### 9.1 Colorbar Range (FIX S, v2.10)

Tightened from the v2.4 default [45, 85] to [50, 70] dB(A) so the full `RdYlGn_r`
gradient spans the realistic HK urban noise range.

| Setting | v2.4 | v2.10 | Effect |
|---------|------|-------|--------|
| `contour_levels_min` | 45 dB | **50 dB** | Removes excess green band at bottom |
| `contour_levels_max` | 75 dB | **70 dB** | Focuses gradient on working range |
| `contour_step` | 5 dB | 5 dB | Unchanged |
| `contour_label_step` | 5 dB | **10 dB** | Halves label density (BUG 7) |

### 9.2 EPD Threshold Annotation (FIX T, v2.10)

HK EPD noise limits are shown as dashed horizontal lines on the colorbar.
Labels are placed in a floating legend box inside the map (lower-left), away from colorbar tick labels.

| Threshold | Level | Line Colour | Reference |
|-----------|-------|-------------|-----------|
| HK EPD Day limit | **65 dB(A)** | Orange `#e67e22` | EIAO Technical Memorandum |
| HK EPD Night limit | **70 dB(A)** | Red `#c0392b` | EIAO Technical Memorandum |

### 9.3 Building Façade Levels

```python
for each building centroid (cx, cy):
    col = round((cx - grid_x0) / grid_resolution)
    row = round((cy - grid_y0) / grid_resolution)
    facade_db = noise[row, col]

buildings.plot(column="facade_db", cmap="RdYlGn_r", norm=levels_norm)
```

### 9.4 Map Elements

| Element | Style / Parameters |
|---------|--------------------|
| Noise contourf | `RdYlGn_r`, alpha=0.60, `extend='both'` |
| Fine contour lines | All 5 dB levels, black, lw=0.4, alpha=0.25 |
| Bold contour lines | Every 10 dB, black, lw=0.9, alpha=0.55 |
| Contour labels | `fmt="%d dB"`, fontsize=7, inline |
| Building façades | `RdYlGn_r` by `facade_db`, edgecolor `#3a3a3a` |
| Road network | `#333333`, lw=0.7, alpha=0.45 |
| Site footprint | Facecolor `#e63946`, edgecolor white, lw=1.5 |
| SITE label | Bold white, path_effects stroke red |
| EPD legend box | Lower-left, floating, framealpha=0.88 |
| North arrow | Lower-left quadrant, annotate arrowstyle |
| Stats box | Lower-right: max / mean / min / source range |
| Colorbar | `fraction=0.028`, `pad=0.02`, `aspect=30` |
| Basemap | CartoDB PositronNoLabels → Positron → OSM (fallback chain) |
| Attribution | © OpenStreetMap · © CARTO · © CSDI HK |
| Output DPI | 200 |

---

## 10. Configuration Reference

All parameters are set in the `CONFIG` dictionary at the top of the script.
No code changes are required for standard use — only `CONFIG` edits.

### 10.1 Input

| Key | Default | Description |
|-----|---------|-------------|
| `input_mode` | `"SINGLE"` | `"SINGLE"` or `"MULTIPLE"` |
| `single_query` | `{type: LOT, value: IL 1657}` | Type: `LOT` / `ADDRESS` / `COORDINATES` |
| `multiple_queries` | `[...]` | List of query dicts for batch mode |

### 10.2 ATC Data

| Key | Default | Description |
|-----|---------|-------------|
| `atc_folder_path` | `/content/ATC_2024/.../Current` | Path to extracted ATC Excel files |
| `atc_zip_path` | `/content/2024.zip` | ZIP archive (alternative to folder) |
| `atc_zip_subfolder` | `"current"` | Subfolder name within ZIP to search |

### 10.3 WFS Endpoints

| Key | Description |
|-----|-------------|
| `atc_wfs_url` | CSDI HK ATC station point WFS (GeoJSON output) |
| `lnrs_wfs_url` | CSDI HK LNRS corridor WFS (GeoJSON output) |
| `lnrs_fallback_path` | Local file path if WFS unreachable (`.geojson`/`.csv`/`.shp`) |
| `wfs_timeout` | WFS request timeout in seconds (default: 30) |

### 10.4 Propagation

| Key | Default | Description |
|-----|---------|-------------|
| `study_radius` | `150` | Domain radius from site centre (metres) |
| `grid_resolution` | `5` | Grid cell size (metres) |
| `densify_spacing` | `5.0` | Road line densification interval (metres) |
| `road_mask_distance` | `80.0` | Max cell-to-road distance for valid cells (m). `None` disables mask |

### 10.5 Acoustics

| Key | Default | Description |
|-----|---------|-------------|
| `ground_absorption` | `0.6` | G: 0 = hard ground, 1 = fully absorptive |
| `ground_term_coeff` | `1.5` | Multiplier on G in attenuation formula (was 5 in v2.4) |
| `base_reflection` | `2.0` | Uniform reflection bonus applied to all road links (dB) |
| `canyon_buffer_m` | `25.0` | Canyon corridor half-width either side of road (metres) |
| `canyon_full_area` | `2000.0` | Covered area threshold for maximum canyon bonus (m²) |
| `canyon_max_bonus` | `8.0` | Maximum canyon reflection gain (dB) |

### 10.6 Fallback Traffic

| Key | Default | Description |
|-----|---------|-------------|
| `default_flow` | `3500` | Fallback veh/hr when no ATC match within 500 m |
| `default_heavy_pct` | `0.18` | Fallback heavy vehicle fraction |
| `default_speed` | `40` | Fallback speed (km/h) |
| `road_flow_table` | see §6.2 | Per road-type fallback flows (veh/hr) |
| `road_speed_table` | see §6.2 | Per road-type default speeds (km/h) |

### 10.7 Visualisation

| Key | Default | Description |
|-----|---------|-------------|
| `contour_levels_min` | `50` | Colorbar / contour lower bound dB(A) |
| `contour_levels_max` | `70` | Colorbar / contour upper bound dB(A) |
| `colorbar_max_db` | `70` | Colorbar maximum annotation (dB) |
| `contour_step` | `5` | Contour interval (dB) |
| `contour_label_step` | `10` | Contour label interval (dB) |
| `noise_floor_db` | `45.0` | Cells below this value set to NaN |
| `smooth_sigma` | `1.5` | Gaussian kernel sigma in grid cells |
| `basemap_zoom` | `19` | contextily tile zoom level |
| `output_dpi` | `200` | Output PNG resolution |

### 10.8 Feature Flags (Stubs)

| Key | Default | Description |
|-----|---------|-------------|
| `enable_building_shielding` | `False` | Diffraction attenuation (not implemented) |
| `enable_terrain_correction` | `False` | DEM height correction (not implemented) |
| `enable_met_correction` | `False` | Meteorological correction (not implemented) |
| `enable_iso9613` | `False` | Full ISO 9613-2 propagation (not implemented) |
| `enable_cnossos` | `False` | CNOSSOS-EU emission model (not implemented) |
| `enable_gpu` | `False` | CuPy GPU acceleration (not implemented) |

---

## 11. Computational Complexity

Let:
- `R` = number of road segments
- `N` = densified nodes per segment ≈ road_length / densify_spacing
- `G` = number of grid cells = (2 × radius / dx)²
- `S` = number of ATC stations
- `F` = number of LNRS features
- `B` = number of building footprints

| Operation | Complexity | Dominant Factor |
|-----------|------------|----------------|
| ATC station snapping | O(R × S) | Fast in practice (S < 500) |
| LNRS road intersection | O(R × log F) | STRtree spatial index |
| Canyon building coverage | O(R × log B) | STRtree + polygon intersection |
| Line source densification | O(R × N) | Linear in road length |
| Propagation energy accumulation | **O(R × N × G)** | **Primary bottleneck** |
| Road proximity mask | O(R × N × G) | Same loop structure as propagation |
| Gaussian smoothing | O(G) | scipy.ndimage C implementation |

**Typical scale with default config:**

```
G    = (300 / 5)² = 3 600 cells
R×N  ≈ 200 roads × 30 nodes = 6 000 source points
Ops  ≈ 6 000 × 3 600 = 21.6 M distance evaluations per site
Time ≈ 5–15 seconds on CPU (NumPy vectorised)
```

---

## 12. Performance Considerations

### 12.1 Current Bottlenecks

1. **Propagation inner loop** — O(R × N × G) distance evaluations; dominant cost
2. **Canyon building intersection** — Per-road polygon area; significant in dense areas
3. **Grid resolution scaling** — Doubling resolution (5 m → 2.5 m) quadruples cell count

### 12.2 Implemented Optimisations

| Technique | Applied To | Benefit |
|-----------|-----------|---------|
| NumPy vectorised arrays | Propagation distance calc | Eliminates Python loops over grid cells |
| STRtree spatial index | LNRS + Canyon queries | O(log N) vs O(N) per road |
| Line densification pre-pass | Source extraction | One-time cost; improves accuracy vs artefacts |
| Gaussian smoothing (scipy) | Post-propagation | Optimised C implementation |
| Per-segment energy averaging | Accumulation | Prevents double-counting for long roads |

### 12.3 Possible Further Optimisations

| Optimisation | Expected Speedup | Complexity |
|-------------|-----------------|------------|
| Spatial cutoff radius (skip sources > R_cutoff from cell) | 2–5× | Low |
| KD-tree for station snapping | Marginal (S is small) | Low |
| NumPy broadcasting over all segments simultaneously | 3–10× | Medium |
| CuPy GPU acceleration (`enable_gpu`) | 20–100× | High |
| Multiprocessing over multiple sites | Linear in N_sites | Medium |

---

## 13. Assumptions

| Assumption | Implication |
|------------|-------------|
| Roads modelled as distributed line sources (densified) | Not ISO 9613 point-source equivalent |
| Flat terrain | No DEM height correction applied |
| No building shielding or diffraction | Façade levels overestimated near tall buildings |
| Steady-state peak hour condition | No time-varying or day/night simulation |
| No meteorological correction | No downwind/crosswind adjustment |
| 2D grid only | No vertical slice or multi-storey modelling |
| Building reflections via canyon bonus only | Specular reflections not explicitly ray-traced |
| EPSG:3857 distance ≈ true metric | Accurate at HK latitudes; small error at high latitudes |

---

## 14. Limitations

### 14.1 Model Compliance

- **Not ISO 9613-2 compliant** — missing diffraction, meteorology, and A_misc terms
- **Not CNOSSOS-EU compliant** — different emission model structure and road surface terms
- **Not field-calibrated** — no comparison to measured noise data

### 14.2 Data Limitations

- LNRS correction requires WFS access **or** a local fallback file to be specified; otherwise silently disabled
- ATC matching requires a station within 500 m — roads beyond this threshold use fallback flows only
- ATC data is annual census; does not reflect temporary closures or construction diversions

### 14.3 Suitable Applications

| Use Case | Suitable? |
|----------|-----------|
| Concept planning and option comparison | ✅ Yes |
| Early-stage EIA screening | ✅ Yes |
| Relative noise impact ranking between sites | ✅ Yes |
| Identifying façades needing detailed assessment | ✅ Yes |
| Detailed noise assessment for planning submission | ❌ No |
| Compliance certification | ❌ No |
| Calibrated prediction at specific façade positions | ❌ No |

---

## 15. Extensibility Roadmap

### 15.1 Feature Stubs — Activate via CONFIG Flags

| Flag | Feature | Implementation Path |
|------|---------|---------------------|
| `enable_building_shielding` | Diffraction attenuation over/around buildings | IL-45 barrier model using OSM building heights |
| `enable_terrain_correction` | DEM-based height correction | SRTM/HK DEM integration; adjust d to slant distance |
| `enable_met_correction` | Meteorological downwind correction | ISO 9613-2 Cmet term; wind/temp profile input |
| `enable_iso9613` | Full ISO 9613-2 propagation | Replace attenuation formula with A_div+A_atm+A_gr+A_bar+A_misc |
| `enable_cnossos` | CNOSSOS-EU emission model | Replace L_light/L_heavy with CNOSSOS road emission |
| `enable_gpu` | CuPy GPU acceleration | Replace NumPy propagation arrays with CuPy kernels |

### 15.2 Priority Development Roadmap

| Priority | Feature | Impact |
|----------|---------|--------|
| 1 | Building shielding (IL-45 diffraction) | Major accuracy improvement near tall buildings |
| 2 | ISO 9613-2 full implementation | Compliance-grade output |
| 3 | Time-banded simulation (AM / IP / PM / Night) | Lden and night-time level computation |
| 4 | GPU acceleration (CuPy) | 20–100× speedup for fine grids |
| 5 | 3D noise mapping | Multi-storey façade exposure per floor |
| 6 | Real-time traffic API | Replace annual ATC census with live count data |
| 7 | CNOSSOS-EU emission | EU-harmonised model for cross-border projects |

---

## 16. Quick Start

### 16.1 Installation

```bash
pip install osmnx geopandas contextily pyproj shapely \
            numpy matplotlib requests pillow openpyxl xlrd scipy
```

### 16.2 Minimum Config (no ATC data — fallback flows active)

```python
CONFIG["input_mode"]   = "SINGLE"
CONFIG["single_query"] = {"type": "LOT", "value": "IL 157"}
```

### 16.3 With ATC Excel Files

```python
# Option A — extracted folder:
CONFIG["atc_folder_path"] = "/content/ATC_2024/ATC_2024/2024/Current"

# Option B — ZIP archive:
CONFIG["atc_zip_path"]      = "/content/2024.zip"
CONFIG["atc_zip_subfolder"] = "current"
```

### 16.4 With Local LNRS Fallback

```python
CONFIG["lnrs_fallback_path"] = "/content/lnrs_hk.geojson"
# Also accepts: .gpkg  .shp  .csv (with WKT 'geometry' column)
```

### 16.5 Multiple Sites (Batch Mode)

```python
CONFIG["input_mode"] = "MULTIPLE"
CONFIG["multiple_queries"] = [
    {"type": "LOT",     "value": "IL 1657"},
    {"type": "LOT",     "value": "IL 904"},
    {"type": "ADDRESS", "value": "1 Austin Road West"},
]
```

### 16.6 Run

```python
pipeline = NoiseModelPipeline(CONFIG)
pipeline.run()
# Output files:
#   NOISE_MODEL_LOT_IL_157.png
#   NOISE_MODEL_LOT_IL_1657.png   (if multiple)
```

---

## 17. Summary

| Component | Technology |
|-----------|------------|
| Government traffic census | HK TD ATC Excel files (2024) |
| Spatial GIS data | OpenStreetMap via osmnx |
| WFS data services | CSDI Hong Kong (EPSG:4326) |
| Acoustic emission model | HK EPD empirical speed-dependent formulae |
| Physics-based propagation | NumPy vectorised grid accumulation |
| Street canyon modelling | Building footprint corridor intersection |
| LNRS correction | CSDI WFS spatial overlay (−3 dB) |
| Road proximity masking | STRtree segment-distance filter (FIX R) |
| Urban spatial visualisation | matplotlib + contextily + geopandas |

This engine forms a modular, extensible, near-field traffic noise modelling framework designed for research, planning screening, and comparative EIA assessment in the Hong Kong urban context.

---

> **Model Version:** 2.10
> **Type:** Near-field screening acoustic model
> **Status:** Not ISO 9613 compliant · Not calibrated to field data
> **Maintainer:** Update `CONFIG` only — all other components are data-driven and dynamic
