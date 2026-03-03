# Traffic-Based Environmental Noise Modelling Engine
## Technical Architecture & System Design Documentation

---

# 1. System Objective

This system computes spatially distributed environmental road traffic noise levels (Leq dB(A)) using:

- Official Annual Traffic Census (ATC) datasets
- Vehicle class composition
- Road geometry from OpenStreetMap
- Low Noise Road Surface (LNRS) spatial overlays
- Physics-based acoustic propagation

The system is designed as a modular, extensible, near-field screening model.

---

# 2. High-Level System Architecture

```
┌───────────────────────────────┐
│        User Input Layer       │
│  (LOT / Address / Coordinates)│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Location Resolver        │
│  (HK GeoData API → EPSG:4326) │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Spatial Data Acquisition    │
│                               │
│  • OSM Road Network           │
│  • ATC Station WFS            │
│  • LNRS WFS                   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Traffic Data Integration    │
│                               │
│  • Load ATC ZIP               │
│  • Extract flow + heavy_pct   │
│  • Spatial snap to roads      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Emission Calculation Layer  │
│                               │
│  • Light vehicle emission     │
│  • Heavy vehicle emission     │
│  • Energy summation           │
│  • LNRS correction            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Propagation Engine (Grid)    │
│                               │
│  • Distance attenuation       │
│  • Ground absorption          │
│  • Energy summation           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Visualization & Export      │
│  • Contours                   │
│  • Façade exposure            │
│  • PNG export                 │
└───────────────────────────────┘
```

---

# 3. Data Architecture

## 3.1 Coordinate Systems

| Layer | CRS |
|-------|------|
| Resolver Output | EPSG:4326 |
| OSM Roads | EPSG:3857 |
| ATC Stations | EPSG:4326 → 3857 |
| LNRS | EPSG:4326 → 3857 |
| Propagation Grid | EPSG:3857 |

All distance-based calculations occur in projected CRS (EPSG:3857).

---

# 4. Data Contracts

## 4.1 ATC ZIP Contract

Input:
```
2024.zip
 ├── S1001.xlsx
 ├── S1002.xlsx
 ...
```

Required fields extracted:

| Variable | Meaning |
|----------|---------|
| AADT | Annual Average Daily Traffic |
| AM Peak | Morning peak |
| PM Peak | Evening peak |
| Commercial % | Heavy vehicle share |

Derived:

```
flow = max(AM_peak, PM_peak)
heavy_pct = commercial_pct / 100
```

---

## 4.2 ATC Station WFS Contract

Required attributes:

| Field | Description |
|-------|-------------|
| Station ID | Must match Excel filename |
| geometry | Point |

Spatial join method:

```
nearest_station = argmin(distance(road_centroid, station_point))
```

Time complexity: O(N_roads × N_stations)

---

## 4.3 LNRS Contract

Geometry: MultiLineString

Applied as:

```
if road intersects LNRS:
    lnrs_corr = -3 dB
else:
    lnrs_corr = 0
```

---

# 5. Emission Model

The model separates traffic into:

- Light vehicles
- Heavy vehicles

### 5.1 Light Vehicle Emission

```
L_light = 27.7 + 10 log10(flow * (1 - heavy_pct)) + 0.02 * speed
```

### 5.2 Heavy Vehicle Emission

```
L_heavy = 23.1 + 10 log10(flow * heavy_pct) + 0.08 * speed
```

### 5.3 Energy Summation

```
L_source = 10 log10(
    10^(L_light/10) + 10^(L_heavy/10)
)
```

### 5.4 Final Link Level

```
L_link = L_source + lnrs_corr
```

---

# 6. Propagation Engine

The domain is discretized into a rectangular grid.

```
Grid Resolution = Δx (meters)
Domain = buffer(site, radius)
```

For each road segment and grid cell:

Distance attenuation:

```
A_div = 20 log10(d + 1)
```

Ground absorption:

```
A_ground = G × 5 log10(d + 1)
```

Cell contribution:

```
L_cell = L_link − A_div − A_ground
```

Energy accumulation:

```
Noise(x,y) = 10 log10( Σ 10^(L_cell/10) )
```

---

# 7. Computational Complexity

Let:

- R = number of road segments
- G = number of grid cells

Propagation complexity:

```
O(R × G)
```

Nearest station snapping:

```
O(R × S)
```

Where S = number of ATC stations.

This is computationally intensive for high-resolution grids.

---

# 8. Performance Considerations

Bottlenecks:

1. Distance computation via shapely (vectorize)
2. Grid resolution scaling
3. Large road networks

Optimizations possible:

- Spatial indexing (R-tree)
- Vectorized distance calculation
- KD-tree for station snapping
- NumPy broadcasting instead of nested loops
- GPU acceleration (CuPy)

---

# 9. Assumptions

- Roads modeled as infinite line sources
- Flat terrain
- No building shielding
- No diffraction modeling
- Steady-state peak hour condition
- No meteorological correction
- No vertical propagation

---

# 10. Limitations

This is a screening-level model.

Not ISO 9613 compliant.
Not CNOSSOS-EU compliant.
Not calibrated to measured field data.

Suitable for:

- Concept planning
- Comparative assessment
- Early-stage design evaluation

---

# 11. Extensibility Roadmap

Possible upgrades:

1. Building shielding diffraction
2. Reflection modelling
3. Terrain (DEM) correction
4. Dynamic time simulation
5. Meteorological correction
6. ISO 9613 implementation
7. CNOSSOS emission model
8. Real-time traffic API integration
9. GPU acceleration
10. 3D noise mapping

---

# 12. Developer Integration Notes

Required configurable parameters:

```
CONFIG["atc_zip_path"]
CONFIG["study_radius"]
CONFIG["grid_resolution"]
CONFIG["ground_absorption"]
CONFIG["scenario"]
```

All other components are API-driven and dynamic.

---

# 13. Summary

This model integrates:

- Government traffic census
- Spatial GIS data
- Acoustic emission modelling
- Physics-based propagation
- Urban spatial visualization

It forms a modular, extensible, near-field traffic noise modelling framework suitable for research and planning applications.

---

Model Version: 1.1  
Type: Near-field screening acoustic model  
