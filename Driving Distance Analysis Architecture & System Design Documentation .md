# Technical Architecture & System Design Documentation

> **Module:** `generate_driving()`
> **Output:** PNG map (130 dpi) via `BytesIO` buffer
> **Dependencies:** osmnx · geopandas · contextily · matplotlib · networkx · shapely · numpy

---

## Table of Contents

1. [Module Objective](#1-module-objective)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Function Signature & Parameters](#3-function-signature--parameters)
4. [Global Constants & Configuration](#4-global-constants--configuration)
5. [Ring Configurations](#5-ring-configurations)
6. [Data Architecture](#6-data-architecture)
7. [Pipeline Phases](#7-pipeline-phases)
   - 7.1 [Location Resolution](#71-location-resolution)
   - 7.2 [Site Polygon Detection](#72-site-polygon-detection)
   - 7.3 [Drive Network Graph](#73-drive-network-graph)
   - 7.4 [MTR Station Processing](#74-mtr-station-processing)
   - 7.5 [Route Computation](#75-route-computation)
   - 7.6 [Map Rendering](#76-map-rendering)
   - 7.7 [Output](#77-output)
8. [Routing Engine](#8-routing-engine)
9. [Route Arrow Placement](#9-route-arrow-placement)
10. [TO/FROM Label Placement](#10-tofrom-label-placement)
11. [MTR Logo Rendering](#11-mtr-logo-rendering)
12. [Map Layers & Styling](#12-map-layers--styling)
13. [Computational Complexity](#13-computational-complexity)
14. [Assumptions & Limitations](#14-assumptions--limitations)
15. [Extensibility Roadmap](#15-extensibility-roadmap)
16. [Quick Start](#16-quick-start)
17. [Summary](#17-summary)

---

## 1. Module Objective

This module generates a **site driving distance analysis map** for any Hong Kong land parcel, address, or coordinate. For a given site and drive-time horizon (5 / 10 / 15 minutes), it:

- Resolves the site location to WGS84 coordinates
- Builds a real road network graph via osmnx within the required radius
- Computes travel-time-weighted shortest paths to the nearest 3 MTR stations
- Renders ingress (station → site) and egress (site → station) routes with directional arrows
- Overlays concentric drive-time rings labelled with distance and time
- Places collision-aware TO/FROM station labels at the map edge
- Renders the MTR logo icon at each station via `matplotlib.offsetbox`
- Returns the final map as a PNG `BytesIO` buffer

---

## 2. High-Level System Architecture

```
┌───────────────────────────────────────────────────────┐
│                     Input Layer                       │
│   data_type : LOT / ADDRESS / COORDINATES             │
│   value     : "IL 157" / "1 Austin Rd" / "22.3,114.1"│
│   max_drive_minutes : 5 | 10 | 15                     │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│              Universal Location Resolver              │
│   resolve_location()  →  (lon, lat) WGS84             │
│   get_lot_boundary()  →  GeoDataFrame or None         │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│              Site Polygon Detection                   │
│                                                       │
│   Priority 1 : Official lot boundary (get_lot_boundary)│
│   Priority 2 : OZP zone_data polygon (if provided)    │
│   Priority 3 : OSM building footprint (80→150→250 m)  │
│   Priority 4 : OSM landuse / amenity polygon          │
│   Priority 5 : 80 m circular buffer (guaranteed)      │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│              Drive Network Graph                      │
│   ox.graph_from_point()  — drive network              │
│   Edge weight : travel_time = length / (35 km/h)      │
│   site_node   = nearest_nodes(G, lon, lat)            │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│              MTR Station Acquisition                  │
│   ox.features_from_point(railway=station)             │
│   Sort by distance → top 3 nearest stations           │
│   Snap each to nearest graph node                     │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│              Route Computation  (per station)         │
│                                                       │
│   Egress  : site_node → station_node  (green)         │
│   Ingress : station_node → site_node  (red)           │
│   Algorithm: nx.shortest_path(weight="travel_time")   │
│   Output  : GeoDataFrame of route geometry (3857)     │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                   Map Rendering                       │
│                                                       │
│   CartoDB PositronNoLabels basemap                    │
│   Road network (grey, low alpha)                      │
│   Drive-time rings (dashed, gold)                     │
│   Ingress / Egress routes (red / green + arrows)      │
│   Station polygons + MTR logo icons                   │
│   TO/FROM labels (edge-placed, collision-aware)       │
│   Site footprint + OUT/IN label                       │
│   North arrow + legend + title                        │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                     Output                            │
│   BytesIO PNG buffer (130 dpi, 11×11 inches)          │
└───────────────────────────────────────────────────────┘
```

---

## 3. Function Signature & Parameters

```python
def generate_driving(
    data_type         : str,                        # "LOT" | "ADDRESS" | "COORDINATES"
    value             : str,                        # e.g. "IL 157" | "1 Austin Rd West" | "22.3,114.1"
    zone_data         : gpd.GeoDataFrame = None,    # Optional OZP zoning polygons (EPSG: any)
    max_drive_minutes : int              = 15        # Drive-time horizon: 5 | 10 | 15
) -> BytesIO:                                       # PNG image buffer
```

**Returns:** `BytesIO` object containing a 130 dpi PNG map at 11×11 inches.

**Fallback behaviour:** if `max_drive_minutes` is not in `{5, 10, 15}`, it is silently reset to `15`.

---

## 4. Global Constants & Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `DRIVE_SPEED` | `35 km/h` | Assumed average urban driving speed for travel-time weighting |
| `INGRESS_COLOR` | `#e74c3c` (red) | Route colour: station → site |
| `EGRESS_COLOR` | `#27ae60` (green) | Route colour: site → station |
| `_MTR_LOGO_PATH` | `../static/HK_MTR_logo.png` | Path to MTR logo image relative to module file |

**Thread safety settings (set at module load):**

```python
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
```

Prevents KMeans / NumPy from spawning excess threads in a server context.

**Matplotlib backend:**

```python
matplotlib.use("Agg")
```

Non-interactive backend — required for headless server rendering (no display required).

---

## 5. Ring Configurations

Drive-time rings, map viewport, and network fetch radius are fully determined by `max_drive_minutes` via `RING_CONFIGS`:

| `max_drive_minutes` | Ring Radii (m) | Ring Labels | `map_extent` (m) | `graph_dist` (m) |
|---------------------|---------------|-------------|-----------------|-----------------|
| **5** | 83, 250, 400 | 1.5 min / 3 min / 5 min | 600 | 800 |
| **10** | 250, 500, 750 | 3 min / 6 min / 10 min | 875 | 1500 |
| **15** | 375, 750, 1125 | 1.5 min / 3 min / 4.5 min | 1400 | 2500 |

**Basemap zoom level selection (derived from `map_extent`):**

```
map_extent ≤ 650 m  →  zoom 16
map_extent ≤ 950 m  →  zoom 15
map_extent >  950 m →  zoom 14
```

**Ring radius derivation (using `DRIVE_SPEED = 35 km/h`):**

```
distance (m) = speed (m/min) × time (min)
             = (35 000 / 60) × t
             ≈ 583 m/min × t

5 min  → 400 m outermost ring   (≈ 583 × 0.67 rounded)
10 min → 750 m outermost ring   (≈ 583 × 1.28 rounded)
15 min → 1125 m outermost ring  (≈ 583 × 1.93 rounded)
```

---

## 6. Data Architecture

### 6.1 Coordinate Reference Systems

| Layer | Input CRS | Working CRS | Notes |
|-------|-----------|-------------|-------|
| Resolver output | EPSG:4326 | EPSG:4326 | WGS84 lon/lat |
| Site polygon | EPSG:3857 | **EPSG:3857** | All geometry ops in metres |
| Drive network graph G | EPSG:4326 (internal) | EPSG:4326 for node lookup | osmnx stores nodes as lon/lat |
| Route GeoDataFrames | EPSG:4326 → 3857 | **EPSG:3857** | `route_to_gdf().to_crs(3857)` |
| MTR stations | EPSG:4326 → 3857 | **EPSG:3857** | `features_from_point().to_crs(3857)` |
| OSM fallback polygons | EPSG:4326 → 3857 | **EPSG:3857** | Building / landuse features |
| Rendered map | EPSG:3857 | EPSG:3857 | contextily in Web Mercator |

### 6.2 Key Intermediate Objects

| Variable | Type | Description |
|----------|------|-------------|
| `site_pt_3857` | `shapely.Point` (3857) | Site centroid; all distance queries use this |
| `site_poly` | `shapely.Polygon` (3857) | Site footprint (from 5-step detection chain) |
| `site_gdf` | GeoDataFrame (3857) | Single-row wrapper for plotting |
| `centroid` | `shapely.Point` (3857) | `site_poly.centroid`; map centre |
| `G` | `networkx.MultiDiGraph` | Drive network with `travel_time` edge weights |
| `site_node` | int | Nearest osmnx graph node ID to site |
| `stations` | GeoDataFrame (3857) | Top 3 nearest MTR stations |
| `egress_gdf` | GeoDataFrame (3857) | Route geometry: site → station |
| `ingress_gdf` | GeoDataFrame (3857) | Route geometry: station → site |

---

## 7. Pipeline Phases

### 7.1 Location Resolution

```python
lon, lat     = resolve_location(data_type, value)
site_pt_3857 = GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
lot_gdf      = get_lot_boundary(lon, lat, data_type)   # → GeoDataFrame or None
```

### 7.2 Site Polygon Detection

A 5-step fallback chain guarantees a visible site footprint under all conditions:

```
Step 1 — Official lot boundary
         if get_lot_boundary() returns GeoDataFrame:
             site_poly = lot_gdf.geometry.iloc[0]
             site_gdf  = lot_gdf
             → DONE

Step 2 — OZP zone polygon (if zone_data provided)
         ozp     = zone_data.to_crs(3857)
         matches = ozp[ ozp.contains(site_pt_3857) ]
         site_poly = matches.geometry.iloc[0]
         → DONE if non-empty

Step 3 — OSM building footprint (progressive radius expansion)
         for dist in [80, 150, 250]:
             fetch buildings within dist m
             sort by area descending
             use largest if centroid.distance(site_pt) < dist×2
         → DONE if found

Step 4 — OSM landuse / amenity polygon
         fetch landuse=True within 100 m, then amenity=True within 100 m
         use nearest polygon centroid to site_pt
         → DONE if found

Step 5 — Guaranteed buffer fallback
         site_poly = site_pt_3857.buffer(80)   # 80 m visible circle
         → ALWAYS succeeds
```

### 7.3 Drive Network Graph

```python
G = ox.graph_from_point((lat, lon),
                         dist=cfg["graph_dist"],
                         network_type="drive",
                         simplify=True)

# Add travel_time edge weight
for u, v, k, data in G.edges(keys=True, data=True):
    data["travel_time"] = data["length"] / (DRIVE_SPEED * 1000 / 60)
    #                   = metres / (35 000 m / 60 min)
    #                   = travel time in minutes

site_node = ox.distance.nearest_nodes(G, lon, lat)
```

`graph_dist` is set per `RING_CONFIGS` to ensure the network covers the full drive-time horizon plus a buffer for routing.

### 7.4 MTR Station Processing

```python
stations = ox.features_from_point(
    (lat, lon),
    tags={"railway": "station"},
    dist=max(cfg["max_radius"] * 2, 1500)
).to_crs(3857)

stations["dist"] = stations.centroid.distance(centroid)
stations = stations.sort_values("dist").head(3)
```

Top 3 nearest stations are processed. Each station is snapped to the drive graph:

```python
st_wgs  = GeoSeries([st_cen], crs=3857).to_crs(4326).iloc[0]
st_node = ox.distance.nearest_nodes(G, st_wgs.x, st_wgs.y)
```

### 7.5 Route Computation

For each station, two routes are computed independently:

```python
def _route(node_from, node_to):
    path = nx.shortest_path(G, node_from, node_to, weight="travel_time")
    return ox.routing.route_to_gdf(G, path).to_crs(3857)

egress_gdf  = _route(site_node, st_node)   # site → station (GREEN)
ingress_gdf = _route(st_node,   site_node) # station → site (RED)
```

Returns `None` silently if no path exists (disconnected graph component). Both routes are plotted independently — they may differ due to one-way streets.

### 7.6 Map Rendering

Rendering order (z-order low → high):

| Z-order | Layer |
|---------|-------|
| 0 | CartoDB PositronNoLabels basemap |
| 1 | Drive road network (grey, 35% alpha) |
| 2 | Outermost ring fill (yellow, 18% alpha) |
| 5 | Ring boundary lines (dashed gold) |
| 8 | Egress routes (green) |
| 9 | Ingress routes (red) |
| 10 | Ring distance/time labels |
| 11 | Site footprint (red fill) |
| 12 | SITE text label |
| 15 | MTR logo icon |
| 16 | OUT/IN label box |
| 18 | TO/FROM station labels |
| 20 | Egress route arrow |
| 21 | Ingress route arrow |
| 23 | Station polygon |
| 24 | MTR logo icon (station) |
| 25 | Station name label |

### 7.7 Output

```python
buf = BytesIO()
plt.tight_layout()
plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)
gc.collect()
return buf
```

`gc.collect()` is called twice (post-graph build, post-render) to release osmnx network objects from memory in a long-running server process.

---

## 8. Routing Engine

### 8.1 Algorithm

```
nx.shortest_path(G, source, target, weight="travel_time")
```

Dijkstra's algorithm on a directed multigraph. Edge weight is travel time in minutes derived from segment length and assumed speed:

```
travel_time (min) = length (m) / (35 000 m / 60 min)
                  = length / 583.3
```

### 8.2 Speed Assumption

`DRIVE_SPEED = 35 km/h` is a fixed urban average. This does not account for:
- Traffic signals or congestion
- Time-of-day variation
- Road category speed limits (motorway vs residential)

All three rings use the same speed; ring radii are computed directly from this constant.

### 8.3 Ingress vs Egress

Both directions are computed because one-way streets can produce materially different routes in dense urban grids. They are plotted in different colours:

| Direction | Colour | Label |
|-----------|--------|-------|
| Egress: site → station | `#27ae60` green | EGRESS ROUTING |
| Ingress: station → site | `#e74c3c` red | INGRESS ROUTING |

---

## 9. Route Arrow Placement

A single directional arrow is placed on the **longest segment** of each route at the **60% position** along that segment:

```python
def _add_route_arrow(ax, gdf_route, color, zorder):
    # 1. Find longest LineString segment across all geometries
    best_len, best_seg = -1, None
    for geom in gdf_route.geometry:
        parts = geom.geoms if MultiLineString else [geom]
        for p in parts:
            if p.length > best_len:
                best_len = p.length; best_seg = p

    # 2. Place arrow at 60% along that segment's coordinate list
    coords = list(best_seg.coords)
    idx    = int(len(coords) * 0.6)
    ax.annotate("",
        xy=coords[idx+1], xytext=coords[idx],
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=20))
```

One arrow per route keeps the map clean — no repeated arrowheads along the full path.

---

## 10. TO/FROM Label Placement

Station names are placed at the map edge using a two-step process:

### 10.1 Base Position (`_tofrom_pos`)

Projects the station centroid direction from the map centre outward to 88% of `MAP_EXTENT`:

```python
direction = normalise(station_centroid - map_centroid)
base_pt   = map_centroid + direction × MAP_EXTENT × 0.88
           (clamped to ±93% of MAP_EXTENT in x and y)
```

### 10.2 Collision Nudge (`_nudge`)

Rotates the label clockwise in 22° increments until a non-overlapping position is found:

```python
for angle in [0, 22, -22, 44, -44, 66, -66, 90, -90, 112, -112, 135, -135, 158, -158, 180]:
    candidate = rotate base_pt by angle around map_centroid
    if all existing labels are >= MAP_EXTENT × 0.18 away:
        return candidate
```

If no non-overlapping position is found after all angles, the original base position is used.

### 10.3 Label Style

```python
ax.text(x, y, f"TO/FROM\n{name}",
        fontsize=8.5, weight="bold",
        bbox=dict(facecolor="white", edgecolor="black",
                  linewidth=1.5, boxstyle="square,pad=0.4"))
```

---

## 11. MTR Logo Rendering

The MTR logo is loaded once at module import time:

```python
_MTR_LOGO_PATH = "../static/HK_MTR_logo.png"
_mtr_img = mpimg.imread(_MTR_LOGO_PATH)   # None if file not found
```

Placed at each station centroid using `matplotlib.offsetbox`:

```python
def _add_mtr_icon(ax, x, y, size=0.035, zorder=15):
    icon = OffsetImage(_mtr_img, zoom=size)
    ab   = AnnotationBbox(icon, (x, y), frameon=False, zorder=zorder,
                          box_alignment=(0.5, 0.5))
    ax.add_artist(ab)
```

`size=0.035` scales the image to approximately 3.5% of its native pixel size. Silently skipped if the logo file is missing.

---

## 12. Map Layers & Styling

| Layer | Style | Notes |
|-------|-------|-------|
| Basemap | CartoDB PositronNoLabels, alpha=0.55 | No street labels — routing routes provide context |
| Road network | `#8a8a8a`, lw=0.3, alpha=0.35 | OSM drive edges from graph G |
| Drive-time ring fill | `#FFD700` (gold), alpha=0.18 | Outermost ring only (solid fill) |
| Ring boundaries | `#e6b800`, lw=2, dashed `(0,(4,3))` | All 3 rings |
| Ring labels | Black bold, fontsize=10 | Placed at ring right edge + 30 m offset |
| Egress route | `#27ae60` green, lw=3.5, alpha=0.92 | site → station |
| Ingress route | `#e74c3c` red, lw=3.5, alpha=0.92 | station → site |
| Route arrows | `-|>` arrowstyle, lw=2.5, mutation_scale=20 | One per route at 60% of longest segment |
| Station polygon | `#5dade2` fill, `#2e86c1` edge, lw=2, alpha=0.5 | Point stations buffered 55 m |
| MTR logo | PNG, zoom=0.035 | `AnnotationBbox`, frameon=False |
| Station name | Black bold, fontsize=10 | Offset +`MAP_EXTENT × 0.06` above centroid |
| TO/FROM labels | White fill, black border, `square,pad=0.4` | Edge-placed, collision-aware |
| Site footprint | Red fill, alpha=0.65 | No edge colour |
| OUT/IN label | White bold on `#cc0000`, `round,pad=0.3` | Centred on site centroid |
| SITE text | Black bold, fontsize=11 | Below site, offset −`MAP_EXTENT × 0.068` |
| North arrow | Black `-|>`, lw=2 | Upper-right quadrant |
| Legend | White bg, black border, lower-right | Ingress + Egress line styles |

---

## 13. Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Location resolution | O(1) | Single HTTP request |
| Site polygon detection | O(B) per fallback step | B = OSM buildings/features in radius |
| Drive graph build | O(E + V) | E edges, V nodes in `graph_dist` radius |
| Travel time annotation | O(E) | Linear pass over all edges |
| Nearest node snapping | O(V) per call | osmnx brute-force for small graphs |
| Shortest path (Dijkstra) | O((E + V) log V) | Per route; called ×2 per station = ×6 total |
| Route to GDF conversion | O(path length) | Per route |
| Collision nudge | O(S × 16) | S = placed labels so far; 16 angle candidates |
| Map rendering | O(F) | F = total rendered features |

**Typical performance:** 15–40 seconds per site depending on network density and graph_dist. Drive graph build and shortest-path computation are dominant.

---

## 14. Assumptions & Limitations

| Assumption / Limitation | Implication |
|-------------------------|-------------|
| `DRIVE_SPEED = 35 km/h` fixed for all road types | Motorways and residential streets treated identically; ring radii are approximate |
| No traffic or time-of-day modelling | Travel times are theoretical minimums, not real-world estimates |
| Ring radii use straight-line buffers | True drive-time isochrones follow road network; rings are indicative only |
| Top 3 nearest stations by Euclidean distance | Nearest by road network may differ; no isochrone filtering |
| Station snapping uses nearest graph node | Station entrance may not be the nearest node; slight routing inaccuracy near complex interchanges |
| `nx.shortest_path` returns `None` on disconnected graph | Route silently not drawn; no warning to user |
| MTR logo silently omitted if file missing | No fallback icon or placeholder |
| `zone_data` is optional | Site polygon falls through to OSM/buffer chain if omitted |
| `ax.set_xlim/ylim` called twice (before/after basemap) | Required to prevent contextily from overriding viewport; relies on undocumented behaviour |

---

## 15. Extensibility Roadmap

| Priority | Feature | Implementation Path |
|----------|---------|---------------------|
| 1 | True drive-time isochrones | Replace circular rings with network-based isochrone polygons via `ox.distance.shortest_path` from all edge nodes |
| 2 | Variable speed per road type | Map OSM `maxspeed` tag to edge `travel_time` instead of fixed 35 km/h |
| 3 | Traffic-adjusted travel times | Integrate Google Maps / HERE Distance Matrix API for time-of-day speeds |
| 4 | Top-N stations by drive time | Sort stations by actual network travel time rather than Euclidean distance |
| 5 | Walking / cycling mode | Accept `network_type` parameter; adjust `DRIVE_SPEED` accordingly |
| 6 | Multiple destination types | Extend beyond MTR stations to hospitals, schools, bus termini |
| 7 | Route summary table | Add inset table with station name, distance (m), and travel time (min) per route |
| 8 | Async graph fetching | Fetch drive graph and station features concurrently |
| 9 | KD-tree node snapping | Replace osmnx `nearest_nodes` (O(V)) with `scipy.spatial.KDTree` (O(log V)) |
| 10 | Interactive HTML output | Replace static PNG with folium / keplergl interactive map |

---

## 16. Quick Start

### 16.1 Installation

```bash
pip install osmnx geopandas contextily matplotlib networkx shapely numpy
```

### 16.2 Basic Usage

```python
from driving_map import generate_driving

# By lot number, 15-minute drive radius (default)
buf = generate_driving("LOT", "IL 157")

# By address, 10-minute radius
buf = generate_driving("ADDRESS", "1 Austin Road West", max_drive_minutes=10)

# By coordinates, 5-minute radius
buf = generate_driving("COORDINATES", "22.3025,114.1714", max_drive_minutes=5)
```

### 16.3 With OZP Zone Data (improves site polygon accuracy)

```python
import geopandas as gpd

zone_data = gpd.read_file("ozp_zones.geojson")
buf = generate_driving("LOT", "IL 157", zone_data=zone_data, max_drive_minutes=15)
```

### 16.4 Save or Serve the Output

```python
# Save to file
with open("driving_analysis.png", "wb") as f:
    f.write(buf.getvalue())

# Serve via Flask
from flask import send_file
return send_file(buf, mimetype="image/png")

# Display in Jupyter
from IPython.display import Image
Image(data=buf.getvalue())
```

### 16.5 MTR Logo Setup

Place the MTR logo image at:

```
<project_root>/static/HK_MTR_logo.png
```

If the file is absent, the module loads without error and silently omits the logo icon from the map.

---

## 17. Summary

| Component | Technology |
|-----------|------------|
| Location resolution | `modules.resolver` — HK GeoData API |
| Site polygon detection | 5-step fallback chain (lot → OZP → building → landuse → buffer) |
| Drive network | osmnx `graph_from_point()`, `network_type="drive"` |
| Travel-time weighting | Edge length / (35 km/h) in minutes |
| Shortest path routing | NetworkX Dijkstra (`shortest_path`, weight=`travel_time`) |
| Route geometry | `ox.routing.route_to_gdf()` reprojected to EPSG:3857 |
| MTR station acquisition | osmnx `features_from_point(railway=station)` |
| Drive-time rings | Shapely buffer circles on site centroid |
| Directional arrows | `ax.annotate` at 60% of longest route segment |
| TO/FROM label placement | Edge-projection + 22° rotation nudge collision avoidance |
| MTR logo | `matplotlib.offsetbox.AnnotationBbox` with PNG OffsetImage |
| Map rendering | matplotlib + contextily (CartoDB PositronNoLabels) |
| Output | BytesIO PNG buffer (11×11 in, 130 dpi) |

This module provides a self-contained, API-driven driving accessibility analysis tool suitable for integration into planning workflows, transport assessment pipelines, and property analysis platforms in the Hong Kong context.

---

> **Module:** `generate_driving()`
> **File:** `driving_map.py`
> **Output:** PNG BytesIO buffer (130 dpi)
> **Depends on:** `modules.resolver.resolve_location`, `modules.resolver.get_lot_boundary`
> **Optional:** `zone_data` GeoDataFrame · `../static/HK_MTR_logo.png`
