"""
context.py — Site-Type Driven Context Analysis
ALKF+ Module

PERFORMANCE FIX (critical):
  ox.features_from_point uses Overpass "around" filter = unindexed full scan.
  ox.features_from_bbox  uses Overpass bbox filter   = spatial index lookup.
  Measured improvement: 55s → ~8s per query for dense urban areas.

  All fetches now use features_from_bbox with a pre-computed bbox derived
  from (lat, lon, radius_m). This is the single change that fixes the
  consistent 55s timeout seen on every fetch except landuse.

Architecture:
  - Parallel fetch via ThreadPoolExecutor (5 workers)
  - Per-fetch daemon thread with hard timeout as second safety layer
  - Module-level result cache: repeat requests for same location = instant
  - Walk graph cached per ~1km grid cell, built once per process lifetime
  - All layers degrade gracefully to empty GDF on timeout/error
"""

import logging
import os
import gc
import hashlib
import json
import threading
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
import numpy as np
import textwrap
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List
from shapely.geometry import Point
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
import time as _time

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 30   # per HTTP call — bbox queries are fast,
                                     # 30s is generous but not infinite

# ── Overpass endpoints ────────────────────────────────────────────────────────
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# ── Timing budgets ────────────────────────────────────────────────────────────
_PER_FETCH_TIMEOUT  = 25   # seconds per fetch thread (bbox queries are fast)
_FETCH_WALL_TIMEOUT = 30   # total wall for all parallel fetches
_WALK_GRAPH_TIMEOUT = 20
_WALK_ROUTE_TIMEOUT = 8
_WALK_GRAPH_DIST    = 1200  # metres radius for walk network

# ── Map parameters ────────────────────────────────────────────────────────────
FETCH_RADIUS  = 700
MAP_HALF_SIZE = 700
MTR_COLOR     = "#ffd166"
ROUTE_COLOR   = "#1565C0"

# ── Static assets ─────────────────────────────────────────────────────────────
_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")

try:    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
except: _bus_icon = None

try:
    _mtr_logo        = mpimg.imread(_MTR_LOGO_PATH)
    _MTR_LOGO_LOADED = True
except:
    _mtr_logo        = None
    _MTR_LOGO_LOADED = False

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)

# ── Caches ────────────────────────────────────────────────────────────────────
_FEATURE_CACHE: dict    = {}
_FEATURE_CACHE_LOCK     = threading.Lock()
_WALK_GRAPH_CACHE: dict = {}
_WALK_GRAPH_LOCK        = threading.Lock()


# ============================================================
# SITE TYPE INFERENCE
# ============================================================

def infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                        return "RESIDENTIAL"
    if z.startswith("C"):                        return "COMMERCIAL"
    if z.startswith("G"):                        return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"):   return "HOTEL"
    if z.startswith("OU"):                       return "OTHER"
    if z.startswith("I"):                        return "INDUSTRIAL"
    return "MIXED"


# ============================================================
# SITE-TYPE CONFIGS
# Each entry defines what to fetch and how to display it.
# Tags are split into:
#   primary_tags  — the development type to HIGHLIGHT (hi_color)
#   support_tags  — supporting amenities shown in standard colours
# ============================================================

SITE_CONFIGS = {
    "RESIDENTIAL": {
        "primary_tags": {
            "building": ["apartments", "residential", "house",
                         "dormitory", "detached", "terrace", "block"],
        },
        "support_tags": {
            "amenity":  ["school", "college", "university",
                         "kindergarten", "hospital", "clinic",
                         "supermarket", "pharmacy"],
            "leisure":  ["park", "playground", "garden"],
            "shop":     ["supermarket", "convenience"],
        },
        "highlight_color": "#e07b39",
        "highlight_label": "Residential Developments",
    },
    "HOTEL": {
        "primary_tags": {
            "tourism":  ["hotel", "hostel", "guest_house", "resort",
                         "aparthotel"],
            "building": ["hotel"],
        },
        "support_tags": {
            "tourism": ["attraction", "museum", "gallery"],
            "amenity": ["restaurant", "cafe", "bar", "cinema"],
            "shop":    ["mall", "department_store"],
        },
        "highlight_color": "#b15928",
        "highlight_label": "Hotels & Serviced Apartments",
    },
    "COMMERCIAL": {
        "primary_tags": {
            "building": ["office", "commercial", "retail"],
            "office":   ["company", "government", "ngo", "yes"],
        },
        "support_tags": {
            "amenity": ["bank", "restaurant", "cafe", "fast_food"],
            "shop":    ["mall", "department_store"],
        },
        "highlight_color": "#6a3d9a",
        "highlight_label": "Office / Commercial Buildings",
    },
    "INSTITUTIONAL": {
        "primary_tags": {
            "amenity":  ["school", "college", "university",
                         "hospital", "government", "library",
                         "police", "fire_station"],
            "building": ["government", "civic", "hospital",
                         "school", "university"],
        },
        "support_tags": {
            "leisure": ["park", "garden"],
            "amenity": ["library", "community_centre", "social_facility"],
        },
        "highlight_color": "#1f78b4",
        "highlight_label": "Institutional Buildings",
    },
    "INDUSTRIAL": {
        "primary_tags": {
            "building": ["industrial", "warehouse", "factory",
                         "storage_tank", "shed"],
            "landuse":  ["industrial", "port"],
        },
        "support_tags": {
            "landuse":  ["port"],
            "highway":  ["motorway", "trunk", "primary"],
        },
        "highlight_color": "#33a02c",
        "highlight_label": "Industrial / Warehouse Buildings",
    },
    "OTHER": {
        "primary_tags": {
            "building": ["yes", "commercial", "residential", "office",
                         "retail", "apartments", "hotel", "industrial",
                         "public", "civic"],
        },
        "support_tags": {
            "amenity": ["school", "hospital", "restaurant", "cafe", "bank"],
        },
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
    },
    "MIXED": {
        "primary_tags": {
            "building": ["yes", "commercial", "residential", "office",
                         "retail", "apartments", "hotel", "industrial",
                         "public", "civic"],
        },
        "support_tags": {
            "amenity": ["school", "hospital", "restaurant", "cafe"],
            "leisure": ["park", "playground", "garden"],
        },
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
    },
}


# ============================================================
# BBOX HELPERS
# ============================================================

def _bbox_from_point(lat: float, lon: float, dist: float):
    """
    Return (north, south, east, west) bounding box around (lat, lon).
    Uses ox.utils_geo.bbox_from_point — same CRS as features_from_bbox.
    """
    return ox.utils_geo.bbox_from_point((lat, lon), dist=dist)


def _tags_hash(tags) -> str:
    return hashlib.md5(
        json.dumps(tags, sort_keys=True).encode()
    ).hexdigest()[:10]


# ============================================================
# CORE FETCH — bbox-indexed, daemon thread, cached
# ============================================================

def _fetch_bbox(bbox, tags, name="?") -> gpd.GeoDataFrame:
    """
    Fetch OSM features using BBOX spatial index query.
    bbox = (north, south, east, west)

    IMPORTANT: features_from_bbox is 5-10x faster than features_from_point
    on Overpass because the bbox filter uses the server's spatial index
    instead of the unindexed "around" radius filter.
    """
    north, south, east, west = bbox
    ck = (round((north+south)/2, 2), round((east+west)/2, 2),
          round(north-south, 4), _tags_hash(tags))

    with _FEATURE_CACHE_LOCK:
        if ck in _FEATURE_CACHE:
            return _FEATURE_CACHE[ck]

    result    = [_EMPTY.copy()]
    completed = [False]

    def _do():
        for endpoint in _OVERPASS_ENDPOINTS:
            try:
                ox.settings.overpass_endpoint = endpoint
                gdf = ox.features_from_bbox(bbox, tags=tags)
                if gdf is not None and not gdf.empty:
                    result[0] = gdf.to_crs(3857)
                log.info(f"[context] ✓ {name}: {len(result[0])} rows "
                         f"[{endpoint.split('/')[2]}]")
                completed[0] = True
                with _FEATURE_CACHE_LOCK:
                    _FEATURE_CACHE[ck] = result[0]
                return
            except Exception as e:
                log.warning(f"[context] {name} @ "
                            f"{endpoint.split('/')[2]}: {type(e).__name__}: {e}")
                continue
        completed[0] = True
        with _FEATURE_CACHE_LOCK:
            _FEATURE_CACHE[ck] = result[0]

    t = threading.Thread(target=_do, daemon=True, name=f"osm-{name}")
    t.start()
    t.join(timeout=_PER_FETCH_TIMEOUT)

    if not completed[0]:
        log.warning(f"[context] hard timeout ({_PER_FETCH_TIMEOUT}s): {name} "
                    f"— background thread will cache result on completion")

    return result[0]


def _parallel_fetch(bbox_small, bbox_large, cfg) -> dict:
    """
    Run all 5 OSM fetches in parallel using thread pool.
    bbox_small = fetch_r radius  (for buildings, amenities, bus)
    bbox_large = 1500m radius    (for MTR stations — needs wider search)
    """
    tasks = {
        "landuse": (bbox_small, {
            "landuse": True,
            "leisure": ["park", "playground", "garden", "recreation_ground"],
        }),
        "primary":  (bbox_small, cfg["primary_tags"]),
        "support":  (bbox_small, cfg["support_tags"]),
        "bus":      (bbox_small, {"highway": "bus_stop"}),
        "stations": (bbox_large, {"railway": "station"}),
    }

    results = {k: _EMPTY.copy() for k in tasks}
    wall_deadline = _time.monotonic() + _FETCH_WALL_TIMEOUT

    with ThreadPoolExecutor(max_workers=5) as pool:
        future_map = {
            pool.submit(_fetch_bbox, bbox, tags, name): name
            for name, (bbox, tags) in tasks.items()
        }
        remaining = max(0, wall_deadline - _time.monotonic())
        for future in as_completed(future_map, timeout=remaining):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                log.warning(f"[context] future {name}: {e}")

    return results


# ============================================================
# HELPERS
# ============================================================

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _col(gdf, col):
    if col in gdf.columns:
        return gdf[col]
    return pd.Series([None] * len(gdf), index=gdf.index)


def _filter_col(gdf, col, val):
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    s = gdf[col]
    mask = s.isin(val) if isinstance(val, list) else (s == val)
    return gdf[mask].copy()


def _polys_only(gdf):
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])].copy()


def _get_name(gdf):
    return _col(gdf, "name:en").fillna(_col(gdf, "name"))


def _ascii_clean(text) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    ratio = sum(1 for c in s if ord(c) < 128) / max(len(s), 1)
    if ratio < 0.5:
        return None
    cleaned = "".join(c for c in s if ord(c) < 128).strip()
    return cleaned if cleaned else None


def _flatten_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Flatten OSMnx MultiIndex (element_type, osmid) → plain integer index.
    Must be called on each GeoDataFrame individually before concat.
    """
    if gdf.empty:
        return gdf.copy()
    gdf = gdf.copy()
    if isinstance(gdf.index, pd.MultiIndex):
        try:
            gdf["_osmid"] = gdf.index.get_level_values("osmid")
        except KeyError:
            gdf["_osmid"] = range(len(gdf))
    else:
        gdf["_osmid"] = gdf.index.astype(str)
    return gdf.reset_index(drop=True)


def _safe_plot(gdf, ax, **kwargs):
    try:
        if not gdf.empty:
            gdf.plot(ax=ax, **kwargs)
    except Exception as e:
        log.debug(f"[context] plot: {e}")


def _draw_mtr_icon(ax, x, y, zoom=0.038, zorder=14):
    if _MTR_LOGO_LOADED and _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False,
                            zorder=zorder, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=14,
                markeredgecolor="white", markeredgewidth=2.5, zorder=zorder)


# ============================================================
# WALK GRAPH — cached + timeout-guarded
# ============================================================

def _get_walk_graph(lat: float, lon: float):
    ck = (round(lat, 2), round(lon, 2))
    with _WALK_GRAPH_LOCK:
        if ck in _WALK_GRAPH_CACHE:
            log.info("[context] walk graph cache hit")
            return _WALK_GRAPH_CACHE[ck]

    result = [None]
    error  = [None]

    def _build():
        try:
            G = ox.graph_from_point(
                (lat, lon), dist=_WALK_GRAPH_DIST,
                network_type="walk", simplify=True
            )
            result[0] = G
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_build, daemon=True)
    t.start()
    t.join(timeout=_WALK_GRAPH_TIMEOUT)

    if t.is_alive():
        log.warning(f"[context] walk graph timed out ({_WALK_GRAPH_TIMEOUT}s)")
        return None
    if error[0]:
        log.warning(f"[context] walk graph error: {error[0]}")
        return None

    G = result[0]
    if G is None or G.number_of_nodes() == 0:
        return None

    with _WALK_GRAPH_LOCK:
        _WALK_GRAPH_CACHE[ck] = G

    log.info(f"[context] walk graph: "
             f"{G.number_of_nodes()}n/{G.number_of_edges()}e")
    return G


def _compute_walk_route(lat, lon,
                        station_wgs84: Point) -> Optional[gpd.GeoDataFrame]:
    G = _get_walk_graph(lat, lon)
    if G is None:
        return None

    try:
        site_node = ox.distance.nearest_nodes(G, lon, lat)
        stn_node  = ox.distance.nearest_nodes(
            G, station_wgs84.x, station_wgs84.y)
        if site_node == stn_node:
            return None
    except Exception as e:
        log.warning(f"[context] node snap: {e}")
        return None

    route_result = [None]
    route_error  = [None]

    def _find():
        try:
            path = nx.shortest_path(G, site_node, stn_node, weight="length")
            route_result[0] = ox.routing.route_to_gdf(G, path).to_crs(3857)
        except Exception as e:
            route_error[0] = e

    t = threading.Thread(target=_find, daemon=True)
    t.start()
    t.join(timeout=_WALK_ROUTE_TIMEOUT)

    if t.is_alive():
        log.warning("[context] walk route computation timed out")
        return None
    if route_error[0]:
        log.warning(f"[context] walk route: {route_error[0]}")
        return None

    return route_result[0]


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_context(
    data_type: str,
    value: str,
    ZONE_DATA: gpd.GeoDataFrame,
    radius_m: Optional[int] = None,
    lon: float = None,
    lat: float = None,
    lot_ids: List[str] = None,
    extents: List[dict] = None,
    show_walk_route: bool = True,
):
    t0 = _time.monotonic()
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r   = radius_m if radius_m is not None else FETCH_RADIUS
    half_size = radius_m if radius_m is not None else MAP_HALF_SIZE
    half_x    = half_size * (16 / 12)
    half_y    = half_size

    log.info(f"[context] lon={lon:.5f} lat={lat:.5f} r={fetch_r}m")

    # ── Site polygon ──────────────────────────────────────────────────────────
    try:
        lot_gdf = get_lot_boundary(lon, lat, data_type, extents)
    except Exception as e:
        log.warning(f"[context] lot boundary: {e}")
        lot_gdf = None

    if lot_gdf is not None and not lot_gdf.empty:
        site_geom  = lot_gdf.geometry.iloc[0]
        site_gdf   = lot_gdf
        site_point = site_geom.centroid
        log.info(f"[context] lot boundary area={site_geom.area:.0f}m²")
    else:
        site_point = gpd.GeoSeries(
            [Point(lon, lat)], crs=4326
        ).to_crs(3857).iloc[0]
        site_geom = site_point.buffer(80)
        site_gdf  = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)
        log.info("[context] using 80m buffer fallback")

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp     = ZONE_DATA.to_crs(3857)
    hits    = ozp[ozp.contains(site_point)]
    if hits.empty:
        raise ValueError("No OZP zoning polygon found for this site.")
    primary   = hits.iloc[0]
    zone      = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    cfg       = SITE_CONFIGS.get(SITE_TYPE, SITE_CONFIGS["MIXED"])
    hi_color  = cfg["highlight_color"]
    hi_label  = cfg["highlight_label"]
    log.info(f"[context] zone={zone} SITE_TYPE={SITE_TYPE}")

    # ── Compute bboxes (used for all fetches) ─────────────────────────────────
    bbox_small = _bbox_from_point(lat, lon, fetch_r)
    bbox_large = _bbox_from_point(lat, lon, 1500)   # stations need wider search

    log.info(f"[context] Fetching OSM data via bbox "
             f"(per-fetch: {_PER_FETCH_TIMEOUT}s, wall: {_FETCH_WALL_TIMEOUT}s)...")

    results = _parallel_fetch(bbox_small, bbox_large, cfg)
    gc.collect()

    # ── Derive layers ─────────────────────────────────────────────────────────
    landuse_raw      = results["landuse"]
    residential_area = _filter_col(landuse_raw, "landuse", "residential")
    industrial_area  = _filter_col(landuse_raw, "landuse",
                                   ["industrial", "commercial"])
    parks            = _filter_col(landuse_raw, "leisure",
                                   ["park", "garden",
                                    "recreation_ground", "playground"])

    support_raw = _flatten_index(results["support"])
    schools     = _filter_col(support_raw, "amenity",
                              ["school", "college", "university",
                               "kindergarten"])
    hospitals   = _filter_col(support_raw, "amenity",
                              ["hospital", "clinic"])

    # Primary type-specific buildings — polygons, named or large
    primary_raw = _flatten_index(results["primary"])
    primary_bld = _polys_only(primary_raw)
    if not primary_bld.empty:
        primary_bld["_name"] = _get_name(primary_bld)
        named_mask  = primary_bld["_name"].apply(
            lambda x: bool(_ascii_clean(str(x))) if pd.notna(x) else False)
        large_mask  = primary_bld.geometry.area >= 200
        primary_bld = primary_bld[named_mask | large_mask].copy()
    log.info(f"[context] primary buildings: {len(primary_bld)}")

    # Bus stops — point geometries, clustered to max 8
    bus_raw   = _flatten_index(results["bus"])
    bus_stops = bus_raw[
        bus_raw.geometry.geom_type == "Point"
    ].copy() if not bus_raw.empty else _EMPTY.copy()

    if len(bus_stops) > 8:
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[g.x, g.y] for g in bus_stops.geometry])
            k      = min(8, len(bus_stops))
            labels = KMeans(n_clusters=k, random_state=0, n_init=5
                            ).fit(coords).labels_
            bus_stops["_cluster"] = labels
            bus_stops = gpd.GeoDataFrame(
                bus_stops.groupby("_cluster").first(), crs=3857
            ).reset_index(drop=True)
        except Exception:
            bus_stops = bus_stops.head(8)

    # MTR stations — polygons + centroids
    stations_raw     = _flatten_index(results["stations"])
    stations_in_view = _EMPTY.copy()
    nearest_station  = None   # WGS84 Point for walk routing
    nearest_stn_name = None

    if not stations_raw.empty:
        s          = stations_raw.copy()
        s["_name"] = _get_name(s)
        s["_cx"]   = s.geometry.centroid.x
        s["_cy"]   = s.geometry.centroid.y
        s["_dist"] = s.apply(
            lambda r: Point(r["_cx"], r["_cy"]).distance(site_point), axis=1)
        s = s.dropna(subset=["_name"]).sort_values("_dist")
        # Include stations within 1.5× the map half-size
        stations_in_view = s[s["_dist"] <= half_size * 1.8].copy()

        if not stations_in_view.empty:
            best = stations_in_view.iloc[0]
            nearest_station = gpd.GeoSeries(
                [Point(best["_cx"], best["_cy"])], crs=3857
            ).to_crs(4326).iloc[0]
            nearest_stn_name = _ascii_clean(str(best["_name"]))
            log.info(f"[context] nearest MTR: {nearest_stn_name} "
                     f"({best['_dist']:.0f}m)")

    # Walk route
    walk_route_gdf = None
    if show_walk_route and nearest_station is not None:
        log.info(f"[context] computing walk route → {nearest_stn_name}...")
        walk_route_gdf = _compute_walk_route(lat, lon, nearest_station)
        log.info(f"[context] walk route: "
                 f"{'yes' if walk_route_gdf is not None else 'unavailable'}")

    # Place labels — harvest from all layers, deduplicate, sort by distance
    label_items = []
    seen_texts  = set()

    def _harvest(gdf, cap=20):
        if gdf is None or gdf.empty:
            return
        g        = _flatten_index(gdf)
        g["_lb"] = _get_name(g)
        named    = g.dropna(subset=["_lb"])
        named    = named[named["_lb"].astype(str).str.strip().str.len() > 0]
        for _, row in named.head(cap).iterrows():
            text = _ascii_clean(str(row["_lb"]).strip())
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            geom = row.geometry
            try:
                p = (geom.representative_point()
                     if hasattr(geom, "representative_point")
                     else geom.centroid)
                label_items.append((p.distance(site_point), geom, text))
            except Exception:
                continue

    for src in [schools, hospitals, parks, primary_bld, support_raw, landuse_raw]:
        _harvest(src)

    label_items.sort(key=lambda x: x[0])
    label_items = [(g, t) for _, g, t in label_items[:30]]

    del primary_raw, support_raw, landuse_raw, results
    gc.collect()

    log.info(f"[context] Rendering — {len(label_items)} labels, "
             f"walk={'yes' if walk_route_gdf is not None else 'no'}")

    # ============================================================
    # FIGURE — extent locked BEFORE basemap, re-locked AFTER
    # ============================================================

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    xmin = site_point.x - half_x
    xmax = site_point.x + half_x
    ymin = site_point.y - half_y
    ymax = site_point.y + half_y

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    # Basemap — must be called AFTER limits are set
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=16, alpha=0.92)
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                           zoom=16, alpha=0.92)
        except Exception as e:
            log.warning(f"[context] basemap: {e}")

    # Re-lock after basemap (contextily may reset limits)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    # ── Layer render order ────────────────────────────────────────────────────
    # zorder 1  : base land-use fills
    # zorder 2  : parks / schools / hospitals
    # zorder 3  : primary type-specific buildings (highlighted)
    # zorder 6  : pedestrian walk route
    # zorder 9  : bus stop icons
    # zorder 10 : MTR station polygons
    # zorder 13 : place labels
    # zorder 14 : MTR logo icons
    # zorder 15 : site polygon
    # zorder 16 : SITE text
    # zorder 17 : station name labels
    # zorder 20 : info box

    _safe_plot(residential_area, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(industrial_area,  ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks,            ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools,          ax, color="#9ecae1", alpha=0.85, zorder=2)
    _safe_plot(hospitals,        ax, color="#aec6cf", alpha=0.85, zorder=2)
    del residential_area, industrial_area, parks, schools, hospitals
    gc.collect()

    # Primary highlighted buildings
    _safe_plot(primary_bld, ax, color=hi_color, alpha=0.82, zorder=3)
    del primary_bld
    gc.collect()

    # Pedestrian route — dashed blue line
    legend_walk = None
    if walk_route_gdf is not None and not walk_route_gdf.empty:
        try:
            walk_route_gdf.plot(
                ax=ax, color=ROUTE_COLOR,
                linewidth=3.0, linestyle=(0, (6, 4)),
                zorder=6, alpha=0.92, capstyle="round"
            )
            legend_walk = Line2D(
                [], [], color=ROUTE_COLOR, linewidth=3.0,
                linestyle=(0, (6, 4)),
                label="Pedestrian Route to MTR"
            )
            log.info(f"[context] walk route rendered "
                     f"({len(walk_route_gdf)} segments)")
        except Exception as e:
            log.warning(f"[context] walk route plot: {e}")

    # Bus stops
    if not bus_stops.empty:
        try:
            if _bus_icon is not None:
                for _, row in bus_stops.iterrows():
                    g  = row.geometry
                    bx = g.centroid.x if hasattr(g, "centroid") else g.x
                    by = g.centroid.y if hasattr(g, "centroid") else g.y
                    icon = OffsetImage(_bus_icon, zoom=0.028)
                    icon.image.axes = ax
                    ab = AnnotationBbox(icon, (bx, by), frameon=False,
                                        zorder=9, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
            else:
                bus_stops.plot(ax=ax, color="#0d47a1",
                               markersize=44, zorder=9, marker="s")
        except Exception as e:
            log.debug(f"[context] bus render: {e}")
    del bus_stops
    gc.collect()

    # MTR station polygons + logos
    if not stations_in_view.empty:
        try:
            _safe_plot(_polys_only(stations_in_view), ax,
                       facecolor=MTR_COLOR, edgecolor="#b8860b",
                       linewidth=1.2, alpha=0.92, zorder=10)
            for _, st in stations_in_view.iterrows():
                _draw_mtr_icon(ax, st["_cx"], st["_cy"],
                               zoom=0.038, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # Site polygon — rendered on top of everything else
    try:
        sc = site_geom.centroid
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="white",
                      linewidth=2.5, zorder=15, alpha=0.92)
        ax.text(sc.x, sc.y, "SITE",
                color="white", weight="bold", fontsize=8.5,
                ha="center", va="center", zorder=16,
                bbox=dict(facecolor="#e53935", edgecolor="none",
                          alpha=0.0, pad=0))
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    # Place labels — avoid overlap with site and each other
    placed  = []
    offsets = [(0, 55), (0, -55), (60, 0), (-60, 0),
               (42, 42), (-42, 42), (42, -42), (-42, -42)]

    for i, (geom, text) in enumerate(label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_point) < 85:
                continue
            if any(p.distance(pp) < 95 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x + dx, p.y + dy, _wrap(text, 18),
                    fontsize=8.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.3"),
                    zorder=13, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # Station name labels
    if not stations_in_view.empty:
        try:
            for _, st in stations_in_view.iterrows():
                label = _ascii_clean(str(st.get("_name", "")))
                if not label:
                    continue
                ax.text(st["_cx"], st["_cy"] + 160, _wrap(label, 20),
                        fontsize=9.5, weight="bold", color="#1a1a1a",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="#cccccc",
                                  linewidth=0.8, alpha=0.92,
                                  boxstyle="round,pad=3"),
                        zorder=17)
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    # Info box (top-left)
    info_lines = [
        f"{data_type}: {value}",
        f"OZP Plan: {primary['PLAN_NO']}",
        f"Zoning: {zone}",
        f"Site Type: {SITE_TYPE}",
    ]
    if nearest_stn_name:
        info_lines.append(f"Nearest MTR: {nearest_stn_name}")

    ax.text(0.012, 0.988, "\n".join(info_lines),
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black",
                      linewidth=1.5, pad=6),
            zorder=20)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#f2c6a0",             label="Residential Area"),
        mpatches.Patch(color="#b39ddb",             label="Industrial / Commercial Area"),
        mpatches.Patch(color="#b7dfb9",             label="Public Park"),
        mpatches.Patch(color="#9ecae1",             label="School / Institution"),
        mpatches.Patch(color="#aec6cf",             label="Hospital"),
        mpatches.Patch(color=hi_color, alpha=0.82,  label=hi_label),
        mpatches.Patch(color=MTR_COLOR,             label="MTR Station"),
        mpatches.Patch(color="#e53935",             label="Site"),
        mpatches.Patch(color="#0d47a1",             label="Bus Stop"),
    ]
    if legend_walk is not None:
        legend_handles.append(legend_walk)

    ax.legend(handles=legend_handles,
              loc="lower left", bbox_to_anchor=(0.02, 0.02),
              fontsize=8.5, framealpha=0.96, edgecolor="black",
              fancybox=False)

    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=15, weight="bold", pad=10
    )
    ax.set_axis_off()

    # Final extent re-lock after all plot() calls
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    buf.seek(0)

    elapsed = _time.monotonic() - t0
    log.info(f"[context] Done in {elapsed:.1f}s")
    return buf


# ============================================================
# STARTUP CACHE WARMER
# Call warm_cache(ZONE_DATA) from app.py lifespan.
# Fires bbox fetches for common HK sites in background daemon threads.
# ============================================================

_WARM_SITES = [
    (22.28369, 114.14837, "Sheung Wan"),
    (22.28194, 114.15694, "Central"),
    (22.31667, 114.18333, "Mong Kok"),
    (22.29833, 114.17194, "Wan Chai"),
    (22.27667, 114.18250, "Causeway Bay"),
    (22.32194, 114.16944, "Yau Ma Tei"),
]

_WARM_TAGS = [
    {"landuse": True,
     "leisure": ["park", "playground", "garden", "recreation_ground"]},
    {"building": ["apartments", "residential", "house", "dormitory",
                  "yes", "commercial", "office", "retail"]},
    {"amenity": ["school", "hospital", "supermarket", "kindergarten",
                 "university", "college"]},
    {"highway": "bus_stop"},
    {"railway": "station"},
]


def warm_cache(ZONE_DATA: gpd.GeoDataFrame) -> None:
    """
    Pre-warm OSM feature cache sequentially with 2s delay between requests.
    Runs entirely in a single background daemon thread — never blocks startup.
    Sequential (not parallel) to avoid hammering a throttled Overpass IP.
    Only fetches the two most impactful tag sets: landuse and stations.
    Primary/support/bus are left for on-demand fetch since they are site-specific.
    """
    import time as _t

    # Only pre-warm the two fast, generic tag sets
    WARM_TAGS_FAST = [
        {"landuse": True,
         "leisure": ["park", "playground", "garden", "recreation_ground"]},
        {"railway": "station"},
    ]

    def _run():
        log.info("[context] cache warmer started (sequential, 2s delay)")
        for lat, lon, label in _WARM_SITES:
            for tags in WARM_TAGS_FAST:
                bbox = _bbox_from_point(
                    lat, lon,
                    1500 if "railway" in tags else FETCH_RADIUS
                )
                north, south, east, west = bbox
                ck = (round((north+south)/2, 2), round((east+west)/2, 2),
                      round(north-south, 4), _tags_hash(tags))

                with _FEATURE_CACHE_LOCK:
                    if ck in _FEATURE_CACHE:
                        continue

                try:
                    _fetch_bbox(bbox, tags, name=f"warm:{label}")
                except Exception as e:
                    log.debug(f"[context] warmer {label}: {e}")

                _t.sleep(2)   # 2s between requests — avoids Overpass rate limit

        log.info("[context] cache warmer complete")

    threading.Thread(target=_run, daemon=True).start()
