"""
context.py — Site-Type Driven Context Analysis
ALKF+ Module

Timeout strategy:
  The Overpass API throttles by queuing requests slowly rather than
  rejecting them. requests_timeout does NOT help — the server accepts
  the connection and then stalls. The only reliable fix is to run each
  OSM fetch in a daemon thread and kill it after a hard wall time.

  Per-fetch hard timeout:  _PER_FETCH_TIMEOUT  (15s default)
  Total parallel wall:     _FETCH_WALL_TIMEOUT (20s default)

  Any fetch that doesn't return within its thread timeout returns an
  empty GeoDataFrame. The map always renders with whatever data arrived.

  Result cache: fetched GDFs cached per (lat2dp, lon2dp, dist, tags_hash)
  so repeat requests for the same location are instant.
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

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache        = True
ox.settings.log_console      = False
# Low per-HTTP-request timeout — if Overpass stalls on a single request
# this fires. But Overpass often accepts connection then queues for minutes,
# so we ALSO use daemon thread timeouts as a second layer.
ox.settings.requests_timeout = 50   # must be < _PER_FETCH_TIMEOUT thread limit


# ── Timing budgets ────────────────────────────────────────────────────────────
_PER_FETCH_TIMEOUT  = 55   # seconds per individual OSM fetch thread
_FETCH_WALL_TIMEOUT = 60   # seconds total wall for all parallel fetches
_WALK_GRAPH_TIMEOUT = 20   # seconds for walk graph build
_WALK_ROUTE_TIMEOUT = 8    # seconds for shortest path

# ── Overpass endpoints ────────────────────────────────────────────────────────
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

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
_FEATURE_CACHE: dict  = {}
_FEATURE_CACHE_LOCK   = threading.Lock()
_WALK_GRAPH_CACHE: dict = {}
_WALK_GRAPH_LOCK        = threading.Lock()


# ============================================================
# SITE TYPE INFERENCE
# ============================================================

def infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                          return "RESIDENTIAL"
    if z.startswith("C"):                          return "COMMERCIAL"
    if z.startswith("G"):                          return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"):     return "HOTEL"
    if z.startswith("OU"):                         return "OTHER"
    if z.startswith("I"):                          return "INDUSTRIAL"
    return "MIXED"


# ============================================================
# SITE-TYPE CONFIGS
# ============================================================

SITE_CONFIGS = {
    "RESIDENTIAL": {
        "primary_tags": {
            "building": ["apartments", "residential", "house",
                         "dormitory", "detached", "terrace"],
        },
        "support_tags": {
            "amenity": ["school", "college", "university",
                        "hospital", "supermarket", "kindergarten"],
            "leisure": ["park", "playground"],
        },
        "highlight_color": "#e07b39",
        "highlight_label": "Residential Developments",
    },
    "HOTEL": {
        "primary_tags": {
            "tourism":  ["hotel", "hostel", "guest_house", "resort"],
            "building": ["hotel"],
        },
        "support_tags": {
            "tourism": ["attraction", "museum"],
            "amenity": ["restaurant", "cafe", "bar"],
        },
        "highlight_color": "#b15928",
        "highlight_label": "Hotels & Serviced Apartments",
    },
    "COMMERCIAL": {
        "primary_tags": {
            "building": ["office", "commercial", "retail"],
            "office":   True,
        },
        "support_tags": {
            "amenity": ["bank", "restaurant", "cafe"],
            "shop":    ["mall", "department_store"],
        },
        "highlight_color": "#6a3d9a",
        "highlight_label": "Office / Commercial Buildings",
    },
    "INSTITUTIONAL": {
        "primary_tags": {
            "amenity":  ["school", "college", "university",
                         "hospital", "government", "library"],
            "building": ["government", "civic", "hospital", "school"],
        },
        "support_tags": {
            "leisure": ["park", "garden"],
            "amenity": ["library", "community_centre"],
        },
        "highlight_color": "#1f78b4",
        "highlight_label": "Institutional Buildings",
    },
    "INDUSTRIAL": {
        "primary_tags": {
            "building": ["industrial", "warehouse", "factory"],
            "landuse":  ["industrial", "port"],
        },
        "support_tags": {
            "highway": ["motorway", "trunk"],
            "landuse": ["port"],
        },
        "highlight_color": "#33a02c",
        "highlight_label": "Industrial / Warehouse Buildings",
    },
    "OTHER":  {
        "primary_tags": {
            "building": ["yes", "commercial", "residential", "office",
                         "retail", "apartments", "hotel", "industrial"],
        },
        "support_tags": {
            "amenity": ["school", "hospital", "restaurant",
                        "cafe", "bank", "library"],
        },
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
    },
    "MIXED":  {
        "primary_tags": {
            "building": ["yes", "commercial", "residential", "office",
                         "retail", "apartments", "hotel", "industrial"],
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
# HELPERS
# ============================================================

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _tags_hash(tags):
    return hashlib.md5(
        json.dumps(tags, sort_keys=True).encode()
    ).hexdigest()[:10]


def _col(gdf, col):
    if col in gdf.columns:
        return gdf[col]
    return pd.Series([None] * len(gdf), index=gdf.index)


def _filter_col(gdf, col, val):
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    s    = gdf[col]
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


def _draw_mtr_icon(ax, x, y, zoom=0.035, zorder=14):
    if _MTR_LOGO_LOADED and _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False,
                            zorder=zorder, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=12,
                markeredgecolor="white", markeredgewidth=2, zorder=zorder)


# ============================================================
# CORE FETCH — daemon thread with hard wall timeout
#
# Why daemon threads instead of requests_timeout:
#   Overpass accepts the TCP connection immediately (so requests_timeout
#   doesn't fire) then stalls the HTTP response for 1-4 minutes while
#   it processes the query queue. The only way to enforce a real deadline
#   is to run the blocking ox.features_from_point call in a daemon thread
#   and abandon it after _PER_FETCH_TIMEOUT seconds.
#
# The thread continues running in the background after we abandon it —
# daemon=True ensures it doesn't prevent process shutdown. When it
# eventually completes, the result is written to _FEATURE_CACHE so
# the NEXT request for the same location benefits from it.
# ============================================================

def _fetch_with_hard_timeout(lat, lon, dist, tags,
                              name="?") -> gpd.GeoDataFrame:
    """
    Fetch OSM features with a hard per-thread timeout.
    Returns cached result instantly if available.
    If fetch completes after timeout, result is cached for next call.
    """
    ck = (round(lat, 2), round(lon, 2), dist, _tags_hash(tags))

    # Cache hit — instant return
    with _FEATURE_CACHE_LOCK:
        if ck in _FEATURE_CACHE:
            log.info(f"[context] cache hit: {name}")
            return _FEATURE_CACHE[ck]

    result    = [_EMPTY.copy()]
    completed = [False]

    def _do_fetch():
        for endpoint in _OVERPASS_ENDPOINTS:
            try:
                ox.settings.overpass_endpoint = endpoint
                gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags)
                if gdf is not None and not gdf.empty:
                    result[0] = gdf.to_crs(3857)
                    log.info(f"[context] ✓ {name}: {len(result[0])} rows "
                             f"[{endpoint.split('/')[2]}]")
                else:
                    log.info(f"[context] ✓ {name}: 0 rows (empty)")
                completed[0] = True
                # Write to cache even if thread was already abandoned
                with _FEATURE_CACHE_LOCK:
                    _FEATURE_CACHE[ck] = result[0]
                return   # success — stop trying endpoints
            except Exception as e:
                log.warning(f"[context] {name} @ "
                            f"{endpoint.split('/')[2]}: {type(e).__name__}")
                continue
        completed[0] = True
        with _FEATURE_CACHE_LOCK:
            _FEATURE_CACHE[ck] = result[0]

    t = threading.Thread(target=_do_fetch, daemon=True, name=f"osm-{name}")
    t.start()
    t.join(timeout=_PER_FETCH_TIMEOUT)

    if not completed[0]:
        log.warning(f"[context] timeout ({_PER_FETCH_TIMEOUT}s): {name} "
                    f"— thread continues in background, result cached on completion")

    return result[0]


def _parallel_fetch(tasks: dict) -> dict:
    """
    Submit all fetches simultaneously, collect results up to
    _FETCH_WALL_TIMEOUT. Each individual fetch also has its own
    _PER_FETCH_TIMEOUT via daemon thread.
    """
    results = {k: _EMPTY.copy() for k in tasks}

    with ThreadPoolExecutor(max_workers=min(len(tasks), 5)) as pool:
        future_map = {
            pool.submit(
                _fetch_with_hard_timeout,
                *args,
                name=key
            ): key
            for key, args in tasks.items()
        }

        # Collect as they complete — stop after wall timeout
        import time
        deadline = time.monotonic() + _FETCH_WALL_TIMEOUT

        for future in as_completed(future_map,
                                   timeout=max(0, deadline - time.monotonic())):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception as e:
                log.warning(f"[context] future {key}: {e}")

    return results


# ============================================================
# WALK GRAPH — cached + timeout-guarded daemon thread
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
                (lat, lon), dist=_WALK_GRAPH_DIST if hasattr(
                    _WALK_GRAPH_DIST, '__int__') else 1000,
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

    log.info(f"[context] walk graph: {G.number_of_nodes()}n / "
             f"{G.number_of_edges()}e")
    return G


_WALK_GRAPH_DIST = 1000   # metres


def _compute_walk_route(lat, lon, station_wgs84: Point):
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
        log.warning("[context] walk route timed out")
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
        log.info("[context] using 80m buffer")

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp     = ZONE_DATA.to_crs(3857)
    primary = ozp[ozp.contains(site_point)]
    if primary.empty:
        raise ValueError("No OZP zoning polygon found for this site.")
    primary   = primary.iloc[0]
    zone      = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    cfg       = SITE_CONFIGS.get(SITE_TYPE, SITE_CONFIGS["MIXED"])
    hi_color  = cfg["highlight_color"]
    hi_label  = cfg["highlight_label"]
    log.info(f"[context] zone={zone} SITE_TYPE={SITE_TYPE}")

    # ── Parallel fetch (all with individual hard timeouts) ────────────────────
    log.info(f"[context] Fetching OSM data "
             f"(per-fetch: {_PER_FETCH_TIMEOUT}s, wall: {_FETCH_WALL_TIMEOUT}s)...")

    results = _parallel_fetch({
        "landuse":  (lat, lon, fetch_r, {
            "landuse": True,
            "leisure": ["park", "playground", "garden", "recreation_ground"],
        }),
        "primary":  (lat, lon, fetch_r, cfg["primary_tags"]),
        "support":  (lat, lon, fetch_r, cfg["support_tags"]),
        "bus":      (lat, lon, fetch_r, {"highway": "bus_stop"}),
        "stations": (lat, lon, 1500,    {"railway": "station"}),
    })
    gc.collect()

    # ── Derive layers ─────────────────────────────────────────────────────────
    landuse_raw      = results["landuse"]
    residential_area = _filter_col(landuse_raw, "landuse", "residential")
    industrial_area  = _filter_col(landuse_raw, "landuse",
                                   ["industrial", "commercial"])
    parks            = _filter_col(landuse_raw, "leisure",
                                   ["park", "garden",
                                    "recreation_ground", "playground"])
    support_raw = results["support"]
    schools     = _filter_col(support_raw, "amenity",
                              ["school", "college", "university", "kindergarten"])
    hospitals   = _filter_col(support_raw, "amenity",
                              ["hospital", "clinic"])

    primary_raw = _flatten_index(results["primary"])
    primary_bld = _polys_only(primary_raw)
    if not primary_bld.empty:
        primary_bld["_name"] = _get_name(primary_bld)
        named_mask  = primary_bld["_name"].apply(
            lambda x: bool(_ascii_clean(str(x))) if pd.notna(x) else False)
        large_mask  = primary_bld.geometry.area >= 300
        primary_bld = primary_bld[named_mask | large_mask].copy()
    log.info(f"[context] primary buildings: {len(primary_bld)}")

    bus_raw   = _flatten_index(results["bus"])
    bus_stops = bus_raw[
        bus_raw.geometry.geom_type == "Point"
    ].copy() if not bus_raw.empty else _EMPTY.copy()

    if len(bus_stops) > 6:
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[g.x, g.y] for g in bus_stops.geometry])
            labels = KMeans(n_clusters=6, random_state=0).fit(coords).labels_
            bus_stops["_cluster"] = labels
            bus_stops = gpd.GeoDataFrame(
                bus_stops.groupby("_cluster").first(), crs=3857
            ).reset_index(drop=True)
        except Exception:
            bus_stops = bus_stops.head(6)

    stations_raw     = _flatten_index(results["stations"])
    stations_in_view = _EMPTY.copy()
    nearest_station  = None
    nearest_stn_name = None

    if not stations_raw.empty:
        s          = stations_raw.copy()
        s["_name"] = _get_name(s)
        s["_cx"]   = s.geometry.centroid.x
        s["_cy"]   = s.geometry.centroid.y
        s["_dist"] = s.apply(
            lambda r: Point(r["_cx"], r["_cy"]).distance(site_point), axis=1)
        s = s.dropna(subset=["_name"]).sort_values("_dist")
        stations_in_view = s[s["_dist"] <= half_size * 1.5].copy()

        if not stations_in_view.empty:
            best = stations_in_view.iloc[0]
            nearest_station = gpd.GeoSeries(
                [Point(best["_cx"], best["_cy"])], crs=3857
            ).to_crs(4326).iloc[0]
            nearest_stn_name = _ascii_clean(str(best["_name"]))

    # Walk route
    walk_route_gdf = None
    if show_walk_route and nearest_station is not None:
        log.info(f"[context] Walk route → {nearest_stn_name}...")
        walk_route_gdf = _compute_walk_route(lat, lon, nearest_station)
        log.info(f"[context] walk route: "
                 f"{'yes' if walk_route_gdf is not None else 'unavailable'}")

    # Labels
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
    # FIGURE
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

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=16, alpha=0.92)
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                           zoom=16, alpha=0.92)
        except Exception as e:
            log.warning(f"[context] basemap: {e}")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    _safe_plot(residential_area, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(industrial_area,  ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks,            ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools,          ax, color="#9ecae1", alpha=0.85, zorder=2)
    _safe_plot(hospitals,        ax, color="#aec6cf", alpha=0.85, zorder=2)
    del residential_area, industrial_area, parks, schools, hospitals
    gc.collect()

    _safe_plot(primary_bld, ax, color=hi_color, alpha=0.82, zorder=3)
    del primary_bld
    gc.collect()

    legend_walk = None
    if walk_route_gdf is not None and not walk_route_gdf.empty:
        try:
            walk_route_gdf.plot(ax=ax, color=ROUTE_COLOR,
                                linewidth=2.8, linestyle=(0, (6, 4)),
                                zorder=6, alpha=0.90)
            legend_walk = Line2D([], [], color=ROUTE_COLOR, linewidth=2.8,
                                 linestyle=(0, (6, 4)),
                                 label="Pedestrian Route to MTR")
        except Exception as e:
            log.warning(f"[context] walk route plot: {e}")

    if not bus_stops.empty:
        try:
            if _bus_icon is not None:
                for _, row in bus_stops.iterrows():
                    g  = row.geometry
                    bx = g.centroid.x if hasattr(g, "centroid") else g.x
                    by = g.centroid.y if hasattr(g, "centroid") else g.y
                    icon = OffsetImage(_bus_icon, zoom=0.025)
                    icon.image.axes = ax
                    ab = AnnotationBbox(icon, (bx, by), frameon=False,
                                        zorder=9, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
            else:
                bus_stops.plot(ax=ax, color="#0d47a1",
                               markersize=40, zorder=9, marker="s")
        except Exception as e:
            log.debug(f"[context] bus render: {e}")
    del bus_stops
    gc.collect()

    if not stations_in_view.empty:
        try:
            _safe_plot(stations_in_view, ax,
                       facecolor=MTR_COLOR, edgecolor="none",
                       alpha=0.90, zorder=10)
            for _, st in stations_in_view.iterrows():
                _draw_mtr_icon(ax, st["_cx"], st["_cy"], zoom=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    try:
        sc = site_geom.centroid
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="white",
                      linewidth=2.5, zorder=15)
        ax.text(sc.x, sc.y, "SITE", color="white", weight="bold",
                fontsize=8, ha="center", va="center", zorder=16)
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    placed  = []
    offsets = [(0, 50), (0, -50), (55, 0), (-55, 0),
               (38, 38), (-38, 38), (38, -38), (-38, -38)]

    for i, (geom, text) in enumerate(label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_point) < 80:
                continue
            if any(p.distance(pp) < 90 for pp in placed):
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

    if not stations_in_view.empty:
        try:
            for _, st in stations_in_view.iterrows():
                label = _ascii_clean(str(st.get("_name", "")))
                if not label:
                    continue
                ax.text(st["_cx"], st["_cy"] + 150, _wrap(label, 18),
                        fontsize=9, weight="bold", color="black",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.88, pad=2.5),
                        zorder=17)
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    info_lines = [
        f"{data_type}: {value}",
        f"OZP Plan: {primary['PLAN_NO']}",
        f"Zoning: {zone}",
        f"Site Type: {SITE_TYPE}",
    ]
    if nearest_stn_name:
        info_lines.append(f"Nearest MTR: {nearest_stn_name}")

    ax.text(0.012, 0.988, "\n".join(info_lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black",
                      linewidth=1.5, pad=6),
            zorder=20)

    legend_handles = [
        mpatches.Patch(color="#f2c6a0",            label="Residential Area"),
        mpatches.Patch(color="#b39ddb",            label="Industrial / Commercial Area"),
        mpatches.Patch(color="#b7dfb9",            label="Public Park"),
        mpatches.Patch(color="#9ecae1",            label="School / Institution"),
        mpatches.Patch(color="#aec6cf",            label="Hospital"),
        mpatches.Patch(color=hi_color, alpha=0.82, label=hi_label),
        mpatches.Patch(color=MTR_COLOR,            label="MTR Station"),
        mpatches.Patch(color="#e53935",            label="Site"),
        mpatches.Patch(color="#0d47a1",            label="Bus Stop"),
    ]
    if legend_walk is not None:
        legend_handles.append(legend_walk)

    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(0.02, 0.02),
              fontsize=8.5, framealpha=0.95, edgecolor="black")

    ax.set_title(f"Automated Site Context Analysis – {data_type} {value}",
                 fontsize=15, weight="bold")
    ax.set_axis_off()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("[context] Done.")
    return buf


# ============================================================
# STARTUP CACHE WARMER
#
# Call warm_cache(ZONE_DATA) from app.py lifespan startup.
# Fetches OSM data for common HK sites in background daemon threads
# so the first real user request hits the module cache instead of
# making cold Overpass calls.
#
# Usage in app.py:
#   from modules.context import warm_cache
#
#   @asynccontextmanager
#   async def lifespan(app: FastAPI):
#       import asyncio, functools
#       loop = asyncio.get_event_loop()
#       loop.run_in_executor(None, functools.partial(warm_cache, ZONE_DATA))
#       yield
#
# ============================================================

# Common HK sites — covers the most frequent analysis requests.
# Each entry: (lat, lon, label)
_WARM_SITES = [
    (22.28369, 114.14837, "Sheung Wan / IL 1657"),
    (22.28194, 114.15694, "Central / IL 157"),
    (22.31667, 114.18333, "Mong Kok"),
    (22.29833, 114.17194, "Wan Chai"),
    (22.27667, 114.18250, "Causeway Bay"),
    (22.32194, 114.16944, "Yau Ma Tei"),
]

_WARM_TAGS = [
    {"landuse": True, "leisure": ["park", "playground", "garden", "recreation_ground"]},
    {"building": ["apartments", "residential", "house", "dormitory"]},
    {"amenity": ["school", "hospital", "supermarket", "kindergarten"]},
    {"highway": "bus_stop"},
    {"railway": "station"},
]


def warm_cache(ZONE_DATA: gpd.GeoDataFrame) -> None:
    """
    Pre-warm the OSM feature cache for common HK sites.
    Runs in a background daemon thread — never blocks startup.
    Safe to call even if ZONE_DATA is None (skips zoning lookup).
    """
    log.info("[context] cache warmer started")

    def _warm_one(lat, lon, label, tags):
        ck = (round(lat, 2), round(lon, 2), FETCH_RADIUS, _tags_hash(tags))
        with _FEATURE_CACHE_LOCK:
            if ck in _FEATURE_CACHE:
                return   # already cached
        try:
            result = [_EMPTY.copy()]
            done   = [False]

            def _fetch():
                for endpoint in _OVERPASS_ENDPOINTS:
                    try:
                        ox.settings.overpass_endpoint = endpoint
                        gdf = ox.features_from_point(
                            (lat, lon), dist=FETCH_RADIUS, tags=tags
                        )
                        if gdf is not None and not gdf.empty:
                            result[0] = gdf.to_crs(3857)
                        done[0] = True
                        with _FEATURE_CACHE_LOCK:
                            _FEATURE_CACHE[ck] = result[0]
                        return
                    except Exception:
                        continue
                done[0] = True

            t = threading.Thread(target=_fetch, daemon=True)
            t.start()
            t.join(timeout=60)
        except Exception as e:
            log.debug(f"[context] warmer {label}: {e}")

    for lat, lon, label in _WARM_SITES:
        for tags in _WARM_TAGS:
            _warm_one(lat, lon, label, tags)

    log.info("[context] cache warmer complete")
