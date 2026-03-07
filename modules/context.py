"""
context.py — Site-Type Driven Context Analysis
ALKF+ Module

Matches Colab reference output:
  - Real site building footprint (red polygon)
  - MTR station yellow polygon
  - Bus stops as clustered dark blue dots
  - Walk route to nearest MTR (dashed blue)
  - Place labels for nearby amenities
  - Type-driven building highlights

Architecture:
  - Single merged OSMnx bbox fetch (like Colab) — avoids empty layer bug
  - ox.settings set once before threads — no race condition
  - Walk graph via OSMnx graph_from_point
  - All rendering matches Colab style exactly
"""

import logging
import os
import gc
import threading
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import contextily as cx
import numpy as np
import textwrap
import pandas as pd
import networkx as nx

from typing import Optional, List
from shapely.geometry import Point
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time as _time

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

# ── OSMnx global settings (set once — never mutated inside threads) ───────────
ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 20
ox.settings.overpass_endpoint = "https://overpass-api.de/api/interpreter"

# ── Constants ─────────────────────────────────────────────────────────────────
FETCH_RADIUS  = 600    # metres — OSM data fetch radius
MAP_HALF_SIZE = 800    # metres — map half-height
MTR_COLOR     = "#ffd166"
ROUTE_COLOR   = "#005eff"
BUS_COLOR     = "#0d47a1"

# ── Static assets ─────────────────────────────────────────────────────────────
_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")

try:
    _mtr_logo        = mpimg.imread(_MTR_LOGO_PATH)
    _MTR_LOGO_LOADED = True
except Exception:
    _mtr_logo        = None
    _MTR_LOGO_LOADED = False

try:
    _bus_img = mpimg.imread(_BUS_ICON_PATH)
except Exception:
    _bus_img = None

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)

# ── Walk graph cache ──────────────────────────────────────────────────────────
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
# SITE-TYPE BUILDING HIGHLIGHT CONFIG
# Controls which building types are highlighted in orange/type-colour
# ============================================================

_HIGHLIGHT = {
    "RESIDENTIAL": {
        "building": ["apartments", "residential", "house",
                     "dormitory", "detached", "terrace", "block"],
        "color":    "#e07b39",
        "label":    "Residential Developments",
    },
    "HOTEL": {
        "tourism":  ["hotel", "hostel", "resort", "guest_house", "aparthotel"],
        "building": ["hotel"],
        "color":    "#b15928",
        "label":    "Hotels & Serviced Apartments",
    },
    "COMMERCIAL": {
        "building": ["office", "commercial", "retail"],
        "office":   True,
        "color":    "#6a3d9a",
        "label":    "Office / Commercial Buildings",
    },
    "INSTITUTIONAL": {
        "amenity":  ["school", "college", "university", "hospital", "government"],
        "building": ["school", "university", "hospital", "government", "civic"],
        "color":    "#1f78b4",
        "label":    "Institutional Buildings",
    },
    "INDUSTRIAL": {
        "building": ["industrial", "warehouse", "factory", "shed"],
        "landuse":  ["industrial"],
        "color":    "#33a02c",
        "label":    "Industrial / Warehouse",
    },
    "OTHER": {
        "building": True,
        "color":    "#888888",
        "label":    "Nearby Buildings",
    },
    "MIXED": {
        "building": True,
        "color":    "#888888",
        "label":    "Nearby Buildings",
    },
}


# ============================================================
# OSM FETCH — single bbox call, all tags merged
# Mirrors the Colab approach: one features_from_point call with all tags.
# Returns raw GeoDataFrame in EPSG:3857.
# ============================================================

_OVERPASS_FALLBACK = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def _fetch_osm(lat: float, lon: float, dist: float,
               tags: dict) -> gpd.GeoDataFrame:
    """
    Fetch OSM features via features_from_bbox.
    Tries 3 Overpass endpoints in sequence.
    ox.settings is set once before the call — no mutation inside.
    """
    bbox = ox.utils_geo.bbox_from_point((lat, lon), dist=dist)

    for ep in _OVERPASS_FALLBACK:
        try:
            ox.settings.overpass_endpoint = ep
            gdf = ox.features_from_bbox(bbox, tags=tags)
            if gdf is not None and not gdf.empty:
                return gdf.to_crs(3857)
        except Exception as e:
            log.debug(f"[context] {ep.split('/')[2]}: {type(e).__name__}")
            continue

    return _EMPTY.copy()


def _safe_get(gdf: gpd.GeoDataFrame, col: str) -> pd.Series:
    """Safe column access — returns None-filled Series if column missing."""
    if col in gdf.columns:
        return gdf[col].fillna("")
    return pd.Series("", index=gdf.index)


def _filter(gdf: gpd.GeoDataFrame, col: str,
            val) -> gpd.GeoDataFrame:
    """Filter GDF rows where column matches value or list of values."""
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    s = gdf[col].fillna("")
    if isinstance(val, list):
        return gdf[s.isin(val)].copy()
    return gdf[s == val].copy()


def _polys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only Polygon / MultiPolygon geometries."""
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])].copy()


# ============================================================
# WALK GRAPH — cached per rounded lat/lon
# ============================================================

def _get_walk_graph(lat: float, lon: float,
                    dist: int = 2000) -> Optional[object]:
    ck = (round(lat, 2), round(lon, 2))
    with _WALK_GRAPH_LOCK:
        if ck in _WALK_GRAPH_CACHE:
            return _WALK_GRAPH_CACHE[ck]

    result = [None]
    def _build():
        try:
            result[0] = ox.graph_from_point(
                (lat, lon), dist=dist, network_type="walk", simplify=True)
        except Exception as e:
            log.warning(f"[context] walk graph: {e}")

    t = threading.Thread(target=_build, daemon=True)
    t.start(); t.join(timeout=25)

    G = result[0]
    if G and G.number_of_nodes() > 0:
        with _WALK_GRAPH_LOCK:
            _WALK_GRAPH_CACHE[ck] = G
        log.info(f"[context] walk graph: {G.number_of_nodes()}n")
        return G
    return None


# ============================================================
# HELPERS
# ============================================================

def _wrap(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(text), width))


def _ascii(text) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    ratio = sum(1 for c in s if ord(c) < 128) / max(len(s), 1)
    if ratio < 0.5:
        return None
    return "".join(c for c in s if ord(c) < 128).strip() or None


def _get_name(gdf: gpd.GeoDataFrame) -> pd.Series:
    en = _safe_get(gdf, "name:en")
    base = _safe_get(gdf, "name")
    return en.where(en != "", base)


def _safe_plot(gdf, ax, **kw):
    try:
        if not gdf.empty:
            gdf.plot(ax=ax, **kw)
    except Exception as e:
        log.debug(f"[context] plot: {e}")


def _draw_mtr_icon(ax, x, y, zoom=0.035, zorder=14):
    if _MTR_LOGO_LOADED and _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ax.add_artist(AnnotationBbox(
            icon, (x, y), frameon=False,
            zorder=zorder, box_alignment=(0.5, 0.5)))
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=12,
                markeredgecolor="white", markeredgewidth=2, zorder=zorder)


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

    lon, lat  = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r   = radius_m if radius_m is not None else FETCH_RADIUS
    half_size = radius_m if radius_m is not None else MAP_HALF_SIZE
    # 4:3 aspect — matches Colab figsize(12,12) with 900m half-size
    half_x    = half_size * (992 / 737)
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
        site_point = site_geom.centroid
        log.info(f"[context] lot boundary area={site_geom.area:.0f}m²")
    else:
        site_point = gpd.GeoSeries(
            [Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
        site_geom  = site_point.buffer(15)
        log.info("[context] fallback point")

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp  = ZONE_DATA.to_crs(3857)
    hits = ozp[ozp.contains(site_point)]
    if hits.empty:
        raise ValueError("No zoning polygon found for this site.")
    zone_row  = hits.iloc[0]
    zone      = zone_row["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    hi_cfg    = _HIGHLIGHT.get(SITE_TYPE, _HIGHLIGHT["MIXED"])
    hi_color  = hi_cfg["color"]
    hi_label  = hi_cfg["label"]
    log.info(f"[context] zone={zone} SITE_TYPE={SITE_TYPE}")

    # ── Parallel fetch: OSM polygons + MTR + bus stops + walk graph ──────────
    # All 4 run concurrently — total time = slowest single fetch (~20s)
    # instead of sequential ~60s which exceeds Render health check window
    from concurrent.futures import ThreadPoolExecutor, as_completed

    log.info("[context] Fetching OSM data + MTR + bus + walk graph (parallel)...")
    t_fetch = _time.monotonic()

    _poly_result  = [_EMPTY.copy()]
    _stn_result   = [_EMPTY.copy()]
    _bus_result   = [_EMPTY.copy()]
    _graph_result = [None]

    def _fetch_polygons():
        _poly_result[0] = _fetch_osm(lat, lon, max(fetch_r, 1200), {
            "landuse":  True,
            "leisure":  True,
            "amenity":  True,
            "building": True,
            "tourism":  True,
            "office":   True,
        })
        log.info(f"[context] polygons: {len(_poly_result[0])} rows "
                 f"in {_time.monotonic()-t_fetch:.1f}s")

    def _fetch_stations():
        _stn_result[0] = _fetch_osm(lat, lon, 2000, {"railway": "station"})
        log.info(f"[context] stations: {len(_stn_result[0])} rows")

    def _fetch_bus():
        _bus_result[0] = _fetch_osm(lat, lon, 900, {"highway": "bus_stop"})
        log.info(f"[context] bus raw: {len(_bus_result[0])} rows")

    def _fetch_graph():
        _graph_result[0] = _get_walk_graph(lat, lon, dist=2000)
        log.info(f"[context] walk graph: "
                 f"{'ready' if _graph_result[0] else 'unavailable'}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [
            pool.submit(_fetch_polygons),
            pool.submit(_fetch_stations),
            pool.submit(_fetch_bus),
            pool.submit(_fetch_graph),
        ]
        # Wait up to 35s total — gives each thread its own 20s Overpass timeout
        for f in as_completed(futs, timeout=35):
            try:
                f.result()
            except Exception as e:
                log.warning(f"[context] parallel fetch error: {e}")

    log.info(f"[context] all fetches done in {_time.monotonic()-t_fetch:.1f}s")

    polygons = _poly_result[0]

    # ── Extract base layers ───────────────────────────────────────────────────
    residential_area = _filter(polygons, "landuse", "residential")
    industrial_area  = _filter(polygons, "landuse",
                                ["industrial", "commercial"])
    parks            = _filter(polygons, "leisure",
                                ["park", "garden", "playground",
                                 "recreation_ground"])
    schools          = _filter(polygons, "amenity",
                                ["school", "college", "university",
                                 "kindergarten"])
    hospitals        = _filter(polygons, "amenity", ["hospital", "clinic"])

    # ── Type-driven highlight buildings ───────────────────────────────────────
    hi_buildings = _EMPTY.copy()
    if not polygons.empty:
        mask = pd.Series(False, index=polygons.index)

        if "building" in hi_cfg:
            val = hi_cfg["building"]
            if val is True:
                b_col = _safe_get(polygons, "building")
                mask |= b_col.str.len() > 0
            else:
                mask |= _safe_get(polygons, "building").isin(val)

        if "tourism" in hi_cfg:
            mask |= _safe_get(polygons, "tourism").isin(hi_cfg["tourism"])

        if "office" in hi_cfg and hi_cfg["office"] is True:
            mask |= _safe_get(polygons, "office").str.len() > 0

        if "amenity" in hi_cfg:
            mask |= _safe_get(polygons, "amenity").isin(hi_cfg["amenity"])

        hi_raw = _polys(polygons[mask].copy())

        if not hi_raw.empty:
            names        = _get_name(hi_raw)
            named        = names.apply(lambda x: bool(_ascii(str(x))))
            large        = hi_raw.geometry.area >= 600
            hi_buildings = hi_raw[named | large].copy()

        log.info(f"[context] highlight buildings: {len(hi_buildings)}")

    # ── Upgrade site polygon to nearest OSM building footprint ───────────────
    site_render_geom = site_geom
    if not polygons.empty:
        try:
            nearby = _polys(polygons[
                polygons.geometry.distance(site_point) < 40
            ])
            if not nearby.empty:
                best = nearby.assign(_a=nearby.area).sort_values(
                    "_a", ascending=False).geometry.iloc[0]
                site_render_geom = best
                log.info(f"[context] site OSM footprint: {best.area:.0f}m²")
            elif site_geom.area < 200:
                site_render_geom = site_point.buffer(20)
        except Exception as e:
            log.debug(f"[context] site footprint: {e}")
    elif site_geom.area < 200:
        site_render_geom = site_point.buffer(20)

    site_gdf = gpd.GeoDataFrame(geometry=[site_render_geom], crs=3857)

    del polygons
    gc.collect()

    # ── Process MTR stations (from parallel result) ───────────────────────────
    stations_gdf  = _stn_result[0]
    stations_plot = _EMPTY.copy()
    nearest_stn   = None
    nearest_name  = None

    if not stations_gdf.empty:
        s          = stations_gdf.copy()
        s["_name"] = _get_name(s)
        s["_cx"]   = s.geometry.centroid.x
        s["_cy"]   = s.geometry.centroid.y
        s["_dist"] = s.apply(
            lambda r: Point(r["_cx"], r["_cy"]).distance(site_point), axis=1)
        s = s.dropna(subset=["_name"]).sort_values("_dist")
        stations_plot = s[s["_dist"] <= half_size * 1.5].head(3).copy()

        if not s.empty:
            best         = s.iloc[0]
            nearest_name = _ascii(str(best["_name"]))
            nearest_stn  = gpd.GeoSeries(
                [Point(best["_cx"], best["_cy"])], crs=3857
            ).to_crs(4326).iloc[0]
            log.info(f"[context] nearest MTR: {nearest_name} "
                     f"({best['_dist']:.0f}m)")

    # ── Process bus stops (from parallel result) ──────────────────────────────
    bus_stops = _EMPTY.copy()
    bus_raw   = _bus_result[0]

    if not bus_raw.empty:
        bs = bus_raw.copy()
        bs["geometry"] = bs.geometry.apply(
            lambda g: g.centroid if g.geom_type != "Point" else g)
        bus_stops = gpd.GeoDataFrame(bs, crs=3857)

        if len(bus_stops) > 6:
            try:
                from sklearn.cluster import KMeans
                coords = np.array([[g.x, g.y] for g in bus_stops.geometry])
                bus_stops["_cl"] = KMeans(
                    n_clusters=6, random_state=0
                ).fit(coords).labels_
                bus_stops = gpd.GeoDataFrame(
                    bus_stops.groupby("_cl").first(), crs=3857
                ).reset_index(drop=True)
            except Exception:
                bus_stops = bus_stops.head(6)

        log.info(f"[context] bus stops: {len(bus_stops)}")

    # ── Walk route (uses graph from parallel result) ──────────────────────────
    walk_routes = []
    G           = _graph_result[0]

    if show_walk_route and nearest_stn is not None and G is not None:
        log.info(f"[context] Walk route → {nearest_name}...")
        try:
            site_node = ox.distance.nearest_nodes(G, lon, lat)
            stn_node  = ox.distance.nearest_nodes(
                G, nearest_stn.x, nearest_stn.y)
            if site_node != stn_node:
                path      = nx.shortest_path(
                    G, site_node, stn_node, weight="length")
                route_gdf = ox.routing.route_to_gdf(G, path).to_crs(3857)
                walk_routes.append(route_gdf)
                log.info(f"[context] walk route: {len(route_gdf)} segments")
        except Exception as e:
            log.warning(f"[context] walk route: {e}")

    # ── Place labels ──────────────────────────────────────────────────────────
    log.info("[context] Fetching labels...")
    labels_rules = {
        "amenity": ["school", "college", "university", "hospital",
                    "clinic", "supermarket", "library", "restaurant"],
        "leisure": ["park", "garden"],
        "place":   ["neighbourhood", "suburb"],
        "tourism": ["attraction"],
    }
    labels_gdf = _fetch_osm(lat, lon, min(fetch_r, 800), labels_rules)

    label_items = []
    seen_text   = set()

    def _harvest(gdf):
        if gdf is None or gdf.empty:
            return
        g        = gdf.copy()
        g["_lb"] = _get_name(g)
        for _, row in g.iterrows():
            text = _ascii(str(row.get("_lb", "") or "").strip())
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            geom = row.geometry
            try:
                p = (geom.representative_point()
                     if hasattr(geom, "representative_point")
                     else geom.centroid)
                label_items.append((p.distance(site_point), geom, text))
            except Exception:
                continue

    _harvest(labels_gdf)
    _harvest(schools)
    _harvest(hospitals)
    _harvest(parks)

    label_items.sort(key=lambda x: x[0])
    label_items = [(g, t) for _, g, t in label_items[:28]]
    log.info(f"[context] labels: {len(label_items)}")

    del labels_gdf
    gc.collect()

    log.info(f"[context] Rendering — "
             f"hi={len(hi_buildings)} bus={len(bus_stops)} "
             f"route={'yes' if walk_routes else 'no'}")

    # ============================================================
    # RENDER — matches Colab style exactly
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 10.5))

    xmin = site_point.x - half_x;  xmax = site_point.x + half_x
    ymin = site_point.y - half_y;  ymax = site_point.y + half_y

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.autoscale(False)

    # Basemap
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=15, alpha=0.95)
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                           zoom=15, alpha=0.95)
        except Exception as e:
            log.warning(f"[context] basemap: {e}")

    # Re-lock extents after basemap
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.autoscale(False)

    # zorder 1: Landuse fills
    _safe_plot(residential_area, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(industrial_area,  ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks,            ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools,          ax, color="#9ecae1", alpha=0.90, zorder=2)
    _safe_plot(hospitals,        ax, color="#aec6cf", alpha=0.90, zorder=2)
    del residential_area, industrial_area, parks, schools, hospitals
    gc.collect()

    # zorder 3: Type-driven highlight buildings
    _safe_plot(hi_buildings, ax, color=hi_color, alpha=0.82,
               zorder=3, edgecolor="white", linewidth=0.3)
    del hi_buildings
    gc.collect()

    # zorder 5: Walk routes (dashed blue — Colab style)
    legend_route = None
    for route_gdf in walk_routes:
        try:
            route_gdf.plot(ax=ax, color=ROUTE_COLOR, linewidth=2.2,
                           linestyle="--", zorder=5, alpha=0.92)
            legend_route = mlines.Line2D(
                [], [], color=ROUTE_COLOR, linewidth=2.2,
                linestyle="--", label="Pedestrian Route to MTR")
        except Exception as e:
            log.debug(f"[context] route plot: {e}")

    # zorder 9: Bus stops — dark blue dots (matches Colab exactly)
    if not bus_stops.empty:
        try:
            bus_stops.plot(ax=ax, color=BUS_COLOR,
                           markersize=35, zorder=9, marker="o")
        except Exception as e:
            log.debug(f"[context] bus plot: {e}")
    del bus_stops
    gc.collect()

    # zorder 10–14: MTR stations
    if not stations_plot.empty:
        try:
            # Yellow polygon fill (Colab style)
            _safe_plot(_polys(stations_plot), ax,
                       facecolor=MTR_COLOR, edgecolor="none",
                       alpha=0.9, zorder=10)
            # MTR logo icon on centroid
            for _, st in stations_plot.iterrows():
                _draw_mtr_icon(ax, st["_cx"], st["_cy"],
                               zoom=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # zorder 14–16: Site polygon
    try:
        site_gdf.plot(ax=ax, facecolor="none", edgecolor="white",
                      linewidth=6.0, zorder=14)
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="#b71c1c",
                      linewidth=2.5, zorder=15)
        sc = site_render_geom.centroid
        ax.text(sc.x, sc.y, "SITE", color="white", weight="bold",
                fontsize=9, ha="center", va="center", zorder=16)
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    # zorder 12: Place labels
    placed  = []
    offsets = [(0,40),(0,-40),(40,0),(-40,0),(28,28),(-28,28),
               (28,-28),(-28,-28)]

    for i, (geom, text) in enumerate(label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point")
                 else geom.centroid)
            if p.distance(site_point) < 100:
                continue
            if any(p.distance(pp) < 120 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x+dx, p.y+dy, _wrap(text, 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=12, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # zorder 17: MTR station name labels (Colab style — white box)
    if not stations_plot.empty:
        try:
            for _, st in stations_plot.iterrows():
                label = _ascii(str(st.get("_name", "") or ""))
                if not label:
                    continue
                ax.text(st["_cx"], st["_cy"] + 130, _wrap(label, 18),
                        fontsize=9, weight="bold", color="black",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.8, pad=1.0),
                        zorder=17)
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    # zorder 20: Info box
    info = (f"{data_type}: {value}\n"
            f"OZP Plan: {zone_row['PLAN_NO']}\n"
            f"Zoning: {zone}\n"
            f"Site Type: {SITE_TYPE}")
    if nearest_name:
        info += f"\nNearest MTR: {nearest_name}"

    ax.text(0.012, 0.988, info, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6),
            zorder=20)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#f2c6a0", label="Residential Area"),
        mpatches.Patch(color="#b39ddb", label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9", label="Public Park"),
        mpatches.Patch(color="#9ecae1", label="School / Institution"),
        mpatches.Patch(color="#aec6cf", label="Hospital / Clinic"),
        mpatches.Patch(color=hi_color, alpha=0.82, label=hi_label),
        mpatches.Patch(color=MTR_COLOR, label="MTR Station"),
        mpatches.Patch(color="#e53935", label="Site"),
        mpatches.Patch(color=BUS_COLOR, label="Bus Stop"),
    ]
    if legend_route is not None:
        legend_handles.append(legend_route)

    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(0.02, 0.02),
              fontsize=8.5, framealpha=0.95)

    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=15, weight="bold")
    ax.set_axis_off()
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    buf = BytesIO()
    plt.tight_layout()
    ax.set_position([0.02, 0.02, 0.96, 0.96])
    plt.savefig(buf, format="png", dpi=100,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)

    log.info(f"[context] Done in {_time.monotonic()-t0:.1f}s")
    return buf


# ============================================================
# STARTUP
# ============================================================

def warm_cache(ZONE_DATA: gpd.GeoDataFrame) -> None:
    """Called from app.py lifespan. No-op in this version."""
    log.info("[context] warm_cache: ready")
