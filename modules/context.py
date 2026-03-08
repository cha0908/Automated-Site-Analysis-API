"""
context.py — Site-Type-Driven Context Analysis Map  v5
=======================================================
Fixes vs v4:
  1. Site polygon always visible — uses lot boundary GDF already in 3857,
     with guaranteed minimum 60m buffer if polygon is too small.
  2. MTR station — icon + name label ONLY, no polygon fill
     (old polygon was the entire station area = huge yellow blob).
  3. Labels — only harvest from landuse / leisure / amenity that match
     the site type. Restaurants/shops filtered out for non-food sites.
  4. Map re-centred on site_point every time.
"""

import gc
import hashlib
import json
import logging
import os
import textwrap
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List, Optional

import contextily as cx
import geopandas as gpd
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from shapely.geometry import Point

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)
ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 30

# ── Overpass endpoints ────────────────────────────────────────────────────────
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# ── Timing ────────────────────────────────────────────────────────────────────
_PER_FETCH_TIMEOUT  = 25
_FETCH_WALL_TIMEOUT = 30

# ── Map parameters ────────────────────────────────────────────────────────────
FETCH_RADIUS  = 700
MAP_HALF_SIZE = 700
MTR_COLOR     = "#ffd166"

# ── Static assets ─────────────────────────────────────────────────────────────
_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")

try:
    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
    log.info("[context] bus icon loaded")
except Exception:
    _bus_icon = None

try:
    _mtr_logo = mpimg.imread(_MTR_LOGO_PATH)
    log.info("[context] MTR logo loaded")
except Exception:
    _mtr_logo = None
    log.warning("[context] MTR logo not found")

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)

# ── Feature cache ─────────────────────────────────────────────────────────────
_FEATURE_CACHE      : dict = {}
_FEATURE_CACHE_LOCK        = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# OZP-based site type  (YOUR dataset only)
# ─────────────────────────────────────────────────────────────────────────────

def _infer_site_type(zone_label: str) -> str:
    z = str(zone_label).upper().strip()
    if z.startswith("R"):                      return "RESIDENTIAL"
    if z.startswith("C"):                      return "COMMERCIAL"
    if z.startswith("G"):                      return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"): return "HOTEL"
    if z.startswith("I"):                      return "INDUSTRIAL"
    if z.startswith("OU"):                     return "OTHER"
    return "MIXED"


# ─────────────────────────────────────────────────────────────────────────────
# Site-type config
# ─────────────────────────────────────────────────────────────────────────────

SITE_CONFIGS = {
    "RESIDENTIAL": {
        "primary_tags": {"building": ["apartments", "residential", "house",
                                       "dormitory", "detached", "terrace"]},
        "support_tags": {"amenity": ["school", "college", "university",
                                      "kindergarten", "hospital", "clinic",
                                      "supermarket", "pharmacy"],
                          "leisure": ["park", "playground", "garden"]},
        "highlight_color": "#e07b39",
        "highlight_label": "Residential Developments",
        # Labels: only these amenity/leisure values shown as text
        "label_amenity": ["school", "college", "university", "hospital",
                           "supermarket", "library", "community_centre"],
        "label_leisure": ["park", "garden", "playground"],
    },
    "HOTEL": {
        "primary_tags": {"tourism": ["hotel", "hostel", "guest_house",
                                      "resort", "aparthotel"],
                          "building": ["hotel"]},
        "support_tags": {"tourism": ["attraction", "museum", "gallery"],
                          "amenity": ["restaurant", "cafe", "theatre", "cinema"],
                          "shop":    ["mall", "department_store"]},
        "highlight_color": "#b15928",
        "highlight_label": "Hotels & Serviced Apartments",
        "label_amenity": ["theatre", "cinema", "museum"],
        "label_leisure": ["park"],
    },
    "COMMERCIAL": {
        "primary_tags": {"building": ["office", "commercial", "retail"],
                          "office":   ["company", "government", "ngo", "yes"]},
        "support_tags": {"amenity": ["bank", "restaurant", "cafe", "fast_food"],
                          "shop":    ["mall", "department_store"]},
        "highlight_color": "#6a3d9a",
        "highlight_label": "Office / Commercial Buildings",
        "label_amenity": ["bank", "government", "library", "post_office"],
        "label_leisure": ["park"],
    },
    "INSTITUTIONAL": {
        "primary_tags": {"amenity":  ["school", "college", "university",
                                       "hospital", "government", "library"],
                          "building": ["government", "civic", "hospital",
                                       "school", "university"]},
        "support_tags": {"leisure": ["park", "garden"],
                          "amenity": ["library", "community_centre"]},
        "highlight_color": "#1f78b4",
        "highlight_label": "Institutional Buildings",
        "label_amenity": ["school", "college", "university", "hospital",
                           "library", "government"],
        "label_leisure": ["park", "garden"],
    },
    "INDUSTRIAL": {
        "primary_tags": {"building": ["industrial", "warehouse", "factory"],
                          "landuse":  ["industrial"]},
        "support_tags": {"landuse":  ["port"]},
        "highlight_color": "#33a02c",
        "highlight_label": "Industrial / Warehouse Buildings",
        "label_amenity": [],
        "label_leisure": [],
    },
    "OTHER": {
        "primary_tags": {"building": ["yes", "commercial", "residential",
                                       "office", "retail", "apartments"]},
        "support_tags": {"amenity": ["school", "hospital", "bank"]},
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
        "label_amenity": ["school", "hospital", "bank", "library"],
        "label_leisure": ["park"],
    },
    "MIXED": {
        "primary_tags": {"building": ["yes", "commercial", "residential",
                                       "office", "retail", "apartments"]},
        "support_tags": {"amenity": ["school", "hospital", "restaurant"],
                          "leisure": ["park", "playground"]},
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
        "label_amenity": ["school", "hospital", "bank", "library"],
        "label_leisure": ["park", "garden"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Bbox fetch
# ─────────────────────────────────────────────────────────────────────────────

def _tags_hash(tags) -> str:
    return hashlib.md5(json.dumps(tags, sort_keys=True).encode()).hexdigest()[:10]


def _bbox_from_point(lat, lon, dist):
    return ox.utils_geo.bbox_from_point((lat, lon), dist=dist)


def _fetch_bbox(bbox, tags, name="?") -> gpd.GeoDataFrame:
    north, south, east, west = bbox
    ck = (round((north + south) / 2, 2), round((east + west) / 2, 2),
          round(north - south, 4), _tags_hash(tags))

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
                log.warning(f"[context] {name} @ {endpoint.split('/')[2]}: {e}")
        completed[0] = True
        with _FEATURE_CACHE_LOCK:
            _FEATURE_CACHE[ck] = result[0]

    t = threading.Thread(target=_do, daemon=True, name=f"osm-{name}")
    t.start()
    t.join(timeout=_PER_FETCH_TIMEOUT)
    if not completed[0]:
        log.warning(f"[context] hard timeout ({_PER_FETCH_TIMEOUT}s): {name}")
    return result[0]


def _parallel_fetch(bbox_small, bbox_large, cfg) -> dict:
    tasks = {
        "landuse":  (bbox_small, {"landuse": True,
                                   "leisure": ["park", "playground", "garden",
                                               "recreation_ground"]}),
        "primary":  (bbox_small, cfg["primary_tags"]),
        "support":  (bbox_small, cfg["support_tags"]),
        "bus":      (bbox_small, {"highway": "bus_stop"}),
        "stations": (bbox_large, {"railway": "station"}),
    }
    results       = {k: _EMPTY.copy() for k in tasks}
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _col(gdf, col):
    return gdf[col] if col in gdf.columns else pd.Series(
        [None] * len(gdf), index=gdf.index)


def _filter_col(gdf, col, val):
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    mask = gdf[col].isin(val) if isinstance(val, list) else (gdf[col] == val)
    return gdf[mask].copy()


def _polys_only(gdf):
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def _get_name(gdf):
    return _col(gdf, "name:en").fillna(_col(gdf, "name"))


def _ascii_clean(text) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    ratio = sum(1 for c in s if ord(c) < 128) / max(len(s), 1)
    if ratio < 0.5:
        return None
    cleaned = "".join(c for c in s if ord(c) < 128).strip()
    return cleaned or None


def _flatten(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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


def _draw_mtr_icon(ax, x, y, zoom=0.042, zorder=14):
    if _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False,
                            zorder=zorder, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=16,
                markeredgecolor="white", markeredgewidth=2.5, zorder=zorder)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_context(
    data_type : str,
    value     : str,
    ZONE_DATA : gpd.GeoDataFrame,
    radius_m  : Optional[int] = None,
    lon       : float         = None,
    lat       : float         = None,
    lot_ids   : List[str]     = None,
    extents   : List[dict]    = None,
) -> BytesIO:

    t0 = _time.monotonic()
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)

    fetch_r   = radius_m if radius_m is not None else FETCH_RADIUS
    half_size = radius_m if radius_m is not None else MAP_HALF_SIZE
    half_x    = half_size * (16 / 12)
    half_y    = half_size

    log.info(f"[context] lon={lon:.5f} lat={lat:.5f} r={fetch_r}m")

    # ── 1. Site point (always in 3857) ────────────────────────────────────────
    site_point_wgs = Point(lon, lat)
    site_point = gpd.GeoSeries(
        [site_point_wgs], crs=4326).to_crs(3857).iloc[0]

    # ── 2. Site polygon ───────────────────────────────────────────────────────
    site_geom = None
    try:
        lot_gdf = get_lot_boundary(lon, lat, data_type, extents)
        if lot_gdf is not None and not lot_gdf.empty:
            candidate = lot_gdf.geometry.iloc[0]
            # Only use if polygon is meaningful size (>100 m²)
            if candidate.area > 100:
                site_geom = candidate
                log.info(f"[context] lot boundary area={site_geom.area:.0f}m²")
    except Exception as e:
        log.warning(f"[context] lot boundary: {e}")

    if site_geom is None:
        # Guaranteed visible 60m buffer around the resolved point
        site_geom = site_point.buffer(60)
        log.info("[context] using 60m buffer fallback")

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # ── 3. OZP zoning → site type ─────────────────────────────────────────────
    ozp  = ZONE_DATA.to_crs(3857)
    hits = ozp[ozp.contains(site_point)]
    if hits.empty:
        ozp_copy      = ozp.copy()
        ozp_copy["_d"] = ozp_copy.geometry.distance(site_point)
        primary       = ozp_copy.sort_values("_d").iloc[0]
        log.warning("[context] site not inside any zone — using nearest")
    else:
        primary = hits.iloc[0]

    zone      = str(primary.get("ZONE_LABEL", "MIXED"))
    plan_no   = str(primary.get("PLAN_NO",    "N/A"))
    SITE_TYPE = _infer_site_type(zone)          # ← OZP only, never OSM
    cfg       = SITE_CONFIGS.get(SITE_TYPE, SITE_CONFIGS["MIXED"])
    hi_color  = cfg["highlight_color"]
    hi_label  = cfg["highlight_label"]
    log.info(f"[context] OZP zone={zone}  SITE_TYPE={SITE_TYPE}")

    # ── 4. Parallel OSM fetch ─────────────────────────────────────────────────
    bbox_small = _bbox_from_point(lat, lon, fetch_r)
    bbox_large = _bbox_from_point(lat, lon, 1500)
    log.info("[context] Fetching OSM via bbox ...")
    results = _parallel_fetch(bbox_small, bbox_large, cfg)
    gc.collect()

    # ── 5. Derive display layers ──────────────────────────────────────────────
    landuse_raw      = results["landuse"]
    residential_area = _filter_col(landuse_raw, "landuse", "residential")
    industrial_area  = _filter_col(landuse_raw, "landuse",
                                   ["industrial", "commercial"])
    parks = _filter_col(landuse_raw, "leisure",
                        ["park", "garden", "recreation_ground", "playground"])

    support_raw = _flatten(results["support"])
    schools     = _filter_col(support_raw, "amenity",
                              ["school", "college", "university",
                               "kindergarten"])
    hospitals   = _filter_col(support_raw, "amenity", ["hospital", "clinic"])

    # Primary highlighted buildings
    primary_raw = _flatten(results["primary"])
    primary_bld = _polys_only(primary_raw)
    if not primary_bld.empty:
        primary_bld["_name"] = _get_name(primary_bld)
        named = primary_bld["_name"].apply(
            lambda x: bool(_ascii_clean(str(x))) if pd.notna(x) else False)
        large = primary_bld.geometry.area >= 200
        primary_bld = primary_bld[named | large].copy()
    log.info(f"[context] primary buildings: {len(primary_bld)}")

    # Bus stops — clustered to ≤8
    bus_raw   = _flatten(results["bus"])
    bus_stops = (bus_raw[bus_raw.geometry.geom_type == "Point"].copy()
                 if not bus_raw.empty else _EMPTY.copy())
    if len(bus_stops) > 8:
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[g.x, g.y] for g in bus_stops.geometry])
            k      = min(8, len(bus_stops))
            lbls   = KMeans(n_clusters=k, random_state=0,
                            n_init=5).fit(coords).labels_
            bus_stops["_cl"] = lbls
            bus_stops = (gpd.GeoDataFrame(
                bus_stops.groupby("_cl").first(), crs=3857)
                .reset_index(drop=True))
        except Exception:
            bus_stops = bus_stops.head(8)

    # ── 6. MTR stations — centroid only (NO polygon fill) ────────────────────
    stations_raw     = _flatten(results["stations"])
    stations_in_view = []   # list of (cx, cy, name_str)
    nearest_stn_name = None

    if not stations_raw.empty:
        s          = stations_raw.copy()
        s["_name"] = _get_name(s)
        s["_cx"]   = s.geometry.centroid.x
        s["_cy"]   = s.geometry.centroid.y
        s["_dist"] = s.apply(
            lambda r: Point(r["_cx"], r["_cy"]).distance(site_point), axis=1)
        s = s.dropna(subset=["_name"]).sort_values("_dist")
        nearby = s[s["_dist"] <= half_size * 1.8].head(3)

        for _, st in nearby.iterrows():
            name = _ascii_clean(str(st["_name"]))
            if name:
                stations_in_view.append((st["_cx"], st["_cy"], name))

        if stations_in_view:
            nearest_stn_name = stations_in_view[0][2]
            log.info(f"[context] nearest MTR: {nearest_stn_name}")

    # ── 7. Place labels — site-type relevant only ─────────────────────────────
    allowed_amenity = set(cfg.get("label_amenity", []))
    allowed_leisure = set(cfg.get("label_leisure", []))

    label_items = []
    seen_texts  = set()

    def _harvest_relevant(gdf, col, allowed_vals, cap=25):
        if gdf is None or gdf.empty or col not in gdf.columns:
            return
        g   = _flatten(gdf)
        sub = g[g[col].isin(allowed_vals)] if allowed_vals else g
        sub["_lb"] = _get_name(sub)
        sub = sub.dropna(subset=["_lb"])
        sub = sub[sub["_lb"].astype(str).str.strip().str.len() > 0]
        for _, row in sub.head(cap).iterrows():
            text = _ascii_clean(str(row["_lb"]).strip())
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            try:
                geom = row.geometry
                p    = (geom.representative_point()
                        if hasattr(geom, "representative_point")
                        else geom.centroid)
                label_items.append((p.distance(site_point), geom, text))
            except Exception:
                continue

    # Harvest only relevant types for this site
    _harvest_relevant(support_raw, "amenity", allowed_amenity)
    _harvest_relevant(landuse_raw, "leisure", allowed_leisure)
    _harvest_relevant(parks,       "leisure", allowed_leisure)
    _harvest_relevant(schools,     "amenity", allowed_amenity)
    _harvest_relevant(hospitals,   "amenity", allowed_amenity)

    label_items.sort(key=lambda x: x[0])
    label_items = [(g, t) for _, g, t in label_items[:28]]

    del primary_raw, support_raw, landuse_raw, results
    gc.collect()

    log.info(f"[context] Rendering — {len(label_items)} labels, "
             f"{len(stations_in_view)} MTR stations")

    # ── 8. Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    # Centre on site_point — always
    cx_map = site_point.x
    cy_map = site_point.y
    xmin, xmax = cx_map - half_x, cx_map + half_x
    ymin, ymax = cy_map - half_y, cy_map + half_y

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

    # Re-lock after basemap
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    # ── Layer draws ───────────────────────────────────────────────────────────
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

    # Bus stops — icon or square marker
    if not bus_stops.empty:
        try:
            if _bus_icon is not None:
                for _, row in bus_stops.iterrows():
                    g  = row.geometry
                    bx = g.x if g.geom_type == "Point" else g.centroid.x
                    by = g.y if g.geom_type == "Point" else g.centroid.y
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

    # MTR stations — ICON + NAME ONLY, no polygon fill
    for (cx_st, cy_st, stn_name) in stations_in_view:
        try:
            _draw_mtr_icon(ax, cx_st, cy_st, zoom=0.042, zorder=14)
            ax.text(cx_st, cy_st - 110,
                    _wrap(stn_name, 20),
                    fontsize=9.5, weight="bold", color="#1a1a1a",
                    ha="center", va="top",
                    bbox=dict(facecolor="white", edgecolor="#cccccc",
                              linewidth=0.8, alpha=0.95,
                              boxstyle="round,pad=3"),
                    zorder=15)
        except Exception as e:
            log.debug(f"[context] MTR icon: {e}")

    # ── Site polygon — always on top, always visible ───────────────────────────
    try:
        sc = site_geom.centroid
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="white",
                      linewidth=2.5, zorder=16, alpha=0.95)
        # "SITE" label — always placed at centroid
        ax.text(sc.x, sc.y, "SITE",
                color="white", weight="bold", fontsize=9,
                ha="center", va="center", zorder=17,
                bbox=dict(facecolor="#e53935", edgecolor="none",
                          alpha=0.0, pad=1))
    except Exception as e:
        log.warning(f"[context] site render: {e}")
        # Ultimate fallback — plot a red dot
        ax.plot(site_point.x, site_point.y, "o", color="#e53935",
                markersize=18, markeredgecolor="white",
                markeredgewidth=2, zorder=17)
        ax.text(site_point.x, site_point.y + 80, "SITE",
                color="#e53935", weight="bold", fontsize=9,
                ha="center", va="bottom", zorder=18)

    # Place labels
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

    # Info box
    info_lines = [
        f"{data_type}: {value}",
        f"OZP Plan: {plan_no}",
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

    # Legend
    legend_handles = [
        mpatches.Patch(color="#f2c6a0",             label="Residential Area"),
        mpatches.Patch(color="#b39ddb",             label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9",             label="Public Park / Garden"),
        mpatches.Patch(color="#9ecae1",             label="School / Institution"),
        mpatches.Patch(color="#aec6cf",             label="Hospital / Clinic"),
        mpatches.Patch(color=hi_color, alpha=0.82,  label=hi_label),
        mpatches.Patch(color=MTR_COLOR,             label="MTR Station"),
        mpatches.Patch(color="#e53935",             label="Site"),
        mpatches.Patch(color="#0d47a1",             label="Bus Stop"),
    ]
    ax.legend(handles=legend_handles,
              loc="lower left", bbox_to_anchor=(0.02, 0.02),
              fontsize=8.5, framealpha=0.96, edgecolor="black",
              fancybox=False)

    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=15, weight="bold", pad=10)
    ax.set_axis_off()

    # Final re-lock
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    buf.seek(0)

    log.info(f"[context] Done in {_time.monotonic() - t0:.1f}s")
    return buf
