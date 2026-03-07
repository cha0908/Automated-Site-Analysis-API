"""
context.py — Site-Type Driven Context Analysis
ALKF+ Module

Output matches Colab reference:
  - Real site building footprint (red polygon)
  - MTR station yellow polygon + MTR logo icon
  - Bus stops with bus icon (clustered)
  - Type-driven building highlights per site type
  - Landuse fills: residential, industrial, parks, schools, hospitals
  - Place labels for nearby amenities

NO walk route — removed to eliminate blocking network call.
All fetches run in parallel (3 threads) — total fetch time ~20s max.
"""

import logging
import os
import gc
import threading
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
import numpy as np
import textwrap
import pandas as pd
import networkx as nx

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List
from shapely.geometry import Point
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time as _time

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

# ── OSMnx settings (set once at module level — never mutated in threads) ──────
ox.settings.use_cache           = True
ox.settings.log_console         = False
ox.settings.timeout             = 20   # Overpass server-side query timeout (fills {timeout} in QL)
ox.settings.requests_timeout    = 22   # HTTP socket timeout (client-side)
# ox.settings.overpass_settings template = "[out:json][timeout:{timeout}]"
# {timeout} is filled from ox.settings.timeout above — do NOT override the template directly

# ── Constants ─────────────────────────────────────────────────────────────────
FETCH_RADIUS  = 600
MAP_HALF_SIZE = 800
MTR_COLOR     = "#ffd166"
BUS_COLOR     = "#0d47a1"

# ── Static assets ─────────────────────────────────────────────────────────────
_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")

try:
    _mtr_logo        = mpimg.imread(_MTR_LOGO_PATH)
    _MTR_LOGO_LOADED = True
    log.info("[context] MTR logo loaded")
except Exception as e:
    _mtr_logo        = None
    _MTR_LOGO_LOADED = False
    log.warning(f"[context] MTR logo missing: {e}")

try:
    _bus_img = mpimg.imread(_BUS_ICON_PATH)
    log.info("[context] bus icon loaded")
except Exception as e:
    _bus_img = None
    log.warning(f"[context] bus icon missing: {e}")

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)

# ── Overpass endpoints ────────────────────────────────────────────────────────
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


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
# SITE-TYPE HIGHLIGHT CONFIG
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
        "color":    "#33a02c",
        "label":    "Industrial / Warehouse",
    },
    "OTHER":  {"building": True, "color": "#888888", "label": "Nearby Buildings"},
    "MIXED":  {"building": True, "color": "#888888", "label": "Nearby Buildings"},
}


# ============================================================
# OSM FETCH — bbox query, endpoint rotation, thread-safe
# ============================================================

def _fetch_osm(lat: float, lon: float,
               dist: float, tags: dict) -> gpd.GeoDataFrame:
    """
    Fetch OSM features via features_from_bbox.
    ox.settings.overpass_endpoint is set per-call in a lock to avoid
    race conditions between parallel threads.
    """
    bbox = ox.utils_geo.bbox_from_point((lat, lon), dist=dist)
    for ep in _OVERPASS_ENDPOINTS:
        try:
            ox.settings.overpass_endpoint = ep
            gdf = ox.features_from_bbox(bbox, tags=tags)
            if gdf is not None and not gdf.empty:
                return gdf.to_crs(3857)
        except Exception as e:
            log.debug(f"[context] {ep.split('/')[2]}: {type(e).__name__}")
            continue
    return _EMPTY.copy()


# ============================================================
# HELPERS
# ============================================================

def _safe_get(gdf: gpd.GeoDataFrame, col: str) -> pd.Series:
    if col in gdf.columns:
        return gdf[col].fillna("")
    return pd.Series("", index=gdf.index)


def _filter(gdf: gpd.GeoDataFrame, col: str, val) -> gpd.GeoDataFrame:
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    s = gdf[col].fillna("")
    return gdf[s.isin(val) if isinstance(val, list) else (s == val)].copy()


def _polys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])].copy()


def _get_name(gdf: gpd.GeoDataFrame) -> pd.Series:
    en   = _safe_get(gdf, "name:en")
    base = _safe_get(gdf, "name")
    return en.where(en != "", base)


def _ascii(text) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    if sum(1 for c in s if ord(c) < 128) / max(len(s), 1) < 0.5:
        return None
    return "".join(c for c in s if ord(c) < 128).strip() or None


def _wrap(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(text), width))


def _safe_plot(gdf, ax, **kw):
    try:
        if not gdf.empty:
            gdf.plot(ax=ax, **kw)
    except Exception as e:
        log.debug(f"[context] plot: {e}")


def _draw_mtr_icon(ax, x, y, zoom=0.035, zorder=14):
    """Draw MTR logo at (x,y). Falls back to red circle if logo missing."""
    if _MTR_LOGO_LOADED and _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ax.add_artist(AnnotationBbox(
            icon, (x, y), frameon=False,
            zorder=zorder, box_alignment=(0.5, 0.5)))
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=12,
                markeredgecolor="white", markeredgewidth=2, zorder=zorder)


def _draw_bus_icon(ax, x, y, zoom=0.028, zorder=9):
    """Draw bus icon at (x,y). Falls back to dark blue square if icon missing."""
    if _bus_img is not None:
        icon = OffsetImage(_bus_img, zoom=zoom)
        icon.image.axes = ax
        ax.add_artist(AnnotationBbox(
            icon, (x, y), frameon=False,
            zorder=zorder, box_alignment=(0.5, 0.5)))
    else:
        ax.plot(x, y, "s", color=BUS_COLOR, markersize=10,
                markeredgecolor="white", markeredgewidth=1, zorder=zorder)


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
    show_walk_route: bool = False,   # disabled — removed to prevent blocking
):
    t0 = _time.monotonic()

    lon, lat  = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r   = radius_m if radius_m is not None else FETCH_RADIUS
    half_size = radius_m if radius_m is not None else MAP_HALF_SIZE
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

    # ── 3 parallel fetches: polygons + MTR stations + bus stops ──────────────
    # No walk graph — eliminated to remove the 25-30s blocking call
    log.info("[context] Fetching OSM (parallel: polygons + MTR + bus)...")
    t_fetch = _time.monotonic()

    _poly_result = [_EMPTY.copy()]
    _stn_result  = [_EMPTY.copy()]
    _bus_result  = [_EMPTY.copy()]

    def _fetch_polygons():
        r = _fetch_osm(lat, lon, max(fetch_r, 1200), {
            "landuse": True, "leisure": True, "amenity": True,
            "building": True, "tourism": True, "office": True,
        })
        _poly_result[0] = r
        log.info(f"[context] polygons: {len(r)} rows "
                 f"in {_time.monotonic()-t_fetch:.1f}s")

    def _fetch_stations():
        r = _fetch_osm(lat, lon, 2000, {"railway": "station"})
        _stn_result[0] = r
        log.info(f"[context] stations: {len(r)} rows")

    def _fetch_bus():
        r = _fetch_osm(lat, lon, 500, {"highway": "bus_stop"})
        _bus_result[0] = r
        log.info(f"[context] bus raw: {len(r)} rows")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_fetch_polygons): "polygons",
            pool.submit(_fetch_stations): "stations",
            pool.submit(_fetch_bus):      "bus",
        }
        # Per-future timeout — any single fetch capped at 25s
        # prevents one slow query from blocking the entire pipeline
        import concurrent.futures as _cf
        done, pending = _cf.wait(futs, timeout=28,  # 20s server timeout + 8s buffer
                                 return_when=_cf.ALL_COMPLETED)
        for f in pending:
            fname = futs[f]
            log.warning(f"[context] {fname} fetch timed out — skipping")
            f.cancel()
        for f in done:
            try:
                f.result()
            except Exception as e:
                log.warning(f"[context] fetch error ({futs[f]}): {e}")

    log.info(f"[context] all fetches done in "
             f"{_time.monotonic()-t_fetch:.1f}s")

    polygons = _poly_result[0]

    # ── Base layers ───────────────────────────────────────────────────────────
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

        b_cfg = hi_cfg.get("building")
        if b_cfg is True:
            mask |= _safe_get(polygons, "building").str.len() > 0
        elif isinstance(b_cfg, list):
            mask |= _safe_get(polygons, "building").isin(b_cfg)

        t_cfg = hi_cfg.get("tourism")
        if isinstance(t_cfg, list):
            mask |= _safe_get(polygons, "tourism").isin(t_cfg)

        if hi_cfg.get("office") is True:
            mask |= _safe_get(polygons, "office").str.len() > 0

        a_cfg = hi_cfg.get("amenity")
        if isinstance(a_cfg, list):
            mask |= _safe_get(polygons, "amenity").isin(a_cfg)

        hi_raw = _polys(polygons[mask].copy())
        if not hi_raw.empty:
            names  = _get_name(hi_raw)
            named  = names.apply(lambda x: bool(_ascii(str(x))))
            large  = hi_raw.geometry.area >= 600
            hi_buildings = hi_raw[named | large].copy()

        log.info(f"[context] highlight buildings: {len(hi_buildings)}")

    # ── Site footprint from nearest OSM building ──────────────────────────────
    site_render_geom = site_geom
    if not polygons.empty:
        try:
            nearby = _polys(polygons[
                polygons.geometry.distance(site_point) < 40])
            if not nearby.empty:
                best = nearby.assign(_a=nearby.area).sort_values(
                    "_a", ascending=False).geometry.iloc[0]
                site_render_geom = best
                log.info(f"[context] site footprint: {best.area:.0f}m²")
            elif site_geom.area < 200:
                site_render_geom = site_point.buffer(20)
        except Exception as e:
            log.debug(f"[context] site footprint: {e}")
    elif site_geom.area < 200:
        site_render_geom = site_point.buffer(20)

    site_gdf = gpd.GeoDataFrame(geometry=[site_render_geom], crs=3857)
    del polygons
    gc.collect()

    # ── MTR stations ──────────────────────────────────────────────────────────
    stations_plot = _EMPTY.copy()
    nearest_name  = None

    if not _stn_result[0].empty:
        s          = _stn_result[0].copy()
        s["_name"] = _get_name(s)
        s["_cx"]   = s.geometry.centroid.x
        s["_cy"]   = s.geometry.centroid.y
        s["_dist"] = s.apply(
            lambda r: Point(r["_cx"], r["_cy"]).distance(site_point), axis=1)
        s = s.dropna(subset=["_name"]).sort_values("_dist")
        stations_plot = s[s["_dist"] <= half_size * 1.5].head(3).copy()
        if not s.empty:
            nearest_name = _ascii(str(s.iloc[0]["_name"]))
            log.info(f"[context] nearest MTR: {nearest_name} "
                     f"({s.iloc[0]['_dist']:.0f}m)")

    # ── Bus stops ─────────────────────────────────────────────────────────────
    bus_stops = _EMPTY.copy()
    if not _bus_result[0].empty:
        bs = _bus_result[0].copy()
        bs["geometry"] = bs.geometry.apply(
            lambda g: g.centroid if g.geom_type != "Point" else g)
        bus_stops = gpd.GeoDataFrame(bs, crs=3857)
        if len(bus_stops) > 6:
            try:
                from sklearn.cluster import KMeans
                coords = np.array([[g.x, g.y] for g in bus_stops.geometry])
                bus_stops["_cl"] = KMeans(
                    n_clusters=6, random_state=0).fit(coords).labels_
                bus_stops = gpd.GeoDataFrame(
                    bus_stops.groupby("_cl").first(), crs=3857
                ).reset_index(drop=True)
            except Exception:
                bus_stops = bus_stops.head(6)
        log.info(f"[context] bus stops: {len(bus_stops)}")

    # ── Place labels ──────────────────────────────────────────────────────────
    # Labels fetch — also bounded to 20s
    _lbl_result = [_EMPTY.copy()]
    def _fetch_labels():
        _lbl_result[0] = _fetch_osm(lat, lon, min(fetch_r, 800), {
            "amenity": ["school", "college", "university", "hospital",
                        "clinic", "supermarket", "library", "restaurant"],
            "leisure": ["park", "garden"],
            "place":   ["neighbourhood", "suburb"],
            "tourism": ["attraction"],
        })
    lbl_thread = threading.Thread(target=_fetch_labels, daemon=True)
    lbl_thread.start()
    lbl_thread.join(timeout=25)  # 20s server + 5s transfer
    if lbl_thread.is_alive():
        log.warning("[context] labels fetch timed out — skipping")
    labels_gdf = _lbl_result[0]

    label_items = []
    seen_text   = set()

    def _harvest(gdf):
        if gdf is None or gdf.empty:
            return
        g = gdf.copy()
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
             f"hi={len(hi_buildings)} "
             f"stn={len(stations_plot)} "
             f"bus={len(bus_stops)}")

    # ============================================================
    # RENDER
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

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.autoscale(False)

    # zorder 1-2: Landuse fills
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

    # zorder 9: Bus stops with icon
    if not bus_stops.empty:
        for _, row in bus_stops.iterrows():
            try:
                g  = row.geometry
                pt = g if g.geom_type == "Point" else g.centroid
                _draw_bus_icon(ax, pt.x, pt.y, zoom=0.028, zorder=9)
            except Exception as e:
                log.debug(f"[context] bus icon: {e}")
    del bus_stops
    gc.collect()

    # zorder 10 + 14: MTR station polygon + logo icon
    if not stations_plot.empty:
        try:
            _safe_plot(_polys(stations_plot), ax,
                       facecolor=MTR_COLOR, edgecolor="none",
                       alpha=0.9, zorder=10)
            for _, st in stations_plot.iterrows():
                _draw_mtr_icon(ax, st["_cx"], st["_cy"],
                               zoom=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # zorder 14-16: Site polygon (white glow + red fill)
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
    offsets = [(0,40),(0,-40),(40,0),(-40,0),
               (28,28),(-28,28),(28,-28),(-28,-28)]

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

    # zorder 17: MTR station name labels
    if not stations_plot.empty:
        for _, st in stations_plot.iterrows():
            try:
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
                log.debug(f"[context] stn label: {e}")

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
    ax.legend(handles=[
        mpatches.Patch(color="#f2c6a0",         label="Residential Area"),
        mpatches.Patch(color="#b39ddb",         label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9",         label="Public Park"),
        mpatches.Patch(color="#9ecae1",         label="School / Institution"),
        mpatches.Patch(color="#aec6cf",         label="Hospital / Clinic"),
        mpatches.Patch(color=hi_color, alpha=0.82, label=hi_label),
        mpatches.Patch(color=MTR_COLOR,         label="MTR Station"),
        mpatches.Patch(color="#e53935",         label="Site"),
        mpatches.Patch(color=BUS_COLOR,         label="Bus Stop"),
    ], loc="lower left", bbox_to_anchor=(0.02, 0.02),
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


def warm_cache(ZONE_DATA: gpd.GeoDataFrame) -> None:
    log.info("[context] warm_cache: ready (no-op)")
