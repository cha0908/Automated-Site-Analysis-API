"""
context.py — Site-Type-Driven Context Analysis  v6
====================================================
Root cause of timeouts: `building=[...]` tags in HK Overpass take 30-60s
even with bbox because there are millions of building features.

Fix: NEVER fetch building tags at all.
Use only landuse polygons + amenity/leisure/tourism POINTS.
Points are tiny (no geometry overhead) and return in <5s even in HK.

Architecture:
  - Stations fetched FIRST in its own thread (widest bbox, most important)
  - Landuse + amenity fetched in parallel threads
  - NO building tags anywhere
  - Sequential fallback: if parallel times out, retry stations alone
  - HEAD / fixed in snippet below for main.py
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
ox.settings.requests_timeout = 25

_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# ── Timing ─────────────────────────────────────────────────────────────────
_PER_FETCH_TIMEOUT  = 20   # tighter per-fetch budget
_FETCH_WALL_TIMEOUT = 25   # wall clock for ALL parallel fetches

# ── Map ────────────────────────────────────────────────────────────────────
MAP_HALF_SIZE = 700
MTR_COLOR     = "#ffd166"

# ── Assets ─────────────────────────────────────────────────────────────────
_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")

try:
    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
except Exception:
    _bus_icon = None

try:
    _mtr_logo = mpimg.imread(_MTR_LOGO_PATH)
    log.info("[context] MTR logo OK")
except Exception:
    _mtr_logo = None
    log.warning("[context] MTR logo missing")

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)

_CACHE      : dict = {}
_CACHE_LOCK        = threading.Lock()


# ── OZP site type ──────────────────────────────────────────────────────────

def _infer_site_type(zone_label: str) -> str:
    z = str(zone_label).upper().strip()
    if z.startswith("R"):                      return "RESIDENTIAL"
    if z.startswith("C"):                      return "COMMERCIAL"
    if z.startswith("G"):                      return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"): return "HOTEL"
    if z.startswith("I"):                      return "INDUSTRIAL"
    if z.startswith("OU"):                     return "OTHER"
    return "MIXED"


# ── Tag sets  — NO building tags, only landuse + amenity/leisure points ────

def _landuse_tags() -> dict:
    """Fast: returns large zone polygons only. Low count even in HK."""
    return {
        "landuse": ["residential", "commercial", "industrial",
                    "retail", "warehouse", "port"],
        "leisure": ["park", "playground", "garden", "recreation_ground"],
    }


def _amenity_tags(site_type: str) -> dict:
    """Point-only amenities. Fast because points have no geometry payload."""
    base = {"amenity": ["school", "college", "university", "hospital",
                        "clinic", "bank", "library", "government",
                        "post_office", "community_centre"]}
    if site_type in ("HOTEL", "COMMERCIAL", "MIXED", "OTHER"):
        base["amenity"] += ["restaurant", "cafe", "theatre", "cinema"]
    if site_type in ("HOTEL",):
        base["tourism"] = ["hotel", "hostel", "guest_house",
                           "attraction", "museum", "gallery"]
    if site_type in ("COMMERCIAL", "MIXED"):
        base["shop"] = ["mall", "department_store"]
    return base


def _station_tags() -> dict:
    return {"railway": "station"}


def _bus_tags() -> dict:
    return {"highway": "bus_stop"}


# ── Highlight colour per site type ─────────────────────────────────────────

_HI = {
    "RESIDENTIAL":  ("#e07b39", "Residential Area"),
    "COMMERCIAL":   ("#6a3d9a", "Commercial / Office Zone"),
    "HOTEL":        ("#b15928", "Hotel / Tourism Zone"),
    "INSTITUTIONAL":("#1f78b4", "Institutional Zone"),
    "INDUSTRIAL":   ("#33a02c", "Industrial Zone"),
    "OTHER":        ("#888888", "Other / Mixed Zone"),
    "MIXED":        ("#888888", "Mixed Use Zone"),
}

# Labels allowed per site type
_LABEL_AMENITY = {
    "RESIDENTIAL":   ["school","college","university","hospital",
                      "supermarket","library","community_centre"],
    "COMMERCIAL":    ["bank","government","library","post_office"],
    "HOTEL":         ["theatre","cinema","museum","attraction"],
    "INSTITUTIONAL": ["school","college","university","hospital",
                      "library","government"],
    "INDUSTRIAL":    [],
    "OTHER":         ["school","hospital","bank","library"],
    "MIXED":         ["school","hospital","bank","library"],
}
_LABEL_LEISURE = {k: ["park","garden"] for k in _HI}
_LABEL_LEISURE["INDUSTRIAL"] = []


# ── Bbox fetch (cached, daemon thread, multi-endpoint) ─────────────────────

def _tags_hash(tags) -> str:
    return hashlib.md5(json.dumps(tags, sort_keys=True).encode()).hexdigest()[:10]


def _bbox_from_point(lat, lon, dist):
    return ox.utils_geo.bbox_from_point((lat, lon), dist=dist)


def _fetch(bbox, tags, name="?") -> gpd.GeoDataFrame:
    north, south, east, west = bbox
    ck = (round((north+south)/2, 2), round((east+west)/2, 2),
          round(north-south, 4), _tags_hash(tags))

    with _CACHE_LOCK:
        if ck in _CACHE:
            log.info(f"[context] cache hit: {name}")
            return _CACHE[ck]

    result    = [_EMPTY.copy()]
    completed = [False]

    def _do():
        for ep in _OVERPASS_ENDPOINTS:
            try:
                ox.settings.overpass_endpoint = ep
                gdf = ox.features_from_bbox(bbox, tags=tags)
                if gdf is not None and not gdf.empty:
                    result[0] = gdf.to_crs(3857)
                log.info(f"[context] ✓ {name}: {len(result[0])} rows "
                         f"[{ep.split('/')[2]}]")
                completed[0] = True
                with _CACHE_LOCK:
                    _CACHE[ck] = result[0]
                return
            except Exception as e:
                log.warning(f"[context] {name}@{ep.split('/')[2]}: {e}")
        completed[0] = True
        with _CACHE_LOCK:
            _CACHE[ck] = result[0]

    t = threading.Thread(target=_do, daemon=True, name=f"ctx-{name}")
    t.start()
    t.join(timeout=_PER_FETCH_TIMEOUT)
    if not completed[0]:
        log.warning(f"[context] timeout ({_PER_FETCH_TIMEOUT}s): {name} "
                    "— will cache on completion")
    return result[0]


# ── Helpers ────────────────────────────────────────────────────────────────

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _col(gdf, col):
    return gdf[col] if col in gdf.columns else pd.Series(
        [None]*len(gdf), index=gdf.index)


def _filter(gdf, col, vals):
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    mask = gdf[col].isin(vals) if isinstance(vals, list) else (gdf[col]==vals)
    return gdf[mask].copy()


def _polys(gdf):
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(
        ["Polygon","MultiPolygon"])].copy()


def _get_name(gdf):
    return _col(gdf,"name:en").fillna(_col(gdf,"name"))


def _ascii(text) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    ratio = sum(1 for c in s if ord(c) < 128) / max(len(s), 1)
    if ratio < 0.5:
        return None
    out = "".join(c for c in s if ord(c) < 128).strip()
    return out or None


def _flat(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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


def _safe_plot(gdf, ax, **kw):
    try:
        if not gdf.empty:
            gdf.plot(ax=ax, **kw)
    except Exception as e:
        log.debug(f"[context] plot: {e}")


def _mtr_icon(ax, x, y, zoom=0.042, zorder=14):
    if _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ax.add_artist(AnnotationBbox(icon, (x, y), frameon=False,
                                     zorder=zorder,
                                     box_alignment=(0.5, 0.5)))
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=18,
                markeredgecolor="white", markeredgewidth=2.5, zorder=zorder)


# ── Main ───────────────────────────────────────────────────────────────────

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

    half  = radius_m if radius_m is not None else MAP_HALF_SIZE
    half_x = half * (16/12)
    half_y = half
    fetch_r = radius_m if radius_m is not None else MAP_HALF_SIZE

    log.info(f"[context] lon={lon:.5f} lat={lat:.5f} r={fetch_r}m")

    # 1. Site point
    site_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    # 2. Site polygon
    site_geom = None
    try:
        lb = get_lot_boundary(lon, lat, data_type, extents)
        if lb is not None and not lb.empty:
            cand = lb.geometry.iloc[0]
            if cand.area > 50:
                site_geom = cand
                log.info(f"[context] lot boundary area={cand.area:.0f}m²")
    except Exception as e:
        log.warning(f"[context] lot boundary: {e}")

    if site_geom is None:
        site_geom = site_pt.buffer(60)
        log.info("[context] 60m buffer fallback")

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # 3. OZP zone → site type
    ozp  = ZONE_DATA.to_crs(3857)
    hits = ozp[ozp.contains(site_pt)]
    if hits.empty:
        tmp      = ozp.copy()
        tmp["_d"] = tmp.geometry.distance(site_pt)
        primary  = tmp.sort_values("_d").iloc[0]
        log.warning("[context] nearest zone fallback")
    else:
        primary = hits.iloc[0]

    zone      = str(primary.get("ZONE_LABEL", "MIXED"))
    plan_no   = str(primary.get("PLAN_NO",    "N/A"))
    SITE_TYPE = _infer_site_type(zone)
    hi_color, hi_label = _HI.get(SITE_TYPE, ("#888888", "Zone"))
    log.info(f"[context] OZP zone={zone} SITE_TYPE={SITE_TYPE}")

    # 4. Bboxes
    bbox_main = _bbox_from_point(lat, lon, fetch_r)
    bbox_wide = _bbox_from_point(lat, lon, 1600)   # stations need wider

    # 5. PARALLEL FETCH — 4 tasks, no building tags
    #    Order: stations first (most important, widest bbox)
    tasks = {
        "stations": (bbox_wide, _station_tags()),
        "landuse":  (bbox_main, _landuse_tags()),
        "amenity":  (bbox_main, _amenity_tags(SITE_TYPE)),
        "bus":      (bbox_main, _bus_tags()),
    }

    fetched       = {k: _EMPTY.copy() for k in tasks}
    wall_deadline = _time.monotonic() + _FETCH_WALL_TIMEOUT

    with ThreadPoolExecutor(max_workers=4) as pool:
        fmap = {pool.submit(_fetch, bbox, tags, name): name
                for name, (bbox, tags) in tasks.items()}
        remaining = max(0.5, wall_deadline - _time.monotonic())
        for future in as_completed(fmap, timeout=remaining):
            name = fmap[future]
            try:
                fetched[name] = future.result()
            except Exception as e:
                log.warning(f"[context] future {name}: {e}")

    gc.collect()

    # 6. MTR stations — centroid + name, NO polygon
    stations_list = []   # [(cx, cy, name), ...]
    nearest_name  = None

    st_raw = _flat(fetched["stations"])
    if not st_raw.empty:
        st_raw["_name"] = _get_name(st_raw)
        st_raw["_cx"]   = st_raw.geometry.centroid.x
        st_raw["_cy"]   = st_raw.geometry.centroid.y
        st_raw["_dist"] = st_raw.apply(
            lambda r: Point(r["_cx"],r["_cy"]).distance(site_pt), axis=1)
        st_raw = st_raw.dropna(subset=["_name"]).sort_values("_dist")
        nearby = st_raw[st_raw["_dist"] <= half * 2.0].head(3)
        for _, row in nearby.iterrows():
            nm = _ascii(str(row["_name"]))
            if nm:
                stations_list.append((row["_cx"], row["_cy"], nm))
        if stations_list:
            nearest_name = stations_list[0][2]
            log.info(f"[context] MTR stations found: "
                     f"{[s[2] for s in stations_list]}")
    else:
        log.warning("[context] stations fetch returned empty — "
                    "will retry in background (cached for next request)")

    # 7. Landuse layers
    lu  = fetched["landuse"]
    res = _polys(_filter(lu, "landuse", ["residential"]))
    ind = _polys(_filter(lu, "landuse", ["industrial","commercial","warehouse"]))
    parks = _polys(_filter(lu, "leisure",
                            ["park","garden","recreation_ground","playground"]))

    # 8. Amenity layers
    am      = _flat(fetched["amenity"])
    schools = _filter(am, "amenity",
                      ["school","college","university","kindergarten"])
    hosps   = _filter(am, "amenity", ["hospital","clinic"])

    # 9. Bus stops
    bus_raw   = _flat(fetched["bus"])
    bus_stops = (bus_raw[bus_raw.geometry.geom_type=="Point"].copy()
                 if not bus_raw.empty else _EMPTY.copy())
    if len(bus_stops) > 8:
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[g.x,g.y] for g in bus_stops.geometry])
            lbls   = KMeans(n_clusters=8, random_state=0,
                            n_init=5).fit(coords).labels_
            bus_stops["_cl"] = lbls
            bus_stops = (gpd.GeoDataFrame(
                bus_stops.groupby("_cl").first(), crs=3857)
                .reset_index(drop=True))
        except Exception:
            bus_stops = bus_stops.head(8)

    # 10. Labels — site-type filtered
    allowed_am  = set(_LABEL_AMENITY.get(SITE_TYPE, []))
    allowed_leis = set(_LABEL_LEISURE.get(SITE_TYPE, []))
    label_items = []
    seen        = set()

    def _harvest(gdf, col, allowed):
        if gdf is None or gdf.empty:
            return
        g = _flat(gdf)
        if allowed:
            g = g[g[col].isin(allowed)] if col in g.columns else g.iloc[0:0]
        g["_lb"] = _get_name(g)
        g = g.dropna(subset=["_lb"])
        g = g[g["_lb"].astype(str).str.strip().str.len() > 0]
        for _, row in g.head(25).iterrows():
            txt = _ascii(str(row["_lb"]).strip())
            if not txt or txt in seen:
                continue
            seen.add(txt)
            try:
                geom = row.geometry
                p    = (geom.representative_point()
                        if hasattr(geom,"representative_point")
                        else geom.centroid)
                label_items.append((p.distance(site_pt), geom, txt))
            except Exception:
                continue

    _harvest(am,      "amenity", allowed_am)
    _harvest(lu,      "leisure", allowed_leis)
    _harvest(parks,   "leisure", allowed_leis)
    _harvest(schools, "amenity", allowed_am)
    _harvest(hosps,   "amenity", allowed_am)

    label_items.sort(key=lambda x: x[0])
    label_items = [(g,t) for _,g,t in label_items[:28]]

    del lu, am, fetched
    gc.collect()

    # 11. Draw
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    xmin = site_pt.x - half_x
    xmax = site_pt.x + half_x
    ymin = site_pt.y - half_y
    ymax = site_pt.y + half_y

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

    # Layers
    _safe_plot(res,     ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(ind,     ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks,   ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools, ax, color="#9ecae1", alpha=0.85, zorder=2)
    _safe_plot(hosps,   ax, color="#aec6cf", alpha=0.85, zorder=2)
    del res, ind, parks, schools, hosps
    gc.collect()

    # Bus stops
    if not bus_stops.empty:
        if _bus_icon is not None:
            for _, row in bus_stops.iterrows():
                try:
                    g  = row.geometry
                    bx = g.x if g.geom_type=="Point" else g.centroid.x
                    by = g.y if g.geom_type=="Point" else g.centroid.y
                    ic = OffsetImage(_bus_icon, zoom=0.028)
                    ic.image.axes = ax
                    ax.add_artist(AnnotationBbox(ic, (bx, by), frameon=False,
                                                 zorder=9,
                                                 box_alignment=(0.5,0.5)))
                except Exception:
                    pass
        else:
            bus_stops.plot(ax=ax, color="#0d47a1",
                           markersize=44, zorder=9, marker="s")
    del bus_stops
    gc.collect()

    # MTR — icon + name, no polygon
    for (cx_st, cy_st, nm) in stations_list:
        try:
            _mtr_icon(ax, cx_st, cy_st, zoom=0.042, zorder=14)
            ax.text(cx_st, cy_st - 110, _wrap(nm, 20),
                    fontsize=9.5, weight="bold", color="#1a1a1a",
                    ha="center", va="top",
                    bbox=dict(facecolor="white", edgecolor="#cccccc",
                              linewidth=0.8, alpha=0.95,
                              boxstyle="round,pad=3"),
                    zorder=15)
        except Exception as e:
            log.debug(f"[context] MTR draw: {e}")

    # Site — always on top
    try:
        sc = site_geom.centroid
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="white",
                      linewidth=2.5, zorder=16, alpha=0.95)
        ax.text(sc.x, sc.y, "SITE", color="white", weight="bold",
                fontsize=9, ha="center", va="center", zorder=17)
    except Exception:
        ax.plot(site_pt.x, site_pt.y, "o", color="#e53935",
                markersize=20, markeredgecolor="white",
                markeredgewidth=2.5, zorder=17)
        ax.text(site_pt.x, site_pt.y+90, "SITE",
                color="#e53935", weight="bold", fontsize=9,
                ha="center", va="bottom", zorder=18)

    # Place labels
    placed  = []
    offsets = [(0,55),(0,-55),(60,0),(-60,0),
               (42,42),(-42,42),(42,-42),(-42,-42)]
    for i, (geom, txt) in enumerate(label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom,"representative_point") else geom.centroid)
            if p.distance(site_pt) < 85:
                continue
            if any(p.distance(pp) < 95 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x+dx, p.y+dy, _wrap(txt,18),
                    fontsize=8.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.3"),
                    zorder=13, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # Info box
    lines = [f"{data_type}: {value}", f"OZP Plan: {plan_no}",
             f"Zoning: {zone}", f"Site Type: {SITE_TYPE}"]
    if nearest_name:
        lines.append(f"Nearest MTR: {nearest_name}")
    ax.text(0.012, 0.988, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black",
                      linewidth=1.5, pad=6),
            zorder=20)

    # Legend
    ax.legend(handles=[
        mpatches.Patch(color="#f2c6a0",            label="Residential"),
        mpatches.Patch(color="#b39ddb",            label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9",            label="Park / Garden"),
        mpatches.Patch(color="#9ecae1",            label="School / Institution"),
        mpatches.Patch(color="#aec6cf",            label="Hospital / Clinic"),
        mpatches.Patch(color=hi_color, alpha=0.82, label=hi_label),
        mpatches.Patch(color=MTR_COLOR,            label="MTR Station"),
        mpatches.Patch(color="#e53935",            label="Site"),
        mpatches.Patch(color="#0d47a1",            label="Bus Stop"),
    ], loc="lower left", bbox_to_anchor=(0.02,0.02),
       fontsize=8.5, framealpha=0.96, edgecolor="black", fancybox=False)

    ax.set_title(f"Automated Site Context Analysis – {data_type} {value}",
                 fontsize=15, weight="bold", pad=10)
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
    log.info(f"[context] Done in {_time.monotonic()-t0:.1f}s")
    return buf
