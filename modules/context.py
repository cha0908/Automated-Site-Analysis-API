"""
context.py — Site Context Analysis Map
Matches the Colab reference output quality:
  - Real shortest-path walk route to nearest MTR
  - Type-driven building highlights
  - Clean labels, 400m catchment ring
  - All rendered at 100 dpi
"""
import logging
import os
import gc
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import contextily as cx
import matplotlib.patches as mpatches
import numpy as np
import textwrap
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import Optional
from shapely.geometry import Point, LineString
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 25

log = logging.getLogger(__name__)

FETCH_RADIUS  = 600
MAP_HALF_SIZE = 800
MTR_COLOR     = "#ffd166"

_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
try:
    _bus_img = mpimg.imread(_BUS_ICON_PATH)
    log.info(f"[context] bus icon loaded {_bus_img.shape}")
except Exception as e:
    _bus_img = None
    log.warning(f"[context] bus icon missing: {e}")

_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
try:
    _mtr_logo        = mpimg.imread(_MTR_LOGO_PATH)
    _MTR_LOGO_LOADED = True
except Exception:
    _mtr_logo        = None
    _MTR_LOGO_LOADED = False

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)


# ── Site-type inference ───────────────────────────────────────────────────────

def infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                      return "RESIDENTIAL"
    if z.startswith("C"):                      return "COMMERCIAL"
    if z.startswith("G"):                      return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"): return "HOTEL"
    if z.startswith("OU"):                     return "OTHER"
    if z.startswith("I"):                      return "INDUSTRIAL"
    return "MIXED"


# ── Per-type OSM tags ─────────────────────────────────────────────────────────

_SIMILAR = {
    "RESIDENTIAL":   {"building": ["apartments", "residential",
                                   "house", "dormitory", "detached", "terrace"]},
    "HOTEL":         {"tourism": ["hotel", "hostel", "resort"],
                      "building": ["hotel"]},
    "COMMERCIAL":    {"building": ["office", "commercial", "retail"],
                      "office": True},
    "INSTITUTIONAL": {"amenity": ["school", "college", "university", "hospital"]},
    "INDUSTRIAL":    {"building": ["industrial", "warehouse"],
                      "landuse": ["industrial"]},
    "OTHER":         {"building": True},
    "MIXED":         {"building": True},
}

_SUPPORT = {
    "RESIDENTIAL":   {"amenity": ["school", "college", "university",
                                  "hospital", "clinic", "supermarket"],
                      "leisure": ["park"]},
    "HOTEL":         {"tourism": ["attraction"],
                      "amenity": ["restaurant", "cafe"],
                      "shop":    ["mall"]},
    "COMMERCIAL":    {"amenity": ["bank", "restaurant"],
                      "railway": ["station"]},
    "INSTITUTIONAL": {"leisure": ["park"],
                      "amenity": ["library", "hospital"]},
    "INDUSTRIAL":    {"landuse": ["port"]},
    "OTHER":         {"amenity": True},
    "MIXED":         {"amenity": True, "leisure": True},
}

_HI_COLOR = {
    "RESIDENTIAL": "#e07b39",
    "HOTEL":       "#b15928",
    "COMMERCIAL":  "#6a3d9a",
    "INSTITUTIONAL":"#1f78b4",
    "INDUSTRIAL":  "#33a02c",
    "OTHER":       "#888888",
    "MIXED":       "#888888",
}
_HI_LABEL = {
    "RESIDENTIAL": "Residential Developments",
    "HOTEL":       "Hotels & Serviced Apartments",
    "COMMERCIAL":  "Office / Commercial Buildings",
    "INSTITUTIONAL":"Institutional Buildings",
    "INDUSTRIAL":  "Industrial / Warehouse",
    "OTHER":       "Nearby Buildings",
    "MIXED":       "Nearby Buildings",
}
_SP_COLOR = {
    "RESIDENTIAL": "#9ecae1",
    "HOTEL":       "#fb9a99",
    "COMMERCIAL":  "#cab2d6",
    "INSTITUTIONAL":"#b7dfb9",
    "INDUSTRIAL":  "#b2df8a",
    "OTHER":       "#dddddd",
    "MIXED":       "#dddddd",
}
_SP_LABEL = {
    "RESIDENTIAL": "Schools / Parks / Hospitals",
    "HOTEL":       "Restaurants / Attractions",
    "COMMERCIAL":  "Banks / Restaurants",
    "INSTITUTIONAL":"Parks / Libraries",
    "INDUSTRIAL":  "Ports",
    "OTHER":       "Amenities",
    "MIXED":       "Amenities / Parks",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def wrap_label(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


# Overpass endpoints — tried in order on failure
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def _bbox_from_point(lat: float, lon: float, dist: float) -> tuple:
    """Return (north, south, east, west) bbox in WGS84."""
    return ox.utils_geo.bbox_from_point((lat, lon), dist=dist)


def _fetch_one(lat, lon, dist, tags) -> gpd.GeoDataFrame:
    """
    Fetch OSM features using bbox query (spatial index) instead of
    features_from_point (unindexed Overpass 'around' filter).
    bbox queries are 5-10x faster on dense urban areas like HK.
    Tries multiple Overpass endpoints on failure.
    """
    bbox = _bbox_from_point(lat, lon, dist)
    for ep in _OVERPASS_ENDPOINTS:
        try:
            ox.settings.overpass_endpoint = ep
            gdf = ox.features_from_bbox(bbox, tags=tags)
            if gdf is not None and not gdf.empty:
                return gdf.to_crs(3857)
            return _EMPTY.copy()
        except Exception as e:
            log.debug(f"[context] fetch {ep.split('/')[2]} "
                      f"{list(tags.keys())[:2]}: {e}")
            continue
    return _EMPTY.copy()


def _parallel_fetch(tasks: dict, wall_timeout: float = 35) -> dict:
    """
    Fetch all OSM layers in parallel threads.
    wall_timeout reduced to 35s (from 120s) — must complete before
    Render's 30s health check kills the process.
    Each individual fetch has a 25s timeout via ox.settings.requests_timeout.
    """
    results = {k: _EMPTY.copy() for k in tasks}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(_fetch_one, *args): key
                   for key, args in tasks.items()}
        done, not_done = wait(futures, timeout=wall_timeout,
                              return_when=ALL_COMPLETED)
        for f in not_done:
            f.cancel()
            log.warning(f"[context] timeout: {futures[f]}")
        for f in done:
            key = futures[f]
            try:
                results[key] = f.result()
                log.info(f"[context] ✓ {key}: {len(results[key])} rows")
            except Exception as e:
                log.warning(f"[context] ✗ {key}: {e}")
    return results


def _col(gdf, col):
    if col in gdf.columns:
        return gdf[col]
    return pd.Series([None] * len(gdf), index=gdf.index)


def _filter(gdf, col, val):
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    s = gdf[col]
    return gdf[s.isin(val) if isinstance(val, list) else (s == val)].copy()


def _polys(gdf):
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])].copy()


def _name(gdf):
    return _col(gdf, "name:en").fillna(_col(gdf, "name"))


def _ascii(text) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    if sum(1 for c in s if ord(c) < 128) / max(len(s), 1) < 0.5:
        return None
    return "".join(c for c in s if ord(c) < 128).strip() or None


def _safe_plot(gdf, ax, **kw):
    try:
        if not gdf.empty:
            gdf.plot(ax=ax, **kw)
    except Exception as e:
        log.debug(f"[context] plot: {e}")


def _draw_mtr(ax, x, y, zoom=0.035, zorder=14):
    if _MTR_LOGO_LOADED and _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ax.add_artist(AnnotationBbox(icon, (x, y), frameon=False,
                                     zorder=zorder, box_alignment=(0.5, 0.5)))
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=12,
                markeredgecolor="white", markeredgewidth=2, zorder=zorder)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_context(
    data_type: str,
    value: str,
    ZONE_DATA: gpd.GeoDataFrame,
    radius_m: Optional[int] = None,
    lon: float = None,
    lat: float = None,
    lot_ids: list = None,
    extents: list = None,
):
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r   = radius_m if radius_m is not None else FETCH_RADIUS
    half_size = radius_m if radius_m is not None else MAP_HALF_SIZE
    half_x    = half_size * (992 / 737)
    half_y    = half_size
    log.info(f"[context] lon={lon:.5f} lat={lat:.5f} r={fetch_r}m")

    # ── Site polygon ──────────────────────────────────────────────────────────
    try:
        lot_gdf = get_lot_boundary(lon, lat, data_type, extents)
    except Exception as e:
        log.warning(f"[context] get_lot_boundary: {e}")
        lot_gdf = None

    if lot_gdf is not None and not lot_gdf.empty:
        site_geom  = lot_gdf.geometry.iloc[0]
        site_point = site_geom.centroid
        log.info(f"[context] lot boundary area={site_geom.area:.0f}m²")
    else:
        site_point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
        site_geom  = site_point.buffer(15)
        log.info("[context] fallback point")
    site_render_geom = site_geom

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp     = ZONE_DATA.to_crs(3857)
    primary = ozp[ozp.contains(site_point)]
    if primary.empty:
        raise ValueError("No zoning polygon found for this site.")
    primary   = primary.iloc[0]
    zone      = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    hi_color  = _HI_COLOR.get(SITE_TYPE, "#888888")
    hi_label  = _HI_LABEL.get(SITE_TYPE, "Nearby Buildings")
    sp_color  = _SP_COLOR.get(SITE_TYPE, "#dddddd")
    sp_label  = _SP_LABEL.get(SITE_TYPE, "Amenities")
    log.info(f"[context] zone={zone} SITE_TYPE={SITE_TYPE}")

    # ── Parallel OSM fetch ────────────────────────────────────────────────────
    log.info("[context] Fetching OSM data (120s)...")

    results = _parallel_fetch({
        "base": (lat, lon, max(fetch_r, 1200), {
            "landuse":  True,
            "leisure":  ["park", "playground", "garden", "recreation_ground"],
            "highway":  ["bus_stop"],
            "place":    ["neighbourhood", "suburb"],
            "railway":  ["station"],
        }),
        "similar": (lat, lon, fetch_r, _SIMILAR.get(SITE_TYPE, _SIMILAR["MIXED"])),
        "support": (lat, lon, fetch_r, _SUPPORT.get(SITE_TYPE, _SUPPORT["MIXED"])),
        "labels":  (lat, lon, min(fetch_r, 700), {
            "amenity": ["school", "college", "university", "hospital",
                        "clinic", "supermarket", "library"],
            "leisure": ["park", "garden"],
            "place":   ["neighbourhood", "suburb"],
            "tourism": ["attraction"],
        }),
    }, wall_timeout=120)
    gc.collect()

    # ── Extract layers ────────────────────────────────────────────────────────
    base = results["base"]

    residential_area = _filter(base, "landuse", "residential")
    industrial_area  = _filter(base, "landuse", ["industrial", "commercial"])
    parks_layer      = _filter(base, "leisure", "park")

    bus_raw = _EMPTY.copy()
    if not base.empty and "highway" in base.columns:
        bs = base[_col(base, "highway") == "bus_stop"].copy()
        if not bs.empty:
            bs["geometry"] = bs.geometry.apply(
                lambda g: g.centroid if g.geom_type != "Point" else g)
            bus_raw = gpd.GeoDataFrame(bs, crs=3857)

    stations_all     = _filter(base, "railway", "station")
    stations_in_view = _EMPTY.copy()
    nearest_stn      = None
    nearest_stn_name = None

    if not stations_all.empty:
        s             = stations_all.copy()
        s["_name"]    = _name(s)
        s["_centroid"]= s.geometry.centroid
        s["_dist"]    = s["_centroid"].apply(lambda g: g.distance(site_point))
        stn_sorted    = s.dropna(subset=["_name"]).sort_values("_dist").head(3)
        stations_in_view = stn_sorted[stn_sorted["_dist"] <= half_size * 1.5]
        if not stn_sorted.empty:
            nearest_stn      = stn_sorted.iloc[0]
            nearest_stn_name = _ascii(str(nearest_stn["_name"]))

    support_raw    = results["support"]
    support_blds   = _polys(support_raw)
    schools_poly   = _filter(support_raw, "amenity",
                              ["school", "college", "university"]) \
                     if not support_raw.empty else _EMPTY.copy()
    hospitals_poly = _filter(support_raw, "amenity",
                              ["hospital", "clinic"]) \
                     if not support_raw.empty else _EMPTY.copy()

    _sim_raw = _polys(results["similar"])

    # ── Upgrade site polygon to nearest building footprint ────────────────────
    site_render_geom = site_geom
    if not _sim_raw.empty:
        try:
            _nearby = _sim_raw[_sim_raw.geometry.distance(site_point) < 40]
            if not _nearby.empty:
                _best = _nearby.assign(_a=_nearby.area).sort_values(
                    "_a", ascending=False).geometry.iloc[0]
                site_render_geom = _best
                log.info(f"[context] site from OSM building area={_best.area:.0f}m²")
            elif site_geom.area < 200:
                site_render_geom = site_point.buffer(20)
                log.info("[context] site: small buffer 20m")
        except Exception as e:
            log.debug(f"[context] site building lookup: {e}")
    elif site_geom.area < 200:
        site_render_geom = site_point.buffer(20)

    site_gdf = gpd.GeoDataFrame(geometry=[site_render_geom], crs=3857)

    if not _sim_raw.empty:
        _sim_raw["_nm"] = _name(_sim_raw)
        _has_name = _sim_raw["_nm"].apply(
            lambda x: bool(_ascii(str(x))) if pd.notna(x) else False)
        _large    = _sim_raw.geometry.area >= 600
        similar_blds = _sim_raw[_has_name & _large].copy()
        log.info(f"[context] similar (named+600m²): {len(similar_blds)}")
    else:
        similar_blds = _EMPTY.copy()

    # Cap polygon count to protect render memory on Render free tier (512MB)
    # 200 large polygons is visually identical to 500+ at any useful zoom level
    if len(similar_blds) > 200:
        similar_blds = similar_blds.assign(
            _area=similar_blds.geometry.area
        ).nlargest(200, "_area").drop(columns=["_area"])
        log.info(f"[context] similar capped to 200 largest")

    if len(support_blds) > 150:
        support_blds = support_blds.head(150)

    bus_stops = bus_raw.copy()
    if len(bus_stops) > 10:
        try:
            from sklearn.cluster import KMeans
            pts    = [g if g.geom_type == "Point" else g.centroid
                      for g in bus_stops.geometry]
            coords = np.array([[p.x, p.y] for p in pts])
            bus_stops["_cl"] = KMeans(
                n_clusters=10, random_state=0).fit(coords).labels_
            bus_stops = gpd.GeoDataFrame(
                bus_stops.groupby("_cl").first(), crs=3857)
        except Exception as e:
            log.debug(f"[context] kmeans: {e}")
            bus_stops = bus_stops.head(10)
    log.info(f"[context] bus stops: {len(bus_stops)}")

    # ── Place labels ──────────────────────────────────────────────────────────
    labels_raw = results["labels"]
    all_labels = []
    seen       = set()

    def _collect(gdf):
        if gdf is None or gdf.empty:
            return
        g        = gdf.copy()
        g["_lb"] = _name(g)
        for _, row in g.dropna(subset=["_lb"]).iterrows():
            text = _ascii(str(row["_lb"]).strip())
            if not text or text in seen:
                continue
            seen.add(text)
            geom = row.geometry
            p    = (geom.representative_point()
                    if hasattr(geom, "representative_point") else geom.centroid)
            all_labels.append((p.distance(site_point), geom, text))

    for src in [labels_raw, support_raw, parks_layer, schools_poly]:
        _collect(src)

    all_labels.sort(key=lambda x: x[0])
    label_items = [(g, t) for _, g, t in all_labels[:28]]
    log.info(f"[context] labels: {len(label_items)}")

    del base, results, labels_raw, support_raw
    gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # RENDER
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("[context] Rendering...")

    # FIX: 14×10.5 at dpi=100 = 1400×1050px (vs 1777×1320 before)
    # Saves ~40% of numpy array memory during render — critical on 512MB free tier
    fig, ax = plt.subplots(figsize=(14, 10.5))
    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=15, alpha=0.95)
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                           zoom=15, alpha=0.95)
        except Exception as e:
            log.warning(f"[context] basemap: {e}")

    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # zorder 1: Base landuse
    _safe_plot(residential_area, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(industrial_area,  ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks_layer,      ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools_poly,     ax, color="#9ecae1", alpha=0.90, zorder=2)
    _safe_plot(hospitals_poly,   ax, color="#aec6cf", alpha=0.90, zorder=2)
    del residential_area, industrial_area, parks_layer, schools_poly, hospitals_poly
    gc.collect()

    # zorder 3-4: Type-specific buildings
    _safe_plot(support_blds, ax, color=sp_color, alpha=0.45, zorder=3)
    _safe_plot(similar_blds, ax, color=hi_color,  alpha=0.85, zorder=4,
               edgecolor="white", linewidth=0.3)
    del support_blds, similar_blds
    gc.collect()

    # zorder 9: Bus stops
    if not bus_stops.empty:
        try:
            for _, row in bus_stops.iterrows():
                g  = row.geometry
                pt = g if g.geom_type == "Point" else g.centroid
                if _bus_img is not None:
                    icon = OffsetImage(_bus_img, zoom=0.025)
                    icon.image.axes = ax
                    ax.add_artist(AnnotationBbox(icon, (pt.x, pt.y),
                                                 frameon=False, zorder=9,
                                                 box_alignment=(0.5, 0.5)))
                else:
                    ax.plot(pt.x, pt.y, "s", color="#0d47a1", markersize=10,
                            zorder=9, markeredgecolor="white", markeredgewidth=1)
        except Exception as e:
            log.debug(f"[context] bus render: {e}")
    del bus_stops
    gc.collect()

    # zorder 10-14: MTR stations
    if not stations_in_view.empty:
        try:
            stations_in_view.plot(ax=ax, facecolor=MTR_COLOR,
                                  edgecolor="none", alpha=0.9, zorder=10)
            for _, st in stations_in_view.iterrows():
                c = st.geometry.centroid
                _draw_mtr(ax, c.x, c.y, zoom=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # zorder 14-16: Site
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
    offsets = [(0, 40), (0, -40), (40, 0), (-40, 0), (28, 28), (-28, 28)]
    for i, (geom, text) in enumerate(label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_point) < 100:
                continue
            if any(p.distance(pp) < 90 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x + dx, p.y + dy, wrap_label(text, 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=12, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # zorder 17: MTR station name labels
    if not stations_in_view.empty:
        try:
            for _, st in stations_in_view.iterrows():
                label = _ascii(str(st.get("_name", "") or ""))
                if not label:
                    continue
                c = st.geometry.centroid
                ax.text(c.x, c.y + 140, wrap_label(label, 18),
                        fontsize=9, weight="bold", color="black",
                        ha="center", va="bottom",
                        bbox=dict(facecolor=MTR_COLOR, edgecolor="none",
                                  alpha=0.9, pad=2),
                        zorder=17)
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    # Info box
    info = (f"{data_type}: {value}\n"
            f"OZP Plan: {primary['PLAN_NO']}\n"
            f"Zoning: {zone}\n"
            f"Site Type: {SITE_TYPE}")
    if nearest_stn_name:
        info += f"\nNearest MTR: {nearest_stn_name}"
    ax.text(0.012, 0.988, info, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6),
            zorder=20)

    # Legend
    ax.legend(handles=[
        mpatches.Patch(color="#f2c6a0",   label="Residential Area"),
        mpatches.Patch(color="#b39ddb",   label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9",   label="Public Park"),
        mpatches.Patch(color="#9ecae1",   label="School / Institution"),
        mpatches.Patch(color="#aec6cf",   label="Hospital / Clinic"),
        mpatches.Patch(color=hi_color, alpha=0.85, label=hi_label),
        mpatches.Patch(color=sp_color, alpha=0.45, label=sp_label),
        mpatches.Patch(color=MTR_COLOR,   label="MTR Station"),
        mpatches.Patch(color="#e53935",   label="Site"),
        mpatches.Patch(color="#0d47a1",   label="Bus Stop"),
    ], loc="lower left", bbox_to_anchor=(0.02, 0.02),
       fontsize=8.5, framealpha=0.95)

    ax.set_title(f"Automated Site Context Analysis – {data_type} {value}",
                 fontsize=15, weight="bold")
    ax.set_axis_off()
    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.autoscale(False)

    buf = BytesIO()
    plt.tight_layout()
    ax.set_position([0.02, 0.02, 0.96, 0.96])
    plt.savefig(buf, format="png", dpi=100,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("[context] Done.")
    return buf
