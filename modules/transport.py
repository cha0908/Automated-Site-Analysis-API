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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Optional
from shapely.geometry import Point
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 20

log = logging.getLogger(__name__)

FETCH_RADIUS     = 800
MAP_HALF_SIZE    = 600
MTR_COLOR        = "#ffd166"
WALK_ROUTE_COLOR = "#005eff"
FETCH_TIMEOUT    = 30   # per-task wall-clock timeout in seconds

_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
try:
    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
except Exception:
    _bus_icon = None

_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
try:
    _mtr_logo        = mpimg.imread(_MTR_LOGO_PATH)
    _MTR_LOGO_LOADED = True
except Exception:
    _mtr_logo        = None
    _MTR_LOGO_LOADED = False

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)


# ── Site type ─────────────────────────────────────────────────────────────────

def infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                      return "RESIDENTIAL"
    if z.startswith("C"):                      return "COMMERCIAL"
    if z.startswith("G"):                      return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"): return "HOTEL"
    if z.startswith("OU"):                     return "OTHER"
    if z.startswith("I"):                      return "INDUSTRIAL"
    return "MIXED"


TYPE_CONFIG = {
    "RESIDENTIAL": {
        "similar_tags":    {"building": ["apartments", "residential",
                                         "house", "dormitory", "detached", "terrace"]},
        "support_tags":    {"amenity": ["school", "hospital", "supermarket"],
                            "leisure": ["park"]},
        "highlight_color": "#e07b39",
        "support_color":   "#9ecae1",
        "highlight_label": "Residential Developments",
        "support_label":   "Schools / Parks / Hospitals",
    },
    "HOTEL": {
        "similar_tags":    {"tourism": ["hotel", "hostel", "resort"],
                            "building": ["hotel"]},
        "support_tags":    {"tourism": ["attraction"],
                            "amenity": ["restaurant"], "shop": ["mall"]},
        "highlight_color": "#b15928",
        "support_color":   "#fb9a99",
        "highlight_label": "Hotels & Serviced Apartments",
        "support_label":   "Restaurants / Attractions",
    },
    "COMMERCIAL": {
        "similar_tags":    {"building": ["office", "commercial"],
                            "office": True, "landuse": ["commercial"]},
        "support_tags":    {"amenity": ["bank", "restaurant"],
                            "railway": ["station"]},
        "highlight_color": "#6a3d9a",
        "support_color":   "#cab2d6",
        "highlight_label": "Office / Commercial Buildings",
        "support_label":   "Banks / Restaurants / Transit",
    },
    "INSTITUTIONAL": {
        "similar_tags":    {"amenity": ["school", "college", "university",
                                        "hospital", "clinic", "government"]},
        "support_tags":    {"leisure": ["park"], "amenity": ["library"]},
        "highlight_color": "#1f78b4",
        "support_color":   "#b7dfb9",
        "highlight_label": "Institutional Buildings",
        "support_label":   "Parks / Libraries",
    },
    "INDUSTRIAL": {
        "similar_tags":    {"building": ["industrial", "warehouse"],
                            "landuse": ["industrial"],
                            "industrial": ["factory"]},
        "support_tags":    {"highway": ["motorway"], "landuse": ["port"]},
        "highlight_color": "#33a02c",
        "support_color":   "#b2df8a",
        "highlight_label": "Industrial / Warehouse Buildings",
        "support_label":   "Ports / Motorways",
    },
    "OTHER": {
        "similar_tags":    {"building": True},
        "support_tags":    {"amenity": True},
        "highlight_color": "#aaaaaa",
        "support_color":   "#dddddd",
        "highlight_label": "Nearby Buildings",
        "support_label":   "Amenities",
    },
    "MIXED": {
        "similar_tags":    {"building": True},
        "support_tags":    {"amenity": True, "leisure": True},
        "highlight_color": "#aaaaaa",
        "support_color":   "#b7dfb9",
        "highlight_label": "Nearby Buildings",
        "support_label":   "Amenities / Parks",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def wrap_label(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _fetch(lat, lon, dist, tags) -> gpd.GeoDataFrame:
    """Single OSMnx fetch — returns empty GDF on any failure."""
    try:
        gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags)
        if gdf is not None and not gdf.empty:
            return gdf.to_crs(3857)
    except Exception as e:
        log.debug(f"[context] fetch {list(tags)[:2]}: {e}")
    return _EMPTY.copy()


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
        ax.plot(x, y, "o", color="#ED1D24", markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, zorder=zorder)


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
        site_gdf   = lot_gdf
        site_point = site_geom.centroid
    else:
        site_point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
        site_geom  = site_point.buffer(60)
        site_gdf   = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp     = ZONE_DATA.to_crs(3857)
    primary = ozp[ozp.contains(site_point)]
    if primary.empty:
        raise ValueError("No zoning polygon found for this site.")
    primary   = primary.iloc[0]
    zone      = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    cfg       = TYPE_CONFIG.get(SITE_TYPE, TYPE_CONFIG["MIXED"])
    log.info(f"[context] zone={zone} site_type={SITE_TYPE}")

    # ── Parallel fetch ────────────────────────────────────────────────────────
    # All OSMnx calls run at the same time — total time = slowest single fetch
    log.info("[context] Parallel fetching all layers...")

    tasks = {
        "base":     (lat, lon, fetch_r,  {"landuse": True,
                                           "leisure": ["park", "playground",
                                                       "garden", "recreation_ground"]}),
        "schools":  (lat, lon, fetch_r,  {"amenity": ["school", "college", "university"]}),
        "similar":  (lat, lon, fetch_r,  cfg["similar_tags"]),
        "support":  (lat, lon, fetch_r,  cfg["support_tags"]),
        "stations": (lat, lon, 1500,     {"railway": "station"}),
        "bus":      (lat, lon, 700,      {"highway": "bus_stop"}),
        "labels":   (lat, lon, min(fetch_r, 600),
                     {"amenity": ["school", "college", "university", "hospital"],
                      "leisure": ["park"],
                      "place":   ["neighbourhood", "suburb"]}),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {pool.submit(_fetch, *args): key
                   for key, args in tasks.items()}
        for future in as_completed(futures, timeout=90):
            key = futures[future]
            try:
                results[key] = future.result()
                log.info(f"[context] ✓ {key}: {len(results[key])} rows")
            except Exception as e:
                log.warning(f"[context] ✗ {key}: {e}")
                results[key] = _EMPTY.copy()

    # Fill any that timed out
    for key in tasks:
        if key not in results:
            log.warning(f"[context] timed out: {key}")
            results[key] = _EMPTY.copy()

    gc.collect()

    # ── Process base ──────────────────────────────────────────────────────────
    base             = results["base"]
    residential_area = _filter_col(base, "landuse", "residential")
    industrial_area  = _filter_col(base, "landuse", ["industrial", "commercial"])
    parks            = _filter_col(base, "leisure", "park")
    del base

    schools      = results["schools"]
    similar_blds = _polys_only(results["similar"])
    support_blds = _polys_only(results["support"])
    bus_stops_raw = results["bus"]
    labels_raw    = results["labels"]

    # ── Process stations ──────────────────────────────────────────────────────
    stations_raw     = results["stations"]
    stations         = _EMPTY.copy()
    stations_in_view = _EMPTY.copy()

    if not stations_raw.empty:
        s             = stations_raw.copy()
        s["name"]     = _get_name(s)
        s["centroid"] = s.geometry.centroid
        s["dist"]     = s["centroid"].apply(lambda g: g.distance(site_point))
        stations      = s.dropna(subset=["name"]).sort_values("dist").head(3)
        stations_in_view = stations[stations["dist"] <= half_size * 1.3]

    # ── Process bus stops ─────────────────────────────────────────────────────
    bus_stops = bus_stops_raw
    if len(bus_stops) > 6:
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[g.centroid.x, g.centroid.y]
                                for g in bus_stops.geometry])
            bus_stops = bus_stops.copy()
            bus_stops["cluster"] = KMeans(
                n_clusters=6, random_state=0).fit(coords).labels_
            bus_stops = gpd.GeoDataFrame(
                bus_stops.groupby("cluster").first(), crs=3857)
        except Exception:
            bus_stops = bus_stops.head(6)

    # ── Process labels ────────────────────────────────────────────────────────
    all_label_items = []
    seen_texts      = set()

    def _collect(gdf):
        if gdf is None or gdf.empty:
            return
        g        = gdf.copy()
        g["_lb"] = _get_name(g)
        named    = g.dropna(subset=["_lb"])
        named    = named[named["_lb"].astype(str).str.strip().str.len() > 0]
        for _, row in named.iterrows():
            text = str(row["_lb"]).strip()
            if text in seen_texts:
                continue
            seen_texts.add(text)
            geom = row.geometry
            p    = (geom.representative_point()
                    if hasattr(geom, "representative_point") else geom.centroid)
            all_label_items.append((p.distance(site_point), geom, text))

    for src in [labels_raw, schools, parks, similar_blds, support_blds]:
        _collect(src)

    all_label_items.sort(key=lambda x: x[0])
    all_label_items = [(g, t) for _, g, t in all_label_items[:35]]

    del labels_raw, results
    gc.collect()

    log.info("[context] Rendering...")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16.15, 12))
    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=16, alpha=0.95)
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                           zoom=16, alpha=0.95)
        except Exception as e:
            log.warning(f"[context] basemap: {e}")

    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # ── Layers ────────────────────────────────────────────────────────────────
    _safe_plot(residential_area, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(industrial_area,  ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks,            ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools,          ax, color="#9ecae1", alpha=0.90, zorder=2)
    _safe_plot(support_blds,     ax, color=cfg["support_color"],   alpha=0.60, zorder=3)
    _safe_plot(similar_blds,     ax, color=cfg["highlight_color"], alpha=0.80, zorder=4)

    del residential_area, industrial_area, parks, schools
    del support_blds, similar_blds
    gc.collect()

    # ── Bus stops ─────────────────────────────────────────────────────────────
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
                bus_stops.plot(ax=ax, color="#0d47a1", markersize=35, zorder=9)
        except Exception as e:
            log.debug(f"[context] bus stop render: {e}")
    del bus_stops
    gc.collect()

    # ── MTR stations ──────────────────────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            stations_in_view.plot(ax=ax, facecolor=MTR_COLOR,
                                  edgecolor="none", alpha=0.9, zorder=10)
            for _, st in stations_in_view.iterrows():
                c = st.geometry.centroid
                _draw_mtr_icon(ax, c.x, c.y, zoom=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # ── Site — always on top ──────────────────────────────────────────────────
    try:
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred",
                      linewidth=2.5, zorder=15)
        sc = site_geom.centroid
        ax.text(sc.x, sc.y, "SITE", color="white", weight="bold",
                fontsize=10, ha="center", va="center", zorder=16)
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    # ── Place labels ──────────────────────────────────────────────────────────
    placed  = []
    offsets = [(0, 40), (0, -40), (40, 0), (-40, 0), (30, 30), (-30, 30)]
    for i, (geom, text) in enumerate(all_label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_point) < 100:
                continue
            if any(p.distance(pp) < 100 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x + dx, p.y + dy, wrap_label(text, 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=13, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # ── MTR station labels ────────────────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            for _, st in stations_in_view.iterrows():
                raw = st.get("name")
                if not raw or not isinstance(raw, str) or not str(raw).strip():
                    continue
                c = st.geometry.centroid
                ax.text(c.x, c.y + 140, wrap_label(str(raw).strip(), 18),
                        fontsize=9, weight="bold", color="black",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=2),
                        zorder=17)
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    # ── Info box ──────────────────────────────────────────────────────────────
    ax.text(
        0.012, 0.988,
        f"{data_type}: {value}\n"
        f"OZP Plan: {primary['PLAN_NO']}\n"
        f"Zoning: {zone}\n"
        f"Site Type: {SITE_TYPE}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9.2,
        bbox=dict(facecolor="white", edgecolor="black", pad=6),
        zorder=20,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color="#f2c6a0",              label="Residential Area"),
        mpatches.Patch(color="#b39ddb",              label="Industrial / Commercial Area"),
        mpatches.Patch(color="#b7dfb9",              label="Public Park"),
        mpatches.Patch(color="#9ecae1",              label="School / Institution"),
        mpatches.Patch(color=cfg["highlight_color"], alpha=0.80,
                       label=cfg["highlight_label"]),
        mpatches.Patch(color=cfg["support_color"],   alpha=0.60,
                       label=cfg["support_label"]),
        mpatches.Patch(color=MTR_COLOR,              label="MTR Station"),
        mpatches.Patch(color="#e53935",              label="Site"),
        mpatches.Patch(color="#0d47a1",              label="Bus Stop"),
    ]

    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(0.02, 0.02),
              fontsize=8.5, framealpha=0.95)

    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=15, weight="bold",
    )
    ax.set_axis_off()
    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.autoscale(False)

    buf = BytesIO()
    plt.tight_layout()
    ax.set_position([0.02, 0.02, 0.96, 0.96])
    plt.savefig(buf, format="png", dpi=110,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("[context] Done.")
    return buf
