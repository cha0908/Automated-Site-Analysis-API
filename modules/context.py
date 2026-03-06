"""
context.py — Site-Type Driven Context Analysis
ALKF+ Module

Generates a context analysis map that adapts its content based on the
inferred site type (Residential, Hotel, Commercial, Institutional,
Industrial). Each type highlights the most analytically relevant
surrounding developments and infrastructure.

Optionally computes a pedestrian walking route from site → nearest
MTR station using the OSM walk network.

Architecture notes:
- All OSM fetches run in a ThreadPoolExecutor (parallel)
- OSMnx MultiIndex flattened before any concat or attribute access
- Map extent locked before cx.add_basemap to prevent tile zoom errors
- ax.set_aspect / ax.autoscale called AFTER basemap to preserve extent
"""

import logging
import os
import gc
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import contextily as cx
import numpy as np
import textwrap
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import Optional, List
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 30

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

FETCH_RADIUS  = 700
MAP_HALF_SIZE = 700
MTR_COLOR     = "#ffd166"
ROUTE_COLOR   = "#1565C0"   # blue dashed pedestrian route

_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")

try:
    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
except Exception:
    _bus_icon = None

try:
    _mtr_logo        = mpimg.imread(_MTR_LOGO_PATH)
    _MTR_LOGO_LOADED = True
except Exception:
    _mtr_logo        = None
    _MTR_LOGO_LOADED = False

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)


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
# SITE-TYPE DRIVEN FETCH CONFIGURATION
#
# Each site type specifies:
#   primary_tags  — OSM tags for type-specific highlighted buildings
#   support_tags  — secondary amenities relevant to this programme
#   highlight_color  — fill color for primary buildings on map
#   highlight_label  — legend label for primary buildings
#   label_priority   — ordered list of OSM tag keys to extract place labels from
# ============================================================

SITE_CONFIGS = {
    "RESIDENTIAL": {
        "primary_tags": {
            "building": ["apartments", "residential", "house",
                         "dormitory", "detached", "terrace", "semidetached_house"],
        },
        "support_tags": {
            "amenity":  ["school", "college", "university",
                         "hospital", "clinic", "supermarket", "kindergarten"],
            "leisure":  ["park", "playground", "garden", "recreation_ground"],
            "shop":     ["supermarket", "convenience"],
        },
        "highlight_color": "#e07b39",
        "highlight_label": "Residential Developments",
        "label_priority":  ["leisure", "amenity", "name"],
    },
    "HOTEL": {
        "primary_tags": {
            "tourism":  ["hotel", "hostel", "guest_house", "resort"],
            "building": ["hotel"],
        },
        "support_tags": {
            "tourism":  ["attraction", "museum", "viewpoint"],
            "amenity":  ["restaurant", "cafe", "bar", "nightclub"],
            "shop":     ["mall", "department_store"],
            "leisure":  ["park"],
        },
        "highlight_color": "#b15928",
        "highlight_label": "Hotels & Serviced Apartments",
        "label_priority":  ["tourism", "amenity", "name"],
    },
    "COMMERCIAL": {
        "primary_tags": {
            "building": ["office", "commercial", "retail"],
            "office":   True,
        },
        "support_tags": {
            "amenity":  ["bank", "restaurant", "cafe", "atm", "fast_food"],
            "shop":     ["mall", "department_store"],
            "railway":  ["station"],
        },
        "highlight_color": "#6a3d9a",
        "highlight_label": "Office / Commercial Buildings",
        "label_priority":  ["office", "amenity", "building", "name"],
    },
    "INSTITUTIONAL": {
        "primary_tags": {
            "amenity":  ["school", "college", "university",
                         "hospital", "government", "library", "courthouse"],
            "building": ["government", "civic", "hospital", "school"],
        },
        "support_tags": {
            "leisure":  ["park", "garden"],
            "amenity":  ["library", "community_centre"],
        },
        "highlight_color": "#1f78b4",
        "highlight_label": "Institutional Buildings",
        "label_priority":  ["amenity", "building", "name"],
    },
    "INDUSTRIAL": {
        "primary_tags": {
            "building": ["industrial", "warehouse", "factory", "storage_tank"],
            "landuse":  ["industrial", "port"],
        },
        "support_tags": {
            "highway":  ["motorway", "trunk", "primary"],
            "railway":  ["rail", "goods"],
            "landuse":  ["port", "logistics"],
        },
        "highlight_color": "#33a02c",
        "highlight_label": "Industrial / Warehouse Buildings",
        "label_priority":  ["landuse", "building", "name"],
    },
    "OTHER": {
        "primary_tags": {"building": True},
        "support_tags":  {"amenity": True, "leisure": True},
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
        "label_priority":  ["amenity", "name"],
    },
    "MIXED": {
        "primary_tags": {"building": True},
        "support_tags":  {"amenity": True, "leisure": True, "shop": True},
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
        "label_priority":  ["amenity", "leisure", "name"],
    },
}


# ============================================================
# HELPERS
# ============================================================

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _fetch_one(lat, lon, dist, tags) -> gpd.GeoDataFrame:
    try:
        gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags)
        if gdf is not None and not gdf.empty:
            return gdf.to_crs(3857)
    except Exception as e:
        log.debug(f"[context] fetch {list(tags.keys())[:2]}: {e}")
    return _EMPTY.copy()


def _parallel_fetch(tasks: dict, wall_timeout: float = 130) -> dict:
    results = {k: _EMPTY.copy() for k in tasks}
    with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as pool:
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


def _filter_col(gdf, col, val):
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    s    = gdf[col]
    mask = s.isin(val) if isinstance(val, list) else (s == val)
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
    if not s:
        return None
    ascii_ratio = sum(1 for c in s if ord(c) < 128) / max(len(s), 1)
    if ascii_ratio < 0.5:
        return None
    cleaned = "".join(c for c in s if ord(c) < 128).strip()
    return cleaned if cleaned else None


def _flatten_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Flatten OSMnx MultiIndex (element_type, osmid) to plain int index.
    Must be called on each GeoDataFrame individually before pd.concat.
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
# PEDESTRIAN ROUTE COMPUTATION
# Site → nearest MTR station via OSM walk network
# ============================================================

def _compute_walk_route(lat: float, lon: float,
                        station_centroid_wgs84: Point,
                        walk_dist: int = 1200) -> Optional[gpd.GeoDataFrame]:
    """
    Compute shortest pedestrian path from site to MTR station centroid
    using OSMnx walk network. Returns GeoDataFrame in EPSG:3857 or None.

    Parameters
    ----------
    lat, lon : float
        Site WGS84 coordinates.
    station_centroid_wgs84 : Point
        Station centroid in WGS84 (EPSG:4326).
    walk_dist : int
        Radius in metres around site to build walk graph.
    """
    try:
        G = ox.graph_from_point(
            (lat, lon),
            dist=walk_dist,
            network_type="walk",
            simplify=True
        )
        if G.number_of_nodes() == 0:
            return None

        site_node = ox.distance.nearest_nodes(G, lon, lat)
        stn_node  = ox.distance.nearest_nodes(
            G,
            station_centroid_wgs84.x,
            station_centroid_wgs84.y
        )

        if site_node == stn_node:
            return None

        path = nx.shortest_path(G, site_node, stn_node, weight="length")
        route_gdf = ox.routing.route_to_gdf(G, path).to_crs(3857)
        return route_gdf

    except Exception as e:
        log.warning(f"[context] walk route failed: {e}")
        return None


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

    # Slightly wider horizontal extent for 16:9 aspect
    half_x = half_size * (16 / 12)
    half_y = half_size

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
        log.info(f"[context] lot boundary: area={site_geom.area:.0f}m²")
    else:
        site_point = gpd.GeoSeries(
            [Point(lon, lat)], crs=4326
        ).to_crs(3857).iloc[0]
        site_geom = site_point.buffer(80)
        site_gdf  = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)
        log.info("[context] using 80m buffer fallback")

    # ── Zoning lookup ─────────────────────────────────────────────────────────
    ozp     = ZONE_DATA.to_crs(3857)
    primary = ozp[ozp.contains(site_point)]
    if primary.empty:
        raise ValueError("No OZP zoning polygon found for this site.")
    primary   = primary.iloc[0]
    zone      = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    cfg       = SITE_CONFIGS.get(SITE_TYPE, SITE_CONFIGS["MIXED"])
    log.info(f"[context] zone={zone} SITE_TYPE={SITE_TYPE}")

    hi_color  = cfg["highlight_color"]
    hi_label  = cfg["highlight_label"]

    # ── Parallel OSM fetches ──────────────────────────────────────────────────
    log.info("[context] Fetching OSM data in parallel...")

    fetch_tasks = {
        # Base urban context — always fetched
        "landuse": (lat, lon, fetch_r, {
            "landuse": True,
            "leisure": ["park", "playground", "garden", "recreation_ground"],
        }),
        # Type-specific primary buildings
        "primary": (lat, lon, fetch_r, cfg["primary_tags"]),
        # Type-specific support amenities
        "support": (lat, lon, fetch_r, cfg["support_tags"]),
        # Bus stops
        "bus":     (lat, lon, fetch_r, {"highway": "bus_stop"}),
        # MTR stations — wider radius
        "stations":(lat, lon, 1500,    {"railway": "station"}),
    }

    results = _parallel_fetch(fetch_tasks, wall_timeout=130)
    gc.collect()

    # ── Derive layers ─────────────────────────────────────────────────────────

    # Land use base layers
    landuse_raw      = results["landuse"]
    residential_area = _filter_col(landuse_raw, "landuse", "residential")
    industrial_area  = _filter_col(landuse_raw, "landuse",
                                   ["industrial", "commercial"])
    parks            = _filter_col(landuse_raw, "leisure",
                                   ["park", "garden", "recreation_ground",
                                    "playground"])

    # Support amenities
    support_raw = results["support"]
    schools     = _filter_col(support_raw, "amenity",
                              ["school", "college", "university", "kindergarten"])
    hospitals   = _filter_col(support_raw, "amenity", ["hospital", "clinic"])

    # Primary type-specific buildings — polygons only, filter named or large
    primary_raw = _flatten_index(results["primary"])
    primary_bld = _polys_only(primary_raw)
    if not primary_bld.empty:
        primary_bld["_name"] = _get_name(primary_bld)
        named_mask = primary_bld["_name"].apply(
            lambda x: bool(_ascii_clean(str(x))) if pd.notna(x) else False
        )
        large_mask = primary_bld.geometry.area >= 300
        primary_bld = primary_bld[named_mask | large_mask].copy()
    log.info(f"[context] primary buildings: {len(primary_bld)}")

    # Bus stops
    bus_raw   = _flatten_index(results["bus"])
    bus_stops = bus_raw[
        bus_raw.geometry.geom_type == "Point"
    ].copy() if not bus_raw.empty else _EMPTY.copy()

    # Cluster bus stops to max 6 representative points
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

    # ── MTR station processing ────────────────────────────────────────────────
    stations_raw     = _flatten_index(results["stations"])
    stations_in_view = _EMPTY.copy()
    nearest_station  = None     # WGS84 Point — for walk route
    nearest_stn_name = None

    if not stations_raw.empty:
        s             = stations_raw.copy()
        s["_name"]    = _get_name(s)
        s["_cx"]      = s.geometry.centroid.x
        s["_cy"]      = s.geometry.centroid.y
        s["_dist"]    = s.apply(
            lambda r: Point(r["_cx"], r["_cy"]).distance(site_point), axis=1
        )
        s = s.dropna(subset=["_name"]).sort_values("_dist")

        # Keep stations within 1.5x map half size
        stations_in_view = s[s["_dist"] <= half_size * 1.5].copy()

        if not stations_in_view.empty:
            best = stations_in_view.iloc[0]
            # Convert centroid back to WGS84 for walk routing
            nearest_station = gpd.GeoSeries(
                [Point(best["_cx"], best["_cy"])], crs=3857
            ).to_crs(4326).iloc[0]
            nearest_stn_name = _ascii_clean(str(best["_name"]))

    # ── Walk route computation ────────────────────────────────────────────────
    walk_route_gdf = None
    if show_walk_route and nearest_station is not None:
        log.info(f"[context] Computing walk route to {nearest_stn_name}...")
        walk_route_gdf = _compute_walk_route(lat, lon, nearest_station,
                                              walk_dist=max(int(half_size * 1.8), 1200))

    # ── Place labels ──────────────────────────────────────────────────────────
    # Collect labels from multiple layers in priority order
    label_items = []    # list of (dist_from_site, geom_3857, label_text)
    seen_texts  = set()

    def _harvest_labels(gdf, max_items=20):
        if gdf is None or gdf.empty:
            return
        g        = _flatten_index(gdf)
        g["_lb"] = _get_name(g)
        named    = g.dropna(subset=["_lb"])
        named    = named[named["_lb"].astype(str).str.strip().str.len() > 0]
        for _, row in named.head(max_items).iterrows():
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

    # Harvest in priority order — most relevant first
    for src in [schools, hospitals, parks, primary_bld, support_raw, landuse_raw]:
        _harvest_labels(src)

    label_items.sort(key=lambda x: x[0])
    label_items = [(g, t) for _, g, t in label_items[:30]]

    # Cleanup large objects before render
    del primary_raw, support_raw, landuse_raw, results
    gc.collect()

    log.info(f"[context] Rendering — {len(label_items)} labels, "
             f"walk_route={'yes' if walk_route_gdf is not None else 'no'}")

    # ============================================================
    # FIGURE SETUP
    # CRITICAL: set_xlim / set_ylim BEFORE cx.add_basemap
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

    # Re-lock extent after basemap (contextily may alter limits)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    # ── Base land-use layers ──────────────────────────────────────────────────
    _safe_plot(residential_area, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(industrial_area,  ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks,            ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools,          ax, color="#9ecae1", alpha=0.85, zorder=2)
    _safe_plot(hospitals,        ax, color="#aec6cf", alpha=0.85, zorder=2)

    del residential_area, industrial_area, parks, schools, hospitals
    gc.collect()

    # ── Primary type-specific buildings ──────────────────────────────────────
    _safe_plot(primary_bld, ax, color=hi_color, alpha=0.82, zorder=3)
    del primary_bld
    gc.collect()

    # ── Pedestrian walk route — dashed blue ───────────────────────────────────
    legend_walk_handle = None
    if walk_route_gdf is not None and not walk_route_gdf.empty:
        try:
            walk_route_gdf.plot(
                ax=ax,
                color=ROUTE_COLOR,
                linewidth=2.8,
                linestyle=(0, (6, 4)),   # dashed
                zorder=6,
                alpha=0.90
            )
            legend_walk_handle = Line2D(
                [], [],
                color=ROUTE_COLOR,
                linewidth=2.8,
                linestyle=(0, (6, 4)),
                label="Pedestrian Route to MTR"
            )
            log.info(f"[context] Walk route plotted ({len(walk_route_gdf)} segments)")
        except Exception as e:
            log.warning(f"[context] walk route plot: {e}")

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
                bus_stops.plot(ax=ax, color="#0d47a1",
                               markersize=40, zorder=9, marker="s")
        except Exception as e:
            log.debug(f"[context] bus stop render: {e}")
    del bus_stops
    gc.collect()

    # ── MTR stations ──────────────────────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            # Render station polygon fill
            _safe_plot(stations_in_view, ax,
                       facecolor=MTR_COLOR, edgecolor="none",
                       alpha=0.90, zorder=10)
            # Render MTR logo icon at centroid
            for _, st in stations_in_view.iterrows():
                _draw_mtr_icon(ax, st["_cx"], st["_cy"], zoom=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # ── Site — red fill with SITE label ──────────────────────────────────────
    try:
        sc = site_geom.centroid
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="white",
                      linewidth=2.5, zorder=15)
        ax.text(sc.x, sc.y, "SITE",
                color="white", weight="bold", fontsize=8,
                ha="center", va="center", zorder=16)
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    # ── Place labels ──────────────────────────────────────────────────────────
    placed  = []
    offsets = [(0, 50), (0, -50), (55, 0), (-55, 0), (38, 38), (-38, 38),
               (38, -38), (-38, -38)]

    for i, (geom, text) in enumerate(label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            # Skip labels too close to site centroid or each other
            if p.distance(site_point) < 80:
                continue
            if any(p.distance(pp) < 90 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(
                p.x + dx, p.y + dy,
                _wrap(text, 18),
                fontsize=8.5,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, boxstyle="round,pad=0.3"),
                zorder=13, clip_on=True
            )
            placed.append(p)
        except Exception:
            continue

    # ── MTR station name labels ───────────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            for _, st in stations_in_view.iterrows():
                raw   = st.get("_name")
                label = _ascii_clean(str(raw)) if raw else None
                if not label:
                    continue
                ax.text(
                    st["_cx"], st["_cy"] + 150,
                    _wrap(label, 18),
                    fontsize=9, weight="bold", color="black",
                    ha="center", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.88, pad=2.5),
                    zorder=17
                )
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    # ── Info box ──────────────────────────────────────────────────────────────
    info_lines = [
        f"{data_type}: {value}",
        f"OZP Plan: {primary['PLAN_NO']}",
        f"Zoning: {zone}",
        f"Site Type: {SITE_TYPE}",
    ]
    if nearest_stn_name:
        info_lines.append(f"Nearest MTR: {nearest_stn_name}")

    ax.text(
        0.012, 0.988,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black",
                  linewidth=1.5, pad=6),
        zorder=20,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#f2c6a0",   label="Residential Area"),
        mpatches.Patch(color="#b39ddb",   label="Industrial / Commercial Area"),
        mpatches.Patch(color="#b7dfb9",   label="Public Park"),
        mpatches.Patch(color="#9ecae1",   label="School / Institution"),
        mpatches.Patch(color="#aec6cf",   label="Hospital"),
        mpatches.Patch(color=hi_color, alpha=0.82, label=hi_label),
        mpatches.Patch(color=MTR_COLOR,   label="MTR Station"),
        mpatches.Patch(color="#e53935",   label="Site"),
        mpatches.Patch(color="#0d47a1",   label="Bus Stop"),
    ]

    if legend_walk_handle is not None:
        legend_handles.append(legend_walk_handle)

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        fontsize=8.5,
        framealpha=0.95,
        edgecolor="black",
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=15, weight="bold",
    )
    ax.set_axis_off()

    # Final extent lock
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("[context] Done.")
    return buf
