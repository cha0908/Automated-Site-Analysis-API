"""
context.py — Site-Type Driven Context Analysis
ALKF+ Module — Local Dataset Edition

Data source: local GeoPackage files (no Overpass API calls at runtime)
  data/osm/buildings.gpkg   — building footprints
  data/osm/landuse.gpkg     — landuse + leisure polygons
  data/osm/amenities.gpkg   — amenity, tourism, shop
  data/osm/transport.gpkg   — bus stops + MTR stations

Query strategy:
  1. At module load: read all 4 GeoPackages into memory (EPSG:3857), build STRtree
  2. At request time: bbox clip via STRtree → <50ms per layer, zero network calls
  3. Overpass retained as fallback ONLY if local files are missing

Performance: full context map in <15s total (was 83-295s with Overpass)
"""

import logging
import os
import gc
import threading
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
import numpy as np
import textwrap
import pandas as pd
import networkx as nx
import osmnx as ox

from typing import Optional, List
from shapely.geometry import Point, box
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
import time as _time

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

# ── OSMnx settings (only used in Overpass fallback mode) ─────────────────────
ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 30

# ── Paths ─────────────────────────────────────────────────────────────────────
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
_DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "osm")

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

# ── Map parameters ────────────────────────────────────────────────────────────
FETCH_RADIUS  = 700
MAP_HALF_SIZE = 700
MTR_COLOR     = "#ffd166"
ROUTE_COLOR   = "#1565C0"

# ── Walk network parameters ───────────────────────────────────────────────────
_WALK_GRAPH_DIST    = 1200
_WALK_GRAPH_TIMEOUT = 20
_WALK_ROUTE_TIMEOUT = 8
_WALK_GRAPH_CACHE: dict = {}
_WALK_GRAPH_LOCK        = threading.Lock()


# ============================================================
# LOCAL DATASET CONTAINER
# Loaded once at startup via warm_cache() or on first request.
# Each attribute is a GeoDataFrame in EPSG:3857 with STRtree built.
# ============================================================

class _LocalOSM:
    buildings: gpd.GeoDataFrame = _EMPTY.copy()
    landuse:   gpd.GeoDataFrame = _EMPTY.copy()
    amenities: gpd.GeoDataFrame = _EMPTY.copy()
    transport: gpd.GeoDataFrame = _EMPTY.copy()
    loaded: bool = False

_LOCAL = _LocalOSM()
_LOAD_LOCK = threading.Lock()


def _load_local_datasets() -> None:
    """
    Load all 4 GeoPackages into memory.
    Thread-safe — only runs once even under concurrent first requests.
    Each dataset is reprojected to EPSG:3857 and its STRtree pre-built.
    Estimated memory: ~400-600 MB for all of HK.
    """
    global _LOCAL
    with _LOAD_LOCK:
        if _LOCAL.loaded:
            return  # already loaded by another thread

        files = {
            "buildings": os.path.join(_DATA_DIR, "buildings.gpkg"),
            "landuse":   os.path.join(_DATA_DIR, "landuse.gpkg"),
            "amenities": os.path.join(_DATA_DIR, "amenities.gpkg"),
            "transport": os.path.join(_DATA_DIR, "transport.gpkg"),
        }

        missing = [k for k, p in files.items() if not os.path.exists(p)]
        if missing:
            log.warning(f"[context] LOCAL datasets missing: {missing} "
                        f"— Overpass API fallback will be used")
            return

        t0 = _time.monotonic()
        for attr, path in files.items():
            try:
                gdf = gpd.read_file(path).to_crs(3857)
                gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
                _ = gdf.sindex   # pre-build STRtree index
                setattr(_LOCAL, attr, gdf)
                log.info(f"[context] loaded {attr}: {len(gdf):,} features")
            except Exception as e:
                log.warning(f"[context] failed to load {attr}: {e}")

        _LOCAL.loaded = True
        log.info(f"[context] all local datasets ready in "
                 f"{_time.monotonic() - t0:.1f}s")


# ============================================================
# STRtree BBOX QUERY
# ============================================================

def _bbox_query(dataset: gpd.GeoDataFrame,
                xmin: float, ymin: float,
                xmax: float, ymax: float) -> gpd.GeoDataFrame:
    """
    Spatial bbox filter using pre-built STRtree index.
    Typical time: <1ms for any HK urban area.

    Steps:
      1. STRtree.intersection() returns candidate indices (bbox overlap)
      2. Exact .intersects(query_box) filters to true intersections
    """
    if dataset.empty:
        return _EMPTY.copy()
    try:
        query_box     = box(xmin, ymin, xmax, ymax)
        candidate_idx = list(dataset.sindex.intersection(
            (xmin, ymin, xmax, ymax)))
        if not candidate_idx:
            return _EMPTY.copy()
        candidates = dataset.iloc[candidate_idx].copy()
        return candidates[candidates.geometry.intersects(query_box)].copy()
    except Exception as e:
        log.debug(f"[context] bbox_query: {e}")
        return _EMPTY.copy()


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
# ============================================================

SITE_CONFIGS = {
    "RESIDENTIAL": {
        "building_types": ["apartments", "residential", "house",
                           "dormitory", "detached", "terrace", "block"],
        "amenity_types":  ["school", "college", "university", "kindergarten",
                           "hospital", "clinic", "supermarket", "pharmacy"],
        "leisure_types":  ["park", "playground", "garden", "recreation_ground"],
        "highlight_color": "#e07b39",
        "highlight_label": "Residential Developments",
    },
    "HOTEL": {
        "building_types": ["hotel"],
        "tourism_types":  ["hotel", "hostel", "guest_house",
                           "resort", "aparthotel"],
        "amenity_types":  ["restaurant", "cafe", "bar", "cinema"],
        "leisure_types":  [],
        "highlight_color": "#b15928",
        "highlight_label": "Hotels & Serviced Apartments",
    },
    "COMMERCIAL": {
        "building_types": ["office", "commercial", "retail"],
        "office_types":   ["company", "government", "ngo", "yes"],
        "amenity_types":  ["bank", "restaurant", "cafe", "fast_food"],
        "leisure_types":  [],
        "highlight_color": "#6a3d9a",
        "highlight_label": "Office / Commercial Buildings",
    },
    "INSTITUTIONAL": {
        "building_types": ["government", "civic", "hospital",
                           "school", "university"],
        "amenity_types":  ["school", "college", "university", "hospital",
                           "government", "library", "police", "fire_station"],
        "leisure_types":  ["park", "garden"],
        "highlight_color": "#1f78b4",
        "highlight_label": "Institutional Buildings",
    },
    "INDUSTRIAL": {
        "building_types": ["industrial", "warehouse", "factory",
                           "storage_tank", "shed"],
        "amenity_types":  [],
        "leisure_types":  [],
        "highlight_color": "#33a02c",
        "highlight_label": "Industrial / Warehouse Buildings",
    },
    "OTHER": {
        "building_types": ["yes", "commercial", "residential", "office",
                           "retail", "apartments", "hotel", "industrial",
                           "public", "civic"],
        "amenity_types":  ["school", "hospital", "restaurant", "cafe"],
        "leisure_types":  [],
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
    },
    "MIXED": {
        "building_types": ["yes", "commercial", "residential", "office",
                           "retail", "apartments", "hotel", "industrial",
                           "public", "civic"],
        "amenity_types":  ["school", "hospital", "restaurant", "cafe"],
        "leisure_types":  ["park", "playground", "garden"],
        "highlight_color": "#aaaaaa",
        "highlight_label": "Nearby Buildings",
    },
}


# ============================================================
# LOCAL SPATIAL FETCH  (<100ms total)
# ============================================================

def _fetch_local(site_point: Point,
                 fetch_r: float,
                 cfg: dict) -> dict:
    """
    Query all 4 local datasets via STRtree bbox index.
    No network I/O. Total time: <100ms for any HK site.
    """
    t0 = _time.monotonic()

    # Compute bbox in EPSG:3857 metres
    xmin = site_point.x - fetch_r;  xmax = site_point.x + fetch_r
    ymin = site_point.y - fetch_r;  ymax = site_point.y + fetch_r
    stn_r = 1500   # wider radius for MTR stations
    sxmin = site_point.x - stn_r;   sxmax = site_point.x + stn_r
    symin = site_point.y - stn_r;   symax = site_point.y + stn_r

    # ── Landuse + leisure ──────────────────────────────────────────────────
    lu_all = _bbox_query(_LOCAL.landuse, xmin, ymin, xmax, ymax)
    residential_area = _EMPTY.copy()
    industrial_area  = _EMPTY.copy()
    parks            = _EMPTY.copy()
    if not lu_all.empty:
        if "landuse" in lu_all.columns:
            residential_area = lu_all[lu_all["landuse"] == "residential"].copy()
            industrial_area  = lu_all[
                lu_all["landuse"].isin(["industrial", "commercial"])].copy()
        if "leisure" in lu_all.columns:
            parks = lu_all[lu_all["leisure"].isin(
                ["park", "garden", "recreation_ground", "playground"])].copy()

    # ── Primary buildings (site-type specific) ─────────────────────────────
    bld_all  = _bbox_query(_LOCAL.buildings, xmin, ymin, xmax, ymax)
    btypes   = cfg.get("building_types", [])
    primary_bld = _EMPTY.copy()
    if not bld_all.empty and btypes:
        b_col  = bld_all.get("building", pd.Series(dtype=str))
        t_mask = b_col.isin(btypes)
        # Extra tag masks for hotel (tourism) and commercial (office) types
        extra  = pd.Series(False, index=bld_all.index)
        if "tourism_types" in cfg and "tourism" in bld_all.columns:
            extra |= bld_all["tourism"].isin(cfg["tourism_types"])
        if "office_types" in cfg and "office" in bld_all.columns:
            extra |= bld_all["office"].isin(cfg["office_types"])
        matched = bld_all[t_mask | extra].copy()
        # Keep polygons only; named or >= 200m²
        matched = matched[
            matched.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if not matched.empty:
            name_col = matched.get("name", pd.Series(dtype=str)).fillna("")
            has_name = name_col.str.strip().str.len() > 0
            large    = matched.geometry.area >= 200
            primary_bld = matched[has_name | large].copy()

    # ── Amenities (schools, hospitals, support) ────────────────────────────
    amen_all  = _bbox_query(_LOCAL.amenities, xmin, ymin, xmax, ymax)
    schools   = _EMPTY.copy()
    hospitals = _EMPTY.copy()
    support   = _EMPTY.copy()
    if not amen_all.empty:
        atypes = cfg.get("amenity_types", [])
        ttypes = cfg.get("tourism_types", [])

        if "amenity" in amen_all.columns:
            schools   = amen_all[amen_all["amenity"].isin(
                ["school", "college", "university", "kindergarten"])].copy()
            hospitals = amen_all[amen_all["amenity"].isin(
                ["hospital", "clinic"])].copy()

        if atypes or ttypes:
            a_mask = pd.Series(False, index=amen_all.index)
            if atypes and "amenity" in amen_all.columns:
                a_mask |= amen_all["amenity"].isin(atypes)
            if ttypes and "tourism" in amen_all.columns:
                a_mask |= amen_all["tourism"].isin(ttypes)
            support = amen_all[a_mask].copy()

    # ── Bus stops ──────────────────────────────────────────────────────────
    trans_s   = _bbox_query(_LOCAL.transport, xmin, ymin, xmax, ymax)
    bus_stops = _EMPTY.copy()
    if not trans_s.empty and "highway" in trans_s.columns:
        bus_stops = trans_s[
            (trans_s["highway"] == "bus_stop") &
            (trans_s.geometry.geom_type == "Point")
        ].copy()

    # ── MTR stations (wider bbox) ──────────────────────────────────────────
    trans_l  = _bbox_query(_LOCAL.transport, sxmin, symin, sxmax, symax)
    stations = _EMPTY.copy()
    if not trans_l.empty and "railway" in trans_l.columns:
        stations = trans_l[
            trans_l["railway"].isin(["station", "halt"])
        ].copy()

    log.info(f"[context] local fetch {(_time.monotonic()-t0)*1000:.0f}ms — "
             f"bld={len(primary_bld)} amen={len(support)} "
             f"bus={len(bus_stops)} stn={len(stations)}")

    return {
        "primary_bld":      primary_bld,
        "support":          support,
        "schools":          schools,
        "hospitals":        hospitals,
        "parks":            parks,
        "residential_area": residential_area,
        "industrial_area":  industrial_area,
        "bus_stops":        bus_stops,
        "stations":         stations,
        "landuse_all":      lu_all,
    }


# ============================================================
# OVERPASS FALLBACK  (used only when local files missing)
# ============================================================

def _fetch_overpass_fallback(lat: float, lon: float,
                              fetch_r: float, cfg: dict) -> dict:
    import hashlib, json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    log.warning("[context] Using Overpass fallback "
                "(run prepare_osm_data.py to enable local mode)")

    OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    def _bbox(la, lo, d):
        return ox.utils_geo.bbox_from_point((la, lo), dist=d)

    def _fetch(bbox, tags):
        for ep in OVERPASS_ENDPOINTS:
            try:
                ox.settings.overpass_endpoint = ep
                gdf = ox.features_from_bbox(bbox, tags=tags)
                if gdf is not None and not gdf.empty:
                    return gdf.to_crs(3857)
            except Exception:
                continue
        return _EMPTY.copy()

    bbox_s = _bbox(lat, lon, fetch_r)
    bbox_l = _bbox(lat, lon, 1500)

    btypes = cfg.get("building_types", [])
    atypes = cfg.get("amenity_types",  [])

    task_defs = {
        "landuse":  (bbox_s, {"landuse": True,
                               "leisure": ["park","playground",
                                           "garden","recreation_ground"]}),
        "primary":  (bbox_s, {"building": btypes or True}),
        "support":  (bbox_s, {"amenity": atypes or True}),
        "bus":      (bbox_s, {"highway": "bus_stop"}),
        "stations": (bbox_l, {"railway": "station"}),
    }
    raw = {k: _EMPTY.copy() for k in task_defs}
    with ThreadPoolExecutor(max_workers=5) as pool:
        fmap = {pool.submit(_fetch, b, t): n
                for n, (b, t) in task_defs.items()}
        for f in as_completed(fmap, timeout=60):
            n = fmap[f]
            try:
                raw[n] = f.result()
                log.info(f"[context] overpass ✓ {n}: {len(raw[n])} rows")
            except Exception as e:
                log.warning(f"[context] overpass ✗ {n}: {e}")

    def _fc(gdf, col, vals):
        if gdf.empty or col not in gdf.columns:
            return _EMPTY.copy()
        s = gdf[col]
        return gdf[s.isin(vals) if isinstance(vals, list)
                   else (s == vals)].copy()

    lu = raw["landuse"]
    sp = raw["support"]
    return {
        "primary_bld":      raw["primary"],
        "support":          sp,
        "schools":          _fc(sp, "amenity",
                                ["school","college","university","kindergarten"]),
        "hospitals":        _fc(sp, "amenity", ["hospital","clinic"]),
        "parks":            _fc(lu, "leisure",
                                ["park","garden","recreation_ground","playground"]),
        "residential_area": _fc(lu, "landuse", "residential"),
        "industrial_area":  _fc(lu, "landuse", ["industrial","commercial"]),
        "bus_stops":        raw["bus"],
        "stations":         raw["stations"],
        "landuse_all":      lu,
    }


# ============================================================
# HELPERS
# ============================================================

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


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


def _get_name(gdf: gpd.GeoDataFrame) -> pd.Series:
    en   = gdf["name:en"] if "name:en" in gdf.columns else pd.Series(dtype=str)
    base = gdf["name"]    if "name"    in gdf.columns else pd.Series(dtype=str)
    return en.fillna(base)


def _safe_plot(gdf, ax, **kwargs):
    try:
        if not gdf.empty:
            gdf.plot(ax=ax, **kwargs)
    except Exception as e:
        log.debug(f"[context] plot: {e}")


def _polys_only(gdf):
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])].copy()


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
# WALK GRAPH — cached, timeout-guarded
# ============================================================

def _get_walk_graph(lat, lon):
    ck = (round(lat, 2), round(lon, 2))
    with _WALK_GRAPH_LOCK:
        if ck in _WALK_GRAPH_CACHE:
            return _WALK_GRAPH_CACHE[ck]

    result = [None]; error = [None]
    def _build():
        try:
            result[0] = ox.graph_from_point(
                (lat, lon), dist=_WALK_GRAPH_DIST,
                network_type="walk", simplify=True)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_build, daemon=True)
    t.start(); t.join(timeout=_WALK_GRAPH_TIMEOUT)
    if t.is_alive():
        log.warning("[context] walk graph timed out")
        return None
    if error[0]:
        log.warning(f"[context] walk graph error: {error[0]}")
        return None
    G = result[0]
    if not G or G.number_of_nodes() == 0:
        return None
    with _WALK_GRAPH_LOCK:
        _WALK_GRAPH_CACHE[ck] = G
    log.info(f"[context] walk graph: "
             f"{G.number_of_nodes()}n/{G.number_of_edges()}e")
    return G


def _compute_walk_route(lat, lon, station_wgs84: Point):
    G = _get_walk_graph(lat, lon)
    if G is None:
        return None
    try:
        sn = ox.distance.nearest_nodes(G, lon, lat)
        tn = ox.distance.nearest_nodes(G, station_wgs84.x, station_wgs84.y)
        if sn == tn:
            return None
    except Exception as e:
        log.warning(f"[context] node snap: {e}")
        return None

    res = [None]; err = [None]
    def _find():
        try:
            path = nx.shortest_path(G, sn, tn, weight="length")
            res[0] = ox.routing.route_to_gdf(G, path).to_crs(3857)
        except Exception as e:
            err[0] = e

    t = threading.Thread(target=_find, daemon=True)
    t.start(); t.join(timeout=_WALK_ROUTE_TIMEOUT)
    if t.is_alive():
        log.warning("[context] walk route timed out")
        return None
    if err[0]:
        log.warning(f"[context] walk route: {err[0]}")
        return None
    return res[0]


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

    # Lazy load on first call if warm_cache wasn't called at startup
    if not _LOCAL.loaded:
        _load_local_datasets()

    lon, lat  = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r   = radius_m if radius_m is not None else FETCH_RADIUS
    half_size = radius_m if radius_m is not None else MAP_HALF_SIZE
    half_x    = half_size * (16 / 12)
    half_y    = half_size

    log.info(f"[context] lon={lon:.5f} lat={lat:.5f} r={fetch_r}m "
             f"[{'local' if _LOCAL.loaded else 'overpass'}]")

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
            [Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
        site_geom = site_point.buffer(80)
        site_gdf  = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)
        log.info("[context] using 80m buffer fallback")

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp  = ZONE_DATA.to_crs(3857)
    hits = ozp[ozp.contains(site_point)]
    if hits.empty:
        raise ValueError("No OZP zoning polygon found for this site.")
    zone_row  = hits.iloc[0]
    zone      = zone_row["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    cfg       = SITE_CONFIGS.get(SITE_TYPE, SITE_CONFIGS["MIXED"])
    hi_color  = cfg["highlight_color"]
    hi_label  = cfg["highlight_label"]
    log.info(f"[context] zone={zone} SITE_TYPE={SITE_TYPE}")

    # ── Fetch layers ──────────────────────────────────────────────────────────
    if _LOCAL.loaded:
        layers = _fetch_local(site_point, fetch_r, cfg)
    else:
        layers = _fetch_overpass_fallback(lat, lon, fetch_r, cfg)

    primary_bld      = layers["primary_bld"]
    support          = layers["support"]
    schools          = layers["schools"]
    hospitals        = layers["hospitals"]
    parks            = layers["parks"]
    residential_area = layers["residential_area"]
    industrial_area  = layers["industrial_area"]
    bus_stops        = layers["bus_stops"]
    stations_raw     = layers["stations"]
    landuse_all      = layers["landuse_all"]
    gc.collect()

    log.info(f"[context] primary buildings: {len(primary_bld)}")

    # ── Bus stop clustering ───────────────────────────────────────────────────
    if len(bus_stops) > 8:
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[g.x, g.y] for g in bus_stops.geometry])
            k = min(8, len(bus_stops))
            labels = KMeans(
                n_clusters=k, random_state=0, n_init=5
            ).fit(coords).labels_
            bus_stops["_cluster"] = labels
            bus_stops = gpd.GeoDataFrame(
                bus_stops.groupby("_cluster").first(), crs=3857
            ).reset_index(drop=True)
        except Exception:
            bus_stops = bus_stops.head(8)

    # ── MTR stations ──────────────────────────────────────────────────────────
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
        stations_in_view = s[s["_dist"] <= half_size * 1.8].copy()

        if not stations_in_view.empty:
            best = stations_in_view.iloc[0]
            nearest_station = gpd.GeoSeries(
                [Point(best["_cx"], best["_cy"])], crs=3857
            ).to_crs(4326).iloc[0]
            nearest_stn_name = _ascii_clean(str(best["_name"]))
            log.info(f"[context] nearest MTR: {nearest_stn_name} "
                     f"({best['_dist']:.0f}m)")

    # ── Walk route ────────────────────────────────────────────────────────────
    walk_route_gdf = None
    if show_walk_route and nearest_station is not None:
        log.info(f"[context] walk route → {nearest_stn_name}...")
        walk_route_gdf = _compute_walk_route(lat, lon, nearest_station)
        log.info(f"[context] walk route: "
                 f"{'yes' if walk_route_gdf is not None else 'unavailable'}")

    # ── Place labels ──────────────────────────────────────────────────────────
    label_items = []
    seen_texts  = set()

    def _harvest(gdf, cap=20):
        if gdf is None or gdf.empty:
            return
        gdf = gdf.copy()
        gdf["_lb"] = _get_name(gdf)
        named = gdf.dropna(subset=["_lb"])
        named = named[named["_lb"].astype(str).str.strip().str.len() > 0]
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

    for src in [schools, hospitals, parks, primary_bld,
                support, landuse_all]:
        _harvest(src)

    label_items.sort(key=lambda x: x[0])
    label_items = [(g, t) for _, g, t in label_items[:30]]
    del layers, landuse_all, support
    gc.collect()

    log.info(f"[context] Rendering — {len(label_items)} labels, "
             f"walk={'yes' if walk_route_gdf is not None else 'no'}")

    # ============================================================
    # FIGURE
    # ============================================================

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    xmin = site_point.x - half_x;  xmax = site_point.x + half_x
    ymin = site_point.y - half_y;  ymax = site_point.y + half_y

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.autoscale(False)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=16, alpha=0.92)
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                           zoom=16, alpha=0.92)
        except Exception as e:
            log.warning(f"[context] basemap: {e}")

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.autoscale(False)

    # zorder 1: landuse fills
    _safe_plot(residential_area, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(industrial_area,  ax, color="#b39ddb", alpha=0.75, zorder=1)
    # zorder 2: parks / schools / hospitals
    _safe_plot(parks,            ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools,          ax, color="#9ecae1", alpha=0.85, zorder=2)
    _safe_plot(hospitals,        ax, color="#aec6cf", alpha=0.85, zorder=2)
    del residential_area, industrial_area, parks, schools, hospitals
    gc.collect()

    # zorder 3: primary highlighted buildings
    _safe_plot(primary_bld, ax, color=hi_color, alpha=0.82, zorder=3)
    del primary_bld; gc.collect()

    # zorder 6: walk route
    legend_walk = None
    if walk_route_gdf is not None and not walk_route_gdf.empty:
        try:
            walk_route_gdf.plot(
                ax=ax, color=ROUTE_COLOR, linewidth=3.0,
                linestyle=(0, (6, 4)), zorder=6, alpha=0.92)
            legend_walk = Line2D(
                [], [], color=ROUTE_COLOR, linewidth=3.0,
                linestyle=(0, (6, 4)), label="Pedestrian Route to MTR")
            log.info(f"[context] walk route: {len(walk_route_gdf)} segments")
        except Exception as e:
            log.warning(f"[context] walk route plot: {e}")

    # zorder 9: bus stops
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
    del bus_stops; gc.collect()

    # zorder 10 / 14: MTR stations + logos
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

    # zorder 15 / 16: site polygon + label
    try:
        sc = site_geom.centroid
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="white",
                      linewidth=2.5, zorder=15, alpha=0.92)
        ax.text(sc.x, sc.y, "SITE", color="white", weight="bold",
                fontsize=8.5, ha="center", va="center", zorder=16)
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    # zorder 13: place labels
    placed  = []
    offsets = [(0,55),(0,-55),(60,0),(-60,0),
               (42,42),(-42,42),(42,-42),(-42,-42)]

    for i, (geom, text) in enumerate(label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_point) < 85:
                continue
            if any(p.distance(pp) < 95 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x+dx, p.y+dy, _wrap(text, 18),
                    fontsize=8.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.3"),
                    zorder=13, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # zorder 17: station name labels
    if not stations_in_view.empty:
        try:
            for _, st in stations_in_view.iterrows():
                label = _ascii_clean(str(st.get("_name", "")))
                if not label:
                    continue
                ax.text(st["_cx"], st["_cy"]+160, _wrap(label, 20),
                        fontsize=9.5, weight="bold", color="#1a1a1a",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="#cccccc",
                                  linewidth=0.8, alpha=0.92,
                                  boxstyle="round,pad=3"),
                        zorder=17)
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    # zorder 20: info box
    info_lines = [
        f"{data_type}: {value}",
        f"OZP Plan: {zone_row['PLAN_NO']}",
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

    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(0.02, 0.02),
              fontsize=8.5, framealpha=0.96, edgecolor="black", fancybox=False)

    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=15, weight="bold", pad=10)
    ax.set_axis_off()
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.autoscale(False)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    buf.seek(0)

    log.info(f"[context] Done in {_time.monotonic()-t0:.1f}s")
    return buf


# ============================================================
# STARTUP
# ============================================================

def warm_cache(ZONE_DATA: gpd.GeoDataFrame) -> None:
    """
    Load local OSM datasets at startup.
    Fast path: local GeoPackages → <10s, zero Overpass calls ever.
    Fallback: if files missing, Overpass used at request time.
    """
    _load_local_datasets()
    log.info(f"[context] warm_cache done "
             f"[{'local' if _LOCAL.loaded else 'overpass fallback'}]")
