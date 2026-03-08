"""
context.py — Site-Type-Driven Context Analysis with Walking Routes
====================================================================
Enhanced version incorporating:
- HK Government API for location resolution
- Walking routes to MTR stations
- Building footprints
- Bus stop clustering

Architecture:
  - Raw HTTP POST to Overpass (GeoJSON output)
  - Parallel thread fetches with timeout
  - osmnx for routing to MTR stations
  - Module-level result cache
"""

import gc
import hashlib
import io
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
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests as _requests
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pyproj import Transformer
from shapely.geometry import Point
from sklearn.cluster import KMeans

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)
ox.settings.use_cache = True
ox.settings.log_console = False

# ── Overpass endpoints (tried in order) ────────────────────────────────────
_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

_PER_FETCH_TIMEOUT = 15
_FETCH_WALL_TIMEOUT = 20

MAP_HALF_SIZE = 900
MTR_COLOR = "#ffd166"

# ── Static assets ──────────────────────────────────────────────────────────
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")

try:
    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
except Exception:
    _bus_icon = None

try:
    _mtr_logo = mpimg.imread(_MTR_LOGO_PATH)
    log.info("[context] MTR logo loaded")
except Exception:
    _mtr_logo = None
    log.warning("[context] MTR logo missing — fallback marker will be used")

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=4326)
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


# ── HK Government API Resolver (from Colab) ───────────────────────────────

def resolve_hk_location(data_type: str, value: str):
    """
    Resolve HK location using government Geodata API.
    Based on user's Colab code.
    """
    base_url = "https://mapapi.geodata.gov.hk/gs/api/v1.0.0"
    data_type = data_type.upper().strip()

    try:
        if data_type == "LOT":
            url = f"{base_url}/lus/lot/SearchNumber?text={value.replace(' ', '%20')}"
        else:
            url = f"{base_url}/lus/{data_type.lower()}/Search?value={value}"

        r = _requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        if "candidates" not in data or len(data["candidates"]) == 0:
            return None

        best = max(data["candidates"], key=lambda x: x["score"])

        x2326 = best["location"]["x"]
        y2326 = best["location"]["y"]

        lon, lat = Transformer.from_crs(
            2326, 4326, always_xy=True
        ).transform(x2326, y2326)

        return lon, lat
    except Exception as e:
        log.warning(f"[context] HK resolver failed: {e}")
        return None


# ── OZP site type ──────────────────────────────────────────────────────────

def _infer_site_type(zone_label: str) -> str:
    z = str(zone_label).upper().strip()
    if z.startswith("R"):
        return "RESIDENTIAL"
    if z.startswith("C"):
        return "COMMERCIAL"
    if z.startswith("G"):
        return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"):
        return "HOTEL"
    if z.startswith("I"):
        return "INDUSTRIAL"
    if z.startswith("OU"):
        return "OTHER"
    return "MIXED"


def _context_rules(site_type: str) -> dict:
    """Get OSM tags based on site type (from Colab)."""
    if site_type == "RESIDENTIAL":
        return {"amenity": ["school", "college", "university"], "leisure": ["park"], "place": ["neighbourhood"]}
    if site_type == "COMMERCIAL":
        return {"amenity": ["bank", "restaurant", "market"], "railway": ["station"]}
    if site_type == "INSTITUTIONAL":
        return {"amenity": ["school", "college", "hospital"], "leisure": ["park"]}
    return {"amenity": True, "leisure": True}


# ── Overpass QL builders ───────────────────────────────────────────────────

def _ql_union(tag_dict: dict, bbox_str: str) -> str:
    parts = []
    for key, vals in tag_dict.items():
        if isinstance(vals, list):
            for v in vals:
                parts.append(f'nwr["{key}"="{v}"]({bbox_str});')
        elif vals is True:
            parts.append(f'nwr["{key}"]({bbox_str});')
        else:
            parts.append(f'nwr["{key}"="{vals}"]({bbox_str});')
    return "\n".join(parts)


def _build_query(tag_dict: dict, south, west, north, east) -> str:
    bbox_str = f"{south},{west},{north},{east}"
    union = _ql_union(tag_dict, bbox_str)
    return f"""
[out:json][timeout:12];
(
{union}
);
out body geom qt;
""".strip()


# ── Raw Overpass fetch ─────────────────────────────────────────────────────

def _fetch_overpass(tag_dict: dict, south, west, north, east,
                    name="?") -> gpd.GeoDataFrame:
    ck = hashlib.md5(
        json.dumps({"tags": tag_dict,
                    "bbox": (round(south, 4), round(west, 4),
                             round(north, 4), round(east, 4))},
                   sort_keys=True).encode()
    ).hexdigest()

    with _CACHE_LOCK:
        if ck in _CACHE:
            log.info(f"[context] cache hit: {name}")
            return _CACHE[ck]

    query = _build_query(tag_dict, south, west, north, east)

    result = [_EMPTY.copy()]
    completed = [False]

    def _do():
        for ep in _ENDPOINTS:
            try:
                resp = _requests.post(
                    ep, data={"data": query},
                    timeout=_PER_FETCH_TIMEOUT,
                    headers={"Accept-Encoding": "gzip"}
                )
                if resp.status_code != 200:
                    log.warning(f"[context] {name}@{ep.split('/')[2]}: HTTP {resp.status_code}")
                    continue

                osm_json = resp.json()
                elements = osm_json.get("elements", [])
                if not elements:
                    log.info(f"[context] {name}: 0 elements from {ep.split('/')[2]}")
                    completed[0] = True
                    with _CACHE_LOCK:
                        _CACHE[ck] = result[0]
                    return

                features = []
                for el in elements:
                    geom = _el_to_geom(el)
                    if geom is None:
                        continue
                    props = {k: v for k, v in el.items()
                             if k not in ("geometry", "nodes", "members", "bounds")}
                    props.update(el.get("tags", {}))
                    features.append({"type": "Feature",
                                     "geometry": geom,
                                     "properties": props})

                if features:
                    fc = {"type": "FeatureCollection", "features": features}
                    gdf = gpd.read_file(io.StringIO(json.dumps(fc)))
                    gdf = gdf.set_crs(4326, allow_override=True)
                    result[0] = gdf
                    log.info(f"[context] ✓ {name}: {len(gdf)} rows [{ep.split('/')[2]}]")

                completed[0] = True
                with _CACHE_LOCK:
                    _CACHE[ck] = result[0]
                return

            except Exception as e:
                log.warning(f"[context] {name}@{ep.split('/')[2]}: {type(e).__name__}: {e}")
                continue

        completed[0] = True
        with _CACHE_LOCK:
            _CACHE[ck] = result[0]

    t = threading.Thread(target=_do, daemon=True, name=f"ctx-{name}")
    t.start()
    t.join(timeout=_PER_FETCH_TIMEOUT + 2)
    if not completed[0]:
        log.warning(f"[context] hard timeout: {name}")
    return result[0]


def _el_to_geom(el: dict) -> Optional[dict]:
    """Convert a raw Overpass JSON element to a GeoJSON geometry dict."""
    t = el.get("type")
    try:
        if t == "node":
            lon = el.get("lon")
            lat = el.get("lat")
            if lon is None or lat is None:
                return None
            return {"type": "Point", "coordinates": [lon, lat]}

        if t in ("way", "relation"):
            geom = el.get("geometry")
            if not geom:
                return None
            coords = [[p["lon"], p["lat"]] for p in geom
                      if "lon" in p and "lat" in p]
            if len(coords) < 2:
                return None
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            if len(coords) >= 4:
                return {"type": "Polygon", "coordinates": [coords]}
            return {"type": "LineString", "coordinates": coords}
    except Exception:
        pass
    return None


# ── Tag dicts ──────────────────────────────────────────────────────────────

def _landuse_tags() -> dict:
    return {
        "landuse": ["residential", "commercial", "industrial", "retail", "warehouse"],
        "leisure": ["park", "playground", "garden", "recreation_ground"],
    }


def _amenity_tags(site_type: str) -> dict:
    base = {
        "amenity": ["school", "college", "university", "hospital",
                    "clinic", "bank", "library", "government",
                    "post_office", "community_centre"]
    }
    if site_type in ("HOTEL", "COMMERCIAL", "MIXED", "OTHER"):
        base["amenity"] += ["restaurant", "cafe", "theatre", "cinema"]
    if site_type == "HOTEL":
        base["tourism"] = ["hotel", "hostel", "guest_house", "attraction", "museum", "gallery"]
    if site_type in ("COMMERCIAL", "MIXED"):
        base["shop"] = ["mall", "department_store"]
    return base


# ── Highlight + label config ───────────────────────────────────────────────

_HI = {
    "RESIDENTIAL": ("#e07b39", "Residential Zone"),
    "COMMERCIAL": ("#6a3d9a", "Commercial / Office Zone"),
    "HOTEL": ("#b15928", "Hotel / Tourism Zone"),
    "INSTITUTIONAL": ("#1f78b4", "Institutional Zone"),
    "INDUSTRIAL": ("#33a02c", "Industrial Zone"),
    "OTHER": ("#888888", "Other / Mixed Zone"),
    "MIXED": ("#888888", "Mixed Use Zone"),
}

_LABEL_AM = {
    "RESIDENTIAL": ["school", "college", "university", "hospital", "library", "community_centre"],
    "COMMERCIAL": ["bank", "government", "library", "post_office"],
    "HOTEL": ["theatre", "cinema", "museum"],
    "INSTITUTIONAL": ["school", "college", "university", "hospital", "library", "government"],
    "INDUSTRIAL": [],
    "OTHER": ["school", "hospital", "bank", "library"],
    "MIXED": ["school", "hospital", "bank", "library"],
}
_LABEL_LE = {k: ["park", "garden", "recreation_ground"] for k in _HI}
_LABEL_LE["INDUSTRIAL"] = []


# ── Helpers ────────────────────────────────────────────────────────────────

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _col(gdf, col):
    return gdf[col] if col in gdf.columns else pd.Series([None] * len(gdf), index=gdf.index)


def _filter(gdf, col, vals):
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    mask = gdf[col].isin(vals) if isinstance(vals, list) else (gdf[col] == vals)
    return gdf[mask].copy()


def _polys(gdf):
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def _get_name(gdf):
    return _col(gdf, "name:en").fillna(_col(gdf, "name"))


def _ascii(text) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    ratio = sum(1 for c in s if ord(c) < 128) / max(len(s), 1)
    if ratio < 0.5:
        return None
    out = "".join(c for c in s if ord(c) < 128).strip()
    return out or None


def _safe_plot(gdf, ax, **kw):
    try:
        if not gdf.empty:
            gdf.to_crs(3857).plot(ax=ax, **kw)
    except Exception as e:
        log.debug(f"[context] plot: {e}")


def _mtr_icon(ax, x, y, zoom=0.042, zorder=14):
    if _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ax.add_artist(AnnotationBbox(icon, (x, y), frameon=False, zorder=zorder, box_alignment=(0.5, 0.5)))
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=18, markeredgecolor="white", markeredgewidth=2.5, zorder=zorder)


# ── Main ───────────────────────────────────────────────────────────────────

def generate_context(
    data_type: str,
    value: str,
    ZONE_DATA: gpd.GeoDataFrame,
    radius_m: Optional[int] = None,
    lon: float = None,
    lat: float = None,
    lot_ids: List[str] = None,
    extents: List[dict] = None,
    use_hk_api: bool = False,
) -> BytesIO:
    """
    Generate context analysis map with walking routes to MTR.

    Args:
        data_type: Input type (LOT, ADDRESS, etc.)
        value: Input value
        ZONE_DATA: OZP zoning GeoDataFrame
        radius_m: Map radius in meters
        lon, lat: Direct coordinates
        lot_ids, extents: For batch processing
        use_hk_api: Use HK Government API for resolution

    Returns:
        BytesIO: PNG image buffer
    """
    t0 = _time.monotonic()

    # Try HK API first if requested
    if use_hk_api:
        hk_coords = resolve_hk_location(data_type, value)
        if hk_coords:
            lon, lat = hk_coords
            log.info(f"[context] HK API resolved: lon={lon:.5f} lat={lat:.5f}")

    # Fall back to standard resolver
    if lon is None or lat is None:
        lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)

    half = radius_m if radius_m is not None else MAP_HALF_SIZE
    half_x = half * (16 / 12)
    half_y = half
    fetch_r = half

    log.info(f"[context] lon={lon:.5f} lat={lat:.5f} r={fetch_r}m")

    # 1. Site point (3857)
    site_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    # 2. Site polygon
    site_geom = None
    try:
        lb = get_lot_boundary(lon, lat, data_type, extents)
        if lb is not None and not lb.empty:
            cand = lb.to_crs(3857).geometry.iloc[0]
            if cand.area > 50:
                site_geom = cand
                log.info(f"[context] lot boundary area={cand.area:.0f}m²")
    except Exception as e:
        log.warning(f"[context] lot boundary: {e}")
    if site_geom is None:
        site_geom = site_pt.buffer(40)
        log.info("[context] 40m buffer fallback")
    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # 3. OZP zone → site type
    ozp = ZONE_DATA.to_crs(3857)
    hits = ozp[ozp.contains(site_pt)]
    if hits.empty:
        tmp = ozp.copy()
        tmp["_d"] = tmp.geometry.distance(site_pt)
        primary = tmp.sort_values("_d").iloc[0]
        log.warning("[context] nearest zone fallback")
    else:
        primary = hits.iloc[0]

    zone = str(primary.get("ZONE_LABEL", "MIXED"))
    plan_no = str(primary.get("PLAN_NO", "N/A"))
    SITE_TYPE = _infer_site_type(zone)
    LABEL_RULES = _context_rules(SITE_TYPE)
    hi_color, hi_label = _HI.get(SITE_TYPE, ("#888", "Zone"))
    log.info(f"[context] OZP zone={zone} SITE_TYPE={SITE_TYPE}")

    # 4. Compute bbox extents
    n, s, e, w = ox.utils_geo.bbox_from_point((lat, lon), dist=fetch_r)
    n2, s2, e2, w2 = ox.utils_geo.bbox_from_point((lat, lon), dist=2000)

    # 5. Parallel raw Overpass fetches
    tasks = {
        "landuse": ({"tags": _landuse_tags(), "bbox": (s, w, n, e)}),
        "amenity": ({"tags": _amenity_tags(SITE_TYPE), "bbox": (s, w, n, e)}),
        "bus": ({"tags": {"highway": "bus_stop"}, "bbox": (s, w, n, e)}),
        "stations": ({"tags": {"railway": "station"}, "bbox": (s2, w2, n2, e2)}),
    }

    fetched = {}
    wall_deadline = _time.monotonic() + _FETCH_WALL_TIMEOUT

    with ThreadPoolExecutor(max_workers=4) as pool:
        fmap = {
            pool.submit(
                _fetch_overpass,
                v["tags"], *v["bbox"], k
            ): k
            for k, v in tasks.items()
        }
        remaining = max(0.5, wall_deadline - _time.monotonic())
        for future in as_completed(fmap, timeout=remaining):
            k = fmap[future]
            try:
                fetched[k] = future.result()
            except Exception as e:
                log.warning(f"[context] future {k}: {e}")
                fetched[k] = _EMPTY.copy()

    for k in tasks:
        if k not in fetched:
            fetched[k] = _EMPTY.copy()

    gc.collect()

    # 6. MTR stations
    stations_list = []
    nearest_name = None
    st_raw = fetched["stations"]
    if not st_raw.empty:
        try:
            st3 = st_raw.to_crs(3857).copy()
            st3["_name"] = _get_name(st3)
            st3["_cx"] = st3.geometry.centroid.x
            st3["_cy"] = st3.geometry.centroid.y
            st3["_dist"] = st3.apply(
                lambda r: Point(r["_cx"], r["_cy"]).distance(site_pt), axis=1)
            st3 = st3.dropna(subset=["_name"]).sort_values("_dist")
            nearby = st3[st3["_dist"] <= half * 2.2].head(3)
            for _, row in nearby.iterrows():
                nm = _ascii(str(row["_name"]))
                if nm:
                    stations_list.append((row["_cx"], row["_cy"], nm))
            if stations_list:
                nearest_name = stations_list[0][2]
                log.info(f"[context] MTR: {[s[2] for s in stations_list]}")
        except Exception as e:
            log.warning(f"[context] stations parse: {e}")

    # 7. Landuse layers
    lu = fetched["landuse"]
    res = _polys(_filter(lu, "landuse", ["residential"]))
    ind = _polys(_filter(lu, "landuse", ["industrial", "commercial", "warehouse"]))
    parks = _polys(_filter(lu, "leisure", ["park", "garden", "recreation_ground", "playground"]))

    # 8. Amenity layers
    am = fetched["amenity"]
    schools = _filter(am, "amenity", ["school", "college", "university", "kindergarten"])
    hosps = _filter(am, "amenity", ["hospital", "clinic"])

    # 9. Bus stops (clustered)
    bus_raw = fetched["bus"]
    bus_stops = (bus_raw[bus_raw.geometry.geom_type == "Point"].copy()
                 if not bus_raw.empty else _EMPTY.copy())
    if not bus_stops.empty:
        bus_stops = bus_stops.to_crs(3857)
    if len(bus_stops) > 6:
        try:
            coords = np.array([[g.x, g.y] for g in bus_stops.geometry])
            bus_stops["cluster"] = KMeans(n_clusters=6, random_state=0).fit(coords).labels_
            bus_stops = bus_stops.groupby("cluster").first().reset_index(drop=True)
        except Exception:
            bus_stops = bus_stops.head(6)

    # 10. Building footprints (from Colab)
    buildings = _EMPTY.copy()
    try:
        building_tags = {"building": True}
        building_raw = _fetch_overpass(building_tags, s, w, n, e, "buildings")
        if not building_raw.empty:
            buildings = _polys(building_raw)
            log.info(f"[context] buildings: {len(buildings)} polygons")
    except Exception as e:
        log.warning(f"[context] buildings fetch: {e}")

    # 11. Walking routes to MTR (from Colab)
    routes = []
    if not stations.empty:
        try:
            G = ox.graph_from_point((lat, lon), dist=2000, network_type="walk")
            site_node = ox.distance.nearest_nodes(G, lon, lat)

            for _, st in stations.iterrows():
                st_geom = st.geometry
                if st_geom.geom_type == "Point":
                    ll = st_geom
                else:
                    ll = st_geom.centroid
                ll_wgs84 = gpd.GeoSeries([ll], crs=3857).to_crs(4326).iloc[0]
                st_node = ox.distance.nearest_nodes(G, ll_wgs84.x, ll_wgs84.y)
                path = nx.shortest_path(G, site_node, st_node, weight="length")
                route_gdf = ox.routing.route_to_gdf(G, path).to_crs(3857)
                routes.append(route_gdf)
            log.info(f"[context] walking routes: {len(routes)}")
        except Exception as e:
            log.warning(f"[context] walking routes: {e}")

    # 12. Labels
    allowed_am = set(_LABEL_AM.get(SITE_TYPE, []))
    allowed_le = set(_LABEL_LE.get(SITE_TYPE, []))
    label_items = []
    seen = set()

    def _harvest(gdf, col, allowed):
        if gdf is None or gdf.empty:
            return
        try:
            g = gdf.copy()
            if allowed and col in g.columns:
                g = g[g[col].isin(allowed)]
            g["_lb"] = _get_name(g)
            g = g.dropna(subset=["_lb"])
            g = g[g["_lb"].astype(str).str.strip().str.len() > 0]
            g3 = g.to_crs(3857)
            for _, row in g3.head(25).iterrows():
                txt = _ascii(str(row["_lb"]).strip())
                if not txt or txt in seen:
                    continue
                seen.add(txt)
                geom = row.geometry
                p = (geom.representative_point()
                     if hasattr(geom, "representative_point")
                     else geom.centroid)
                label_items.append((p.distance(site_pt), geom, txt))
        except Exception as e:
            log.debug(f"[context] harvest: {e}")

    _harvest(am, "amenity", allowed_am)
    _harvest(lu, "leisure", allowed_le)
    _harvest(parks, "leisure", allowed_le)
    _harvest(schools, "amenity", allowed_am)
    _harvest(hosps, "amenity", allowed_am)

    label_items.sort(key=lambda x: x[0])
    label_items = [(g, t) for _, g, t in label_items[:24]]

    del fetched, lu, am
    gc.collect()

    # 13. Draw
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor("white")

    xmin = site_pt.x - half_x
    xmax = site_pt.x + half_x
    ymin = site_pt.y - half_y
    ymax = site_pt.y + half_y

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.autoscale(False)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=16, alpha=0.95)
    except Exception as e:
        log.warning(f"[context] basemap: {e}")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.autoscale(False)

    # Landuse layers
    _safe_plot(res, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(ind, ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks, ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools, ax, color="#9ecae1", alpha=0.90, zorder=2)

    # Building footprints (from Colab)
    _safe_plot(buildings, ax, color="#d9d9d9", alpha=0.35, zorder=1)

    # Walking routes to MTR (from Colab)
    for r in routes:
        _safe_plot(r, ax, color="#005eff", linewidth=2.2, linestyle="--", zorder=3)

    del res, ind, parks, schools, routes
    gc.collect()

    # Bus stops
    if not bus_stops.empty:
        if _bus_icon is not None:
            for _, row in bus_stops.iterrows():
                try:
                    g = row.geometry
                    bx = g.x if g.geom_type == "Point" else g.centroid.x
                    by = g.y if g.geom_type == "Point" else g.centroid.y
                    ic = OffsetImage(_bus_icon, zoom=0.028)
                    ic.image.axes = ax
                    ax.add_artist(AnnotationBbox(ic, (bx, by), frameon=False, zorder=9, box_alignment=(0.5, 0.5)))
                except Exception:
                    pass
        else:
            bus_stops.plot(ax=ax, color="#0d47a1", markersize=35, zorder=9, marker="s")
    del bus_stops
    gc.collect()

    # MTR stations
    for (cx_st, cy_st, nm) in stations_list:
        try:
            _mtr_icon(ax, cx_st, cy_st, zoom=0.042, zorder=14)
            ax.text(cx_st, cy_st - 110, _wrap(nm, 20),
                    fontsize=9, weight="bold", color="#1a1a1a",
                    ha="center", va="top",
                    bbox=dict(facecolor="white", edgecolor="#cccccc",
                              linewidth=0.8, alpha=0.95, boxstyle="round,pad=3"),
                    zorder=15)
        except Exception as e:
            log.debug(f"[context] MTR draw: {e}")

    # Site
    try:
        sc = site_geom.centroid
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred",
                      linewidth=2, zorder=11, alpha=0.95)
        ax.text(sc.x, sc.y, "SITE", color="white", weight="bold",
                fontsize=9, ha="center", va="center", zorder=12)
    except Exception:
        ax.plot(site_pt.x, site_pt.y, "o", color="#e53935",
                markersize=20, markeredgecolor="white", markeredgewidth=2.5, zorder=17)
        ax.text(site_pt.x, site_pt.y + 90, "SITE", color="#e53935", weight="bold",
                fontsize=9, ha="center", va="bottom", zorder=18)

    # Labels (from Colab with offsets)
    placed = []
    offsets = [(0, 35), (0, -35), (35, 0), (-35, 0), (25, 25), (-25, 25),
               (25, -25), (-25, -25)]
    for i, (geom, txt) in enumerate(label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_pt) < 140:
                continue
            if any(p.distance(pp) < 120 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x + dx, p.y + dy, _wrap(txt, 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=13, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # Info box (from Colab)
    ax.text(0.015, 0.985,
            f"Lot: {value}\n"
            f"OZP Plan: {plan_no}\n"
            f"Zoning: {zone}\n"
            f"Site Type: {SITE_TYPE}\n",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6))

    # Legend (from Colab)
    ax.legend(handles=[
        mpatches.Patch(color="#f2c6a0", label="Residential"),
        mpatches.Patch(color="#b39ddb", label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9", label="Public Park"),
        mpatches.Patch(color="#9ecae1", label="School / Institution"),
        mpatches.Patch(color="#d9d9d9", alpha=0.35, label="Buildings"),
        mpatches.Patch(color=MTR_COLOR, label="MTR Station"),
        mpatches.Patch(color="#e53935", label="Site"),
        mpatches.Patch(color="#005eff", label="Pedestrian Route to MTR"),
        mpatches.Patch(color="#0d47a1", label="Bus Stop"),
    ], loc="lower left", bbox_to_anchor=(0.02, 0.08),
       fontsize=8.5, framealpha=0.95)

    ax.set_title("Automated Site Context Analysis (Building-Type Driven)",
                 fontsize=15, weight="bold")
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info(f"[context] Done in {_time.monotonic() - t0:.1f}s")
    return buf
