import logging
import os
import tempfile
import requests
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union

BASE_URL = "https://mapapi.geodata.gov.hk/gs/api/v1.0.0"
LOT_INDEX_BBOX_M = 300

ALLOWED_TYPES = [
    "LOT", "STT", "GLA", "LPP", "UN",
    "BUILDINGCSUID", "LOTCSUID", "PRN", "ADDRESS",
]

_LOT_INDEX_TYPE = {"GLA": "gla", "STT": "stt"}
_LOT_BOUNDARY_CACHE = {}
_transformer_2326_to_4326 = Transformer.from_crs(2326, 4326, always_xy=True)
_transformer_4326_to_2326 = Transformer.from_crs(4326, 2326, always_xy=True)


def _fetch_lot_gml(lit, minx, miny, maxx, maxy):
    """Fetch lot boundary GML from HK Gov GIS API."""
    url = f"{BASE_URL}/iC1000/{lit}?bbox={minx},{miny},{maxx},{maxy},EPSG:2326"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200 or not resp.content.strip():
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
            tmp.write(resp.content)
            tmp.flush()
            tmp_path = tmp.name
        gdf = gpd.read_file(tmp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(2326)
        return gdf.to_crs(4326)
    except Exception as e:
        logging.getLogger(__name__).debug("GML parse failed: %s", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def get_lot_boundary(lon: float, lat: float, data_type: str,
                     extents: list = None):
    """
    Fetch lot boundary polygon(s).

    If extents is provided (multi-lot), fetches each lot boundary using
    its exact HK2326 bbox and merges adjacent ones into a single polygon.
    Falls back to single-lot centroid search when extents is empty.
    """
    data_type = data_type.upper()
    if data_type == "ADDRESS":
        return None

    # ── MULTI-LOT: merge boundaries from real extents ─────────
    if extents and len(extents) > 1:
        merged_geoms = []
        lit = _LOT_INDEX_TYPE.get(data_type, "lot")
        for ext in extents:
            if not ext:
                continue
            try:
                xmin = float(ext["xmin"]); ymin = float(ext["ymin"])
                xmax = float(ext["xmax"]); ymax = float(ext["ymax"])
                # Expand search slightly to ensure we capture the lot
                pad = 5
                gdf = _fetch_lot_gml(lit, xmin - pad, ymin - pad, xmax + pad, ymax + pad)
                if gdf is None or gdf.empty:
                    # Fallback: create bbox polygon from extent
                    bbox_2326 = box(xmin, ymin, xmax, ymax)
                    bbox_gdf  = gpd.GeoDataFrame(geometry=[bbox_2326], crs=2326).to_crs(4326)
                    merged_geoms.append(bbox_gdf.geometry.iloc[0])
                    continue
                # Find the lot polygon that contains the extent centroid
                cx_2326 = (xmin + xmax) / 2
                cy_2326 = (ymin + ymax) / 2
                cx_wgs, cy_wgs = _transformer_2326_to_4326.transform(cx_2326, cy_2326)
                pt = Point(cx_wgs, cy_wgs)
                gdf["_dist"] = gdf.geometry.distance(pt)
                best = gdf.sort_values("_dist").iloc[0]
                merged_geoms.append(best.geometry)
            except Exception as e:
                logging.getLogger(__name__).debug("multi-lot extent fetch failed: %s", e)

        if merged_geoms:
            combined = unary_union(merged_geoms)
            out = gpd.GeoDataFrame(geometry=[combined], crs=4326).to_crs(3857)
            return out

    # ── SINGLE LOT: centroid-based search ─────────────────────
    cache_key = (round(lon, 5), round(lat, 5), data_type)
    if cache_key in _LOT_BOUNDARY_CACHE:
        return _LOT_BOUNDARY_CACHE[cache_key]

    lit = _LOT_INDEX_TYPE.get(data_type, "lot")
    try:
        to_2326  = Transformer.from_crs(4326, 2326, always_xy=True)
        x2326, y2326 = to_2326.transform(lon, lat)
        minx = x2326 - LOT_INDEX_BBOX_M
        miny = y2326 - LOT_INDEX_BBOX_M
        maxx = x2326 + LOT_INDEX_BBOX_M
        maxy = y2326 + LOT_INDEX_BBOX_M
        gdf = _fetch_lot_gml(lit, minx, miny, maxx, maxy)
        if gdf is None or gdf.empty:
            _LOT_BOUNDARY_CACHE[cache_key] = None
            return None

        pt = Point(lon, lat)
        for idx, row in gdf.iterrows():
            if row.geometry and row.geometry.contains(pt):
                out = gpd.GeoDataFrame(geometry=[row.geometry], crs=4326).to_crs(3857)
                _LOT_BOUNDARY_CACHE[cache_key] = out
                return out

        gdf["_dist"] = gdf.geometry.distance(pt)
        idx = gdf["_dist"].idxmin()
        if gdf.loc[idx, "_dist"] < 0.001:
            out = gpd.GeoDataFrame(geometry=[gdf.loc[idx, "geometry"]], crs=4326).to_crs(3857)
            _LOT_BOUNDARY_CACHE[cache_key] = out
            return out

    except Exception as e:
        logging.getLogger(__name__).debug("get_lot_boundary failed: %s", e)

    _LOT_BOUNDARY_CACHE[cache_key] = None
    return None


def resolve_location(data_type: str, value: str,
                     lon: float = None, lat: float = None,
                     lot_ids: list = None, extents: list = None):
    """
    Resolves to (lon, lat) in WGS84.
    ADDRESS type: returns pre-resolved coords directly.
    Multi-lot: uses centroid of merged extents as the analysis point.
    All other types: calls HK Gov GIS SearchNumber API.
    """
    data_type = data_type.upper()

    if data_type == "ADDRESS":
        if lon is not None and lat is not None:
            return lon, lat
        raise ValueError(
            "ADDRESS type requires pre-resolved lon/lat from /search results."
        )

    # Multi-lot: derive centroid from merged extents
    if extents and len(extents) > 1:
        valid_extents = [e for e in extents if e and e.get("xmin")]
        if valid_extents:
            all_xmin = min(e["xmin"] for e in valid_extents)
            all_ymin = min(e["ymin"] for e in valid_extents)
            all_xmax = max(e["xmax"] for e in valid_extents)
            all_ymax = max(e["ymax"] for e in valid_extents)
            cx = (all_xmin + all_xmax) / 2
            cy = (all_ymin + all_ymax) / 2
            lon_out, lat_out = _transformer_2326_to_4326.transform(cx, cy)
            return lon_out, lat_out

    # If we already have pre-resolved coords (from search), use them
    if lon is not None and lat is not None:
        return lon, lat

    if data_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported data type: {data_type}")

    url = (
        f"{BASE_URL}/lus/{data_type.lower()}/SearchNumber"
        f"?text={value.replace(' ', '%20')}"
    )
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError("Failed to resolve number.")

    data = response.json()

    if "candidates" not in data or len(data["candidates"]) == 0:
        raise ValueError("No matching result found.")

    best  = max(data["candidates"], key=lambda x: x.get("score", 0))
    x2326 = best["location"]["x"]
    y2326 = best["location"]["y"]

    lon_out, lat_out = Transformer.from_crs(
        2326, 4326, always_xy=True
    ).transform(x2326, y2326)

    return lon_out, lat_out
