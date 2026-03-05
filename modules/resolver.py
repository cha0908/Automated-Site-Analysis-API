import logging
import os
import tempfile
import requests
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union

BASE_URL         = "https://mapapi.geodata.gov.hk/gs/api/v1.0.0"
LOT_INDEX_BBOX_M = 300   # half-side metres for single-lot centroid search

ALLOWED_TYPES = [
    "LOT", "STT", "GLA", "LPP", "UN",
    "BUILDINGCSUID", "LOTCSUID", "PRN", "ADDRESS",
]

_LOT_INDEX_TYPE      = {"GLA": "gla", "STT": "stt"}
_LOT_BOUNDARY_CACHE  = {}
_t2326_4326 = Transformer.from_crs(2326, 4326, always_xy=True)
_t4326_2326 = Transformer.from_crs(4326, 2326, always_xy=True)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: fetch + parse GML from LandsD iC1000 API
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_lot_gml(lit, minx, miny, maxx, maxy):
    url = f"{BASE_URL}/iC1000/{lit}?bbox={minx},{miny},{maxx},{maxy},EPSG:2326"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200 or not resp.content.strip():
            return None
    except Exception as e:
        log.debug("GML fetch error: %s", e)
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
        log.debug("GML parse failed: %s", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: get_lot_boundary
#
# Single lot  → centroid-based search → returns 1-row GDF in EPSG:3857
# Multi-lot   → per-extent fetch → merge into ONE unified polygon → returns
#               1-row GDF in EPSG:3857 (no individual lot highlighting)
# ADDRESS     → returns None (site polygon determined by OSM building lookup)
# ─────────────────────────────────────────────────────────────────────────────
def get_lot_boundary(lon: float, lat: float, data_type: str,
                     extents: list = None):
    data_type = data_type.upper()

    if data_type == "ADDRESS":
        return None

    # ── MULTI-LOT: one extent per selected lot ────────────────
    if extents and len(extents) > 1:
        merged_geoms = []
        lit = _LOT_INDEX_TYPE.get(data_type, "lot")

        for ext in extents:
            if not ext:
                continue
            try:
                xmin = float(ext["xmin"]); ymin = float(ext["ymin"])
                xmax = float(ext["xmax"]); ymax = float(ext["ymax"])
                pad  = 5  # small padding to catch boundary polygons
                gdf  = _fetch_lot_gml(lit, xmin-pad, ymin-pad, xmax+pad, ymax+pad)

                if gdf is None or gdf.empty:
                    # Fallback: use extent bbox converted to WGS84
                    bbox_gdf = gpd.GeoDataFrame(
                        geometry=[box(xmin, ymin, xmax, ymax)], crs=2326
                    ).to_crs(4326)
                    merged_geoms.append(bbox_gdf.geometry.iloc[0])
                    continue

                # Pick the polygon closest to this lot's centroid in HK2326
                cx_wgs, cy_wgs = _t2326_4326.transform(
                    (xmin+xmax)/2, (ymin+ymax)/2
                )
                pt = Point(cx_wgs, cy_wgs)
                gdf["_dist"] = gdf.geometry.distance(pt)
                merged_geoms.append(gdf.sort_values("_dist").geometry.iloc[0])

            except Exception as e:
                log.debug("multi-lot extent fetch failed: %s", e)

        if merged_geoms:
            # Merge ALL selected lots into ONE unified polygon
            combined = unary_union(merged_geoms)
            return gpd.GeoDataFrame(geometry=[combined], crs=4326).to_crs(3857)

    # ── SINGLE LOT: centroid-based search ─────────────────────
    cache_key = (round(lon, 5), round(lat, 5), data_type)
    if cache_key in _LOT_BOUNDARY_CACHE:
        return _LOT_BOUNDARY_CACHE[cache_key]

    lit = _LOT_INDEX_TYPE.get(data_type, "lot")
    try:
        x2326, y2326 = _t4326_2326.transform(lon, lat)
        gdf = _fetch_lot_gml(
            lit,
            x2326 - LOT_INDEX_BBOX_M, y2326 - LOT_INDEX_BBOX_M,
            x2326 + LOT_INDEX_BBOX_M, y2326 + LOT_INDEX_BBOX_M,
        )
        if gdf is None or gdf.empty:
            _LOT_BOUNDARY_CACHE[cache_key] = None
            return None

        pt = Point(lon, lat)

        # Prefer a polygon that actually contains the resolved point
        for _, row in gdf.iterrows():
            if row.geometry and row.geometry.contains(pt):
                out = gpd.GeoDataFrame(
                    geometry=[row.geometry], crs=4326
                ).to_crs(3857)
                _LOT_BOUNDARY_CACHE[cache_key] = out
                return out

        # Fallback: closest polygon within ~50m (in degrees at HK lat ≈ 0.0005°)
        gdf["_dist"] = gdf.geometry.distance(pt)
        best = gdf.sort_values("_dist").iloc[0]
        if best["_dist"] < 0.001:
            out = gpd.GeoDataFrame(
                geometry=[best.geometry], crs=4326
            ).to_crs(3857)
            _LOT_BOUNDARY_CACHE[cache_key] = out
            return out

    except Exception as e:
        log.debug("get_lot_boundary single-lot failed: %s", e)

    _LOT_BOUNDARY_CACHE[cache_key] = None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: resolve_location  → (lon, lat) WGS84
#
# Priority order:
#   1. ADDRESS   → return pre-resolved lon/lat directly
#   2. Multi-lot → centroid of merged extents bbox
#   3. Pre-resolved lon/lat from search results
#   4. HK Gov GIS SearchNumber API call
# ─────────────────────────────────────────────────────────────────────────────
def resolve_location(data_type: str, value: str,
                     lon: float = None, lat: float = None,
                     lot_ids: list = None, extents: list = None):
    data_type = data_type.upper()

    # ADDRESS: coords come pre-resolved from frontend search
    if data_type == "ADDRESS":
        if lon is not None and lat is not None:
            return lon, lat
        raise ValueError(
            "ADDRESS type requires pre-resolved lon/lat from /search results."
        )

    # Multi-lot: use centroid of merged bounding box of all extents
    if extents and len(extents) > 1:
        valid = [e for e in extents if e and e.get("xmin")]
        if valid:
            cx = (min(e["xmin"] for e in valid) + max(e["xmax"] for e in valid)) / 2
            cy = (min(e["ymin"] for e in valid) + max(e["ymax"] for e in valid)) / 2
            lon_out, lat_out = _t2326_4326.transform(cx, cy)
            return lon_out, lat_out

    # Single lot with pre-resolved coords from search
    if lon is not None and lat is not None:
        return lon, lat

    if data_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Fall back to live API lookup
    url = (
        f"{BASE_URL}/lus/{data_type.lower()}/SearchNumber"
        f"?text={value.replace(' ', '%20')}"
    )
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise ValueError("Failed to resolve location: API returned non-200.")

    data = resp.json()
    if not data.get("candidates"):
        raise ValueError(f"No matching result found for {data_type} {value}.")

    best    = max(data["candidates"], key=lambda x: x.get("score", 0))
    x2326   = best["location"]["x"]
    y2326   = best["location"]["y"]
    lon_out, lat_out = _t2326_4326.transform(x2326, y2326)
    return lon_out, lat_out
