import logging
import os
import tempfile
import requests
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point

BASE_URL = "https://mapapi.geodata.gov.hk/gs/api/v1.0.0"
LOT_INDEX_BBOX_M = 300  # half-side in metres; full bbox ≤ 750×600m per API

ALLOWED_TYPES = [
    "LOT",
    "STT",
    "GLA",
    "LPP",
    "UN",
    "BUILDINGCSUID",
    "LOTCSUID",
    "PRN"
]

# Lot Index API path: LOT/GLA/STT only; others fall back to "lot"
_LOT_INDEX_TYPE = {"GLA": "gla", "STT": "stt"}
_LOT_BOUNDARY_CACHE = {}


def get_lot_boundary(lon: float, lat: float, data_type: str):
    """
    Fetch official lot boundary from LandsD iC1000 API (HK80). Returns a single-row
    GeoDataFrame in EPSG:3857, or None if unavailable. Result is cached by (lon, lat, type).
    """
    data_type = data_type.upper()
    cache_key = (round(lon, 5), round(lat, 5), data_type)
    if cache_key in _LOT_BOUNDARY_CACHE:
        return _LOT_BOUNDARY_CACHE[cache_key]

    lit = _LOT_INDEX_TYPE.get(data_type, "lot")
    try:
        to_2326 = Transformer.from_crs(4326, 2326, always_xy=True)
        x2326, y2326 = to_2326.transform(lon, lat)
        minx = x2326 - LOT_INDEX_BBOX_M
        miny = y2326 - LOT_INDEX_BBOX_M
        maxx = x2326 + LOT_INDEX_BBOX_M
        maxy = y2326 + LOT_INDEX_BBOX_M
        url = f"{BASE_URL}/iC1000/{lit}?bbox={minx},{miny},{maxx},{maxy},EPSG:2326"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200 or not resp.content.strip():
            _LOT_BOUNDARY_CACHE[cache_key] = None
            return None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
                tmp.write(resp.content)
                tmp.flush()
                tmp_path = tmp.name
            gdf = gpd.read_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        if gdf.empty or "geometry" not in gdf.columns:
            _LOT_BOUNDARY_CACHE[cache_key] = None
            return None

        if gdf.crs is None:
            gdf.set_crs(2326, inplace=True)
        gdf = gdf.to_crs(4326)
        pt = Point(lon, lat)
        for idx, row in gdf.iterrows():
            if row.geometry and row.geometry.contains(pt):
                out = gpd.GeoDataFrame(geometry=[row.geometry], crs=4326).to_crs(3857)
                _LOT_BOUNDARY_CACHE[cache_key] = out
                return out

        # Point not inside any polygon: use closest (e.g. on boundary)
        gdf["_dist"] = gdf.geometry.distance(pt)
        idx = gdf["_dist"].idxmin()
        if gdf.loc[idx, "_dist"] < 0.0005:  # ~50m in degrees at HK lat
            out = gpd.GeoDataFrame(geometry=[gdf.loc[idx, "geometry"]], crs=4326).to_crs(3857)
            _LOT_BOUNDARY_CACHE[cache_key] = out
            return out
    except Exception as e:
        logging.getLogger(__name__).debug("get_lot_boundary failed: %s", e)

    _LOT_BOUNDARY_CACHE[cache_key] = None
    return None


def resolve_location(data_type: str, value: str):

    data_type = data_type.upper()

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

    best = max(data["candidates"], key=lambda x: x.get("score", 0))

    x2326 = best["location"]["x"]
    y2326 = best["location"]["y"]

    lon, lat = Transformer.from_crs(
        2326, 4326, always_xy=True
    ).transform(x2326, y2326)

    return lon, lat
