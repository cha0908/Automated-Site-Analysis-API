"""
osm_builder.py — Runtime OSM data builder for context.py
Runs at server startup (inside warm_cache), NOT at build time.
Render free tier blocks downloads during build but allows them at runtime.
"""

import os
import logging
import threading
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import shape, Point

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "osm")
_PBF_PATH = os.path.join(_DATA_DIR, "hong-kong-latest.osm.pbf")
_MIN_PBF_MB = 30

_GPKG_FILES = {
    "buildings": os.path.join(_DATA_DIR, "buildings.gpkg"),
    "landuse":   os.path.join(_DATA_DIR, "landuse.gpkg"),
    "amenities": os.path.join(_DATA_DIR, "amenities.gpkg"),
    "transport": os.path.join(_DATA_DIR, "transport.gpkg"),
}

_PBF_SOURCES = [
    "https://download.geofabrik.de/asia/hong-kong-latest.osm.pbf",
    "https://download.bbbike.org/osm/bbbike/HongKong/HongKong.osm.pbf",
]

HK_BBOX = (113.82, 22.14, 114.45, 22.58)


def gpkg_files_ready() -> bool:
    """Return True if all 4 GeoPackages exist and are non-empty."""
    for path in _GPKG_FILES.values():
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            return False
    return True


def download_pbf() -> bool:
    """
    Download HK OSM PBF at runtime using requests (works on Render free tier).
    Returns True on success, False on failure.
    """
    import requests

    Path(_DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Skip if already valid
    if os.path.exists(_PBF_PATH):
        mb = os.path.getsize(_PBF_PATH) / 1e6
        if mb >= _MIN_PBF_MB:
            log.info(f"[osm_builder] PBF already exists ({mb:.1f} MB)")
            return True
        log.warning(f"[osm_builder] PBF corrupt ({mb:.1f} MB) — re-downloading")
        os.remove(_PBF_PATH)

    for url in _PBF_SOURCES:
        host = url.split('/')[2]
        log.info(f"[osm_builder] Downloading PBF from {host}...")
        try:
            resp = requests.get(url, stream=True, timeout=300,
                                headers={"User-Agent": "ALKF-OSM-Builder/1.0"})
            resp.raise_for_status()
            total      = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(_PBF_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            log.info(f"[osm_builder] {downloaded/1e6:.1f}/"
                                     f"{total/1e6:.1f} MB ({pct:.0f}%)")

            mb = os.path.getsize(_PBF_PATH) / 1e6
            if mb < _MIN_PBF_MB:
                log.warning(f"[osm_builder] Too small ({mb:.1f} MB) — "
                            f"trying next source")
                os.remove(_PBF_PATH)
                continue

            log.info(f"[osm_builder] PBF downloaded: {mb:.1f} MB")
            return True

        except Exception as e:
            log.warning(f"[osm_builder] {host} failed: {e}")
            if os.path.exists(_PBF_PATH):
                os.remove(_PBF_PATH)
            continue

    log.error("[osm_builder] All PBF sources failed")
    return False


def parse_pbf() -> bool:
    """
    Parse PBF into 4 GeoPackages using osmium.
    Returns True on success.
    """
    try:
        import osmium
    except ImportError:
        log.error("[osm_builder] osmium not installed")
        return False

    class _BuildingHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.features = []
            self._f = osmium.geom.GeoJSONFactory()
        def area(self, a):
            tags = dict(a.tags)
            if not tags.get("building"):
                return
            try:
                geom = shape(json.loads(self._f.create_multipolygon(a)))
                if not geom.is_valid:
                    geom = geom.buffer(0)
                self.features.append({
                    "building": tags.get("building", ""),
                    "name":     tags.get("name:en") or tags.get("name") or "",
                    "tourism":  tags.get("tourism", ""),
                    "office":   tags.get("office", ""),
                    "geometry": geom,
                })
            except Exception:
                pass

    class _LanduseHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.features = []
            self._f = osmium.geom.GeoJSONFactory()
        def area(self, a):
            tags = dict(a.tags)
            lu = tags.get("landuse", "")
            le = tags.get("leisure", "")
            if not (lu or le):
                return
            try:
                geom = shape(json.loads(self._f.create_multipolygon(a)))
                if not geom.is_valid:
                    geom = geom.buffer(0)
                self.features.append({
                    "landuse": lu, "leisure": le,
                    "name": tags.get("name:en") or tags.get("name") or "",
                    "geometry": geom,
                })
            except Exception:
                pass

    class _AmenityHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.features = []
            self._f = osmium.geom.GeoJSONFactory()
        def _add(self, tags, geom):
            am = tags.get("amenity",""); to = tags.get("tourism","")
            sh = tags.get("shop","");    of = tags.get("office","")
            if not (am or to or sh or of):
                return
            self.features.append({
                "amenity": am, "tourism": to, "shop": sh, "office": of,
                "name": tags.get("name:en") or tags.get("name") or "",
                "geometry": geom,
            })
        def node(self, n):
            try:
                self._add(dict(n.tags), Point(n.location.lon, n.location.lat))
            except Exception:
                pass
        def area(self, a):
            try:
                geom = shape(json.loads(self._f.create_multipolygon(a)))
                if not geom.is_valid:
                    geom = geom.buffer(0)
                self._add(dict(a.tags), geom)
            except Exception:
                pass

    class _TransportHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.features = []
            self._f = osmium.geom.GeoJSONFactory()
        def node(self, n):
            tags = dict(n.tags)
            hw = tags.get("highway",""); rw = tags.get("railway","")
            if hw != "bus_stop" and rw not in ("station","halt"):
                return
            try:
                self.features.append({
                    "highway": hw, "railway": rw,
                    "name": tags.get("name:en") or tags.get("name") or "",
                    "geometry": Point(n.location.lon, n.location.lat),
                })
            except Exception:
                pass
        def area(self, a):
            tags = dict(a.tags)
            rw = tags.get("railway","")
            if rw not in ("station","halt"):
                return
            try:
                geom = shape(json.loads(self._f.create_multipolygon(a)))
                if not geom.is_valid:
                    geom = geom.buffer(0)
                self.features.append({
                    "highway": "", "railway": rw,
                    "name": tags.get("name:en") or tags.get("name") or "",
                    "geometry": geom,
                })
            except Exception:
                pass

    handlers = [
        (_BuildingHandler,  _GPKG_FILES["buildings"],  "buildings"),
        (_LanduseHandler,   _GPKG_FILES["landuse"],    "landuse"),
        (_AmenityHandler,   _GPKG_FILES["amenities"],  "amenities"),
        (_TransportHandler, _GPKG_FILES["transport"],  "transport"),
    ]

    xmin, ymin, xmax, ymax = HK_BBOX

    for HandlerCls, out_path, label in handlers:
        log.info(f"[osm_builder] Parsing {label}...")
        try:
            h = HandlerCls()
            h.apply_file(_PBF_PATH, locations=True, idx="flex_mem")
            if not h.features:
                log.warning(f"[osm_builder] No features for {label}")
                continue
            gdf = gpd.GeoDataFrame(h.features, crs=4326)
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
            gdf = gdf.cx[xmin:xmax, ymin:ymax]
            gdf.to_file(out_path, driver="GPKG", layer=label)
            mb = os.path.getsize(out_path) / 1e6
            log.info(f"[osm_builder] {label}: {len(gdf):,} features "
                     f"({mb:.1f} MB)")
        except Exception as e:
            log.error(f"[osm_builder] Failed to parse {label}: {e}")
            return False

    return True


def build_local_datasets() -> bool:
    """
    Full pipeline: download PBF → parse → write GeoPackages.
    Called from warm_cache() in a background daemon thread.
    Returns True if all files are ready.
    """
    if gpkg_files_ready():
        log.info("[osm_builder] GeoPackages already present — skipping build")
        return True

    log.info("[osm_builder] Starting OSM data build pipeline...")

    if not download_pbf():
        return False

    if not parse_pbf():
        return False

    # Clean up PBF to save disk space (~70MB freed)
    if os.path.exists(_PBF_PATH):
        os.remove(_PBF_PATH)
        log.info("[osm_builder] PBF removed after parsing")

    if not gpkg_files_ready():
        log.error("[osm_builder] GeoPackages not all present after build")
        return False

    log.info("[osm_builder] OSM data build complete ✓")
    return True
