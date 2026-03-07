"""
prepare_osm_data.py — One-time OSM data preparation script
Run locally (not on Render) to generate the GeoPackage files.

Usage:
    pip install osmium geopandas pyogrio shapely
    python prepare_osm_data.py

Downloads HK OSM extract from Geofabrik and extracts:
    data/osm/buildings.gpkg   — building footprints with type + name
    data/osm/landuse.gpkg     — landuse + leisure polygons
    data/osm/amenities.gpkg   — amenity, tourism, shop points + polygons
    data/osm/transport.gpkg   — bus stops (points) + MTR stations (polygons)

All outputs in EPSG:4326. context.py reprojects to 3857 at load time.
Total file size estimate: ~80–120 MB for HK island + Kowloon.
"""

import os
import sys
import subprocess
import urllib.request
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Point, Polygon, MultiPolygon
import osmium
import json

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "osm")
os.makedirs(OUT_DIR, exist_ok=True)

PBF_URL  = "https://download.geofabrik.de/asia/hong-kong-latest.osm.pbf"
PBF_PATH = os.path.join(OUT_DIR, "hong-kong-latest.osm.pbf")

# ── HK bounding box (WGS84) ───────────────────────────────────────────────────
# Covers HK Island, Kowloon, New Territories
HK_BBOX = (113.82, 22.14, 114.45, 22.58)   # (xmin, ymin, xmax, ymax)


# ============================================================
# STEP 1 — Download PBF
# ============================================================

def download_pbf():
    if os.path.exists(PBF_PATH):
        print(f"PBF already exists: {PBF_PATH}")
        return
    print(f"Downloading HK OSM extract (~70MB)...")
    urllib.request.urlretrieve(PBF_URL, PBF_PATH,
        reporthook=lambda b, bs, ts: print(
            f"\r  {b*bs/1e6:.1f} / {ts/1e6:.1f} MB", end="", flush=True))
    print(f"\nDownloaded: {PBF_PATH}")


# ============================================================
# STEP 2 — OSM handlers using osmium
# ============================================================

class BuildingHandler(osmium.SimpleHandler):
    """Extract building footprints with type + name."""
    def __init__(self):
        super().__init__()
        self.features = []
        self._factory = osmium.geom.GeoJSONFactory()

    def _process(self, elem, geom_fn):
        tags = dict(elem.tags)
        btype = tags.get("building", "")
        if not btype:
            return
        name = tags.get("name:en") or tags.get("name") or ""
        try:
            geojson = geom_fn(elem)
            geom = shape(json.loads(geojson))
            if not geom.is_valid:
                geom = geom.buffer(0)
            self.features.append({
                "building": btype,
                "name":     name,
                "amenity":  tags.get("amenity", ""),
                "tourism":  tags.get("tourism", ""),
                "office":   tags.get("office", ""),
                "shop":     tags.get("shop", ""),
                "geometry": geom,
            })
        except Exception:
            pass

    def way(self, w):
        self._process(w, self._factory.create_linestring)

    def area(self, a):
        self._process(a, self._factory.create_multipolygon)


class LanduseHandler(osmium.SimpleHandler):
    """Extract landuse + leisure polygons."""
    def __init__(self):
        super().__init__()
        self.features = []
        self._factory = osmium.geom.GeoJSONFactory()

    def area(self, a):
        tags = dict(a.tags)
        landuse = tags.get("landuse", "")
        leisure = tags.get("leisure", "")
        natural = tags.get("natural", "")
        if not (landuse or leisure or natural):
            return
        name = tags.get("name:en") or tags.get("name") or ""
        try:
            geojson = self._factory.create_multipolygon(a)
            geom = shape(json.loads(geojson))
            if not geom.is_valid:
                geom = geom.buffer(0)
            self.features.append({
                "landuse": landuse,
                "leisure": leisure,
                "natural": natural,
                "name":    name,
                "geometry": geom,
            })
        except Exception:
            pass


class AmenityHandler(osmium.SimpleHandler):
    """Extract amenity, tourism, shop — both nodes and areas."""
    def __init__(self):
        super().__init__()
        self.features = []
        self._factory = osmium.geom.GeoJSONFactory()

    def _process_tags(self, tags, geom):
        amenity = tags.get("amenity", "")
        tourism = tags.get("tourism", "")
        shop    = tags.get("shop", "")
        office  = tags.get("office", "")
        if not (amenity or tourism or shop or office):
            return
        name = tags.get("name:en") or tags.get("name") or ""
        self.features.append({
            "amenity":  amenity,
            "tourism":  tourism,
            "shop":     shop,
            "office":   office,
            "name":     name,
            "geometry": geom,
        })

    def node(self, n):
        tags = dict(n.tags)
        try:
            geom = Point(n.location.lon, n.location.lat)
            self._process_tags(tags, geom)
        except Exception:
            pass

    def area(self, a):
        tags = dict(a.tags)
        try:
            geojson = self._factory.create_multipolygon(a)
            geom = shape(json.loads(geojson))
            if not geom.is_valid:
                geom = geom.buffer(0)
            self._process_tags(tags, geom)
        except Exception:
            pass


class TransportHandler(osmium.SimpleHandler):
    """Extract bus stops (nodes) and MTR/rail stations (areas + nodes)."""
    def __init__(self):
        super().__init__()
        self.features = []
        self._factory = osmium.geom.GeoJSONFactory()

    def node(self, n):
        tags = dict(n.tags)
        hw  = tags.get("highway", "")
        rw  = tags.get("railway", "")
        if hw != "bus_stop" and rw not in ("station", "halt", "tram_stop"):
            return
        name = tags.get("name:en") or tags.get("name") or ""
        try:
            geom = Point(n.location.lon, n.location.lat)
            self.features.append({
                "type":     hw or rw,
                "highway":  hw,
                "railway":  rw,
                "name":     name,
                "geometry": geom,
            })
        except Exception:
            pass

    def area(self, a):
        tags = dict(a.tags)
        rw = tags.get("railway", "")
        if rw not in ("station", "halt"):
            return
        name = tags.get("name:en") or tags.get("name") or ""
        try:
            geojson = self._factory.create_multipolygon(a)
            geom = shape(json.loads(geojson))
            if not geom.is_valid:
                geom = geom.buffer(0)
            self.features.append({
                "type":    rw,
                "highway": "",
                "railway": rw,
                "name":    name,
                "geometry": geom,
            })
        except Exception:
            pass


# ============================================================
# STEP 3 — Parse PBF and write GeoPackages
# ============================================================

def parse_and_save(handler_cls, out_path, label):
    print(f"\nParsing {label}...")
    h = handler_cls()
    h.apply_file(PBF_PATH, locations=True, idx="flex_mem")
    if not h.features:
        print(f"  WARNING: no features found for {label}")
        return

    gdf = gpd.GeoDataFrame(h.features, crs=4326)

    # Drop invalid / null geometries
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]

    # Clip to HK bbox to remove any out-of-bounds artefacts
    xmin, ymin, xmax, ymax = HK_BBOX
    gdf = gdf.cx[xmin:xmax, ymin:ymax]

    print(f"  {len(gdf):,} features → {out_path}")
    gdf.to_file(out_path, driver="GPKG", layer=label)
    print(f"  Saved: {os.path.getsize(out_path)/1e6:.1f} MB")


def main():
    download_pbf()
    parse_and_save(BuildingHandler, os.path.join(OUT_DIR, "buildings.gpkg"),  "buildings")
    parse_and_save(LanduseHandler,  os.path.join(OUT_DIR, "landuse.gpkg"),    "landuse")
    parse_and_save(AmenityHandler,  os.path.join(OUT_DIR, "amenities.gpkg"),  "amenities")
    parse_and_save(TransportHandler,os.path.join(OUT_DIR, "transport.gpkg"),  "transport")

    print("\n✓ All GeoPackages ready.")
    print("Copy data/osm/ to your Render project and redeploy.")


if __name__ == "__main__":
    main()
