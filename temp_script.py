"""
Inspect OSM polygon data used by context analysis.
Shows which layers have name/name:en and how many rows, so we can confirm usage.
Run from repo root: python scripts/inspect_context_osm_data.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# Same as context: one point and radius (e.g. HK)
LAT, LON = 22.2783, 114.1577
DIST = 200

print("Fetching polygons (same as context)...")
polygons = ox.features_from_point(
    (LAT, LON), dist=DIST,
    tags={"landuse": True, "leisure": True, "amenity": True, "building": True}
)
polygons = polygons.to_crs(3857)
site_point = gpd.GeoSeries([Point(LON, LAT)], crs=4326).to_crs(3857).iloc[0]

# Same splits as context
residential = polygons[polygons.get("landuse") == "residential"]
industrial = polygons[polygons.get("landuse").isin(["industrial", "commercial"])]
parks = polygons[polygons.get("leisure") == "park"]
schools = polygons[polygons.get("amenity").isin(["school", "college", "university"])]
buildings = polygons[polygons.get("building").notnull()]

def summarize(gdf, name):
    print(f"\n--- {name} ---")
    print(f"  rows: {len(gdf)}")
    if gdf.empty:
        return
    cols = list(gdf.columns)
    has_name = "name" in cols
    has_name_en = "name:en" in cols
    print(f"  has 'name': {has_name}, has 'name:en': {has_name_en}")
    label = gdf.get("name:en").fillna(gdf.get("name"))
    if label is not None:
        non_null = label.notna()
        non_empty = label.astype(str).str.strip().str.len() > 0
        ok = non_null & non_empty
        print(f"  label (name:en fillna name): non_null={non_null.sum()}, non_empty={non_empty.sum()}, usable={ok.sum()}")
        if ok.any():
            print(f"  sample labels: {label[ok].head(3).tolist()}")
    # For buildings, show building tag values
    if name == "buildings" and "building" in cols:
        b = gdf["building"]
        print(f"  building tag value_counts: {b.value_counts(dropna=False).head(10).to_dict()}")

for gdf, name in [
    (residential, "residential"),
    (industrial, "industrial"),
    (parks, "parks"),
    (schools, "schools"),
    (buildings, "buildings"),
]:
    summarize(gdf, name)

print("\n--- done ---")