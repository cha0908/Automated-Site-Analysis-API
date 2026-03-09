"""
Diagnostic script for View Analysis at K11 Musea.
Runs the view pipeline with detailed logging to diagnose sea view detection.
Run from the Automated-Site-Analysis-API directory:
  python test_view_k11.py

Output: view_diagnostic_k11.log and optionally view_k11_output.png

Experiment by changing the radius overrides below; the script patches the view
module so generate_view() and all diagnostics use the same values.
"""

import os
import sys
import logging
import requests

# Run from API directory: python scripts/test_view_k11.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.dirname(SCRIPT_DIR)
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
os.chdir(API_ROOT)

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from pyproj import Transformer

import osmnx as ox
from modules.resolver import resolve_location, get_lot_boundary
from modules import view as view_mod

_t2326_4326 = Transformer.from_crs(2326, 4326, always_xy=True)

# Isolate test from cache so results reflect current WATER_TAGS / fetch (no stale OSM data)
ox.settings.use_cache = False
ox.settings.log_console = False

# ── Config ────────────────────────────────────────────────────────────────────
DATA_TYPE = "ADDRESS"
VALUE = "K11 Musea"
LOG_FILE = os.path.join(SCRIPT_DIR, "view_diagnostic_k11.log")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "view_k11_output.png")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
BUILDING_GPKG = os.path.join(DATA_DIR, "BUILDINGS_FINAL .gpkg")

# Radius overrides for this test (patched into view module so generate_view uses them too).
# Default view module: FETCH=500, MAP=200, VIEW=200. Use larger to pull in more water.
TEST_FETCH_RADIUS = 600
TEST_MAP_RADIUS = 400
TEST_VIEW_RADIUS = 400
# Optional: uncomment to experiment with city-candidate radius (default 30).
# TEST_CITY_RADIUS = 50  # then set view_mod.CITY_RADIUS = TEST_CITY_RADIUS in main()

# Sectors to log in detail (south / sea direction: ~140–260°)
SOUTH_SECTOR_START = 140
SOUTH_SECTOR_END = 260


def log_line(msg: str, logger):
    logger.info(msg)
    print(msg)


def resolve_address_search(query: str):
    """
    Resolve an address (e.g. 'K11 Musea') to lon, lat using the same
    geodata.gov.hk API as the app's /search endpoint. Returns (lon, lat) or (None, None).
    """
    try:
        resp = requests.get(
            f"https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q={requests.utils.quote(query)}",
            timeout=10,
        )
        data = resp.json()
        if not data:
            return None, None
        s = data[0]
        x, y = s.get("x"), s.get("y")
        if x is None or y is None:
            return None, None
        lon, lat = _t2326_4326.transform(x, y)
        return round(lon, 6), round(lat, 6)
    except Exception as e:
        print(f"Address search failed: {e}")
        return None, None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger()
    log = lambda s: log_line(s, logger)

    log("=" * 70)
    log("VIEW ANALYSIS DIAGNOSTIC – K11 Musea")
    log("=" * 70)

    # Apply radius overrides so this run and generate_view() use the same values
    view_mod.FETCH_RADIUS = TEST_FETCH_RADIUS
    view_mod.MAP_RADIUS = TEST_MAP_RADIUS
    view_mod.VIEW_RADIUS = TEST_VIEW_RADIUS
    log("── RADII (overrides applied) ──")
    log(f"  FETCH_RADIUS={view_mod.FETCH_RADIUS} m, MAP_RADIUS={view_mod.MAP_RADIUS} m, VIEW_RADIUS={view_mod.VIEW_RADIUS} m")
    log(f"  CITY_RADIUS={view_mod.CITY_RADIUS} m, SECTOR_SIZE={view_mod.SECTOR_SIZE}°")
    log("")

    # Load building data
    if not os.path.isfile(BUILDING_GPKG):
        log(f"ERROR: Building data not found: {BUILDING_GPKG}")
        return
    BUILDING_DATA = gpd.read_file(BUILDING_GPKG).to_crs(3857)
    if "HEIGHT_M" not in BUILDING_DATA.columns:
        log("ERROR: HEIGHT_M column missing in building data")
        return
    BUILDING_DATA = BUILDING_DATA[BUILDING_DATA["HEIGHT_M"] > 5]
    log(f"Loaded BUILDING_DATA: {len(BUILDING_DATA)} buildings (HEIGHT_M > 5)\n")

    # 1. Resolve location (ADDRESS requires pre-resolved lon/lat from search)
    log("── 1. RESOLUTION ──")
    lon, lat = resolve_address_search(VALUE)
    if lon is None or lat is None:
        log(f"  ERROR: Could not resolve '{VALUE}' via geodata.gov.hk locationSearch")
        return
    log(f"  data_type={DATA_TYPE}, value={VALUE}")
    log(f"  Resolved via geodata.gov.hk locationSearch: lon={lon}, lat={lat}")
    # Pass into resolver so view module gets same coords when we call generate_view
    lon, lat = resolve_location(DATA_TYPE, VALUE, lon, lat)
    log("")

    # 2. Site polygon
    log("── 2. SITE POLYGON ──")
    lot_gdf = get_lot_boundary(lon, lat, DATA_TYPE)
    if lot_gdf is not None:
        site_geom = lot_gdf.geometry.iloc[0]
        center = site_geom.centroid
        log("  Source: lot boundary (get_lot_boundary)")
    else:
        site_building = ox.features_from_point(
            (lat, lon), dist=60, tags={"building": True}
        ).to_crs(3857)
        if len(site_building):
            site_geom = (
                site_building.assign(area=site_building.area)
                .sort_values("area", ascending=False)
                .geometry.iloc[0]
            )
        else:
            site_geom = (
                gpd.GeoSeries([Point(lon, lat)], crs=4326)
                .to_crs(3857)
                .iloc[0]
                .buffer(25)
            )
        center = site_geom.centroid
        log("  Source: OSM building or buffer fallback")
    log(f"  center (3857): x={center.x:.2f}, y={center.y:.2f}\n")

    analysis_circle = center.buffer(view_mod.MAP_RADIUS)

    # 3. Context data (same workflow as view.generate_view)
    def fetch_layer(tags):
        gdf = ox.features_from_point(
            (lat, lon), dist=view_mod.FETCH_RADIUS, tags=tags
        ).to_crs(3857)
        return gdf[gdf.intersects(analysis_circle)]

    buildings = fetch_layer({"building": True})
    parks = fetch_layer({"leisure": "park", "landuse": "grass", "natural": "wood"})

    # Water: one call for natural in [strait, bay, water, coastline], same as view.py
    log("── 3. WATER LAYER (same as view.py) ──")
    gdf_all = fetch_layer({"natural": ["strait", "bay", "water", "coastline"]})
    water_parts = []
    if not gdf_all.empty and "geometry" in gdf_all.columns:
        poly_mask = gdf_all.geometry.type.isin(["Polygon", "MultiPolygon"])
        if poly_mask.any():
            water_parts.append(gdf_all.loc[poly_mask, ["geometry"]].copy())
        line_mask = gdf_all.geometry.type.isin(["LineString", "MultiLineString"])
        if line_mask.any():
            coast_lines = gdf_all.loc[line_mask]
            buffered = coast_lines.geometry.buffer(view_mod.COASTLINE_BUFFER_M)
            water_parts.append(gpd.GeoDataFrame(geometry=buffered, crs=gdf_all.crs))
    if water_parts:
        water = gpd.GeoDataFrame(
            geometry=pd.concat([p.geometry for p in water_parts], ignore_index=True),
            crs=water_parts[0].crs,
        )
    else:
        water = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    log(f"  FETCH_RADIUS={view_mod.FETCH_RADIUS}, MAP_RADIUS={view_mod.MAP_RADIUS}")
    log(f"  water features (poly+coastline buffered): {len(water)}")
    if len(water):
        water_in_circle = water.intersection(analysis_circle)
        total_water_area = water_in_circle.area.sum()
        log(f"  total water area inside analysis_circle: {total_water_area:.0f} m²")
    log(f"  parks (in circle): {len(parks)} features\n")

    # 4. Site height
    nearby = BUILDING_DATA[BUILDING_DATA.intersects(analysis_circle)].copy()
    H_max = view_mod._get_site_height(nearby, center)
    H_mid = H_max / 2.0
    log("── 4. SITE HEIGHT ──")
    log(f"  H_max={H_max:.2f} m, H_mid={H_mid:.2f} m")
    log(f"  nearby buildings (in analysis_circle): {len(nearby)}\n")

    # 5. City candidates
    city_circle = center.buffer(view_mod.CITY_RADIUS)
    city_candidates = nearby[nearby.intersects(city_circle)].copy()
    log("── 5. CITY CANDIDATES (intersect circle around site) ──")
    log(f"  CITY_RADIUS={view_mod.CITY_RADIUS} m, count={len(city_candidates)}")

    if len(city_candidates):
        city_candidates = city_candidates.copy()
        city_candidates["_angle"] = city_candidates.geometry.centroid.apply(
            lambda pt: np.degrees(np.arctan2(pt.y - center.y, pt.x - center.x)) % 360
        )
        city_candidates["_dist"] = city_candidates.geometry.centroid.apply(
            lambda pt: center.distance(pt)
        )
        for i, (_, row) in enumerate(city_candidates.iterrows()):
            log(f"    [{i}] HEIGHT_M={row['HEIGHT_M']:.1f}, _angle={row['_angle']:.1f}°, _dist={row['_dist']:.1f}m")
        log("  → Any sector containing one of these angles with HEIGHT_M > h_ref becomes CITY first.\n")
    else:
        log("  (none)\n")

    # 6. Classify sectors (same as view pipeline)
    raw_mid = view_mod._classify_sectors(
        center, parks, water, city_candidates, H_mid, nearby, H_max
    )
    raw_max = view_mod._classify_sectors(
        center, parks, water, city_candidates, H_max, nearby, H_max
    )

    # 6a. Per-wedge data (shares and resulting view)
    log("── 6a. SECTOR SHARES & CLASSIFICATION (VIEW_RADIUS={} m) ──".format(view_mod.VIEW_RADIUS))
    one_sector = view_mod._make_sector(
        center.x, center.y, view_mod.VIEW_RADIUS, 0, view_mod.SECTOR_SIZE
    )
    log(f"  sector_area (one wedge): {one_sector.area:.0f} m²")
    for angle in range(0, 360, view_mod.SECTOR_SIZE):
        start, end = angle, angle + view_mod.SECTOR_SIZE
        sector = view_mod._make_sector(
            center.x, center.y, view_mod.VIEW_RADIUS, start, end
        )
        sector_area = sector.area or 1.0
        green_share = (
            (parks.intersection(sector).area.sum() if len(parks) else 0) / sector_area
        )
        water_share = (
            (water.intersection(sector).area.sum() if len(water) else 0) / sector_area
        )
        row_mid = next(r for r in raw_mid if r["start"] == start)
        row_max = next(r for r in raw_max if r["start"] == start)
        in_south = SOUTH_SECTOR_START <= start < SOUTH_SECTOR_END
        if in_south or start % 60 == 0:
            log(
                f"  sector {start:3d}-{end:3d}° | "
                f"green={green_share:.3f} water={water_share:.3f} | "
                f"mid={row_mid['view']} max={row_max['view']}"
            )
    log("")

    # 6b. City-view logic only: why CITY, and when PARK/SEA is overridden by blocking
    log("── 6b. CITY VIEW LOGIC (blocking / city candidate) ──")
    SECTOR_SIZE = view_mod.SECTOR_SIZE
    VIEW_RADIUS = view_mod.VIEW_RADIUS
    if len(city_candidates):
        cand = city_candidates.copy()
        cand["_angle"] = cand.geometry.centroid.apply(
            lambda pt: np.degrees(np.arctan2(pt.y - center.y, pt.x - center.x)) % 360
        )
    else:
        cand = pd.DataFrame(columns=["HEIGHT_M", "_angle"])
    if len(nearby):
        nb = nearby.copy()
        nb["_angle"] = nb.geometry.centroid.apply(
            lambda pt: np.degrees(np.arctan2(pt.y - center.y, pt.x - center.x)) % 360
        )
        nb["_dist"] = nb.geometry.centroid.apply(lambda pt: center.distance(pt))
    else:
        nb = pd.DataFrame(columns=["HEIGHT_M", "_angle", "_dist"])

    for panel_name, h_ref, h_site, raw_rows in [
        ("MID", H_mid, H_max, raw_mid),
        ("MAX", H_max, H_max, raw_max),
    ]:
        log(f"  [{panel_name}] h_ref={h_ref:.1f}m, h_site={h_site:.1f}m")
        for angle in range(0, 360, SECTOR_SIZE):
            start, end = angle, angle + SECTOR_SIZE
            sector = view_mod._make_sector(
                center.x, center.y, VIEW_RADIUS, start, end
            )
            sector_area = sector.area or 1.0
            green_share = (
                (parks.intersection(sector).area.sum() if len(parks) else 0) / sector_area
            )
            water_share = (
                (water.intersection(sector).area.sum() if len(water) else 0) / sector_area
            )
            is_city = False
            if len(cand):
                in_wedge = cand[cand["_angle"].between(start, end, inclusive="left")]
                if len(in_wedge) and (in_wedge["HEIGHT_M"] > h_ref).any():
                    is_city = True
            if is_city:
                in_wedge = cand[cand["_angle"].between(start, end, inclusive="left")]
                taller = in_wedge[in_wedge["HEIGHT_M"] > h_ref]
                parts = [f"HEIGHT_M={r['HEIGHT_M']:.1f}m @ {r['_angle']:.1f}°" for _, r in taller.iterrows()]
                log(
                    f"    sector {start:3d}-{end:3d}° | CITY (city candidate in wedge, HEIGHT_M > h_ref): "
                    f"{parts}"
                )
                continue
            view = "CITY"
            if water_share > 0.02 and water_share > green_share:
                view = "SEA"
            elif green_share > 0.02:
                view = "PARK"
            if view not in ("PARK", "SEA"):
                continue
            in_wedge_nb = nb[
                nb["_angle"].between(start, end, inclusive="left") &
                (nb["_dist"] <= VIEW_RADIUS)
            ].sort_values("_dist")
            taller = in_wedge_nb[in_wedge_nb["HEIGHT_M"] > h_site]
            if len(taller) == 0:
                log(
                    f"    sector {start:3d}-{end:3d}° | {view} (no building taller than site in wedge)"
                )
                continue
            d_block = float(taller.iloc[0]["_dist"])
            if view == "PARK":
                park_geom = parks.intersection(sector).unary_union
                content_dist = float("inf") if park_geom.is_empty else center.distance(park_geom)
            else:
                water_geom = water.intersection(sector).unary_union
                content_dist = float("inf") if water_geom.is_empty else center.distance(water_geom)
            blocked = d_block < content_dist
            blocker = taller.iloc[0]
            log(
                f"    sector {start:3d}-{end:3d}° | {view} -> {'CITY (BLOCKED)' if blocked else view}: "
                f"nearest taller building HEIGHT_M={blocker['HEIGHT_M']:.1f}m at {d_block:.1f}m, "
                f"content at {content_dist:.1f}m"
            )
        log("")

    # Summary by view type
    log("── 7. SUMMARY (MID panel) ──")
    from collections import Counter
    mid_views = Counter(r["view"] for r in raw_mid)
    for v, c in mid_views.most_common():
        log(f"  {v}: {c} sectors")
    log("")
    log("── 8. SUMMARY (MAX panel) ──")
    max_views = Counter(r["view"] for r in raw_max)
    for v, c in max_views.most_common():
        log(f"  {v}: {c} sectors")
    log("")

    # Generate image (uses same patched radii)
    log("── 9. GENERATING IMAGE ──")
    try:
        buf = view_mod.generate_view(DATA_TYPE, VALUE, BUILDING_DATA, lon=lon, lat=lat)
        with open(OUTPUT_PNG, "wb") as f:
            f.write(buf.getvalue())
        log(f"  Saved: {OUTPUT_PNG}")
    except Exception as e:
        log(f"  ERROR generating image: {e}")
    log("")
    log("Done. Share view_diagnostic_k11.log to diagnose sea view detection.")
    log("")
    log("── HOW TO EXPERIMENT ──")
    log("  • Edit TEST_FETCH_RADIUS, TEST_MAP_RADIUS, TEST_VIEW_RADIUS at top of this script.")
    log("  • In modules/view.py: change CITY_RADIUS (default 30) to exclude fewer buildings as city candidates.")
    log("  • In modules/view.py: change the 0.02 threshold in _classify_sectors (water_share > 0.02) to be more lenient.")
    log("  • In modules/view.py: consider excluding the site building from city_candidates so it does not claim harbour sectors.")


if __name__ == "__main__":
    main()
