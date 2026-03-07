import logging
import os
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import contextily as cx

from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, wait
from io import BytesIO

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 20

FETCH_RADIUS = 600
MAP_HALF_SIZE = 600

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)


# ─────────────────────────────────────────────────────────────
# Fast bbox fetch instead of features_from_point
# ─────────────────────────────────────────────────────────────

def fetch_bbox(lat, lon, dist, tags):

    try:

        north = lat + (dist / 111320)
        south = lat - (dist / 111320)
        east  = lon + (dist / (111320 * np.cos(np.radians(lat))))
        west  = lon - (dist / (111320 * np.cos(np.radians(lat))))

        gdf = ox.features_from_bbox(north, south, east, west, tags=tags)

        if gdf is not None and not gdf.empty:
            return gdf.to_crs(3857)

    except Exception as e:
        log.warning(f"[context] fetch failed {tags}: {e}")

    return _EMPTY.copy()


# ─────────────────────────────────────────────────────────────
# Parallel fetch
# ─────────────────────────────────────────────────────────────

def parallel_fetch(tasks):

    results = {}

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:

        futures = {pool.submit(fetch_bbox, *args): key
                   for key, args in tasks.items()}

        done, _ = wait(futures, timeout=30)

        for f in done:
            key = futures[f]
            try:
                results[key] = f.result()
                log.info(f"[context] {key}: {len(results[key])}")
            except:
                results[key] = _EMPTY.copy()

    return results


# ─────────────────────────────────────────────────────────────
# Site type inference
# ─────────────────────────────────────────────────────────────

def infer_site_type(zone):

    z = zone.upper()

    if z.startswith("R"):
        return "RESIDENTIAL"

    if z.startswith("C"):
        return "COMMERCIAL"

    if z.startswith("G"):
        return "INSTITUTIONAL"

    if z.startswith("I"):
        return "INDUSTRIAL"

    return "MIXED"


# ─────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────

def generate_context(
    data_type,
    value,
    ZONE_DATA,
    radius_m=None,
    lon=None,
    lat=None,
    lot_ids=None,
    extents=None,
):

    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)

    log.info(f"[context] {lon} {lat}")

    fetch_r = radius_m or FETCH_RADIUS

    # ─────────────────────────────────
    # Site geometry
    # ─────────────────────────────────

    lot = get_lot_boundary(lon, lat, data_type, extents)

    if lot is not None and not lot.empty:

        site_geom = lot.geometry.iloc[0]
        site_point = site_geom.centroid
        site_gdf = lot

    else:

        site_point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
        site_geom = site_point.buffer(80)
        site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # ─────────────────────────────────
    # Zoning
    # ─────────────────────────────────

    zones = ZONE_DATA.to_crs(3857)
    primary = zones[zones.contains(site_point)]

    if primary.empty:
        raise ValueError("No zoning")

    zone = primary.iloc[0]["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)

    log.info(f"[context] zone={zone} type={SITE_TYPE}")

    # ─────────────────────────────────
    # Fetch OSM
    # ─────────────────────────────────

    tasks = {

        "landuse": (lat, lon, fetch_r, {"landuse": True}),

        "parks": (lat, lon, fetch_r,
                  {"leisure": ["park", "playground"]}),

        "bus": (lat, lon, fetch_r,
                {"highway": "bus_stop"}),

        "buildings": (lat, lon, fetch_r,
                      {"building": True}),

        "streets": (lat, lon, fetch_r,
                    {"highway": ["primary",
                                 "secondary",
                                 "tertiary",
                                 "residential"]}),
    }

    results = parallel_fetch(tasks)

    buildings = results["buildings"]
    bus = results["bus"]
    landuse = results["landuse"]
    parks = results["parks"]
    streets = results["streets"]

    # ─────────────────────────────────
    # Plot
    # ─────────────────────────────────

    fig, ax = plt.subplots(figsize=(16, 12))

    cx.add_basemap(ax,
                   source=cx.providers.CartoDB.PositronNoLabels,
                   zoom=16)

    if not landuse.empty:
        landuse.plot(ax=ax,
                     color="#b39ddb",
                     alpha=0.5)

    if not parks.empty:
        parks.plot(ax=ax,
                   color="#b7dfb9",
                   alpha=0.9)

    if not buildings.empty:
        buildings.plot(ax=ax,
                       color="#e07b39",
                       alpha=0.8,
                       edgecolor="white",
                       linewidth=0.2)

    if not bus.empty:
        bus.plot(ax=ax,
                 color="#0d47a1",
                 markersize=40)

    site_gdf.plot(ax=ax,
                  color="#e53935",
                  edgecolor="black",
                  linewidth=2)

    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=16,
        weight="bold",
    )

    ax.set_axis_off()

    buf = BytesIO()

    plt.savefig(buf,
                format="png",
                dpi=120,
                bbox_inches="tight")

    plt.close(fig)

    buf.seek(0)

    log.info("[context] done")

    return buf
