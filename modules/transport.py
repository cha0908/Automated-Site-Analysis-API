from typing import Optional, List
import matplotlib
matplotlib.use("Agg")

import os
import gc
import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.legend_handler as lh
import numpy as np
import pandas as pd

from shapely.geometry import Point, box
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from modules.resolver import resolve_location, get_lot_boundary


# ============================================================
# SETTINGS
# ============================================================

ox.settings.use_cache = True
ox.settings.log_console = False

MAP_RADIUS = 3000

COLOR_ROADS = "#e85d9e"
COLOR_WATER = "#6fa8dc"
COLOR_BUILDINGS = "#d6d6d6"
COLOR_SITE = "#FF0000"
COLOR_LIGHT_RAIL = "#D3A809"

STATION_LOGO_ZOOM = 0.055
STATION_MIN_DISTANCE = 250


# ============================================================
# MTR LINE COLORS
# ============================================================

MTR_LINE_COLORS = {
    "island": "#007DC5",
    "kwun tong": "#00AB4E",
    "tsuen wan": "#ED1D24",
    "tseung kwan o": "#7D499D",
    "tung chung": "#F7943E",
    "east rail": "#5EB6E4",
    "tuen ma": "#923011",
    "south island": "#BAC429",
    "airport express": "#888B8D",
    "lantau and airport": "#888B8D",
    "guangzhoushen": "#007DC5",
}

DEFAULT_MTR_COLOR = "#3f78b5"


MTR_LEGEND_LINES = [
    ("island", "#007DC5", "Island Line"),
    ("kwun tong", "#00AB4E", "Kwun Tong Line"),
    ("tsuen wan", "#ED1D24", "Tsuen Wan Line"),
    ("tseung kwan o", "#7D499D", "Tseung Kwan O Line"),
    ("tung chung", "#F7943E", "Tung Chung Line"),
    ("east rail", "#5EB6E4", "East Rail Line"),
    ("tuen ma", "#923011", "Tuen Ma Line"),
    ("south island", "#BAC429", "South Island Line"),
    ("airport express", "#888B8D", "Airport Express"),
    ("lantau and airport", "#888B8D", "Airport Express"),
    ("guangzhoushen", "#007DC5", "Express Rail Link"),
]


def get_mtr_color(name: str) -> str:
    name = name.lower()
    for key, col in MTR_LINE_COLORS.items():
        if key in name:
            return col
    return DEFAULT_MTR_COLOR


# ============================================================
# LOAD MTR LOGO
# ============================================================

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")

try:
    _mtr_img = mpimg.imread(_MTR_LOGO_PATH)
    MTR_LOGO_LOADED = True
except:
    _mtr_img = None
    MTR_LOGO_LOADED = False


# ============================================================
# DRAW STATION
# ============================================================

def draw_station(ax, x, y, zoom=STATION_LOGO_ZOOM,
                 fallback_color="#ED1D24", zorder=9):

    if MTR_LOGO_LOADED and _mtr_img is not None:

        icon = OffsetImage(_mtr_img, zoom=zoom)
        icon.image.axes = ax

        ab = AnnotationBbox(
            icon,
            (x, y),
            frameon=False,
            zorder=zorder
        )

        ax.add_artist(ab)

    else:

        ax.add_patch(
            plt.Circle((x, y), 60, color=fallback_color, zorder=zorder)
        )


# ============================================================
# SAFE FETCH
# ============================================================

def safe_fetch(lat, lon, dist, tags):

    try:

        gdf = ox.features_from_point(
            (lat, lon),
            dist=dist,
            tags=tags
        )

        if not gdf.empty:
            return gdf.to_crs(3857)

        return gpd.GeoDataFrame(geometry=[], crs=3857)

    except:

        return gpd.GeoDataFrame(geometry=[], crs=3857)


def keep_lines(gdf):

    if not gdf.empty:

        return gdf[
            gdf.geometry.type.isin(
                ["LineString", "MultiLineString"]
            )
        ]

    return gpd.GeoDataFrame(geometry=[], crs=3857)


# ============================================================
# TRANSPORT MAP GENERATOR
# ============================================================

def generate_transport(data_type: str,
                       value: str,
                       radius_m: Optional[int] = None,
                       lon: float = None,
                       lat: float = None,
                       lot_ids: List[str] = None,
                       extents: List[dict] = None):


    lon, lat = resolve_location(
        data_type,
        value,
        lon,
        lat,
        lot_ids,
        extents
    )

    r = radius_m if radius_m else MAP_RADIUS

    lot_gdf = get_lot_boundary(lon, lat, data_type, extents)


    # ========================================================
    # SITE POINT
    # ========================================================

    if lot_gdf is not None:

        site_geom = lot_gdf.geometry.iloc[0]
        site_gdf = lot_gdf
        site_point = site_geom.centroid

    else:

        site_point = gpd.GeoSeries(
            [Point(lon, lat)],
            crs=4326
        ).to_crs(3857).iloc[0]


    # ========================================================
    # FETCH DATA
    # ========================================================

    buildings = safe_fetch(lat, lon, r, {"building": True})
    roads = keep_lines(
        safe_fetch(
            lat,
            lon,
            r,
            {"highway": ["motorway", "trunk", "primary", "secondary"]}
        )
    )

    light_rail = keep_lines(
        safe_fetch(
            lat,
            lon,
            r,
            {"railway": "light_rail"}
        )
    )

    stations = safe_fetch(
        lat,
        lon,
        r,
        {"railway": "station"}
    )

    water = safe_fetch(
        lat,
        lon,
        r,
        {"natural": "water"}
    )

    mtr_routes = keep_lines(
        safe_fetch(
            lat,
            lon,
            r,
            {"railway": ["rail", "subway"]}
        )
    )


    # ========================================================
    # SITE GEOMETRY
    # ========================================================

    if lot_gdf is None:

        if not buildings.empty:

            distances = buildings.geometry.distance(site_point)
            nearest_idx = distances.idxmin()
            site_geom = buildings.loc[nearest_idx, "geometry"]

        else:

            site_geom = site_point.buffer(40)

        site_gdf = gpd.GeoDataFrame(
            geometry=[site_geom],
            crs=3857
        )


    # ========================================================
    # MAP EXTENT
    # ========================================================

    HALF_W = 1900
    HALF_H = 1100

    xmin = site_point.x - HALF_W
    xmax = site_point.x + HALF_W
    ymin = site_point.y - HALF_H
    ymax = site_point.y + HALF_H

    clip_box = box(xmin, ymin, xmax, ymax)
    clip_gdf = gpd.GeoDataFrame(
        geometry=[clip_box],
        crs=3857
    )


    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(figsize=(18, 10))

    fig.patch.set_facecolor("#f4f4f4")
    ax.set_facecolor("#f4f4f4")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cx.add_basemap(
        ax,
        crs="EPSG:3857",
        source=cx.providers.CartoDB.Positron,
        zoom=15,
        alpha=0.5
    )


    # ========================================================
    # BASE LAYERS
    # ========================================================

    if not buildings.empty:
        buildings.plot(ax=ax, color=COLOR_BUILDINGS, alpha=0.5, zorder=1)

    if not water.empty:
        water.plot(ax=ax, color=COLOR_WATER, alpha=0.8, zorder=2)

    if not roads.empty:
        roads.plot(ax=ax, color=COLOR_ROADS, linewidth=2.2, zorder=3)


    # ========================================================
    # MTR ROUTES
    # ========================================================

    lines_on_map = {}

    if not mtr_routes.empty:

        mtr_visible = gpd.clip(mtr_routes, clip_gdf)

        if not mtr_visible.empty:

            name_col = "name" if "name" in mtr_visible.columns else None

            if name_col:

                for name in mtr_visible[name_col].dropna().unique():

                    clean = ''.join(c for c in name if ord(c) < 128).strip()

                    if clean == "":
                        continue

                    color = get_mtr_color(clean)

                    subset = mtr_visible[mtr_visible[name_col] == name]

                    subset.plot(ax=ax, color="white", linewidth=8, zorder=4)
                    subset.plot(ax=ax, color=color, linewidth=4.5, zorder=5)

                    if color not in lines_on_map:
                        lines_on_map[color] = clean.title()

                    merged = subset.union_all()

                    if merged.length < 600:
                        continue

                    midpoint = merged.interpolate(
                        0.5,
                        normalized=True
                    )

                    ax.text(
                        midpoint.x,
                        midpoint.y,
                        clean.upper(),
                        fontsize=9,
                        weight="bold",
                        color=color,
                        ha="center",
                        va="center",
                        zorder=12,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.85,
                            pad=2
                        )
                    )


    # ========================================================
    # LIGHT RAIL
    # ========================================================

    if not light_rail.empty:

        light_rail.plot(ax=ax, color="white", linewidth=6, zorder=4)
        light_rail.plot(ax=ax, color=COLOR_LIGHT_RAIL, linewidth=3.5, zorder=5)


    # ========================================================
    # STATIONS
    # ========================================================

    if not stations.empty:

        station_pts = stations.copy()

        station_pts["geometry"] = station_pts.centroid

        station_pts = station_pts[
            (station_pts.geometry.x >= xmin)
            & (station_pts.geometry.x <= xmax)
            & (station_pts.geometry.y >= ymin)
            & (station_pts.geometry.y <= ymax)
        ]

        placed = []

        for _, row in station_pts.iterrows():

            sx = row.geometry.x
            sy = row.geometry.y

            pt = Point(sx, sy)

            if any(pt.distance(p) < STATION_MIN_DISTANCE for p in placed):
                continue

            placed.append(pt)

            draw_station(ax, sx, sy)


    # ========================================================
    # SITE
    # ========================================================

    site_gdf.plot(
        ax=ax,
        facecolor=COLOR_SITE,
        edgecolor="none",
        zorder=13
    )

    centroid = site_geom.centroid

    ax.text(
        centroid.x,
        centroid.y - 120,
        "SITE",
        fontsize=14,
        weight="bold",
        ha="center",
        zorder=14
    )


    # ========================================================
    # NORTH ARROW
    # ========================================================

    ax.annotate(
        '',
        xy=(0.07, 0.85),
        xytext=(0.07, 0.80),
        xycoords=ax.transAxes,
        arrowprops=dict(
            facecolor='black',
            width=1.5,
            headwidth=8,
            headlength=10
        )
    )

    ax.text(
        0.07,
        0.86,
        "N",
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=12
    )


    # ========================================================
    # LEGEND
    # ========================================================

    legend_handles = []

    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color=COLOR_LIGHT_RAIL,
            linewidth=4,
            label="Light Rail"
        )
    )

    for color, label in lines_on_map.items():

        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=4,
                label=label
            )
        )

    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color=COLOR_ROADS,
            linewidth=4,
            label="Vehicle Circulation"
        )
    )

    legend_handles.append(
        mpatches.Patch(
            facecolor=COLOR_SITE,
            label="Site"
        )
    )

    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            marker='o',
            linestyle='None',
            markerfacecolor='#ED1D24',
            markeredgecolor='white',
            markeredgewidth=2,
            markersize=10,
            label="MTR Station"
        )
    )

    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        title="Legend"
    )

    legend.get_frame().set_linewidth(2)


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        f"SITE ANALYSIS – Transportation ({data_type} {value})",
        fontsize=18,
        weight="bold"
    )

    ax.set_axis_off()


    # ========================================================
    # EXPORT
    # ========================================================

    buffer = BytesIO()

    plt.savefig(
        buffer,
        format="png",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    gc.collect()

    buffer.seek(0)

    return buffer
