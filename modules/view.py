import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon
from matplotlib.patches import Wedge, Patch
from io import BytesIO

# ✅ IMPORT UNIVERSAL RESOLVER
from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache = True
ox.settings.log_console = False

FETCH_RADIUS = 1500
MAP_RADIUS = 800
VIEW_RADIUS = 360
ARC_WIDTH = 40
SECTOR_SIZE = 20


# ------------------------------------------------------------
# MAIN GENERATOR
# ------------------------------------------------------------

def generate_view(data_type: str, value: str, BUILDING_DATA: gpd.GeoDataFrame):

    # ✅ Dynamic resolver
    lon, lat = resolve_location(data_type, value)

    # --------------------------------------------------------
    # SITE POLYGON (official lot boundary or OSM fallback)
    # --------------------------------------------------------

    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_geom = lot_gdf.geometry.iloc[0]
        center = site_geom.centroid
    else:
        site_building = ox.features_from_point(
            (lat, lon),
            dist=60,
            tags={"building": True}
        ).to_crs(3857)

        if len(site_building):
            site_geom = (
                site_building.assign(area=site_building.area)
                .sort_values("area", ascending=False)
                .geometry.iloc[0]
            )
        else:
            site_geom = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0].buffer(25)

        center = site_geom.centroid
    analysis_circle = center.buffer(MAP_RADIUS)

    # --------------------------------------------------------
    # CONTEXT DATA
    # --------------------------------------------------------

    def fetch_layer(tags):
        gdf = ox.features_from_point((lat, lon), dist=FETCH_RADIUS, tags=tags).to_crs(3857)
        return gdf[gdf.intersects(analysis_circle)]

    buildings = fetch_layer({"building": True})
    parks = fetch_layer({"leisure":"park","landuse":"grass","natural":"wood"})
    water = fetch_layer({"waterway":True,"natural":"water"})

    nearby = BUILDING_DATA[BUILDING_DATA.intersects(analysis_circle)].copy()

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_facecolor("#f2f2f2")
    ax.set_xlim(center.x - MAP_RADIUS, center.x + MAP_RADIUS)
    ax.set_ylim(center.y - MAP_RADIUS, center.y + MAP_RADIUS)
    ax.set_aspect("equal")

    if len(parks):
        parks.plot(ax=ax, color="#b8c8a0", edgecolor="none", zorder=1)

    if len(water):
        water.plot(ax=ax, color="#6bb6d9", edgecolor="none", zorder=2)

    buildings.plot(ax=ax, color="#e3e3e3", edgecolor="none", zorder=3)

    # --------------------------------------------------------
    # RADIAL GUIDES
    # --------------------------------------------------------

    for angle in range(0, 360, SECTOR_SIZE):
        rad = np.radians(angle)
        ax.plot(
            [center.x, center.x + MAP_RADIUS*np.cos(rad)],
            [center.y, center.y + MAP_RADIUS*np.sin(rad)],
            linestyle=(0,(2,4)),
            linewidth=0.8,
            color="#d49a2a",
            alpha=0.35,
            zorder=4
        )

    # --------------------------------------------------------
    # MPD RINGS
    # --------------------------------------------------------

    for d in [80,160,200]:
        circle = center.buffer(d)
        gpd.GeoSeries([circle], crs=3857).boundary.plot(
            ax=ax,
            linestyle=(0,(4,4)),
            linewidth=1.2,
            color="#555555",
            alpha=0.9,
            zorder=5
        )

        ax.text(
            center.x + d,
            center.y,
            f"{d} mPD",
            fontsize=8,
            weight="bold",
            color="white",
            bbox=dict(facecolor="black", edgecolor="none", pad=2),
            zorder=6
        )

    # --------------------------------------------------------
    # SECTOR CREATION
    # --------------------------------------------------------

    def create_sector(start_angle, end_angle):
        angles = np.linspace(start_angle, end_angle, 40)
        points = [(center.x, center.y)]
        for angle in angles:
            rad = np.radians(angle)
            x = center.x + VIEW_RADIUS * np.cos(rad)
            y = center.y + VIEW_RADIUS * np.sin(rad)
            points.append((x, y))
        return Polygon(points)

    sector_data = []

    for angle in range(0, 360, SECTOR_SIZE):

        sector = create_sector(angle, angle + SECTOR_SIZE)
        sector_area = sector.area

        green_area = parks.intersection(sector).area.sum() if len(parks) else 0
        water_area = water.intersection(sector).area.sum() if len(water) else 0
        building_area = buildings.intersection(sector).area.sum() if len(buildings) else 0

        sector_heights = nearby[nearby.intersects(sector)]
        avg_height = sector_heights["HEIGHT_M"].mean() if len(sector_heights) else 0

        sector_data.append({
            "start": angle,
            "end": angle + SECTOR_SIZE,
            "green": green_area / sector_area if sector_area else 0,
            "water": water_area / sector_area if sector_area else 0,
            "building": building_area / sector_area if sector_area else 0,
            "avg_height": avg_height
        })

    df = pd.DataFrame(sector_data)

    # --------------------------------------------------------
    # NORMALIZATION + SCORING
    # --------------------------------------------------------

    def normalize(series):
        if series.max() - series.min() == 0:
            return series * 0
        return (series - series.min()) / (series.max() - series.min())

    df["green_n"] = normalize(df["green"])
    df["water_n"] = normalize(df["water"])
    df["height_n"] = normalize(df["avg_height"])
    df["density_n"] = normalize(df["building"])

    df["city_score"] = df["height_n"] * df["density_n"]
    df["green_score"] = df["green_n"]
    df["water_score"] = df["water_n"]
    df["open_score"] = (1 - df["density_n"]) * (1 - df["height_n"])

    df["view"] = df[["green_score","water_score","city_score","open_score"]].idxmax(axis=1)
    df["view"] = df["view"].str.replace("_score","").str.upper()

    # --------------------------------------------------------
    # MERGE SECTORS
    # --------------------------------------------------------

    merged = []
    current_start = df.iloc[0]["start"]
    current_end = df.iloc[0]["end"]
    current_type = df.iloc[0]["view"]

    for i in range(1, len(df)):
        row = df.iloc[i]
        if row["view"] == current_type:
            current_end = row["end"]
        else:
            merged.append((current_start, current_end, current_type))
            current_start = row["start"]
            current_end = row["end"]
            current_type = row["view"]

    merged.append((current_start, current_end, current_type))

    if merged[0][2] == merged[-1][2]:
        merged[0] = (merged[-1][0], merged[0][1], merged[0][2])
        merged.pop()

    # --------------------------------------------------------
    # DRAW VIEW ARCS
    # --------------------------------------------------------

    color_map = {
        "GREEN": "#3dbb74",
        "WATER": "#4fa3d1",
        "CITY": "#e75b8c",
        "OPEN": "#f0a25a"
    }

    for start, end, view_type in merged:
        arc = Wedge(
            (center.x, center.y),
            VIEW_RADIUS,
            start,
            end,
            width=ARC_WIDTH,
            facecolor=color_map[view_type],
            edgecolor="white",
            linewidth=2,
            zorder=7
        )
        ax.add_patch(arc)

    # --------------------------------------------------------
    # HEIGHT LABELS
    # --------------------------------------------------------

    top_buildings = nearby.sort_values("HEIGHT_M", ascending=False).head(25)

    for _, row in top_buildings.iterrows():
        centroid = row.geometry.centroid
        ax.text(
            centroid.x,
            centroid.y,
            f"{row['HEIGHT_M']:.1f} m",
            fontsize=7,
            color="white",
            bbox=dict(facecolor="black", edgecolor="none", pad=1.5),
            zorder=12
        )

    # --------------------------------------------------------
    # SITE
    # --------------------------------------------------------

    gpd.GeoSeries([site_geom]).plot(
        ax=ax,
        facecolor="#e74c3c",
        edgecolor="white",
        linewidth=1.5,
        zorder=13
    )

    ax.text(center.x, center.y - 35, "SITE", fontsize=12, weight="bold", ha="center", va="top")

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend_elements = [
        Patch(facecolor="#3dbb74", label="Green View"),
        Patch(facecolor="#4fa3d1", label="Water View"),
        Patch(facecolor="#e75b8c", label="City View"),
        Patch(facecolor="#f0a25a", label="Open View"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#444444",
        framealpha=0.95,
        fontsize=9
    )

    # ✅ UPDATED TITLE
    ax.set_title(
        f"SITE ANALYSIS – View Analysis ({data_type} {value})",
        fontsize=16,
        weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=200)
    plt.close(fig)

    return buffer
