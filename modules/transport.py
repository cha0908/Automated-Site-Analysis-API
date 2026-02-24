import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from shapely.geometry import Point, box
from io import BytesIO

# ✅ IMPORT UNIVERSAL RESOLVER
from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache = True
ox.settings.log_console = False

MAP_RADIUS = 3000

COLOR_EXISTING_RAIL = "#e06a2b"
COLOR_MTR = "#3f78b5"
COLOR_ROADS = "#e85d9e"
COLOR_WATER = "#6fa8dc"
COLOR_BUILDINGS = "#d6d6d6"
COLOR_SITE = "#FF0000"


# ------------------------------------------------------------
# MAIN GENERATOR
# ------------------------------------------------------------

def generate_transport(data_type: str, value: str):

    # ✅ Dynamic resolver
    lon, lat = resolve_location(data_type, value)

    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_geom = lot_gdf.geometry.iloc[0]
        site_gdf = lot_gdf
        site_point = site_geom.centroid
    else:
        site_point = gpd.GeoSeries(
            [Point(lon, lat)],
            crs=4326
        ).to_crs(3857).iloc[0]

    # --------------------------------------------------------
    # SAFE FETCH
    # --------------------------------------------------------

    def safe_fetch(tags):
        try:
            gdf = ox.features_from_point((lat, lon), dist=MAP_RADIUS, tags=tags)
            if not gdf.empty:
                return gdf.to_crs(3857)
            return gpd.GeoDataFrame(geometry=[], crs=3857)
        except:
            return gpd.GeoDataFrame(geometry=[], crs=3857)

    def keep_lines(gdf):
        if not gdf.empty:
            return gdf[gdf.geometry.type.isin(["LineString","MultiLineString"])]
        return gpd.GeoDataFrame(geometry=[], crs=3857)

    # --------------------------------------------------------
    # FETCH DATA
    # --------------------------------------------------------

    buildings = safe_fetch({"building": True})
    roads = keep_lines(safe_fetch({"highway":["motorway","trunk","primary","secondary"]}))
    light_rail = keep_lines(safe_fetch({"railway":"light_rail"}))
    stations = safe_fetch({"railway":"station"})
    water = safe_fetch({"natural":"water"})
    mtr_routes = keep_lines(safe_fetch({"railway":["rail","subway"]}))

    # --------------------------------------------------------
    # SITE POLYGON (only when not using official lot boundary)
    # --------------------------------------------------------

    if lot_gdf is None:
        if not buildings.empty:
            distances = buildings.geometry.distance(site_point)
            nearest_idx = distances.idxmin()
            site_geom = buildings.loc[nearest_idx, "geometry"]
        else:
            site_geom = site_point.buffer(40)
        site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(18,10))
    fig.patch.set_facecolor("#f4f4f4")
    ax.set_facecolor("#f4f4f4")

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=15, alpha=0.5)

    if not buildings.empty:
        buildings.plot(ax=ax, color=COLOR_BUILDINGS, alpha=0.5, zorder=1)

    if not water.empty:
        water.plot(ax=ax, color=COLOR_WATER, alpha=0.8, zorder=2)

    if not roads.empty:
        roads.plot(ax=ax, color=COLOR_ROADS, linewidth=2.2, zorder=3)

    # --------------------------------------------------------
    # MTR ROUTES
    # --------------------------------------------------------

    xmin = site_point.x - 1600
    xmax = site_point.x + 2200
    ymin = site_point.y - 1100
    ymax = site_point.y + 1100

    clip_box = box(xmin, ymin, xmax, ymax)
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs=3857)

    if not mtr_routes.empty:

        mtr_visible = gpd.clip(mtr_routes, clip_gdf)

        if not mtr_visible.empty:

            mtr_visible.plot(ax=ax, color="white", linewidth=8, zorder=4)
            mtr_visible.plot(ax=ax, color=COLOR_MTR, linewidth=4.5, zorder=5)

            placed_positions = []

            if "name" in mtr_visible.columns:

                unique_names = mtr_visible["name"].dropna().unique()

                for name in unique_names:

                    clean_name = ''.join(c for c in name if ord(c) < 128).strip()
                    if clean_name == "":
                        continue

                    subset = mtr_visible[mtr_visible["name"] == name]
                    merged = subset.union_all()

                    if merged.length < 600:
                        continue

                    midpoint = merged.interpolate(0.5, normalized=True)

                    offset_y = 0
                    for pt in placed_positions:
                        if midpoint.distance(pt) < 500:
                            offset_y += 150

                    new_point = Point(midpoint.x, midpoint.y + offset_y)
                    placed_positions.append(new_point)

                    ax.text(
                        new_point.x,
                        new_point.y,
                        clean_name.upper(),
                        fontsize=9,
                        weight="bold",
                        color=COLOR_MTR,
                        ha="center",
                        va="center",
                        zorder=10,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2)
                    )

    # --------------------------------------------------------
    # LIGHT RAIL
    # --------------------------------------------------------

    if not light_rail.empty:
        light_rail.plot(ax=ax, color="white", linewidth=6, zorder=4)
        light_rail.plot(ax=ax, color=COLOR_EXISTING_RAIL, linewidth=3.5, zorder=5)

    # --------------------------------------------------------
    # STATIONS
    # --------------------------------------------------------

    if not stations.empty:
        station_pts = stations.copy()
        station_pts["geometry"] = station_pts.centroid

        station_pts.plot(
            ax=ax,
            facecolor="white",
            edgecolor=COLOR_EXISTING_RAIL,
            markersize=120,
            linewidth=2,
            zorder=6
        )

    # --------------------------------------------------------
    # SITE
    # --------------------------------------------------------

    site_gdf.plot(ax=ax, facecolor=COLOR_SITE, edgecolor="none", zorder=7)

    centroid = site_geom.centroid

    ax.text(
        centroid.x,
        centroid.y - 120,
        "SITE",
        fontsize=14,
        weight="bold",
        ha="center"
    )

    # --------------------------------------------------------
    # NORTH ARROW
    # --------------------------------------------------------

    ax.annotate(
        '',
        xy=(0.07, 0.85),
        xytext=(0.07, 0.80),
        xycoords=ax.transAxes,
        arrowprops=dict(facecolor='black', width=1.5, headwidth=8, headlength=10)
    )

    ax.text(
        0.07, 0.86,
        "N",
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=12
    )

    # --------------------------------------------------------
    # EXTENT
    # --------------------------------------------------------

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend = ax.legend(
        handles=[
            mlines.Line2D([], [], color=COLOR_EXISTING_RAIL, linewidth=5, label="Existing Light Rail"),
            mlines.Line2D([], [], color=COLOR_MTR, linewidth=5, label="MTR Line"),
            mlines.Line2D([], [], color=COLOR_ROADS, linewidth=5, label="Vehicle Circulation"),
            mpatches.Patch(facecolor=COLOR_SITE, label="Site"),
            mlines.Line2D([], [], marker='o', linestyle='None',
                         markerfacecolor='white',
                         markeredgecolor=COLOR_EXISTING_RAIL,
                         markeredgewidth=2,
                         markersize=9,
                         label="Rail Station")
        ],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.15),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        title="Legend"
    )

    legend.get_frame().set_linewidth(2)

    # ✅ UPDATED TITLE
    ax.set_title(
        f"SITE ANALYSIS – Transportation ({data_type} {value})",
        fontsize=18,
        weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=200)
    plt.close(fig)

    return buffer
