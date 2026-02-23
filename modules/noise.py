import osmnx as ox
import geopandas as gpd
import contextily as cx
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point
from io import BytesIO

ox.settings.use_cache = True
ox.settings.log_console = False

# ------------------------------------------------------------
# OPTIMIZED SETTINGS
# ------------------------------------------------------------

STUDY_RADIUS = 140     # slightly reduced
GRID_RES = 10          # BIG performance boost (was 6)
ZOOM = 18              # reduced from 19

TRAFFIC_FLOW = 1200
HEAVY_PERCENT = 0.12
SPEED = 40
GROUND_ABSORPTION = 0.6


# ------------------------------------------------------------
# TRAFFIC EMISSION MODEL
# ------------------------------------------------------------

def traffic_emission(flow, heavy_pct, speed):

    L_light = 27.7 + 10*np.log10(flow*(1-heavy_pct)) + 0.02*speed
    L_heavy = 23.1 + 10*np.log10(flow*heavy_pct) + 0.08*speed

    energy = 10**(L_light/10) + 10**(L_heavy/10)
    return 10*np.log10(energy)


# ------------------------------------------------------------
# MAIN GENERATOR (OPTIMIZED)
# ------------------------------------------------------------

def generate_noise(lon: float, lat: float):

    site_point = gpd.GeoSeries(
        [Point(lon, lat)],
        crs=4326
    ).to_crs(3857).iloc[0]

    # --------------------------------------------------------
    # SITE FOOTPRINT
    # --------------------------------------------------------

    site_candidates = ox.features_from_point(
        (lat, lon),
        dist=60,
        tags={"building": True}
    ).to_crs(3857)

    if site_candidates.empty:
        raise ValueError("Site building footprint not found.")

    site_candidates["area"] = site_candidates.area
    site_polygon = site_candidates.sort_values("area", ascending=False).geometry.iloc[0]
    site_gdf = gpd.GeoDataFrame(geometry=[site_polygon], crs=3857)

    # --------------------------------------------------------
    # ROADS
    # --------------------------------------------------------

    roads = ox.features_from_point(
        (lat, lon),
        dist=STUDY_RADIUS,
        tags={"highway": True}
    ).to_crs(3857)

    roads = roads[roads.geometry.type.isin(["LineString","MultiLineString"])]

    if roads.empty:
        raise ValueError("No roads found.")

    # --------------------------------------------------------
    # SOURCE LEVEL
    # --------------------------------------------------------

    L_source = traffic_emission(
        TRAFFIC_FLOW,
        HEAVY_PERCENT,
        SPEED
    )

    # --------------------------------------------------------
    # GRID (COARSER)
    # --------------------------------------------------------

    minx, miny, maxx, maxy = site_polygon.buffer(STUDY_RADIUS).bounds

    x_vals = np.arange(minx, maxx, GRID_RES)
    y_vals = np.arange(miny, maxy, GRID_RES)

    X, Y = np.meshgrid(x_vals, y_vals)

    noise_energy = np.zeros_like(X)

    # --------------------------------------------------------
    # FASTER DISTANCE COMPUTATION
    # --------------------------------------------------------

    for geom in roads.geometry:

        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]

        for line in lines:

            # Use simple bounding box distance approximation
            d = np.sqrt((X - line.centroid.x)**2 + (Y - line.centroid.y)**2)

            A_div = 20*np.log10(d + 1)
            A_ground = GROUND_ABSORPTION * 5*np.log10(d + 1)
            A_reflect = -2

            L = L_source - A_div - A_ground + A_reflect
            noise_energy += 10**(L/10)

    noise = 10*np.log10(noise_energy + 1e-9)

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9,9))

    center = site_polygon.centroid

    ax.set_xlim(center.x - STUDY_RADIUS, center.x + STUDY_RADIUS)
    ax.set_ylim(center.y - STUDY_RADIUS, center.y + STUDY_RADIUS)

    cx.add_basemap(
        ax,
        source=cx.providers.Esri.WorldImagery,
        crs=3857,
        zoom=ZOOM,
        alpha=1
    )

    levels = np.arange(45, 105, 5)

    cont = ax.contourf(
        X, Y, noise,
        levels=levels,
        cmap="RdYlGn_r",
        alpha=0.45
    )

    ax.contour(
        X, Y, noise,
        levels=levels,
        colors="black",
        linewidths=0.3,
        alpha=0.25
    )

    site_gdf.plot(
        ax=ax,
        facecolor="red",
        edgecolor="none",
        zorder=10
    )

    ax.text(
        center.x,
        center.y,
        "SITE",
        fontsize=12,
        weight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=20
    )

    cbar = plt.colorbar(cont, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Noise Level Leq dB(A)")

    ax.set_title(
        "Near-Site Environmental Noise Assessment",
        fontsize=13,
        weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=130)  # reduced DPI
    plt.close(fig)

    return buffer
