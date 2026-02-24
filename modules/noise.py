import osmnx as ox
import geopandas as gpd
import contextily as cx
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString
from io import BytesIO

# ✅ IMPORT UNIVERSAL RESOLVER
from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache = True
ox.settings.log_console = False

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

STUDY_RADIUS = 150
GRID_RES = 6
ZOOM = 19

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
# MAIN GENERATOR
# ------------------------------------------------------------

def generate_noise(data_type: str, value: str):

    # ✅ Dynamic resolver
    lon, lat = resolve_location(data_type, value)

    # --------------------------------------------------------
    # SITE POLYGON (official lot boundary or OSM fallback)
    # --------------------------------------------------------

    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_polygon = lot_gdf.geometry.iloc[0]
        site_gdf = lot_gdf
    else:
        site_point = gpd.GeoSeries(
            [Point(lon, lat)],
            crs=4326
        ).to_crs(3857).iloc[0]

        site_candidates = ox.features_from_point(
            (lat, lon),
            dist=60,
            tags={"building": True}
        ).to_crs(3857)

        site_candidates["area"] = site_candidates.area

        if len(site_candidates) == 0:
            raise ValueError("Site building footprint not found.")

        site_polygon = site_candidates.sort_values(
            "area", ascending=False
        ).geometry.iloc[0]

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

    if len(roads) == 0:
        raise ValueError("No roads found in study radius.")

    # --------------------------------------------------------
    # SOURCE LEVEL
    # --------------------------------------------------------

    L_source = traffic_emission(
        TRAFFIC_FLOW,
        HEAVY_PERCENT,
        SPEED
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    minx, miny, maxx, maxy = site_polygon.buffer(STUDY_RADIUS).bounds

    x_vals = np.arange(minx, maxx, GRID_RES)
    y_vals = np.arange(miny, maxy, GRID_RES)

    X, Y = np.meshgrid(x_vals, y_vals)

    noise_energy = np.zeros_like(X)

    # --------------------------------------------------------
    # PROPAGATION
    # --------------------------------------------------------

    for geom in roads.geometry:

        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]

        for line in lines:

            d = np.vectorize(
                lambda xx, yy: line.distance(Point(xx, yy))
            )(X, Y)

            A_div = 20*np.log10(d + 1)
            A_ground = GROUND_ABSORPTION * 5*np.log10(d + 1)

            A_reflect = -2

            L = L_source - A_div - A_ground + A_reflect

            noise_energy += 10**(L/10)

    noise = 10*np.log10(noise_energy + 1e-9)

    # --------------------------------------------------------
    # BUILDING FAÇADE EXPOSURE
    # --------------------------------------------------------

    buildings = ox.features_from_point(
        (lat, lon),
        dist=STUDY_RADIUS,
        tags={"building": True}
    ).to_crs(3857)

    buildings = buildings[
        buildings.geometry.type.isin(["Polygon","MultiPolygon"])
    ]

    facade_levels = []

    for geom in buildings.geometry:
        centroid = geom.centroid
        val = np.mean(noise[
            (np.abs(X-centroid.x)<GRID_RES) &
            (np.abs(Y-centroid.y)<GRID_RES)
        ])
        facade_levels.append(val)

    buildings["facade_db"] = facade_levels

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10,10))

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
        linewidths=0.4,
        alpha=0.3
    )

    buildings.plot(
        ax=ax,
        column="facade_db",
        cmap="RdYlGn_r",
        linewidth=0.2,
        edgecolor="black",
        alpha=0.9
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
        fontsize=14,
        weight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=20
    )

    cbar = plt.colorbar(cont, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Noise Level Leq dB(A)")

    # ✅ UPDATED TITLE
    ax.set_title(
        f"Near-Site Environmental Noise Assessment\n{data_type} {value}",
        fontsize=14,
        weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=200)
    plt.close(fig)

    return buffer
