import matplotlib
matplotlib.use("Agg")

import gc
import os
import osmnx as ox
import geopandas as gpd
import contextily as cx
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point
from io import BytesIO

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache   = True
ox.settings.log_console = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

STUDY_RADIUS      = 300
GRID_RES          = 6
ZOOM              = 19
GROUND_ABSORPTION = 0.6
TRAFFIC_FLOW      = 1200
HEAVY_PERCENT     = 0.12
SPEED             = 40


# ------------------------------------------------------------
# TRAFFIC EMISSION MODEL
# ------------------------------------------------------------

def traffic_emission(flow, heavy_pct, speed):
    L_light = 27.7 + 10*np.log10(flow*(1 - heavy_pct) + 1e-9) + 0.02*speed
    L_heavy = 23.1 + 10*np.log10(flow*heavy_pct + 1e-9)        + 0.08*speed
    energy  = 10**(L_light/10) + 10**(L_heavy/10)
    return 10*np.log10(energy)


# ------------------------------------------------------------
# MAIN GENERATOR
# ------------------------------------------------------------

def generate_noise(data_type: str, value: str):

    lon, lat = resolve_location(data_type, value)

    # Anchor point is always raw lon/lat, NOT lot polygon centroid
    anchor_pt_3857 = gpd.GeoSeries(
        [Point(lon, lat)], crs=4326
    ).to_crs(3857).iloc[0]

    cx_val = anchor_pt_3857.x
    cy_val = anchor_pt_3857.y

    # --------------------------------------------------------
    # SITE POLYGON — only for display, does NOT affect grid
    # --------------------------------------------------------

    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_polygon = lot_gdf.geometry.iloc[0]
        site_gdf     = lot_gdf
    else:
        try:
            site_candidates = ox.features_from_point(
                (lat, lon), dist=60, tags={"building": True}
            ).to_crs(3857)
            site_candidates["area"] = site_candidates.area

            if len(site_candidates) == 0:
                raise ValueError("No buildings found.")

            site_polygon = site_candidates.sort_values(
                "area", ascending=False
            ).geometry.iloc[0]

        except Exception:
            site_polygon = anchor_pt_3857.buffer(40)

        site_gdf = gpd.GeoDataFrame(geometry=[site_polygon], crs=3857)

    # --------------------------------------------------------
    # ROADS
    # --------------------------------------------------------

    try:
        roads = ox.features_from_point(
            (lat, lon), dist=STUDY_RADIUS, tags={"highway": True}
        ).to_crs(3857)
        roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
    except Exception as e:
        raise ValueError(f"Roads fetch failed: {e}")

    if len(roads) == 0:
        raise ValueError("No roads found in study radius.")

    # --------------------------------------------------------
    # SOURCE LEVEL
    # --------------------------------------------------------

    L_source = traffic_emission(TRAFFIC_FLOW, HEAVY_PERCENT, SPEED)

    # --------------------------------------------------------
    # GRID — anchored to raw lon/lat point
    # --------------------------------------------------------

    minx = cx_val - STUDY_RADIUS
    maxx = cx_val + STUDY_RADIUS
    miny = cy_val - STUDY_RADIUS
    maxy = cy_val + STUDY_RADIUS

    x_vals = np.arange(minx, maxx, GRID_RES)
    y_vals = np.arange(miny, maxy, GRID_RES)

    X, Y         = np.meshgrid(x_vals, y_vals)
    noise_energy = np.zeros_like(X)

    # --------------------------------------------------------
    # PROPAGATION
    # --------------------------------------------------------

    for geom in roads.geometry:
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]

        for line in lines:
            d         = np.vectorize(lambda xx, yy: line.distance(Point(xx, yy)))(X, Y)
            A_div     = 20 * np.log10(d + 1)
            A_ground  = GROUND_ABSORPTION * 5 * np.log10(d + 1)
            A_reflect = -2
            L         = L_source - A_div - A_ground + A_reflect
            noise_energy += 10**(L/10)

    noise = 10 * np.log10(noise_energy + 1e-9)

    gc.collect()

    # --------------------------------------------------------
    # BUILDING FAÇADE EXPOSURE
    # --------------------------------------------------------

    try:
        buildings = ox.features_from_point(
            (lat, lon), dist=STUDY_RADIUS, tags={"building": True}
        ).to_crs(3857)
        buildings = buildings[
            buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
        ]

        facade_levels = []
        for geom in buildings.geometry:
            bcentroid = geom.centroid
            mask      = (np.abs(X - bcentroid.x) < GRID_RES) & \
                        (np.abs(Y - bcentroid.y) < GRID_RES)
            val       = float(np.mean(noise[mask])) if mask.any() else float(np.nan)
            facade_levels.append(val)

        buildings["facade_db"] = facade_levels

    except Exception:
        buildings = gpd.GeoDataFrame(geometry=[], crs=3857)

    gc.collect()

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#f4f4f4")
    ax.set_facecolor("#f4f4f4")

    ax.set_xlim(cx_val - STUDY_RADIUS, cx_val + STUDY_RADIUS)
    ax.set_ylim(cy_val - STUDY_RADIUS, cy_val + STUDY_RADIUS)
    ax.set_aspect("equal")

    # No labels 2D flat basemap
    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        crs=3857,
        zoom=ZOOM,
        alpha=1.0
    )

    # --------------------------------------------------------
    # NOISE CONTOURS
    # --------------------------------------------------------

    levels = np.arange(45, 105, 5)

    cont = ax.contourf(
        X, Y, noise,
        levels=levels,
        cmap="RdYlGn_r",
        alpha=0.45,
        zorder=2
    )

    ax.contour(
        X, Y, noise,
        levels=levels,
        colors="black",
        linewidths=0.4,
        alpha=0.3,
        zorder=3
    )

    # --------------------------------------------------------
    # BUILDINGS — façade color
    # --------------------------------------------------------

    if len(buildings) > 0 and "facade_db" in buildings.columns:
        buildings.plot(
            ax=ax,
            column="facade_db",
            cmap="RdYlGn_r",
            vmin=45,
            vmax=100,
            linewidth=0.8,
            edgecolor="#555555",
            alpha=0.85,
            zorder=4
        )

    # --------------------------------------------------------
    # SITE
    # --------------------------------------------------------

    site_gdf.plot(
        ax=ax,
        facecolor="red",
        edgecolor="none",
        zorder=10
    )

    # FIX: use site polygon centroid for SITE label, not anchor point
    site_center = site_gdf.geometry.iloc[0].centroid
    ax.text(
        site_center.x, site_center.y,
        "SITE",
        fontsize=14, weight="bold",
        color="white",
        ha="center", va="center",
        zorder=20
    )

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    cbar = plt.colorbar(cont, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Noise Level Leq dB(A)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # --------------------------------------------------------
    # NORTH ARROW
    # --------------------------------------------------------

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    nx   = xlim[0] + 0.07 * (xlim[1] - xlim[0])
    ny   = ylim[0] + 0.88 * (ylim[1] - ylim[0])
    L_ar = STUDY_RADIUS * 0.06

    ax.annotate(
        "", xy=(nx, ny + L_ar), xytext=(nx, ny - L_ar),
        arrowprops=dict(arrowstyle="-|>", color="black",
                        lw=2, mutation_scale=14),
        zorder=25
    )
    ax.text(
        nx, ny + L_ar * 1.4, "N",
        fontsize=12, weight="bold",
        ha="center", va="center", zorder=25
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ax.set_title(
        f"Near-Site Environmental Noise Assessment\n"
        f"{data_type} {value}\n"
        f"Traffic + Ground Absorption + Reflection Effects",
        fontsize=14, weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()

    return buffer
