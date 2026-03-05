import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon
from matplotlib.patches import Wedge, Patch
from io import BytesIO

# IMPORT UNIVERSAL RESOLVER
from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache = True
ox.settings.log_console = False

# ── Radii ──────────────────────────────────────────────────────────────────────
FETCH_RADIUS  = 500    # OSM data fetch radius (kept larger than MAP_RADIUS)
MAP_RADIUS    = 200    # visible map extent  ← changed from 800
VIEW_RADIUS   = 160    # arc ring outer edge (80 % of MAP_RADIUS feels balanced)
ARC_WIDTH     = 30     # arc ring thickness
CITY_RADIUS   = 30     # only buildings within 30 m count for CITY view
SECTOR_SIZE   = 20     # degrees per wedge


# ── Colour palette ─────────────────────────────────────────────────────────────
COLOR_MAP = {
    "GREEN": "#3dbb74",
    "WATER": "#4fa3d1",
    "CITY":  "#e75b8c",
    "OPEN":  "#f0a25a",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: build sector polygon
# ──────────────────────────────────────────────────────────────────────────────
def _sector_polygon(cx, cy, radius, start_deg, end_deg, steps=40):
    angles = np.linspace(start_deg, end_deg, steps)
    pts = [(cx, cy)]
    for a in angles:
        r = np.radians(a)
        pts.append((cx + radius * np.cos(r), cy + radius * np.sin(r)))
    return Polygon(pts)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: draw one view-analysis panel
# ──────────────────────────────────────────────────────────────────────────────
def _draw_panel(ax, center, site_geom, buildings, parks, water,
                sector_rows, title_suffix):
    """Render a single view-analysis panel onto *ax*."""

    ax.set_facecolor("#f2f2f2")
    ax.set_xlim(center.x - MAP_RADIUS, center.x + MAP_RADIUS)
    ax.set_ylim(center.y - MAP_RADIUS, center.y + MAP_RADIUS)
    ax.set_aspect("equal")

    # --- background layers ---
    if len(parks):
        parks.plot(ax=ax, color="#b8c8a0", edgecolor="none", zorder=1)
    if len(water):
        water.plot(ax=ax, color="#6bb6d9", edgecolor="none", zorder=2)
    buildings.plot(ax=ax, color="#e3e3e3", edgecolor="none", zorder=3)

    # --- radial guide lines ---
    for angle in range(0, 360, SECTOR_SIZE):
        rad = np.radians(angle)
        ax.plot(
            [center.x, center.x + MAP_RADIUS * np.cos(rad)],
            [center.y, center.y + MAP_RADIUS * np.sin(rad)],
            linestyle=(0, (2, 4)), linewidth=0.8,
            color="#d49a2a", alpha=0.35, zorder=4,
        )

    # --- distance rings (80 / 160 / 200 m) ---
    for d, label in [(80, "80 m"), (160, "160 m"), (200, "200 m")]:
        gpd.GeoSeries([center.buffer(d)], crs=3857).boundary.plot(
            ax=ax, linestyle=(0, (4, 4)), linewidth=1.2,
            color="#555555", alpha=0.9, zorder=5,
        )
        ax.text(
            center.x + d, center.y, label,
            fontsize=7, weight="bold", color="white",
            bbox=dict(facecolor="black", edgecolor="none", pad=2), zorder=6,
        )

    # --- view arcs ---
    for row in sector_rows:
        arc = Wedge(
            (center.x, center.y), VIEW_RADIUS,
            row["start"], row["end"],
            width=ARC_WIDTH,
            facecolor=COLOR_MAP[row["view"]],
            edgecolor="white", linewidth=1.5, zorder=7,
        )
        ax.add_patch(arc)

    # --- site polygon ---
    gpd.GeoSeries([site_geom]).plot(
        ax=ax, facecolor="#e74c3c", edgecolor="white",
        linewidth=1.5, zorder=13,
    )
    ax.text(center.x, center.y - 25, "SITE",
            fontsize=10, weight="bold", ha="center", va="top", zorder=14)

    # --- legend ---
    legend_elements = [Patch(facecolor=v, label=k.capitalize() + " View")
                       for k, v in COLOR_MAP.items()]
    ax.legend(handles=legend_elements, loc="lower right",
              frameon=True, facecolor="white", edgecolor="#444444",
              framealpha=0.95, fontsize=8)

    ax.set_title(title_suffix, fontsize=13, weight="bold", pad=10)
    ax.set_axis_off()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ──────────────────────────────────────────────────────────────────────────────
def generate_view(data_type: str, value: str, BUILDING_DATA: gpd.GeoDataFrame,
                  lon: float = None, lat: float = None,
                  lot_ids: list = None, extents: list = None):

    # ── 1. Resolve location ────────────────────────────────────────────────────
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)

    # ── 2. Site polygon ────────────────────────────────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_geom = lot_gdf.geometry.iloc[0]
        center = site_geom.centroid
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
            site_geom = (gpd.GeoSeries([Point(lon, lat)], crs=4326)
                         .to_crs(3857).iloc[0].buffer(25))
        center = site_geom.centroid

    analysis_circle = center.buffer(MAP_RADIUS)

    # ── 3. Context data (200 m radius) ─────────────────────────────────────────
    def fetch_layer(tags):
        gdf = ox.features_from_point(
            (lat, lon), dist=FETCH_RADIUS, tags=tags
        ).to_crs(3857)
        return gdf[gdf.intersects(analysis_circle)]

    buildings = fetch_layer({"building": True})
    parks     = fetch_layer({"leisure": "park", "landuse": "grass",
                              "natural": "wood"})
    water     = fetch_layer({"waterway": True, "natural": "water"})

    nearby = BUILDING_DATA[BUILDING_DATA.intersects(analysis_circle)].copy()

    # ── 4. Site building height ────────────────────────────────────────────────
    # Find building in BUILDING_DATA whose geometry contains the centroid,
    # or fall back to the closest building.
    H_max = 10.0   # default
    site_pt = center   # already in EPSG:3857

    if len(nearby):
        containing = nearby[nearby.geometry.contains(site_pt)]
        if len(containing):
            H_max = float(containing.iloc[0]["HEIGHT_M"])
        else:
            nearby_copy = nearby.copy()
            nearby_copy["_dist"] = nearby_copy.geometry.distance(site_pt)
            H_max = float(nearby_copy.sort_values("_dist").iloc[0]["HEIGHT_M"])

    H_mid = H_max / 2.0

    # ── 5. Buildings within CITY_RADIUS (30 m) ─────────────────────────────────
    city_circle = center.buffer(CITY_RADIUS)
    city_candidates = nearby[nearby.intersects(city_circle)].copy()

    # Compute bearing (°, 0 = East, CCW) for each city candidate centroid
    if len(city_candidates):
        city_candidates = city_candidates.copy()
        city_candidates["_angle"] = city_candidates.geometry.centroid.apply(
            lambda pt: np.degrees(np.arctan2(pt.y - center.y, pt.x - center.x)) % 360
        )

    # ── 6. Per-sector analysis ─────────────────────────────────────────────────
    def analyse_sectors(h_ref):
        rows = []
        for angle in range(0, 360, SECTOR_SIZE):
            start, end = angle, angle + SECTOR_SIZE
            sector = _sector_polygon(center.x, center.y,
                                     VIEW_RADIUS, start, end)
            sector_area = sector.area or 1.0

            # green / water / building coverage
            green_frac   = (parks.intersection(sector).area.sum()
                            if len(parks) else 0) / sector_area
            water_frac   = (water.intersection(sector).area.sum()
                            if len(water) else 0) / sector_area
            building_frac = (buildings.intersection(sector).area.sum()
                             if len(buildings) else 0) / sector_area

            # CITY: nearby building (≤ 30 m) in this wedge AND taller than h_ref
            is_city = False
            if len(city_candidates):
                in_wedge = city_candidates[
                    city_candidates["_angle"].apply(
                        lambda a: (start <= a < end)
                                  or (end > 360 and a < end - 360)
                    )
                ]
                if len(in_wedge) and (in_wedge["HEIGHT_M"] > h_ref).any():
                    is_city = True

            # Simple dominant-type logic
            # Priority: CITY (if flagged) → WATER → GREEN → OPEN
            if is_city:
                view = "CITY"
            elif water_frac > 0.05:
                view = "WATER"
            elif green_frac > building_frac and green_frac > 0.05:
                view = "GREEN"
            else:
                view = "OPEN"

            rows.append({
                "start": start, "end": end,
                "green": green_frac, "water": water_frac,
                "building": building_frac, "is_city": is_city,
                "view": view,
            })
        return rows

    rows_mid = analyse_sectors(H_mid)
    rows_max = analyse_sectors(H_max)

    # ── 7. Render two panels side-by-side ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))

    _draw_panel(
        axes[0], center, site_geom, buildings, parks, water,
        rows_mid,
        f"Mid-Height View  (ref = {H_mid:.1f} m)\n"
        f"Site building: {H_max:.1f} m  |  H_mid = {H_mid:.1f} m",
    )

    _draw_panel(
        axes[1], center, site_geom, buildings, parks, water,
        rows_max,
        f"Max-Height View  (ref = {H_max:.1f} m)\n"
        f"Site building: {H_max:.1f} m  |  H_max = {H_max:.1f} m",
    )

    fig.suptitle(
        f"SITE ANALYSIS – View Analysis ({data_type} {value})",
        fontsize=18, weight="bold", y=1.01,
    )

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buffer
