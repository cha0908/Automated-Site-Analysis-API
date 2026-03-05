import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
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
FETCH_RADIUS = 500    # OSM data fetch radius
MAP_RADIUS   = 200    # visible map extent
VIEW_RADIUS  = 200    # outer edge of the view ring
ARC_WIDTH    = 30     # arc ring thickness
CITY_RADIUS  = 30     # only buildings within 30 m count for CITY view
SECTOR_SIZE  = 20     # degrees per wedge

# ── Colour palette ─────────────────────────────────────────────────────────────
COLOR_MAP = {
    "GREEN": "#3dbb74",
    "WATER": "#4fa3d1",
    "CITY":  "#e75b8c",
    "OPEN":  "#f0a25a",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_sector(cx, cy, radius, start_deg, end_deg, steps=40):
    """Return a pie-slice Polygon for one wedge."""
    angles = np.linspace(start_deg, end_deg, steps)
    pts = [(cx, cy)]
    for a in angles:
        r = np.radians(a)
        pts.append((cx + radius * np.cos(r), cy + radius * np.sin(r)))
    return Polygon(pts)


def _get_site_height(landsd_gdf, site_centroid):
    """
    Return HEIGHT_M of the building that contains the site centroid,
    or the closest building if none contains it.
    Falls back to 10.0 if the dataset is empty.
    """
    if not len(landsd_gdf):
        return 10.0
    containing = landsd_gdf[landsd_gdf.geometry.contains(site_centroid)]
    if len(containing):
        return float(containing.iloc[0]["HEIGHT_M"])
    tmp = landsd_gdf.copy()
    tmp["_dist"] = tmp.geometry.distance(site_centroid)
    return float(tmp.sort_values("_dist").iloc[0]["HEIGHT_M"])


def _classify_sectors(center, parks, water, city_candidates, h_ref):
    """
    Classify every SECTOR_SIZE wedge as GREEN / WATER / CITY / OPEN.
    Priority: CITY > WATER > GREEN > OPEN

    Returns list of dicts: [{start, end, view}, ...]
    """
    # Pre-compute bearing of every city candidate from site centre
    if len(city_candidates):
        cand = city_candidates.copy()
        cand["_angle"] = cand.geometry.centroid.apply(
            lambda pt: np.degrees(
                np.arctan2(pt.y - center.y, pt.x - center.x)
            ) % 360
        )
    else:
        cand = pd.DataFrame(columns=["HEIGHT_M", "_angle"])

    rows = []
    for angle in range(0, 360, SECTOR_SIZE):
        start, end  = angle, angle + SECTOR_SIZE
        sector      = _make_sector(center.x, center.y, VIEW_RADIUS, start, end)
        sector_area = sector.area or 1.0

        green_share = (parks.intersection(sector).area.sum()
                       if len(parks) else 0) / sector_area
        water_share = (water.intersection(sector).area.sum()
                       if len(water) else 0) / sector_area

        # CITY: candidate in this wedge AND taller than h_ref
        is_city = False
        if len(cand):
            in_wedge = cand[cand["_angle"].between(start, end, inclusive="left")]
            if len(in_wedge) and (in_wedge["HEIGHT_M"] > h_ref).any():
                is_city = True

        if is_city:
            view = "CITY"
        elif water_share > green_share and water_share > 0.02:
            view = "WATER"
        elif green_share > 0.02:
            view = "GREEN"
        else:
            view = "OPEN"

        rows.append({"start": start, "end": end, "view": view})

    return rows


def _merge_sectors(sector_rows):
    """
    Merge adjacent same-view wedges into larger arcs.
    Handles wrap-around (last arc merges with first if same type).
    Returns list of dicts: [{start, end, view}, ...]
    """
    merged = []
    cur = dict(sector_rows[0])

    for row in sector_rows[1:]:
        if row["view"] == cur["view"]:
            cur["end"] = row["end"]
        else:
            merged.append(cur)
            cur = dict(row)

    merged.append(cur)

    # Wrap-around: merge last into first if same view type
    if len(merged) > 1 and merged[0]["view"] == merged[-1]["view"]:
        merged[0]["start"] = merged[-1]["start"]
        merged.pop()

    return merged


def _draw_panel(ax, center, site_geom, buildings, parks, water,
                nearby, sector_rows, title):
    """
    Render one complete view-analysis panel onto *ax*.

    sector_rows : merged arc list [{start, end, view}, ...]
    nearby      : GeoDataFrame with HEIGHT_M column (BUILDING_DATA subset)
    """
    ax.set_facecolor("#f2f2f2")
    ax.set_xlim(center.x - MAP_RADIUS, center.x + MAP_RADIUS)
    ax.set_ylim(center.y - MAP_RADIUS, center.y + MAP_RADIUS)
    ax.set_aspect("equal")

    # ── background layers ────────────────────────────────────────────────────
    if len(parks):
        parks.plot(ax=ax, color="#b8c8a0", edgecolor="none", zorder=1)
    if len(water):
        water.plot(ax=ax, color="#6bb6d9", edgecolor="none", zorder=2)
    buildings.plot(ax=ax, color="#e3e3e3", edgecolor="none", zorder=3)

    # ── radial guide lines ───────────────────────────────────────────────────
    for angle in range(0, 360, SECTOR_SIZE):
        rad = np.radians(angle)
        ax.plot(
            [center.x, center.x + MAP_RADIUS * np.cos(rad)],
            [center.y, center.y + MAP_RADIUS * np.sin(rad)],
            linestyle=(0, (2, 4)), linewidth=0.8,
            color="#d49a2a", alpha=0.35, zorder=4,
        )

    # ── distance rings (80 / 160 / 200 m) ───────────────────────────────────
    for d, label in [(80, "80 m"), (160, "160 m"), (200, "200 m")]:
        gpd.GeoSeries([center.buffer(d)], crs=3857).boundary.plot(
            ax=ax, linestyle=(0, (4, 4)), linewidth=1.2,
            color="#555555", alpha=0.9, zorder=5,
        )
        ax.text(
            center.x + d + 3, center.y, label,
            fontsize=7, weight="bold", color="white",
            bbox=dict(facecolor="black", edgecolor="none", pad=2), zorder=6,
        )

    # ── view arcs + arc labels ───────────────────────────────────────────────
    for row in sector_rows:
        start, end, view_type = row["start"], row["end"], row["view"]

        # Normalise wrap-around arcs (e.g. start=340, end=40)
        arc_span = end - start
        if arc_span <= 0:
            arc_span += 360
        draw_end = start + arc_span   # Wedge handles values > 360 correctly

        arc = Wedge(
            (center.x, center.y),
            VIEW_RADIUS,
            start, draw_end,
            width=ARC_WIDTH,
            facecolor=COLOR_MAP[view_type],
            edgecolor="white", linewidth=1.5, zorder=7,
        )
        ax.add_patch(arc)

        # Label only when arc ≥ one sector wide
        if arc_span >= SECTOR_SIZE:
            mid_angle = start + arc_span / 2
            mid_rad   = np.radians(mid_angle)
            label_r   = VIEW_RADIUS - ARC_WIDTH / 2   # radial midpoint of ring

            lx = center.x + label_r * np.cos(mid_rad)
            ly = center.y + label_r * np.sin(mid_rad)

            # Tangent rotation so text follows the ring
            rotation = mid_angle - 90
            if 90 < mid_angle % 360 < 270:
                rotation += 180

            ax.text(
                lx, ly,
                f"{view_type} VIEW",
                fontsize=10, weight="bold", color="white",
                ha="center", va="center",
                rotation=rotation, rotation_mode="anchor",
                zorder=8,
            )

    # ── building height labels ────────────────────────────────────────────────
    # Top 12 tallest, outside 80 m inner ring, within 200 m analysis radius
    label_candidates = nearby.copy()
    label_candidates["_dist"] = label_candidates.geometry.centroid.apply(
        lambda pt: center.distance(pt)
    )
    label_candidates = label_candidates[
        (label_candidates["_dist"] > 80) &
        (label_candidates["_dist"] <= MAP_RADIUS)
    ]
    top_buildings = label_candidates.sort_values(
        "HEIGHT_M", ascending=False
    ).head(12)

    for _, brow in top_buildings.iterrows():
        centroid = brow.geometry.centroid
        ax.text(
            centroid.x, centroid.y,
            f"{brow['HEIGHT_M']:.1f} m",
            fontsize=7, color="white",
            bbox=dict(facecolor="black", edgecolor="none", pad=1.5),
            ha="center", va="center", zorder=12,
        )

    # ── site polygon ─────────────────────────────────────────────────────────
    gpd.GeoSeries([site_geom]).plot(
        ax=ax, facecolor="#e74c3c", edgecolor="white",
        linewidth=1.5, zorder=13,
    )
    ax.text(
        center.x, center.y - 20, "SITE",
        fontsize=11, weight="bold", ha="center", va="top", zorder=14,
    )

    # ── legend (bottom-right) ────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor="#3dbb74", label="Green View"),
        Patch(facecolor="#4fa3d1", label="Water View"),
        Patch(facecolor="#e75b8c", label="City View"),
        Patch(facecolor="#f0a25a", label="Open View"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        frameon=True, facecolor="white", edgecolor="#444444",
        framealpha=0.95, fontsize=8,
    )

    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    ax.set_axis_off()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR  (called by the API)
# ══════════════════════════════════════════════════════════════════════════════

def generate_view(data_type: str, value: str, BUILDING_DATA: gpd.GeoDataFrame,
                  lon: float = None, lat: float = None,
                  lot_ids: list = None, extents: list = None):
    """
    Generate a dual-panel (MID HEIGHT / MAX HEIGHT) view-analysis map.

    Parameters
    ----------
    data_type     : resolver data-type string (e.g. "LOT")
    value         : resolver value string    (e.g. "IL 1657")
    BUILDING_DATA : GeoDataFrame with HEIGHT_M column, CRS=3857
    lon, lat      : optional override coordinates (EPSG:4326)
    lot_ids       : optional lot ID list for resolver
    extents       : optional extent list for resolver

    Returns
    -------
    BytesIO  PNG image buffer
    """

    # ── 1. Resolve location ────────────────────────────────────────────────────
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)

    # ── 2. Site polygon ────────────────────────────────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_geom = lot_gdf.geometry.iloc[0]
        center    = site_geom.centroid
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

    # ── 3. Context data ────────────────────────────────────────────────────────
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
    H_max = _get_site_height(nearby, center)
    H_mid = H_max / 2.0

    # ── 5. City candidates (≤ 30 m from site centre) ──────────────────────────
    city_circle     = center.buffer(CITY_RADIUS)
    city_candidates = nearby[nearby.intersects(city_circle)].copy()

    # ── 6. Classify + merge sectors for both height levels ────────────────────
    raw_mid    = _classify_sectors(center, parks, water, city_candidates, H_mid)
    raw_max    = _classify_sectors(center, parks, water, city_candidates, H_max)
    merged_mid = _merge_sectors(raw_mid)
    merged_max = _merge_sectors(raw_max)

    title_mid = (f"SITE ANALYSIS – View Analysis (MID HEIGHT)\n"
                 f"ref = H_mid = {H_mid:.1f} m  (site building = {H_max:.1f} m)")
    title_max = (f"SITE ANALYSIS – View Analysis (MAX HEIGHT)\n"
                 f"ref = H_max = {H_max:.1f} m  (site building = {H_max:.1f} m)")

    # ── 7. Render two panels side-by-side ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))

    fig.suptitle(
        f"SITE ANALYSIS – View Analysis ({data_type} {value})",
        fontsize=18, weight="bold", y=1.01,
    )

    _draw_panel(axes[0], center, site_geom, buildings, parks, water,
                nearby, merged_mid, title_mid)

    _draw_panel(axes[1], center, site_geom, buildings, parks, water,
                nearby, merged_max, title_max)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buffer
