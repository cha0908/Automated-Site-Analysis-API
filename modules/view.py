import logging
import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon

log = logging.getLogger(__name__)
from shapely.ops import unary_union
from matplotlib.patches import Wedge, Patch
from io import BytesIO

# IMPORT UNIVERSAL RESOLVER
from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache = True
ox.settings.log_console = False

# ── Radii ──────────────────────────────────────────────────────────────────────
FETCH_RADIUS         = 500  # OSM data fetch radius
MAP_RADIUS           = 200  # visible map extent and view-analysis radius
VIEW_RADIUS          = 200  # outer edge of the view ring
ARC_WIDTH            = 30   # arc ring thickness
CITY_RADIUS          = 50   # only buildings within 30 m count for CITY view
SITE_HEIGHT_RADIUS_M = 60   # fallback radius for site height when no lot boundary
SECTOR_SIZE          = 20   # degrees per wedge
COASTLINE_BUFFER_M   = 2    # buffer coastline lines to polygon for sea-view area

# ── Colour palette ─────────────────────────────────────────────────────────────
COLOR_MAP = {
    "PARK":      "#3dbb74",
    "GREEN":     "#86efac",
    "MOUNTAIN":  "#4b5563",
    "HARBOR":    "#2e86ab",
    "RESERVOIR": "#7eb8da",
    "SEA":       "#4fa3d1",
    "CITY":      "#e75b8c",
}

# Water view types for sector classification (precedence: reservoir > harbor > sea)
WATER_VIEW_TYPES = ("RESERVOIR", "HARBOR", "SEA")


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


def _water_type_from_osm(row) -> str:
    """
    Classify OSM water feature as RESERVOIR, HARBOR, or SEA.
    Conservative: harbor only when explicit (e.g. harbour=yes); else seaview.
    """
    natural = str(row.get("natural") or "").lower()
    water = str(row.get("water") or "").lower()
    harbour_val = row.get("harbour") or row.get("harbor")
    harbour = str(harbour_val or "").lower() if harbour_val is not None else ""
    if natural == "water" and water == "reservoir":
        return "RESERVOIR"
    if harbour in ("yes", "true", "1"):
        return "HARBOR"
    return "SEA"


def _build_water_layers(gdf_all, crs="EPSG:3857"):
    """
    Build non-overlapping water layers: reservoir, harbor, sea.
    Precedence: reservoir > harbor > sea (so overlapping areas are assigned once).
    Returns (water_reservoir_gdf, water_harbor_gdf, water_sea_gdf, water_combined_gdf).
    """
    empty = gpd.GeoDataFrame(geometry=[], crs=crs)
    if gdf_all.empty or "geometry" not in gdf_all.columns:
        return empty, empty, empty, empty

    geoms_by_type = {"RESERVOIR": [], "HARBOR": [], "SEA": []}
    for idx, row in gdf_all.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.type in ("LineString", "MultiLineString"):
            geom = geom.buffer(COASTLINE_BUFFER_M)
            wtype = "SEA"
        else:
            wtype = _water_type_from_osm(row)
        geoms_by_type[wtype].append(geom)

    def _to_gdf(geoms):
        if not geoms:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
        return gpd.GeoDataFrame(geometry=geoms, crs=crs)

    gdf_res = _to_gdf(geoms_by_type["RESERVOIR"])
    gdf_har = _to_gdf(geoms_by_type["HARBOR"])
    gdf_sea = _to_gdf(geoms_by_type["SEA"])

    # Precedence: reservoir > harbor > sea — clip lower by higher so no double-count
    union_res = unary_union(gdf_res.geometry) if len(gdf_res) else None
    union_har = unary_union(gdf_har.geometry) if len(gdf_har) else None

    if union_res is not None and not union_res.is_empty:
        if len(gdf_har):
            gdf_har = gpd.GeoDataFrame(
                geometry=gdf_har.geometry.difference(union_res),
                crs=crs,
            )
            gdf_har = gdf_har[~gdf_har.geometry.is_empty & ~gdf_har.geometry.isna()]
        if len(gdf_sea):
            gdf_sea = gpd.GeoDataFrame(
                geometry=gdf_sea.geometry.difference(union_res),
                crs=crs,
            )
            gdf_sea = gdf_sea[~gdf_sea.geometry.is_empty & ~gdf_sea.geometry.isna()]
    if union_har is not None and not union_har.is_empty and len(gdf_sea):
        gdf_sea = gpd.GeoDataFrame(
            geometry=gdf_sea.geometry.difference(union_har),
            crs=crs,
        )
        gdf_sea = gdf_sea[~gdf_sea.geometry.is_empty & ~gdf_sea.geometry.isna()]

    # Combined for basemap (all water)
    all_geoms = []
    for g in (gdf_res, gdf_har, gdf_sea):
        if len(g):
            all_geoms.append(g.geometry)
    water_combined = gpd.GeoDataFrame(
        geometry=pd.concat(all_geoms, ignore_index=True) if all_geoms else [],
        crs=crs,
    )
    return gdf_res, gdf_har, gdf_sea, water_combined


def _get_site_height(landsd_gdf, site_centroid):
    """
    Return HEIGHT_M of the building that contains the site centroid,
    or the closest building if none contains it.
    Falls back to 10.0 if the dataset is empty.
    """
    if not len(landsd_gdf):
        log.info("[view] site height: no buildings in radius → fallback 10.0 m")
        return 10.0
    containing = landsd_gdf[landsd_gdf.geometry.contains(site_centroid)]
    if len(containing):
        h = float(containing.iloc[0]["HEIGHT_M"])
        log.info("[view] site height: building contains centroid → HEIGHT_M=%.1f m (from %d containing)", h, len(containing))
        return h
    tmp = landsd_gdf.copy()
    tmp["_dist"] = tmp.geometry.distance(site_centroid)
    best = tmp.sort_values("_dist").iloc[0]
    h = float(best["HEIGHT_M"])
    dist_m = float(best["_dist"])
    log.info("[view] site height: no containing building → using closest at %.1f m → HEIGHT_M=%.1f m", dist_m, h)
    return h


def _fallback_view_from_neighbors(left_view: str, right_view: str) -> str:
    """Pick view from two adjacent wedges by hierarchy: water > green > city."""
    WATER = ("RESERVOIR", "HARBOR", "SEA")
    GREEN = ("PARK", "MOUNTAIN", "GREEN")
    v1 = left_view if left_view != "FALLBACK" else "CITY"
    v2 = right_view if right_view != "FALLBACK" else "CITY"
    if v1 in WATER or v2 in WATER:
        return v1 if v1 in WATER else v2
    if v1 in GREEN or v2 in GREEN:
        return v1 if v1 in GREEN else v2
    return "CITY"


def _classify_sectors(center, parks, mountains, green,
                     water_reservoir, water_harbor, water_sea,
                     city_candidates, h_ref, nearby, h_site):
    """
    Classify every SECTOR_SIZE wedge as PARK / HARBOR / RESERVOIR / SEA / CITY.
    Priority: water/green views first, then CITY where buildings block.
    The previous \"city-candidate\" rule (buildings within CITY_RADIUS forcing
    CITY) is disabled; we now rely solely on the blocking logic that compares
    building distance against the distance to view content.
    """
    # City-candidate rule disabled → keep an empty placeholder so the signature
    # stays compatible, but we never treat a sector as CITY purely because of
    # proximity to a tall building. All CITY views come from blocking logic.
    cand = pd.DataFrame(columns=["HEIGHT_M", "_angle"])

    # Pre-compute bearing and distance for all nearby (for blocking check)
    if len(nearby):
        nb = nearby.copy()
        nb["_angle"] = nb.geometry.centroid.apply(
            lambda pt: np.degrees(
                np.arctan2(pt.y - center.y, pt.x - center.x)
            ) % 360
        )
        nb["_dist"] = nb.geometry.centroid.apply(lambda pt: center.distance(pt))
    else:
        nb = pd.DataFrame(columns=["HEIGHT_M", "_angle", "_dist"])

    # Combined water for blocking distance (union of all three)
    water_combined = _to_combined_water(water_reservoir, water_harbor, water_sea)

    rows = []
    for angle in range(0, 360, SECTOR_SIZE):
        start, end  = angle, angle + SECTOR_SIZE
        sector      = _make_sector(center.x, center.y, VIEW_RADIUS, start, end)
        sector_area = sector.area or 1.0

        park_share = (parks.intersection(sector).area.sum()
                      if len(parks) else 0) / sector_area
        mountain_share = (mountains.intersection(sector).area.sum()
                          if len(mountains) else 0) / sector_area
        green_share = (green.intersection(sector).area.sum()
                       if len(green) else 0) / sector_area
        res_share = (water_reservoir.intersection(sector).area.sum()
                     if len(water_reservoir) else 0) / sector_area
        har_share = (water_harbor.intersection(sector).area.sum()
                     if len(water_harbor) else 0) / sector_area
        sea_share = (water_sea.intersection(sector).area.sum()
                     if len(water_sea) else 0) / sector_area
        water_share = res_share + har_share + sea_share

        # CITY via city-candidates disabled; we always start with non-CITY and
        # let the blocking logic convert to CITY when appropriate.
        is_city = False

        # Mountain presence: any cliff/peak feature intersecting this sector
        has_mountain = False
        n_mountains = 0
        min_mountain_dist = None
        if len(mountains):
            try:
                mts_in_sector = mountains[mountains.intersects(sector)]
                n_mountains = len(mts_in_sector)
                has_mountain = n_mountains > 0
                if n_mountains:
                    # Distance from sector centre to nearest mountain element (by centroid)
                    dists = mts_in_sector.geometry.centroid.apply(lambda g: center.distance(g))
                    if len(dists):
                        min_mountain_dist = float(dists.min())
            except Exception:
                # Fallback: conservative default - treat any intersection as mountain present
                try:
                    mts_in_sector = mountains[mountains.intersects(sector)]
                    n_mountains = len(mts_in_sector)
                    has_mountain = n_mountains > 0
                except Exception:
                    has_mountain = len(mountains.intersection(sector)) > 0

        if is_city:
            view = "CITY"

            in_wedge = cand[cand["_angle"].between(start, end, inclusive="left")]
            taller = in_wedge[in_wedge["HEIGHT_M"] > h_ref]
            if len(taller) > 0:
                log.info(
                    "[view-city-candidate] sector start=%3d end=%3d: %d bldg(s) within 30m (taller than h_ref=%.1f m), heights=%s; water_share=%.2f (would-be view if not city)",
                    start, end, len(taller), h_ref, taller["HEIGHT_M"].tolist(), water_share,
                )

        elif water_share > 0.02 and water_share > max(park_share, mountain_share, green_share):
            view = max(
                (("RESERVOIR", res_share), ("HARBOR", har_share), ("SEA", sea_share)),
                key=lambda x: x[1],
            )[0]
        # Mountain view only when there is a mountain feature AND either:
        # - substantial green share (forested slope), or
        # - multiple cliff/peak features in this sector.
        elif has_mountain and (green_share > 0.10 or n_mountains >= 2):
            view = "MOUNTAIN"
        elif park_share > 0.02:
            view = "PARK"
        elif green_share > 0.02:
            view = "GREEN"
        else:
            # No strong signal: will be set from the two adjacent wedges (water > green > city)
            view = "FALLBACK"

        # Blocking check: if view is PARK / MOUNTAIN / GREEN or any water type,
        # override to CITY when a building taller than the site in this wedge
        # is closer than the content.
        if view in ("PARK", "MOUNTAIN", "GREEN", "HARBOR", "RESERVOIR", "SEA") and len(nb):
            in_wedge_nb = nb[
                nb["_angle"].between(start, end, inclusive="left") &
                (nb["_dist"] <= VIEW_RADIUS)
            ].sort_values("_dist")
            taller = in_wedge_nb[in_wedge_nb["HEIGHT_M"] > h_site]
            if len(taller):
                d_block = float(taller.iloc[0]["_dist"])
                if view == "PARK":
                    park_geom = parks.intersection(sector).unary_union
                    content_dist = float("inf") if park_geom.is_empty else center.distance(park_geom)
                elif view == "MOUNTAIN":
                    mountain_geom = mountains.intersection(sector).unary_union if len(mountains) else None
                    content_dist = float("inf") if not mountain_geom or mountain_geom.is_empty else center.distance(mountain_geom)
                else:
                    water_geom = water_combined.intersection(sector).unary_union
                    content_dist = float("inf") if water_geom.is_empty else center.distance(water_geom)
                if d_block < content_dist:
                    try:
                        log.info(
                            "[view-blocked] start=%3d end=%3d prev_view=%-8s "
                            "d_block=%.1f content_dist=%.1f",
                            start,
                            end,
                            view,
                            d_block,
                            content_dist,
                        )
                    except Exception:
                        pass
                    view = "CITY"

        # Collect per-sector data; logging happens after fallback + blocking so
        # we can report the final view. We keep whether this sector originally
        # came from FALLBACK so logs can show e.g. "SEA(FALLBACK)".
        rows.append({
            "start": start,
            "end": end,
            "view": view,
            "origin": "FALLBACK" if view == "FALLBACK" else "PRIMARY",
            "park_share": park_share,
            "mountain_share": mountain_share,
            "green_share": green_share,
            "water_share": water_share,
            "is_city": is_city,
            "n_mountains": n_mountains,
            "min_mountain_dist": min_mountain_dist,
        })

    # Resolve FALLBACK sectors from the two adjacent wedges (water > green > city)
    n = len(rows)
    for i in range(n):
        if rows[i]["view"] == "FALLBACK":
            left_v = rows[(i - 1) % n]["view"]
            right_v = rows[(i + 1) % n]["view"]
            rows[i]["view"] = _fallback_view_from_neighbors(left_v, right_v)

    # Blocking check for sectors that were FALLBACK (and any water/park/green/mountain):
    # override to CITY when a building taller than the site in that wedge is
    # closer than the content.
    for i in range(n):
        view = rows[i]["view"]
        if view not in ("PARK", "MOUNTAIN", "GREEN", "HARBOR", "RESERVOIR", "SEA") or not len(nb):
            continue
        start, end = rows[i]["start"], rows[i]["end"]
        sector = _make_sector(center.x, center.y, VIEW_RADIUS, start, end)
        in_wedge_nb = nb[
            nb["_angle"].between(start, end, inclusive="left") &
            (nb["_dist"] <= VIEW_RADIUS)
        ].sort_values("_dist")
        taller = in_wedge_nb[in_wedge_nb["HEIGHT_M"] > h_site]
        if not len(taller):
            continue
        d_block = float(taller.iloc[0]["_dist"])
        if view == "PARK":
            park_geom = parks.intersection(sector).unary_union
            content_dist = float("inf") if park_geom.is_empty else center.distance(park_geom)
        elif view == "MOUNTAIN":
            mountain_geom = mountains.intersection(sector).unary_union if len(mountains) else None
            content_dist = float("inf") if not mountain_geom or mountain_geom.is_empty else center.distance(mountain_geom)
        else:
            water_geom = water_combined.intersection(sector).unary_union
            content_dist = float("inf") if water_geom.is_empty else center.distance(water_geom)
        if d_block < content_dist:
            try:
                log.info(
                    "[view-blocked] start=%3d end=%3d prev_view=%-8s "
                    "d_block=%.1f content_dist=%.1f (post-fallback)",
                    start,
                    end,
                    view,
                    d_block,
                    content_dist,
                )
            except Exception:
                pass
            rows[i]["view"] = "CITY"

    # Per-sector debug log (after fallback + blocking) so we see the final
    # classification. If the sector originally came from FALLBACK, annotate
    # the view as e.g. "SEA(FALLBACK)".
    for row in rows:
        try:
            logged_view = row["view"]
            if row.get("origin") == "FALLBACK":
                logged_view = f"{logged_view}(FALLBACK)"
            log.info(
                "[view-sector] start=%3d end=%3d view=%-16s park=%.2f mountain=%.2f "
                "green=%.2f water=%.2f city=%s n_mtn=%d min_mtn_dist=%s",
                row["start"],
                row["end"],
                logged_view,
                row["park_share"],
                row["mountain_share"],
                row["green_share"],
                row["water_share"],
                "Y" if row.get("is_city") else "N",
                row["n_mountains"],
                f"{row['min_mountain_dist']:.1f}" if row["min_mountain_dist"] is not None else "n/a",
            )
        except Exception:
            # Logging must not affect core classification logic
            pass

    return rows


def _to_combined_water(water_reservoir, water_harbor, water_sea):
    """Union of the three water layers for blocking-distance check."""
    parts = []
    for gdf in (water_reservoir, water_harbor, water_sea):
        if len(gdf):
            parts.append(gdf.geometry)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=water_reservoir.crs if hasattr(water_reservoir, "crs") else "EPSG:3857")
    return gpd.GeoDataFrame(
        geometry=pd.concat(parts, ignore_index=True),
        crs=water_reservoir.crs,
    )


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

    # Basemap (same pattern as walking): extent set first so scope matches
    view_extent = 2 * MAP_RADIUS
    zoom_level = 17 if view_extent <= 500 else (16 if view_extent <= 900 else 15)
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.PositronNoLabels,
                   zoom=zoom_level, alpha=0.9)

    # ── background layers ────────────────────────────────────────────────────
    if len(parks):
        parks.plot(ax=ax, color="#b8c8a0", edgecolor="none", zorder=1)
    if len(water):
        water.plot(ax=ax, color="#6bb6d9", edgecolor="none", zorder=2)
    # buildings.plot(ax=ax, color="#e3e3e3", edgecolor="none", zorder=3)  # commented: basemap shows roads/buildings

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
    # for d, label in [(80, "80 m"), (160, "160 m"), (200, "200 m")]:
    #     gpd.GeoSeries([center.buffer(d)], crs=3857).boundary.plot(
    #         ax=ax, linestyle=(0, (4, 4)), linewidth=1.2,
    #         color="#555555", alpha=0.9, zorder=5,
    #     )
    #     ax.text(
    #         center.x + d + 3, center.y, label,
    #         fontsize=7, weight="bold", color="white",
    #         bbox=dict(facecolor="black", edgecolor="none", pad=2), zorder=6,
    #     )

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

            # Tangent rotation so text follows the ring; flip if upside down (keep readable)
            rotation = (mid_angle - 90) % 360
            if rotation > 180:
                rotation -= 360
            if rotation > 90 or rotation <= -90:
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
    # Top 12 tallest, from 30 m to 200 m from site centre
    label_candidates = nearby.copy()
    label_candidates["_dist"] = label_candidates.geometry.centroid.apply(
        lambda pt: center.distance(pt)
    )
    label_candidates = label_candidates[
        (label_candidates["_dist"] >= 30) &
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
        Patch(facecolor=COLOR_MAP["PARK"],     label="Park View"),
        Patch(facecolor=COLOR_MAP["GREEN"],    label="Green View"),
        Patch(facecolor=COLOR_MAP["MOUNTAIN"], label="Mountain View"),
        Patch(facecolor=COLOR_MAP["HARBOR"],   label="Harbor View"),
        Patch(facecolor=COLOR_MAP["RESERVOIR"],label="Reservoir View"),
        Patch(facecolor=COLOR_MAP["SEA"],      label="Sea View"),
        Patch(facecolor=COLOR_MAP["CITY"],     label="City View"),
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
    log.info("[view] site coordinate (lon, lat) = (%.6f, %.6f)", lon, lat)

    # ── 2. Site polygon ────────────────────────────────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_geom = lot_gdf.geometry.iloc[0]
        center    = site_geom.centroid
        # Buildings that intersect the lot polygon – used for site height (tallest on lot)
        buildings_on_lot = BUILDING_DATA[BUILDING_DATA.intersects(site_geom)].copy()
        if len(buildings_on_lot):
            max_h_lot = float(buildings_on_lot["HEIGHT_M"].max())
            log.info(
                "[view] buildings intersecting lot polygon: count=%d, HEIGHT_M max=%.1f",
                len(buildings_on_lot),
                max_h_lot,
            )
        else:
            log.info(
                "[view] buildings intersecting lot polygon: count=0, HEIGHT_M max=n/a"
            )
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
        buildings_on_lot = gpd.GeoDataFrame(geometry=[], crs=BUILDING_DATA.crs)

    analysis_circle = center.buffer(MAP_RADIUS)
    log.info("[view] analysis radius = %d m (MAP_RADIUS)", MAP_RADIUS)

    # ── 3. Context data ────────────────────────────────────────────────────────
    def fetch_layer(tags):
        """
        Fetch OSM features for the given tags around the site.
        IMPORTANT: If Overpass/osmnx fails or returns nothing, we log and
        return an empty GeoDataFrame so the rest of the view analysis
        can still proceed.
        """
        try:
            gdf = ox.features_from_point(
                (lat, lon), dist=FETCH_RADIUS, tags=tags
            ).to_crs(3857)
            if gdf is None or gdf.empty:
                log.info(
                    "[view] fetch_layer: no features for tags=%s at (lon,lat)=(%.6f, %.6f)",
                    tags, lon, lat,
                )
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
            return gdf[gdf.intersects(analysis_circle)]
        except Exception as e:
            log.warning(
                "[view] fetch_layer error for tags=%s at (lon,lat)=(%.6f, %.6f): %s",
                tags, lon, lat, e,
            )
            # Return an empty layer so other fetches and the classification can continue.
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    # buildings = fetch_layer({"building": True})  # commented: use basemap for roads/buildings
    buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    # Distinguish explicit parks, mountain features, and other green
    parks      = fetch_layer({"leisure": "park"})
    mountains  = fetch_layer({"natural": ["cliff", "peak"]})
    green      = fetch_layer({"landuse": "grass", "natural": ["wood", "grassland"], "boundary":"national_park"})

    # Water: one call for natural in [strait, bay, water, coastline]; split into reservoir/harbor/sea
    gdf_all = fetch_layer({"natural": ["strait", "bay", "water", "coastline"]})
    water_reservoir, water_harbor, water_sea, water_combined = _build_water_layers(
        gdf_all, crs="EPSG:3857"
    )

    nearby = BUILDING_DATA[BUILDING_DATA.intersects(analysis_circle)].copy()
    # Exclude on-site buildings so the site does not \"block itself\" in the
    # view analysis. Site buildings are still used for H_max / H_mid, but only
    # off-site buildings participate in CITY / blocking logic and labels.
    if len(buildings_on_lot):
        nearby = nearby[~nearby.index.isin(buildings_on_lot.index)].copy()
    n_nearby = len(nearby)
    if n_nearby > 0:
        heights = nearby["HEIGHT_M"].astype(float)
        log.info(
            "[view] nearby OFF-SITE buildings in %d m radius: count=%d, HEIGHT_M min=%.1f max=%.1f mean=%.1f",
            MAP_RADIUS, n_nearby, heights.min(), heights.max(), heights.mean(),
        )
    else:
        log.info("[view] nearby OFF-SITE buildings in %d m radius: count=0", MAP_RADIUS)

    # ── 4. Site building height ────────────────────────────────────────────────
    if len(buildings_on_lot):
        H_max = float(buildings_on_lot["HEIGHT_M"].max())
        log.info(
            "[view] site height from lot buildings: H_max (max HEIGHT_M on lot)=%.1f m",
            H_max,
        )
    else:
        # Fallback: use tallest building within a smaller radius around the site centre
        site_height_circle = center.buffer(SITE_HEIGHT_RADIUS_M)
        nearby_site = BUILDING_DATA[BUILDING_DATA.intersects(site_height_circle)].copy()
        n_site = len(nearby_site)
        if n_site > 0:
            H_max = float(nearby_site["HEIGHT_M"].max())
            log.info(
                "[view] site height from radius fallback: H_max (max HEIGHT_M in %d m)=%.1f m (no buildings_on_lot, %d buildings in fallback radius)",
                SITE_HEIGHT_RADIUS_M,
                H_max,
                n_site,
            )
        else:
            H_max = 10.0
            log.info(
                "[view] site height from radius fallback: no buildings within %d m → H_max=10.0 m (no buildings_on_lot)",
                SITE_HEIGHT_RADIUS_M,
            )
    H_mid = H_max / 2.0
    log.info("[view] heights used: H_max=%.1f m, H_mid=%.1f m", H_max, H_mid)

    # ── 5. City candidates (DISABLED – rely on blocking only) ─────────────────
    # We previously treated buildings within CITY_RADIUS of the site centre as
    # \"city candidates\" that could force CITY even when water/green existed
    # further out. That rule is now disabled; all CITY views come from the
    # blocking logic that compares building distance against the distance to
    # view content. We keep an empty GeoDataFrame here to preserve the call
    # signature without affecting behaviour.
    city_candidates = gpd.GeoDataFrame(geometry=[], crs=BUILDING_DATA.crs)
    log.info(
        "[view] city_candidates disabled; CITY_RADIUS=%d, count=%d",
        CITY_RADIUS,
        len(city_candidates),
    )

    # ── 6. Classify + merge sectors for both height levels ────────────────────
    raw_mid    = _classify_sectors(center, parks, mountains, green,
                                   water_reservoir, water_harbor, water_sea,
                                   city_candidates, H_mid, nearby, H_mid)
    raw_max    = _classify_sectors(center, parks, mountains, green,
                                   water_reservoir, water_harbor, water_sea,
                                   city_candidates, H_max, nearby, H_max)
    merged_mid = _merge_sectors(raw_mid)
    merged_max = _merge_sectors(raw_max)

    title_mid = (f"SITE ANALYSIS – View Analysis (MID HEIGHT)\n"
                 f"ref = H_mid = {H_mid:.1f} m  (site building = {H_max:.1f} m)")
    title_max = (f"SITE ANALYSIS – View Analysis (MAX HEIGHT)\n"
                 f"ref = H_max = {H_max:.1f} m  (site building = {H_max:.1f} m)")

    # ── 7. Render two panels side-by-side ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))

    fig.suptitle(
        f"SITE ANALYSIS – View Analysis ({data_type} {value})",
        fontsize=18, weight="bold", y=1.01,
    )

    _draw_panel(axes[0], center, site_geom, buildings, parks, water_combined,
                nearby, merged_mid, title_mid)

    _draw_panel(axes[1], center, site_geom, buildings, parks, water_combined,
                nearby, merged_max, title_max)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buffer
