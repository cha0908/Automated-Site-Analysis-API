"""
modules/context.py  v4
──────────────────────────────────────────────────────────────────────────────
Site Context Analysis — matches the Colab reference output.

v4 fixes:
  ✓ Bus stops spaced out — min 80 m apart, shown as solid navy dots (no icon clutter)
  ✓ MTR logo larger and clearly visible
  ✓ Site polygon uses lot boundary properly; fallback buffer is bigger (80 m)
  ✓ Place labels filtered to schools/parks/neighbourhoods only (no restaurants)
  ✓ Legend moved below map (no overlap)
  ✓ Buildings layer rendered
  ✓ All OSM fetches isolated with timeout + gc
"""

import io
import gc
import os
import textwrap
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.legend_handler as lh

import numpy as np
import geopandas as gpd
import osmnx as ox
import contextily as cx

from shapely.geometry import Point
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 25

# ── Paths ─────────────────────────────────────────────────────────────────────
_STATIC       = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC, "HK_MTR_logo.png")

# ── Constants ─────────────────────────────────────────────────────────────────
FETCH_RADIUS    = 1500
MAP_HALF_SIZE   = 900
MTR_COLOR       = "#ffd166"
BUS_COUNT       = 10
BUS_MIN_DIST_M  = 80     # minimum spacing between displayed bus stops
MTR_RADIUS_M    = 2000
LABEL_DIST_M    = 800    # radius for place labels

# ── Logo loader ───────────────────────────────────────────────────────────────

def _load_mtr_logo():
    try:
        img = mpimg.imread(_MTR_LOGO_PATH)
        log.info("context: MTR logo loaded")
        return img
    except Exception as e:
        log.warning("context: MTR logo missing (%s)", e)
        return None

_MTR_IMG = _load_mtr_logo()


def _place_logo(ax, x, y, img_arr, zoom=0.07, zorder=13):
    if img_arr is None:
        return
    icon = OffsetImage(img_arr, zoom=zoom)
    icon.image.axes = ax
    ab = AnnotationBbox(icon, (x, y), frameon=False,
                        zorder=zorder, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)


def _draw_roundel(ax, x, y, size=70, color="#ED1D24", zorder=11):
    """Fallback MTR roundel if logo PNG missing."""
    ax.add_patch(plt.Circle((x, y), size, color=color, zorder=zorder))
    bw, bh = size * 2.0, size * 0.55
    ax.add_patch(plt.Rectangle((x-bw/2, y-bh/2), bw, bh,
                                color="white", zorder=zorder+1))
    ax.add_patch(plt.Circle((x, y), size*0.55, color="white", zorder=zorder+2))
    ax.add_patch(plt.Rectangle((x-bw/2, y-bh/2), bw, bh,
                                color=color, zorder=zorder+3))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                  return "RESIDENTIAL"
    if z.startswith("C"):                  return "COMMERCIAL"
    if z.startswith("G"):                  return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU"): return "HOTEL"
    return "MIXED"


def _label_rules(site_type: str) -> dict:
    """
    Always fetch schools, parks and neighbourhoods regardless of site type —
    these are the contextually meaningful labels (same as Colab reference).
    For commercial sites we additionally include railway stations.
    """
    base = {
        "amenity": ["school", "college", "university", "hospital"],
        "leisure": ["park"],
        "place":   ["neighbourhood", "suburb"],
    }
    if site_type == "COMMERCIAL":
        base["railway"] = ["station"]
    return base


def _empty_gdf():
    return gpd.GeoDataFrame(geometry=[], crs=3857)


def _safe_osm(lat, lon, dist, tags, label=""):
    try:
        gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags).to_crs(3857)
        gc.collect()
        return gdf
    except Exception as e:
        log.warning("context: OSM fetch failed [%s]: %s", label, e)
        gc.collect()
        return _empty_gdf()


def _space_points(gdf, min_dist_m):
    """Return subset of point GeoDataFrame with minimum spacing between points."""
    if gdf.empty:
        return gdf
    kept = []
    kept_pts = []
    for _, row in gdf.iterrows():
        pt = row.geometry if row.geometry.geom_type == "Point" else row.geometry.centroid
        if all(pt.distance(p) >= min_dist_m for p in kept_pts):
            kept.append(row)
            kept_pts.append(pt)
    if not kept:
        return _empty_gdf()
    return gpd.GeoDataFrame(kept, crs=gdf.crs).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_context(
    data_type: str,
    value:     str,
    zone_data: gpd.GeoDataFrame,
    radius_m:  int   = None,
    lon:       float = None,
    lat:       float = None,
    lot_ids:   list  = None,
    extents:   list  = None,
) -> io.BytesIO:

    lot_ids = lot_ids or []
    extents = extents or []

    # 1. Resolve coordinates
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    log.info("context: resolved %.6f, %.6f", lon, lat)

    site_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    # 2. OZP zoning from dataset
    hits = zone_data[zone_data.contains(site_pt)]
    if hits.empty:
        zd2 = zone_data.copy()
        zd2["_d"] = zd2.geometry.distance(site_pt)
        hits = zd2[zd2["_d"] < 200].sort_values("_d")
    if hits.empty:
        raise ValueError("No OZP zone found near site.")

    ozp_row  = hits.iloc[0]
    zone     = ozp_row.get("ZONE_LABEL") or ozp_row.get("ZONE") or "N/A"
    plan_no  = ozp_row.get("PLAN_NO")    or ozp_row.get("PLAN")  or "N/A"
    s_type   = _infer_site_type(zone)
    lbl_tags = _label_rules(s_type)
    log.info("context: zone=%s type=%s", zone, s_type)

    # 3. OSM landuse / building polygons
    log.info("context: fetching landuse polygons")
    polys = _safe_osm(lat, lon, FETCH_RADIUS,
                      {"landuse": True, "leisure": True,
                       "amenity": True, "building": True}, "landuse")

    if not polys.empty:
        lu = polys.get("landuse", gpd.GeoSeries(dtype=object))
        le = polys.get("leisure", gpd.GeoSeries(dtype=object))
        am = polys.get("amenity", gpd.GeoSeries(dtype=object))
        bu = polys.get("building", gpd.GeoSeries(dtype=object))

        residential = polys[lu == "residential"]
        industrial  = polys[lu.isin(["industrial", "commercial"])]
        parks       = polys[le == "park"]
        schools     = polys[am.isin(["school", "college", "university"])]
        buildings   = polys[
            bu.notnull() &
            polys.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
    else:
        residential = industrial = parks = schools = buildings = _empty_gdf()

    # 4. Site footprint — lot boundary first, then OSM, then buffer
    lot_gdf = get_lot_boundary(lon, lat, data_type,
                               extents if len(extents) > 1 else None)
    if lot_gdf is not None and not lot_gdf.empty:
        site_geom = lot_gdf.geometry.iloc[0]
        log.info("context: site from lot boundary")
    elif not polys.empty:
        cands = polys[
            polys.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) &
            (polys.geometry.distance(site_pt) < 60)
        ]
        if len(cands):
            site_geom = (cands.assign(a=cands.area)
                              .sort_values("a", ascending=False)
                              .geometry.iloc[0])
            log.info("context: site from OSM polygon")
        else:
            site_geom = site_pt.buffer(80)
            log.info("context: site from point buffer (80 m)")
    else:
        site_geom = site_pt.buffer(80)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # 5. MTR stations — polygon footprints
    log.info("context: fetching MTR stations")
    stations = _safe_osm(lat, lon, MTR_RADIUS_M,
                         {"railway": "station"}, "stations")

    if not stations.empty:
        ne = stations.get("name:en")
        nz = stations.get("name")
        if ne is not None and nz is not None:
            stations["_name"] = ne.fillna(nz)
        elif nz is not None:
            stations["_name"] = nz
        else:
            stations["_name"] = "MTR"
        stations["_cx"] = stations.geometry.centroid.x
        stations["_cy"] = stations.geometry.centroid.y
        stations["_d"]  = stations.geometry.centroid.distance(site_pt)
        stations = (stations.dropna(subset=["_name"])
                             .sort_values("_d")
                             .head(4))

    # 6. Bus stops — 10 nearest, spaced ≥ 80 m apart
    log.info("context: fetching bus stops")
    bus_raw = _safe_osm(lat, lon, 1200,
                        {"highway": "bus_stop"}, "bus_stops")

    if not bus_raw.empty:
        bus_raw["_d"] = bus_raw.geometry.distance(site_pt)
        bus_sorted    = bus_raw.sort_values("_d").head(BUS_COUNT * 3)
        bus_stops     = _space_points(bus_sorted, BUS_MIN_DIST_M).head(BUS_COUNT)
    else:
        bus_stops = _empty_gdf()
    log.info("context: %d bus stops after spacing", len(bus_stops))

    # 7. Place labels — schools, parks, neighbourhoods
    log.info("context: fetching place labels")
    labels = _safe_osm(lat, lon, LABEL_DIST_M, lbl_tags, "labels")

    if not labels.empty:
        ne = labels.get("name:en")
        nz = labels.get("name")
        if ne is not None and nz is not None:
            labels["label"] = ne.fillna(nz)
        elif nz is not None:
            labels["label"] = nz
        else:
            labels["label"] = None
        labels = (labels.dropna(subset=["label"])
                        .drop_duplicates("label")
                        .head(24))

    # ── 8. PLOT ───────────────────────────────────────────────────────────────
    log.info("context: rendering")
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.subplots_adjust(bottom=0.16)   # space for legend below map

    xmin = site_pt.x - MAP_HALF_SIZE
    xmax = site_pt.x + MAP_HALF_SIZE
    ymin = site_pt.y - MAP_HALF_SIZE
    ymax = site_pt.y + MAP_HALF_SIZE

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.autoscale(False)

    try:
        cx.add_basemap(ax, crs="EPSG:3857",
                       source=cx.providers.CartoDB.Positron,
                       zoom=16, alpha=0.95)
    except Exception as e:
        log.warning("context: basemap error: %s", e)

    # Landuse layers (zorder 1-5)
    if not residential.empty: residential.plot(ax=ax, color="#f2c6a0", alpha=0.75, zorder=1)
    if not industrial.empty:  industrial.plot(ax=ax,  color="#b39ddb", alpha=0.75, zorder=1)
    if not parks.empty:       parks.plot(ax=ax,       color="#b7dfb9", alpha=0.90, zorder=2)
    if not schools.empty:     schools.plot(ax=ax,     color="#9ecae1", alpha=0.90, zorder=2)
    if not buildings.empty:   buildings.plot(ax=ax,   color="#d9d9d9", alpha=0.40, zorder=3)

    # MTR station polygons (zorder 8) + logo (zorder 13)
    if not stations.empty:
        stations.plot(ax=ax, facecolor=MTR_COLOR, edgecolor="#b8860b",
                      linewidth=1.5, alpha=0.95, zorder=8)
        for _, st in stations.iterrows():
            cx_ = float(st["_cx"])
            cy_ = float(st["_cy"])
            if _MTR_IMG is not None:
                _place_logo(ax, cx_, cy_, _MTR_IMG, zoom=0.07, zorder=13)
            else:
                _draw_roundel(ax, cx_, cy_, size=70, zorder=11)
            nm = st.get("_name", "")
            if nm and isinstance(nm, str):
                ax.text(cx_, cy_ + 130, _wrap(nm, 18),
                        fontsize=8.5, ha="center", va="bottom", weight="bold",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=2),
                        zorder=14, clip_on=True)

    # Bus stops — solid navy dots (zorder 10), clearly spaced
    if not bus_stops.empty:
        for _, bs in bus_stops.iterrows():
            pt = bs.geometry if bs.geometry.geom_type == "Point" else bs.geometry.centroid
            # Draw as a larger solid dot
            ax.plot(pt.x, pt.y, "o",
                    color="#0d47a1", markersize=10,
                    markeredgecolor="white", markeredgewidth=1.5,
                    zorder=10)

    # Site polygon (zorder 14 — always on top)
    site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred",
                  linewidth=2.5, zorder=14)
    cx_site = site_geom.centroid.x
    cy_site = site_geom.centroid.y
    ax.text(cx_site, cy_site, "SITE",
            color="white", weight="bold", ha="center", va="center",
            fontsize=9, zorder=15)

    # Place labels
    offsets = [(0,40),(0,-40),(40,0),(-40,0),(30,30),(-30,30),
               (30,-30),(-30,-30),(0,55),(55,0)]
    placed = []
    if not labels.empty and "label" in labels.columns:
        for i, (_, lrow) in enumerate(labels.iterrows()):
            try:
                p = lrow.geometry.representative_point()
            except Exception:
                continue
            if p.distance(site_pt) < 120:
                continue
            if any(p.distance(pp) < 100 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x+dx, p.y+dy, _wrap(lrow["label"], 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.3"),
                    zorder=12, clip_on=True)
            placed.append(p)

    # Info box (top-left)
    ax.text(0.015, 0.985,
            f"Lot: {value}\nOZP Plan: {plan_no}\nZoning: {zone}\nSite Type: {s_type}\n",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#f2c6a0", label="Residential"),
        mpatches.Patch(color="#b39ddb", label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9", label="Public Park"),
        mpatches.Patch(color="#9ecae1", label="School / Institution"),
        mpatches.Patch(color=MTR_COLOR,  label="MTR Station"),
        mpatches.Patch(color="#e53935",  label="Site"),
        mlines.Line2D([], [], marker="o", linestyle="None",
                      markerfacecolor="#0d47a1", markeredgecolor="white",
                      markeredgewidth=1.5, markersize=9, label="Bus Stop"),
    ]

    handler_map = {}

    # MTR logo in legend (if available)
    if _MTR_IMG is not None:
        try:
            raw   = (_MTR_IMG*255).astype(np.uint8) if _MTR_IMG.dtype != np.uint8 else _MTR_IMG
            thumb = np.array(Image.fromarray(raw).resize((20, 20), Image.LANCZOS))

            class _MLH(mlines.Line2D): pass
            class _MLHH(lh.HandlerBase):
                def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                    ib = OffsetImage(thumb, zoom=1.0)
                    return [AnnotationBbox(ib, (xd+w/2, yd+h/2),
                                          xycoords=trans, frameon=False)]
            legend_handles.append(_MLH([], [], label="MTR Logo"))
            handler_map[_MLH] = _MLHH()
        except Exception:
            pass

    ax.legend(
        handles=legend_handles,
        handler_map=handler_map or None,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.02),
        bbox_transform=ax.transAxes,
        ncol=4,
        fontsize=8.5,
        framealpha=0.95,
        edgecolor="#333",
        title="Legend",
        title_fontsize=9,
        labelspacing=0.3,
        handlelength=1.4,
        borderpad=0.5,
        columnspacing=1.0,
    )

    ax.set_title("Automated Site Context Analysis (Building-Type Driven)",
                 fontsize=15, weight="bold")
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("context: done")
    return buf
