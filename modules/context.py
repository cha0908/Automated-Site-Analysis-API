"""
modules/context.py  v5
──────────────────────────────────────────────────────────────────────────────
v5 fixes:
  ✓ Bus stops spread across 4 directional quadrants around site (not clustered)
  ✓ Basemap switched to CartoDB.PositronNoLabels (no street label clutter)
  ✓ Map extent centred correctly with equal padding on all 4 sides
  ✓ Bus stop static/bus.png icon used if available, else styled dot
  ✓ MTR logo from static/HK_MTR_logo.png
  ✓ No walk graph — crash-free
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

from shapely.geometry import Point, box
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 25

# ── Paths ─────────────────────────────────────────────────────────────────────
_STATIC        = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC, "HK_MTR_logo.png")
_BUS_LOGO_PATH = os.path.join(_STATIC, "bus.png")

# ── Constants ─────────────────────────────────────────────────────────────────
FETCH_RADIUS   = 1500    # landuse/building fetch (m)
MAP_HALF_SIZE  = 900     # half-extent in Web-Mercator metres — equal on all sides
MTR_COLOR      = "#ffd166"
BUS_COUNT      = 10      # total bus stops to show
BUS_FETCH_R    = 1200    # bus stop search radius (m)
MTR_RADIUS_M   = 2000
LABEL_RADIUS   = 800


# ── Logo loaders ──────────────────────────────────────────────────────────────

def _load_img(path, label):
    try:
        img = mpimg.imread(path)
        log.info("context: %s loaded", label)
        return img
    except Exception as e:
        log.warning("context: %s not found (%s)", label, e)
        return None

_MTR_IMG = _load_img(_MTR_LOGO_PATH, "MTR logo")
_BUS_IMG = _load_img(_BUS_LOGO_PATH, "bus icon")


def _place_logo(ax, x, y, img_arr, zoom=0.07, zorder=13):
    if img_arr is None:
        return
    icon = OffsetImage(img_arr, zoom=zoom)
    icon.image.axes = ax
    ab = AnnotationBbox(icon, (x, y), frameon=False,
                        zorder=zorder, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)


def _draw_roundel(ax, x, y, size=70, color="#ED1D24", zorder=11):
    ax.add_patch(plt.Circle((x, y), size, color=color, zorder=zorder))
    bw, bh = size*2.0, size*0.55
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
    """Always show schools, parks, neighbourhoods — contextually meaningful."""
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


def _spread_bus_stops(gdf, site_pt, n_total=10, min_dist_m=60):
    """
    Return up to n_total bus stops spread across all 4 quadrants around the
    site.  Each quadrant gets up to ceil(n_total/4) stops; within a quadrant
    stops must be at least min_dist_m apart.
    """
    if gdf.empty:
        return _empty_gdf()

    gdf = gdf.copy()
    gdf["_cx"] = gdf.geometry.apply(
        lambda g: g.x if g.geom_type == "Point" else g.centroid.x)
    gdf["_cy"] = gdf.geometry.apply(
        lambda g: g.y if g.geom_type == "Point" else g.centroid.y)
    gdf["_dx"] = gdf["_cx"] - site_pt.x
    gdf["_dy"] = gdf["_cy"] - site_pt.y
    gdf["_d"]  = np.hypot(gdf["_dx"], gdf["_dy"])
    gdf = gdf.sort_values("_d")

    quota = max(1, (n_total + 3) // 4)   # stops per quadrant

    def quadrant(row):
        return (int(row["_dx"] >= 0), int(row["_dy"] >= 0))

    selected   = []
    q_counts   = {}
    q_placed   = {}   # quadrant → list of placed Point coords

    for _, row in gdf.iterrows():
        q = quadrant(row)
        if q_counts.get(q, 0) >= quota:
            continue
        pt = Point(row["_cx"], row["_cy"])
        prev = q_placed.get(q, [])
        if any(pt.distance(p) < min_dist_m for p in prev):
            continue
        selected.append(row)
        q_counts[q]  = q_counts.get(q, 0) + 1
        q_placed.setdefault(q, []).append(pt)
        if len(selected) >= n_total:
            break

    if not selected:
        return _empty_gdf()
    return gpd.GeoDataFrame(selected, crs=gdf.crs).reset_index(drop=True)


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

    ozp_row = hits.iloc[0]
    zone    = ozp_row.get("ZONE_LABEL") or ozp_row.get("ZONE") or "N/A"
    plan_no = ozp_row.get("PLAN_NO")    or ozp_row.get("PLAN")  or "N/A"
    s_type  = _infer_site_type(zone)
    log.info("context: zone=%s type=%s", zone, s_type)

    # 3. OSM landuse / building polygons
    log.info("context: fetching landuse polygons")
    polys = _safe_osm(lat, lon, FETCH_RADIUS,
                      {"landuse": True, "leisure": True,
                       "amenity": True, "building": True}, "landuse")

    if not polys.empty:
        lu = polys.get("landuse",  gpd.GeoSeries(dtype=object))
        le = polys.get("leisure",  gpd.GeoSeries(dtype=object))
        am = polys.get("amenity",  gpd.GeoSeries(dtype=object))
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

    # 4. Site footprint
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
        else:
            site_geom = site_pt.buffer(80)
    else:
        site_geom = site_pt.buffer(80)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # 5. MTR stations
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
                             .sort_values("_d").head(4))

    # 6. Bus stops — spread across quadrants
    log.info("context: fetching bus stops")
    bus_raw = _safe_osm(lat, lon, BUS_FETCH_R,
                        {"highway": "bus_stop"}, "bus_stops")
    bus_stops = _spread_bus_stops(bus_raw, site_pt,
                                  n_total=BUS_COUNT, min_dist_m=60)
    log.info("context: %d bus stops selected", len(bus_stops))

    # 7. Place labels
    log.info("context: fetching place labels")
    lbl_tags = _label_rules(s_type)
    labels   = _safe_osm(lat, lon, LABEL_RADIUS, lbl_tags, "labels")
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
                        .drop_duplicates("label").head(24))

    # ── 8. PLOT ───────────────────────────────────────────────────────────────
    log.info("context: rendering")
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.subplots_adjust(bottom=0.14)

    # Map extent — equal padding all four sides, centred on site
    xmin = site_pt.x - MAP_HALF_SIZE
    xmax = site_pt.x + MAP_HALF_SIZE
    ymin = site_pt.y - MAP_HALF_SIZE
    ymax = site_pt.y + MAP_HALF_SIZE

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # Basemap — NO labels
    try:
        cx.add_basemap(ax, crs="EPSG:3857",
                       source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=16, alpha=0.95)
    except Exception as e:
        log.warning("context: basemap error: %s", e)

    # Landuse layers
    if not residential.empty: residential.plot(ax=ax, color="#f2c6a0", alpha=0.75, zorder=1)
    if not industrial.empty:  industrial.plot(ax=ax,  color="#b39ddb", alpha=0.75, zorder=1)
    if not parks.empty:       parks.plot(ax=ax,       color="#b7dfb9", alpha=0.90, zorder=2)
    if not schools.empty:     schools.plot(ax=ax,     color="#9ecae1", alpha=0.90, zorder=2)
    if not buildings.empty:   buildings.plot(ax=ax,   color="#d9d9d9", alpha=0.40, zorder=3)

    # MTR station polygons + logo
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

    # Bus stops
    if not bus_stops.empty:
        for _, bs in bus_stops.iterrows():
            bx = float(bs["_cx"])
            by = float(bs["_cy"])
            if _BUS_IMG is not None:
                _place_logo(ax, bx, by, _BUS_IMG, zoom=0.06, zorder=10)
            else:
                ax.plot(bx, by, "o",
                        color="#0d47a1", markersize=11,
                        markeredgecolor="white", markeredgewidth=1.5,
                        zorder=10)

    # Site polygon
    site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred",
                  linewidth=2.5, zorder=14)
    ax.text(site_geom.centroid.x, site_geom.centroid.y, "SITE",
            color="white", weight="bold", ha="center", va="center",
            fontsize=9, zorder=15)

    # Place labels
    offsets = [(0,40),(0,-40),(40,0),(-40,0),(30,30),(-30,30),
               (30,-30),(-30,-30),(0,55),(55,0),(-55,0),(0,-55)]
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

    # Info box (top-left, inside map)
    ax.text(0.015, 0.985,
            f"Lot: {value}\nOZP Plan: {plan_no}\nZoning: {zone}\nSite Type: {s_type}\n",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6))

    # ── Legend below map ──────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#f2c6a0", label="Residential"),
        mpatches.Patch(color="#b39ddb", label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9", label="Public Park"),
        mpatches.Patch(color="#9ecae1", label="School / Institution"),
        mpatches.Patch(color=MTR_COLOR,  label="MTR Station"),
        mpatches.Patch(color="#e53935",  label="Site"),
    ]
    handler_map = {}

    # MTR logo legend entry
    if _MTR_IMG is not None:
        try:
            raw   = (_MTR_IMG*255).astype(np.uint8) if _MTR_IMG.dtype != np.uint8 else _MTR_IMG
            thumb = np.array(Image.fromarray(raw).resize((20, 20), Image.LANCZOS))
            class _MLH(mlines.Line2D): pass
            class _MLHH(lh.HandlerBase):
                def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                    ib = OffsetImage(thumb, zoom=1.0)
                    return [AnnotationBbox(ib,(xd+w/2,yd+h/2),
                                          xycoords=trans,frameon=False)]
            legend_handles.append(_MLH([], [], label="MTR Station"))
            handler_map[_MLH] = _MLHH()
        except Exception:
            legend_handles.append(
                mlines.Line2D([], [], marker="*", linestyle="None",
                              markerfacecolor="#ED1D24", markersize=10,
                              label="MTR Station"))
    else:
        legend_handles.append(
            mlines.Line2D([], [], marker="*", linestyle="None",
                          markerfacecolor="#ED1D24", markersize=10,
                          label="MTR Station"))

    # Bus icon legend entry
    if _BUS_IMG is not None:
        try:
            bus_arr   = (_BUS_IMG*255).astype(np.uint8) if _BUS_IMG.dtype != np.uint8 else _BUS_IMG
            bus_thumb = np.array(Image.fromarray(bus_arr).resize((20, 20), Image.LANCZOS))
            class _BLH(mlines.Line2D): pass
            class _BLHH(lh.HandlerBase):
                def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                    ib = OffsetImage(bus_thumb, zoom=1.0)
                    return [AnnotationBbox(ib,(xd+w/2,yd+h/2),
                                          xycoords=trans,frameon=False)]
            legend_handles.append(_BLH([], [], label="Bus Stop"))
            handler_map[_BLH] = _BLHH()
        except Exception:
            legend_handles.append(
                mlines.Line2D([], [], marker="o", linestyle="None",
                              markerfacecolor="#0d47a1", markeredgecolor="white",
                              markeredgewidth=1.5, markersize=9, label="Bus Stop"))
    else:
        legend_handles.append(
            mlines.Line2D([], [], marker="o", linestyle="None",
                          markerfacecolor="#0d47a1", markeredgecolor="white",
                          markeredgewidth=1.5, markersize=9, label="Bus Stop"))

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
