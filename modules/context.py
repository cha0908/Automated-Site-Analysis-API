"""
modules/context.py  v6
──────────────────────────────────────────────────────────────────────────────
v6 crash fix:
  - zoom=16 on a 12x12 fig downloads ~100 tiles → OOM kill on Render free tier
  - Fix: drop to zoom=15, reduce fig to 10x10, lower dpi to 120
  - Basemap fetch wrapped in try/except with graceful fallback (plain bg)
  - All heavy numpy/gdf objects explicitly deleted + gc after each step
  - Rendering split into logical sections so partial failure is visible in logs
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
_STATIC        = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC, "HK_MTR_logo.png")
_BUS_LOGO_PATH = os.path.join(_STATIC, "bus.png")

# ── Constants ─────────────────────────────────────────────────────────────────
FETCH_RADIUS   = 1500
MAP_HALF_SIZE  = 900
MTR_COLOR      = "#ffd166"
BUS_COUNT      = 10
BUS_FETCH_R    = 1200
MTR_RADIUS_M   = 2000
LABEL_RADIUS   = 800

FIG_SIZE   = 10     # inches — reduced from 12 to cut tile count
FIG_DPI    = 120    # reduced from 150
BASEMAP_ZOOM = 15   # reduced from 16 — 4× fewer tiles, massive memory saving


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
    try:
        icon = OffsetImage(img_arr, zoom=zoom)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False,
                            zorder=zorder, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    except Exception as e:
        log.debug("context: logo place failed: %s", e)


def _draw_roundel(ax, x, y, size=70, color="#ED1D24", zorder=11):
    try:
        ax.add_patch(plt.Circle((x, y), size, color=color, zorder=zorder))
        bw, bh = size*2.0, size*0.55
        ax.add_patch(plt.Rectangle((x-bw/2, y-bh/2), bw, bh,
                                    color="white", zorder=zorder+1))
        ax.add_patch(plt.Circle((x, y), size*0.55, color="white", zorder=zorder+2))
        ax.add_patch(plt.Rectangle((x-bw/2, y-bh/2), bw, bh,
                                    color=color, zorder=zorder+3))
    except Exception as e:
        log.debug("context: roundel failed: %s", e)


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
        log.info("context: OSM [%s] → %d rows", label, len(gdf))
        return gdf
    except Exception as e:
        log.warning("context: OSM fetch failed [%s]: %s", label, e)
        gc.collect()
        return _empty_gdf()


def _spread_bus_stops(gdf, site_pt, n_total=10, min_dist_m=60):
    """Distribute up to n_total stops across 4 quadrants, min spacing enforced."""
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

    quota    = max(1, (n_total + 3) // 4)
    selected = []
    q_counts = {}
    q_placed = {}

    for _, row in gdf.iterrows():
        q = (int(row["_dx"] >= 0), int(row["_dy"] >= 0))
        if q_counts.get(q, 0) >= quota:
            continue
        pt   = Point(row["_cx"], row["_cy"])
        prev = q_placed.get(q, [])
        if any(pt.distance(p) < min_dist_m for p in prev):
            continue
        selected.append(row)
        q_counts[q] = q_counts.get(q, 0) + 1
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

    # 1. Resolve
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    log.info("context: resolved %.6f, %.6f", lon, lat)
    site_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    # 2. OZP zoning
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

    # 3. Landuse polygons
    log.info("context: fetching landuse")
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
        del polys; gc.collect()
    else:
        residential = industrial = parks = schools = buildings = _empty_gdf()

    # 4. Site footprint
    lot_gdf = get_lot_boundary(lon, lat, data_type,
                               extents if len(extents) > 1 else None)
    if lot_gdf is not None and not lot_gdf.empty:
        site_geom = lot_gdf.geometry.iloc[0]
        log.info("context: site from lot boundary")
    else:
        site_geom = site_pt.buffer(80)
        log.info("context: site from buffer fallback")
    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # 5. MTR stations
    log.info("context: fetching stations")
    stations = _safe_osm(lat, lon, MTR_RADIUS_M, {"railway": "station"}, "stations")
    if not stations.empty:
        ne = stations.get("name:en")
        nz = stations.get("name")
        stations["_name"] = ne.fillna(nz) if (ne is not None and nz is not None) \
                            else (nz if nz is not None else "MTR")
        stations["_cx"] = stations.geometry.centroid.x
        stations["_cy"] = stations.geometry.centroid.y
        stations["_d"]  = stations.geometry.centroid.distance(site_pt)
        stations = stations.dropna(subset=["_name"]).sort_values("_d").head(4)

    # 6. Bus stops
    log.info("context: fetching bus stops")
    bus_raw   = _safe_osm(lat, lon, BUS_FETCH_R, {"highway": "bus_stop"}, "bus_stops")
    bus_stops = _spread_bus_stops(bus_raw, site_pt, BUS_COUNT, min_dist_m=60)
    del bus_raw; gc.collect()
    log.info("context: %d bus stops selected", len(bus_stops))

    # 7. Labels
    log.info("context: fetching labels")
    labels = _safe_osm(lat, lon, LABEL_RADIUS, _label_rules(s_type), "labels")
    if not labels.empty:
        ne = labels.get("name:en")
        nz = labels.get("name")
        labels["label"] = ne.fillna(nz) if (ne is not None and nz is not None) \
                          else (nz if nz is not None else None)
        labels = labels.dropna(subset=["label"]).drop_duplicates("label").head(24)

    # ── 8. FIGURE ─────────────────────────────────────────────────────────────
    log.info("context: building figure (zoom=%d, figsize=%d)", BASEMAP_ZOOM, FIG_SIZE)

    xmin = site_pt.x - MAP_HALF_SIZE
    xmax = site_pt.x + MAP_HALF_SIZE
    ymin = site_pt.y - MAP_HALF_SIZE
    ymax = site_pt.y + MAP_HALF_SIZE

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    fig.subplots_adjust(bottom=0.14)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # Basemap — no labels, lower zoom = fewer tiles = less memory
    log.info("context: adding basemap")
    try:
        cx.add_basemap(ax, crs="EPSG:3857",
                       source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=BASEMAP_ZOOM, alpha=0.95)
    except Exception as e:
        log.warning("context: basemap failed (%s) — plain background", e)
        ax.set_facecolor("#f0efe9")
    gc.collect()
    log.info("context: basemap done")

    # Landuse
    log.info("context: plotting landuse layers")
    if not residential.empty: residential.plot(ax=ax, color="#f2c6a0", alpha=0.75, zorder=1)
    if not industrial.empty:  industrial.plot(ax=ax,  color="#b39ddb", alpha=0.75, zorder=1)
    if not parks.empty:       parks.plot(ax=ax,       color="#b7dfb9", alpha=0.90, zorder=2)
    if not schools.empty:     schools.plot(ax=ax,     color="#9ecae1", alpha=0.90, zorder=2)
    if not buildings.empty:   buildings.plot(ax=ax,   color="#d9d9d9", alpha=0.40, zorder=3)
    del residential, industrial, parks, schools, buildings; gc.collect()

    # MTR stations
    log.info("context: plotting MTR stations")
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
    log.info("context: plotting bus stops")
    if not bus_stops.empty:
        for _, bs in bus_stops.iterrows():
            bx = float(bs["_cx"])
            by = float(bs["_cy"])
            if _BUS_IMG is not None:
                _place_logo(ax, bx, by, _BUS_IMG, zoom=0.06, zorder=10)
            else:
                ax.plot(bx, by, "o", color="#0d47a1", markersize=11,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=10)

    # Site
    log.info("context: plotting site")
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

    # Info box
    ax.text(0.015, 0.985,
            f"Lot: {value}\nOZP Plan: {plan_no}\nZoning: {zone}\nSite Type: {s_type}\n",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6))

    # Legend
    legend_handles = [
        mpatches.Patch(color="#f2c6a0", label="Residential"),
        mpatches.Patch(color="#b39ddb", label="Industrial / Commercial"),
        mpatches.Patch(color="#b7dfb9", label="Public Park"),
        mpatches.Patch(color="#9ecae1", label="School / Institution"),
        mpatches.Patch(color=MTR_COLOR,  label="MTR Station"),
        mpatches.Patch(color="#e53935",  label="Site"),
    ]
    handler_map = {}

    # MTR legend icon
    if _MTR_IMG is not None:
        try:
            raw   = (_MTR_IMG*255).astype(np.uint8) if _MTR_IMG.dtype != np.uint8 else _MTR_IMG
            thumb = np.array(Image.fromarray(raw).resize((20,20), Image.LANCZOS))
            class _MLH(mlines.Line2D): pass
            class _MLHH(lh.HandlerBase):
                def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                    return [AnnotationBbox(OffsetImage(thumb, zoom=1.0),
                                          (xd+w/2,yd+h/2), xycoords=trans, frameon=False)]
            legend_handles.append(_MLH([], [], label="MTR Station Icon"))
            handler_map[_MLH] = _MLHH()
        except Exception:
            pass

    # Bus legend icon
    if _BUS_IMG is not None:
        try:
            ba    = (_BUS_IMG*255).astype(np.uint8) if _BUS_IMG.dtype != np.uint8 else _BUS_IMG
            bthumb = np.array(Image.fromarray(ba).resize((20,20), Image.LANCZOS))
            class _BLH(mlines.Line2D): pass
            class _BLHH(lh.HandlerBase):
                def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                    return [AnnotationBbox(OffsetImage(bthumb, zoom=1.0),
                                          (xd+w/2,yd+h/2), xycoords=trans, frameon=False)]
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
        fontsize=8.5, framealpha=0.95, edgecolor="#333",
        title="Legend", title_fontsize=9,
        labelspacing=0.3, handlelength=1.4,
        borderpad=0.5, columnspacing=1.0,
    )

    ax.set_title("Automated Site Context Analysis (Building-Type Driven)",
                 fontsize=15, weight="bold")
    ax.set_axis_off()

    log.info("context: saving PNG")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("context: done")
    return buf
