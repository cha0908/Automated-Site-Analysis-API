"""
modules/context.py  v9
- All 4 OSM fetches run IN PARALLEL (ThreadPoolExecutor, shutdown(wait=False))
  cutting total fetch time from ~55s sequential to ~18s (slowest single fetch).
  This prevents Render health-check kills during the fetch phase.
- Bus stops: bus.png icon (OffsetImage zoom=0.028), fallback to navy dot
- MTR: yellow polygon footprint + small MTR logo on centroid
- Legend: single-column vertical list, bottom-left inside map
- Basemap: PositronNoLabels, zoom=16
"""

import io
import gc
import os
import textwrap
import logging
import concurrent.futures as _cf
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg

import numpy as np
import geopandas as gpd
import osmnx as ox
import contextily as cx

from shapely.geometry import Point, box as sbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 25

_STATIC        = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC, "HK_MTR_logo.png")
_BUS_ICON_PATH = os.path.join(_STATIC, "bus.png")

FETCH_RADIUS  = 1500
MAP_HALF_SIZE = 900
MTR_COLOR     = "#ffd166"
BUS_COUNT     = 10
BUS_FETCH_R   = 1200
MTR_RADIUS_M  = 2000
LABEL_RADIUS  = 800
FIG_SIZE      = 12
FIG_DPI       = 150
BASEMAP_ZOOM  = 16

# ── Static assets — loaded once at module init ────────────────────────────────

def _load_mtr():
    try:
        img = mpimg.imread(_MTR_LOGO_PATH)
        log.info("context: MTR logo loaded")
        return img
    except Exception as e:
        log.warning("context: MTR logo missing (%s)", e)
        return None

_MTR_IMG = _load_mtr()

_MTR_THUMB = None
if _MTR_IMG is not None:
    try:
        raw        = (_MTR_IMG * 255).astype(np.uint8) if _MTR_IMG.dtype != np.uint8 else _MTR_IMG
        _MTR_THUMB = np.array(Image.fromarray(raw).resize((22, 22), Image.LANCZOS))
        log.info("context: MTR thumb pre-built")
    except Exception as e:
        log.warning("context: MTR thumb failed (%s)", e)

_BUS_IMG = None
try:
    _BUS_IMG = mpimg.imread(_BUS_ICON_PATH)
    log.info("context: bus icon loaded")
except Exception as e:
    log.warning("context: bus icon missing (%s)", e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _infer_site_type(zone):
    z = zone.upper()
    if z.startswith("R"):                  return "RESIDENTIAL"
    if z.startswith("C"):                  return "COMMERCIAL"
    if z.startswith("G"):                  return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU"): return "HOTEL"
    return "MIXED"


def _label_rules(site_type):
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
        log.warning("context: OSM [%s] failed: %s", label, e)
        gc.collect()
        return _empty_gdf()


def _clip_to_map(gdf, xmin, ymin, xmax, ymax):
    if gdf.empty:
        return gdf
    try:
        cb      = gpd.GeoDataFrame(geometry=[sbox(xmin, ymin, xmax, ymax)], crs=3857)
        clipped = gpd.clip(gdf, cb)
        log.info("context: clipped %d → %d rows", len(gdf), len(clipped))
        return clipped
    except Exception as e:
        log.warning("context: clip failed (%s)", e)
        return gdf


def _spread_bus_stops(gdf, site_pt, n_total=10, min_dist_m=60):
    if gdf.empty:
        return _empty_gdf()
    gdf = gdf.copy()
    gdf["_cx"] = gdf.geometry.apply(lambda g: g.x if g.geom_type == "Point" else g.centroid.x)
    gdf["_cy"] = gdf.geometry.apply(lambda g: g.y if g.geom_type == "Point" else g.centroid.y)
    gdf["_dx"] = gdf["_cx"] - site_pt.x
    gdf["_dy"] = gdf["_cy"] - site_pt.y
    gdf["_d"]  = np.hypot(gdf["_dx"], gdf["_dy"])
    gdf = gdf.sort_values("_d")

    quota    = max(1, (n_total + 3) // 4)
    selected, q_counts, q_placed = [], {}, {}

    for _, row in gdf.iterrows():
        q    = (int(row["_dx"] >= 0), int(row["_dy"] >= 0))
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
    data_type, value, zone_data,
    radius_m=None, lon=None, lat=None,
    lot_ids=None, extents=None,
):
    lot_ids = lot_ids or []
    extents = extents or []

    # 1. Resolve
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    log.info("context: resolved %.6f, %.6f", lon, lat)
    site_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    xmin = site_pt.x - MAP_HALF_SIZE
    xmax = site_pt.x + MAP_HALF_SIZE
    ymin = site_pt.y - MAP_HALF_SIZE
    ymax = site_pt.y + MAP_HALF_SIZE

    # 2. OZP zoning (local — instant)
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

    # 3. Site footprint (local — instant)
    lot_gdf   = get_lot_boundary(lon, lat, data_type,
                                 extents if len(extents) > 1 else None)
    site_geom = (lot_gdf.geometry.iloc[0]
                 if lot_gdf is not None and not lot_gdf.empty
                 else site_pt.buffer(80))
    site_gdf  = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # 4. ALL OSM fetches in parallel — total time = slowest single fetch
    #    (~18s) instead of sum of all fetches (~55s sequential).
    #
    #    CRITICAL: shutdown(wait=False) so zombie Overpass threads never
    #    block the render phase. Timed-out fetches fall back to empty GDF.
    log.info("context: launching 4 OSM fetches in parallel")

    _results = {
        "landuse":  _empty_gdf(),
        "stations": _empty_gdf(),
        "bus":      _empty_gdf(),
        "labels":   _empty_gdf(),
    }

    def _fetch_landuse():
        _results["landuse"] = _safe_osm(
            lat, lon, FETCH_RADIUS,
            {"landuse": True, "leisure": True, "amenity": True},
            "landuse")

    def _fetch_stations():
        _results["stations"] = _safe_osm(
            lat, lon, MTR_RADIUS_M,
            {"railway": "station"},
            "stations")

    def _fetch_bus():
        _results["bus"] = _safe_osm(
            lat, lon, BUS_FETCH_R,
            {"highway": "bus_stop"},
            "bus_stops")

    def _fetch_labels():
        _results["labels"] = _safe_osm(
            lat, lon, LABEL_RADIUS,
            _label_rules(s_type),
            "labels")

    _pool = ThreadPoolExecutor(max_workers=4)
    _futs = {
        _pool.submit(_fetch_landuse):  "landuse",
        _pool.submit(_fetch_stations): "stations",
        _pool.submit(_fetch_bus):      "bus",
        _pool.submit(_fetch_labels):   "labels",
    }
    done, pending = _cf.wait(_futs, timeout=35, return_when=_cf.ALL_COMPLETED)
    for f in pending:
        log.warning("context: OSM [%s] timed out — using empty", _futs[f])
    for f in done:
        try:
            f.result()   # re-raise any exception from the thread
        except Exception as e:
            log.warning("context: OSM fetch error (%s): %s", _futs[f], e)
    _pool.shutdown(wait=False)  # do NOT block on zombie threads
    log.info("context: all OSM fetches done (%d/%d completed)", len(done), len(_futs))

    # 5. Post-process landuse
    polys_raw = _results["landuse"]
    polys     = _clip_to_map(polys_raw, xmin, ymin, xmax, ymax)
    del polys_raw; gc.collect()

    if not polys.empty:
        lu = polys.get("landuse", gpd.GeoSeries(dtype=object))
        le = polys.get("leisure", gpd.GeoSeries(dtype=object))
        am = polys.get("amenity", gpd.GeoSeries(dtype=object))
        residential = polys[lu == "residential"]
        industrial  = polys[lu.isin(["industrial", "commercial"])]
        parks       = polys[le == "park"]
        schools     = polys[am.isin(["school", "college", "university"])]
        del polys; gc.collect()
    else:
        residential = industrial = parks = schools = _empty_gdf()

    # 6. Post-process stations
    stations = _results["stations"]
    if not stations.empty:
        ne = stations.get("name:en")
        nz = stations.get("name")
        stations["_name"] = (ne.fillna(nz) if (ne is not None and nz is not None)
                             else (nz if nz is not None else "MTR"))
        stations["_cx"] = stations.geometry.centroid.x
        stations["_cy"] = stations.geometry.centroid.y
        stations["_d"]  = stations.geometry.centroid.distance(site_pt)
        stations = (stations.dropna(subset=["_name"])
                             .sort_values("_d").head(4))
        stations = stations[
            (stations["_cx"].between(xmin, xmax)) &
            (stations["_cy"].between(ymin, ymax))
        ]

    # 7. Post-process bus stops
    bus_stops = _spread_bus_stops(_results["bus"], site_pt, BUS_COUNT, min_dist_m=60)
    gc.collect()
    log.info("context: %d bus stops selected", len(bus_stops))

    # 8. Post-process labels
    labels = _results["labels"]
    if not labels.empty:
        ne = labels.get("name:en")
        nz = labels.get("name")
        labels["label"] = (ne.fillna(nz) if (ne is not None and nz is not None)
                           else (nz if nz is not None else None))
        labels = labels.dropna(subset=["label"]).drop_duplicates("label").head(24)

    # ── FIGURE ────────────────────────────────────────────────────────────────
    log.info("context: building figure zoom=%d figsize=%d dpi=%d",
             BASEMAP_ZOOM, FIG_SIZE, FIG_DPI)

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.autoscale(False)

    log.info("context: adding basemap")
    try:
        cx.add_basemap(ax, crs="EPSG:3857",
                       source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=BASEMAP_ZOOM, alpha=0.95)
    except Exception as e:
        log.warning("context: basemap failed (%s)", e)
        ax.set_facecolor("#f0efe9")
    gc.collect()
    log.info("context: basemap done")

    # Landuse layers
    if not residential.empty: residential.plot(ax=ax, color="#f2c6a0", alpha=0.75, zorder=1)
    if not industrial.empty:  industrial.plot(ax=ax,  color="#b39ddb", alpha=0.75, zorder=1)
    if not parks.empty:       parks.plot(ax=ax,       color="#b7dfb9", alpha=0.90, zorder=2)
    if not schools.empty:     schools.plot(ax=ax,     color="#9ecae1", alpha=0.90, zorder=2)
    del residential, industrial, parks, schools; gc.collect()
    log.info("context: landuse done")

    # MTR station polygons (yellow) + logo on centroid
    log.info("context: plotting MTR stations")
    if not stations.empty:
        stations.plot(ax=ax, facecolor=MTR_COLOR, edgecolor="none",
                      linewidth=0, alpha=0.90, zorder=10)
        for _, st in stations.iterrows():
            cx_ = float(st["_cx"])
            cy_ = float(st["_cy"])
            if _MTR_THUMB is not None:
                icon = OffsetImage(_MTR_THUMB, zoom=1.0)
                icon.image.axes = ax
                ax.add_artist(AnnotationBbox(
                    icon, (cx_, cy_), frameon=False,
                    zorder=12, box_alignment=(0.5, 0.5)))
            else:
                ax.scatter([cx_], [cy_], s=300, c="#ED1D24",
                           edgecolors="white", linewidths=1.5, zorder=12)
            nm = st.get("_name", "")
            if nm and isinstance(nm, str):
                ax.text(cx_, cy_ + 120, _wrap(nm, 16),
                        fontsize=8.5, ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.8, pad=1.0),
                        zorder=13, clip_on=True)
    gc.collect()
    log.info("context: stations done")

    # Bus stops — icon if available, else navy dot
    log.info("context: plotting bus stops")
    if not bus_stops.empty:
        bxs = bus_stops["_cx"].tolist()
        bys = bus_stops["_cy"].tolist()
        if _BUS_IMG is not None:
            for bx, by in zip(bxs, bys):
                icon = OffsetImage(_BUS_IMG, zoom=0.028)
                icon.image.axes = ax
                ax.add_artist(AnnotationBbox(
                    icon, (bx, by), frameon=False,
                    zorder=9, box_alignment=(0.5, 0.5)))
        else:
            ax.scatter(bxs, bys, s=55, c="#0d47a1",
                       edgecolors="white", linewidths=1.2, zorder=9)
    log.info("context: bus stops done")

    # Site polygon
    site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred",
                  linewidth=2, zorder=11)
    ax.text(site_geom.centroid.x, site_geom.centroid.y, "SITE",
            color="white", weight="bold", ha="center", va="center",
            fontsize=9, zorder=12)

    # Place labels
    offsets = [(0,35),(0,-35),(35,0),(-35,0),(25,25),(-25,25),
               (25,-25),(-25,-25),(0,50),(50,0),(-50,0),(0,-50)]
    placed = []
    if not labels.empty and "label" in labels.columns:
        for i, (_, lrow) in enumerate(labels.iterrows()):
            try:
                p = lrow.geometry.representative_point()
            except Exception:
                continue
            if p.distance(site_pt) < 140:
                continue
            if any(p.distance(pp) < 120 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x + dx, p.y + dy, _wrap(lrow["label"], 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=12, clip_on=True)
            placed.append(p)

    # Info box — top-left
    ax.text(0.015, 0.985,
            f"Lot: {value}\nOZP Plan: {plan_no}\nZoning: {zone}\nSite Type: {s_type}\n",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6))

    # Legend — single column, bottom-left inside map
    ax.legend(
        handles=[
            mpatches.Patch(color="#f2c6a0", label="Residential"),
            mpatches.Patch(color="#b39ddb", label="Industrial / Commercial"),
            mpatches.Patch(color="#b7dfb9", label="Public Park"),
            mpatches.Patch(color="#9ecae1", label="School / Institution"),
            mpatches.Patch(color=MTR_COLOR,  label="MTR Station"),
            mpatches.Patch(color="#e53935",  label="Site"),
            mlines.Line2D([], [], marker="o", linestyle="None",
                          markerfacecolor="#0d47a1", markeredgecolor="white",
                          markeredgewidth=1.2, markersize=7, label="Bus Stop"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=1,
        fontsize=8.5,
        framealpha=0.95,
        edgecolor="none",
        labelspacing=0.4,
        handlelength=1.2,
        borderpad=0.6,
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
