"""
modules/context.py  v7
──────────────────────────────────────────────────────────────────────────────
v7 crash fixes:
  - Bus stops: AnnotationBbox removed entirely — plain ax.scatter() dots only
  - MTR logo: single shared OffsetImage reused across stations (not recreated)
  - Landuse: clip to map bbox BEFORE plotting (5976→~200 rows)
  - buildings layer removed (too many polygons, not in reference output)
  - del + gc after every heavy operation
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

FETCH_RADIUS  = 1500
MAP_HALF_SIZE = 900
MTR_COLOR     = "#ffd166"
BUS_COUNT     = 10
BUS_FETCH_R   = 1200
MTR_RADIUS_M  = 2000
LABEL_RADIUS  = 800
FIG_SIZE      = 10
FIG_DPI       = 120
BASEMAP_ZOOM  = 15


def _load_mtr():
    try:
        img = mpimg.imread(_MTR_LOGO_PATH)
        log.info("context: MTR logo loaded")
        return img
    except Exception as e:
        log.warning("context: MTR logo missing (%s)", e)
        return None

_MTR_IMG = _load_mtr()


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
    """Clip GeoDataFrame to map bbox to reduce polygon count before plotting."""
    if gdf.empty:
        return gdf
    try:
        clip_box = gpd.GeoDataFrame(geometry=[sbox(xmin, ymin, xmax, ymax)], crs=3857)
        clipped  = gpd.clip(gdf, clip_box)
        log.info("context: clipped %d → %d rows", len(gdf), len(clipped))
        return clipped
    except Exception as e:
        log.warning("context: clip failed (%s), using full gdf", e)
        return gdf


def _spread_bus_stops(gdf, site_pt, n_total=10, min_dist_m=60):
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
        q  = (int(row["_dx"] >= 0), int(row["_dy"] >= 0))
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

    # 3. Landuse — fetch then CLIP to map extent immediately
    log.info("context: fetching landuse")
    polys_raw = _safe_osm(lat, lon, FETCH_RADIUS,
                          {"landuse": True, "leisure": True, "amenity": True},
                          "landuse")
    polys = _clip_to_map(polys_raw, xmin, ymin, xmax, ymax)
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

    # 4. Site footprint
    lot_gdf = get_lot_boundary(lon, lat, data_type,
                               extents if len(extents) > 1 else None)
    site_geom = (lot_gdf.geometry.iloc[0]
                 if lot_gdf is not None and not lot_gdf.empty
                 else site_pt.buffer(80))
    site_gdf  = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)
    log.info("context: site footprint ready")

    # 5. MTR stations
    log.info("context: fetching stations")
    stations = _safe_osm(lat, lon, MTR_RADIUS_M, {"railway": "station"}, "stations")
    if not stations.empty:
        ne = stations.get("name:en")
        nz = stations.get("name")
        stations["_name"] = (ne.fillna(nz) if (ne is not None and nz is not None)
                             else (nz if nz is not None else "MTR"))
        stations["_cx"] = stations.geometry.centroid.x
        stations["_cy"] = stations.geometry.centroid.y
        stations["_d"]  = stations.geometry.centroid.distance(site_pt)
        stations = stations.dropna(subset=["_name"]).sort_values("_d").head(4)
        # keep only stations inside map extent
        stations = stations[
            (stations["_cx"] >= xmin) & (stations["_cx"] <= xmax) &
            (stations["_cy"] >= ymin) & (stations["_cy"] <= ymax)
        ]

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
        labels["label"] = (ne.fillna(nz) if (ne is not None and nz is not None)
                           else (nz if nz is not None else None))
        labels = labels.dropna(subset=["label"]).drop_duplicates("label").head(24)

    # ── 8. FIGURE ─────────────────────────────────────────────────────────────
    log.info("context: building figure")
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    fig.subplots_adjust(bottom=0.14)
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

    # Landuse
    log.info("context: plotting landuse")
    if not residential.empty: residential.plot(ax=ax, color="#f2c6a0", alpha=0.75, zorder=1)
    if not industrial.empty:  industrial.plot(ax=ax,  color="#b39ddb", alpha=0.75, zorder=1)
    if not parks.empty:       parks.plot(ax=ax,       color="#b7dfb9", alpha=0.90, zorder=2)
    if not schools.empty:     schools.plot(ax=ax,     color="#9ecae1", alpha=0.90, zorder=2)
    del residential, industrial, parks, schools; gc.collect()
    log.info("context: landuse done")

    # MTR stations — polygon fill + logo using scatter (no AnnotationBbox loop)
    log.info("context: plotting MTR stations")
    if not stations.empty:
        stations.plot(ax=ax, facecolor=MTR_COLOR, edgecolor="#b8860b",
                      linewidth=1.5, alpha=0.95, zorder=8)

        if _MTR_IMG is not None:
            # Resize once, reuse for all stations
            raw   = (_MTR_IMG * 255).astype(np.uint8) if _MTR_IMG.dtype != np.uint8 else _MTR_IMG
            thumb = np.array(Image.fromarray(raw).resize((28, 28), Image.LANCZOS))
            for _, st in stations.iterrows():
                cx_ = float(st["_cx"])
                cy_ = float(st["_cy"])
                icon = OffsetImage(thumb, zoom=1.0)
                icon.image.axes = ax
                ax.add_artist(AnnotationBbox(icon, (cx_, cy_),
                                             frameon=False, zorder=13,
                                             box_alignment=(0.5, 0.5)))
                nm = st.get("_name", "")
                if nm and isinstance(nm, str):
                    ax.text(cx_, cy_ + 130, _wrap(nm, 18),
                            fontsize=8, ha="center", va="bottom", weight="bold",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.85, pad=2),
                            zorder=14, clip_on=True)
            del thumb, raw
        else:
            # Fallback: draw roundel circles via scatter
            xs = stations["_cx"].tolist()
            ys = stations["_cy"].tolist()
            ax.scatter(xs, ys, s=600, c="#ED1D24",
                       edgecolors="white", linewidths=2, zorder=13)

    gc.collect()
    log.info("context: stations done")

    # Bus stops — plain scatter dots, zero memory overhead
    log.info("context: plotting bus stops")
    if not bus_stops.empty:
        bxs = bus_stops["_cx"].tolist()
        bys = bus_stops["_cy"].tolist()
        ax.scatter(bxs, bys, s=120, c="#0d47a1",
                   edgecolors="white", linewidths=1.5, zorder=10)
    log.info("context: bus stops done")

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
        mlines.Line2D([], [], marker="o", linestyle="None",
                      markerfacecolor="#0d47a1", markeredgecolor="white",
                      markeredgewidth=1.5, markersize=9, label="Bus Stop"),
    ]
    handler_map = {}

    if _MTR_IMG is not None:
        try:
            raw   = (_MTR_IMG*255).astype(np.uint8) if _MTR_IMG.dtype != np.uint8 else _MTR_IMG
            lthumb = np.array(Image.fromarray(raw).resize((20, 20), Image.LANCZOS))
            class _MLH(mlines.Line2D): pass
            class _MLHH(lh.HandlerBase):
                def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                    return [AnnotationBbox(OffsetImage(lthumb, zoom=1.0),
                                          (xd+w/2, yd+h/2),
                                          xycoords=trans, frameon=False)]
            legend_handles.append(_MLH([], [], label="MTR Icon"))
            handler_map[_MLH] = _MLHH()
        except Exception:
            pass

    ax.legend(
        handles=legend_handles,
        handler_map=handler_map or None,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.02),
        bbox_transform=ax.transAxes,
        ncol=4, fontsize=8.5, framealpha=0.95,
        edgecolor="#333", title="Legend", title_fontsize=9,
        labelspacing=0.3, handlelength=1.4,
        borderpad=0.5, columnspacing=1.0,
    )

    ax.set_title("Automated Site Context Analysis (Building-Type Driven)",
                 fontsize=15, weight="bold")
    ax.set_axis_off()

    log.info("context: saving PNG dpi=%d", FIG_DPI)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("context: done")
    return buf
