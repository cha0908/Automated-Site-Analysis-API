"""
modules/context.py  v3
──────────────────────────────────────────────────────────────────────────────
Site Context Analysis — matches the Colab reference output.

Changes from v2:
  ✓ Walk-graph / walk-routes REMOVED entirely (was the crash source)
  ✓ MTR stations drawn as coloured polygon footprints + MTR logo on centroid
  ✓ Bus stops: up to 10 nearest, drawn with a custom bus-stop icon
  ✓ Zoning type inferred from OZP dataset (same as Colab)
  ✓ Every OSM fetch wrapped in try/except + gc.collect()
  ✓ requests_timeout = 25 s throughout
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
import matplotlib.image as mpimg
import matplotlib.lines as mlines

import numpy as np
import geopandas as gpd
import osmnx as ox
import contextily as cx

from shapely.geometry import Point
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 25

# ── Paths ─────────────────────────────────────────────────────────────────────
_STATIC = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC, "HK_MTR_logo.png")

# ── Constants ─────────────────────────────────────────────────────────────────
FETCH_RADIUS  = 1500
MAP_HALF_SIZE = 900
MTR_COLOR     = "#ffd166"
BUS_COUNT     = 10
MTR_RADIUS_M  = 2000


# ── Logo loaders ──────────────────────────────────────────────────────────────

def _load_mtr_logo():
    try:
        img = mpimg.imread(_MTR_LOGO_PATH)
        log.info("context: MTR logo loaded")
        return img
    except Exception as e:
        log.warning("context: MTR logo not found (%s) — using roundel fallback", e)
        return None


def _make_bus_icon(size=40):
    """Create a simple navy bus-stop circle icon as a numpy array."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    d.ellipse([0, 0, size-1, size-1], fill=(13, 71, 161, 230))   # #0d47a1
    # white "B" label
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        d.text((size//2 - 4, size//2 - 6), "B", fill=(255,255,255,255), font=font)
    except Exception:
        pass
    return np.array(img)


_MTR_IMG = _load_mtr_logo()
_BUS_ICON = _make_bus_icon(40)


def _place_logo(ax, x, y, img_arr, zoom=0.045, zorder=12):
    """Place a numpy-array image centred at (x, y)."""
    if img_arr is None:
        return
    icon = OffsetImage(img_arr, zoom=zoom)
    icon.image.axes = ax
    ab = AnnotationBbox(icon, (x, y), frameon=False,
                        zorder=zorder, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)


def _draw_roundel(ax, x, y, size=55, color="#ED1D24", zorder=11):
    """Fallback MTR roundel when logo PNG is missing."""
    ax.add_patch(plt.Circle((x, y), size, color=color, zorder=zorder))
    bw, bh = size * 2.0, size * 0.55
    ax.add_patch(plt.Rectangle((x-bw/2, y-bh/2), bw, bh,
                                color="white", zorder=zorder+1))
    ax.add_patch(plt.Circle((x, y), size*0.55, color="white", zorder=zorder+2))
    ax.add_patch(plt.Rectangle((x-bw/2, y-bh/2), bw, bh,
                                color=color, zorder=zorder+3))


# ── Text helpers ──────────────────────────────────────────────────────────────

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
    if site_type == "RESIDENTIAL":
        return {"amenity": ["school","college","university"],
                "leisure": ["park"], "place": ["neighbourhood"]}
    if site_type == "COMMERCIAL":
        return {"amenity": ["bank","restaurant","market"], "railway": ["station"]}
    if site_type == "INSTITUTIONAL":
        return {"amenity": ["school","college","hospital"], "leisure": ["park"]}
    return {"amenity": True, "leisure": True}


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

    # 1. Resolve coords
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    log.info("context: %.6f, %.6f", lon, lat)

    site_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    # 2. OZP zoning (from dataset — same as Colab)
    hits = zone_data[zone_data.contains(site_pt)]
    if hits.empty:
        zone_data = zone_data.copy()
        zone_data["_d"] = zone_data.geometry.distance(site_pt)
        hits = zone_data[zone_data["_d"] < 200].sort_values("_d")
    if hits.empty:
        raise ValueError("No OZP zone found near site.")

    row      = hits.iloc[0]
    zone     = row.get("ZONE_LABEL") or row.get("ZONE") or "N/A"
    plan_no  = row.get("PLAN_NO")    or row.get("PLAN")  or "N/A"
    s_type   = _infer_site_type(zone)
    lbl_tags = _label_rules(s_type)

    # 3. OSM landuse polygons
    log.info("context: fetching landuse polygons")
    polys = _safe_osm(lat, lon, FETCH_RADIUS,
                      {"landuse":True,"leisure":True,"amenity":True,"building":True},
                      "landuse")

    if not polys.empty:
        lu = polys.get("landuse", gpd.GeoSeries(dtype=object))
        le = polys.get("leisure", gpd.GeoSeries(dtype=object))
        am = polys.get("amenity", gpd.GeoSeries(dtype=object))
        bu = polys.get("building", gpd.GeoSeries(dtype=object))
        residential = polys[lu == "residential"]
        industrial  = polys[lu.isin(["industrial","commercial"])]
        parks       = polys[le == "park"]
        schools     = polys[am.isin(["school","college","university"])]
        buildings   = polys[bu.notnull()]
    else:
        residential = industrial = parks = schools = buildings = _empty_gdf()

    # 4. Site footprint — lot boundary first, then OSM fallback
    lot_gdf = get_lot_boundary(lon, lat, data_type,
                               extents if len(extents) > 1 else None)
    if lot_gdf is not None and not lot_gdf.empty:
        site_geom = lot_gdf.geometry.iloc[0]
    elif not polys.empty:
        cands = polys[
            polys.geometry.geom_type.isin(["Polygon","MultiPolygon"]) &
            (polys.geometry.distance(site_pt) < 40)
        ]
        site_geom = (
            cands.assign(a=cands.area).sort_values("a",ascending=False).geometry.iloc[0]
            if len(cands) else site_pt.buffer(40)
        )
    else:
        site_geom = site_pt.buffer(40)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # 5. MTR stations (polygon footprints + centroid for logo)
    log.info("context: fetching MTR stations")
    stations = _safe_osm(lat, lon, MTR_RADIUS_M, {"railway":"station"}, "stations")

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
                             .head(4))   # keep nearest 4 station polygons

    # 6. Bus stops — 10 nearest
    log.info("context: fetching bus stops")
    bus_raw = _safe_osm(lat, lon, BUS_COUNT * 150 + 500,
                        {"highway":"bus_stop"}, "bus_stops")

    if not bus_raw.empty:
        bus_raw["_d"] = bus_raw.geometry.distance(site_pt)
        bus_stops = bus_raw.sort_values("_d").head(BUS_COUNT)
    else:
        bus_stops = _empty_gdf()

    # 7. Place labels (no walk graph — removed)
    log.info("context: fetching place labels")
    labels = _safe_osm(lat, lon, 800, lbl_tags, "labels")

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

    # Landuse layers
    if not residential.empty: residential.plot(ax=ax, color="#f2c6a0", alpha=0.75)
    if not industrial.empty:  industrial.plot(ax=ax,  color="#b39ddb", alpha=0.75)
    if not parks.empty:       parks.plot(ax=ax,       color="#b7dfb9", alpha=0.90)
    if not schools.empty:     schools.plot(ax=ax,     color="#9ecae1", alpha=0.90)
    if not buildings.empty:   buildings.plot(ax=ax,   color="#d9d9d9", alpha=0.35)

    # MTR station polygon footprints (yellow)
    if not stations.empty:
        stations.plot(ax=ax, facecolor=MTR_COLOR, edgecolor="#cc8800",
                      linewidth=1.2, alpha=0.90, zorder=10)
        # MTR logo or roundel on each station centroid
        for _, st in stations.iterrows():
            cx_ = st["_cx"]
            cy_ = st["_cy"]
            if _MTR_IMG is not None:
                _place_logo(ax, cx_, cy_, _MTR_IMG, zoom=0.045, zorder=12)
            else:
                _draw_roundel(ax, cx_, cy_, size=55, zorder=11)
            # station name label
            nm = st.get("_name","")
            if nm and isinstance(nm, str):
                ax.text(cx_, cy_ + 110, _wrap(nm, 18),
                        fontsize=8, ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=1.5),
                        zorder=13, clip_on=True)

    # Bus stops with icon
    if not bus_stops.empty:
        for _, bs in bus_stops.iterrows():
            bx = bs.geometry.x if bs.geometry.geom_type == "Point" else bs.geometry.centroid.x
            by = bs.geometry.y if bs.geometry.geom_type == "Point" else bs.geometry.centroid.y
            _place_logo(ax, bx, by, _BUS_ICON, zoom=0.55, zorder=11)

    # Site polygon
    site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred",
                  linewidth=2, zorder=14)
    ax.text(site_geom.centroid.x, site_geom.centroid.y, "SITE",
            color="white", weight="bold", ha="center", va="center",
            fontsize=9, zorder=15)

    # Place labels
    offsets = [(0,35),(0,-35),(35,0),(-35,0),(25,25),(-25,25)]
    placed  = []
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
            ax.text(p.x+dx, p.y+dy, _wrap(lrow["label"], 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
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
        mpatches.Patch(color="#0d47a1",  label="Bus Stop"),
    ]

    # MTR logo in legend
    if _MTR_IMG is not None:
        try:
            raw = (_MTR_IMG*255).astype(np.uint8) if _MTR_IMG.dtype != np.uint8 else _MTR_IMG
            thumb = np.array(Image.fromarray(raw).resize((18,18), Image.LANCZOS))
            import matplotlib.legend_handler as lh

            class _LH(mlines.Line2D): pass
            class _LHH(lh.HandlerBase):
                def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                    ib = OffsetImage(thumb, zoom=0.9)
                    return [AnnotationBbox(ib,(xd+w/2,yd+h/2),
                                          xycoords=trans,frameon=False)]
            legend_handles.append(_LH([], [], label="MTR Logo"))
            handler_map = {_LH: _LHH()}
        except Exception:
            handler_map = {}
    else:
        handler_map = {}

    # Bus icon in legend
    try:
        bus_thumb = np.array(Image.fromarray(_BUS_ICON).resize((18,18), Image.LANCZOS))
        import matplotlib.legend_handler as lh2

        class _BH(mlines.Line2D): pass
        class _BHH(lh2.HandlerBase):
            def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                ib = OffsetImage(bus_thumb, zoom=0.9)
                return [AnnotationBbox(ib,(xd+w/2,yd+h/2),
                                       xycoords=trans,frameon=False)]
        legend_handles.append(_BH([], [], label="Bus Stop Icon"))
        handler_map[_BH] = _BHH()
    except Exception:
        pass

    ax.legend(
        handles=legend_handles,
        handler_map=handler_map if handler_map else None,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.08),
        fontsize=8.5,
        framealpha=0.95,
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
