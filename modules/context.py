import logging
import os
import gc
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import contextily as cx
import matplotlib.patches as mpatches
import numpy as np
import textwrap
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import Optional
from shapely.geometry import Point, MultiPoint
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 25

log = logging.getLogger(__name__)

FETCH_RADIUS  = 600
MAP_HALF_SIZE = 600
MTR_COLOR     = "#ffd166"

_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
try:
    _bus_img = mpimg.imread(_BUS_ICON_PATH)
    log.info(f"[context] bus icon loaded {_bus_img.shape}")
except Exception as e:
    _bus_img = None
    log.warning(f"[context] bus icon not found: {e}")

_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
try:
    _mtr_logo        = mpimg.imread(_MTR_LOGO_PATH)
    _MTR_LOGO_LOADED = True
except Exception:
    _mtr_logo        = None
    _MTR_LOGO_LOADED = False

_EMPTY = gpd.GeoDataFrame(geometry=[], crs=3857)


# ── Site type ─────────────────────────────────────────────────────────────────

def infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                      return "RESIDENTIAL"
    if z.startswith("C"):                      return "COMMERCIAL"
    if z.startswith("G"):                      return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"): return "HOTEL"
    if z.startswith("OU"):                     return "OTHER"
    if z.startswith("I"):                      return "INDUSTRIAL"
    return "MIXED"


# ── Per site-type OSM fetch tags ──────────────────────────────────────────────

SIMILAR_TAGS = {
    "RESIDENTIAL":   {"building": ["apartments", "residential",
                                   "house", "dormitory", "detached", "terrace"]},
    "HOTEL":         {"tourism": ["hotel", "hostel", "resort"],
                      "building": ["hotel"]},
    "COMMERCIAL":    {"building": ["office", "commercial", "retail"],
                      "office": True, "shop": ["mall", "supermarket"]},
    "INSTITUTIONAL": {"amenity": ["school", "college", "university",
                                  "hospital", "government"]},
    "INDUSTRIAL":    {"building": ["industrial", "warehouse"],
                      "landuse": ["industrial"]},
    "OTHER":         {"building": True},
    "MIXED":         {"building": True},
}

SUPPORT_TAGS = {
    "RESIDENTIAL":   {"amenity": ["school", "college", "university",
                                  "hospital", "supermarket", "clinic"],
                      "leisure": ["park"], "shop": ["supermarket"]},
    "HOTEL":         {"tourism": ["attraction"],
                      "amenity": ["restaurant", "cafe"],
                      "shop":    ["mall"]},
    "COMMERCIAL":    {"amenity": ["bank", "restaurant"],
                      "railway": ["station"]},
    "INSTITUTIONAL": {"leisure": ["park"], "amenity": ["library", "hospital"]},
    "INDUSTRIAL":    {"landuse": ["port"], "highway": ["motorway"]},
    "OTHER":         {"amenity": True},
    "MIXED":         {"amenity": True, "leisure": True},
}

HIGHLIGHT_COLOR = {
    "RESIDENTIAL":   "#e07b39",
    "HOTEL":         "#b15928",
    "COMMERCIAL":    "#6a3d9a",
    "INSTITUTIONAL": "#1f78b4",
    "INDUSTRIAL":    "#33a02c",
    "OTHER":         "#888888",
    "MIXED":         "#888888",
}
HIGHLIGHT_LABEL = {
    "RESIDENTIAL":   "Residential Developments",
    "HOTEL":         "Hotels & Serviced Apartments",
    "COMMERCIAL":    "Office / Commercial Buildings",
    "INSTITUTIONAL": "Institutional Buildings",
    "INDUSTRIAL":    "Industrial / Warehouse Buildings",
    "OTHER":         "Nearby Buildings",
    "MIXED":         "Nearby Buildings",
}
SUPPORT_COLOR = {
    "RESIDENTIAL":   "#9ecae1",
    "HOTEL":         "#fb9a99",
    "COMMERCIAL":    "#cab2d6",
    "INSTITUTIONAL": "#b7dfb9",
    "INDUSTRIAL":    "#b2df8a",
    "OTHER":         "#dddddd",
    "MIXED":         "#dddddd",
}
SUPPORT_LABEL = {
    "RESIDENTIAL":   "Schools / Parks / Hospitals",
    "HOTEL":         "Restaurants / Attractions",
    "COMMERCIAL":    "Banks / Restaurants / Transit",
    "INSTITUTIONAL": "Parks / Libraries",
    "INDUSTRIAL":    "Ports / Motorways",
    "OTHER":         "Amenities",
    "MIXED":         "Amenities / Parks",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def wrap_label(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _fetch_one(lat, lon, dist, tags) -> gpd.GeoDataFrame:
    try:
        gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags)
        if gdf is not None and not gdf.empty:
            return gdf.to_crs(3857)
    except Exception as e:
        log.debug(f"[context] fetch {list(tags.keys())[:3]}: {e}")
    return _EMPTY.copy()


def _parallel_fetch(tasks: dict, wall_timeout: float = 120) -> dict:
    results = {k: _EMPTY.copy() for k in tasks}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(_fetch_one, *args): key
                   for key, args in tasks.items()}
        done, not_done = wait(futures, timeout=wall_timeout,
                              return_when=ALL_COMPLETED)
        for f in not_done:
            f.cancel()
            log.warning(f"[context] timeout: {futures[f]}")
        for f in done:
            key = futures[f]
            try:
                results[key] = f.result()
                log.info(f"[context] ✓ {key}: {len(results[key])} rows")
            except Exception as e:
                log.warning(f"[context] ✗ {key}: {e}")
    return results


def _col(gdf, col):
    if col in gdf.columns:
        return gdf[col]
    return pd.Series([None] * len(gdf), index=gdf.index)


def _filter_col(gdf, col, val):
    if gdf.empty or col not in gdf.columns:
        return _EMPTY.copy()
    s    = gdf[col]
    mask = s.isin(val) if isinstance(val, list) else (s == val)
    return gdf[mask].copy()


def _polys_only(gdf):
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])].copy()


def _points_only(gdf):
    if gdf.empty:
        return _EMPTY.copy()
    return gdf[gdf.geometry.geom_type.isin(["Point"])].copy()


def _get_name(gdf):
    return _col(gdf, "name:en").fillna(_col(gdf, "name"))


def _ascii_only(text) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    ascii_chars = sum(1 for c in s if ord(c) < 128)
    if ascii_chars / max(len(s), 1) < 0.5:
        return None
    cleaned = "".join(c for c in s if ord(c) < 128).strip()
    return cleaned if cleaned else None


def _safe_plot(gdf, ax, **kwargs):
    try:
        if not gdf.empty:
            gdf.plot(ax=ax, **kwargs)
    except Exception as e:
        log.debug(f"[context] plot: {e}")


def _draw_mtr_icon(ax, x, y, zoom=0.035, zorder=14):
    if _MTR_LOGO_LOADED and _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False,
                            zorder=zorder, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=12,
                markeredgecolor="white", markeredgewidth=2, zorder=zorder)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_context(
    data_type: str,
    value: str,
    ZONE_DATA: gpd.GeoDataFrame,
    radius_m: Optional[int] = None,
    lon: float = None,
    lat: float = None,
    lot_ids: list = None,
    extents: list = None,
):
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r   = radius_m if radius_m is not None else FETCH_RADIUS
    half_size = radius_m if radius_m is not None else MAP_HALF_SIZE
    half_x    = half_size * (992 / 737)
    half_y    = half_size
    log.info(f"[context] lon={lon:.5f} lat={lat:.5f} r={fetch_r}m")

    # ── Site polygon ──────────────────────────────────────────────────────────
    try:
        lot_gdf = get_lot_boundary(lon, lat, data_type, extents)
    except Exception as e:
        log.warning(f"[context] get_lot_boundary: {e}")
        lot_gdf = None

    if lot_gdf is not None and not lot_gdf.empty:
        site_geom  = lot_gdf.geometry.iloc[0]
        site_gdf   = lot_gdf
        site_point = site_geom.centroid
        log.info(f"[context] lot boundary area={site_geom.area:.0f}m²")
    else:
        site_point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
        site_geom  = site_point.buffer(80)
        site_gdf   = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)
        log.info("[context] using fallback buffer")

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp     = ZONE_DATA.to_crs(3857)
    primary = ozp[ozp.contains(site_point)]
    if primary.empty:
        raise ValueError("No zoning polygon found for this site.")
    primary   = primary.iloc[0]
    zone      = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    sim_tags  = SIMILAR_TAGS.get(SITE_TYPE, SIMILAR_TAGS["MIXED"])
    sup_tags  = SUPPORT_TAGS.get(SITE_TYPE, SUPPORT_TAGS["MIXED"])
    hi_color  = HIGHLIGHT_COLOR.get(SITE_TYPE, "#888888")
    hi_label  = HIGHLIGHT_LABEL.get(SITE_TYPE, "Nearby Buildings")
    sp_color  = SUPPORT_COLOR.get(SITE_TYPE, "#dddddd")
    sp_label  = SUPPORT_LABEL.get(SITE_TYPE, "Amenities")
    log.info(f"[context] zone={zone} SITE_TYPE={SITE_TYPE}")

    # ── 4 parallel fetches ────────────────────────────────────────────────────
    # base     = landuse + leisure + bus_stops + place labels (one merged call)
    # similar  = type-specific similar buildings
    # support  = type-specific supporting amenities
    # stations = MTR stations
    log.info("[context] Fetching 4 tasks in parallel (120s limit)...")
    results = _parallel_fetch({
        "base": (lat, lon, fetch_r, {
            "landuse": True,
            "leisure": ["park", "playground", "garden", "recreation_ground"],
            "highway": ["bus_stop"],
            "place":   ["neighbourhood", "suburb"],
        }),
        "similar":  (lat, lon, fetch_r, sim_tags),
        "support":  (lat, lon, fetch_r, sup_tags),
        "stations": (lat, lon, 1200,    {"railway": "station"}),
    }, wall_timeout=120)
    gc.collect()

    # ── Process base ──────────────────────────────────────────────────────────
    base             = results["base"]
    residential_area = _filter_col(base, "landuse", "residential")
    industrial_area  = _filter_col(base, "landuse", ["industrial", "commercial"])
    parks            = _filter_col(base, "leisure", "park")

    # Bus stops: points tagged highway=bus_stop
    bus_stops_raw = _EMPTY.copy()
    if not base.empty and "highway" in base.columns:
        bs = base[_col(base, "highway") == "bus_stop"].copy()
        if not bs.empty:
            # Ensure we have Point geometry for bus stops
            bs["geometry"] = bs.geometry.apply(
                lambda g: g.centroid if g.geom_type != "Point" else g)
            bus_stops_raw = gpd.GeoDataFrame(bs, crs=3857)
    log.info(f"[context] bus stops raw: {len(bus_stops_raw)}")

    # ── Process support ───────────────────────────────────────────────────────
    support_raw = results["support"]
    schools     = _filter_col(support_raw, "amenity",
                              ["school", "college", "university"]) \
                  if not support_raw.empty else _EMPTY.copy()
    hospitals   = _filter_col(support_raw, "amenity", ["hospital", "clinic"]) \
                  if not support_raw.empty else _EMPTY.copy()
    support_blds = _polys_only(support_raw)

    # ── Process similar buildings ─────────────────────────────────────────────
    _sim_raw = _polys_only(results["similar"])
    if not _sim_raw.empty:
        _sim_raw        = _sim_raw.copy()
        _sim_raw["_nm"] = _get_name(_sim_raw)
        _named  = _sim_raw["_nm"].apply(
            lambda x: bool(_ascii_only(str(x))) if pd.notna(x) else False)
        _large  = _sim_raw.geometry.area >= 500
        similar_blds = _sim_raw[_named | _large].copy()
        log.info(f"[context] similar after filter: {len(similar_blds)}")
    else:
        similar_blds = _EMPTY.copy()

    # ── Process stations ──────────────────────────────────────────────────────
    stations_in_view = _EMPTY.copy()
    nearest_stn_name = None
    stations_raw     = results["stations"]

    if not stations_raw.empty:
        s             = stations_raw.copy()
        s["name"]     = _get_name(s)
        s["centroid"] = s.geometry.centroid
        s["dist"]     = s["centroid"].apply(lambda g: g.distance(site_point))
        stn           = s.dropna(subset=["name"]).sort_values("dist").head(3)
        stations_in_view = stn[stn["dist"] <= half_size * 1.4]
        if not stn.empty:
            raw_name = stn.iloc[0]["name"]
            nearest_stn_name = _ascii_only(str(raw_name)) if raw_name else None

    # ── Cluster bus stops to max 6 ────────────────────────────────────────────
    bus_stops = bus_stops_raw.copy()
    if len(bus_stops) > 6:
        try:
            from sklearn.cluster import KMeans
            pts    = [g.centroid if hasattr(g, "centroid") and g.geom_type != "Point"
                      else g for g in bus_stops.geometry]
            coords = np.array([[p.x, p.y] for p in pts])
            bus_stops["cluster"] = KMeans(
                n_clusters=6, random_state=0).fit(coords).labels_
            bus_stops = gpd.GeoDataFrame(
                bus_stops.groupby("cluster").first(), crs=3857)
        except Exception as e:
            log.debug(f"[context] KMeans: {e}")
            bus_stops = bus_stops.head(6)
    log.info(f"[context] bus stops after cluster: {len(bus_stops)}")

    # ── Place labels ──────────────────────────────────────────────────────────
    all_label_items = []
    seen_texts      = set()

    def _collect(gdf):
        if gdf is None or gdf.empty:
            return
        g        = gdf.copy()
        g["_lb"] = _get_name(g)
        named    = g.dropna(subset=["_lb"])
        named    = named[named["_lb"].astype(str).str.strip().str.len() > 0]
        for _, row in named.iterrows():
            text = _ascii_only(str(row["_lb"]).strip())
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            geom = row.geometry
            p    = (geom.representative_point()
                    if hasattr(geom, "representative_point") else geom.centroid)
            all_label_items.append((p.distance(site_point), geom, text))

    for src in [base, schools, hospitals, parks, similar_blds, support_blds]:
        _collect(src)

    all_label_items.sort(key=lambda x: x[0])
    all_label_items = [(g, t) for _, g, t in all_label_items[:30]]

    del base, support_raw, results
    gc.collect()

    log.info(f"[context] Rendering — {len(all_label_items)} labels, "
             f"bus={len(bus_stops)}, stn={len(stations_in_view)}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16.15, 12))
    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=16, alpha=0.95)
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                           zoom=16, alpha=0.95)
        except Exception as e:
            log.warning(f"[context] basemap: {e}")

    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # ── Base layers (zorder 1-2) ──────────────────────────────────────────────
    _safe_plot(residential_area, ax, color="#f2c6a0", alpha=0.75, zorder=1)
    _safe_plot(industrial_area,  ax, color="#b39ddb", alpha=0.75, zorder=1)
    _safe_plot(parks,            ax, color="#b7dfb9", alpha=0.90, zorder=2)
    _safe_plot(schools,          ax, color="#9ecae1", alpha=0.90, zorder=2)
    _safe_plot(hospitals,        ax, color="#aec6cf", alpha=0.90, zorder=2)

    del residential_area, industrial_area, parks, schools, hospitals
    gc.collect()

    # ── Type-specific layers (zorder 3-4) ─────────────────────────────────────
    _safe_plot(support_blds, ax, color=sp_color, alpha=0.60, zorder=3)
    _safe_plot(similar_blds, ax, color=hi_color,  alpha=0.80, zorder=4)

    del support_blds, similar_blds
    gc.collect()

    # ── 400m pedestrian catchment ring (zorder 5) ────────────────────────────
    try:
        catchment = gpd.GeoDataFrame(
            geometry=[site_point.buffer(400)], crs=3857)
        catchment.plot(ax=ax, facecolor="none", edgecolor="#555555",
                       linewidth=1.2, linestyle="--", alpha=0.55, zorder=5)
        # Label the ring
        ax.text(site_point.x, site_point.y + 410, "400m walk",
                fontsize=7.5, color="#555555", ha="center", va="bottom",
                alpha=0.75, zorder=5)
    except Exception as e:
        log.debug(f"[context] catchment: {e}")

    # ── Straight-line connector to nearest MTR (zorder 6) ────────────────────
    if not stations_in_view.empty:
        try:
            nearest = stations_in_view.iloc[0]
            nc      = nearest.geometry.centroid
            ax.annotate("",
                xy=(nc.x, nc.y), xytext=(site_point.x, site_point.y),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#005eff",
                    lw=2.0,
                    linestyle="dashed",
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=6,
            )
            # Distance label on midpoint
            mid_x = (site_point.x + nc.x) / 2
            mid_y = (site_point.y + nc.y) / 2
            dist_m = int(site_point.distance(nc))
            ax.text(mid_x, mid_y, f"~{dist_m}m walk",
                    fontsize=7.5, color="#005eff", ha="center", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=1),
                    zorder=6)
        except Exception as e:
            log.debug(f"[context] walk line: {e}")

    # ── Bus stops (zorder 9) ──────────────────────────────────────────────────
    if not bus_stops.empty:
        try:
            for _, row in bus_stops.iterrows():
                g  = row.geometry
                pt = g if g.geom_type == "Point" else g.centroid
                bx, by = pt.x, pt.y
                if _bus_img is not None:
                    icon = OffsetImage(_bus_img, zoom=0.025)
                    icon.image.axes = ax
                    ab = AnnotationBbox(icon, (bx, by), frameon=False,
                                        zorder=9, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
                else:
                    ax.plot(bx, by, "s", color="#0d47a1",
                            markersize=10, zorder=9,
                            markeredgecolor="white", markeredgewidth=1)
        except Exception as e:
            log.debug(f"[context] bus render: {e}")
    del bus_stops
    gc.collect()

    # ── MTR stations (zorder 10-14) ───────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            stations_in_view.plot(ax=ax, facecolor=MTR_COLOR,
                                  edgecolor="none", alpha=0.9, zorder=10)
            for _, st in stations_in_view.iterrows():
                c = st.geometry.centroid
                _draw_mtr_icon(ax, c.x, c.y, zoom=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # ── Site (zorder 15-16) ───────────────────────────────────────────────────
    try:
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="white",
                      linewidth=2.5, zorder=15)
        sc = site_geom.centroid
        ax.text(sc.x, sc.y, "SITE", color="white", weight="bold",
                fontsize=8, ha="center", va="center", zorder=16)
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    # ── Place labels (zorder 13) ──────────────────────────────────────────────
    placed  = []
    offsets = [(0, 40), (0, -40), (40, 0), (-40, 0), (28, 28), (-28, 28)]
    for i, (geom, text) in enumerate(all_label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_point) < 100:
                continue
            if any(p.distance(pp) < 95 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x + dx, p.y + dy, wrap_label(text, 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=13, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # ── MTR station name labels (zorder 17) ──────────────────────────────────
    if not stations_in_view.empty:
        try:
            for _, st in stations_in_view.iterrows():
                raw   = st.get("name")
                label = _ascii_only(str(raw).strip()) if raw else None
                if not label:
                    continue
                c = st.geometry.centroid
                ax.text(c.x, c.y + 140, wrap_label(label, 18),
                        fontsize=9, weight="bold", color="black",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=2),
                        zorder=17)
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    # ── Info box ──────────────────────────────────────────────────────────────
    info_lines = (
        f"{data_type}: {value}\n"
        f"OZP Plan: {primary['PLAN_NO']}\n"
        f"Zoning: {zone}\n"
        f"Site Type: {SITE_TYPE}"
    )
    if nearest_stn_name:
        info_lines += f"\nNearest MTR: {nearest_stn_name}"

    ax.text(0.012, 0.988, info_lines,
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6),
            zorder=20)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color="#f2c6a0", label="Residential Area"),
        mpatches.Patch(color="#b39ddb", label="Industrial / Commercial Area"),
        mpatches.Patch(color="#b7dfb9", label="Public Park"),
        mpatches.Patch(color="#9ecae1", label="School / Institution"),
        mpatches.Patch(color="#aec6cf", label="Hospital / Clinic"),
        mpatches.Patch(color=hi_color,  alpha=0.80, label=hi_label),
        mpatches.Patch(color=sp_color,  alpha=0.60, label=sp_label),
        mpatches.Patch(color=MTR_COLOR, label="MTR Station"),
        mpatches.Patch(color="#e53935", label="Site"),
        mpatches.Patch(color="#0d47a1", label="Bus Stop"),
        mlines.Line2D([], [], color="#005eff", linewidth=2,
                      linestyle="--", label="Route to MTR"),
        mlines.Line2D([], [], color="#555555", linewidth=1.2,
                      linestyle="--", label="400m Walk Catchment"),
    ]
    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(0.02, 0.02),
              fontsize=8.5, framealpha=0.95)

    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=15, weight="bold",
    )
    ax.set_axis_off()
    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.autoscale(False)

    buf = BytesIO()
    plt.tight_layout()
    ax.set_position([0.02, 0.02, 0.96, 0.96])
    plt.savefig(buf, format="png", dpi=110,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("[context] Done.")
    return buf
