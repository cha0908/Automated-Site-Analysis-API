import logging
import os
import gc
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import contextily as cx
import matplotlib.patches as mpatches
import numpy as np
import textwrap
import pandas as pd
from typing import Optional
from shapely.geometry import Point
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 20

log = logging.getLogger(__name__)

FETCH_RADIUS            = 800
MAP_HALF_SIZE           = 600
MTR_COLOR               = "#ffd166"   # yellow — matches Colab target
WALK_ROUTE_COLOR        = "#005eff"

_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
try:
    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
except Exception:
    _bus_icon = None

_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
try:
    _mtr_logo        = mpimg.imread(_MTR_LOGO_PATH)
    _MTR_LOGO_LOADED = True
except Exception:
    _mtr_logo        = None
    _MTR_LOGO_LOADED = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def wrap_label(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                        return "RESIDENTIAL"
    if z.startswith("C"):                        return "COMMERCIAL"
    if z.startswith("G"):                        return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU(H)"):   return "HOTEL"
    if z.startswith("OU"):                       return "OTHER"
    if z.startswith("I"):                        return "INDUSTRIAL"
    return "MIXED"


# Building tags to highlight per site type
BUILDING_TAGS = {
    "RESIDENTIAL":   {"residential": ["apartments", "house", "residential",
                                      "detached", "semidetached_house", "terrace"]},
    "COMMERCIAL":    {"commercial": True, "office": True, "retail": True,
                      "shop": True},
    "HOTEL":         {"tourism": ["hotel", "motel", "hostel", "guest_house"]},
    "INSTITUTIONAL": {"amenity": ["school", "college", "university",
                                  "hospital", "clinic", "government"]},
    "INDUSTRIAL":    {"industrial": True, "man_made": ["works", "warehouse"]},
    "OTHER":         {},
    "MIXED":         {},
}

# OSM amenity/leisure labels to pull per site type
LABEL_RULES = {
    "RESIDENTIAL":   {"amenity": ["school", "college", "university"],
                      "leisure": ["park"], "place": ["neighbourhood"]},
    "COMMERCIAL":    {"amenity": ["bank", "restaurant", "market"],
                      "railway": ["station"]},
    "INSTITUTIONAL": {"amenity": ["school", "college", "hospital"],
                      "leisure": ["park"]},
    "HOTEL":         {"amenity": ["restaurant", "hotel"],
                      "tourism": ["hotel", "attraction"]},
    "INDUSTRIAL":    {"amenity": ["factory"], "man_made": True},
    "OTHER":         {"amenity": True, "leisure": True},
    "MIXED":         {"amenity": True, "leisure": True},
}


def _safe_fetch(lat, lon, dist, tags, timeout=20):
    old = ox.settings.requests_timeout
    try:
        ox.settings.requests_timeout = timeout
        gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags)
        ox.settings.requests_timeout = old
        if gdf is not None and not gdf.empty:
            result = gdf.to_crs(3857)
            del gdf
            gc.collect()
            return result
    except Exception as e:
        ox.settings.requests_timeout = old
        log.debug(f"[context] fetch failed {tags}: {e}")
    return gpd.GeoDataFrame(geometry=[], crs=3857)


def _col(gdf, col):
    if col in gdf.columns:
        return gdf[col]
    return pd.Series([None] * len(gdf), index=gdf.index)


def _filter(gdf, col, val):
    if gdf.empty or col not in gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(gdf, "crs", 3857))
    s = gdf[col]
    mask = s.isin(val) if isinstance(val, list) else (s == val)
    return gdf[mask].copy()


def _get_name(gdf):
    ne = _col(gdf, "name:en")
    n  = _col(gdf, "name")
    return ne.fillna(n)


def _draw_mtr_station(ax, x, y, zoom=0.04, zorder=12):
    if _MTR_LOGO_LOADED and _mtr_logo is not None:
        icon = OffsetImage(_mtr_logo, zoom=zoom)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False,
                            zorder=zorder, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.plot(x, y, "o", color="#ED1D24", markersize=10,
                markeredgecolor="white", markeredgewidth=1.5,
                zorder=zorder)


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
    else:
        site_point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
        site_geom  = site_point.buffer(40)
        site_gdf   = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp     = ZONE_DATA.to_crs(3857)
    primary = ozp[ozp.contains(site_point)]
    if primary.empty:
        raise ValueError("No zoning polygon found for this site.")
    primary   = primary.iloc[0]
    zone      = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    log.info(f"[context] zone={zone} site_type={SITE_TYPE}")

    # ── Fetch base polygons (one call) ────────────────────────────────────────
    log.info("[context] Fetching base polygons...")
    polygons = _safe_fetch(lat, lon, fetch_r,
                           {"landuse": True, "leisure": True,
                            "amenity": True, "building": True})
    gc.collect()

    residential = _filter(polygons, "landuse", "residential")
    industrial  = _filter(polygons, "landuse", ["industrial", "commercial"])
    parks       = _filter(polygons, "leisure", "park")
    schools     = _filter(polygons, "amenity", ["school", "college", "university"])
    buildings   = (polygons[_col(polygons, "building").notnull()].copy()
                   if not polygons.empty else gpd.GeoDataFrame(geometry=[], crs=3857))

    # Site footprint fallback
    if lot_gdf is None and not polygons.empty:
        cands = polygons[
            polygons.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) &
            (polygons.geometry.distance(site_point) < 40)
        ]
        if len(cands):
            site_geom = (cands.assign(area=cands.area)
                         .sort_values("area", ascending=False)
                         .geometry.iloc[0])
            site_gdf  = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    del polygons
    gc.collect()

    # ── Fetch type-specific buildings ─────────────────────────────────────────
    type_buildings = gpd.GeoDataFrame(geometry=[], crs=3857)
    bt = BUILDING_TAGS.get(SITE_TYPE, {})
    if bt:
        log.info(f"[context] Fetching {SITE_TYPE} buildings...")
        type_buildings = _safe_fetch(lat, lon, fetch_r, bt)
        gc.collect()

    # ── MTR stations ──────────────────────────────────────────────────────────
    log.info("[context] Fetching MTR stations...")
    stations_raw = _safe_fetch(lat, lon, 1500, {"railway": "station"})
    gc.collect()

    stations         = gpd.GeoDataFrame(geometry=[], crs=3857)
    stations_in_view = gpd.GeoDataFrame(geometry=[], crs=3857)

    if not stations_raw.empty:
        s = stations_raw.copy()
        s["name"]     = _get_name(s)
        s["centroid"] = s.geometry.centroid
        s["dist"]     = s["centroid"].apply(lambda g: g.distance(site_point))
        s = s.dropna(subset=["name"]).sort_values("dist").head(3)
        stations = s
        stations_in_view = s[s["dist"] <= half_size * 1.2]
    del stations_raw
    gc.collect()

    # ── Walking routes to MTR (fast — graph only if stations exist) ───────────
    routes = []
    if not stations.empty:
        log.info("[context] Fetching walk graph...")
        try:
            G = ox.graph_from_point((lat, lon), dist=min(fetch_r + 500, 1500),
                                    network_type="walk",
                                    simplify=True)
            site_node = ox.distance.nearest_nodes(G, lon, lat)

            for _, st in stations.head(2).iterrows():
                try:
                    ll = gpd.GeoSeries([st["centroid"]], crs=3857).to_crs(4326).iloc[0]
                    st_node = ox.distance.nearest_nodes(G, ll.x, ll.y)
                    path    = nx.shortest_path(G, site_node, st_node, weight="length")
                    route_gdf = ox.routing.route_to_gdf(G, path).to_crs(3857)
                    routes.append(route_gdf)
                except Exception as re:
                    log.debug(f"[context] route failed: {re}")

            del G
            gc.collect()
        except Exception as e:
            log.warning(f"[context] walk graph failed: {e}")
            gc.collect()

    # ── Bus stops ─────────────────────────────────────────────────────────────
    log.info("[context] Fetching bus stops...")
    bus_stops = _safe_fetch(lat, lon, 700, {"highway": "bus_stop"})
    gc.collect()

    if len(bus_stops) > 6:
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[g.centroid.x, g.centroid.y]
                                for g in bus_stops.geometry])
            bus_stops = bus_stops.copy()
            bus_stops["cluster"] = KMeans(n_clusters=6, random_state=0).fit(coords).labels_
            bus_stops = gpd.GeoDataFrame(bus_stops.groupby("cluster").first(), crs=3857)
        except Exception as e:
            log.debug(f"[context] KMeans: {e}")
            bus_stops = bus_stops.head(6)

    # ── Place labels ──────────────────────────────────────────────────────────
    log.info("[context] Fetching labels...")
    lrules     = LABEL_RULES.get(SITE_TYPE, {"amenity": True, "leisure": True})
    labels_raw = _safe_fetch(lat, lon, fetch_r, lrules)
    gc.collect()

    all_label_items = []

    def _collect_labels(gdf):
        if gdf.empty:
            return
        g = gdf.copy()
        g["_lbl"] = _get_name(g)
        named = g.dropna(subset=["_lbl"])
        named = named[named["_lbl"].astype(str).str.strip().str.len() > 0]
        for _, row in named.iterrows():
            geom = row.geometry
            text = str(row["_lbl"]).strip()
            p    = (geom.representative_point()
                    if hasattr(geom, "representative_point") else geom.centroid)
            all_label_items.append((p.distance(site_point), geom, text))

    for src in [labels_raw, residential, parks, schools,
                type_buildings if not type_buildings.empty else None]:
        if src is not None:
            _collect_labels(src)

    all_label_items.sort(key=lambda x: x[0])
    # Deduplicate by text
    seen_texts = set()
    deduped = []
    for d, g, t in all_label_items:
        if t not in seen_texts:
            seen_texts.add(t)
            deduped.append((d, g, t))
    all_label_items = [(g, t) for _, g, t in deduped[:40]]

    del labels_raw
    gc.collect()

    log.info("[context] Rendering...")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16.15, 12))
    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       zoom=16, alpha=0.95)
    except Exception as e:
        log.warning(f"[context] basemap failed: {e}")

    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # ── Base layers ───────────────────────────────────────────────────────────
    for gdf, color, alpha in [
        (residential,   "#f2c6a0", 0.75),
        (industrial,    "#b39ddb", 0.75),
        (parks,         "#b7dfb9", 0.90),
        (schools,       "#9ecae1", 0.90),
        (buildings,     "#d9d9d9", 0.35),
    ]:
        if not gdf.empty:
            try: gdf.plot(ax=ax, color=color, alpha=alpha)
            except Exception: pass

    # ── Type-specific buildings (highlighted) ─────────────────────────────────
    type_color = {
        "RESIDENTIAL":   "#e07b39",
        "COMMERCIAL":    "#6a3d9a",
        "HOTEL":         "#b15928",
        "INSTITUTIONAL": "#1f78b4",
        "INDUSTRIAL":    "#b2df8a",
    }.get(SITE_TYPE, "#aaaaaa")

    if not type_buildings.empty:
        try:
            type_buildings.plot(ax=ax, color=type_color, alpha=0.65, zorder=3)
        except Exception:
            pass

    del residential, industrial, parks, schools, buildings, type_buildings
    gc.collect()

    # ── Walking routes ────────────────────────────────────────────────────────
    for r in routes:
        try:
            r.plot(ax=ax, color=WALK_ROUTE_COLOR, linewidth=2.2,
                   linestyle="--", zorder=6)
        except Exception:
            pass

    # ── Bus stops ─────────────────────────────────────────────────────────────
    if not bus_stops.empty:
        try:
            if _bus_icon is not None:
                for _, row in bus_stops.iterrows():
                    g  = row.geometry
                    bx = g.centroid.x if hasattr(g, "centroid") else g.x
                    by = g.centroid.y if hasattr(g, "centroid") else g.y
                    icon = OffsetImage(_bus_icon, zoom=0.025)
                    icon.image.axes = ax
                    ab = AnnotationBbox(icon, (bx, by), frameon=False,
                                        zorder=9, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
            else:
                bus_stops.plot(ax=ax, color="#0d47a1", markersize=35, zorder=9)
        except Exception as e:
            log.debug(f"[context] bus stop render: {e}")
    del bus_stops
    gc.collect()

    # ── MTR station footprint + icon ──────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            stations_in_view.plot(ax=ax, facecolor=MTR_COLOR,
                                  edgecolor="none", alpha=0.9, zorder=10)
            for _, st in stations_in_view.iterrows():
                c = st.geometry.centroid
                _draw_mtr_station(ax, c.x, c.y, zoom=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # ── Site ──────────────────────────────────────────────────────────────────
    try:
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred",
                      linewidth=2, zorder=11)
        sc = site_geom.centroid
        ax.text(sc.x, sc.y, "SITE",
                color="white", weight="bold", fontsize=9,
                ha="center", va="center", zorder=13)
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    # ── Place labels ──────────────────────────────────────────────────────────
    placed = []
    offsets = [(0, 35), (0, -35), (35, 0), (-35, 0), (25, 25), (-25, 25)]
    for i, (geom, text) in enumerate(all_label_items):
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_point) < 100:
                continue
            if any(p.distance(pp) < 100 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x + dx, p.y + dy, wrap_label(text, 18),
                    fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=12, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # ── MTR station name labels ───────────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            for _, st in stations_in_view.iterrows():
                raw = st.get("name")
                if not raw or not isinstance(raw, str) or not raw.strip():
                    continue
                c = st.geometry.centroid
                ax.text(c.x, c.y + 130, wrap_label(str(raw).strip(), 18),
                        fontsize=9, weight="bold", color="black",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.8, pad=1.0),
                        zorder=15)
        except Exception as e:
            log.debug(f"[context] station labels: {e}")

    # ── Info box ──────────────────────────────────────────────────────────────
    ax.text(
        0.015, 0.985,
        f"{data_type}: {value}\n"
        f"OZP Plan: {primary['PLAN_NO']}\n"
        f"Zoning: {zone}\n"
        f"Site Type: {SITE_TYPE}\n",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9.2,
        bbox=dict(facecolor="white", edgecolor="black", pad=6),
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    type_label = {
        "RESIDENTIAL":   "Residential Buildings",
        "COMMERCIAL":    "Commercial / Office Buildings",
        "HOTEL":         "Hotel Buildings",
        "INSTITUTIONAL": "Institutional Buildings",
        "INDUSTRIAL":    "Industrial Buildings",
    }.get(SITE_TYPE)

    handles = [
        mpatches.Patch(color="#f2c6a0", label="Residential Area"),
        mpatches.Patch(color="#b39ddb", label="Industrial / Commercial Area"),
        mpatches.Patch(color="#b7dfb9", label="Public Park"),
        mpatches.Patch(color="#9ecae1", label="School / Institution"),
    ]
    if type_label:
        handles.append(mpatches.Patch(color=type_color, alpha=0.65,
                                      label=type_label))
    handles += [
        mpatches.Patch(color=MTR_COLOR,  label="MTR Station"),
        mpatches.Patch(color="#e53935",  label="Site"),
        mlines.Line2D([], [], color=WALK_ROUTE_COLOR, linewidth=2,
                      linestyle="--", label="Pedestrian Route to MTR"),
        mpatches.Patch(color="#0d47a1",  label="Bus Stop"),
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
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("[context] Done.")
    return buf
