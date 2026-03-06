import logging
import os
import gc
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
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
from modules.driving import _add_mtr_icon

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 25

log = logging.getLogger(__name__)

FETCH_RADIUS             = 800
MAP_HALF_SIZE            = 600
NEAREST_NAMED_STATION_M  = 150
MTR_COLOR                = "#d84315"

_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
try:
    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
except Exception:
    _bus_icon = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def wrap_label(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def infer_site_type(zone):
    if zone.startswith("R"):            return "RESIDENTIAL"
    if zone.startswith("C"):            return "COMMERCIAL"
    if zone.startswith("G"):            return "INSTITUTIONAL"
    if "HOTEL" in zone.upper() or zone.startswith("OU"): return "HOTEL"
    return "MIXED"


def context_rules(site_type):
    if site_type == "RESIDENTIAL":
        return {"amenity": ["school", "college", "university"],
                "leisure": ["park"], "place": ["neighbourhood"]}
    if site_type == "COMMERCIAL":
        return {"amenity": ["bank", "restaurant", "market"],
                "railway": ["station"]}
    if site_type == "INSTITUTIONAL":
        return {"amenity": ["school", "college", "hospital"],
                "leisure": ["park"]}
    return {"amenity": True, "leisure": True}


def _safe_fetch(lat, lon, dist, tags):
    """Fetch OSM features, return empty GDF on any error."""
    try:
        gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags)
        if gdf is not None and not gdf.empty:
            result = gdf.to_crs(3857)
            del gdf
            gc.collect()
            return result
    except Exception as e:
        log.debug(f"[context] fetch failed {tags}: {e}")
    return gpd.GeoDataFrame(geometry=[], crs=3857)


def _col(gdf, col):
    """Safely get a column, returning a Series of None if missing."""
    if col in gdf.columns:
        return gdf[col]
    return pd.Series([None] * len(gdf), index=gdf.index)


def _filter(gdf, col, val):
    """Filter gdf where column equals val (or is in list val)."""
    if gdf.empty or col not in gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs if not gdf.empty else 3857)
    s = gdf[col]
    if isinstance(val, list):
        mask = s.isin(val)
    else:
        mask = s == val
    return gdf[mask].copy()


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
    log.info(f"[context] extent half_x={half_x:.0f} half_y={half_y:.0f} r={fetch_r}m")

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
        site_geom  = None
        site_gdf   = None

    # ── Zoning ────────────────────────────────────────────────────────────────
    ozp     = ZONE_DATA.to_crs(3857)
    primary = ozp[ozp.contains(site_point)]
    if primary.empty:
        raise ValueError("No zoning polygon found for this site.")
    primary     = primary.iloc[0]
    zone        = primary["ZONE_LABEL"]
    SITE_TYPE   = infer_site_type(zone)
    LABEL_RULES = context_rules(SITE_TYPE)
    log.info(f"[context] zone={zone} site_type={SITE_TYPE}")

    # ── Fetch OSM layers one-by-one with gc ───────────────────────────────────
    log.info("[context] Fetching polygons...")
    polygons = _safe_fetch(lat, lon, fetch_r,
                           {"landuse": True, "leisure": True,
                            "amenity": True, "building": True})
    gc.collect()

    residential = _filter(polygons, "landuse", "residential")
    industrial  = _filter(polygons, "landuse", ["industrial", "commercial"])
    parks       = _filter(polygons, "leisure", "park")
    schools     = _filter(polygons, "amenity", ["school", "college", "university"])
    buildings   = polygons[_col(polygons, "building").notnull()].copy() \
                  if not polygons.empty else gpd.GeoDataFrame(geometry=[], crs=3857)

    # Site footprint fallback
    if site_gdf is None:
        if not polygons.empty:
            cands = polygons[
                polygons.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) &
                (polygons.geometry.distance(site_point) < 40)
            ]
            if len(cands):
                site_geom = (cands.assign(area=cands.area)
                             .sort_values("area", ascending=False)
                             .geometry.iloc[0])
            else:
                site_geom = site_point.buffer(40)
        else:
            site_geom = site_point.buffer(40)
        site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    del polygons
    gc.collect()

    # ── MTR stations ──────────────────────────────────────────────────────────
    log.info("[context] Fetching MTR stations...")
    stations = _safe_fetch(lat, lon, 1200, {"railway": "station"})
    gc.collect()

    stations_in_view = gpd.GeoDataFrame(geometry=[], crs=3857)
    if not stations.empty:
        stations = stations.copy()
        name_en = _col(stations, "name:en")
        name    = _col(stations, "name")
        stations["name"]     = name_en.fillna(name)
        stations["centroid"] = stations.geometry.centroid
        stations["dist"]     = stations["centroid"].apply(lambda g: g.distance(site_point))
        stations = stations.dropna(subset=["name"]).sort_values("dist").head(2)
        stations_in_view = stations[stations["dist"] <= half_size]

    # ── Bus stops ─────────────────────────────────────────────────────────────
    log.info("[context] Fetching bus stops...")
    bus_stops = _safe_fetch(lat, lon, 700, {"highway": "bus_stop"})
    gc.collect()

    if len(bus_stops) > 6:
        try:
            from sklearn.cluster import KMeans
            coords = np.array([[g.centroid.x, g.centroid.y] for g in bus_stops.geometry])
            bus_stops = bus_stops.copy()
            bus_stops["cluster"] = KMeans(n_clusters=6, random_state=0).fit(coords).labels_
            bus_stops = bus_stops.groupby("cluster").first()
            bus_stops = gpd.GeoDataFrame(bus_stops, crs=3857)
        except Exception as e:
            log.debug(f"[context] KMeans failed: {e}")
            bus_stops = bus_stops.head(6)

    # ── Place labels ──────────────────────────────────────────────────────────
    log.info("[context] Fetching labels...")
    labels_raw = _safe_fetch(lat, lon, 600, LABEL_RULES)
    gc.collect()

    if not labels_raw.empty:
        labels_raw         = labels_raw.copy()
        name_en            = _col(labels_raw, "name:en")
        name               = _col(labels_raw, "name")
        labels_raw["label"] = name_en.fillna(name)
        labels_raw          = (labels_raw.dropna(subset=["label"])
                               .drop_duplicates("label").head(24))
    else:
        labels_raw = gpd.GeoDataFrame(geometry=[], crs=3857)

    # ── Collect all label candidates ─────────────────────────────────────────
    label_sources = [labels_raw, residential, parks, schools]
    if SITE_TYPE in ("COMMERCIAL", "MIXED"):
        label_sources.append(industrial)

    all_label_items = []
    for gdf in label_sources:
        if gdf.empty:
            continue
        gdf = gdf.copy()
        ne  = _col(gdf, "name:en")
        n   = _col(gdf, "name")
        gdf["label"] = ne.fillna(n)
        named = gdf.dropna(subset=["label"])
        named = named[named["label"].astype(str).str.strip().str.len() > 0]
        for _, row in named.iterrows():
            geom = row.geometry
            text = str(row["label"]).strip()
            p    = (geom.representative_point()
                    if hasattr(geom, "representative_point") else geom.centroid)
            dist = p.distance(site_point)
            all_label_items.append((dist, geom, text))

    all_label_items.sort(key=lambda x: x[0])
    all_label_items = [(geom, text) for _, geom, text in all_label_items[:40]]

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
        cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerNoLabels,
                       zoom=16, alpha=0.95)
    except Exception as e:
        log.warning(f"[context] basemap failed: {e}")

    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # ── Coloured layers ───────────────────────────────────────────────────────
    if not residential.empty:
        try: residential.plot(ax=ax, color="#f2c6a0", alpha=0.9)
        except Exception: pass
    if not industrial.empty:
        try: industrial.plot(ax=ax, color="#b39ddb", alpha=0.9)
        except Exception: pass
    if not parks.empty:
        try: parks.plot(ax=ax, color="#b7dfb9", alpha=0.9)
        except Exception: pass
    if not schools.empty:
        try: schools.plot(ax=ax, color="#9ecae1", alpha=0.9)
        except Exception: pass
    if not buildings.empty:
        try: buildings.plot(ax=ax, color="#d9d9d9", alpha=0.35)
        except Exception: pass

    del residential, industrial, parks, schools, buildings
    gc.collect()

    # ── Bus stops ─────────────────────────────────────────────────────────────
    if not bus_stops.empty:
        try:
            if _bus_icon is not None:
                for _, row in bus_stops.iterrows():
                    geom = row.geometry
                    bx   = geom.centroid.x if hasattr(geom, "centroid") else geom.x
                    by   = geom.centroid.y if hasattr(geom, "centroid") else geom.y
                    icon = OffsetImage(_bus_icon, zoom=0.02)
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

    # ── MTR stations ──────────────────────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            stations_in_view.plot(ax=ax, facecolor=MTR_COLOR, edgecolor="none",
                                  alpha=0.9, zorder=10)
            for _, st in stations_in_view.iterrows():
                c = st.geometry.centroid
                _add_mtr_icon(ax, c.x, c.y, size=0.035, zorder=14)
        except Exception as e:
            log.debug(f"[context] station render: {e}")

    # ── Site ──────────────────────────────────────────────────────────────────
    try:
        site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred",
                      linewidth=2, zorder=11)
        ax.text(site_geom.centroid.x, site_geom.centroid.y, "SITE",
                color="white", weight="bold", ha="center", va="center", zorder=12)
    except Exception as e:
        log.warning(f"[context] site render: {e}")

    # ── Place labels ──────────────────────────────────────────────────────────
    placed = []
    for geom, text in all_label_items:
        try:
            p = (geom.representative_point()
                 if hasattr(geom, "representative_point") else geom.centroid)
            if p.distance(site_point) < 140:
                continue
            if any(p.distance(pp) < 100 for pp in placed):
                continue
            ax.text(p.x, p.y, wrap_label(text, 18),
                    fontfamily="Arial", fontsize=12, weight="bold",
                    ha="center", va="center", color="black",
                    zorder=12, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # ── MTR station name labels ───────────────────────────────────────────────
    if not stations_in_view.empty:
        try:
            named_stations = [
                (row.geometry.centroid, str(row["name"]).strip())
                for _, row in stations_in_view.iterrows()
                if row.get("name") and isinstance(row["name"], str)
                   and str(row["name"]).strip()
            ]
            for _, st in stations_in_view.iterrows():
                raw_name = st.get("name")
                if (raw_name is None
                        or (isinstance(raw_name, float) and raw_name != raw_name)
                        or not isinstance(raw_name, str)
                        or not str(raw_name).strip()):
                    label_name = "UNNAMED"
                    if named_stations:
                        cent  = st.geometry.centroid
                        label_name = min(named_stations,
                                         key=lambda p: cent.distance(p[0]))[1]
                else:
                    label_name = str(raw_name).strip()
                ax.text(st.geometry.centroid.x, st.geometry.centroid.y + 120,
                        wrap_label(label_name, 18),
                        fontsize=10, weight="bold", color="black",
                        ha="center", va="bottom", zorder=15)
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
    ax.legend(
        handles=[
            mpatches.Patch(color="#f2c6a0", label="Residential"),
            mpatches.Patch(color="#b39ddb", label="Industrial / Commercial"),
            mpatches.Patch(color="#b7dfb9", label="Public Park"),
            mpatches.Patch(color="#9ecae1", label="School / Institution"),
            mpatches.Patch(color=MTR_COLOR,  label="MTR Station"),
            mpatches.Patch(color="#e53935",  label="Site"),
            mpatches.Patch(color="#0d47a1",  label="Bus Stop"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.08),
        fontsize=8.5,
        framealpha=0.95,
    )

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
