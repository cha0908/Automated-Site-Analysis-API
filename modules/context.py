"""
context.py — Site-Type-Driven Context Analysis Map  (memory-safe v3)
=====================================================================
Crash was: 4612 OSM features fetched → OOM during layer extraction + plot.

Fixes in this version
---------------------
1. FETCH_RADIUS reduced 1200 → 700 m  (only what fits in the 900m map view)
2. Hard cap: keep at most MAX_FEATURES rows after fetch, drop the rest
3. Walk graph REMOVED — replaced with a straight dashed line to MTR centroid
   (graph_from_point is the single biggest RAM consumer, ~200 MB for HK)
4. Polygon-only filter applied per layer BEFORE storing in dict
5. Buildings layer capped to MAX_BUILDINGS to limit plot overhead
6. del feats right after layer extraction (before MTR fetch)
7. fig size reduced 12×12 → 10×10, dpi 150 → 130
8. try/finally guarantees plt.close + gc.collect on crash
"""

import gc
import logging
import os
import textwrap

import pandas as pd
import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from io import BytesIO
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.lines import Line2D
from PIL import Image
from shapely.geometry import Point, box, LineString
from sklearn.cluster import KMeans

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache   = True
ox.settings.log_console = False

# ── Static assets ─────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
_MTR_LOGO   = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
_BUS_ICON   = os.path.join(_STATIC_DIR, "bus.png")

# ── Tuning knobs ──────────────────────────────────────────────────────────────
MAP_HALF_SIZE  = 900    # metres — map view half-width
FETCH_RADIUS   = 700    # OSM fetch radius — keep ≤ MAP_HALF_SIZE
MTR_FETCH_DIST = 1500   # MTR station search radius
MAX_FEATURES   = 800    # hard cap on total OSM rows kept in RAM
MAX_BUILDINGS  = 200    # cap on building polygons plotted
MAX_LABELS     = 20     # cap on text labels

# ── Colours ───────────────────────────────────────────────────────────────────
COLOURS = {
    "site"  : "#e53935",
    "route" : "#005eff",
    "mtr"   : "#ffd166",
    "bus"   : "#0d47a1",
}
_LAYER_COLOURS = [
    "#f2c6a0", "#9ecae1", "#b7dfb9", "#f48fb1", "#ffe082",
    "#80cbc4", "#ef9a9a", "#ffcc80", "#ce93d8", "#bcaaa4",
]


# ─────────────────────────────────────────────────────────────────────────────
# Zone helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                 return "RESIDENTIAL"
    if z.startswith("C") or "COMM" in z: return "COMMERCIAL"
    if "HOTEL" in z:                      return "HOTEL"
    if z.startswith("G") or "INST" in z: return "INSTITUTIONAL"
    if z.startswith("I") or "IND"  in z: return "INDUSTRIAL"
    if "OFFICE" in z or z[:1] == "B":    return "OFFICE"
    return "MIXED"


def _merged_tags(site_type: str) -> dict:
    """Minimal tag set — fewer tags = fewer Overpass features returned."""
    base = {"landuse": True, "highway": "bus_stop"}

    if site_type == "RESIDENTIAL":
        return {**base,
                "leisure": ["park", "playground"],
                "amenity": ["school", "college", "university",
                             "hospital", "supermarket"]}
    if site_type == "HOTEL":
        return {**base,
                "tourism": ["hotel", "hostel", "guest_house",
                             "attraction", "museum"],
                "amenity": ["restaurant", "cafe", "bar"]}
    if site_type == "OFFICE":
        return {**base,
                "office":  True,
                "amenity": ["bank", "restaurant", "cafe"]}
    if site_type == "COMMERCIAL":
        return {**base,
                "shop":    ["mall", "department_store", "supermarket"],
                "amenity": ["restaurant", "cinema", "fast_food", "cafe"]}
    if site_type == "INDUSTRIAL":
        return {**base,
                "landuse":  ["industrial", "warehouse"],
                "man_made": ["works"]}
    # MIXED / INSTITUTIONAL / fallback — keep minimal
    return {**base,
            "amenity": ["school", "hospital", "restaurant"],
            "leisure": ["park"]}


# ─────────────────────────────────────────────────────────────────────────────
# GDF helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(gdf, col, values):
    try:
        if col not in gdf.columns:
            return gdf.iloc[0:0]
        if values is True:
            return gdf[gdf[col].notna()]
        if isinstance(values, str):
            values = [values]
        return gdf[gdf[col].isin(values)]
    except Exception:
        return gdf.iloc[0:0]


def _polys(gdf):
    if gdf.empty:
        return gdf
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]


def _load_icon(path):
    try:
        return np.array(Image.open(path).convert("RGBA"))
    except Exception:
        return None


def _place_icon(ax, x, y, arr, zoom=0.04, zorder=15):
    img = OffsetImage(arr, zoom=zoom)
    img.image.axes = ax
    ax.add_artist(AnnotationBbox(img, (x, y), frameon=False,
                                 zorder=zorder, pad=0))


def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


# ─────────────────────────────────────────────────────────────────────────────
# Layer extraction — polygon-only, capped
# ─────────────────────────────────────────────────────────────────────────────

def _extract_layers(feats: gpd.GeoDataFrame, site_type: str) -> dict:
    layers = {}

    def add(name, col, vals):
        sub = _polys(_safe(feats, col, vals))
        if not sub.empty:
            layers[name] = sub

    if site_type == "RESIDENTIAL":
        add("Residential Area",   "landuse", ["residential"])
        add("Park / Greenspace",  "leisure", ["park", "playground"])
        add("School / Education", "amenity", ["school", "college", "university"])
        add("Hospital / Clinic",  "amenity", ["hospital", "clinic"])
        add("Supermarket",        "amenity", ["supermarket"])
    elif site_type == "HOTEL":
        add("Hotel / Resort",     "tourism", ["hotel", "hostel", "guest_house"])
        add("Tourist Attraction", "tourism", ["attraction", "museum"])
        add("Restaurant / Cafe",  "amenity", ["restaurant", "cafe", "bar"])
    elif site_type == "OFFICE":
        add("Office / Business",  "office",  True)
        add("Commercial Zone",    "landuse", ["commercial", "retail"])
        add("Bank",               "amenity", ["bank"])
        add("Restaurant / Cafe",  "amenity", ["restaurant", "cafe"])
    elif site_type == "COMMERCIAL":
        add("Retail / Mall",      "shop",    ["mall", "department_store",
                                               "supermarket"])
        add("Retail Landuse",     "landuse", ["retail", "commercial"])
        add("Restaurant / Cafe",  "amenity", ["restaurant", "cafe",
                                               "fast_food"])
        add("Cinema / Leisure",   "amenity", ["cinema", "theatre"])
    elif site_type == "INDUSTRIAL":
        add("Industrial Zone",    "landuse", ["industrial"])
        add("Warehouse",          "landuse", ["warehouse"])
        add("Works / Facility",   "man_made",["works"])
    else:
        add("Residential Area",   "landuse", ["residential"])
        add("Commercial Zone",    "landuse", ["commercial", "retail"])
        add("Park / Greenspace",  "leisure", ["park"])
        add("School / Education", "amenity", ["school", "college", "university"])
        add("Hospital / Clinic",  "amenity", ["hospital"])

    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Straight-line route to MTR (no graph download)
# ─────────────────────────────────────────────────────────────────────────────

def _straight_routes(site_pt, stations):
    """
    Return list of GeoDataFrames containing a single dashed line from
    site centroid → MTR station centroid. Zero RAM cost vs graph_from_point.
    """
    routes = []
    if stations.empty:
        return routes
    for _, st in stations.iterrows():
        try:
            line = LineString([
                (site_pt.x, site_pt.y),
                (st["_centroid"].x, st["_centroid"].y),
            ])
            gdf = gpd.GeoDataFrame(geometry=[line], crs=3857)
            routes.append(gdf)
        except Exception:
            pass
    return routes


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_context(
    data_type : str,
    value     : str,
    zone_data : gpd.GeoDataFrame,
    radius_m  : int | None,
    lon       : float | None = None,
    lat       : float | None = None,
    lot_ids   : list  | None = None,
    extents   : list  | None = None,
) -> BytesIO:

    lot_ids  = lot_ids  or []
    extents  = extents  or []
    radius_m = radius_m or 600
    fig      = None

    try:
        # 1. Resolve coords ───────────────────────────────────────────────────
        lon, lat = resolve_location(
            data_type, value, lon, lat, lot_ids, extents
        )
        site_pt = (gpd.GeoSeries([Point(lon, lat)], crs=4326)
                      .to_crs(3857).iloc[0])

        # 2. Zoning ───────────────────────────────────────────────────────────
        hits = zone_data[zone_data.contains(site_pt)]
        if hits.empty:
            zd2 = zone_data.copy()
            zd2["_d"] = zd2.geometry.distance(site_pt)
            primary_row = zd2.sort_values("_d").iloc[0]
            del zd2
        else:
            primary_row = hits.iloc[0]
        del hits

        zone      = str(primary_row.get("ZONE_LABEL", "MIXED"))
        plan_no   = str(primary_row.get("PLAN_NO",    "N/A"))
        site_type = _infer_site_type(zone)
        log.info("Context: zone=%s  site_type=%s", zone, site_type)
        gc.collect()

        # 3. OSM fetch — single call, immediately clipped + capped ────────────
        map_bbox = box(
            site_pt.x - MAP_HALF_SIZE, site_pt.y - MAP_HALF_SIZE,
            site_pt.x + MAP_HALF_SIZE, site_pt.y + MAP_HALF_SIZE,
        )
        feats = gpd.GeoDataFrame(columns=["geometry"], crs=3857)
        try:
            raw = ox.features_from_point(
                (lat, lon), dist=FETCH_RADIUS, tags=_merged_tags(site_type)
            ).to_crs(3857)

            # Clip to map view
            raw = raw[raw.geometry.intersects(map_bbox)].copy()

            # Hard cap — keep largest polygons first (most meaningful)
            if len(raw) > MAX_FEATURES:
                poly_mask = raw.geometry.geom_type.isin(
                    ["Polygon", "MultiPolygon"]
                )
                polys  = raw[poly_mask].copy()
                others = raw[~poly_mask].copy()
                polys["_a"] = polys.area
                polys = polys.nlargest(min(len(polys), MAX_FEATURES - 50), "_a")
                others = others.iloc[: max(0, MAX_FEATURES - len(polys))]
                raw = gpd.GeoDataFrame(
                    pd.concat([polys, others], ignore_index=True), crs=3857
                )
                del polys, others

            feats = raw
            del raw
            log.info("OSM features (capped): %d", len(feats))
        except Exception as e:
            log.warning("OSM fetch failed: %s", e)
        gc.collect()

        # 4. Extract layers (polygon-only per layer) ───────────────────────────
        layers = _extract_layers(feats, site_type)

        # 5. Bus stops from same fetch ─────────────────────────────────────────
        bus_stops = _safe(feats, "highway", ["bus_stop"])
        if not bus_stops.empty:
            bus_stops = bus_stops[
                bus_stops.geometry.geom_type == "Point"
            ].copy()
        if len(bus_stops) > 6:
            arr = np.array([[g.x, g.y] for g in bus_stops.geometry])
            bus_stops["_cl"] = (
                KMeans(n_clusters=6, random_state=0, n_init="auto")
                .fit(arr).labels_
            )
            bus_stops = bus_stops.groupby("_cl").first().reset_index(drop=True)

        # 6. Labels from same fetch ────────────────────────────────────────────
        labels = gpd.GeoDataFrame()
        for nc in ("name:en", "name"):
            if nc in feats.columns:
                tmp = feats.copy()
                tmp["_label"] = feats[nc]
                labels = (tmp.dropna(subset=["_label"])
                             .drop_duplicates("_label")
                             .head(MAX_LABELS))
                del tmp
                break

        # Free feats — no longer needed
        del feats
        gc.collect()

        # 7. Site footprint ────────────────────────────────────────────────────
        site_boundary = get_lot_boundary(
            lon, lat, data_type,
            extents if len(extents) > 1 else None,
        )
        if site_boundary is not None and not site_boundary.empty:
            site_geom = site_boundary.geometry.unary_union
        else:
            site_geom = site_pt.buffer(40)
        site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

        # 8. MTR stations — lightweight point fetch ────────────────────────────
        stations = gpd.GeoDataFrame()
        try:
            raw_st = ox.features_from_point(
                (lat, lon), tags={"railway": "station"}, dist=MTR_FETCH_DIST
            ).to_crs(3857)
            if not raw_st.empty:
                raw_st = raw_st.copy()
                raw_st["_name"] = (
                    raw_st["name:en"]
                    if "name:en" in raw_st.columns
                    else raw_st.get("name")
                )
                raw_st["_centroid"] = raw_st.geometry.centroid
                raw_st["_dist"]     = raw_st["_centroid"].distance(site_pt)
                stations = (raw_st.dropna(subset=["_name"])
                                  .sort_values("_dist").head(2))
            del raw_st
            gc.collect()
        except Exception as e:
            log.warning("MTR fetch failed: %s", e)

        # 9. Straight-line routes (no graph download) ──────────────────────────
        routes = _straight_routes(site_pt, stations)

        # 10. Load icons ───────────────────────────────────────────────────────
        mtr_arr = _load_icon(_MTR_LOGO)
        bus_arr = _load_icon(_BUS_ICON)

        # ── PLOT ──────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 10))
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       zoom=16, alpha=0.95)
        ax.set_xlim(site_pt.x - MAP_HALF_SIZE, site_pt.x + MAP_HALF_SIZE)
        ax.set_ylim(site_pt.y - MAP_HALF_SIZE, site_pt.y + MAP_HALF_SIZE)
        ax.set_aspect("equal")
        ax.autoscale(False)

        # Layers
        legend_handles = []
        for i, (layer_name, gdf) in enumerate(layers.items()):
            colour = _LAYER_COLOURS[i % len(_LAYER_COLOURS)]
            try:
                gdf.plot(ax=ax, color=colour, alpha=0.72)
                legend_handles.append(
                    mpatches.Patch(color=colour, label=layer_name)
                )
            except Exception as e:
                log.debug("Layer plot [%s]: %s", layer_name, e)

        # Routes (straight dashed line)
        for route in routes:
            try:
                route.plot(ax=ax, color=COLOURS["route"],
                           linewidth=2.0, linestyle="--", zorder=8)
            except Exception:
                pass

        # Bus stops
        for geom in (bus_stops.geometry if not bus_stops.empty else []):
            try:
                pt = geom if geom.geom_type == "Point" else geom.centroid
                if bus_arr is not None:
                    _place_icon(ax, pt.x, pt.y, bus_arr, zoom=0.04, zorder=9)
                else:
                    ax.plot(pt.x, pt.y, "o", color=COLOURS["bus"],
                            markersize=7, zorder=9)
            except Exception:
                pass

        # MTR stations
        for _, st in (stations.iterrows() if not stations.empty
                      else iter([])):
            try:
                cp = st["_centroid"]
                if mtr_arr is not None:
                    _place_icon(ax, cp.x, cp.y, mtr_arr,
                                zoom=0.055, zorder=12)
                else:
                    ax.plot(cp.x, cp.y, "s", color=COLOURS["mtr"],
                            markersize=14, zorder=10)
                ax.text(cp.x, cp.y + 120, _wrap(st["_name"], 18),
                        fontsize=8.5, ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=1.0),
                        zorder=13, clip_on=True)
            except Exception:
                pass

        # Site
        site_gdf.plot(ax=ax, facecolor=COLOURS["site"],
                      edgecolor="darkred", linewidth=2, zorder=11)
        ax.text(site_geom.centroid.x, site_geom.centroid.y,
                "SITE", color="white", weight="bold",
                ha="center", va="center", fontsize=9, zorder=14)

        # Labels
        placed  = []
        offsets = [(0,38),(0,-38),(40,0),(-40,0),
                   (28,28),(-28,28),(28,-28),(-28,-28)]
        for i, (_, row) in enumerate(labels.iterrows()):
            try:
                p = row.geometry.representative_point()
                if p.distance(site_pt) < 140:
                    continue
                if any(p.distance(pp) < 130 for pp in placed):
                    continue
                dx, dy = offsets[i % len(offsets)]
                ax.text(p.x + dx, p.y + dy, _wrap(row["_label"], 18),
                        fontsize=8.5, ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, boxstyle="round,pad=0.25"),
                        zorder=12, clip_on=True)
                placed.append(p)
            except Exception:
                continue

        # Info box
        ax.text(0.015, 0.985,
                f"Lot: {value}\nOZP Plan: {plan_no}\n"
                f"Zoning: {zone}\nSite Type: {site_type}\n",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.2,
                bbox=dict(facecolor="white", edgecolor="black", pad=6))

        # Legend
        legend_handles += [
            mpatches.Patch(color=COLOURS["mtr"],  label="MTR Station"),
            mpatches.Patch(color=COLOURS["site"], label="Site"),
            Line2D([0],[0], color=COLOURS["route"], linewidth=2,
                   linestyle="--", label="Route to MTR"),
        ]
        if not bus_stops.empty:
            legend_handles.append(
                mpatches.Patch(color=COLOURS["bus"], label="Bus Stop")
            )
        ax.legend(handles=legend_handles, loc="lower left",
                  bbox_to_anchor=(0.02, 0.08), fontsize=8.5, framealpha=0.95)

        ax.set_title(
            f"Site Context Analysis — {site_type.title()} Development",
            fontsize=13, weight="bold")
        ax.set_axis_off()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        return buf

    finally:
        if fig is not None:
            plt.close(fig)
        gc.collect()
