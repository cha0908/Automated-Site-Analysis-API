"""
context.py — Site-Type-Driven Context Analysis Map  (v4 — HK-density safe)
===========================================================================
Previous crash: 3432 features even at 700m in Hong Kong.
Root cause:  {"building": True, "landuse": True} in HK returns thousands of
             tiny building footprints.  We do NOT need individual buildings
             for a context map — only zone polygons + key amenities matter.

Key changes in v4
-----------------
* NO building=True fetch at all  — biggest single source of feature bloat
* NO landuse=True (wildcard)     — replaced with explicit value lists only
* Tags are now site-type specific AND minimal (5-8 values max)
* Two separate tiny fetches:
    A) landuse polygons  (low count in HK — only large zone polygons)
    B) amenity/leisure points  (points only — negligible RAM)
* Bus stops fetched separately with a hard limit of 200 m radius
* Straight-line MTR route kept (no graph download)
* Hard row cap = 300 after each fetch
* gc.collect() after every allocation
* fig 10×10 dpi 120
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
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
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

# ── Tuning ────────────────────────────────────────────────────────────────────
MAP_HALF    = 900    # metres — map view
ZONE_RADIUS = 800    # landuse polygon fetch radius
POI_RADIUS  = 600    # amenity/leisure point fetch radius
BUS_RADIUS  = 400    # bus stop fetch radius
MTR_RADIUS  = 1500   # MTR station fetch radius
MAX_ROWS    = 300    # hard row cap per fetch result
MAX_LABELS  = 18     # max text labels on map

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


def _landuse_tags(site_type: str) -> dict:
    """Only explicit landuse VALUES — never True wildcard."""
    if site_type == "RESIDENTIAL":
        return {"landuse": ["residential", "recreation_ground"],
                "leisure": ["park", "playground"]}
    if site_type in ("COMMERCIAL", "HOTEL", "OFFICE"):
        return {"landuse": ["commercial", "retail", "mixed_use"]}
    if site_type == "INDUSTRIAL":
        return {"landuse": ["industrial", "warehouse"]}
    # MIXED / INSTITUTIONAL
    return {"landuse": ["residential", "commercial", "retail"],
            "leisure": ["park"]}


def _poi_tags(site_type: str) -> dict:
    """Amenity/tourism points only — never building=True."""
    if site_type == "RESIDENTIAL":
        return {"amenity": ["school", "college", "university",
                             "hospital", "supermarket"]}
    if site_type == "HOTEL":
        return {"tourism": ["hotel", "hostel", "attraction", "museum"],
                "amenity": ["restaurant", "cafe"]}
    if site_type == "OFFICE":
        return {"amenity": ["bank", "restaurant", "cafe"]}
    if site_type == "COMMERCIAL":
        return {"amenity": ["restaurant", "cinema", "fast_food", "cafe"],
                "shop":    ["mall", "department_store"]}
    if site_type == "INDUSTRIAL":
        return {"man_made": ["works"]}
    return {"amenity": ["school", "hospital", "restaurant"],
            "leisure": ["park"]}


# ─────────────────────────────────────────────────────────────────────────────
# Fetch helpers — each returns a small, capped GeoDataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(lat, lon, radius, tags, clip_box, polygons_only=False) -> gpd.GeoDataFrame:
    """Safe fetch with clip + row cap. Returns empty GDF on any error."""
    empty = gpd.GeoDataFrame(columns=["geometry"], crs=3857)
    try:
        gdf = ox.features_from_point(
            (lat, lon), dist=radius, tags=tags
        ).to_crs(3857)

        # Clip to map view
        gdf = gdf[gdf.geometry.intersects(clip_box)].copy()

        if polygons_only:
            gdf = gdf[gdf.geometry.geom_type.isin(
                ["Polygon", "MultiPolygon"]
            )].copy()

        # Cap rows — keep largest polygons if poly-only, else head
        if len(gdf) > MAX_ROWS:
            if polygons_only:
                gdf["_a"] = gdf.area
                gdf = gdf.nlargest(MAX_ROWS, "_a").drop(columns=["_a"])
            else:
                gdf = gdf.head(MAX_ROWS)

        log.info("fetch(%dm, %s): %d rows", radius,
                 list(tags.keys())[:2], len(gdf))
        return gdf
    except Exception as e:
        log.warning("fetch failed: %s", e)
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# Layer name → colour mapping
# ─────────────────────────────────────────────────────────────────────────────

_LAYER_DEFS = {
    "RESIDENTIAL": [
        ("Residential Area",   "landuse", ["residential", "recreation_ground"]),
        ("Park / Greenspace",  "leisure", ["park", "playground"]),
        ("School / Education", "amenity", ["school", "college", "university"]),
        ("Hospital / Clinic",  "amenity", ["hospital"]),
        ("Supermarket",        "amenity", ["supermarket"]),
    ],
    "HOTEL": [
        ("Hotel / Resort",     "tourism", ["hotel", "hostel", "guest_house"]),
        ("Tourist Attraction", "tourism", ["attraction", "museum"]),
        ("Restaurant / Cafe",  "amenity", ["restaurant", "cafe"]),
        ("Commercial Zone",    "landuse", ["commercial", "retail"]),
    ],
    "OFFICE": [
        ("Office / Business",  "office",  None),   # None = notna()
        ("Commercial Zone",    "landuse", ["commercial", "retail"]),
        ("Bank",               "amenity", ["bank"]),
        ("Restaurant / Cafe",  "amenity", ["restaurant", "cafe"]),
    ],
    "COMMERCIAL": [
        ("Retail / Mall",      "shop",    ["mall", "department_store"]),
        ("Retail Landuse",     "landuse", ["retail", "commercial"]),
        ("Restaurant / Cafe",  "amenity", ["restaurant", "cafe", "fast_food"]),
        ("Cinema / Leisure",   "amenity", ["cinema", "theatre"]),
    ],
    "INDUSTRIAL": [
        ("Industrial Zone",    "landuse", ["industrial"]),
        ("Warehouse",          "landuse", ["warehouse"]),
        ("Works / Facility",   "man_made",["works"]),
    ],
}
_LAYER_DEFS["MIXED"]         = _LAYER_DEFS["RESIDENTIAL"][:3]
_LAYER_DEFS["INSTITUTIONAL"] = _LAYER_DEFS["RESIDENTIAL"][2:4]


def _extract_layers(feats: gpd.GeoDataFrame, site_type: str) -> list:
    """Return [(name, colour, GeoDataFrame), …] — polygons only, non-empty."""
    defs   = _LAYER_DEFS.get(site_type, _LAYER_DEFS["MIXED"])
    result = []
    for i, (name, col, vals) in enumerate(defs):
        try:
            if col not in feats.columns:
                continue
            if vals is None:
                sub = feats[feats[col].notna()]
            else:
                sub = feats[feats[col].isin(vals)]
            sub = sub[sub.geometry.geom_type.isin(
                ["Polygon", "MultiPolygon"]
            )].copy()
            if not sub.empty:
                result.append((name, _LAYER_COLOURS[i % len(_LAYER_COLOURS)], sub))
        except Exception:
            continue
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _straight_routes(site_pt, stations):
    routes = []
    for _, st in (stations.iterrows() if not stations.empty else []):
        try:
            line = LineString([(site_pt.x, site_pt.y),
                               (st["_centroid"].x, st["_centroid"].y)])
            routes.append(gpd.GeoDataFrame(geometry=[line], crs=3857))
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
    fig      = None

    try:
        # 1. Resolve ──────────────────────────────────────────────────────────
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

        map_bbox = box(
            site_pt.x - MAP_HALF, site_pt.y - MAP_HALF,
            site_pt.x + MAP_HALF, site_pt.y + MAP_HALF,
        )

        # 3a. Landuse polygons (small count) ──────────────────────────────────
        landuse_gdf = _fetch(lat, lon, ZONE_RADIUS,
                             _landuse_tags(site_type), map_bbox,
                             polygons_only=True)
        gc.collect()

        # 3b. POI points (small count) ────────────────────────────────────────
        poi_gdf = _fetch(lat, lon, POI_RADIUS,
                         _poi_tags(site_type), map_bbox,
                         polygons_only=False)
        gc.collect()

        # 3c. Bus stops (very small radius) ───────────────────────────────────
        bus_gdf = _fetch(lat, lon, BUS_RADIUS,
                         {"highway": "bus_stop"}, map_bbox,
                         polygons_only=False)
        bus_stops = bus_gdf[bus_gdf.geometry.geom_type == "Point"].copy()
        del bus_gdf
        if len(bus_stops) > 6:
            arr = np.array([[g.x, g.y] for g in bus_stops.geometry])
            bus_stops["_cl"] = (
                KMeans(n_clusters=6, random_state=0, n_init="auto")
                .fit(arr).labels_
            )
            bus_stops = bus_stops.groupby("_cl").first().reset_index(drop=True)
        gc.collect()

        # 4. Merge landuse + poi for layer extraction ─────────────────────────
        try:
            combined = gpd.GeoDataFrame(
                pd.concat([landuse_gdf, poi_gdf], ignore_index=True),
                crs=3857
            )
        except Exception:
            combined = landuse_gdf if not landuse_gdf.empty else poi_gdf
        del landuse_gdf, poi_gdf
        gc.collect()

        layers = _extract_layers(combined, site_type)

        # Labels from combined (name:en preferred)
        labels = gpd.GeoDataFrame()
        for nc in ("name:en", "name"):
            if nc in combined.columns:
                tmp = combined.copy()
                tmp["_label"] = combined[nc]
                labels = (tmp.dropna(subset=["_label"])
                             .drop_duplicates("_label")
                             .head(MAX_LABELS))
                del tmp
                break

        del combined
        gc.collect()

        # 5. Site footprint ────────────────────────────────────────────────────
        site_boundary = get_lot_boundary(
            lon, lat, data_type,
            extents if len(extents) > 1 else None,
        )
        site_geom = (
            site_boundary.geometry.unary_union
            if (site_boundary is not None and not site_boundary.empty)
            else site_pt.buffer(40)
        )
        site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

        # 6. MTR stations ─────────────────────────────────────────────────────
        stations = gpd.GeoDataFrame()
        try:
            raw_st = ox.features_from_point(
                (lat, lon), tags={"railway": "station"}, dist=MTR_RADIUS
            ).to_crs(3857)
            if not raw_st.empty:
                raw_st = raw_st.copy()
                raw_st["_name"] = (
                    raw_st["name:en"] if "name:en" in raw_st.columns
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

        routes = _straight_routes(site_pt, stations)

        # 7. Icons ────────────────────────────────────────────────────────────
        mtr_arr = _load_icon(_MTR_LOGO)
        bus_arr = _load_icon(_BUS_ICON)

        # ── PLOT ──────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 10))
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       zoom=16, alpha=0.95)
        ax.set_xlim(site_pt.x - MAP_HALF, site_pt.x + MAP_HALF)
        ax.set_ylim(site_pt.y - MAP_HALF, site_pt.y + MAP_HALF)
        ax.set_aspect("equal")
        ax.autoscale(False)

        # Layers
        legend_handles = []
        for name, colour, gdf in layers:
            try:
                gdf.plot(ax=ax, color=colour, alpha=0.72)
                legend_handles.append(
                    mpatches.Patch(color=colour, label=name)
                )
            except Exception as e:
                log.debug("Layer plot [%s]: %s", name, e)

        # Routes
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

        # MTR
        for _, st in (stations.iterrows() if not stations.empty else []):
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
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        return buf

    finally:
        if fig is not None:
            plt.close(fig)
        gc.collect()
