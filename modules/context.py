"""
context.py — Site-Type-Driven Context Analysis Map  (memory-optimised)
=======================================================================
Key memory fixes vs previous version:
  - ONE ox.features_from_point call (merged tags) instead of 4-5 separate calls
  - Bus stops extracted from the same single fetch — no extra Overpass query
  - Place labels extracted from the same single fetch — no extra Overpass query
  - Immediate bbox clip after OSM fetch — drops features outside map view
  - walk graph: simplify=True + retain_all=False — smaller NetworkX graph
  - gc.collect() after every heavy allocation
  - plt.close(fig) + gc.collect() guaranteed in finally block
  - dpi reduced to 150 (was 180) — smaller output buffer
"""

import gc
import logging
import os
import textwrap

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from io import BytesIO
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from shapely.geometry import Point, box
from sklearn.cluster import KMeans

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache   = True
ox.settings.log_console = False

# ── Static asset paths ────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
_MTR_LOGO   = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
_BUS_ICON   = os.path.join(_STATIC_DIR, "bus.png")

# ── Constants ─────────────────────────────────────────────────────────────────
MAP_HALF_SIZE  = 900    # metres (web-mercator) — map view crop
FETCH_RADIUS   = 1200   # OSM feature fetch radius (reduced from 1500)
MTR_FETCH_DIST = 1800   # MTR station search + walk graph radius

# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "residential" : "#f2c6a0",
    "industrial"  : "#b39ddb",
    "park"        : "#b7dfb9",
    "school"      : "#9ecae1",
    "hospital"    : "#f48fb1",
    "hotel"       : "#ffe082",
    "office"      : "#80cbc4",
    "retail"      : "#ef9a9a",
    "restaurant"  : "#ffcc80",
    "attraction"  : "#ce93d8",
    "warehouse"   : "#bcaaa4",
    "building"    : "#d9d9d9",
    "site"        : "#e53935",
    "route"       : "#005eff",
    "mtr"         : "#ffd166",
    "bus"         : "#0d47a1",
}

_LAYER_COLOURS = [
    "#f2c6a0", "#9ecae1", "#b7dfb9", "#f48fb1", "#ffe082",
    "#80cbc4", "#ef9a9a", "#ffcc80", "#ce93d8", "#bcaaa4",
]


# ─────────────────────────────────────────────────────────────────────────────
# Zone → site type
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


# ─────────────────────────────────────────────────────────────────────────────
# ONE merged OSM tag dict — single Overpass call
# ─────────────────────────────────────────────────────────────────────────────

def _merged_tags(site_type: str) -> dict:
    base = {"building": True, "landuse": True, "highway": "bus_stop"}

    if site_type == "RESIDENTIAL":
        return {**base,
                "leisure": ["park", "playground"],
                "amenity": ["school", "college", "university",
                             "hospital", "clinic",
                             "supermarket", "marketplace"]}
    if site_type == "HOTEL":
        return {**base,
                "tourism": True,
                "amenity": ["restaurant", "cafe", "bar", "theatre", "cinema"],
                "shop":    ["mall", "department_store"]}
    if site_type == "OFFICE":
        return {**base,
                "office": True,
                "amenity": ["bank", "restaurant", "cafe", "parking"]}
    if site_type == "COMMERCIAL":
        return {**base,
                "shop":    ["mall", "department_store", "supermarket"],
                "amenity": ["restaurant", "cinema", "parking",
                             "fast_food", "cafe"]}
    if site_type == "INDUSTRIAL":
        return {**base,
                "landuse":  ["industrial", "warehouse", "logistics"],
                "man_made": ["works", "storage_tank"]}
    # MIXED / INSTITUTIONAL / fallback
    return {**base, "amenity": True, "leisure": True, "tourism": True}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(gdf: gpd.GeoDataFrame, col: str, values) -> gpd.GeoDataFrame:
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


def _polygons_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]


def _load_icon_array(path: str):
    try:
        return np.array(Image.open(path).convert("RGBA"))
    except Exception as e:
        log.debug("Icon load failed %s: %s", path, e)
        return None


def _place_icon(ax, x, y, arr, zoom=0.04, zorder=15):
    img = OffsetImage(arr, zoom=zoom)
    img.image.axes = ax
    ax.add_artist(AnnotationBbox(img, (x, y), frameon=False,
                                 zorder=zorder, pad=0))


def _wrap(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(text), width))


# ─────────────────────────────────────────────────────────────────────────────
# Layer extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_layers(feats: gpd.GeoDataFrame, site_type: str) -> dict:
    layers = {}

    def add(name, col, vals):
        layers[name] = _safe(feats, col, vals)

    if site_type == "RESIDENTIAL":
        add("Residential Area",   "landuse", ["residential"])
        add("Park / Greenspace",  "leisure", ["park", "playground"])
        add("School / Education", "amenity", ["school", "college", "university"])
        add("Hospital / Clinic",  "amenity", ["hospital", "clinic"])
        add("Supermarket",        "amenity", ["supermarket", "marketplace"])
    elif site_type == "HOTEL":
        add("Hotel / Resort",     "tourism", ["hotel", "hostel",
                                               "resort", "guest_house"])
        add("Tourist Attraction", "tourism", ["attraction", "museum",
                                               "gallery", "viewpoint"])
        add("Restaurant / Cafe",  "amenity", ["restaurant", "cafe",
                                               "bar", "fast_food"])
        add("Shopping",           "shop",    ["mall", "department_store"])
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
        add("Parking",            "amenity", ["parking"])
    elif site_type == "INDUSTRIAL":
        add("Industrial Zone",    "landuse", ["industrial"])
        add("Warehouse",          "landuse", ["warehouse", "logistics"])
        add("Works / Facility",   "man_made",["works", "storage_tank"])
    else:  # MIXED / INSTITUTIONAL / fallback
        add("Residential Area",   "landuse", ["residential"])
        add("Commercial Zone",    "landuse", ["commercial", "retail"])
        add("Park / Greenspace",  "leisure", ["park"])
        add("School / Education", "amenity", ["school", "college", "university"])
        add("Hospital / Clinic",  "amenity", ["hospital", "clinic"])

    layers["Buildings (context)"] = _safe(feats, "building", True)
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Walk routes — memory-safe
# ─────────────────────────────────────────────────────────────────────────────

def _compute_walk_routes(lat, lon, stations):
    routes = []
    if stations.empty:
        return routes
    try:
        G = ox.graph_from_point(
            (lat, lon),
            dist=MTR_FETCH_DIST,
            network_type="walk",
            simplify=True,
            retain_all=False,
        )
        origin = ox.distance.nearest_nodes(G, lon, lat)
        for _, st in stations.iterrows():
            try:
                ll   = (gpd.GeoSeries([st["_centroid"]], crs=3857)
                            .to_crs(4326).iloc[0])
                dest = ox.distance.nearest_nodes(G, ll.x, ll.y)
                path = nx.shortest_path(G, origin, dest, weight="length")
                routes.append(
                    ox.routing.route_to_gdf(G, path).to_crs(3857)
                )
            except Exception as e:
                log.debug("Walk route failed: %s", e)
        del G
        gc.collect()
    except Exception as e:
        log.warning("Walk graph failed: %s — skipping routes", e)
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
        # 1. Resolve coords
        lon, lat = resolve_location(
            data_type, value, lon, lat, lot_ids, extents
        )
        site_pt = (gpd.GeoSeries([Point(lon, lat)], crs=4326)
                      .to_crs(3857).iloc[0])

        # 2. Zoning
        hits = zone_data[zone_data.contains(site_pt)]
        if hits.empty:
            zd2 = zone_data.copy()
            zd2["_d"] = zd2.geometry.distance(site_pt)
            primary_row = zd2.sort_values("_d").iloc[0]
            del zd2
        else:
            primary_row = hits.iloc[0]

        zone      = str(primary_row.get("ZONE_LABEL", "MIXED"))
        plan_no   = str(primary_row.get("PLAN_NO",    "N/A"))
        site_type = _infer_site_type(zone)
        log.info("Context: zone=%s  site_type=%s  radius=%dm",
                 zone, site_type, radius_m)
        gc.collect()

        # 3. Single OSM fetch for ALL layers + bus stops + labels
        map_bbox = box(
            site_pt.x - MAP_HALF_SIZE, site_pt.y - MAP_HALF_SIZE,
            site_pt.x + MAP_HALF_SIZE, site_pt.y + MAP_HALF_SIZE,
        )
        try:
            feats = ox.features_from_point(
                (lat, lon), dist=FETCH_RADIUS, tags=_merged_tags(site_type)
            ).to_crs(3857)
            # Clip to map view immediately
            feats = feats[feats.geometry.intersects(map_bbox)].copy()
            log.info("OSM features (clipped): %d", len(feats))
        except Exception as e:
            log.warning("OSM features fetch failed: %s", e)
            feats = gpd.GeoDataFrame(columns=["geometry"], crs=3857)
        gc.collect()

        layers = _extract_layers(feats, site_type)

        # 4. Bus stops — from same fetch, no extra query
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
            bus_stops = (bus_stops.groupby("_cl").first()
                                  .reset_index(drop=True))
        gc.collect()

        # 5. Site footprint
        site_boundary = get_lot_boundary(
            lon, lat, data_type,
            extents if len(extents) > 1 else None,
        )
        if site_boundary is not None and not site_boundary.empty:
            site_geom = site_boundary.geometry.unary_union
        else:
            near = _polygons_only(feats)
            near = near[near.geometry.distance(site_pt) < 40]
            if not near.empty:
                site_geom = (
                    near.assign(_a=near.area)
                        .sort_values("_a", ascending=False)
                        .geometry.iloc[0]
                )
            else:
                site_geom = site_pt.buffer(40)
        site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

        # 6. MTR stations — small separate fetch (points only, fast)
        try:
            stations = ox.features_from_point(
                (lat, lon), tags={"railway": "station"}, dist=MTR_FETCH_DIST
            ).to_crs(3857)
            if not stations.empty:
                stations = stations.copy()
                stations["_name"] = (
                    stations["name:en"]
                    if "name:en" in stations.columns
                    else stations.get("name")
                )
                stations["_centroid"] = stations.geometry.centroid
                stations["_dist"]     = stations["_centroid"].distance(site_pt)
                stations = (stations.dropna(subset=["_name"])
                                    .sort_values("_dist").head(2))
            gc.collect()
        except Exception as e:
            log.warning("MTR fetch failed: %s", e)
            stations = gpd.GeoDataFrame()

        # 7. Walk routes
        routes = _compute_walk_routes(lat, lon, stations)

        # 8. Labels — reuse feats, prefer name:en then name
        labels = gpd.GeoDataFrame()
        for nc in ("name:en", "name"):
            if nc in feats.columns:
                tmp = feats.copy()
                tmp["_label"] = feats[nc]
                labels = (tmp.dropna(subset=["_label"])
                             .drop_duplicates("_label").head(28))
                break

        # Free feats now — no longer needed
        del feats
        gc.collect()

        # 9. Load icons once
        mtr_arr = _load_icon_array(_MTR_LOGO)
        bus_arr = _load_icon_array(_BUS_ICON)

        # ── PLOT ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 12))

        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.Positron,
            zoom=16,
            alpha=0.95,
        )
        ax.set_xlim(site_pt.x - MAP_HALF_SIZE, site_pt.x + MAP_HALF_SIZE)
        ax.set_ylim(site_pt.y - MAP_HALF_SIZE, site_pt.y + MAP_HALF_SIZE)
        ax.set_aspect("equal")
        ax.autoscale(False)

        # Layer polygons
        legend_handles = []
        for i, (layer_name, gdf) in enumerate(layers.items()):
            if gdf.empty:
                continue
            is_ctx = "context" in layer_name.lower()
            colour = "#d9d9d9" if is_ctx else _LAYER_COLOURS[i % len(_LAYER_COLOURS)]
            alpha  = 0.30      if is_ctx else 0.75
            try:
                polys = _polygons_only(gdf)
                if not polys.empty:
                    polys.plot(ax=ax, color=colour, alpha=alpha)
                    if not is_ctx:
                        legend_handles.append(
                            mpatches.Patch(color=colour, label=layer_name)
                        )
            except Exception as e:
                log.debug("Layer plot [%s]: %s", layer_name, e)

        # Walk routes
        for route in routes:
            try:
                route.plot(ax=ax, color=COLOURS["route"],
                           linewidth=2.2, linestyle="--", zorder=8)
            except Exception:
                pass

        # Bus stops
        if not bus_stops.empty:
            for geom in bus_stops.geometry:
                try:
                    pt = geom if geom.geom_type == "Point" else geom.centroid
                    if bus_arr is not None:
                        _place_icon(ax, pt.x, pt.y, bus_arr,
                                    zoom=0.04, zorder=9)
                    else:
                        ax.plot(pt.x, pt.y, "o", color=COLOURS["bus"],
                                markersize=7, zorder=9)
                except Exception:
                    pass

        # MTR stations
        if not stations.empty:
            for _, st in stations.iterrows():
                try:
                    cp = st["_centroid"]
                    if mtr_arr is not None:
                        _place_icon(ax, cp.x, cp.y, mtr_arr,
                                    zoom=0.055, zorder=12)
                    else:
                        gpd.GeoDataFrame(
                            geometry=[st.geometry], crs=3857
                        ).plot(ax=ax, facecolor=COLOURS["mtr"],
                               edgecolor="none", alpha=0.9, zorder=10)
                    ax.text(
                        cp.x, cp.y + 120,
                        _wrap(st["_name"], 18),
                        fontsize=8.5, ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=1.0),
                        zorder=13, clip_on=True,
                    )
                except Exception:
                    pass

        # Site
        site_gdf.plot(ax=ax, facecolor=COLOURS["site"],
                      edgecolor="darkred", linewidth=2, zorder=11)
        ax.text(
            site_geom.centroid.x, site_geom.centroid.y,
            "SITE", color="white", weight="bold",
            ha="center", va="center", fontsize=9, zorder=14,
        )

        # Place labels
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
                ax.text(
                    p.x + dx, p.y + dy,
                    _wrap(row["_label"], 18),
                    fontsize=8.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=12, clip_on=True,
                )
                placed.append(p)
            except Exception:
                continue

        # Info box
        ax.text(
            0.015, 0.985,
            f"Lot: {value}\nOZP Plan: {plan_no}\n"
            f"Zoning: {zone}\nSite Type: {site_type}\n",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6),
        )

        # Legend
        legend_handles += [
            mpatches.Patch(color=COLOURS["mtr"],   label="MTR Station"),
            mpatches.Patch(color=COLOURS["site"],  label="Site"),
            mpatches.Patch(color=COLOURS["route"], label="Walk Route to MTR"),
        ]
        if not bus_stops.empty:
            legend_handles.append(
                mpatches.Patch(color=COLOURS["bus"], label="Bus Stop")
            )
        ax.legend(handles=legend_handles, loc="lower left",
                  bbox_to_anchor=(0.02, 0.08),
                  fontsize=8.5, framealpha=0.95)

        ax.set_title(
            f"Site Context Analysis — {site_type.title()} Development",
            fontsize=14, weight="bold",
        )
        ax.set_axis_off()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return buf

    finally:
        if fig is not None:
            plt.close(fig)
        gc.collect()
