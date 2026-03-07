"""
context.py — Site-Type-Driven Context Analysis Map
====================================================
Replaces the generic land-use context map with a development-type-driven
analysis.  The layers shown depend on the OZP zone of the queried site.

Site types and their relevant layers
-------------------------------------
RESIDENTIAL   : residential, schools, parks, hospitals, supermarkets, MTR
HOTEL         : hotels/tourism, restaurants, shopping, attractions, MTR
OFFICE        : offices/business, banks, restaurants, transport nodes, MTR
COMMERCIAL    : retail/malls, restaurants, cinemas, parking, MTR
INDUSTRIAL    : industrial, warehouses, logistics, highways, freight rail
(fallback)    : all amenity + leisure + landuse layers
"""

import logging
import os

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import textwrap
from io import BytesIO
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point
from sklearn.cluster import KMeans

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache = True
ox.settings.log_console = False

# ── Static asset paths ────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
_MTR_LOGO   = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
_BUS_ICON   = os.path.join(_STATIC_DIR, "bus.png")

# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "residential"  : "#f2c6a0",
    "industrial"   : "#b39ddb",
    "park"         : "#b7dfb9",
    "school"       : "#9ecae1",
    "hospital"     : "#f48fb1",
    "hotel"        : "#ffe082",
    "office"       : "#80cbc4",
    "retail"       : "#ef9a9a",
    "restaurant"   : "#ffcc80",
    "attraction"   : "#ce93d8",
    "warehouse"    : "#bcaaa4",
    "building"     : "#d9d9d9",
    "site"         : "#e53935",
    "route"        : "#005eff",
    "mtr"          : "#ffd166",
}

MAP_HALF_SIZE   = 900   # metres (web-mercator)
FETCH_RADIUS    = 1500  # OSM fetch radius in metres
MTR_FETCH_DIST  = 2000  # MTR station search radius


# ─────────────────────────────────────────────────────────────────────────────
# Zone → site type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                    return "RESIDENTIAL"
    if z.startswith("C") or "COMM" in z:    return "COMMERCIAL"
    if "HOTEL" in z:                         return "HOTEL"
    if z.startswith("G") or "INST" in z:    return "INSTITUTIONAL"
    if z.startswith("I") or "IND" in z:     return "INDUSTRIAL"
    if "OFFICE" in z or "B" == z[:1]:       return "OFFICE"
    if z.startswith("OU"):                   return "MIXED"
    return "MIXED"


def _osm_tags_for(site_type: str) -> dict:
    """Return OSM tag dict for features_from_point based on site type."""
    base = {"building": True, "landuse": True}
    if site_type == "RESIDENTIAL":
        return {**base, "leisure": ["park", "playground"],
                "amenity": ["school", "college", "university",
                             "hospital", "clinic",
                             "supermarket", "marketplace"]}
    if site_type == "HOTEL":
        return {**base, "tourism": True,
                "amenity": ["restaurant", "cafe", "bar",
                             "theatre", "cinema"],
                "shop":    ["mall", "department_store"]}
    if site_type == "OFFICE":
        return {**base,
                "office": True,
                "amenity": ["bank", "restaurant", "cafe",
                             "parking", "bus_station"],
                "landuse": ["commercial", "retail"]}
    if site_type == "COMMERCIAL":
        return {**base,
                "shop":   ["mall", "department_store", "supermarket"],
                "amenity": ["restaurant", "cinema", "parking",
                             "fast_food", "cafe"],
                "landuse": ["retail", "commercial"]}
    if site_type == "INDUSTRIAL":
        return {**base,
                "landuse": ["industrial", "warehouse", "logistics"],
                "man_made": ["works", "storage_tank"],
                "railway":  ["rail", "freight"]}
    # fallback / INSTITUTIONAL / MIXED
    return {**base, "amenity": True, "leisure": True, "tourism": True}


# ─────────────────────────────────────────────────────────────────────────────
# Layer extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(gdf, col, values):
    """Filter a GeoDataFrame by column values; returns empty GDF on failure."""
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


def _extract_layers(feats: gpd.GeoDataFrame, site_type: str) -> dict:
    """Return a dict of {layer_name: GeoDataFrame} for the given site type."""
    layers: dict[str, gpd.GeoDataFrame] = {}

    def add(name, col, vals):
        layers[name] = _safe(feats, col, vals)

    buildings = _safe(feats, "building", True)

    if site_type == "RESIDENTIAL":
        add("Residential Area",  "landuse",  ["residential"])
        add("Park / Greenspace", "leisure",  ["park", "playground"])
        add("School / Education","amenity",  ["school", "college", "university"])
        add("Hospital / Clinic", "amenity",  ["hospital", "clinic"])
        add("Supermarket",       "amenity",  ["supermarket", "marketplace"])
        layers["Buildings (context)"] = buildings

    elif site_type == "HOTEL":
        add("Hotel / Resort",    "tourism",  ["hotel", "hostel",
                                               "resort", "guest_house"])
        add("Tourist Attraction","tourism",  ["attraction", "museum",
                                               "gallery", "viewpoint"])
        add("Restaurant / Cafe", "amenity",  ["restaurant", "cafe",
                                               "bar", "fast_food"])
        add("Shopping",          "shop",     ["mall", "department_store"])
        layers["Buildings (context)"] = buildings

    elif site_type == "OFFICE":
        add("Office / Business", "office",   True)
        add("Commercial Zone",   "landuse",  ["commercial", "retail"])
        add("Bank",              "amenity",  ["bank"])
        add("Restaurant / Cafe", "amenity",  ["restaurant", "cafe"])
        layers["Buildings (context)"] = buildings

    elif site_type == "COMMERCIAL":
        add("Retail / Mall",     "shop",     ["mall", "department_store",
                                               "supermarket"])
        add("Retail Landuse",    "landuse",  ["retail", "commercial"])
        add("Restaurant / Cafe", "amenity",  ["restaurant", "cafe",
                                               "fast_food"])
        add("Cinema / Leisure",  "amenity",  ["cinema", "theatre"])
        add("Parking",           "amenity",  ["parking"])
        layers["Buildings (context)"] = buildings

    elif site_type == "INDUSTRIAL":
        add("Industrial Zone",   "landuse",  ["industrial"])
        add("Warehouse / Logistics","landuse",["warehouse", "logistics"])
        add("Works / Facility",  "man_made", ["works", "storage_tank"])
        layers["Buildings (context)"] = buildings

    else:  # MIXED / INSTITUTIONAL / fallback
        add("Residential Area",  "landuse",  ["residential"])
        add("Commercial Zone",   "landuse",  ["commercial", "retail"])
        add("Park / Greenspace", "leisure",  ["park"])
        add("School / Education","amenity",  ["school", "college", "university"])
        add("Hospital / Clinic", "amenity",  ["hospital", "clinic"])
        layers["Buildings (context)"] = buildings

    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

_LAYER_COLOURS = [
    "#f2c6a0", "#9ecae1", "#b7dfb9", "#f48fb1", "#ffe082",
    "#80cbc4", "#ef9a9a", "#ffcc80", "#ce93d8", "#bcaaa4",
]


def _load_icon(path: str, zoom: float = 0.035) -> OffsetImage | None:
    try:
        img = Image.open(path).convert("RGBA")
        return OffsetImage(np.array(img), zoom=zoom)
    except Exception as e:
        log.debug("Icon load failed %s: %s", path, e)
        return None


def _place_icon(ax, x: float, y: float, imagebox: OffsetImage):
    ab = AnnotationBbox(imagebox, (x, y),
                        frameon=False, zorder=15, pad=0)
    ax.add_artist(ab)


def _wrap(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(text), width))


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_context(
    data_type: str,
    value:     str,
    zone_data: gpd.GeoDataFrame,
    radius_m:  int | None,
    lon:       float | None = None,
    lat:       float | None = None,
    lot_ids:   list  | None = None,
    extents:   list  | None = None,
) -> BytesIO:

    lot_ids  = lot_ids  or []
    extents  = extents  or []
    radius_m = radius_m or 600

    # ── 1. Resolve coordinates ────────────────────────────────────────────────
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    site_pt_wgs = Point(lon, lat)
    site_pt  = gpd.GeoSeries([site_pt_wgs], crs=4326).to_crs(3857).iloc[0]

    # ── 2. Zoning lookup ──────────────────────────────────────────────────────
    primary_hits = zone_data[zone_data.contains(site_pt)]
    if primary_hits.empty:
        # nearest zone as fallback
        zone_data["_d"] = zone_data.geometry.distance(site_pt)
        primary_row = zone_data.sort_values("_d").iloc[0]
    else:
        primary_row = primary_hits.iloc[0]

    zone      = str(primary_row.get("ZONE_LABEL", "MIXED"))
    plan_no   = str(primary_row.get("PLAN_NO",    "N/A"))
    site_type = _infer_site_type(zone)

    log.info("Context: zone=%s  site_type=%s  radius=%dm", zone, site_type, radius_m)

    # ── 3. Fetch OSM features ─────────────────────────────────────────────────
    tags = _osm_tags_for(site_type)
    try:
        feats = ox.features_from_point(
            (lat, lon), dist=FETCH_RADIUS, tags=tags
        ).to_crs(3857)
    except Exception as e:
        log.warning("OSM features fetch failed: %s", e)
        feats = gpd.GeoDataFrame(columns=["geometry"], crs=3857)

    layers = _extract_layers(feats, site_type)

    # ── 4. Site footprint ─────────────────────────────────────────────────────
    site_boundary = get_lot_boundary(lon, lat, data_type,
                                     extents if len(extents) > 1 else None)
    if site_boundary is not None and not site_boundary.empty:
        site_geom = site_boundary.geometry.unary_union
    else:
        # fall back to OSM polygon near point
        polys = feats[
            feats.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) &
            (feats.geometry.distance(site_pt) < 40)
        ] if not feats.empty else gpd.GeoDataFrame()
        if not polys.empty:
            site_geom = (
                polys.assign(_a=polys.area)
                     .sort_values("_a", ascending=False)
                     .geometry.iloc[0]
            )
        else:
            site_geom = site_pt.buffer(40)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # ── 5. MTR stations ───────────────────────────────────────────────────────
    try:
        stations = ox.features_from_point(
            (lat, lon), tags={"railway": "station"}, dist=MTR_FETCH_DIST
        ).to_crs(3857)
        if not stations.empty:
            name_col = (stations.get("name:en")
                        if "name:en" in stations.columns
                        else stations.get("name"))
            stations = stations.copy()
            stations["_name"]    = name_col
            stations["_centroid"] = stations.geometry.centroid
            stations["_dist"]    = stations["_centroid"].distance(site_pt)
            stations = (stations.dropna(subset=["_name"])
                                .sort_values("_dist").head(2))
    except Exception:
        stations = gpd.GeoDataFrame()

    # ── 6. Bus stops (clustered) ──────────────────────────────────────────────
    try:
        bus_stops = ox.features_from_point(
            (lat, lon), tags={"highway": "bus_stop"}, dist=900
        ).to_crs(3857)
        if len(bus_stops) > 6:
            arr = np.array([[g.x, g.y] for g in bus_stops.geometry])
            bus_stops = bus_stops.copy()
            bus_stops["_cl"] = (KMeans(n_clusters=6, random_state=0)
                                .fit(arr).labels_)
            bus_stops = bus_stops.groupby("_cl").first()
    except Exception:
        bus_stops = gpd.GeoDataFrame()

    # ── 7. Walk route to nearest MTR ──────────────────────────────────────────
    routes = []
    if not stations.empty:
        try:
            G = ox.graph_from_point((lat, lon), dist=MTR_FETCH_DIST,
                                    network_type="walk")
            origin = ox.distance.nearest_nodes(G, lon, lat)
            for _, st in stations.iterrows():
                ll = (gpd.GeoSeries([st["_centroid"]], crs=3857)
                         .to_crs(4326).iloc[0])
                dest = ox.distance.nearest_nodes(G, ll.x, ll.y)
                path = nx.shortest_path(G, origin, dest, weight="length")
                routes.append(ox.routing.route_to_gdf(G, path).to_crs(3857))
        except Exception as e:
            log.warning("Walk-route failed: %s", e)

    # ── 8. Place labels ───────────────────────────────────────────────────────
    label_tags = {
        "amenity": True, "leisure": True,
        "tourism": True, "shop": True,
    }
    try:
        labels = ox.features_from_point(
            (lat, lon), dist=radius_m, tags=label_tags
        ).to_crs(3857)
        name_col = (labels.get("name:en")
                    if "name:en" in labels.columns
                    else labels.get("name"))
        labels = labels.copy()
        labels["_label"] = name_col
        labels = (labels.dropna(subset=["_label"])
                         .drop_duplicates("_label").head(28))
    except Exception:
        labels = gpd.GeoDataFrame()

    # ── 9. Draw ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12))
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                   zoom=16, alpha=0.95)
    ax.set_xlim(site_pt.x - MAP_HALF_SIZE, site_pt.x + MAP_HALF_SIZE)
    ax.set_ylim(site_pt.y - MAP_HALF_SIZE, site_pt.y + MAP_HALF_SIZE)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # Layer polygons
    legend_handles = []
    for i, (layer_name, gdf) in enumerate(layers.items()):
        if gdf.empty:
            continue
        colour = ("#d9d9d9" if "context" in layer_name.lower()
                  else _LAYER_COLOURS[i % len(_LAYER_COLOURS)])
        alpha  = 0.30 if "context" in layer_name.lower() else 0.75
        try:
            polys = gdf[gdf.geometry.geom_type.isin(
                ["Polygon", "MultiPolygon"])]
            if not polys.empty:
                polys.plot(ax=ax, color=colour, alpha=alpha)
                if "context" not in layer_name.lower():
                    legend_handles.append(
                        mpatches.Patch(color=colour, label=layer_name))
        except Exception as e:
            log.debug("Layer plot error [%s]: %s", layer_name, e)

    # Walk routes
    for route in routes:
        route.plot(ax=ax, color=COLOURS["route"],
                   linewidth=2.2, linestyle="--", zorder=8)

    # Bus stops — icon or fallback dot
    if not bus_stops.empty:
        bus_img = _load_icon(_BUS_ICON, zoom=0.04)
        for geom in bus_stops.geometry:
            pt = geom if geom.geom_type == "Point" else geom.centroid
            if bus_img:
                _place_icon(ax, pt.x, pt.y,
                            OffsetImage(bus_img.get_data(), zoom=0.04))
            else:
                ax.plot(pt.x, pt.y, "o", color="#0d47a1",
                        markersize=7, zorder=9)

    # MTR stations — logo icon or coloured polygon
    mtr_img = _load_icon(_MTR_LOGO, zoom=0.055)
    if not stations.empty:
        for _, st in stations.iterrows():
            cx_pt = st["_centroid"]
            if mtr_img:
                _place_icon(ax, cx_pt.x, cx_pt.y,
                            OffsetImage(mtr_img.get_data(), zoom=0.055))
            else:
                # fallback: orange polygon
                row_gdf = gpd.GeoDataFrame(
                    geometry=[st.geometry], crs=3857)
                row_gdf.plot(ax=ax, facecolor=COLOURS["mtr"],
                             edgecolor="none", alpha=0.9, zorder=10)
            # station name label
            ax.text(cx_pt.x, cx_pt.y + 120,
                    _wrap(st["_name"], 18),
                    fontsize=8.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=1.0),
                    zorder=12, clip_on=True)

    # Site footprint
    site_gdf.plot(ax=ax, facecolor=COLOURS["site"],
                  edgecolor="darkred", linewidth=2, zorder=11)
    ax.text(site_geom.centroid.x, site_geom.centroid.y,
            "SITE", color="white", weight="bold",
            ha="center", va="center", fontsize=9, zorder=12)

    # Place labels
    placed = []
    offsets = [(0, 38), (0, -38), (40, 0), (-40, 0),
               (28, 28), (-28, 28), (28, -28), (-28, -28)]
    for i, (_, row) in enumerate(labels.iterrows()):
        try:
            p = row.geometry.representative_point()
            if p.distance(site_pt) < 140:
                continue
            if any(p.distance(pp) < 130 for pp in placed):
                continue
            dx, dy = offsets[i % len(offsets)]
            ax.text(p.x + dx, p.y + dy,
                    _wrap(row["_label"], 18),
                    fontsize=8.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, boxstyle="round,pad=0.25"),
                    zorder=12, clip_on=True)
            placed.append(p)
        except Exception:
            continue

    # ── Info box ──────────────────────────────────────────────────────────────
    ax.text(0.015, 0.985,
            f"Lot: {value}\n"
            f"OZP Plan: {plan_no}\n"
            f"Zoning: {zone}\n"
            f"Site Type: {site_type}\n",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="black", pad=6))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles += [
        mpatches.Patch(color=COLOURS["mtr"],   label="MTR Station"),
        mpatches.Patch(color=COLOURS["site"],  label="Site"),
        mpatches.Patch(color=COLOURS["route"], label="Walk Route to MTR"),
    ]
    if not bus_stops.empty:
        legend_handles.append(
            mpatches.Patch(color="#0d47a1", label="Bus Stop"))

    ax.legend(handles=legend_handles,
              loc="lower left", bbox_to_anchor=(0.02, 0.08),
              fontsize=8.5, framealpha=0.95)

    ax.set_title(
        f"Site Context Analysis — {site_type.title()} Development",
        fontsize=14, weight="bold")
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
