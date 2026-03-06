import logging
import os
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
import matplotlib.patches as mpatches
import numpy as np
import textwrap
from typing import Optional
from shapely.geometry import Point
from sklearn.cluster import KMeans
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from modules.resolver import resolve_location, get_lot_boundary
from modules.driving import _add_mtr_icon

ox.settings.use_cache = True
ox.settings.log_console = False

FETCH_RADIUS = 800
MAP_HALF_SIZE = 600
NEAREST_NAMED_STATION_M = 150  # fallback unnamed to this named station if within distance (m)
MTR_COLOR = "#d84315"  # dark orange highlight for MTR stations

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
_BUS_ICON_PATH = os.path.join(_STATIC_DIR, "bus.png")
try:
    _bus_icon = mpimg.imread(_BUS_ICON_PATH)
except Exception:
    _bus_icon = None


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def wrap_label(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))

def infer_site_type(zone):
    if zone.startswith("R"): return "RESIDENTIAL"
    if zone.startswith("C"): return "COMMERCIAL"
    if zone.startswith("G"): return "INSTITUTIONAL"
    if "HOTEL" in zone.upper() or zone.startswith("OU"): return "HOTEL"
    return "MIXED"

def context_rules(site_type):
    if site_type == "RESIDENTIAL":
        return {"amenity":["school","college","university"],"leisure":["park"],"place":["neighbourhood"]}
    if site_type == "COMMERCIAL":
        return {"amenity":["bank","restaurant","market"],"railway":["station"]}
    if site_type == "INSTITUTIONAL":
        return {"amenity":["school","college","hospital"],"leisure":["park"]}
    return {"amenity":True,"leisure":True}


# ------------------------------------------------------------
# MAIN GENERATOR
# ------------------------------------------------------------

def generate_context(data_type: str, value: str, ZONE_DATA: gpd.GeoDataFrame,
                     radius_m: Optional[int] = None,
                     lon: float = None, lat: float = None,
                     lot_ids: list = None, extents: list = None):

    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r = radius_m if radius_m is not None else FETCH_RADIUS
    half_size    = radius_m if radius_m is not None else MAP_HALF_SIZE
    half_x = half_size * (992 / 737)
    half_y = half_size
    logging.info("CONTEXT extent: half_x=%.0f half_y=%.0f (m)", half_x, half_y)
    lot_gdf = get_lot_boundary(lon, lat, data_type, extents if extents else None)
    if lot_gdf is not None:
        site_geom = lot_gdf.geometry.iloc[0]
        site_gdf = lot_gdf
        site_point = site_geom.centroid
    else:
        site_point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    # --------------------------------------------------------
    # ZONING
    # --------------------------------------------------------

    ozp = ZONE_DATA.to_crs(3857)
    primary = ozp[ozp.contains(site_point)]

    if primary.empty:
        raise ValueError("No zoning polygon found.")

    primary = primary.iloc[0]
    zone = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    LABEL_RULES = context_rules(SITE_TYPE)

    # --------------------------------------------------------
    # FETCH OSM DATA
    # --------------------------------------------------------

    polygons = ox.features_from_point(
        (lat, lon),
        dist=fetch_r,
        tags={"landuse":True,"leisure":True,"amenity":True,"building":True}
    ).to_crs(3857)

    residential = polygons[polygons.get("landuse")=="residential"]
    industrial  = polygons[polygons.get("landuse").isin(["industrial","commercial"])]
    parks       = polygons[polygons.get("leisure")=="park"]
    schools     = polygons[polygons.get("amenity").isin(["school","college","university"])]
    buildings   = polygons[polygons.get("building").notnull()]

    # --------------------------------------------------------
    # SITE FOOTPRINT (only when not using official lot boundary)
    # --------------------------------------------------------

    if lot_gdf is None:
        candidates = polygons[
            polygons.geometry.geom_type.isin(["Polygon","MultiPolygon"]) &
            (polygons.geometry.distance(site_point) < 40)
        ]

        if len(candidates):
            site_geom = (
                candidates.assign(area=candidates.area)
                .sort_values("area", ascending=False)
                .geometry.iloc[0]
            )
        else:
            site_geom = site_point.buffer(40)

        site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # --------------------------------------------------------
    # MTR STATIONS
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon), tags={"railway":"station"}, dist=1200
    ).to_crs(3857)

    if not stations.empty:
        stations["name"] = stations.get("name:en").fillna(stations.get("name"))
        stations["centroid"] = stations.geometry.centroid
        stations["dist"] = stations["centroid"].distance(site_point)
        stations = stations.dropna(subset=["name"]).sort_values("dist").head(2)

    # --------------------------------------------------------
    # BUS STOPS
    # --------------------------------------------------------

    bus_stops = ox.features_from_point(
        (lat, lon), tags={"highway":"bus_stop"}, dist=700
    ).to_crs(3857)

    if len(bus_stops) > 6:
        coords_array = np.array([[g.x, g.y] for g in bus_stops.geometry])
        bus_stops["cluster"] = KMeans(n_clusters=6, random_state=0).fit(coords_array).labels_
        bus_stops = bus_stops.groupby("cluster").first()

    # --------------------------------------------------------
    # PLACE LABELS
    # --------------------------------------------------------

    labels = ox.features_from_point(
        (lat, lon), dist=600, tags=LABEL_RULES
    ).to_crs(3857)

    labels["label"] = labels.get("name:en").fillna(labels.get("name"))
    labels = labels.dropna(subset=["label"]).drop_duplicates("label").head(24)

    # --------------------------------------------------------
    # ALL LABEL SOURCES (by site type: only relevant layers)
    # --------------------------------------------------------
    if SITE_TYPE == "RESIDENTIAL":
        label_sources = (labels, residential, parks, schools)
    elif SITE_TYPE == "COMMERCIAL":
        label_sources = (labels, industrial, parks)
    elif SITE_TYPE == "INSTITUTIONAL":
        label_sources = (labels, parks, schools, buildings)
    else:
        label_sources = (labels, residential, buildings, parks, schools, industrial)
    all_label_items = []  # list of (distance, geometry, label)
    for gdf in label_sources:
        if gdf.empty:
            continue
        gdf = gdf.copy()
        gdf["label"] = gdf.get("name:en").fillna(gdf.get("name"))
        named = gdf.dropna(subset=["label"])
        named["label"] = named["label"].astype(str).str.strip()
        named = named[named["label"].str.len() > 0]
        for _, row in named.iterrows():
            geom = row.geometry
            text = row["label"]
            p = geom.representative_point() if hasattr(geom, "representative_point") else geom.centroid
            dist = p.distance(site_point)
            all_label_items.append((dist, geom, text))
    all_label_items.sort(key=lambda x: x[0])
    all_label_items = [(geom, text) for _, geom, text in all_label_items[:40]]

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16.15,12))

    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerNoLabels, zoom=16, alpha=0.95)

    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.set_aspect("equal")
    ax.autoscale(False)

    residential.plot(ax=ax, color="#f2c6a0", alpha=0.9)
    industrial.plot(ax=ax, color="#b39ddb", alpha=0.9)
    parks.plot(ax=ax, color="#b7dfb9", alpha=0.9)
    schools.plot(ax=ax, color="#9ecae1", alpha=0.9)
    buildings.plot(ax=ax, color="#d9d9d9", alpha=0.35)

    if not bus_stops.empty:
        if _bus_icon is not None:
            for _, row in bus_stops.iterrows():
                geom = row.geometry
                bx, by = geom.centroid.x, geom.centroid.y
                icon = OffsetImage(_bus_icon, zoom=0.02)
                icon.image.axes = ax
                ab = AnnotationBbox(icon, (bx, by), frameon=False, zorder=9, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)
        else:
            bus_stops.plot(ax=ax, color="#0d47a1", markersize=35, zorder=9)

    stations_in_view = stations[stations["dist"] <= half_size] if not stations.empty else stations
    if not stations_in_view.empty:
        stations_in_view.plot(ax=ax, facecolor=MTR_COLOR, edgecolor="none", alpha=0.9, zorder=10)
        for _, st in stations_in_view.iterrows():
            c = st.geometry.centroid
            _add_mtr_icon(ax, c.x, c.y, size=0.035, zorder=14)

    site_gdf.plot(ax=ax, facecolor="#e53935", edgecolor="darkred", linewidth=2, zorder=11)

    ax.text(site_geom.centroid.x, site_geom.centroid.y, "SITE",
            color="white", weight="bold", ha="center", va="center", zorder=12)

    # --------------------------------------------------------
    # DRAW PLACE LABELS (all sources: labels, residential, buildings, parks, schools, industrial)
    # --------------------------------------------------------
    placed = []
    for geom, text in all_label_items:
        p = geom.representative_point() if hasattr(geom, "representative_point") else geom.centroid
        if p.distance(site_point) < 140:
            continue
        if any(p.distance(pp) < 100 for pp in placed):
            continue
        ax.text(
            p.x, p.y,
            wrap_label(text, 18),
            fontfamily="Arial",
            fontsize=12, weight="bold",
            ha="center", va="center",
            color="black",
            zorder=12,
            clip_on=True,
        )
        placed.append(p)

    # --------------------------------------------------------
    # MTR NAME LABELS
    # --------------------------------------------------------

    if not stations_in_view.empty:
        named_stations = [
            (row.geometry.centroid, str(row["name"]).strip())
            for _, row in stations_in_view.iterrows()
            if row["name"] is not None and isinstance(row["name"], str) and str(row["name"]).strip()
        ]
        for _, st in stations_in_view.iterrows():
            raw_name = st["name"]
            if raw_name is None or (isinstance(raw_name, float) and raw_name != raw_name) or not isinstance(raw_name, str) or not str(raw_name).strip():
                label_name = "UNNAMED"
                if named_stations:
                    cent = st.geometry.centroid
                    min_d = min((cent.distance(c) for c, _ in named_stations), default=float("inf"))
                    if min_d <= NEAREST_NAMED_STATION_M:
                        _, label_name = min(named_stations, key=lambda p: cent.distance(p[0]))
            else:
                label_name = str(raw_name).strip()
            ax.text(
                st.centroid.x,
                st.centroid.y + 120,
                wrap_label(label_name, 18),
                fontsize=10, weight="bold", color="black",
                ha="center", va="bottom", zorder=15
            )

    # --------------------------------------------------------
    # INFO BOX
    # --------------------------------------------------------

    ax.text(
        0.015, 0.985,
        f"{data_type}: {value}\n"
        f"OZP Plan: {primary['PLAN_NO']}\n"
        f"Zoning: {zone}\n"
        f"Site Type: {SITE_TYPE}\n",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9.2,
        bbox=dict(facecolor="white", edgecolor="black", pad=6)
    )

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    ax.legend(
        handles=[
            mpatches.Patch(color="#f2c6a0",label="Residential"),
            mpatches.Patch(color="#b39ddb",label="Industrial / Commercial"),
            mpatches.Patch(color="#b7dfb9",label="Public Park"),
            mpatches.Patch(color="#9ecae1",label="School / Institution"),
            mpatches.Patch(color=MTR_COLOR,label="MTR Station"),
            mpatches.Patch(color="#e53935",label="Site"),
            mpatches.Patch(color="#0d47a1",label="Bus Stop"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.02,0.08),
        fontsize=8.5,
        framealpha=0.95
    )

    ax.set_title(
        f"Automated Site Context Analysis – {data_type} {value}",
        fontsize=15,
        weight="bold"
    )

    ax.set_axis_off()

    # Re-apply extent so basemap/plotting cannot leave the map zoomed in
    ax.set_xlim(site_point.x - half_x, site_point.x + half_x)
    ax.set_ylim(site_point.y - half_y, site_point.y + half_y)
    ax.autoscale(False)

    buffer = BytesIO()
    plt.tight_layout()
    ax.set_position([0.02, 0.02, 0.96, 0.96])
    plt.savefig(buffer, format="png", dpi=120)
    plt.close(fig)

    return buffer
