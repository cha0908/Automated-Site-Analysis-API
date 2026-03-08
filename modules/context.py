"""
modules/context.py
──────────────────────────────────────────────────────────────────────────────
Site Context Analysis map — replicates the Colab output exactly.

Produces: coloured landuse polygons + MTR station footprints + bus stop
clusters + walk-routes to nearest MTR + place labels + info box + legend,
all on a CartoDB Positron basemap.

Called by main.py:
    generate_context(data_type, value, ZONE_DATA, radius_m, lon, lat,
                     lot_ids, extents)   → BytesIO (PNG)
"""

import io
import textwrap
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import geopandas as gpd
import osmnx as ox
import contextily as cx
import networkx as nx

from shapely.geometry import Point
from sklearn.cluster import KMeans

from modules.resolver import resolve_location, get_lot_boundary

log = logging.getLogger(__name__)

ox.settings.use_cache  = True
ox.settings.log_console = False

# ── Tuneable constants ────────────────────────────────────────────────────────
FETCH_RADIUS  = 1500    # OSM fetch radius (m)
MAP_HALF_SIZE = 900     # half-width / half-height of final map (m in Web-Mercator)
MTR_COLOR     = "#ffd166"
BUS_CLUSTER_N = 6
BUS_RADIUS_M  = 900


# ── Text helpers ──────────────────────────────────────────────────────────────

def _wrap(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))


def _infer_site_type(zone: str) -> str:
    z = zone.upper()
    if z.startswith("R"):                        return "RESIDENTIAL"
    if z.startswith("C"):                        return "COMMERCIAL"
    if z.startswith("G"):                        return "INSTITUTIONAL"
    if "HOTEL" in z or z.startswith("OU"):       return "HOTEL"
    return "MIXED"


def _label_rules(site_type: str) -> dict:
    if site_type == "RESIDENTIAL":
        return {
            "amenity":  ["school", "college", "university"],
            "leisure":  ["park"],
            "place":    ["neighbourhood"],
        }
    if site_type == "COMMERCIAL":
        return {
            "amenity":  ["bank", "restaurant", "market"],
            "railway":  ["station"],
        }
    if site_type == "INSTITUTIONAL":
        return {
            "amenity":  ["school", "college", "hospital"],
            "leisure":  ["park"],
        }
    return {"amenity": True, "leisure": True}


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_context(
    data_type: str,
    value:     str,
    zone_data: gpd.GeoDataFrame,
    radius_m:  int   = None,      # kept for API compatibility; not used here
    lon:       float = None,
    lat:       float = None,
    lot_ids:   list  = None,
    extents:   list  = None,
) -> io.BytesIO:

    lot_ids = lot_ids or []
    extents = extents or []

    # ── 1. Resolve location ───────────────────────────────────────────────────
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    log.info("context: resolved location %.6f, %.6f", lon, lat)

    site_point_wm = (
        gpd.GeoSeries([Point(lon, lat)], crs=4326)
        .to_crs(3857)
        .iloc[0]
    )

    # ── 2. OZP zoning ─────────────────────────────────────────────────────────
    primary_rows = zone_data[zone_data.contains(site_point_wm)]
    if primary_rows.empty:
        # fall back to nearest polygon within 200 m
        zone_data["_dist"] = zone_data.geometry.distance(site_point_wm)
        primary_rows = zone_data[zone_data["_dist"] < 200].sort_values("_dist")

    if primary_rows.empty:
        raise ValueError("No OZP zoning polygon found near this location.")

    primary   = primary_rows.iloc[0]
    zone      = primary.get("ZONE_LABEL") or primary.get("ZONE") or "N/A"
    plan_no   = primary.get("PLAN_NO")    or primary.get("PLAN")  or "N/A"
    site_type = _infer_site_type(zone)
    label_rules = _label_rules(site_type)

    # ── 3. OSM landuse / amenity polygons ─────────────────────────────────────
    log.info("context: fetching OSM polygons (r=%d m)", FETCH_RADIUS)
    polygons = ox.features_from_point(
        (lat, lon),
        dist=FETCH_RADIUS,
        tags={"landuse": True, "leisure": True, "amenity": True, "building": True},
    ).to_crs(3857)

    landuse  = polygons.get("landuse",  gpd.GeoSeries(dtype=object))
    leisure  = polygons.get("leisure",  gpd.GeoSeries(dtype=object))
    amenity  = polygons.get("amenity",  gpd.GeoSeries(dtype=object))
    building = polygons.get("building", gpd.GeoSeries(dtype=object))

    residential = polygons[landuse == "residential"]
    industrial  = polygons[landuse.isin(["industrial", "commercial"])]
    parks       = polygons[leisure  == "park"]
    schools     = polygons[amenity.isin(["school", "college", "university"])]
    buildings   = polygons[building.notnull()]

    # ── 4. Site footprint ─────────────────────────────────────────────────────
    # Try lot boundary from the resolver first
    lot_gdf = get_lot_boundary(lon, lat, data_type,
                               extents if len(extents) > 1 else None)

    if lot_gdf is not None and not lot_gdf.empty:
        site_geom = lot_gdf.geometry.iloc[0]
    else:
        # fall back: nearest OSM polygon within 40 m
        candidates = polygons[
            polygons.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) &
            (polygons.geometry.distance(site_point_wm) < 40)
        ]
        if len(candidates):
            site_geom = (
                candidates.assign(area=candidates.area)
                .sort_values("area", ascending=False)
                .geometry.iloc[0]
            )
        else:
            site_geom = site_point_wm.buffer(40)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # ── 5. MTR stations ───────────────────────────────────────────────────────
    log.info("context: fetching MTR stations")
    try:
        stations = ox.features_from_point(
            (lat, lon), tags={"railway": "station"}, dist=2000
        ).to_crs(3857)
    except Exception:
        stations = gpd.GeoDataFrame()

    if not stations.empty:
        name_en = stations.get("name:en")
        name_zh = stations.get("name")
        if name_en is not None and name_zh is not None:
            stations["name"] = name_en.fillna(name_zh)
        elif name_zh is not None:
            stations["name"] = name_zh
        else:
            stations["name"] = "Station"

        stations["centroid"] = stations.geometry.centroid
        stations["dist"]     = stations["centroid"].distance(site_point_wm)
        stations = (
            stations.dropna(subset=["name"])
            .sort_values("dist")
            .head(2)
        )

    # ── 6. Bus stops (clustered) ──────────────────────────────────────────────
    log.info("context: fetching bus stops")
    try:
        bus_stops = ox.features_from_point(
            (lat, lon), tags={"highway": "bus_stop"}, dist=BUS_RADIUS_M
        ).to_crs(3857)
    except Exception:
        bus_stops = gpd.GeoDataFrame()

    if len(bus_stops) > BUS_CLUSTER_N:
        coords_arr = np.array([[g.x, g.y] for g in bus_stops.geometry])
        bus_stops["cluster"] = (
            KMeans(n_clusters=BUS_CLUSTER_N, random_state=0, n_init="auto")
            .fit(coords_arr)
            .labels_
        )
        bus_stops = bus_stops.groupby("cluster").first()

    # ── 7. Walk routes to MTR ─────────────────────────────────────────────────
    routes = []

    if not stations.empty:
        log.info("context: computing walk routes")
        try:
            G = ox.graph_from_point((lat, lon), dist=2000, network_type="walk")
            site_node = ox.distance.nearest_nodes(G, lon, lat)

            for _, st in stations.iterrows():
                ll = (
                    gpd.GeoSeries([st.centroid], crs=3857)
                    .to_crs(4326)
                    .iloc[0]
                )
                st_node = ox.distance.nearest_nodes(G, ll.x, ll.y)
                try:
                    path  = nx.shortest_path(G, site_node, st_node, weight="length")
                    route = ox.routing.route_to_gdf(G, path).to_crs(3857)
                    routes.append(route)
                except nx.NetworkXNoPath:
                    log.debug("No walk path to %s", st.get("name", "station"))
        except Exception as e:
            log.warning("Walk route error: %s", e)

    # ── 8. Place labels ───────────────────────────────────────────────────────
    log.info("context: fetching place labels")
    try:
        labels = ox.features_from_point(
            (lat, lon), dist=800, tags=label_rules
        ).to_crs(3857)

        name_en = labels.get("name:en")
        name_zh = labels.get("name")
        if name_en is not None and name_zh is not None:
            labels["label"] = name_en.fillna(name_zh)
        elif name_zh is not None:
            labels["label"] = name_zh
        else:
            labels["label"] = None

        labels = (
            labels.dropna(subset=["label"])
            .drop_duplicates("label")
            .head(24)
        )
    except Exception as e:
        log.warning("Label fetch error: %s", e)
        labels = gpd.GeoDataFrame()

    # ── 9. Plot ───────────────────────────────────────────────────────────────
    log.info("context: rendering map")
    fig, ax = plt.subplots(figsize=(12, 12))

    # basemap
    try:
        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.Positron,
            zoom=16,
            alpha=0.95,
        )
    except Exception as e:
        log.warning("Basemap error: %s", e)

    # map extent
    ax.set_xlim(site_point_wm.x - MAP_HALF_SIZE, site_point_wm.x + MAP_HALF_SIZE)
    ax.set_ylim(site_point_wm.y - MAP_HALF_SIZE, site_point_wm.y + MAP_HALF_SIZE)
    ax.set_aspect("equal")
    ax.autoscale(False)

    # landuse layers
    if not residential.empty:
        residential.plot(ax=ax, color="#f2c6a0", alpha=0.75)
    if not industrial.empty:
        industrial.plot(ax=ax, color="#b39ddb", alpha=0.75)
    if not parks.empty:
        parks.plot(ax=ax, color="#b7dfb9", alpha=0.90)
    if not schools.empty:
        schools.plot(ax=ax, color="#9ecae1", alpha=0.90)
    if not buildings.empty:
        buildings.plot(ax=ax, color="#d9d9d9", alpha=0.35)

    # walk routes
    for r in routes:
        r.plot(ax=ax, color="#005eff", linewidth=2.2, linestyle="--")

    # bus stops
    if not bus_stops.empty:
        bus_stops.plot(ax=ax, color="#0d47a1", markersize=35, zorder=9)

    # MTR station footprints
    if not stations.empty:
        stations.plot(
            ax=ax,
            facecolor=MTR_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=0.90,
            zorder=10,
        )

    # site polygon
    site_gdf.plot(
        ax=ax, facecolor="#e53935", edgecolor="darkred", linewidth=2, zorder=11
    )
    ax.text(
        site_geom.centroid.x, site_geom.centroid.y,
        "SITE",
        color="white", weight="bold", ha="center", va="center", zorder=12,
    )

    # MTR station name labels
    if not stations.empty:
        for _, st in stations.iterrows():
            ax.text(
                st.centroid.x,
                st.centroid.y + 120,
                _wrap(st["name"], 18),
                fontsize=9,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0),
                zorder=12,
                clip_on=True,
            )

    # place labels
    offsets = [(0,35),(0,-35),(35,0),(-35,0),(25,25),(-25,25)]
    placed  = []

    for i, (_, row) in enumerate(labels.iterrows()):
        p = row.geometry.representative_point()

        if p.distance(site_point_wm) < 140:
            continue
        if any(p.distance(pp) < 120 for pp in placed):
            continue

        dx, dy = offsets[i % len(offsets)]
        ax.text(
            p.x + dx, p.y + dy,
            _wrap(row["label"], 18),
            fontsize=9,
            ha="center", va="center",
            bbox=dict(
                facecolor="white", edgecolor="none",
                alpha=0.85, boxstyle="round,pad=0.25",
            ),
            zorder=12,
            clip_on=True,
        )
        placed.append(p)

    # info box (top-left)
    ax.text(
        0.015, 0.985,
        f"Lot: {value}\n"
        f"OZP Plan: {plan_no}\n"
        f"Zoning: {zone}\n"
        f"Site Type: {site_type}\n",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9.2,
        bbox=dict(facecolor="white", edgecolor="black", pad=6),
    )

    # legend
    ax.legend(
        handles=[
            mpatches.Patch(color="#f2c6a0", label="Residential"),
            mpatches.Patch(color="#b39ddb", label="Industrial / Commercial"),
            mpatches.Patch(color="#b7dfb9", label="Public Park"),
            mpatches.Patch(color="#9ecae1", label="School / Institution"),
            mpatches.Patch(color=MTR_COLOR,  label="MTR Station"),
            mpatches.Patch(color="#e53935",  label="Site"),
            mpatches.Patch(color="#005eff",  label="Pedestrian Route to MTR"),
            mpatches.Patch(color="#0d47a1",  label="Bus Stop"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.08),
        fontsize=8.5,
        framealpha=0.95,
    )

    ax.set_title(
        "Automated Site Context Analysis (Building-Type Driven)",
        fontsize=15, weight="bold",
    )
    ax.set_axis_off()

    # ── 10. Serialise to PNG BytesIO ──────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
