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

v2 fixes:
  - requests_timeout = 25 on every Overpass call (prevents Render restart)
  - try/except + gc.collect() after every OSM fetch
  - walk-graph radius capped at 1500 m (was 2000 — too heavy)
  - basemap extent set BEFORE cx.add_basemap so tiles are fetched correctly
  - all OSM fetches degrade gracefully to empty GeoDataFrame on failure
"""

import io
import gc
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

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 25   # hard cap – prevents Render health-check kill

# ── Tuneable constants ────────────────────────────────────────────────────────
FETCH_RADIUS   = 1500   # OSM polygon fetch radius (m)
MAP_HALF_SIZE  = 900    # half-width / half-height of final map (m Web-Mercator)
MTR_COLOR      = "#ffd166"
BUS_CLUSTER_N  = 6
BUS_RADIUS_M   = 900
WALK_GRAPH_R   = 1500   # walk-graph fetch radius — capped to avoid OOM/timeout
MTR_STATION_R  = 2000   # MTR station search radius (m)


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
        return {
            "amenity": ["school", "college", "university"],
            "leisure": ["park"],
            "place":   ["neighbourhood"],
        }
    if site_type == "COMMERCIAL":
        return {
            "amenity": ["bank", "restaurant", "market"],
            "railway": ["station"],
        }
    if site_type == "INSTITUTIONAL":
        return {
            "amenity": ["school", "college", "hospital"],
            "leisure": ["park"],
        }
    return {"amenity": True, "leisure": True}


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[], crs=3857)


# ── Main entry point ──────────────────────────────────────────────────────────

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
        zone_data["_dist"] = zone_data.geometry.distance(site_point_wm)
        primary_rows = zone_data[zone_data["_dist"] < 200].sort_values("_dist")

    if primary_rows.empty:
        raise ValueError("No OZP zoning polygon found near this location.")

    primary     = primary_rows.iloc[0]
    zone        = primary.get("ZONE_LABEL") or primary.get("ZONE") or "N/A"
    plan_no     = primary.get("PLAN_NO")    or primary.get("PLAN")  or "N/A"
    site_type   = _infer_site_type(zone)
    label_rules = _label_rules(site_type)

    # ── 3. OSM landuse / amenity polygons ─────────────────────────────────────
    log.info("context: fetching OSM polygons (r=%d m)", FETCH_RADIUS)
    try:
        polygons = ox.features_from_point(
            (lat, lon),
            dist=FETCH_RADIUS,
            tags={"landuse": True, "leisure": True, "amenity": True, "building": True},
        ).to_crs(3857)
    except Exception as e:
        log.warning("context: OSM polygon fetch failed: %s", e)
        polygons = _empty_gdf()
    gc.collect()

    if not polygons.empty:
        landuse  = polygons.get("landuse",  gpd.GeoSeries(dtype=object))
        leisure  = polygons.get("leisure",  gpd.GeoSeries(dtype=object))
        amenity  = polygons.get("amenity",  gpd.GeoSeries(dtype=object))
        building = polygons.get("building", gpd.GeoSeries(dtype=object))

        residential = polygons[landuse == "residential"]
        industrial  = polygons[landuse.isin(["industrial", "commercial"])]
        parks       = polygons[leisure  == "park"]
        schools     = polygons[amenity.isin(["school", "college", "university"])]
        buildings   = polygons[building.notnull()]
    else:
        residential = industrial = parks = schools = buildings = _empty_gdf()

    # ── 4. Site footprint ─────────────────────────────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type,
                               extents if len(extents) > 1 else None)

    if lot_gdf is not None and not lot_gdf.empty:
        site_geom = lot_gdf.geometry.iloc[0]
    else:
        if not polygons.empty:
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
        else:
            site_geom = site_point_wm.buffer(40)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # ── 5. MTR stations ───────────────────────────────────────────────────────
    log.info("context: fetching MTR stations")
    try:
        stations = ox.features_from_point(
            (lat, lon), tags={"railway": "station"}, dist=MTR_STATION_R
        ).to_crs(3857)
    except Exception as e:
        log.warning("context: station fetch failed: %s", e)
        stations = _empty_gdf()
    gc.collect()

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
    except Exception as e:
        log.warning("context: bus stop fetch failed: %s", e)
        bus_stops = _empty_gdf()
    gc.collect()

    if len(bus_stops) > BUS_CLUSTER_N:
        try:
            coords_arr = np.array([[g.x, g.y] for g in bus_stops.geometry])
            bus_stops["cluster"] = (
                KMeans(n_clusters=BUS_CLUSTER_N, random_state=0, n_init="auto")
                .fit(coords_arr)
                .labels_
            )
            bus_stops = bus_stops.groupby("cluster").first()
        except Exception as e:
            log.warning("context: bus stop clustering failed: %s", e)

    # ── 7. Walk routes to MTR ─────────────────────────────────────────────────
    routes = []

    if not stations.empty:
        log.info("context: computing walk routes (graph r=%d m)", WALK_GRAPH_R)
        try:
            # Timeout is already set globally on ox.settings.requests_timeout
            G = ox.graph_from_point(
                (lat, lon), dist=WALK_GRAPH_R, network_type="walk"
            )
            gc.collect()
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

            del G
            gc.collect()

        except Exception as e:
            log.warning("context: walk route error (skipped): %s", e)

    # ── 8. Place labels ───────────────────────────────────────────────────────
    log.info("context: fetching place labels")
    try:
        labels = ox.features_from_point(
            (lat, lon), dist=800, tags=label_rules
        ).to_crs(3857)
        gc.collect()

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
        log.warning("context: label fetch failed: %s", e)
        labels = _empty_gdf()

    # ── 9. Plot ───────────────────────────────────────────────────────────────
    log.info("context: rendering map")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Set extent FIRST so contextily fetches the correct tile region
    xmin = site_point_wm.x - MAP_HALF_SIZE
    xmax = site_point_wm.x + MAP_HALF_SIZE
    ymin = site_point_wm.y - MAP_HALF_SIZE
    ymax = site_point_wm.y + MAP_HALF_SIZE
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.autoscale(False)

    try:
        cx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=cx.providers.CartoDB.Positron,
            zoom=16,
            alpha=0.95,
        )
    except Exception as e:
        log.warning("context: basemap error: %s", e)

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

    if not labels.empty and "label" in labels.columns:
        for i, (_, row) in enumerate(labels.iterrows()):
            try:
                p = row.geometry.representative_point()
            except Exception:
                continue

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
    gc.collect()
    buf.seek(0)
    log.info("context: done")
    return buf
