import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from shapely.geometry import Point
from io import BytesIO

ox.settings.log_console = False
ox.settings.use_cache = True

# ------------------------------------------------------------
# OPTIMIZED SETTINGS
# ------------------------------------------------------------

WALK_SPEED_KMPH = 5
GRAPH_RADIUS = 2200      # reduced from 3000
MAP_EXTENT = 1600        # reduced from 2000
STATION_RADIUS = 2200    # reduced


# ------------------------------------------------------------
# MAIN GENERATOR (UPDATED + OPTIMIZED)
# ------------------------------------------------------------

def generate_walking(lon: float, lat: float):

    # --------------------------------------------------------
    # SITE FOOTPRINT
    # --------------------------------------------------------

    osm_site = ox.features_from_point(
        (lat, lon),
        dist=60,
        tags={"building": True}
    ).to_crs(3857)

    if len(osm_site):
        osm_site["area_calc"] = osm_site.geometry.area
        site_geom = osm_site.sort_values(
            "area_calc", ascending=False
        ).geometry.iloc[0]
    else:
        site_geom = gpd.GeoSeries(
            [Point(lon, lat)],
            crs=4326
        ).to_crs(3857).iloc[0].buffer(40)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)
    site_point = site_geom.centroid

    # --------------------------------------------------------
    # WALK NETWORK (Reduced Radius)
    # --------------------------------------------------------

    G_walk = ox.graph_from_point(
        (lat, lon),
        dist=GRAPH_RADIUS,
        network_type="walk"
    )

    roads = ox.graph_to_gdfs(
        G_walk,
        nodes=False
    ).to_crs(3857)

    site_centroid_wgs = gpd.GeoSeries(
        [site_point],
        crs=3857
    ).to_crs(4326).iloc[0]

    site_node = ox.distance.nearest_nodes(
        G_walk,
        site_centroid_wgs.x,
        site_centroid_wgs.y
    )

    # --------------------------------------------------------
    # FETCH STATIONS (Reduced Radius)
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon),
        tags={"railway": "station"},
        dist=STATION_RADIUS
    ).to_crs(3857)

    stations = stations[stations.geometry.notnull()]

    if stations.empty:
        raise ValueError("No nearby stations found.")

    stations["station_name"] = stations.apply(
        lambda r: r.get("name:en")
        if isinstance(r.get("name:en"), str)
        else r.get("name")
        if isinstance(r.get("name"), str)
        else "MTR Station",
        axis=1
    )

    stations["dist"] = stations.geometry.centroid.distance(site_point)
    stations = stations.sort_values("dist").head(3)

    # --------------------------------------------------------
    # ROUTES
    # --------------------------------------------------------

    routes = []

    for _, row in stations.iterrows():

        st_centroid = row.geometry.centroid
        st_wgs = gpd.GeoSeries(
            [st_centroid],
            crs=3857
        ).to_crs(4326).iloc[0]

        st_node = ox.distance.nearest_nodes(
            G_walk,
            st_wgs.x,
            st_wgs.y
        )

        try:
            path = nx.shortest_path(
                G_walk,
                site_node,
                st_node,
                weight="length"
            )
        except:
            continue

        route = ox.routing.route_to_gdf(
            G_walk,
            path
        ).to_crs(3857)

        dist_km = round(route.length.sum() / 1000, 2)
        time_min = max(1, round((dist_km / WALK_SPEED_KMPH) * 60))

        routes.append({
            "route": route,
            "distance": dist_km,
            "time": time_min,
            "station_polygon": row.geometry,
            "station_centroid": st_centroid,
            "name": row["station_name"]
        })

    # --------------------------------------------------------
    # PLOT (Smaller + Lower DPI)
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    roads.plot(ax=ax, linewidth=0.2, color="#8a8a8a", alpha=0.35)

    # Walking Rings
    gpd.GeoSeries([site_point.buffer(1125)], crs=3857).plot(
        ax=ax, color="#2aa9ff", alpha=0.12
    )

    for d, lbl in [(375, "5 min"), (750, "10 min"), (1125, "15 min")]:
        gpd.GeoSeries([site_point.buffer(d)], crs=3857).boundary.plot(
            ax=ax,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            color="#2aa9ff"
        )
        ax.text(site_point.x + d + 100, site_point.y, lbl, fontsize=8)

    colors = ["#4caf50", "#ef5350", "#42a5f5"]

    for i, r in enumerate(routes):

        route_color = colors[i % len(colors)]

        r["route"].plot(
            ax=ax,
            linewidth=2.4,
            color=route_color,
            alpha=0.85,
            zorder=5
        )

        station_geom = r["station_polygon"]

        if station_geom.geom_type == "Point":
            station_geom = station_geom.buffer(55)

        gpd.GeoSeries([station_geom], crs=3857).plot(
            ax=ax,
            facecolor=route_color,
            edgecolor=route_color,
            linewidth=1,
            alpha=0.25,
            zorder=4
        )

        mid = r["route"].geometry.iloc[len(r["route"]) // 2].centroid

        ax.text(
            mid.x,
            mid.y,
            f"{r['time']} min\n{r['distance']} km",
            fontsize=8,
            weight="bold",
            color=route_color,
            ha="center",
            zorder=6
        )

        ax.text(
            r["station_centroid"].x,
            r["station_centroid"].y + 100,
            r["name"].upper(),
            fontsize=9,
            weight="bold",
            ha="center",
            zorder=7
        )

    # Site
    site_gdf.plot(ax=ax, facecolor="red", edgecolor="none")

    ax.text(
        site_point.x,
        site_point.y - 100,
        "SITE",
        color="red",
        weight="bold",
        ha="center"
    )

    ax.set_xlim(site_point.x - MAP_EXTENT, site_point.x + MAP_EXTENT)
    ax.set_ylim(site_point.y - MAP_EXTENT, site_point.y + MAP_EXTENT)

    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        zoom=14,   # reduced from 15
        alpha=0.4
    )

    ax.set_title(
        "Walking Accessibility",
        fontsize=14,
        weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=130)  # reduced from 200
    plt.close(fig)

    return buffer
