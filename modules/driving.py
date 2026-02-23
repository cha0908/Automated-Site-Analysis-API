import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.lines as mlines

from shapely.geometry import Point, LineString, MultiLineString
from io import BytesIO

ox.settings.use_cache = True
ox.settings.log_console = False

DRIVE_SPEED = 35  # km/h
MAP_EXTENT = 1200   # reduced for performance
GRAPH_RADIUS = 2200  # reduced from 3000


# ------------------------------------------------------------
# MAIN GENERATOR (UPDATED + OPTIMIZED)
# ------------------------------------------------------------

def generate_driving(lon: float, lat: float, ZONE_DATA: gpd.GeoDataFrame):

    # --------------------------------------------------------
    # SITE POINT
    # --------------------------------------------------------

    site_point_wgs = Point(lon, lat)

    site_point = gpd.GeoSeries(
        [site_point_wgs],
        crs=4326
    ).to_crs(3857).iloc[0]

    # --------------------------------------------------------
    # ZONING
    # --------------------------------------------------------

    ozp = ZONE_DATA.to_crs(3857)

    matching = ozp[ozp.contains(site_point)]

    if matching.empty:
        raise ValueError("Site not within zoning dataset.")

    site_polygon = matching.geometry.iloc[0]
    site_gdf = gpd.GeoDataFrame(geometry=[site_polygon], crs=3857)
    centroid = site_polygon.centroid

    # --------------------------------------------------------
    # DRIVE NETWORK (REDUCED RADIUS)
    # --------------------------------------------------------

    G = ox.graph_from_point(
        (lat, lon),
        dist=GRAPH_RADIUS,
        network_type="drive"
    )

    # Precompute travel time once
    meters_per_min = DRIVE_SPEED * 1000 / 60

    for _, _, _, data in G.edges(keys=True, data=True):
        data["travel_time"] = data["length"] / meters_per_min

    site_node = ox.distance.nearest_nodes(G, lon, lat)

    # --------------------------------------------------------
    # FETCH STATIONS (REDUCED DIST)
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon),
        tags={"railway": "station"},
        dist=GRAPH_RADIUS
    ).to_crs(3857)

    if stations.empty:
        raise ValueError("No nearby stations found.")

    stations["dist"] = stations.centroid.distance(centroid)
    stations = stations.sort_values("dist").head(3)

    # --------------------------------------------------------
    # ROUTE FUNCTION
    # --------------------------------------------------------

    def get_route(node_from, node_to):
        try:
            route = nx.shortest_path(G, node_from, node_to, weight="travel_time")
            return ox.routing.route_to_gdf(G, route).to_crs(3857)
        except:
            return None

    def add_route_arrow(ax, gdf_route, color):

        if gdf_route is None or gdf_route.empty:
            return

        merged = gdf_route.geometry.union_all()

        if isinstance(merged, MultiLineString):
            merged = max(list(merged.geoms), key=lambda g: g.length)

        if not isinstance(merged, LineString):
            return

        coords = list(merged.coords)
        if len(coords) < 2:
            return

        idx = int(len(coords) * 0.6)

        ax.annotate(
            "",
            xy=coords[idx + 1],
            xytext=coords[idx],
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2,
                mutation_scale=16
            ),
            zorder=20
        )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 11))  # slightly smaller

    fig.patch.set_facecolor("#f2f2f2")
    ax.set_facecolor("#f2f2f2")

    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        zoom=16,   # reduced zoom slightly
        alpha=1.0
    )

    # Plot road network lightly
    edges = ox.graph_to_gdfs(G, nodes=False).to_crs(3857)
    edges.plot(ax=ax, linewidth=0.5, color="#8f8f8f", alpha=0.3, zorder=1)

    # Rings
    ring_distances = [350, 700, 1050]

    for d in ring_distances[::-1]:
        gpd.GeoSeries([centroid.buffer(d)], crs=3857).plot(
            ax=ax,
            color="#f4d03f",
            alpha=0.06,
            zorder=2
        )

    for d in ring_distances:
        gpd.GeoSeries([centroid.buffer(d)], crs=3857).boundary.plot(
            ax=ax,
            color="#c8a600",
            linewidth=1.6,
            linestyle=(0, (6, 5)),
            zorder=4
        )

    # --------------------------------------------------------
    # ROUTES
    # --------------------------------------------------------

    for _, station in stations.iterrows():

        st_centroid = station.geometry.centroid
        st_wgs = gpd.GeoSeries([st_centroid], crs=3857).to_crs(4326).iloc[0]
        station_node = ox.distance.nearest_nodes(G, st_wgs.x, st_wgs.y)

        ingress = get_route(station_node, site_node)
        egress = get_route(site_node, station_node)

        if ingress is not None:
            ingress.plot(ax=ax, linewidth=2.2, color="#e74c3c", zorder=10)
            add_route_arrow(ax, ingress, "#e74c3c")

        if egress is not None:
            egress.plot(ax=ax, linewidth=2.2, color="#27ae60", zorder=10)
            add_route_arrow(ax, egress, "#27ae60")

        station_geom = station.geometry
        if station_geom.geom_type == "Point":
            station_geom = station_geom.buffer(55)

        gpd.GeoSeries([station_geom], crs=3857).plot(
            ax=ax,
            facecolor="#5dade2",
            edgecolor="#2e86c1",
            linewidth=1.2,
            alpha=0.6,
            zorder=7
        )

        name = station.get("name:en") or station.get("name") or "STATION"

        ax.text(
            st_centroid.x,
            st_centroid.y + 110,
            name.upper(),
            fontsize=8.5,
            weight="bold",
            ha="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
            zorder=8
        )

    # --------------------------------------------------------
    # SITE
    # --------------------------------------------------------

    site_gdf.plot(
        ax=ax,
        facecolor="#ff4d4d",
        edgecolor="none",
        alpha=0.45,
        zorder=9
    )

    ax.text(
        centroid.x,
        centroid.y - 60,
        "SITE",
        color="black",
        weight="bold",
        ha="center",
        fontsize=10,
        zorder=10
    )

    # Legend
    ingress_line = mlines.Line2D([], [], color="#e74c3c", linewidth=2.2, label="Ingress")
    egress_line = mlines.Line2D([], [], color="#27ae60", linewidth=2.2, label="Egress")

    ax.legend(
        handles=[ingress_line, egress_line],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=9
    )

    ax.set_xlim(centroid.x - MAP_EXTENT, centroid.x + MAP_EXTENT)
    ax.set_ylim(centroid.y - MAP_EXTENT, centroid.y + MAP_EXTENT)

    ax.set_title(
        "SITE ANALYSIS - Driving Distance",
        fontsize=14,
        weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=130)  # reduced DPI
    plt.close(fig)

    return buffer
