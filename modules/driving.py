import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.lines as mlines

from shapely.geometry import Point, LineString, MultiLineString
from io import BytesIO

# ✅ IMPORT UNIVERSAL RESOLVER
from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache = True
ox.settings.log_console = False

DRIVE_SPEED = 35  # km/h
MAP_EXTENT = 1400


# ------------------------------------------------------------
# MAIN GENERATOR
# ------------------------------------------------------------

def generate_driving(data_type: str, value: str, ZONE_DATA: gpd.GeoDataFrame):

    # ✅ Dynamic resolver
    lon, lat = resolve_location(data_type, value)

    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_polygon = lot_gdf.geometry.iloc[0]
        site_gdf = lot_gdf
        centroid = site_polygon.centroid
    else:
        site_point = gpd.GeoSeries(
            [Point(lon, lat)],
            crs=4326
        ).to_crs(3857).iloc[0]

        # --------------------------------------------------------
        # SITE POLYGON FROM PRELOADED ZONE DATA
        # --------------------------------------------------------

        ozp = ZONE_DATA.to_crs(3857)
        matching = ozp[ozp.contains(site_point)]

        if matching.empty:
            raise ValueError("Site not within zoning dataset.")

        site_polygon = matching.geometry.iloc[0]
        site_gdf = gpd.GeoDataFrame(geometry=[site_polygon], crs=3857)
        centroid = site_polygon.centroid

    # --------------------------------------------------------
    # DRIVE NETWORK
    # --------------------------------------------------------

    G = ox.graph_from_point((lat, lon), dist=3000, network_type="drive")

    for u, v, k, data in G.edges(keys=True, data=True):
        data["travel_time"] = data["length"] / (DRIVE_SPEED * 1000 / 60)

    site_node = ox.distance.nearest_nodes(G, lon, lat)

    # --------------------------------------------------------
    # FETCH STATIONS
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon),
        tags={"railway": "station"},
        dist=3000
    ).to_crs(3857)

    if stations.empty:
        raise ValueError("No nearby stations found.")

    stations["dist"] = stations.centroid.distance(centroid)
    stations = stations.sort_values("dist").head(3)

    # --------------------------------------------------------
    # ROUTE FUNCTIONS
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
                mutation_scale=18
            ),
            zorder=20
        )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor("#f2f2f2")
    ax.set_facecolor("#f2f2f2")

    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        zoom=17,
        alpha=1.0
    )

    edges = ox.graph_to_gdfs(G, nodes=False).to_crs(3857)
    edges.plot(ax=ax, linewidth=0.6, color="#8f8f8f", alpha=0.35, zorder=1)

    # Rings
    ring1 = centroid.buffer(375)
    ring2 = centroid.buffer(750)
    ring3 = centroid.buffer(1125)

    for ring, alpha in zip([ring3, ring2, ring1], [0.05, 0.07, 0.10]):
        gpd.GeoSeries([ring], crs=3857).plot(
            ax=ax,
            color="#f4d03f",
            alpha=alpha,
            zorder=2
        )

    for ring in [ring3, ring2, ring1]:
        gpd.GeoSeries([ring], crs=3857).boundary.plot(
            ax=ax,
            color="#c8a600",
            linewidth=2,
            linestyle=(0, (6, 5)),
            zorder=5
        )

    # Routes
    for _, station in stations.iterrows():

        st_centroid = station.geometry.centroid
        st_wgs = gpd.GeoSeries([st_centroid], crs=3857).to_crs(4326).iloc[0]
        station_node = ox.distance.nearest_nodes(G, st_wgs.x, st_wgs.y)

        ingress = get_route(station_node, site_node)
        egress = get_route(site_node, station_node)

        if ingress is not None:
            ingress.plot(ax=ax, linewidth=2.5, color="#e74c3c", zorder=10)
            add_route_arrow(ax, ingress, "#e74c3c")

        if egress is not None:
            egress.plot(ax=ax, linewidth=2.5, color="#27ae60", zorder=10)
            add_route_arrow(ax, egress, "#27ae60")

        station_geom = station.geometry
        if station_geom.geom_type == "Point":
            station_geom = station_geom.buffer(60)

        gpd.GeoSeries([station_geom], crs=3857).plot(
            ax=ax,
            facecolor="#5dade2",
            edgecolor="#2e86c1",
            linewidth=1.5,
            alpha=0.6,
            zorder=7
        )

        name = station.get("name:en") or station.get("name") or "STATION"

        ax.text(
            st_centroid.x,
            st_centroid.y + 120,
            name.upper(),
            fontsize=9,
            weight="bold",
            ha="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
            zorder=8
        )

    # Site
    site_gdf.plot(
        ax=ax,
        facecolor="#ff4d4d",
        edgecolor="none",
        alpha=0.45,
        zorder=9
    )

    ax.text(
        centroid.x,
        centroid.y - 70,
        "SITE",
        color="black",
        weight="bold",
        ha="center",
        fontsize=11,
        zorder=10
    )

    ingress_line = mlines.Line2D([], [], color="#e74c3c", linewidth=2.5, label="Ingress Route")
    egress_line = mlines.Line2D([], [], color="#27ae60", linewidth=2.5, label="Egress Route")

    ax.legend(
        handles=[ingress_line, egress_line],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black"
    )

    ax.set_xlim(centroid.x - MAP_EXTENT, centroid.x + MAP_EXTENT)
    ax.set_ylim(centroid.y - MAP_EXTENT, centroid.y + MAP_EXTENT)

    # ✅ UPDATED TITLE
    ax.set_title(
        f"SITE ANALYSIS - Driving Distance ({data_type} {value})",
        fontsize=16,
        weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=200)
    plt.close(fig)

    return buffer
