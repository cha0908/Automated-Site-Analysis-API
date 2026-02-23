def generate_driving(lon: float, lat: float, ZONE_DATA):

    # --------------------------------------------------------
    # LOCAL IMPORTS (reduce base memory usage)
    # --------------------------------------------------------

    import osmnx as ox
    import geopandas as gpd
    import networkx as nx
    import matplotlib.pyplot as plt
    import contextily as cx
    import matplotlib.lines as mlines
    from shapely.geometry import Point, LineString, MultiLineString
    from io import BytesIO

    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 60

    # --------------------------------------------------------
    # OPTIMIZED SETTINGS
    # --------------------------------------------------------

    DRIVE_SPEED = 35
    GRAPH_RADIUS = 1200      # 🔥 reduced from 2200
    MAP_EXTENT = 1000        # 🔥 reduced extent

    # --------------------------------------------------------
    # SITE + ZONING
    # --------------------------------------------------------

    site_point_wgs = Point(lon, lat)

    site_point = gpd.GeoSeries(
        [site_point_wgs], crs=4326
    ).to_crs(3857).iloc[0]

    ozp = ZONE_DATA
    matching = ozp[ozp.contains(site_point)]

    if matching.empty:
        raise ValueError("Site not within zoning dataset.")

    site_polygon = matching.geometry.iloc[0]
    centroid = site_polygon.centroid
    site_gdf = gpd.GeoDataFrame(geometry=[site_polygon], crs=3857)

    # --------------------------------------------------------
    # DRIVE NETWORK (SIMPLIFIED)
    # --------------------------------------------------------

    G = ox.graph_from_point(
        (lat, lon),
        dist=GRAPH_RADIUS,
        network_type="drive",
        simplify=True,
        retain_all=False
    )

    meters_per_min = DRIVE_SPEED * 1000 / 60

    for _, _, _, data in G.edges(keys=True, data=True):
        data["travel_time"] = data["length"] / meters_per_min

    site_node = ox.distance.nearest_nodes(G, lon, lat)

    # --------------------------------------------------------
    # FETCH STATIONS (LIMITED)
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon),
        tags={"railway": "station"},
        dist=GRAPH_RADIUS
    )

    if stations.empty:
        raise ValueError("No nearby stations found.")

    stations = stations.to_crs(3857).head(3)

    # --------------------------------------------------------
    # ROUTE FUNCTION
    # --------------------------------------------------------

    def get_route(node_from, node_to):
        try:
            route = nx.shortest_path(G, node_from, node_to, weight="travel_time")
            return route
        except:
            return None

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_facecolor("#f2f2f2")

    # 🔥 Draw network WITHOUT converting to GeoDataFrame
    ox.plot_graph(
        G,
        ax=ax,
        node_size=0,
        edge_color="#8f8f8f",
        edge_linewidth=0.4,
        show=False,
        close=False
    )

    # Basemap (optional – comment if still heavy)
    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        zoom=14,
        alpha=0.6
    )

    # Rings
    for d in [350, 700, 1050]:
        gpd.GeoSeries([centroid.buffer(d)], crs=3857).boundary.plot(
            ax=ax,
            color="#c8a600",
            linewidth=1.2,
            linestyle=(0, (6, 5))
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

        if ingress:
            ox.plot_graph_route(
                G,
                ingress,
                ax=ax,
                route_color="#e74c3c",
                route_linewidth=2.2,
                node_size=0,
                show=False,
                close=False
            )

        if egress:
            ox.plot_graph_route(
                G,
                egress,
                ax=ax,
                route_color="#27ae60",
                route_linewidth=2.2,
                node_size=0,
                show=False,
                close=False
            )

        # Station bubble
        station_geom = station.geometry
        if station_geom.geom_type == "Point":
            station_geom = station_geom.buffer(45)

        gpd.GeoSeries([station_geom], crs=3857).plot(
            ax=ax,
            facecolor="#5dade2",
            edgecolor="#2e86c1",
            alpha=0.6
        )

    # --------------------------------------------------------
    # SITE
    # --------------------------------------------------------

    site_gdf.plot(
        ax=ax,
        facecolor="#ff4d4d",
        alpha=0.45
    )

    ax.text(
        centroid.x,
        centroid.y - 60,
        "SITE",
        weight="bold",
        ha="center"
    )

    ax.set_xlim(centroid.x - MAP_EXTENT, centroid.x + MAP_EXTENT)
    ax.set_ylim(centroid.y - MAP_EXTENT, centroid.y + MAP_EXTENT)

    ax.set_title("SITE ANALYSIS - Driving Distance", fontsize=13, weight="bold")
    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=120)
    plt.close(fig)

    buffer.seek(0)
    return buffer
