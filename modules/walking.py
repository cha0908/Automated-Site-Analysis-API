def generate_walking(lon: float, lat: float):

    # --------------------------------------------------------
    # LOCAL IMPORTS (reduce base memory usage)
    # --------------------------------------------------------

    import osmnx as ox
    import geopandas as gpd
    import networkx as nx
    import matplotlib.pyplot as plt
    import contextily as cx
    from shapely.geometry import Point
    from io import BytesIO

    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.timeout = 60

    # --------------------------------------------------------
    # OPTIMIZED SETTINGS (FREE TIER SAFE)
    # --------------------------------------------------------

    WALK_SPEED_KMPH = 5
    GRAPH_RADIUS = 1000     
    STATION_RADIUS = 1000    
    MAP_EXTENT = 900         

    # --------------------------------------------------------
    # SITE FOOTPRINT
    # --------------------------------------------------------

    try:
        osm_site = ox.features_from_point(
            (lat, lon),
            dist=50,
            tags={"building": True}
        ).to_crs(3857)

        if len(osm_site):
            osm_site["area_calc"] = osm_site.geometry.area
            site_geom = osm_site.sort_values(
                "area_calc", ascending=False
            ).geometry.iloc[0]
        else:
            raise Exception()

    except:
        site_geom = gpd.GeoSeries(
            [Point(lon, lat)],
            crs=4326
        ).to_crs(3857).iloc[0].buffer(35)

    site_point = site_geom.centroid

    # --------------------------------------------------------
    # WALK GRAPH (SIMPLIFIED)
    # --------------------------------------------------------

    G_walk = ox.graph_from_point(
        (lat, lon),
        dist=GRAPH_RADIUS,
        network_type="walk",
        simplify=True,
        retain_all=False
    )

    # Get nearest node
    site_node = ox.distance.nearest_nodes(G_walk, lon, lat)

    # --------------------------------------------------------
    # FETCH STATIONS (LIMITED)
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon),
        tags={"railway": "station"},
        dist=STATION_RADIUS
    )

    if stations.empty:
        raise ValueError("No nearby stations found.")

    stations = stations.to_crs(3857)
    stations = stations.head(3)   # 🔥 limit early

    # --------------------------------------------------------
    # ROUTE CALCULATION
    # --------------------------------------------------------

    routes = []

    for _, row in stations.iterrows():

        try:
            st_lon, st_lat = row.geometry.centroid.to_crs(4326).x, row.geometry.centroid.to_crs(4326).y
            st_node = ox.distance.nearest_nodes(G_walk, st_lon, st_lat)

            path = nx.shortest_path(
                G_walk,
                site_node,
                st_node,
                weight="length"
            )

            route_length = sum(
                ox.utils_graph.get_route_edge_attributes(G_walk, path, "length")
            )

            dist_km = round(route_length / 1000, 2)
            time_min = max(1, round((dist_km / WALK_SPEED_KMPH) * 60))

            routes.append({
                "path": path,
                "distance": dist_km,
                "time": time_min
            })

        except:
            continue

    # --------------------------------------------------------
    # PLOTTING (LIGHTWEIGHT)
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 9))

    # Draw walk network directly (no GeoDataFrame conversion)
    ox.plot_graph(
        G_walk,
        ax=ax,
        node_size=0,
        edge_color="#8a8a8a",
        edge_linewidth=0.3,
        show=False,
        close=False
    )

    # Draw walking rings
    gpd.GeoSeries([site_point.buffer(375)], crs=3857).boundary.plot(
        ax=ax, linestyle="--", linewidth=1, color="#2aa9ff"
    )
    gpd.GeoSeries([site_point.buffer(750)], crs=3857).boundary.plot(
        ax=ax, linestyle="--", linewidth=1, color="#2aa9ff"
    )

    # Plot routes
    route_colors = ["#4caf50", "#ef5350", "#42a5f5"]

    for i, r in enumerate(routes):
        route_color = route_colors[i % len(route_colors)]

        ox.plot_graph_route(
            G_walk,
            r["path"],
            ax=ax,
            route_color=route_color,
            route_linewidth=2.5,
            node_size=0,
            show=False,
            close=False
        )

    # Plot site
    gpd.GeoSeries([site_geom], crs=3857).plot(
        ax=ax,
        facecolor="red",
        edgecolor="none",
        zorder=5
    )

    ax.text(
        site_point.x,
        site_point.y - 80,
        "SITE",
        color="red",
        weight="bold",
        ha="center"
    )

    ax.set_xlim(site_point.x - MAP_EXTENT, site_point.x + MAP_EXTENT)
    ax.set_ylim(site_point.y - MAP_EXTENT, site_point.y + MAP_EXTENT)

    # Optional: comment this out if memory still high
    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        zoom=13,
        alpha=0.4
    )

    ax.set_title("Walking Accessibility", fontsize=13, weight="bold")
    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=120)   # 🔥 reduced DPI
    plt.close(fig)

    buffer.seek(0)
    return buffer
