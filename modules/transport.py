def generate_transport(lon: float, lat: float):

    # --------------------------------------------------------
    # LOCAL IMPORTS (reduce base memory)
    # --------------------------------------------------------

    import osmnx as ox
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import contextily as cx
    from shapely.geometry import Point
    from io import BytesIO

    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 60

    # --------------------------------------------------------
    # OPTIMIZED SETTINGS (FREE TIER SAFE)
    # --------------------------------------------------------

    MAP_RADIUS = 1200        # 🔥 reduced from 2200
    MAP_EXTENT = 1000        # 🔥 reduced extent
    FETCH_LIMIT = 800        # 🔥 hard cap rows

    COLOR_EXISTING_RAIL = "#e06a2b"
    COLOR_MTR = "#3f78b5"
    COLOR_ROADS = "#e85d9e"
    COLOR_WATER = "#6fa8dc"
    COLOR_BUILDINGS = "#d6d6d6"
    COLOR_SITE = "#FF0000"

    # --------------------------------------------------------
    # SITE POINT
    # --------------------------------------------------------

    site_point = gpd.GeoSeries(
        [Point(lon, lat)],
        crs=4326
    ).to_crs(3857).iloc[0]

    # --------------------------------------------------------
    # SAFE FETCH (LIMITED)
    # --------------------------------------------------------

    def safe_fetch(tags):
        try:
            gdf = ox.features_from_point(
                (lat, lon),
                dist=MAP_RADIUS,
                tags=tags
            )
            if gdf.empty:
                return gpd.GeoDataFrame(geometry=[], crs=3857)

            gdf = gdf.head(FETCH_LIMIT)   # 🔥 limit rows early
            return gdf.to_crs(3857)

        except:
            return gpd.GeoDataFrame(geometry=[], crs=3857)

    def keep_lines(gdf):
        if not gdf.empty:
            return gdf[gdf.geometry.type.isin(["LineString","MultiLineString"])]
        return gpd.GeoDataFrame(geometry=[], crs=3857)

    # --------------------------------------------------------
    # FETCH DATA (REDUCED)
    # --------------------------------------------------------

    buildings = safe_fetch({"building": True})
    roads = keep_lines(safe_fetch({"highway":["primary","secondary"]}))  # 🔥 reduced types
    light_rail = keep_lines(safe_fetch({"railway":"light_rail"}))
    stations = safe_fetch({"railway":"station"})
    water = safe_fetch({"natural":"water"})
    mtr_routes = keep_lines(safe_fetch({"railway":["rail","subway"]}))

    # --------------------------------------------------------
    # SITE FOOTPRINT
    # --------------------------------------------------------

    if not buildings.empty:
        buildings["dist"] = buildings.geometry.distance(site_point)
        site_geom = buildings.sort_values("dist").geometry.iloc[0]
    else:
        site_geom = site_point.buffer(35)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # --------------------------------------------------------
    # EXTENT
    # --------------------------------------------------------

    xmin = site_point.x - MAP_EXTENT
    xmax = site_point.x + MAP_EXTENT
    ymin = site_point.y - MAP_EXTENT
    ymax = site_point.y + MAP_EXTENT

    # --------------------------------------------------------
    # PLOT (LIGHTWEIGHT)
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 7))  # 🔥 smaller figure

    ax.set_facecolor("#f4f4f4")

    # Optional: comment if memory still high
    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.Positron,
        zoom=13,        # 🔥 reduced zoom
        alpha=0.5
    )

    # Buildings
    if not buildings.empty:
        buildings.plot(ax=ax, color=COLOR_BUILDINGS, alpha=0.5, linewidth=0)

    # Water
    if not water.empty:
        water.plot(ax=ax, color=COLOR_WATER, alpha=0.8, linewidth=0)

    # Roads
    if not roads.empty:
        roads.plot(ax=ax, color=COLOR_ROADS, linewidth=1.2)

    # MTR
    if not mtr_routes.empty:
        mtr_routes.plot(ax=ax, color="white", linewidth=4)
        mtr_routes.plot(ax=ax, color=COLOR_MTR, linewidth=2.5)

        if "name" in mtr_routes.columns:
            labeled = mtr_routes.dropna(subset=["name"]).head(5)

            for _, row in labeled.iterrows():
                name = str(row["name"])[:20]
                midpoint = row.geometry.interpolate(0.5, normalized=True)

                ax.text(
                    midpoint.x,
                    midpoint.y,
                    name.upper(),
                    fontsize=8,
                    weight="bold",
                    color=COLOR_MTR,
                    ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1)
                )

    # Light Rail
    if not light_rail.empty:
        light_rail.plot(ax=ax, color=COLOR_EXISTING_RAIL, linewidth=2)

    # Stations
    if not stations.empty:
        station_pts = stations.copy()
        station_pts["geometry"] = station_pts.centroid

        station_pts.plot(
            ax=ax,
            facecolor="white",
            edgecolor=COLOR_EXISTING_RAIL,
            markersize=70,
            linewidth=1.5
        )

    # Site
    site_gdf.plot(ax=ax, facecolor=COLOR_SITE, edgecolor="none")

    ax.text(
        site_geom.centroid.x,
        site_geom.centroid.y - 80,
        "SITE",
        fontsize=12,
        weight="bold",
        ha="center"
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title("SITE ANALYSIS – Transportation", fontsize=14, weight="bold")
    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=120)  # 🔥 reduced DPI
    plt.close(fig)

    buffer.seek(0)
    return buffer
