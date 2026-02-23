def generate_context(lon: float, lat: float, ZONE_DATA):

    # --------------------------------------------------------
    # LOCAL IMPORTS (memory safe)
    # --------------------------------------------------------

    import osmnx as ox
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as cx
    import matplotlib.patches as mpatches
    import numpy as np
    import textwrap
    from shapely.geometry import Point
    from io import BytesIO

    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 60

    # --------------------------------------------------------
    # OPTIMIZED SETTINGS
    # --------------------------------------------------------

    FETCH_RADIUS = 1000     # 🔥 reduced
    MAP_HALF_SIZE = 850     # 🔥 reduced
    LABEL_RADIUS = 700      # 🔥 reduced
    MAX_FEATURES = 900      # 🔥 limit rows
    MTR_COLOR = "#ffd166"

    # --------------------------------------------------------
    # SITE POINT
    # --------------------------------------------------------

    site_point = gpd.GeoSeries(
        [Point(lon, lat)],
        crs=4326
    ).to_crs(3857).iloc[0]

    # --------------------------------------------------------
    # ZONING (no repeated to_crs)
    # --------------------------------------------------------

    ozp = ZONE_DATA
    primary = ozp[ozp.contains(site_point)]

    if primary.empty:
        raise ValueError("No zoning polygon found.")

    primary = primary.iloc[0]
    zone = primary["ZONE_LABEL"]

    # --------------------------------------------------------
    # FETCH POLYGONS (LIMITED)
    # --------------------------------------------------------

    polygons = ox.features_from_point(
        (lat, lon),
        dist=FETCH_RADIUS,
        tags={"landuse":True,"leisure":True,"amenity":True,"building":True}
    )

    if polygons.empty:
        polygons = gpd.GeoDataFrame(geometry=[], crs=3857)
    else:
        polygons = polygons.head(MAX_FEATURES).to_crs(3857)

    residential = polygons[polygons.get("landuse")=="residential"]
    industrial  = polygons[polygons.get("landuse").isin(["industrial","commercial"])]
    parks       = polygons[polygons.get("leisure")=="park"]
    schools     = polygons[polygons.get("amenity").isin(["school","college","university"])]
    buildings   = polygons[polygons.get("building").notnull()]

    # --------------------------------------------------------
    # SITE FOOTPRINT
    # --------------------------------------------------------

    candidates = polygons[
        polygons.geometry.geom_type.isin(["Polygon","MultiPolygon"]) &
        (polygons.geometry.distance(site_point) < 40)
    ]

    if len(candidates):
        site_geom = candidates.iloc[0].geometry
    else:
        site_geom = site_point.buffer(40)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # --------------------------------------------------------
    # MTR STATIONS (LIMITED)
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon),
        tags={"railway":"station"},
        dist=FETCH_RADIUS
    )

    if not stations.empty:
        stations = stations.to_crs(3857).head(2)

    # --------------------------------------------------------
    # BUS STOPS (NO KMEANS – lighter)
    # --------------------------------------------------------

    bus_stops = ox.features_from_point(
        (lat, lon),
        tags={"highway":"bus_stop"},
        dist=700
    )

    if not bus_stops.empty:
        bus_stops = bus_stops.to_crs(3857).head(8)

    # --------------------------------------------------------
    # LABELS (FIXED + DRAWN)
    # --------------------------------------------------------

    labels = ox.features_from_point(
        (lat, lon),
        dist=LABEL_RADIUS,
        tags={"amenity":True,"leisure":True}
    )

    if not labels.empty:
        labels = labels.to_crs(3857)
        labels["label"] = labels.get("name:en").fillna(labels.get("name"))
        labels = labels.dropna(subset=["label"]).head(20)

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10,10))  # 🔥 smaller

    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.Positron,
        zoom=14,     # 🔥 reduced zoom
        alpha=0.9
    )

    ax.set_xlim(site_point.x-MAP_HALF_SIZE, site_point.x+MAP_HALF_SIZE)
    ax.set_ylim(site_point.y-MAP_HALF_SIZE, site_point.y+MAP_HALF_SIZE)
    ax.set_aspect("equal")

    residential.plot(ax=ax,color="#f2c6a0",alpha=0.75)
    industrial.plot(ax=ax,color="#b39ddb",alpha=0.75)
    parks.plot(ax=ax,color="#b7dfb9",alpha=0.9)
    schools.plot(ax=ax,color="#9ecae1",alpha=0.9)
    buildings.plot(ax=ax,color="#d9d9d9",alpha=0.3)

    if not bus_stops.empty:
        bus_stops.plot(ax=ax,color="#0d47a1",markersize=30,zorder=9)

    if not stations.empty:
        stations.plot(ax=ax,facecolor=MTR_COLOR,edgecolor="none",alpha=0.9,zorder=10)

    site_gdf.plot(ax=ax,facecolor="#e53935",edgecolor="darkred",linewidth=2,zorder=11)

    ax.text(
        site_geom.centroid.x,
        site_geom.centroid.y,
        "SITE",
        color="white",
        weight="bold",
        ha="center",
        va="center",
        zorder=12
    )

    # 🔥 DRAW LABEL TEXT
    if not labels.empty:
        for _, row in labels.iterrows():
            ax.text(
                row.geometry.centroid.x,
                row.geometry.centroid.y,
                str(row["label"])[:18],
                fontsize=7,
                color="black",
                bbox=dict(facecolor="white",alpha=0.7,pad=1),
                zorder=13
            )

    # INFO BOX
    ax.text(
        0.015,0.985,
        f"OZP Plan: {primary['PLAN_NO']}\nZoning: {zone}",
        transform=ax.transAxes,
        ha="left",va="top",fontsize=9,
        bbox=dict(facecolor="white",edgecolor="black",pad=5)
    )

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
        fontsize=8,
        framealpha=0.95
    )

    ax.set_title("Automated Site Context Analysis",fontsize=13,weight="bold")
    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=120)  # 🔥 reduced DPI
    plt.close(fig)

    buffer.seek(0)
    return buffer
