import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from shapely.geometry import Point
from io import BytesIO

# ✅ IMPORT UNIVERSAL RESOLVER
from modules.resolver import resolve_location, get_lot_boundary

ox.settings.log_console = False
ox.settings.use_cache = True

WALK_SPEED_KMPH = 5
MAP_EXTENT = 2000


# ------------------------------------------------------------
# MAIN GENERATOR
# ------------------------------------------------------------

def generate_walking(data_type: str, value: str):

    # ✅ Dynamic resolver
    lon, lat = resolve_location(data_type, value)

    # --------------------------------------------------------
    # GET SITE FOOTPRINT (official lot boundary or OSM fallback)
    # --------------------------------------------------------

    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_geom = lot_gdf.geometry.iloc[0]
        site_gdf = lot_gdf
        site_point = site_geom.centroid
    else:
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
    # WALK NETWORK
    # --------------------------------------------------------

    G_walk = ox.graph_from_point(
        (lat, lon),
        dist=1500,
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
    # FETCH STATIONS
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon),
        tags={"railway": "station"},
        dist=3000
    ).to_crs(3857)

    stations = stations[stations.geometry.notnull()]

    if stations.empty:
        raise ValueError("No nearby stations found.")

    def _safe_station_name(r):
        a, b = r.get("name:en"), r.get("name")
        if isinstance(a, str) and a.strip():
            return a.strip()
        if isinstance(b, str) and b.strip():
            return b.strip()
        return "UNNAMED"

    NEAREST_NAMED_STATION_M = 150  # fallback unnamed to this named station if within distance (m)

    stations["station_name"] = stations.apply(_safe_station_name, axis=1)

    # Replace UNNAMED with nearest named station's name if within threshold
    unnamed_mask = stations["station_name"] == "UNNAMED"
    if unnamed_mask.any():
        named = stations[~unnamed_mask]
        for idx in stations.index[unnamed_mask]:
            cent = stations.at[idx, "geometry"].centroid
            if named.empty:
                break
            named_copy = named.copy()
            named_copy["_d"] = named_copy.geometry.centroid.distance(cent)
            nearest = named_copy.loc[named_copy["_d"].idxmin()]
            if nearest["_d"] <= NEAREST_NAMED_STATION_M:
                stations.at[idx, "station_name"] = nearest["station_name"]

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
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 12))

    roads.plot(ax=ax, linewidth=0.25, color="#8a8a8a", alpha=0.4)

    gpd.GeoSeries([site_point.buffer(1125)], crs=3857).plot(
        ax=ax, color="#2aa9ff", alpha=0.15
    )

    for d, lbl in [(375, "5 min"), (750, "10 min"), (1125, "15 min")]:
        gpd.GeoSeries([site_point.buffer(d)], crs=3857).boundary.plot(
            ax=ax,
            linestyle=(0, (4, 3)),
            linewidth=2,
            color="#2aa9ff"
        )
        ax.text(site_point.x + d + 120, site_point.y, lbl, fontsize=9)

    colors = ["#4caf50", "#ef5350", "#42a5f5"]

    for i, r in enumerate(routes):

        route_color = colors[i % len(colors)]

        r["route"].plot(
            ax=ax,
            linewidth=2.8,
            color=route_color,
            alpha=0.85,
            zorder=5
        )

        mid = r["route"].geometry.iloc[len(r["route"]) // 2].centroid

        ax.text(
            mid.x,
            mid.y,
            f"{r['time']} min\n{r['distance']} km",
            fontsize=9,
            weight="bold",
            color=route_color,
            ha="center",
            zorder=6
        )

        if r["station_centroid"].distance(site_point) <= MAP_EXTENT:
            station_geom = r["station_polygon"]

            if station_geom.geom_type == "Point":
                station_geom = station_geom.buffer(60)

            gpd.GeoSeries([station_geom], crs=3857).plot(
                ax=ax,
                facecolor=route_color,
                edgecolor=route_color,
                linewidth=1,
                alpha=0.25,
                zorder=4
            )

            label_name = r["name"]
            if not isinstance(label_name, str) or not str(label_name).strip():
                label_name = "UNNAMED"
            else:
                label_name = str(label_name).strip()
            ax.text(
                r["station_centroid"].x,
                r["station_centroid"].y + 120,
                label_name.upper(),
                fontsize=10,
                weight="bold",
                ha="center",
                zorder=7
            )

    site_gdf.plot(ax=ax, facecolor="red", edgecolor="none")

    ax.text(
        site_point.x,
        site_point.y - 120,
        "SITE",
        color="red",
        weight="bold",
        ha="center"
    )

    ax.set_xlim(site_point.x - MAP_EXTENT, site_point.x + MAP_EXTENT)
    ax.set_ylim(site_point.y - MAP_EXTENT, site_point.y + MAP_EXTENT)
    ax.set_aspect("equal")

    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        zoom=15,
        alpha=0.4
    )

    # ✅ UPDATED TITLE
    ax.set_title(
        f"Walking Accessibility - {data_type} {value}",
        fontsize=15,
        weight="bold"
    )

    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=120)
    plt.close(fig)

    return buffer
