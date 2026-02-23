import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
import matplotlib.patches as mpatches
import numpy as np
import textwrap

from shapely.geometry import Point
from sklearn.cluster import KMeans
from io import BytesIO

ox.settings.use_cache = True
ox.settings.log_console = False

FETCH_RADIUS = 1500
MAP_HALF_SIZE = 900
MTR_COLOR = "#ffd166"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def wrap_label(text, width=18):
    return "\n".join(textwrap.wrap(str(text), width))

def infer_site_type(zone):
    if zone.startswith("R"): return "RESIDENTIAL"
    if zone.startswith("C"): return "COMMERCIAL"
    if zone.startswith("G"): return "INSTITUTIONAL"
    if "HOTEL" in zone.upper() or zone.startswith("OU"): return "HOTEL"
    return "MIXED"

def context_rules(site_type):
    if site_type == "RESIDENTIAL":
        return {"amenity":["school","college","university"],"leisure":["park"],"place":["neighbourhood"]}
    if site_type == "COMMERCIAL":
        return {"amenity":["bank","restaurant","market"],"railway":["station"]}
    if site_type == "INSTITUTIONAL":
        return {"amenity":["school","college","hospital"],"leisure":["park"]}
    return {"amenity":True,"leisure":True}

# ------------------------------------------------------------
# MAIN GENERATOR (UPDATED)
# ------------------------------------------------------------

def generate_context(lon: float, lat: float, ZONE_DATA: gpd.GeoDataFrame):

    site_point = gpd.GeoSeries(
        [Point(lon, lat)],
        crs=4326
    ).to_crs(3857).iloc[0]

    # --------------------------------------------------------
    # ZONING
    # --------------------------------------------------------

    ozp = ZONE_DATA.to_crs(3857)

    primary = ozp[ozp.contains(site_point)]

    if primary.empty:
        raise ValueError("No zoning polygon found.")

    primary = primary.iloc[0]
    zone = primary["ZONE_LABEL"]
    SITE_TYPE = infer_site_type(zone)
    LABEL_RULES = context_rules(SITE_TYPE)

    # --------------------------------------------------------
    # FETCH OSM DATA
    # --------------------------------------------------------

    polygons = ox.features_from_point(
        (lat, lon),
        dist=FETCH_RADIUS,
        tags={"landuse":True,"leisure":True,"amenity":True,"building":True}
    ).to_crs(3857)

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
        site_geom = (
            candidates.assign(area=candidates.area)
            .sort_values("area", ascending=False)
            .geometry.iloc[0]
        )
    else:
        site_geom = site_point.buffer(40)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # --------------------------------------------------------
    # MTR STATIONS
    # --------------------------------------------------------

    stations = ox.features_from_point(
        (lat, lon),
        tags={"railway":"station"},
        dist=2000
    ).to_crs(3857)

    if not stations.empty:
        stations["name"] = stations.get("name:en").fillna(stations.get("name"))
        stations["centroid"] = stations.geometry.centroid
        stations["dist"] = stations["centroid"].distance(site_point)
        stations = stations.dropna(subset=["name"]).sort_values("dist").head(2)

    # --------------------------------------------------------
    # BUS STOPS CLUSTERING
    # --------------------------------------------------------

    bus_stops = ox.features_from_point(
        (lat, lon),
        tags={"highway":"bus_stop"},
        dist=900
    ).to_crs(3857)

    if len(bus_stops) > 6:
        coords_array = np.array([[g.x,g.y] for g in bus_stops.geometry])
        bus_stops["cluster"] = KMeans(n_clusters=6, random_state=0).fit(coords_array).labels_
        bus_stops = bus_stops.groupby("cluster").first()

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    labels = ox.features_from_point(
        (lat, lon),
        dist=800,
        tags=LABEL_RULES
    ).to_crs(3857)

    labels["label"] = labels.get("name:en").fillna(labels.get("name"))
    labels = labels.dropna(subset=["label"]).drop_duplicates("label").head(24)

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12,12))

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=16, alpha=0.95)

    ax.set_xlim(site_point.x-MAP_HALF_SIZE, site_point.x+MAP_HALF_SIZE)
    ax.set_ylim(site_point.y-MAP_HALF_SIZE, site_point.y+MAP_HALF_SIZE)
    ax.set_aspect("equal")
    ax.autoscale(False)

    residential.plot(ax=ax,color="#f2c6a0",alpha=0.75)
    industrial.plot(ax=ax,color="#b39ddb",alpha=0.75)
    parks.plot(ax=ax,color="#b7dfb9",alpha=0.9)
    schools.plot(ax=ax,color="#9ecae1",alpha=0.9)
    buildings.plot(ax=ax,color="#d9d9d9",alpha=0.35)

    if not bus_stops.empty:
        bus_stops.plot(ax=ax,color="#0d47a1",markersize=35,zorder=9)

    if not stations.empty:
        stations.plot(ax=ax,facecolor=MTR_COLOR,edgecolor="none",alpha=0.9,zorder=10)

    site_gdf.plot(ax=ax,facecolor="#e53935",edgecolor="darkred",linewidth=2,zorder=11)

    ax.text(site_geom.centroid.x, site_geom.centroid.y,"SITE",
            color="white",weight="bold",ha="center",va="center",zorder=12)

    # --------------------------------------------------------
    # INFO BOX (UPDATED)
    # --------------------------------------------------------

    ax.text(
        0.015,0.985,
        f"OZP Plan: {primary['PLAN_NO']}\n"
        f"Zoning: {zone}\n"
        f"Site Type: {SITE_TYPE}\n",
        transform=ax.transAxes,
        ha="left",va="top",fontsize=9.2,
        bbox=dict(facecolor="white",edgecolor="black",pad=6)
    )

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

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
        bbox_to_anchor=(0.02,0.08),
        fontsize=8.5,
        framealpha=0.95
    )

    ax.set_title("Automated Site Context Analysis (Building-Type Driven)",fontsize=15,weight="bold")
    ax.set_axis_off()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=130)  # reduced resolution for speed
    plt.close(fig)

    return buffer
