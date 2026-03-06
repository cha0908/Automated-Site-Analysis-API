from typing import Optional, List
import matplotlib
matplotlib.use("Agg")

import os
import gc
import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.legend_handler as lh
import numpy as np

from shapely.geometry import Point, box
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# internal modules

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache = True
ox.settings.log_console = False

MAP_RADIUS = 3000

COLOR_ROADS = "#e85d9e"
COLOR_WATER = "#6fa8dc"
COLOR_BUILDINGS = "#d6d6d6"
COLOR_SITE = "#FF0000"
COLOR_LIGHT_RAIL = "#D3A809"

STATION_MIN_DISTANCE = 250
STATION_LOGO_ZOOM = 0.055

# ------------------------------------------------------------

# MTR LINE COLORS

# ------------------------------------------------------------

MTR_LINE_COLORS = {
"island": "#007DC5",
"kwun tong": "#00AB4E",
"tsuen wan": "#ED1D24",
"tseung kwan o": "#7D499D",
"tung chung": "#F7943E",
"east rail": "#5EB6E4",
"tuen ma": "#923011",
"south island": "#BAC429",
"airport express": "#888B8D",
}

DEFAULT_MTR_COLOR = "#3f78b5"

MTR_LEGEND_LINES = [
("island", "#007DC5", "Island Line"),
("kwun tong", "#00AB4E", "Kwun Tong Line"),
("tsuen wan", "#ED1D24", "Tsuen Wan Line"),
("tseung kwan o", "#7D499D", "Tseung Kwan O Line"),
("tung chung", "#F7943E", "Tung Chung Line"),
("east rail", "#5EB6E4", "East Rail Line"),
("tuen ma", "#923011", "Tuen Ma Line"),
("south island", "#BAC429", "South Island Line"),
("airport express", "#888B8D", "Airport Express"),
]

def get_mtr_color(name: str):
name = name.lower()
for k, v in MTR_LINE_COLORS.items():
if k in name:
return v
return DEFAULT_MTR_COLOR

# ------------------------------------------------------------

# LOAD MTR LOGO

# ------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(**file**), "..", "static")
MTR_LOGO_PATH = os.path.join(STATIC_DIR, "HK_MTR_logo.png")

try:
MTR_LOGO = mpimg.imread(MTR_LOGO_PATH)
LOGO_LOADED = True
except:
MTR_LOGO = None
LOGO_LOADED = False

# ------------------------------------------------------------

# DRAW STATION ICON

# ------------------------------------------------------------

def draw_station(ax, x, y, zoom=STATION_LOGO_ZOOM, fallback="#ED1D24"):

```
if LOGO_LOADED and MTR_LOGO is not None:

    img = OffsetImage(MTR_LOGO, zoom=zoom)
    img.image.axes = ax

    ab = AnnotationBbox(
        img,
        (x, y),
        frameon=False,
        zorder=9
    )

    ax.add_artist(ab)

else:

    ax.add_patch(
        plt.Circle((x, y), 60, color=fallback, zorder=9)
    )
```

# ------------------------------------------------------------

# SAFE FETCH

# ------------------------------------------------------------

def safe_fetch(lat, lon, radius, tags):

```
try:
    gdf = ox.features_from_point((lat, lon), dist=radius, tags=tags)

    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=3857)

    return gdf.to_crs(3857)

except:
    return gpd.GeoDataFrame(geometry=[], crs=3857)
```

def keep_lines(gdf):

```
if gdf.empty:
    return gpd.GeoDataFrame(geometry=[], crs=3857)

return gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
```

# ------------------------------------------------------------

# MAIN FUNCTION

# ------------------------------------------------------------

def generate_transport(
data_type: str,
value: str,
lon: float = None,
lat: float = None,
lot_ids: List[str] = None,
extents: List[dict] = None):

```
lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)

lot_gdf = get_lot_boundary(lon, lat, data_type, extents)

if lot_gdf is not None:

    site_geom = lot_gdf.geometry.iloc[0]
    site_gdf = lot_gdf
    site_point = site_geom.centroid

else:

    site_point = gpd.GeoSeries(
        [Point(lon, lat)], crs=4326
    ).to_crs(3857).iloc[0]


# --------------------------------------------------------
# FETCH OSM DATA
# --------------------------------------------------------

buildings = safe_fetch(lat, lon, MAP_RADIUS, {"building": True})

roads = keep_lines(
    safe_fetch(lat, lon, MAP_RADIUS,
               {"highway": ["motorway", "trunk", "primary", "secondary"]})
)

light_rail = keep_lines(
    safe_fetch(lat, lon, MAP_RADIUS, {"railway": "light_rail"})
)

stations = safe_fetch(lat, lon, MAP_RADIUS, {"railway": "station"})

water = safe_fetch(lat, lon, MAP_RADIUS, {"natural": "water"})

mtr_routes = keep_lines(
    safe_fetch(lat, lon, MAP_RADIUS, {"railway": ["rail", "subway"]})
)

gc.collect()


# --------------------------------------------------------
# SITE GEOMETRY
# --------------------------------------------------------

if lot_gdf is None:

    if not buildings.empty:

        distances = buildings.geometry.distance(site_point)
        nearest = distances.idxmin()
        site_geom = buildings.loc[nearest, "geometry"]

    else:

        site_geom = site_point.buffer(40)

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)


# --------------------------------------------------------
# MAP EXTENT
# --------------------------------------------------------

xmin = site_point.x - 1900
xmax = site_point.x + 1900
ymin = site_point.y - 1100
ymax = site_point.y + 1100

clip_box = box(xmin, ymin, xmax, ymax)
clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs=3857)


# --------------------------------------------------------
# PLOT
# --------------------------------------------------------

fig, ax = plt.subplots(figsize=(18, 10))

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

cx.add_basemap(
    ax,
    crs="EPSG:3857",
    source=cx.providers.CartoDB.Positron,
    zoom=15,
    alpha=0.5
)


if not buildings.empty:
    buildings.plot(ax=ax, color=COLOR_BUILDINGS, alpha=0.5)

if not water.empty:
    water.plot(ax=ax, color=COLOR_WATER, alpha=0.8)

if not roads.empty:
    roads.plot(ax=ax, color=COLOR_ROADS, linewidth=2.2)


# --------------------------------------------------------
# MTR ROUTES
# --------------------------------------------------------

if not mtr_routes.empty:

    routes = gpd.clip(mtr_routes, clip_gdf)

    if "name" in routes.columns:

        for name in routes["name"].dropna().unique():

            subset = routes[routes["name"] == name]

            color = get_mtr_color(name)

            subset.plot(ax=ax, color="white", linewidth=8)
            subset.plot(ax=ax, color=color, linewidth=4)


# --------------------------------------------------------
# LIGHT RAIL
# --------------------------------------------------------

if not light_rail.empty:

    light_rail.plot(ax=ax, color="white", linewidth=6)
    light_rail.plot(ax=ax, color=COLOR_LIGHT_RAIL, linewidth=3.5)


# --------------------------------------------------------
# STATIONS
# --------------------------------------------------------

if not stations.empty:

    stations = stations.copy()
    stations["geometry"] = stations.centroid

    for _, row in stations.iterrows():

        x = row.geometry.x
        y = row.geometry.y

        if xmin < x < xmax and ymin < y < ymax:

            draw_station(ax, x, y)


# --------------------------------------------------------
# SITE
# --------------------------------------------------------

site_gdf.plot(ax=ax, color=COLOR_SITE)

c = site_geom.centroid

ax.text(
    c.x,
    c.y - 120,
    "SITE",
    fontsize=14,
    weight="bold",
    ha="center"
)


# --------------------------------------------------------
# NORTH ARROW
# --------------------------------------------------------

ax.annotate(
    '',
    xy=(0.07, 0.85),
    xytext=(0.07, 0.80),
    xycoords=ax.transAxes,
    arrowprops=dict(
        facecolor='black',
        width=1.5,
        headwidth=8,
        headlength=10
    )
)

ax.text(
    0.07,
    0.86,
    "N",
    transform=ax.transAxes,
    ha='center'
)


ax.set_axis_off()

buffer = BytesIO()

plt.savefig(
    buffer,
    format="png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

buffer.seek(0)

return buffer
```
