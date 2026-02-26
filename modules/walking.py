import matplotlib
matplotlib.use("Agg")

import os
import gc
import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import numpy as np

from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from shapely.geometry import Point
from io import BytesIO

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.log_console = False
ox.settings.use_cache   = True   # cache OSM downloads to disk → avoids re-fetching

# ── Memory: restrict matplotlib thread usage ─────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ── Ring configs ──────────────────────────────────────────────
RING_CONFIGS = {
    5: {
        "rings":      [(83,  "1 min\n0.08 km"),
                       (250, "3 min\n0.25 km"),
                       (400, "5 min\n0.40 km")],
        "shade_r":    400,
        "map_extent": 600,
        "walk_dist":  600,    # ↓ smaller fetch = less RAM
    },
    10: {
        "rings":      [(250, "3 min\n0.25 km"),
                       (500, "6 min\n0.50 km"),
                       (750, "10 min\n0.75 km")],
        "shade_r":    750,
        "map_extent": 875,
        "walk_dist":  900,    # ↓ trimmed
    },
    15: {
        "rings":      [(375,  "5 min\n0.375 km"),
                       (750,  "10 min\n0.75 km"),
                       (1125, "15 min\n1.125 km")],
        "shade_r":    1125,
        "map_extent": 1300,
        "walk_dist":  1400,   # ↓ trimmed from 1800
    },
}

WALK_SPEED_KMPH = 5

# ── MTR logo ──────────────────────────────────────────────────
_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
try:
    _mtr_img = mpimg.imread(_MTR_LOGO_PATH)
except Exception:
    _mtr_img = None


def _add_mtr_icon(ax, x, y, size=0.04, zorder=15):
    if _mtr_img is not None:
        icon = OffsetImage(_mtr_img, zoom=size)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False, zorder=zorder,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        r = size * 1400
        ax.add_patch(Circle((x, y), r, facecolor="#C01933", edgecolor="white",
                            linewidth=r*0.05, zorder=zorder,
                            transform=ax.transData, clip_on=True))
        lw = r * 0.22
        ax.add_patch(Circle((x, y + r*0.52), r*0.15, facecolor="white",
                            edgecolor="none", zorder=zorder+1,
                            transform=ax.transData))
        for sign in [-1, 1]:
            ax.plot([x, x + sign*r*0.42], [y + r*0.10, y - r*0.52],
                    color="white", linewidth=lw, solid_capstyle="round",
                    zorder=zorder+1, transform=ax.transData)
        ax.plot([x, x], [y + r*0.35, y + r*0.10],
                color="white", linewidth=lw, solid_capstyle="round",
                zorder=zorder+1, transform=ax.transData)


# ── Main generator ────────────────────────────────────────────

def generate_walking(data_type: str, value: str,
                     max_walk_minutes: int = 15):

    if max_walk_minutes not in RING_CONFIGS:
        max_walk_minutes = 15
    cfg        = RING_CONFIGS[max_walk_minutes]
    MAP_EXTENT = cfg["map_extent"]

    lon, lat = resolve_location(data_type, value)

    # ── Site footprint ────────────────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_geom  = lot_gdf.geometry.iloc[0]
        site_gdf   = lot_gdf
        site_point = site_geom.centroid
    else:
        try:
            osm_site = ox.features_from_point(
                (lat, lon), dist=60, tags={"building": True}
            ).to_crs(3857)
            if len(osm_site):
                osm_site["area_calc"] = osm_site.geometry.area
                site_geom = osm_site.sort_values(
                    "area_calc", ascending=False).geometry.iloc[0]
                del osm_site
            else:
                site_geom = gpd.GeoSeries(
                    [Point(lon, lat)], crs=4326
                ).to_crs(3857).iloc[0].buffer(40)
        except Exception:
            site_geom = gpd.GeoSeries(
                [Point(lon, lat)], crs=4326
            ).to_crs(3857).iloc[0].buffer(40)

        site_gdf   = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)
        site_point = site_geom.centroid

    gc.collect()   # free site fetch memory before graph fetch

    # ── Walk network — simplified graph for low RAM ───────────
    # simplify=True (default) keeps fewer nodes → less RAM
    G_walk = ox.graph_from_point(
        (lat, lon),
        dist=cfg["walk_dist"],
        network_type="walk",
        simplify=True
    )
    roads = ox.graph_to_gdfs(G_walk, nodes=False).to_crs(3857)

    site_wgs  = gpd.GeoSeries([site_point], crs=3857).to_crs(4326).iloc[0]
    site_node = ox.distance.nearest_nodes(G_walk, site_wgs.x, site_wgs.y)

    # ── Stations ──────────────────────────────────────────────
    # Limit fetch radius to max_radius * 1.5 (not *2) to save memory
    station_fetch_r = max(int(cfg["shade_r"] * 1.5), 1200)
    stations = ox.features_from_point(
        (lat, lon), tags={"railway": "station"}, dist=station_fetch_r
    ).to_crs(3857)
    stations = stations[stations.geometry.notnull()]
    if stations.empty:
        raise ValueError("No nearby stations found.")

    def _name(r):
        a = r.get("name:en"); b = r.get("name")
        if isinstance(a, str) and a.strip(): return a.strip()
        if isinstance(b, str) and b.strip(): return b.strip()
        return "MTR Station"

    stations["station_name"] = stations.apply(_name, axis=1)
    stations["dist"] = stations.geometry.centroid.distance(site_point)
    # ↓ Only top 2 stations — reduces route computations
    stations = stations.sort_values("dist").head(2)

    gc.collect()

    # ── Routes ────────────────────────────────────────────────
    routes = []
    for _, row in stations.iterrows():
        st_centroid = row.geometry.centroid
        st_wgs      = gpd.GeoSeries([st_centroid], crs=3857).to_crs(4326).iloc[0]
        st_node     = ox.distance.nearest_nodes(G_walk, st_wgs.x, st_wgs.y)
        try:
            path  = nx.shortest_path(G_walk, site_node, st_node, weight="length")
            route = ox.routing.route_to_gdf(G_walk, path).to_crs(3857)
        except Exception:
            continue
        dist_km  = round(route.length.sum() / 1000, 2)
        time_min = max(1, round((dist_km / WALK_SPEED_KMPH) * 60))
        routes.append({
            "route": route, "distance": dist_km, "time": time_min,
            "station_polygon": row.geometry,
            "station_centroid": st_centroid,
            "name": row["station_name"]
        })

    # ── Plot — use lower DPI to save RAM during rendering ─────
    fig, ax = plt.subplots(figsize=(10, 10))   # slightly smaller figure

    roads.plot(ax=ax, linewidth=0.25, color="#8a8a8a", alpha=0.4)
    del roads   # free road memory before basemap fetch
    gc.collect()

    gpd.GeoSeries([site_point.buffer(cfg["shade_r"])], crs=3857).plot(
        ax=ax, color="#2aa9ff", alpha=0.15)

    for ring_r, ring_lbl in cfg["rings"]:
        gpd.GeoSeries([site_point.buffer(ring_r)], crs=3857).boundary.plot(
            ax=ax, linestyle=(0, (4, 3)), linewidth=2, color="#2aa9ff")
        ax.text(site_point.x + ring_r + 30, site_point.y,
                ring_lbl, fontsize=10, color="black", weight="bold",
                va="center", zorder=10, clip_on=False)

    colors = ["#4caf50", "#ef5350", "#42a5f5"]

    for i, r in enumerate(routes):
        rc = colors[i % len(colors)]
        r["route"].plot(ax=ax, linewidth=2.8, color=rc, alpha=0.85, zorder=5)

        sg = r["station_polygon"]
        if sg.geom_type == "Point":
            sg = sg.buffer(60)
        gpd.GeoSeries([sg], crs=3857).plot(
            ax=ax, facecolor=rc, edgecolor=rc, linewidth=1, alpha=0.25, zorder=4)

        mid = r["route"].geometry.iloc[len(r["route"]) // 2].centroid
        ax.text(mid.x, mid.y, f"{r['time']} min\n{r['distance']} km",
                fontsize=9, weight="bold", color=rc, ha="center", zorder=6)

        icon_x = r["station_centroid"].x
        icon_y = r["station_centroid"].y + (MAP_EXTENT * 0.045)
        _add_mtr_icon(ax, icon_x, icon_y, size=0.04, zorder=8)
        ax.text(icon_x, icon_y + (MAP_EXTENT * 0.045),
                r["name"].upper(),
                fontsize=11, weight="bold", color="black",
                ha="center", va="center", zorder=9)

    site_gdf.plot(ax=ax, facecolor="red", edgecolor="none")
    ax.text(site_point.x, site_point.y - (MAP_EXTENT * 0.06),
            "SITE", color="red", weight="bold", ha="center", fontsize=11)

    ax.set_xlim(site_point.x - MAP_EXTENT, site_point.x + MAP_EXTENT * 1.15)
    ax.set_ylim(site_point.y - MAP_EXTENT, site_point.y + MAP_EXTENT)
    ax.set_aspect("equal")

    zoom_level = 16 if MAP_EXTENT <= 650 else (15 if MAP_EXTENT <= 950 else 14)

    # ↓ alpha=0.35 reduces tile download size slightly
    cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                   zoom=zoom_level, alpha=0.35)

    ax.set_title(f"Walking Accessibility ({max_walk_minutes} min)"
                 f" - {data_type} {value}",
                 fontsize=15, weight="bold")
    ax.set_axis_off()

    buf = BytesIO()
    plt.tight_layout()
    # ↓ dpi=100 instead of 120 — saves ~30% image RAM
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    gc.collect()   # final cleanup
    return buf
