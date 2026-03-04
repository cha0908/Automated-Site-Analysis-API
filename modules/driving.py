import matplotlib
matplotlib.use("Agg")

import os
import gc
import math
import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from shapely.geometry import Point, LineString, MultiLineString
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache   = True
ox.settings.log_console = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

DRIVE_SPEED = 35  # km/h

RING_CONFIGS = {
    5: {
        "rings":      [(83,  "1.5 MINS\n0.083 KM"),
                       (250, "3 MINS\n0.25 KM"),
                       (400, "5 MINS\n0.40 KM")],
        "max_radius": 400, "map_extent": 600,  "graph_dist": 800,
    },
    10: {
        "rings":      [(250, "3 MINS\n0.25 KM"),
                       (500, "6 MINS\n0.50 KM"),
                       (750, "10 MINS\n0.75 KM")],
        "max_radius": 750, "map_extent": 875,  "graph_dist": 1500,
    },
    15: {
        "rings":      [(375,  "1.5 MINS\n0.375 KM"),
                       (750,  "3 MINS\n0.75 KM"),
                       (1125, "4.5 MINS\n1.125 KM")],
        "max_radius": 1300, "map_extent": 1400, "graph_dist": 2500,
    },
}

INGRESS_COLOR = "#e74c3c"
EGRESS_COLOR  = "#27ae60"

# ── MTR logo ──────────────────────────────────────────────────
_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")
try:
    _mtr_img = mpimg.imread(_MTR_LOGO_PATH)
except Exception:
    _mtr_img = None

def _add_mtr_icon(ax, x, y, size=0.035, zorder=15):
    if _mtr_img is not None:
        icon = OffsetImage(_mtr_img, zoom=size)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False, zorder=zorder,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)


# ── Arrow: one clean annotate arrow per route ─────────────────
def _add_route_arrow(ax, gdf_route, color, position=1.0, zorder=20):
    """
    Draw a single directional arrow along the full route geometry.

    The arrow is placed at a distance `position` * total_length
    along the route, where `position` is in [0, 1].
    """
    if gdf_route is None or gdf_route.empty:
        return

    # Flatten all line geometries into one ordered coordinate list
    coords: list[tuple[float, float]] = []
    for geom in gdf_route.geometry:
        parts = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        for p in parts:
            if isinstance(p, LineString):
                pts = list(p.coords)
                if not coords:
                    coords.extend(pts)
                else:
                    # Avoid duplicating the connecting point
                    if pts and pts[0] == coords[-1]:
                        coords.extend(pts[1:])
                    else:
                        coords.extend(pts)

    if len(coords) < 2:
        return

    # Compute segment lengths and total length
    seg_lengths = []
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        seg_lengths.append(math.hypot(x1 - x0, y1 - y0))

    total_length = sum(seg_lengths)
    if total_length == 0:
        return

    # Clamp position into [0, 1]
    pos = max(0.0, min(1.0, float(position)))
    target_dist = pos * total_length

    # Special cases: exactly at start or end of the route
    if pos <= 0.0:
        (x0, y0) = coords[0]
        (x1, y1) = coords[1]
        head_x, head_y = x0, y0
        # Tail a bit forward along first segment
        back_t = 0.15
        tail_x = x0 + back_t * (x1 - x0)
        tail_y = y0 + back_t * (y1 - y0)
    elif pos >= 1.0:
        (x0, y0) = coords[-2]
        (x1, y1) = coords[-1]
        head_x, head_y = x1, y1
        # Tail a bit back along last segment
        back_t = 0.15
        tail_x = x1 - back_t * (x1 - x0)
        tail_y = y1 - back_t * (y1 - y0)
    else:
        # Find the segment where target_dist falls
        dist_so_far = 0.0
        seg_index = 0
        for i, seg_len in enumerate(seg_lengths):
            if dist_so_far + seg_len >= target_dist:
                seg_index = i
                break
            dist_so_far += seg_len

        (x0, y0) = coords[seg_index]
        (x1, y1) = coords[seg_index + 1]
        seg_len = seg_lengths[seg_index]
        if seg_len == 0:
            return

        # Local fraction along this segment
        local_t = (target_dist - dist_so_far) / seg_len
        # Arrow head at this point
        head_x = x0 + local_t * (x1 - x0)
        head_y = y0 + local_t * (y1 - y0)

        # Tail slightly before the head along the same segment
        back_t = max(0.0, local_t - 0.15)
        tail_x = x0 + back_t * (x1 - x0)
        tail_y = y0 + back_t * (y1 - y0)

    if head_x == tail_x and head_y == tail_y:
        return

    ax.annotate(
        "",
        xy=(head_x, head_y),
        xytext=(tail_x, tail_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=3,
            mutation_scale=26,
        ),
        zorder=zorder,
    )


# ── Helpers ───────────────────────────────────────────────────
def _safe_name(station):
    raw = station.get("name:en") or station.get("name")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "STATION"
    return ''.join(c for c in str(raw).upper() if ord(c) < 128).strip() or "STATION"


def _tofrom_pos(st_x, st_y, cx_val, cy_val, extent):
    dx = st_x - cx_val; dy = st_y - cy_val
    d  = np.sqrt(dx**2 + dy**2)
    if d < 1:
        return Point(st_x, st_y)
    dx /= d; dy /= d
    return Point(
        np.clip(cx_val + dx*extent*0.88, cx_val - extent*0.93, cx_val + extent*0.93),
        np.clip(cy_val + dy*extent*0.88, cy_val - extent*0.93, cy_val + extent*0.93),
    )


def _nudge(pt, placed, cx_val, cy_val, extent, min_sep=0.18):
    bx = pt.x - cx_val; by = pt.y - cy_val
    bd = np.sqrt(bx**2 + by**2)
    if bd < 1:
        return pt
    bx /= bd; by /= bd
    for deg in [0,22,-22,44,-44,66,-66,90,-90,112,-112,135,-135,158,-158,180]:
        rad = np.radians(deg)
        rx  = bx*np.cos(rad) - by*np.sin(rad)
        ry  = bx*np.sin(rad) + by*np.cos(rad)
        c   = Point(
            np.clip(cx_val + rx*extent*0.88, cx_val-extent*0.93, cx_val+extent*0.93),
            np.clip(cy_val + ry*extent*0.88, cy_val-extent*0.93, cy_val+extent*0.93))
        if all(c.distance(p) >= extent*min_sep for p in placed):
            return c
    return pt


def _draw_tofrom(ax, x, y, name, zorder=18):
    ax.text(x, y, f"TO/FROM\n{name}",
            fontsize=8.5, weight="bold", color="black",
            ha="center", va="center", zorder=zorder,
            multialignment="center",
            bbox=dict(facecolor="white", edgecolor="black",
                      linewidth=1.5, pad=4, boxstyle="square,pad=0.4"))


def _north_arrow(ax, xlim, ylim, extent):
    x = xlim[0] + 0.965*(xlim[1]-xlim[0])
    y = ylim[0] + 0.940*(ylim[1]-ylim[0])
    L = extent * 0.055
    ax.annotate("", xy=(x, y+L*0.6), xytext=(x, y-L*0.6),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=2, mutation_scale=16), zorder=25)
    ax.text(x, y+L*0.85, "N", fontsize=13, weight="bold",
            color="black", ha="center", va="center", zorder=25)


# ── Main generator ────────────────────────────────────────────
def generate_driving(data_type: str, value: str,
                     zone_data: gpd.GeoDataFrame = None,
                     lon: float = None, lat: float = None,
                     lot_ids: list = None, extents: list = None,
                     max_drive_minutes: int = 15):

    if max_drive_minutes not in RING_CONFIGS:
        max_drive_minutes = 15
    cfg        = RING_CONFIGS[max_drive_minutes]
    MAP_EXTENT = cfg["map_extent"]
                         
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)

    site_pt_3857 = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    # ── Site polygon ──────────────────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type)
    if lot_gdf is not None:
        site_poly = lot_gdf.geometry.iloc[0]
        site_gdf  = lot_gdf
    else:
        site_poly = None
        # 1. Preloaded zone data
        if zone_data is not None:
            try:
                ozp     = zone_data.to_crs(3857)
                matches = ozp[ozp.contains(site_pt_3857)]
                if not matches.empty:
                    site_poly = matches.geometry.iloc[0]
            except Exception:
                pass
        # 2. OSM building footprint fallback
        if site_poly is None:
            for dist in [80, 150, 250]:   # expand search radius progressively
                try:
                    osm = ox.features_from_point(
                        (lat, lon), dist=dist, tags={"building": True}
                    ).to_crs(3857)
                    if len(osm):
                        osm["_a"] = osm.geometry.area
                        candidate = osm.sort_values("_a", ascending=False).geometry.iloc[0]
                        # Only use if centroid is close to our point
                        if candidate.centroid.distance(site_pt_3857) < dist * 2:
                            site_poly = candidate
                            break
                except Exception:
                    pass

        # 3. OSM landuse / amenity fallback
        if site_poly is None:
            for tags in [{"landuse": True}, {"amenity": True}]:
                try:
                    osm = ox.features_from_point(
                        (lat, lon), dist=100, tags=tags
                    ).to_crs(3857)
                    polys = osm[osm.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
                    if len(polys):
                        polys = polys.copy()
                        polys["_d"] = polys.geometry.centroid.distance(site_pt_3857)
                        site_poly = polys.sort_values("_d").geometry.iloc[0]
                        break
                except Exception:
                    pass

        # 4. Guaranteed buffer fallback — always shows SOMETHING visible
        if site_poly is None:
            site_poly = site_pt_3857.buffer(80)   # 80m visible circle

        site_gdf = gpd.GeoDataFrame(geometry=[site_poly], crs=3857)

    centroid       = site_poly.centroid
    cx_val, cy_val = centroid.x, centroid.y

    gc.collect()

    # ── Drive network ─────────────────────────────────────────
    G = ox.graph_from_point((lat, lon), dist=cfg["graph_dist"],
                            network_type="drive", simplify=True)
    for u, v, k, data in G.edges(keys=True, data=True):
        data["travel_time"] = data["length"] / (DRIVE_SPEED * 1000 / 60)
    site_node = ox.distance.nearest_nodes(G, lon, lat)

    # ── Stations ──────────────────────────────────────────────
    stations = ox.features_from_point(
        (lat, lon), tags={"railway": "station"},
        dist=max(cfg["max_radius"] * 2, 1500)
    ).to_crs(3857)
    stations["dist"] = stations.centroid.distance(centroid)
    # Keep up to 3 nearest stations, but only within the largest ring radius
    stations = stations.sort_values("dist")
    stations = stations[stations["dist"] <= cfg["max_radius"]].head(3)

    def _route(nf, nt):
        try:
            path = nx.shortest_path(G, nf, nt, weight="travel_time")
            return ox.routing.route_to_gdf(G, path).to_crs(3857)
        except Exception:
            return None

    gc.collect()

    # ── Figure ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12))
    zoom = 16 if MAP_EXTENT <= 650 else (15 if MAP_EXTENT <= 950 else 14)

    cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                   zoom=zoom, alpha=0.55)

    ox.graph_to_gdfs(G, nodes=False).to_crs(3857).plot(
        ax=ax, linewidth=0.3, color="#8a8a8a", alpha=0.35, zorder=1)

    # Yellow rings
    max_r = cfg["rings"][-1][0]
    gpd.GeoSeries([centroid.buffer(max_r)], crs=3857).plot(
        ax=ax, color="#FFD700", alpha=0.18, zorder=2)
    for ring_r, ring_lbl in cfg["rings"]:
        gpd.GeoSeries([centroid.buffer(ring_r)], crs=3857).boundary.plot(
            ax=ax, color="#e6b800", linewidth=2,
            linestyle=(0, (4, 3)), zorder=5)
        ax.text(cx_val + ring_r + 30, cy_val,
                ring_lbl, fontsize=10, color="black", weight="bold",
                ha="left", va="center", zorder=10, clip_on=False)

    # ── Stations + routes ─────────────────────────────────────
    placed_tofrom = []

    for _, station in stations.iterrows():
        st_cen  = station.geometry.centroid
        st_wgs  = gpd.GeoSeries([st_cen], crs=3857).to_crs(4326).iloc[0]
        st_node = ox.distance.nearest_nodes(G, st_wgs.x, st_wgs.y)
        st_name = _safe_name(station)

        egress_gdf  = _route(site_node, st_node)
        ingress_gdf = _route(st_node,   site_node)

        # If ingress and egress share the same geometry, draw one centreline
        # but keep both directional arrows so overlap is still clear.
        overlap = False
        if egress_gdf is not None and ingress_gdf is not None:
            if len(egress_gdf) == len(ingress_gdf):
                overlap = all(
                    e.equals(i) for e, i in zip(egress_gdf.geometry, ingress_gdf.geometry)
                )

        if overlap:
            # Single centreline (use ingress style), arrows for both directions.
            # Egress arrow near the station end (position ~0.9),
            # ingress arrow exactly at the site end (position = 1.0).
            ingress_gdf.plot(ax=ax, linewidth=3.5, color=INGRESS_COLOR,
                             alpha=0.92, zorder=8)
            _add_route_arrow(ax, egress_gdf,  EGRESS_COLOR,  position=0.9, zorder=20)
            _add_route_arrow(ax, ingress_gdf, INGRESS_COLOR, position=1.0, zorder=21)
        else:
            if egress_gdf is not None:
                egress_gdf.plot(ax=ax, linewidth=3.5, color=EGRESS_COLOR,
                                alpha=0.92, zorder=8)
                _add_route_arrow(ax, egress_gdf, EGRESS_COLOR, position=0.9, zorder=20)

            if ingress_gdf is not None:
                ingress_gdf.plot(ax=ax, linewidth=3.5, color=INGRESS_COLOR,
                                 alpha=0.92, zorder=9)
                # Arrow head at the site end of the route
                _add_route_arrow(ax, ingress_gdf, INGRESS_COLOR, position=1.0, zorder=21)

        # TO/FROM label at map edge
        base_pt  = _tofrom_pos(st_cen.x, st_cen.y, cx_val, cy_val, MAP_EXTENT)
        final_pt = _nudge(base_pt, placed_tofrom, cx_val, cy_val, MAP_EXTENT)
        _draw_tofrom(ax, final_pt.x, final_pt.y, st_name)
        placed_tofrom.append(final_pt)

        # Station polygon
        st_geom = station.geometry
        if st_geom.geom_type == "Point":
            st_geom = st_geom.buffer(55)
        gpd.GeoSeries([st_geom], crs=3857).plot(
            ax=ax, facecolor="#5dade2", edgecolor="#2e86c1",
            linewidth=2, alpha=0.5, zorder=23)

        _add_mtr_icon(ax, st_cen.x, st_cen.y, size=0.035, zorder=24)
        ax.text(st_cen.x, st_cen.y + MAP_EXTENT*0.060, st_name,
                fontsize=10, weight="bold", color="black",
                ha="center", va="bottom", zorder=25)

    # ── Site ──────────────────────────────────────────────────
    site_gdf.plot(ax=ax, facecolor="red", edgecolor="none",
                  alpha=0.85, zorder=11)
    ax.text(cx_val, cy_val - MAP_EXTENT*0.06, "SITE",
            color="black", weight="bold", ha="center",
            fontsize=11, zorder=12)

    # ── Legend ────────────────────────────────────────────────
    leg = ax.legend(
        handles=[
            mlines.Line2D([], [], color=INGRESS_COLOR, linewidth=3,
                          marker=">", markersize=9,
                          markerfacecolor=INGRESS_COLOR, label="INGRESS ROUTING"),
            mlines.Line2D([], [], color=EGRESS_COLOR, linewidth=3,
                          marker=">", markersize=9,
                          markerfacecolor=EGRESS_COLOR, label="EGRESS ROUTING"),
        ],
        loc="lower right", frameon=True, facecolor="white",
        edgecolor="black", fontsize=10,
        title="LEGEND", title_fontsize=10, framealpha=1.0)
    leg.get_title().set_fontweight("bold")

    # ── Axes ──────────────────────────────────────────────────
    ax.set_xlim(cx_val - MAP_EXTENT, cx_val + MAP_EXTENT)
    ax.set_ylim(cy_val - MAP_EXTENT, cy_val + MAP_EXTENT)
    ax.set_aspect("equal")
    _north_arrow(ax, ax.get_xlim(), ax.get_ylim(), MAP_EXTENT)
    ax.set_axis_off()
    ax.set_title(
        f"SITE ANALYSIS – Driving Distance ({max_drive_minutes} min)"
        f" – {data_type} {value}",
        fontsize=15, weight="bold")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130, facecolor="white")
    plt.close(fig)
    gc.collect()
    return buf
