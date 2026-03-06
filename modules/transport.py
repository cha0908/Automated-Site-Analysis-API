import matplotlib
matplotlib.use("Agg")

from typing import Optional, List
import os
import gc
import logging
import osmnx as ox
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.legend_handler as lh
import numpy as np
import pandas as pd

from shapely.geometry import Point, box, MultiLineString, LineString
from shapely.ops import unary_union
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache   = True
ox.settings.log_console = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

log = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

MAP_FETCH_RADIUS     = 4000   # fetch radius — larger catches more tunnel sections
HALF_W               = 1900
HALF_H               = 1100

COLOR_ROADS          = "#e85d9e"
COLOR_WATER          = "#6fa8dc"
COLOR_BUILDINGS      = "#d6d6d6"
COLOR_SITE           = "#FF0000"
COLOR_LIGHT_RAIL     = "#D3A809"

STATION_MIN_DISTANCE = 250
STATION_LOGO_ZOOM    = 0.055

# ============================================================
# MTR LINE COLOURS
# ============================================================

MTR_LINE_COLORS = {
    "island":              "#007DC5",
    "kwun tong":           "#00AB4E",
    "tsuen wan":           "#ED1D24",
    "tseung kwan o":       "#7D499D",
    "tung chung":          "#F7943E",
    "east rail":           "#5EB6E4",
    "tuen ma":             "#923011",
    "south island":        "#BAC429",
    "airport express":     "#888B8D",
    "lantau and airport":  "#888B8D",
    "guangzhoushen":       "#007DC5",
}

DEFAULT_MTR_COLOR = "#3f78b5"

MTR_LEGEND_LINES = [
    ("island",             "#007DC5", "Island Line"),
    ("kwun tong",          "#00AB4E", "Kwun Tong Line"),
    ("tsuen wan",          "#ED1D24", "Tsuen Wan Line"),
    ("tseung kwan o",      "#7D499D", "Tseung Kwan O Line"),
    ("tung chung",         "#F7943E", "Tung Chung Line"),
    ("east rail",          "#5EB6E4", "East Rail Line"),
    ("tuen ma",            "#923011", "Tuen Ma Line"),
    ("south island",       "#BAC429", "South Island Line"),
    ("airport express",    "#888B8D", "Airport Express"),
    ("lantau and airport", "#888B8D", "Airport Express"),
    ("guangzhoushen",      "#007DC5", "Express Rail Link"),
]

_LABEL_REMAP = {
    "LANTAU AND AIRPORT RAILWAY": "AIRPORT EXPRESS",
    "GUANGZHOUSHEN EXPRESS RAIL LINK": "EXPRESS RAIL LINK",
}


def get_mtr_color(name: str) -> str:
    nl = name.lower()
    for key, color in MTR_LINE_COLORS.items():
        if key in nl:
            return color
    return DEFAULT_MTR_COLOR


def _make_label(clean_name: str) -> str:
    upper = clean_name.upper().strip()
    for long_nm, short in _LABEL_REMAP.items():
        if long_nm in upper:
            return f"MTR {short}"
    if upper.startswith("MTR "):
        upper = upper[4:]
    return f"MTR {upper}"


# ============================================================
# LOAD MTR LOGO
# ============================================================

_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")

try:
    _mtr_img        = mpimg.imread(_MTR_LOGO_PATH)
    MTR_LOGO_LOADED = True
    log.info(f"[transport] MTR logo loaded: {_MTR_LOGO_PATH}")
except Exception as _e:
    _mtr_img        = None
    MTR_LOGO_LOADED = False
    log.warning(f"[transport] MTR logo not found ({_e}). Using fallback roundel.")


# ============================================================
# FALLBACK ROUNDEL
# ============================================================

def _draw_roundel(ax, x, y, size=60, color="#ED1D24", zorder=9):
    ax.add_patch(plt.Circle((x, y), size, color=color, zorder=zorder))
    bw, bh = size * 2.0, size * 0.55
    ax.add_patch(plt.Rectangle((x - bw/2, y - bh/2), bw, bh,
                                color="white", zorder=zorder + 1))
    ax.add_patch(plt.Circle((x, y), size * 0.55,
                             color="white", zorder=zorder + 2))
    ax.add_patch(plt.Rectangle((x - bw/2, y - bh/2), bw, bh,
                                color=color, zorder=zorder + 3))


def _draw_station(ax, x, y, zoom=STATION_LOGO_ZOOM,
                  fallback_color="#ED1D24", zorder=9):
    if MTR_LOGO_LOADED and _mtr_img is not None:
        icon = OffsetImage(_mtr_img, zoom=zoom)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False,
                            zorder=zorder, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        _draw_roundel(ax, x, y, size=60, color=fallback_color, zorder=zorder)


# ============================================================
# SAFE FETCH  — always returns EPSG:3857 GeoDataFrame
# ============================================================

def _safe_fetch(lat: float, lon: float, dist: int, tags: dict) -> gpd.GeoDataFrame:
    try:
        gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags)
        if gdf is not None and not gdf.empty:
            return gdf.to_crs(3857)
    except Exception as e:
        log.debug(f"[transport] fetch failed tags={tags}: {e}")
    return gpd.GeoDataFrame(geometry=[], crs=3857)


# ============================================================
# FLATTEN MultiIndex
# MUST be called immediately after every _safe_fetch() call,
# before any column access or geometry filtering.
# ============================================================

def _flatten(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    out = gdf.copy()
    if isinstance(out.index, pd.MultiIndex):
        try:
            out["_osmid"] = out.index.get_level_values("osmid")
        except KeyError:
            out["_osmid"] = range(len(out))
    else:
        out["_osmid"] = out.index.astype(str)
    return out.reset_index(drop=True)


# ============================================================
# CLEAN NAME
# OSMnx "name" column can be: str | float(NaN) | list | dict | pd.Series
# This always returns a clean ASCII str or None.
# ============================================================

def _clean_name(val) -> Optional[str]:
    if val is None:
        return None
    # Unwrap list / tuple
    if isinstance(val, (list, tuple)):
        val = next((v for v in val if v is not None), None)
    # Unwrap pandas Series
    if hasattr(val, "iloc"):
        val = val.iloc[0] if len(val) else None
    # Unwrap dict (relations sometimes store name as dict)
    if isinstance(val, dict):
        val = val.get("name") or next(iter(val.values()), None)
    # Now must be str or numeric
    if not isinstance(val, str):
        return None
    ascii_only = ''.join(c for c in val if ord(c) < 128).strip()
    return ascii_only if ascii_only else None


# ============================================================
# KEEP LINES — filter to line geometries only
# ============================================================

def _keep_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    mask = gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
    return gdf[mask].copy().reset_index(drop=True)


# ============================================================
# FETCH MTR ROUTES
#
# Strategy: fetch railway=rail AND railway=subway, flatten each
# immediately, keep lines, concat, deduplicate.
#
# Also tries a graph-based fallback using ox.graph_from_point
# which often captures tunnel sections that features_from_point misses.
# ============================================================

def _fetch_mtr_routes(lat: float, lon: float, dist: int) -> gpd.GeoDataFrame:
    frames = []

    # ── Method A: features_from_point (standard) ──────────────
    for tags in [{"railway": "rail"}, {"railway": "subway"}]:
        raw = _safe_fetch(lat, lon, dist, tags)
        if raw.empty:
            log.debug(f"[transport] features empty for tags={tags}")
            continue
        flat  = _flatten(raw)
        lines = _keep_lines(flat)
        if not lines.empty:
            log.info(f"[transport] Method A: {len(lines)} lines for {tags}, "
                     f"name col exists={('name' in lines.columns)}")
            frames.append(lines)
        else:
            log.debug(f"[transport] Method A: no line geometries for {tags}")

    # ── Method B: graph_from_point (catches tunnels) ──────────
    # ox.graph_from_point with custom_filter fetches the street network
    # for rail, which includes underground/tunnel segments.
    try:
        G = ox.graph_from_point(
            (lat, lon), dist=dist,
            custom_filter='["railway"~"rail|subway|light_rail"]',
            retain_all=True,
            simplify=True
        )
        edges = ox.graph_to_gdfs(G, nodes=False).to_crs(3857)
        if not edges.empty:
            flat_e  = _flatten(edges)
            lines_e = _keep_lines(flat_e)
            if not lines_e.empty:
                log.info(f"[transport] Method B (graph): {len(lines_e)} rail edges, "
                         f"name col exists={('name' in lines_e.columns)}")
                frames.append(lines_e)
    except Exception as e:
        log.debug(f"[transport] Method B (graph) failed: {e}")

    if not frames:
        log.warning("[transport] _fetch_mtr_routes: no rail data found at all")
        return gpd.GeoDataFrame(geometry=[], crs=3857)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate by osmid
    if "_osmid" in combined.columns:
        before = len(combined)
        combined = (combined
                    .drop_duplicates(subset=["_osmid"])
                    .drop(columns=["_osmid"])
                    .reset_index(drop=True))
        log.info(f"[transport] After dedup: {before} → {len(combined)} rail rows")

    # Log what names we actually got
    if "name" in combined.columns:
        names = combined["name"].apply(_clean_name).dropna().unique().tolist()
        log.info(f"[transport] MTR route names found: {names}")
    else:
        log.warning("[transport] 'name' column missing from combined MTR routes")

    return gpd.GeoDataFrame(combined, crs=3857)


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_transport(data_type: str, value: str,
                       radius_m: Optional[int] = None,
                       lon: float = None, lat: float = None,
                       lot_ids: List[str] = None,
                       extents: List[dict] = None):

    log.info(f"[transport] Starting: {data_type} {value}")

    # ── 1. Resolve coordinates ────────────────────────────────
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r  = radius_m if radius_m else MAP_FETCH_RADIUS
    log.info(f"[transport] Resolved: lon={lon:.5f} lat={lat:.5f} fetch_r={fetch_r}")

    # ── 2. Site polygon ───────────────────────────────────────
    lot_gdf = get_lot_boundary(lon, lat, data_type, extents)
    if lot_gdf is not None:
        site_geom  = lot_gdf.geometry.iloc[0]
        site_gdf   = lot_gdf
        site_point = site_geom.centroid
        log.info("[transport] Using lot boundary for site polygon")
    else:
        site_point = (gpd.GeoSeries([Point(lon, lat)], crs=4326)
                      .to_crs(3857).iloc[0])
        site_geom  = site_point.buffer(40)
        site_gdf   = None
        log.info("[transport] No lot boundary — will use nearest building")

    # ── 3. Fetch all OSM layers (flatten immediately) ─────────

    # Buildings
    _bld  = _safe_fetch(lat, lon, fetch_r, {"building": True})
    _bld_flat = _flatten(_bld)
    buildings = (_bld_flat[
        _bld_flat.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy() if not _bld_flat.empty else gpd.GeoDataFrame(geometry=[], crs=3857))
    log.info(f"[transport] Buildings: {len(buildings)}")

    # Refine site polygon
    if site_gdf is None:
        if not buildings.empty:
            dists     = buildings.geometry.distance(site_point)
            site_geom = buildings.loc[dists.idxmin(), "geometry"]
        site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # Roads
    _rds = _safe_fetch(lat, lon, fetch_r, {
        "highway": ["motorway", "trunk", "primary", "secondary",
                    "tertiary", "residential", "service", "unclassified"]
    })
    roads = _keep_lines(_flatten(_rds))
    log.info(f"[transport] Roads: {len(roads)}")

    # Light rail
    _lr = _safe_fetch(lat, lon, fetch_r, {"railway": "light_rail"})
    light_rail = _keep_lines(_flatten(_lr))
    log.info(f"[transport] Light rail segments: {len(light_rail)}")

    # MTR routes (dedicated function)
    mtr_routes = _fetch_mtr_routes(lat, lon, fetch_r)
    log.info(f"[transport] MTR route rows total: {len(mtr_routes)}")

    # Stations
    _sta     = _safe_fetch(lat, lon, fetch_r, {"railway": "station"})
    stations = _flatten(_sta)
    log.info(f"[transport] Stations: {len(stations)}")

    # Water
    _wf = []
    for wtags in [{"natural": "water"}, {"natural": "bay"}]:
        _w = _safe_fetch(lat, lon, fetch_r, wtags)
        if not _w.empty:
            _wf.append(_flatten(_w))
    if _wf:
        _wc   = pd.concat(_wf, ignore_index=True)
        water = gpd.GeoDataFrame(
            _wc[_wc.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy(),
            crs=3857)
    else:
        water = gpd.GeoDataFrame(geometry=[], crs=3857)
    log.info(f"[transport] Water polygons: {len(water)}")

    gc.collect()

    # ── 4. Map extent ─────────────────────────────────────────
    xmin = site_point.x - HALF_W
    xmax = site_point.x + HALF_W
    ymin = site_point.y - HALF_H
    ymax = site_point.y + HALF_H

    clip_box = box(xmin, ymin, xmax, ymax)
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs=3857)

    # ── 5. Figure — lock extent BEFORE basemap ────────────────
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#f4f4f4")
    ax.set_facecolor("#f4f4f4")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cx.add_basemap(ax, crs="EPSG:3857",
                   source=cx.providers.CartoDB.Positron,
                   zoom=15, alpha=0.6)

    # ── 6. Base layers ────────────────────────────────────────

    if not buildings.empty:
        buildings.plot(ax=ax, color=COLOR_BUILDINGS, alpha=0.5, zorder=1)

    if not water.empty:
        water.plot(ax=ax, color=COLOR_WATER, alpha=0.8, zorder=2)

    if not roads.empty:
        try:
            rc = gpd.clip(roads, clip_gdf)
            if not rc.empty:
                rc.plot(ax=ax, color=COLOR_ROADS, linewidth=1.4, zorder=3)
        except Exception:
            roads.plot(ax=ax, color=COLOR_ROADS, linewidth=1.4, zorder=3)

    # ── 7. MTR routes ─────────────────────────────────────────

    lines_on_map: dict = {}

    if not mtr_routes.empty:
        try:
            mtr_vis = gpd.clip(mtr_routes, clip_gdf)
        except Exception:
            mtr_vis = mtr_routes.copy()

        log.info(f"[transport] MTR visible after clip: {len(mtr_vis)}")

        if not mtr_vis.empty:
            placed_lbl_pts = []

            # Find name column — try "name" then "name:en"
            name_col = None
            for candidate in ["name", "name:en"]:
                if candidate in mtr_vis.columns:
                    has_val = mtr_vis[candidate].apply(
                        lambda v: _clean_name(v) is not None).any()
                    if has_val:
                        name_col = candidate
                        log.info(f"[transport] Using name column: '{name_col}'")
                        break

            if name_col is None:
                log.warning("[transport] No usable name column — drawing all MTR default color")
                mtr_vis.plot(ax=ax, color="white",           linewidth=8,   zorder=4)
                mtr_vis.plot(ax=ax, color=DEFAULT_MTR_COLOR, linewidth=4.5, zorder=5)
            else:
                # Group rows by cleaned name
                mtr_vis["_clean_name"] = mtr_vis[name_col].apply(_clean_name)

                named   = mtr_vis[mtr_vis["_clean_name"].notna()]
                unnamed = mtr_vis[mtr_vis["_clean_name"].isna()]

                for clean_nm, grp in named.groupby("_clean_name"):
                    lc = get_mtr_color(clean_nm)
                    log.info(f"[transport] Drawing: '{clean_nm}' → color={lc} rows={len(grp)}")

                    grp.plot(ax=ax, color="white", linewidth=8,   zorder=4)
                    grp.plot(ax=ax, color=lc,      linewidth=4.5, zorder=5)

                    if lc not in lines_on_map:
                        official = clean_nm.title()
                        for k, c, lbl in MTR_LEGEND_LINES:
                            if k in clean_nm.lower():
                                official = lbl
                                break
                        lines_on_map[lc] = official

                    # Label
                    merged = grp.union_all()
                    if merged.length < 600:
                        continue
                    mid   = merged.interpolate(0.5, normalized=True)
                    off_y = 0
                    for pp in placed_lbl_pts:
                        if mid.distance(pp) < 500:
                            off_y += 170
                    lp = Point(mid.x, mid.y + off_y)
                    placed_lbl_pts.append(lp)
                    if xmin <= lp.x <= xmax and ymin <= lp.y <= ymax:
                        ax.text(lp.x, lp.y, _make_label(clean_nm),
                                fontsize=9, weight="bold", color=lc,
                                ha="center", va="center", zorder=12,
                                bbox=dict(facecolor="white", edgecolor="none",
                                          alpha=0.85, pad=2))

                if not unnamed.empty:
                    unnamed.plot(ax=ax, color="white",           linewidth=8,   zorder=4)
                    unnamed.plot(ax=ax, color=DEFAULT_MTR_COLOR, linewidth=4.5, zorder=5)

    # ── 8. Light rail ─────────────────────────────────────────

    if not light_rail.empty:
        try:
            lrc = gpd.clip(light_rail, clip_gdf)
        except Exception:
            lrc = light_rail.copy()
        if not lrc.empty:
            lrc.plot(ax=ax, color="white",          linewidth=6,   zorder=4)
            lrc.plot(ax=ax, color=COLOR_LIGHT_RAIL, linewidth=3.5, zorder=5)

    # ── 9. Stations ───────────────────────────────────────────

    if not stations.empty:
        stns        = stations.copy()
        stns["_cx"] = stns.geometry.centroid.x
        stns["_cy"] = stns.geometry.centroid.y

        in_view = stns[
            (stns["_cx"] >= xmin) & (stns["_cx"] <= xmax) &
            (stns["_cy"] >= ymin) & (stns["_cy"] <= ymax)
        ]
        log.info(f"[transport] Stations in viewport: {len(in_view)}")

        placed_stn_pts = []
        for _, row in in_view.iterrows():
            sx, sy = row["_cx"], row["_cy"]
            pt = Point(sx, sy)
            if any(pt.distance(p) < STATION_MIN_DISTANCE for p in placed_stn_pts):
                continue
            placed_stn_pts.append(pt)

            fb_color  = "#ED1D24"
            name_str  = _clean_name(row.get("name"))
            if name_str:
                c = get_mtr_color(name_str)
                if c != DEFAULT_MTR_COLOR:
                    fb_color = c
                elif not mtr_routes.empty and "name" in mtr_routes.columns:
                    min_d = float("inf")
                    for _, grp in mtr_routes.groupby("name"):
                        d = grp.union_all().distance(pt)
                        if d < min_d:
                            min_d    = d
                            raw_ln   = grp["name"].iloc[0]
                            ln_clean = _clean_name(raw_ln)
                            if ln_clean:
                                fb_color = get_mtr_color(ln_clean)

            _draw_station(ax, sx, sy, zoom=STATION_LOGO_ZOOM,
                          fallback_color=fb_color, zorder=9)

            if name_str:
                ax.text(sx, sy - 130, name_str,
                        fontsize=7.5, weight="bold",
                        ha="center", va="top", color="#333333", zorder=10,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.75, pad=1.5))

    # ── 10. Site ──────────────────────────────────────────────

    site_gdf.plot(ax=ax, facecolor=COLOR_SITE, edgecolor="white",
                  linewidth=2, zorder=13)
    centroid = site_geom.centroid
    ax.text(centroid.x, centroid.y - 130, "SITE",
            fontsize=14, weight="bold", color="black",
            ha="center", va="top", zorder=14,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=3))

    # ── 11. North arrow ───────────────────────────────────────

    ax.annotate('', xy=(0.07, 0.85), xytext=(0.07, 0.80),
                xycoords=ax.transAxes,
                arrowprops=dict(facecolor='black', width=1.5,
                                headwidth=8, headlength=10))
    ax.text(0.07, 0.86, "N", transform=ax.transAxes,
            ha='center', va='bottom', fontsize=12, weight='bold')

    # ── 12. Re-lock extent ────────────────────────────────────

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ── 13. Legend ────────────────────────────────────────────

    legend_handles = []
    legend_handles.append(
        mlines.Line2D([], [], color=COLOR_LIGHT_RAIL, linewidth=4, label="Light Rail"))

    seen_lc = set()
    for _, color, label in MTR_LEGEND_LINES:
        if color in lines_on_map and color not in seen_lc:
            legend_handles.append(
                mlines.Line2D([], [], color=color, linewidth=4, label=label))
            seen_lc.add(color)

    legend_handles.append(
        mlines.Line2D([], [], color=COLOR_ROADS, linewidth=4, label="Vehicle Circulation"))
    legend_handles.append(
        mpatches.Patch(facecolor=COLOR_SITE, label="Site"))

    handler_map = {}
    if MTR_LOGO_LOADED and _mtr_img is not None:
        raw_arr = ((_mtr_img * 255).astype(np.uint8)
                   if _mtr_img.dtype != np.uint8 else _mtr_img)
        thumb   = np.array(Image.fromarray(raw_arr).resize((20, 20), Image.LANCZOS))

        class _LogoHandle(mlines.Line2D):
            pass

        class _LogoHandler(lh.HandlerBase):
            def create_artists(self, legend, orig_handle,
                               xd, yd, width, height, fontsize, trans):
                imgbox = OffsetImage(thumb, zoom=0.8)
                ab = AnnotationBbox(imgbox, (xd + width/2, yd + height/2),
                                    xycoords=trans, frameon=False)
                return [ab]

        legend_handles.append(_LogoHandle([], [], label="MTR Station"))
        handler_map = {_LogoHandle: _LogoHandler()}
    else:
        legend_handles.append(
            mlines.Line2D([], [], marker='o', linestyle='None',
                          markerfacecolor='#ED1D24', markeredgecolor='white',
                          markeredgewidth=2, markersize=10, label="MTR Station"))

    legend = ax.legend(
        handles=legend_handles,
        handler_map=handler_map if handler_map else None,
        loc="lower left", bbox_to_anchor=(0.02, 0.02),
        frameon=True, facecolor="white", edgecolor="black",
        fontsize=10, title_fontsize=10,
        labelspacing=0.4, handlelength=1.8, handleheight=1.0,
        borderpad=0.6, title="Legend")
    legend.get_frame().set_linewidth(2)

    # ── 14. Title & export ────────────────────────────────────

    ax.set_title(f"SITE ANALYSIS – Transportation ({data_type} {value})",
                 fontsize=18, weight="bold")
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf
