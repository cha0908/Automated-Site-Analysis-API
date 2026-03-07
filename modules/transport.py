from typing import Optional, List
import matplotlib
matplotlib.use("Agg")

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
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from modules.resolver import resolve_location, get_lot_boundary

ox.settings.use_cache        = True
ox.settings.log_console      = False
ox.settings.requests_timeout = 20   # hard timeout per tile request

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAP_RADIUS           = 2500
HALF_W               = 1900
HALF_H               = 1100
STATION_MIN_DISTANCE = 250
STATION_LOGO_ZOOM    = 0.055

COLOR_ROADS      = "#e85d9e"
COLOR_BUILDINGS  = "#d6d6d6"
COLOR_SITE       = "#FF0000"
COLOR_LIGHT_RAIL = "#D3A809"
DEFAULT_MTR_COLOR = "#3f78b5"

# ── MTR colour map ─────────────────────────────────────────────────────────────
# ORDER MATTERS: more-specific keys must come before substrings that would
# accidentally match them (e.g. "south island" before "island").

MTR_LINE_COLORS = [
    ("south island",       "#BAC429"),
    ("island",             "#007DC5"),
    ("kwun tong",          "#00AB4E"),
    ("tsuen wan",          "#ED1D24"),
    ("tseung kwan o",      "#7D499D"),
    ("tung chung",         "#F7943E"),
    ("east rail",          "#5EB6E4"),
    ("tuen ma",            "#923011"),
    ("airport express",    "#888B8D"),
    ("lantau and airport", "#888B8D"),
    ("guangzhou",          "#888B8D"),   # GuangzhouShenzhen / guangzhoushen
    ("express rail",       "#888B8D"),
]

# Legend entries shown when that line is on the map (same order)
MTR_LEGEND_LINES = [
    ("south island",       "#BAC429", "South Island Line"),
    ("island",             "#007DC5", "Island Line"),
    ("kwun tong",          "#00AB4E", "Kwun Tong Line"),
    ("tsuen wan",          "#ED1D24", "Tsuen Wan Line"),
    ("tseung kwan o",      "#7D499D", "Tseung Kwan O Line"),
    ("tung chung",         "#F7943E", "Tung Chung Line"),
    ("east rail",          "#5EB6E4", "East Rail Line"),
    ("tuen ma",            "#923011", "Tuen Ma Line"),
    ("airport express",    "#888B8D", "Airport Express"),
    ("lantau and airport", "#888B8D", "Airport Express"),
    ("guangzhou",          "#888B8D", "Express Rail Link"),
    ("express rail",       "#888B8D", "Express Rail Link"),
]

_LABEL_REMAP = {
    "LANTAU AND AIRPORT": "AIRPORT EXPRESS",
    "GUANGZHOU":          "EXPRESS RAIL LINK",
    "EXPRESS RAIL":       "EXPRESS RAIL LINK",
}


def get_mtr_color(name: str) -> str:
    nl = name.lower()
    for key, color in MTR_LINE_COLORS:          # list, not dict — order preserved
        if key in nl:
            return color
    return DEFAULT_MTR_COLOR


def _make_label(name: str) -> str:
    upper = name.upper().strip()
    # strip leading "MTR " prefix first
    if upper.startswith("MTR "):
        upper = upper[4:]
    for fragment, replacement in _LABEL_REMAP.items():
        if fragment in upper:
            return f"MTR {replacement}"
    return f"MTR {upper}"


# ── MTR logo ──────────────────────────────────────────────────────────────────

_STATIC_DIR    = os.path.join(os.path.dirname(__file__), "..", "static")
_MTR_LOGO_PATH = os.path.join(_STATIC_DIR, "HK_MTR_logo.png")

try:
    _mtr_img        = mpimg.imread(_MTR_LOGO_PATH)
    MTR_LOGO_LOADED = True
    log.info("[transport] MTR logo loaded")
except Exception as _e:
    _mtr_img        = None
    MTR_LOGO_LOADED = False
    log.warning(f"[transport] MTR logo not found: {_e}")


def _draw_roundel(ax, x, y, size=60, color="#ED1D24", zorder=9):
    ax.add_patch(plt.Circle((x, y), size, color=color, zorder=zorder))
    bw, bh = size * 2.0, size * 0.55
    ax.add_patch(plt.Rectangle((x - bw/2, y - bh/2), bw, bh,
                                color="white", zorder=zorder + 1))
    ax.add_patch(plt.Circle((x, y), size * 0.55, color="white", zorder=zorder + 2))
    ax.add_patch(plt.Rectangle((x - bw/2, y - bh/2), bw, bh,
                                color=color, zorder=zorder + 3))


def draw_station(ax, x, y, zoom=STATION_LOGO_ZOOM, fallback="#ED1D24", zorder=9):
    if MTR_LOGO_LOADED and _mtr_img is not None:
        icon = OffsetImage(_mtr_img, zoom=zoom)
        icon.image.axes = ax
        ab = AnnotationBbox(icon, (x, y), frameon=False,
                            zorder=zorder, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        _draw_roundel(ax, x, y, size=60, color=fallback, zorder=zorder)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _extract_lines(geom):
    """Pull LineStrings out of any geometry type (handles GeometryCollection)."""
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "LineString":
        return geom
    if gt == "MultiLineString":
        return geom
    if hasattr(geom, "geoms"):
        lines = []
        for part in geom.geoms:
            ex = _extract_lines(part)
            if ex is None:
                continue
            if ex.geom_type == "MultiLineString":
                lines.extend(list(ex.geoms))
            elif ex.geom_type == "LineString":
                lines.append(ex)
        if lines:
            return MultiLineString(lines) if len(lines) > 1 else lines[0]
    return None


def _to_line_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    rows = []
    for _, row in gdf.iterrows():
        lg = _extract_lines(row.geometry)
        if lg is not None:
            r = row.copy()
            r.geometry = lg
            rows.append(r)
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)


def _keep_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    return gdf[gdf.geometry.geom_type.isin(
        ["LineString", "MultiLineString"])].copy().reset_index(drop=True)


# ── Safe fetch ────────────────────────────────────────────────────────────────

def _safe_fetch(lat: float, lon: float, dist: int, tags: dict,
                timeout: int = 20) -> gpd.GeoDataFrame:
    try:
        old_t = ox.settings.requests_timeout
        ox.settings.requests_timeout = timeout
        gdf = ox.features_from_point((lat, lon), dist=dist, tags=tags)
        ox.settings.requests_timeout = old_t
        if gdf is not None and not gdf.empty:
            result = gdf.to_crs(3857)
            del gdf
            return result
    except Exception as e:
        log.debug(f"[transport] fetch failed {tags}: {e}")
    return gpd.GeoDataFrame(geometry=[], crs=3857)


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


def _clean_name(val) -> Optional[str]:
    if val is None or isinstance(val, float):
        return None
    if isinstance(val, (list, tuple)):
        val = next((v for v in val if v is not None), None)
    if hasattr(val, "iloc"):
        val = val.iloc[0] if len(val) else None
    if isinstance(val, dict):
        val = val.get("name") or next(iter(val.values()), None)
    if not isinstance(val, str):
        return None
    ascii_only = "".join(c for c in val if ord(c) < 128).strip()
    return ascii_only if ascii_only else None


# ── MTR route fetch ───────────────────────────────────────────────────────────

def _fetch_mtr_routes(lat: float, lon: float, dist: int) -> gpd.GeoDataFrame:
    frames = []
    for tags in [{"railway": "rail"}, {"railway": "subway"}]:
        try:
            raw = _safe_fetch(lat, lon, dist, tags, timeout=25)
            if raw.empty:
                continue
            flat = _flatten(raw);  del raw;  gc.collect()
            log.info(f"[transport] MTR {tags}: {len(flat)} rows "
                     f"types={flat.geometry.geom_type.value_counts().to_dict()}")
            line_gdf = _to_line_gdf(flat);  del flat;  gc.collect()
            if not line_gdf.empty:
                if "name" in line_gdf.columns:
                    names = line_gdf["name"].apply(_clean_name).dropna().unique().tolist()
                    log.info(f"[transport] MTR names {tags}: {names[:10]}")
                frames.append(line_gdf)
        except Exception as e:
            log.warning(f"[transport] MTR fetch error {tags}: {e}")
            gc.collect()

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=3857)

    try:
        combined = pd.concat(frames, ignore_index=True);  del frames;  gc.collect()
        if "_osmid" in combined.columns:
            combined = (combined
                        .drop_duplicates(subset=["_osmid"])
                        .drop(columns=["_osmid"])
                        .reset_index(drop=True))
        log.info(f"[transport] MTR combined: {len(combined)} rows")
        return gpd.GeoDataFrame(combined, crs=3857)
    except Exception as e:
        log.warning(f"[transport] MTR concat error: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=3857)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_transport(
    data_type: str,
    value: str,
    lon: float = None,
    lat: float = None,
    lot_ids: List[str] = None,
    extents: List[dict] = None,
    radius_m: Optional[int] = None,
):
    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)
    fetch_r  = radius_m if radius_m else MAP_RADIUS
    log.info(f"[transport] {data_type} {value} → lon={lon:.5f} lat={lat:.5f} r={fetch_r}m")

    # ── Site polygon ──────────────────────────────────────────────────────────
    try:
        lot_gdf = get_lot_boundary(lon, lat, data_type, extents)
    except Exception as e:
        log.warning(f"[transport] get_lot_boundary: {e}")
        lot_gdf = None

    if lot_gdf is not None and not lot_gdf.empty:
        site_geom  = lot_gdf.geometry.iloc[0]
        site_gdf   = lot_gdf
        site_point = site_geom.centroid
    else:
        site_point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]
        site_geom  = site_point.buffer(120)   # visible fallback circle
        site_gdf   = gpd.GeoDataFrame(geometry=[site_geom], crs=3857)

    # ── Map extent ────────────────────────────────────────────────────────────
    xmin = site_point.x - HALF_W
    xmax = site_point.x + HALF_W
    ymin = site_point.y - HALF_H
    ymax = site_point.y + HALF_H
    clip_box = box(xmin, ymin, xmax, ymax)
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs=3857)

    # ── Fetch layers ──────────────────────────────────────────────────────────
    log.info("[transport] Fetching roads...")
    roads = _keep_lines(_flatten(_safe_fetch(lat, lon, fetch_r, {
        "highway": ["motorway", "trunk", "primary", "secondary", "tertiary"],
    }, timeout=20)))
    gc.collect()

    # Skip light_rail — it times out (3+ min) and rarely appears in HK Island
    light_rail = gpd.GeoDataFrame(geometry=[], crs=3857)
    light_rail_plotted = False

    log.info("[transport] Fetching MTR routes...")
    mtr_routes = _fetch_mtr_routes(lat, lon, fetch_r)
    gc.collect()

    log.info("[transport] Fetching stations...")
    stations = _flatten(_safe_fetch(lat, lon, fetch_r, {"railway": "station"}, timeout=20))
    gc.collect()

    log.info("[transport] Rendering...")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#f4f4f4")
    ax.set_facecolor("#f4f4f4")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # No-label basemap
    try:
        cx.add_basemap(ax, crs="EPSG:3857",
                       source=cx.providers.CartoDB.PositronNoLabels,
                       zoom=14, alpha=0.5)
    except Exception as e:
        log.warning(f"[transport] basemap failed: {e}")

    # ── Roads ─────────────────────────────────────────────────────────────────
    if not roads.empty:
        try:
            rc = gpd.clip(roads, clip_gdf)
            (rc if not rc.empty else roads).plot(
                ax=ax, color=COLOR_ROADS, linewidth=2.0, zorder=3, alpha=0.85)
        except Exception:
            pass
    del roads;  gc.collect()

    # ── MTR routes ────────────────────────────────────────────────────────────
    lines_on_map: dict = {}      # color → display label (for legend)

    if not mtr_routes.empty:
        try:
            try:
                mtr_vis = gpd.clip(mtr_routes, clip_gdf)
            except Exception:
                mtr_vis = mtr_routes.copy()

            log.info(f"[transport] mtr_vis: {len(mtr_vis)} rows")

            if not mtr_vis.empty:
                placed_lbl = []

                name_col = None
                for c in ["name", "name:en"]:
                    if c in mtr_vis.columns and mtr_vis[c].apply(_clean_name).notna().any():
                        name_col = c
                        break

                if name_col is None:
                    mtr_vis.plot(ax=ax, color="white",           linewidth=8,   zorder=4)
                    mtr_vis.plot(ax=ax, color=DEFAULT_MTR_COLOR, linewidth=4.5, zorder=5)
                    lines_on_map[DEFAULT_MTR_COLOR] = "MTR"
                else:
                    mtr_vis        = mtr_vis.copy()
                    mtr_vis["_cn"] = mtr_vis[name_col].apply(_clean_name)
                    named          = mtr_vis[mtr_vis["_cn"].notna()]
                    unnamed        = mtr_vis[mtr_vis["_cn"].isna()]

                    for cname, grp in named.groupby("_cn", sort=False):
                        lc = get_mtr_color(cname)
                        log.info(f"[transport] Drawing '{cname}' → color={lc} n={len(grp)}")
                        grp.plot(ax=ax, color="white", linewidth=8,   zorder=4)
                        grp.plot(ax=ax, color=lc,      linewidth=4.5, zorder=5)

                        # Legend entry — deduplicated by color
                        if lc not in lines_on_map:
                            official = cname.title()
                            for k, _c, lbl in MTR_LEGEND_LINES:
                                if k in cname.lower():
                                    official = lbl
                                    break
                            lines_on_map[lc] = official

                        # Line label annotation
                        try:
                            merged = grp.geometry.unary_union
                            if merged is None or merged.is_empty or merged.length < 600:
                                continue
                            mid   = merged.interpolate(0.5, normalized=True)
                            off_y = sum(150 for pp in placed_lbl if mid.distance(pp) < 500)
                            lp    = Point(mid.x, mid.y + off_y)
                            placed_lbl.append(lp)
                            if xmin <= lp.x <= xmax and ymin <= lp.y <= ymax:
                                ax.text(lp.x, lp.y, _make_label(cname),
                                        fontsize=9, weight="bold", color=lc,
                                        ha="center", va="center", zorder=12,
                                        bbox=dict(facecolor="white", edgecolor="none",
                                                  alpha=0.85, pad=2))
                        except Exception as le:
                            log.debug(f"[transport] label '{cname}': {le}")

                    # Unnamed only if no named lines visible at all
                    if not unnamed.empty and named.empty:
                        unnamed.plot(ax=ax, color="white",           linewidth=8,   zorder=4)
                        unnamed.plot(ax=ax, color=DEFAULT_MTR_COLOR, linewidth=4.5, zorder=5)
                        lines_on_map[DEFAULT_MTR_COLOR] = "MTR"

        except Exception as e:
            log.warning(f"[transport] MTR render error: {e}")

    del mtr_routes;  gc.collect()

    # ── Stations ──────────────────────────────────────────────────────────────
    if not stations.empty:
        try:
            stns        = stations.copy()
            stns["_cx"] = stns.geometry.centroid.x
            stns["_cy"] = stns.geometry.centroid.y
            in_view = stns[
                (stns["_cx"] >= xmin) & (stns["_cx"] <= xmax) &
                (stns["_cy"] >= ymin) & (stns["_cy"] <= ymax)
            ]
            placed_pts = []
            for _, row in in_view.iterrows():
                sx, sy = row["_cx"], row["_cy"]
                pt = Point(sx, sy)
                if any(pt.distance(p) < STATION_MIN_DISTANCE for p in placed_pts):
                    continue
                placed_pts.append(pt)
                fb     = "#ED1D24"
                name_s = _clean_name(row.get("name"))
                if name_s:
                    c = get_mtr_color(name_s)
                    if c != DEFAULT_MTR_COLOR:
                        fb = c
                draw_station(ax, sx, sy, zoom=STATION_LOGO_ZOOM, fallback=fb, zorder=9)
                if name_s:
                    ax.text(sx, sy - 130, name_s,
                            fontsize=7.5, weight="bold",
                            ha="center", va="top", color="#333333", zorder=10,
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.75, pad=1.5))
        except Exception as e:
            log.warning(f"[transport] Station render: {e}")
    del stations;  gc.collect()

    # ── Site — always drawn on top ────────────────────────────────────────────
    try:
        site_gdf.plot(ax=ax, facecolor=COLOR_SITE, edgecolor="white",
                      linewidth=2.5, zorder=13)
        centroid = site_geom.centroid
        ax.text(centroid.x, centroid.y - 160, "SITE",
                fontsize=14, weight="bold", color="black",
                ha="center", va="top", zorder=14,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))
    except Exception as e:
        log.warning(f"[transport] Site render: {e}")

    # ── North arrow ───────────────────────────────────────────────────────────
    ax.annotate("", xy=(0.07, 0.85), xytext=(0.07, 0.80),
                xycoords=ax.transAxes,
                arrowprops=dict(facecolor="black", width=1.5,
                                headwidth=8, headlength=10))
    ax.text(0.07, 0.86, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, weight="bold")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []

    # MTR lines — in official order, deduplicated by color
    seen_lc = set()
    for _, color, label in MTR_LEGEND_LINES:
        if color in lines_on_map and color not in seen_lc:
            legend_handles.append(
                mlines.Line2D([], [], color=color, linewidth=4, label=label))
            seen_lc.add(color)

    # Generic MTR fallback only if nothing named was found
    if DEFAULT_MTR_COLOR in lines_on_map and not seen_lc:
        legend_handles.append(
            mlines.Line2D([], [], color=DEFAULT_MTR_COLOR, linewidth=4, label="MTR"))

    legend_handles.append(
        mlines.Line2D([], [], color=COLOR_ROADS, linewidth=4, label="Vehicle Circulation"))
    legend_handles.append(
        mpatches.Patch(facecolor=COLOR_SITE, label="Site"))

    handler_map = {}
    if MTR_LOGO_LOADED and _mtr_img is not None:
        try:
            raw_arr = (_mtr_img * 255).astype(np.uint8) if _mtr_img.dtype != np.uint8 else _mtr_img
            thumb   = np.array(Image.fromarray(raw_arr).resize((20, 20), Image.LANCZOS))

            class _LogoHandle(mlines.Line2D):
                pass

            class _LogoHandler(lh.HandlerBase):
                def create_artists(self, legend, orig_handle, xd, yd, w, h, fontsize, trans):
                    imgbox = OffsetImage(thumb, zoom=0.8)
                    ab = AnnotationBbox(imgbox, (xd + w / 2, yd + h / 2),
                                        xycoords=trans, frameon=False)
                    return [ab]

            legend_handles.append(_LogoHandle([], [], label="MTR Station"))
            handler_map = {_LogoHandle: _LogoHandler()}
        except Exception:
            legend_handles.append(
                mlines.Line2D([], [], marker="o", linestyle="None",
                              markerfacecolor="#ED1D24", markeredgecolor="white",
                              markeredgewidth=2, markersize=10, label="MTR Station"))
    else:
        legend_handles.append(
            mlines.Line2D([], [], marker="o", linestyle="None",
                          markerfacecolor="#ED1D24", markeredgecolor="white",
                          markeredgewidth=2, markersize=10, label="MTR Station"))

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            handler_map=handler_map if handler_map else None,
            loc="lower left", bbox_to_anchor=(0.02, 0.02),
            frameon=True, facecolor="white", edgecolor="black",
            fontsize=10, title_fontsize=10,
            labelspacing=0.4, handlelength=1.8, handleheight=1.0,
            borderpad=0.6, title="Legend",
        )
        legend.get_frame().set_linewidth(2)

    ax.set_title(f"SITE ANALYSIS – Transportation ({data_type} {value})",
                 fontsize=18, weight="bold")
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    log.info("[transport] Done.")
    return buf
