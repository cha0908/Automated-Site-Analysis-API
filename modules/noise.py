# ============================================================
# modules/noise.py
# Traffic-Based Environmental Noise Modelling Engine v2.11
# Adapted for Automated-Site-Analysis-API
# ============================================================
# FIX U — Plot zoom: added "plot_radius" config key (default 100m)
#          Map view is now cropped to plot_radius independently of
#          study_radius (150m). Matches the tight zoom of the Colab
#          output. Change plot_radius in CFG to adjust zoom level.
# + all v2.10 fixes retained
# ============================================================

import gc
import io
import re
import time
import warnings
import logging

import numpy as np
import requests
import geopandas as gpd
import contextily as cx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

from io import BytesIO
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.validation import make_valid
from scipy.ndimage import gaussian_filter

from modules.resolver import resolve_location, get_lot_boundary

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ox.settings.use_cache   = True
ox.settings.log_console = False


# ============================================================
# CONFIG
# ============================================================

CFG = {
    # WFS endpoints (CSDI HK)
    "atc_wfs_url": (
        "https://portal.csdi.gov.hk/server/services/common/"
        "td_rcd_1638950691670_3837/MapServer/WFSServer"
        "?service=wfs&request=GetFeature&typenames=ATC_STATION_PT"
        "&outputFormat=geojson&srsName=EPSG:4326&count=10000"
    ),
    "lnrs_wfs_url": (
        "https://portal.csdi.gov.hk/server/services/common/"
        "epd_rcd_1696984809000_78322/MapServer/WFSServer"
        "?service=wfs&request=GetFeature&typenames=noise_lnrs"
        "&outputFormat=geojson&count=10000"
    ),
    "wfs_timeout": 30,

    # Propagation (unchanged — noise is computed over full study_radius)
    "study_radius":      150,
    "grid_resolution":     5,
    "densify_spacing":     5.0,
    "road_mask_distance": 80.0,

    # Acoustics
    "ground_absorption":  0.6,
    "ground_term_coeff":  1.5,
    "base_reflection":    2.0,
    "smooth_sigma":        1.5,
    "noise_floor_db":     45.0,

    # Canyon
    "canyon_buffer_m":   25.0,
    "canyon_full_area": 2000.0,
    "canyon_max_bonus":   8.0,

    # ATC snap + fallback
    "atc_snap_threshold": 500,
    "default_flow":       3500,
    "default_heavy_pct":  0.18,
    "default_speed":        40,

    # Road-type fallback tables
    "road_flow_table": {
        "motorway":      9000, "trunk":         7000,
        "primary":       5500, "secondary":     4000,
        "tertiary":      2500, "residential":   1000,
        "unclassified":  1500, "living_street":  300,
        "service":        500, "default":       3500,
    },
    "road_speed_table": {
        "motorway": 110, "trunk":         80, "primary":       60,
        "secondary":  50, "tertiary":     40, "residential":   30,
        "unclassified":40, "living_street":20, "service":      20,
        "default":    40,
    },

    # LNRS
    "lnrs_correction_db": -3.0,

    # Visualisation
    # FIX U: plot_radius controls map view crop, independent of study_radius.
    # study_radius=150 fetches & propagates over 150m, but the map only
    # displays the inner plot_radius metres — matching the Colab tight zoom.
    # Decrease to zoom in more, increase to show more context.
    "plot_radius":         100,   # metres shown on map (was implicitly 150 = study_radius)
    "basemap_zoom":        19,
    "contour_levels_min":  50,
    "contour_levels_max":  70,
    "colorbar_max_db":     70,
    "contour_step":         5,
    "contour_label_step":  10,
    "output_dpi":          130,
}


# ============================================================
# UTILITIES
# ============================================================

def _normalise_station_id(raw):
    s = str(raw).strip()
    s = re.sub(r'^[A-Za-z_\-]+', '', s)
    s = s.lstrip('0') or '0'
    return s.lower()


def _densify_line(line, spacing):
    total = line.length
    if total < spacing:
        return np.array(line.coords)
    n_pts = max(2, int(np.ceil(total / spacing)) + 1)
    dists = np.linspace(0, total, n_pts)
    return np.array([line.interpolate(d).coords[0] for d in dists])


def _hw_lookup(hw, table):
    if isinstance(hw, list):
        hw = hw[0]
    return float(table.get(str(hw), table["default"]))


# ============================================================
# PHASE 1A — ATC WFS LOADER
# ============================================================

class ATCWFSLoader:
    _FLOW_COLS  = ["AM_PEAK", "PM_PEAK", "AADT", "DAILY_FLOW", "ADT", "TOTAL",
                   "AM Peak", "PM Peak", "Annual Average Daily Traffic"]
    _HEAVY_COLS = ["COMMERCIAL_PCT", "HEAVY_PCT", "HV_PCT", "CV_PCT",
                   "Commercial %", "Heavy %", "Heavy Vehicle %"]
    _SPEED_COLS = ["SPEED", "Speed", "Speed (km/h)", "SPEED_KMH"]
    _ID_COLS    = ["ATC_STATION_NO", "STATION_NO", "STATIONNO", "STATION_ID",
                   "STN_NO", "OBJECTID", "FID", "NO", "ID"]

    def __init__(self, cfg):
        self.url     = cfg["atc_wfs_url"]
        self.timeout = cfg.get("wfs_timeout", 30)

    def _find_col(self, cols, candidates):
        for cand in candidates:
            if cand in cols:
                return cand
        for cand in candidates:
            for c in cols:
                if cand.lower() in c.lower():
                    return c
        return None

    def load(self):
        try:
            r = requests.get(self.url, timeout=self.timeout)
            r.raise_for_status()
            gdf = gpd.read_file(io.StringIO(r.text))
            if not len(gdf):
                log.warning("  ATC WFS: 0 features")
                return {}
            if gdf.crs is None:
                gdf = gdf.set_crs(4326)
            gdf = gdf.to_crs(3857)
            log.info(f"  ATC WFS: {len(gdf)} stations  cols={list(gdf.columns[:8])}")
        except Exception as e:
            log.warning(f"  ATC WFS failed: {e} — road-type fallback active")
            return {}

        cols      = list(gdf.columns)
        id_col    = self._find_col(cols, self._ID_COLS)
        heavy_col = self._find_col(cols, self._HEAVY_COLS)
        speed_col = self._find_col(cols, self._SPEED_COLS)
        flow_cols = [c for c in self._FLOW_COLS if c in cols]

        result = {}
        for _, row in gdf.iterrows():
            sid = _normalise_station_id(row[id_col] if id_col else str(row.name))

            flow = None
            for fc in flow_cols:
                try:
                    v = float(row[fc])
                    if v == v and v > 0:
                        if fc.upper() in ("AADT", "ADT", "TOTAL", "DAILY_FLOW",
                                          "ANNUAL AVERAGE DAILY TRAFFIC"):
                            v = v / 16.0
                        flow = v
                        break
                except Exception:
                    continue

            heavy = None
            if heavy_col:
                try:
                    hv = float(row[heavy_col])
                    if hv == hv:
                        heavy = hv / 100.0 if hv > 1 else hv
                except Exception:
                    pass

            speed = None
            if speed_col:
                try:
                    sp = float(row[speed_col])
                    if sp == sp:
                        speed = sp
                except Exception:
                    pass

            result[sid] = {
                "flow":      flow,
                "heavy_pct": heavy,
                "speed":     speed,
                "x":         row.geometry.x if row.geometry else None,
                "y":         row.geometry.y if row.geometry else None,
            }

        log.info(f"  ATC WFS parsed: {len(result)} stations")
        return result


# ============================================================
# PHASE 1B — LNRS WFS LOADER
# ============================================================

class LNRSWFSLoader:
    _TYPENAMES = ["noise_lnrs", "NOISE_LNRS", "td:noise_lnrs",
                  "epd:noise_lnrs", "lnrs", "LNRS"]

    def __init__(self, cfg):
        self.url     = cfg["lnrs_wfs_url"]
        self.timeout = cfg.get("wfs_timeout", 30)

    def load(self):
        for tn in self._TYPENAMES:
            url = re.sub(r'typenames=[^&]+', f'typenames={tn}', self.url)
            try:
                r = requests.get(url, timeout=self.timeout)
                if r.status_code != 200:
                    continue
                gdf = gpd.read_file(io.StringIO(r.text))
                if len(gdf) > 0:
                    if gdf.crs is None:
                        gdf = gdf.set_crs(4326)
                    gdf = gdf.to_crs(3857)
                    log.info(f"  LNRS WFS '{tn}': {len(gdf)} zones")
                    return gdf
            except Exception as e:
                log.warning(f"  LNRS WFS '{tn}': {e}")

        log.warning("  LNRS WFS: all typenames failed — no LNRS correction")
        return gpd.GeoDataFrame(geometry=[], crs=3857)


# ============================================================
# PHASE 2A — TRAFFIC ASSIGNMENT
# ============================================================

class TrafficAssigner:
    def __init__(self, atc_data, cfg):
        self.atc  = atc_data
        self.cfg  = cfg
        self.snap = float(cfg.get("atc_snap_threshold", 500))

        stations = [
            (v["x"], v["y"], k)
            for k, v in atc_data.items()
            if v.get("x") is not None and v.get("y") is not None
        ]
        if stations:
            self._coords = np.array([(s[0], s[1]) for s in stations])
            self._sids   = [s[2] for s in stations]
        else:
            self._coords = None
            self._sids   = []

    def _nearest(self, cx, cy):
        if self._coords is None or not len(self._coords):
            return None
        d = np.hypot(*(self._coords - [cx, cy]).T)
        i = int(np.argmin(d))
        return self._sids[i] if d[i] <= self.snap else None

    def _val(self, v, default):
        if v is None:
            return default
        try:
            f = float(v)
            return default if f != f else f
        except Exception:
            return default

    def assign(self, roads):
        flows, heavys, speeds = [], [], []
        matched = 0

        for _, row in roads.iterrows():
            c   = row.geometry.centroid
            sid = self._nearest(c.x, c.y)
            rec = self.atc.get(sid) if sid else None
            hw  = row.get("highway", "default")

            flow = self._val(
                rec.get("flow") if rec else None,
                _hw_lookup(hw, self.cfg["road_flow_table"]),
            )
            if rec and rec.get("flow") is not None:
                matched += 1

            heavy = np.clip(
                self._val(
                    rec.get("heavy_pct") if rec else None,
                    self.cfg["default_heavy_pct"],
                ), 0, 1,
            )

            speed = self._val(
                rec.get("speed") if rec else None,
                _hw_lookup(hw, self.cfg["road_speed_table"]),
            )

            flows.append(flow)
            heavys.append(heavy)
            speeds.append(speed)

        log.info(f"  Traffic: {matched}/{len(roads)} roads ATC-matched")
        roads = roads.copy()
        roads["flow"]      = flows
        roads["heavy_pct"] = heavys
        roads["speed"]     = speeds
        roads["lnrs_corr"] = 0.0
        return roads


# ============================================================
# PHASE 2B — LNRS ASSIGNMENT
# ============================================================

class LNRSAssigner:
    def __init__(self, lnrs_gdf, cfg):
        self.corr  = float(cfg.get("lnrs_correction_db", -3.0))
        self._tree = None
        if len(lnrs_gdf) > 0:
            self._tree = STRtree(lnrs_gdf.geometry.tolist())
            log.info(f"  LNRS tree: {len(lnrs_gdf)} zones")

    def assign(self, roads):
        if not self._tree:
            return roads
        corr = [
            self.corr if len(self._tree.query(g, predicate="intersects")) > 0
            else 0.0
            for g in roads.geometry
        ]
        roads = roads.copy()
        roads["lnrs_corr"] = corr
        log.info(
            f"  LNRS: {sum(v < 0 for v in corr)}/{len(roads)} roads corrected"
        )
        return roads


# ============================================================
# PHASE 2C — CANYON ASSIGNMENT
# ============================================================

class CanyonAssigner:
    def __init__(self, buildings_gdf, cfg):
        self.cfg    = cfg
        self._tree  = None
        self._geoms = []
        if buildings_gdf is not None and len(buildings_gdf) > 0:
            valid = [g for g in buildings_gdf.geometry
                     if g is not None and not g.is_empty]
            if valid:
                self._geoms = valid
                self._tree  = STRtree(valid)
                log.info(f"  CanyonAssigner: {len(valid)} buildings")

    def assign(self, roads):
        if self._tree is None:
            roads = roads.copy()
            roads["canyon_gain"] = 0.0
            return roads

        buf_m   = float(self.cfg["canyon_buffer_m"])
        full_a  = float(self.cfg["canyon_full_area"])
        max_bon = float(self.cfg["canyon_max_bonus"])
        gains   = []

        for _, row in roads.iterrows():
            g = row.geometry
            if g is None or g.is_empty:
                gains.append(0.0); continue
            try:
                corridor = g.buffer(buf_m)
            except Exception:
                gains.append(0.0); continue
            hits = self._tree.query(corridor, predicate="intersects")
            if not len(hits):
                gains.append(0.0); continue
            area = sum(corridor.intersection(self._geoms[i]).area for i in hits)
            gains.append(max_bon * min(area / full_a, 1.0))

        roads = roads.copy()
        roads["canyon_gain"] = gains
        log.info(f"  Canyon: max={max(gains):.1f}  mean={np.mean(gains):.1f} dB")
        return roads


# ============================================================
# PHASE 3 — EMISSION
# ============================================================

class EmissionEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _log10s(x):
        return np.log10(np.maximum(x, 1e-9))

    def compute(self, roads):
        flow  = roads["flow"].values.astype(float)
        hpct  = roads["heavy_pct"].values.astype(float)
        speed = roads["speed"].values.astype(float)
        corr  = roads["lnrs_corr"].values.astype(float)
        cany  = roads["canyon_gain"].values.astype(float)

        Ql = np.maximum(flow * (1 - hpct), 1)
        Qh = np.maximum(flow * hpct,       1)
        Ll = 27.7 + 10 * self._log10s(Ql) + 0.02 * speed
        Lh = 23.1 + 10 * self._log10s(Qh) + 0.08 * speed
        Ls = 10 * self._log10s(10 ** (Ll / 10) + 10 ** (Lh / 10))
        Lk = Ls + corr + cany

        roads = roads.copy()
        roads["L_light"]  = Ll
        roads["L_heavy"]  = Lh
        roads["L_source"] = Ls
        roads["L_link"]   = Lk

        log.info(
            f"  Emission: flow {flow.min():.0f}-{flow.max():.0f} veh/hr | "
            f"canyon {cany.min():.1f}-{cany.max():.1f} dB | "
            f"L_link {Lk.min():.1f}-{Lk.max():.1f} dB(A)"
        )
        return roads


# ============================================================
# PHASE 4 — PROPAGATION
# ============================================================

class PropagationEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def _grid(self, bounds):
        minx, miny, maxx, maxy = bounds
        res = float(self.cfg["grid_resolution"])
        return np.meshgrid(
            np.arange(minx, maxx, res),
            np.arange(miny, maxy, res),
        )

    def _road_proximity_mask(self, X, Y, road_lines):
        dist_thresh = self.cfg.get("road_mask_distance", 80.0)
        if dist_thresh is None or dist_thresh <= 0:
            return np.ones_like(X, dtype=bool)
        dist_thresh = float(dist_thresh)
        min_dist = np.full(X.shape, np.inf, dtype=np.float64)

        for coords, _ in road_lines:
            n = len(coords) - 1
            if n < 1:
                continue
            for i in range(n):
                x1, y1 = coords[i];  x2, y2 = coords[i + 1]
                dx, dy = x2 - x1, y2 - y1
                q = dx * dx + dy * dy
                if q < 1e-6:
                    d = np.sqrt((X - x1) ** 2 + (Y - y1) ** 2)
                else:
                    t = np.clip(((X - x1) * dx + (Y - y1) * dy) / q, 0, 1)
                    d = np.sqrt((X - x1 - t * dx) ** 2 + (Y - y1 - t * dy) ** 2)
                min_dist = np.minimum(min_dist, d)

        mask = min_dist <= dist_thresh
        log.info(
            f"  Road mask: {100*mask.sum()/mask.size:.1f}% cells "
            f"within {dist_thresh}m of a road"
        )
        return mask

    def _extract_lines(self, roads):
        spacing = float(self.cfg.get("densify_spacing", 5.0))
        out = []; null = inv = short = 0

        for _, row in roads.iterrows():
            g = row.geometry
            L = float(row["L_link"])
            if g is None:
                null += 1; continue
            if not g.is_valid:
                try:
                    g = make_valid(g)
                except Exception:
                    inv += 1; continue
            if g is None or g.is_empty or not g.is_valid:
                inv += 1; continue
            if g.geom_type not in ("LineString", "MultiLineString"):
                inv += 1; continue
            parts = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
            for p in parts:
                if p.geom_type != "LineString":
                    inv += 1; continue
                if p.length < 1e-3:
                    short += 1; continue
                out.append((_densify_line(p, spacing), L))

        log.info(
            f"  Lines: {len(out)} valid | "
            f"null={null} invalid={inv} short={short}"
        )
        return out

    @staticmethod
    def _seg_dist(X, Y, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        q = dx * dx + dy * dy
        if q < 1e-6:
            return np.sqrt((X - x1) ** 2 + (Y - y1) ** 2)
        t = np.clip(((X - x1) * dx + (Y - y1) * dy) / q, 0, 1)
        return np.sqrt((X - x1 - t * dx) ** 2 + (Y - y1 - t * dy) ** 2)

    def run(self, roads, site_polygon):
        bounds = site_polygon.buffer(self.cfg["study_radius"]).bounds
        X, Y   = self._grid(bounds)
        energy = np.zeros_like(X, dtype=np.float64)
        G      = float(self.cfg["ground_absorption"])
        GC     = float(self.cfg["ground_term_coeff"])
        Rg     = float(self.cfg["base_reflection"])

        lines = self._extract_lines(roads)
        log.info(f"  Propagation: {len(lines)} sources | {X.size:,} cells")
        t0 = time.time()

        for coords, L_link in lines:
            n = len(coords) - 1
            if n < 1:
                continue
            le = np.zeros_like(X, dtype=np.float64)
            for i in range(n):
                x1, y1 = coords[i];  x2, y2 = coords[i + 1]
                d  = self._seg_dist(X, Y, x1, y1, x2, y2)
                Lc = (L_link
                      - 20 * np.log10(d + 1)
                      - G * GC * np.log10(d + 1)
                      + Rg)
                le += 10 ** (Lc / 10)
            energy += le / n

        noise = 10 * np.log10(energy + 1e-12)

        sigma = float(self.cfg.get("smooth_sigma", 1.5))
        if sigma > 0:
            noise = gaussian_filter(noise, sigma=sigma)

        road_mask = self._road_proximity_mask(X, Y, lines)
        noise[~road_mask] = np.nan

        nf = float(self.cfg.get("noise_floor_db", 45.0))
        noise[np.isfinite(noise) & (noise < nf)] = np.nan

        v = noise[np.isfinite(noise)]
        if len(v):
            log.info(
                f"  Noise >={nf} dB: "
                f"min={v.min():.1f} max={v.max():.1f} mean={v.mean():.1f} dB(A)"
            )
        log.info(f"  Propagation done: {time.time() - t0:.1f}s")
        gc.collect()
        return X, Y, noise


# ============================================================
# PHASE 5 — VISUALISATION
# ============================================================

class NoiseVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def _facade_levels(self, X, Y, noise, bld):
        if not len(bld):
            return bld
        res = float(self.cfg["grid_resolution"])
        x0, y0 = float(X[0, 0]), float(Y[0, 0])
        vals = []
        for c in bld.geometry.centroid:
            col = int(np.clip(round((c.x - x0) / res), 0, noise.shape[1] - 1))
            row = int(np.clip(round((c.y - y0) / res), 0, noise.shape[0] - 1))
            v   = float(noise[row, col])
            vals.append(v if np.isfinite(v) else 0.0)
        bld = bld.copy()
        bld["facade_db"] = vals
        return bld

    def _get_levels(self, noise):
        step = self.cfg["contour_step"]
        cmin = int(self.cfg.get("contour_levels_min") or 50)
        cmax = int(
            self.cfg.get("contour_levels_max")
            or self.cfg.get("colorbar_max_db", 70)
        )
        if cmax - cmin < step * 2:
            cmax = cmin + step * 6
        levels = np.arange(cmin, cmax + step, step)
        v = noise[np.isfinite(noise)]
        dmin = float(v.min()) if len(v) else float(cmin)
        dmax = float(v.max()) if len(v) else float(cmax)
        log.info(f"  Colorbar: {cmin}-{cmax} dB | data: {dmin:.1f}-{dmax:.1f}")
        return levels

    def _add_basemap(self, ax):
        for name, prov in [
            ("CartoDB PositronNoLabels", cx.providers.CartoDB.PositronNoLabels),
            ("CartoDB Positron",         cx.providers.CartoDB.Positron),
            ("OpenStreetMap",            cx.providers.OpenStreetMap.Mapnik),
        ]:
            try:
                cx.add_basemap(
                    ax, source=prov, crs=3857,
                    zoom=self.cfg["basemap_zoom"], alpha=1.0,
                )
                log.info(f"  Basemap: {name}")
                return
            except Exception as e:
                log.warning(f"  Basemap {name}: {e}")

    def render(self, X, Y, noise, site_poly, site_gdf,
               bld, roads, meta):
        levels = self._get_levels(noise)
        bld    = self._facade_levels(X, Y, noise, bld)

        fig, ax = plt.subplots(figsize=(11, 11))
        fig.patch.set_facecolor("#f0f0ee")
        ax.set_facecolor("#e8f0e8")

        c = site_poly.centroid

        # FIX U: use plot_radius for the map VIEW, not study_radius.
        # Propagation still runs over the full study_radius (150m) so
        # noise near edges is accurate, but the displayed area is the
        # tighter plot_radius crop — matching the Colab zoom level.
        R_study = float(self.cfg["study_radius"])
        R_plot  = float(self.cfg.get("plot_radius", R_study))  # default: no crop
        ax.set_xlim(c.x - R_plot, c.x + R_plot)
        ax.set_ylim(c.y - R_plot, c.y + R_plot)
        ax.set_aspect("equal")
        self._add_basemap(ax)

        nc   = np.where(np.isfinite(noise),
                        np.clip(noise, levels[0], levels[-1]), np.nan)
        cont = ax.contourf(X, Y, nc, levels=levels,
                           cmap="RdYlGn_r", alpha=0.60, extend="both")

        lbl_lvls = [
            l for l in levels
            if int(round(l)) % self.cfg.get("contour_label_step", 10) == 0
        ]
        ax.contour(X, Y, nc, levels=levels,
                   colors="black", linewidths=0.4, alpha=0.25)
        cs = ax.contour(X, Y, nc, levels=lbl_lvls,
                        colors="black", linewidths=0.9, alpha=0.55)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%d dB", inline_spacing=4)

        if len(bld) > 0 and "facade_db" in bld.columns:
            norm = mcolors.Normalize(vmin=levels[0], vmax=levels[-1])
            bld.plot(ax=ax, column="facade_db", cmap="RdYlGn_r",
                     norm=norm, linewidth=0.9, edgecolor="#3a3a3a",
                     alpha=0.88, zorder=6)

        if len(roads) > 0:
            roads.plot(ax=ax, color="#333333", linewidth=0.7,
                       alpha=0.45, zorder=7)

        site_gdf.plot(ax=ax, facecolor="#e63946",
                      edgecolor="#ffffff", linewidth=1.5, zorder=10)
        ax.text(c.x, c.y, "SITE", fontsize=13, weight="bold",
                color="white", ha="center", va="center", zorder=20,
                path_effects=[pe.withStroke(linewidth=3,
                                            foreground="#e63946")])

        cbar = plt.colorbar(cont, ax=ax, fraction=0.028, pad=0.02, aspect=30)
        cbar.set_label("Noise Level  Leq dB(A)", fontsize=10, labelpad=8)
        cbar.set_ticks(levels)
        cbar.set_ticklabels([f"{int(l)}" for l in levels])
        cbar.ax.tick_params(labelsize=8)

        epd_items = []
        for thresh, lbl, col in [
            (65, "65 dB(A) - HK EPD Day limit",   "#e67e22"),
            (70, "70 dB(A) - HK EPD Night limit",  "#c0392b"),
        ]:
            if levels[0] <= thresh <= levels[-1]:
                pos = (thresh - levels[0]) / (levels[-1] - levels[0])
                cbar.ax.axhline(y=pos, color=col, lw=2.0, ls="--", zorder=10)
                epd_items.append(
                    mpatches.Patch(facecolor=col, edgecolor=col, label=lbl)
                )

        if epd_items:
            leg = ax.legend(
                handles=epd_items, loc="lower left",
                fontsize=7, framealpha=0.88,
                edgecolor="#aaaaaa", facecolor="white",
                title="EPD Noise Limits", title_fontsize=7,
            )
            leg.set_zorder(35)

        # North arrow — scaled to plot_radius
        xl, yl = ax.get_xlim(), ax.get_ylim()
        nx = xl[0] + 0.06 * (xl[1] - xl[0])
        ny = yl[0] + 0.88 * (yl[1] - yl[0])
        La = R_plot * 0.055
        ax.annotate(
            "", xy=(nx, ny + La), xytext=(nx, ny - La),
            arrowprops=dict(arrowstyle="-|>", color="#111",
                            lw=2, mutation_scale=14),
            zorder=25,
        )
        ax.text(nx, ny + La * 1.45, "N", fontsize=11, weight="bold",
                color="#111", ha="center", va="center", zorder=25)

        v = noise[np.isfinite(noise)]
        if len(v):
            ax.text(
                xl[1] - 0.01 * (xl[1] - xl[0]),
                yl[0] + 0.01 * (yl[1] - yl[0]),
                f"Max:  {v.max():.1f} dB(A)\n"
                f"Mean: {v.mean():.1f} dB(A)\n"
                f"Min:  {v.min():.1f} dB(A)\n"
                f"Src:  {meta.get('L_source_range', '-')}",
                fontsize=8, ha="right", va="bottom", zorder=30,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#aaa", alpha=0.88),
            )

        mask_d = self.cfg.get("road_mask_distance", 80)
        cb_max = int(self.cfg.get("colorbar_max_db", 70))
        ax.set_title(
            f"Near-Site Environmental Noise Assessment\n"
            f"{meta['type']} {meta['value']}  "
            f"[R={int(R_study)}m  view={int(R_plot)}m  "
            f"LNRS={meta.get('lnrs_roads', 0)} roads]\n"
            f"Canyon per-road - Road mask {mask_d}m - "
            f"Colorbar 50-{cb_max} dB  [v2.11]",
            fontsize=12, weight="bold", pad=10,
        )
        ax.text(
            xl[0], yl[0],
            "(C) OpenStreetMap  (C) CARTO  |  ATC+LNRS: (C) CSDI HK\n"
            "Screening model - not ISO 9613 compliant",
            fontsize=6.5, color="#777", va="bottom", zorder=30,
        )

        ax.set_axis_off()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png",
                    dpi=self.cfg["output_dpi"],
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        gc.collect()
        buf.seek(0)
        return buf


# ============================================================
# PUBLIC API ENTRY POINT
# ============================================================

def generate_noise(data_type: str, value: str,
                   lon: float = None, lat: float = None,
                   lot_ids: list = None, extents: list = None) -> BytesIO:
    cfg = CFG.copy()

    lon, lat = resolve_location(data_type, value, lon, lat, lot_ids, extents)



    site_polygon = None
    site_gdf     = None
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    lot_gdf = get_lot_boundary(lon, lat, data_type, extents)
    if lot_gdf is not None:
        geom = lot_gdf.geometry.iloc[0]
        if geom is not None and geom.geom_type in ("Polygon", "MultiPolygon") and geom.area > 100:
            site_polygon = geom
            site_gdf     = lot_gdf
            log.info(f"  Site: lot boundary (area={geom.area:.0f}m²)")
        else:
            log.info(f"  Site: lot boundary rejected (type={geom.geom_type if geom else None}) — trying OSM")

    if site_polygon is None:
        try:
            cands = ox.features_from_point(
                (lat, lon), dist=80, tags={"building": True}
            ).to_crs(3857)
            cands["area"] = cands.area
            cands = cands[cands.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not len(cands):
                raise ValueError("no building polygons near site")
            site_polygon = cands.sort_values("area", ascending=False).geometry.iloc[0]
            log.info(f"  Site: OSM largest building (area={site_polygon.area:.0f}m²)")
        except Exception as e:
            log.info(f"  Site: buffer fallback ({e})")
            site_polygon = pt.buffer(40)
        site_gdf = gpd.GeoDataFrame(geometry=[site_polygon], crs=3857)

    try:
        roads = ox.features_from_point(
            (lat, lon), dist=cfg["study_radius"], tags={"highway": True}
        ).to_crs(3857)
        roads = roads[
            roads.geometry.type.isin(["LineString", "MultiLineString"])
        ]
        if not len(roads):
            raise ValueError("no roads found in study radius")
        log.info(f"  OSM roads: {len(roads)}")
    except Exception as e:
        raise ValueError(f"Road fetch failed: {e}") from e

    try:
        bld = ox.features_from_point(
            (lat, lon), dist=cfg["study_radius"], tags={"building": True}
        ).to_crs(3857)
        bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])]
        log.info(f"  OSM buildings: {len(bld)}")
    except Exception as e:
        log.warning(f"  Buildings fetch failed: {e}")
        bld = gpd.GeoDataFrame(geometry=[], crs=3857)

    atc_data = ATCWFSLoader(cfg).load()
    lnrs_gdf = LNRSWFSLoader(cfg).load()

    roads = TrafficAssigner(atc_data, cfg).assign(roads)
    roads = LNRSAssigner(lnrs_gdf, cfg).assign(roads)
    roads = CanyonAssigner(bld, cfg).assign(roads)
    roads = EmissionEngine(cfg).compute(roads)

    Lv = roads["L_link"].values
    X, Y, noise = PropagationEngine(cfg).run(roads, site_polygon)

    meta = {
        "type":           data_type,
        "value":          value,
        "L_source_range": f"{Lv.min():.0f}-{Lv.max():.0f} dB(A)",
        "lnrs_roads":     int((roads["lnrs_corr"] < 0).sum()),
    }

    return NoiseVisualizer(cfg).render(
        X, Y, noise, site_polygon, site_gdf, bld, roads, meta
    )
