import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from io import BytesIO
import geopandas as gpd
import os
import time
import asyncio
import functools
import logging
import hashlib
import requests
from pyproj import Transformer
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer, Paragraph, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import utils
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response as _FResponse

from modules.walking import generate_walking
from modules.driving import generate_driving
from modules.transport import generate_transport
from modules.context import generate_context
from modules.view import generate_view
from modules.noise import generate_noise

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Automated Site Analysis API", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Cache ─────────────────────────────────────────────────────
CACHE_STORE = {}

def cache_key(data_type: str, value: str, analysis_type: str):
    return hashlib.md5(f"{data_type}_{value}_{analysis_type}".encode()).hexdigest()

# ── Static data ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("Loading zoning dataset...")
ZONE_DATA = gpd.read_file(os.path.join(DATA_DIR, "ZONE_REDUCED.gpkg")).to_crs(3857)

print("Loading building height dataset...")
BUILDING_DATA = gpd.read_file(os.path.join(DATA_DIR, "BUILDINGS_FINAL .gpkg")).to_crs(3857)
if "HEIGHT_M" not in BUILDING_DATA.columns:
    raise ValueError(f"HEIGHT_M column not found. Available: {BUILDING_DATA.columns}")
BUILDING_DATA = BUILDING_DATA[BUILDING_DATA["HEIGHT_M"] > 5]
print("Startup complete.")

# ── Request model ─────────────────────────────────────────────
class LocationRequest(BaseModel):
    data_type: str
    value:     str
    lon:       Optional[float]      = None
    lat:       Optional[float]      = None
    lot_ids:   Optional[List[str]]  = None
    extents:   Optional[List[dict]] = None
    max_walk_minutes:   Optional[int] = None
    max_drive_minutes:  Optional[int] = None
    context_radius_m:   Optional[int] = None

def image_response(buf: BytesIO):
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

def normalise_request(req: LocationRequest):
    dt      = req.data_type.upper()
    lot_ids = req.lot_ids or []
    extents = req.extents or []
    if dt == "ADDRESS" and (req.lon is None or req.lat is None):
        raise ValueError("ADDRESS type requires pre-resolved lon/lat from search results.")
    return dt, req.value, req.lon, req.lat, lot_ids, extents

# ── Generic wrapper ───────────────────────────────────────────
def run_analysis(data_type, value, analysis_type, func, *args):
    logging.info(f"Incoming {analysis_type.upper()} request for {data_type} {value}")
    start = time.time()
    key = cache_key(data_type, value, analysis_type)
    if key in CACHE_STORE:
        logging.info(f"{analysis_type.upper()} cache hit for {data_type} {value}")
        return CACHE_STORE[key]
    result = func(*args)
    CACHE_STORE[key] = result
    logging.info(f"{analysis_type.upper()} completed in {round(time.time()-start,2)}s")
    return result

# ── Search helpers ────────────────────────────────────────────
_t2326_4326 = Transformer.from_crs(2326, 4326, always_xy=True)
LOT_PREFIXES = ("IL","NKIL","KIL","STTL","STL","TML","TPTL","DD","RBL","KCTL","ML","GLA","LPP")

def _looks_like_lot_id(q):
    return any(q.upper().strip().startswith(p) for p in LOT_PREFIXES)

# ── /search ───────────────────────────────────────────────────
@app.get("/search")
def search(q: str, limit: int = 100):
    q = q.strip()
    if not q:
        return {"count": 0, "results": []}

    results, seen = [], set()
    is_lot = _looks_like_lot_id(q)

    try:
        resp = requests.get(
            "https://mapapi.geodata.gov.hk/gs/api/v1.0.0/lus/lot/SearchNumber"
            f"?text={requests.utils.quote(q)}", timeout=10)
        for c in resp.json().get("candidates", [])[:limit]:
            attrs  = c.get("attributes", {})
            lot_id = attrs.get("Descr", c.get("address", q)).strip()
            ref_id = attrs.get("Ref_ID", "")
            key    = f"LOT_{ref_id or lot_id}"
            if key in seen: continue
            seen.add(key)
            loc = c.get("location", {}); ext = c.get("extent", {})
            x, y = loc.get("x") or attrs.get("X"), loc.get("y") or attrs.get("Y")
            lon_w = lat_w = None
            if x and y:
                try:
                    lon_w, lat_w = _t2326_4326.transform(x, y)
                    lon_w, lat_w = round(lon_w, 6), round(lat_w, 6)
                except Exception: pass
            results.append({
                "lot_id": lot_id, "name": lot_id,
                "address": c.get("address","").strip(),
                "district": attrs.get("City",""),
                "ref_id": ref_id, "score": c.get("score",0),
                "data_type": "LOT", "source": "lot_search",
                "lon": lon_w, "lat": lat_w,
                "x2326": round(x,2) if x else None,
                "y2326": round(y,2) if y else None,
                "extent": {
                    "xmin": ext.get("xmin") or attrs.get("Xmin"),
                    "ymin": ext.get("ymin") or attrs.get("Ymin"),
                    "xmax": ext.get("xmax") or attrs.get("Xmax"),
                    "ymax": ext.get("ymax") or attrs.get("Ymax"),
                } if (ext or attrs.get("Xmin")) else None,
            })
    except Exception as e:
        logging.warning(f"/search lot lookup failed: {e}")

    if not is_lot:
        try:
            resp = requests.get(
                f"https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q={requests.utils.quote(q)}",
                timeout=10)
            for s in resp.json()[:limit]:
                ne = str(s.get("nameEN","")).strip()
                ae = str(s.get("addressEN","")).strip()
                key = f"ADDR_{ae}_{ne}"
                if key in seen: continue
                seen.add(key)
                try:
                    lon_r, lat_r = _t2326_4326.transform(s.get("x"), s.get("y"))
                except Exception:
                    lon_r = lat_r = None
                display = ne if ne else ae
                results.append({
                    "lot_id": display, "name": display, "address": ae,
                    "district": str(s.get("districtEN","")).strip(),
                    "ref_id": "", "data_type": "ADDRESS", "source": "address_search",
                    "lon": round(lon_r,6) if lon_r else None,
                    "lat": round(lat_r,6) if lat_r else None,
                    "extent": None,
                })
        except Exception as e:
            logging.warning(f"/search address lookup failed: {e}")

    logging.info(f"Search '{q}' → {len(results)} results")
    return {"count": len(results), "results": results}


# ── /lot-boundary ─────────────────────────────────────────────
@app.get("/lot-boundary")
def lot_boundary(
    lon: float,
    lat: float,
    data_type: str = "LOT",
    extents: Optional[str] = None,
):
    import json
    from modules.resolver import get_lot_boundary

    ext_list = []
    if extents:
        try:
            ext_list = json.loads(extents)
        except Exception:
            pass

    try:
        gdf = get_lot_boundary(lon, lat, data_type, ext_list if ext_list else None)
        if gdf is None or gdf.empty:
            raise HTTPException(status_code=404, detail="Lot boundary not found")
        return gdf.to_crs(4326).__geo_interface__
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Analysis endpoints ────────────────────────────────────────
CONTEXT_RADIUS_ALLOWED = {600, 800, 1000}

@app.post("/walking")
def walking(req: LocationRequest):
    try:
        dt, v, lon, lat, lot_ids, extents = normalise_request(req)
        max_walk = req.max_walk_minutes if req.max_walk_minutes is not None else 15
        max_walk = max(5, min(20, max_walk))
        img = run_analysis(dt, v, f"walking_{max_walk}",
            generate_walking, dt, v, max_walk, lon, lat, lot_ids, extents)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/driving")
def driving(req: LocationRequest):
    try:
        dt, v, lon, lat, lot_ids, extents = normalise_request(req)
        max_drive = req.max_drive_minutes if req.max_drive_minutes is not None else 15
        max_drive = max(5, min(20, max_drive))
        img = run_analysis(dt, v, f"driving_{max_drive}",
            generate_driving, dt, v, ZONE_DATA, lon, lat, lot_ids, extents, max_drive)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transport")
def transport(req: LocationRequest):
    try:
        dt, v, lon, lat, lot_ids, extents = normalise_request(req)
        img = run_analysis(dt, v, "transport",
            generate_transport, dt, v, lon, lat, lot_ids, extents)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context")
async def context(req: LocationRequest):
    try:
        dt, v, lon, lat, lot_ids, extents = normalise_request(req)
        if extents:
            logging.info(f"  extents count: {len(extents)}")
        radius_m = req.context_radius_m if req.context_radius_m is not None else 600
        if radius_m not in CONTEXT_RADIUS_ALLOWED:
            radius_m = 600
        loop = asyncio.get_event_loop()
        img  = await loop.run_in_executor(
            None,
            functools.partial(
                run_analysis, dt, v, f"context_{radius_m}",
                generate_context, dt, v, ZONE_DATA, radius_m,
                lon, lat, lot_ids, extents
            )
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/view")
def view(req: LocationRequest):
    try:
        dt, v, lon, lat, lot_ids, extents = normalise_request(req)
        img = run_analysis(dt, v, "view",
            generate_view, dt, v, BUILDING_DATA, lon, lat, lot_ids, extents)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/noise")
def noise(req: LocationRequest):
    try:
        dt, v, lon, lat, lot_ids, extents = normalise_request(req)
        if extents:
            logging.info(f"  extents count: {len(extents)}")
        img = run_analysis(dt, v, "noise",
            generate_noise, dt, v, lon, lat, lot_ids, extents)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── PDF report ────────────────────────────────────────────────

def generate_pdf_report(data_type: str, value: str,
                         lon: float = None, lat: float = None,
                         lot_ids: list = None, extents: list = None):
    """
    Generates all analysis images IN PARALLEL using a ThreadPoolExecutor,
    then assembles them into a PDF.

    Each module runs concurrently — total wall time ≈ slowest single module
    (~60-80s) instead of the sum of all modules (~400s+ sequential).

    Uses shutdown(wait=False) after cf.wait so zombie threads (e.g. a slow
    Overpass fetch) do not block PDF assembly after the timeout fires.
    """
    import concurrent.futures as _cf
    from concurrent.futures import ThreadPoolExecutor

    lot_ids = lot_ids or []
    extents = extents or []

    # ── Define all analysis tasks ─────────────────────────────
    tasks = [
        ("walking_5",    generate_walking,   [data_type, value, 5,  lon, lat, lot_ids, extents]),
        ("walking_15",   generate_walking,   [data_type, value, 15, lon, lat, lot_ids, extents]),
        ("driving_15",   generate_driving,   [data_type, value, ZONE_DATA, lon, lat, lot_ids, extents, 15]),
        ("transport",    generate_transport, [data_type, value, lon, lat, lot_ids, extents]),
        ("context_600",  generate_context,   [data_type, value, ZONE_DATA, 600, lon, lat, lot_ids, extents]),
        ("noise",        generate_noise,     [data_type, value, lon, lat, lot_ids, extents]),
    ]
    titles = [
        "Walking Accessibility (5 min)",
        "Walking Accessibility (15 min)",
        "Driving Distance",
        "Transport Network",
        "Context & Zoning",
        "Noise Assessment",
    ]

    # ── Submit all tasks in parallel ──────────────────────────
    # run_analysis handles cache hits — cached modules return instantly.
    logging.info(f"[report] Launching {len(tasks)} analyses in parallel...")
    t0 = time.time()

    _pool = ThreadPoolExecutor(max_workers=len(tasks))
    future_to_idx = {
        _pool.submit(run_analysis, data_type, value, atype, func, *args): i
        for i, (atype, func, args) in enumerate(tasks)
    }

    # Wait up to 180s for all tasks — enough for 2 sequential slow modules
    # if one cache-hits and another has a slow Overpass fetch.
    done, pending = _cf.wait(future_to_idx, timeout=180,
                              return_when=_cf.ALL_COMPLETED)

    for f in pending:
        idx = future_to_idx[f]
        logging.warning(f"[report] '{tasks[idx][0]}' timed out — using blank image")

    # CRITICAL: shutdown(wait=False) — do not block on zombie threads
    _pool.shutdown(wait=False)

    logging.info(f"[report] All analyses done in {time.time()-t0:.1f}s "
                 f"({len(done)} completed, {len(pending)} timed out)")

    # ── Collect results in order ──────────────────────────────
    imgs = [None] * len(tasks)
    for f, idx in future_to_idx.items():
        if f in done:
            try:
                imgs[idx] = f.result()
            except Exception as e:
                logging.warning(f"[report] '{tasks[idx][0]}' failed: {e}")

    # ── Build PDF ─────────────────────────────────────────────
    buffer = BytesIO()
    landscape_a4 = (A4[1], A4[0])
    margin_pt    = 0.3 * inch
    frame_w_pt   = A4[1] - 2 * margin_pt
    frame_h_pt   = A4[0] - 2 * margin_pt

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="SlideTitle", parent=styles["Heading1"],
        fontSize=18, spaceAfter=4, alignment=1)

    reserved_h_pt  = (title_style.fontSize * 1.2 * 2 + title_style.spaceAfter) + 0.25 * inch
    max_image_w_pt = frame_w_pt
    max_image_h_pt = frame_h_pt - reserved_h_pt

    doc = SimpleDocTemplate(buffer, pagesize=landscape_a4,
        leftMargin=margin_pt, rightMargin=margin_pt,
        topMargin=margin_pt,  bottomMargin=margin_pt)

    elements = []
    elements.append(Spacer(1, 3 * inch))
    elements.append(Paragraph("Site Analysis Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"{data_type} {value}", styles["Heading2"]))
    elements.append(PageBreak())

    for i, (title, buf_img) in enumerate(zip(titles, imgs), 1):
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 0.25 * inch))

        if buf_img is None:
            # Module timed out or failed — insert a placeholder paragraph
            elements.append(Paragraph(
                f"[{title} — analysis unavailable]", styles["Normal"]))
        else:
            buf_img.seek(0)
            data   = buf_img.read()
            reader = utils.ImageReader(BytesIO(data))
            iw, ih = reader.getSize()
            scale  = min(max_image_w_pt / iw, max_image_h_pt / ih)
            elements.append(RLImage(BytesIO(data), width=iw * scale, height=ih * scale))

        if i < len(titles):
            elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ── /report endpoint — async to keep event loop free ─────────
@app.post("/report")
async def report(req: LocationRequest):
    """
    Async endpoint — runs the blocking PDF generation in a thread executor.
    Keeps the FastAPI event loop free to answer Render health checks (HEAD /)
    during the ~60-90s it takes to generate all analysis images.
    """
    try:
        dt, v, lon, lat, lot_ids, extents = normalise_request(req)
        logging.info(f"Generating FULL PDF report for {dt} {v}")
        loop = asyncio.get_event_loop()
        pdf  = await loop.run_in_executor(
            None,
            functools.partial(
                generate_pdf_report, dt, v, lon, lat, lot_ids, extents
            )
        )
        return StreamingResponse(pdf, media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=site_analysis_report.pdf"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health / root ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Automated Site Analysis API v3.0"}

@app.head("/")
def root_head():
    return _FResponse(status_code=200)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.head("/health")
def health_head():
    return _FResponse(status_code=200)
