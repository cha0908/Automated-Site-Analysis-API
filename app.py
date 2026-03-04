import matplotlib
matplotlib.use("Agg")
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional                          # ← NEW
from io import BytesIO
import geopandas as gpd
import os
import time
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

# -----------------------------------------------------
# IMPORT MODULES
# -----------------------------------------------------

from modules.walking import generate_walking
from modules.driving import generate_driving
from modules.transport import generate_transport
from modules.context import generate_context
from modules.view import generate_view
from modules.noise import generate_noise

# -----------------------------------------------------
# FASTAPI INIT
# -----------------------------------------------------

app = FastAPI(
    title="Automated Site Analysis API",
    version="3.0"
)

# -----------------------------------------------------
# CORS CONFIGURATION
# -----------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOGGING CONFIG
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------------
# CACHE STORE
# -----------------------------------------------------

CACHE_STORE = {}

def cache_key(data_type: str, value: str, analysis_type: str):
    raw = f"{data_type}_{value}_{analysis_type}"
    return hashlib.md5(raw.encode()).hexdigest()

# -----------------------------------------------------
# LOAD STATIC DATA
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

ZONE_PATH      = os.path.join(DATA_DIR, "ZONE_REDUCED.gpkg")
BUILDINGS_PATH = os.path.join(DATA_DIR, "BUILDINGS_FINAL .gpkg")

print("Loading zoning dataset...")
ZONE_DATA = gpd.read_file(ZONE_PATH).to_crs(3857)

print("Loading building height dataset...")
BUILDING_DATA = gpd.read_file(BUILDINGS_PATH).to_crs(3857)

if "HEIGHT_M" not in BUILDING_DATA.columns:
    raise ValueError(
        f"HEIGHT_M column not found. Available columns: {BUILDING_DATA.columns}"
    )

BUILDING_DATA = BUILDING_DATA[BUILDING_DATA["HEIGHT_M"] > 5]

print("Startup complete.")

# -----------------------------------------------------
# REQUEST MODELS
# -----------------------------------------------------

class LocationRequest(BaseModel):
    data_type: str
    value:     str
class WalkingRequest(BaseModel):
    data_type: str
    value: str
    max_walk_minutes: Optional[int] = None l    lon:       Optional[float] = None   # pre-resolved coords for ADDRESS type
    lat:       Optional[float] = None   # None for LOT type — resolver handles it

def image_response(buffer: BytesIO):
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

# -----------------------------------------------------
# GENERIC ANALYSIS WRAPPER
# -----------------------------------------------------

def run_analysis(data_type: str, value: str, analysis_type: str, func, *args):

    logging.info(f"Incoming {analysis_type.upper()} request for {data_type} {value}")
    start = time.time()

    key = cache_key(data_type, value, analysis_type)

    if key in CACHE_STORE:
        logging.info(f"{analysis_type.upper()} cache hit for {data_type} {value}")
        return CACHE_STORE[key]

    result = func(*args)

    CACHE_STORE[key] = result

    duration = round(time.time() - start, 2)
    logging.info(f"{analysis_type.upper()} completed in {duration}s")

    return result

# -----------------------------------------------------
# SEARCH HELPERS
# -----------------------------------------------------

_transformer_2326_to_4326 = Transformer.from_crs(2326, 4326, always_xy=True)

LOT_PREFIXES = (
    "IL", "NKIL", "KIL", "STTL", "STL", "TML", "TPTL",
    "DD", "RBL", "KCTL", "ML", "GLA", "LPP"
)

def _looks_like_lot_id(q: str) -> bool:
    q_upper = q.upper().strip()
    return any(q_upper.startswith(p) for p in LOT_PREFIXES)

# -----------------------------------------------------
# SEARCH ENDPOINT
# -----------------------------------------------------

@app.get("/search")
def search(q: str, limit: int = 100):
    q = q.strip()
    if not q:
        return {"count": 0, "results": []}

    results = []
    seen    = set()
    is_lot  = _looks_like_lot_id(q)

    # ── 1. LOT ID SEARCH ──────────────────────────────────────
    try:
        resp = requests.get(
            "https://mapapi.geodata.gov.hk/gs/api/v1.0.0/lus/lot/SearchNumber"
            f"?text={requests.utils.quote(q)}",
            timeout=10
        )
        for c in resp.json().get("candidates", [])[:limit]:
            attrs   = c.get("attributes", {})
            lot_id  = attrs.get("Descr", c.get("address", q)).strip()
            ref_id  = attrs.get("Ref_ID", "")
            address = c.get("address", "").strip()
            key     = f"LOT_{ref_id or lot_id}"
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "lot_id":    lot_id,
                "name":      lot_id,
                "address":   address,
                "district":  attrs.get("City", ""),
                "ref_id":    ref_id,
                "data_type": "LOT",
                "source":    "lot_search",
                "lon":       None,
                "lat":       None,
            })
    except Exception as e:
        logging.warning(f"/search lot lookup failed: {e}")

    # ── 2. ADDRESS / BUILDING NAME SEARCH ─────────────────────
    if not is_lot:
        try:
            resp = requests.get(
                "https://geodata.gov.hk/gs/api/v1.0.0/locationSearch"
                f"?q={requests.utils.quote(q)}",
                timeout=10
            )
            for s in resp.json()[:limit]:
                name_en    = str(s.get("nameEN",    "")).strip()
                address_en = str(s.get("addressEN", "")).strip()
                district   = str(s.get("districtEN","")).strip()
                key        = f"ADDR_{address_en}_{name_en}"
                if key in seen:
                    continue
                seen.add(key)
                try:
                    lon, lat = _transformer_2326_to_4326.transform(
                        s.get("x"), s.get("y")
                    )
                except Exception:
                    lon, lat = None, None
                results.append({
                    "lot_id":    address_en,
                    "name":      name_en,
                    "address":   address_en,
                    "district":  district,
                    "ref_id":    "",
                    "data_type": "ADDRESS",
                    "source":    "address_search",
                    "lon":       round(lon, 6) if lon else None,
                    "lat":       round(lat, 6) if lat else None,
                })
        except Exception as e:
            logging.warning(f"/search address lookup failed: {e}")

    logging.info(f"Search '{q}' → {len(results)} results")
    return {"count": len(results), "results": results}

# -----------------------------------------------------
# ANALYSIS ENDPOINTS
# -----------------------------------------------------

@app.post("/walking")
def walking(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type, req.value, "walking",
            generate_walking, req.data_type, req.value
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/driving")
def driving(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type, req.value, "driving",
            generate_driving, req.data_type, req.value, ZONE_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transport")
def transport(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type, req.value, "transport",
            generate_transport, req.data_type, req.value
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context")
def context(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type, req.value, "context",
            generate_context, req.data_type, req.value, ZONE_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/view")
def view(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type, req.value, "view",
            generate_view, req.data_type, req.value, BUILDING_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/noise")
def noise(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type, req.value, "noise",
            generate_noise, req.data_type, req.value
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# PDF REPORT
# -----------------------------------------------------

def generate_pdf_report(data_type: str, value: str):

    buffer = BytesIO()
    landscape_a4 = (A4[1], A4[0])
    page_w_pt, page_h_pt = A4[1], A4[0]
    margin_pt  = 0.3 * inch
    frame_w_pt = page_w_pt - 2 * margin_pt
    frame_h_pt = page_h_pt - 2 * margin_pt

    elements = []
    styles   = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="SlideTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=4,
        alignment=1,
    )
    footer_style = ParagraphStyle(
        name="Footer",
        parent=styles["Normal"],
        fontSize=9,
        textColor="gray",
        alignment=1,
        spaceBefore=1,
    )

    _title_line_pt  = title_style.fontSize * 1.2
    _title_block_pt = 2 * _title_line_pt + title_style.spaceAfter
    _spacer_pt      = 0.25 * inch
    reserved_h_pt   = _title_block_pt + _spacer_pt

    max_image_w_pt = frame_w_pt
    max_image_h_pt = frame_h_pt - reserved_h_pt

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape_a4,
        leftMargin=margin_pt,
        rightMargin=margin_pt,
        topMargin=margin_pt,
        bottomMargin=margin_pt,
    )

    logging.info("Generating report images...")

    walking_5     = generate_walking(data_type, value, 5)
    walking_15    = generate_walking(data_type, value, 15)
    driving_img   = generate_driving(data_type, value, ZONE_DATA)
    transport_img = generate_transport(data_type, value)
    context_img   = generate_context(data_type, value, ZONE_DATA)
    view_img      = generate_view(data_type, value, BUILDING_DATA)
    noise_img     = generate_noise(data_type, value)

    images = [
        walking_5, walking_15, driving_img, transport_img,
        context_img, view_img, noise_img,
    ]

    site_label = f"{data_type} {value}"
    elements.append(Spacer(1, 3 * inch))
    elements.append(Paragraph("Site Analysis Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(site_label, styles["Heading2"]))
    elements.append(PageBreak())

    titles = [
        "Walking Accessibility (5 min)",
        "Walking Accessibility (15 min)",
        "Driving Distance",
        "Transport Network",
        "Context & Zoning",
        "View Analysis",
        "Noise Assessment",
    ]

    n_pages = len(titles)

    for i, (slide_title, img_buffer) in enumerate(zip(titles, images), start=1):
        elements.append(Paragraph(slide_title, title_style))
        elements.append(Spacer(1, 0.25 * inch))
        img_buffer.seek(0)
        data     = img_buffer.read()
        copy_buf = BytesIO(data)
        reader   = utils.ImageReader(copy_buf)
        iw, ih   = reader.getSize()
        scale    = min(max_image_w_pt / iw, max_image_h_pt / ih)
        w        = iw * scale
        h        = ih * scale
        elements.append(RLImage(BytesIO(data), width=w, height=h))
        if i < n_pages:
            elements.append(PageBreak())

    doc.build(elements)
    buffer.seek(0)
    return buffer


@app.post("/report")
def report(req: LocationRequest):
    try:
        logging.info(f"Generating FULL PDF report for {req.data_type} {req.value}")
        pdf_buffer = generate_pdf_report(req.data_type, req.value)
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=site_analysis_report.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------

@app.get("/")
def root():
    return {"status": "Automated Site Analysis API Running - Multi Identifier Enabled"}
