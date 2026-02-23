from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import geopandas as gpd
import os
import time
import logging
import hashlib
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
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
    version="2.0"
)

# -----------------------------------------------------
# CORS CONFIGURATION (ALLOW ALL ORIGINS)
# -----------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all domains
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],      # Allow all HTTP methods
    allow_headers=["*"],      # Allow all headers
)

# -----------------------------------------------------
# LOGGING CONFIG
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------------
# CACHE STORE (FREE TIER SAFE)
# -----------------------------------------------------

CACHE_STORE = {}

def cache_key(lot_id: str, analysis_type: str):
    raw = f"{lot_id}_{analysis_type}"
    return hashlib.md5(raw.encode()).hexdigest()

# -----------------------------------------------------
# LOAD STATIC DATA ONCE
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

ZONE_PATH = os.path.join(DATA_DIR, "ZONE_REDUCED.gpkg")
BUILDINGS_PATH = os.path.join(DATA_DIR, "BUILDINGS_FINAL .gpkg")  # removed extra space

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
# REQUEST MODEL
# -----------------------------------------------------

class LotRequest(BaseModel):
    lot_id: str

def image_response(buffer: BytesIO):
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

# -----------------------------------------------------
# GENERIC WRAPPER (Logging + Timing + Cache)
# -----------------------------------------------------

def run_analysis(lot_id: str, analysis_type: str, func, *args):

    logging.info(f"Incoming {analysis_type.upper()} request for lot {lot_id}")
    start = time.time()

    key = cache_key(lot_id, analysis_type)

    if key in CACHE_STORE:
        logging.info(f"{analysis_type.upper()} cache hit for {lot_id}")
        return CACHE_STORE[key]

    result = func(*args)

    CACHE_STORE[key] = result

    duration = round(time.time() - start, 2)
    logging.info(f"{analysis_type.upper()} completed in {duration}s")

    return result

# -----------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------

@app.post("/walking")
def walking(req: LotRequest):
    try:
        img = run_analysis(
            req.lot_id,
            "walking",
            generate_walking,
            req.lot_id
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/driving")
def driving(req: LotRequest):
    try:
        img = run_analysis(
            req.lot_id,
            "driving",
            generate_driving,
            req.lot_id,
            ZONE_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transport")
def transport(req: LotRequest):
    try:
        img = run_analysis(
            req.lot_id,
            "transport",
            generate_transport,
            req.lot_id
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context")
def context(req: LotRequest):
    try:
        img = run_analysis(
            req.lot_id,
            "context",
            generate_context,
            req.lot_id,
            ZONE_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/view")
def view(req: LotRequest):
    try:
        img = run_analysis(
            req.lot_id,
            "view",
            generate_view,
            req.lot_id,
            BUILDING_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/noise")
def noise(req: LotRequest):
    try:
        img = run_analysis(
            req.lot_id,
            "noise",
            generate_noise,
            req.lot_id
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# PDF REPORT ENDPOINT
# -----------------------------------------------------

def generate_pdf_report(lot_id: str):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    elements = []

    view_img = generate_view(lot_id, BUILDING_DATA)
    noise_img = generate_noise(lot_id)

    view_img.seek(0)
    noise_img.seek(0)

    elements.append(RLImage(view_img, width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(RLImage(noise_img, width=6*inch, height=4*inch))

    doc.build(elements)

    buffer.seek(0)
    return buffer

@app.post("/report")
def report(req: LotRequest):
    try:
        logging.info(f"Generating PDF report for {req.lot_id}")
        pdf_buffer = generate_pdf_report(req.lot_id)

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=site_report.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------

@app.get("/")
def root():
    return {"status": "Automated Site Analysis API Running"}
