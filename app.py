import matplotlib
matplotlib.use("Agg")

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

ZONE_PATH = os.path.join(DATA_DIR, "ZONE_REDUCED.gpkg")
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
# REQUEST MODEL
# -----------------------------------------------------

class LocationRequest(BaseModel):
    data_type: str
    value: str

def image_response(buffer: BytesIO):
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

# -----------------------------------------------------
# GENERIC WRAPPER
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
# ENDPOINTS
# -----------------------------------------------------

@app.post("/walking")
def walking(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type,
            req.value,
            "walking",
            generate_walking,
            req.data_type,
            req.value
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/driving")
def driving(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type,
            req.value,
            "driving",
            generate_driving,
            req.data_type,
            req.value,
            ZONE_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transport")
def transport(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type,
            req.value,
            "transport",
            generate_transport,
            req.data_type,
            req.value
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context")
def context(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type,
            req.value,
            "context",
            generate_context,
            req.data_type,
            req.value,
            ZONE_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/view")
def view(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type,
            req.value,
            "view",
            generate_view,
            req.data_type,
            req.value,
            BUILDING_DATA
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/noise")
def noise(req: LocationRequest):
    try:
        img = run_analysis(
            req.data_type,
            req.value,
            "noise",
            generate_noise,
            req.data_type,
            req.value
        )
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# PDF REPORT
# -----------------------------------------------------

def generate_pdf_report(data_type: str, value: str):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    elements = []

    logging.info("Generating report images...")

    # --------------------------------------------------
    # GENERATE ALL ANALYSIS IMAGES
    # --------------------------------------------------

    walking_img = generate_walking(data_type, value)
    driving_img = generate_driving(data_type, value, ZONE_DATA)
    transport_img = generate_transport(data_type, value)
    context_img = generate_context(data_type, value, ZONE_DATA)
    view_img = generate_view(data_type, value, BUILDING_DATA)
    noise_img = generate_noise(data_type, value)

    images = [
        walking_img,
        driving_img,
        transport_img,
        context_img,
        view_img,
        noise_img
    ]

    # --------------------------------------------------
    # ADD TO PDF
    # --------------------------------------------------

    for img_buffer in images:
        img_buffer.seek(0)
        elements.append(RLImage(img_buffer, width=6*inch, height=4.5*inch))
        elements.append(Spacer(1, 0.6*inch))

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
