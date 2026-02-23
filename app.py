from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator
from io import BytesIO
import geopandas as gpd
import os
import time
import logging
import hashlib
import requests
from pyproj import Transformer
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------
# IMPORT MODULES (UPDATED TO ACCEPT lon, lat)
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
# CORS
# -----------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOGGING
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------------
# ALLOWED INPUT TYPES
# -----------------------------------------------------

ALLOWED_TYPES = {
    "LOT","STT","GLA","LPP","UN",
    "BUILDINGCSUID","LOTCSUID","PRN"
}

# -----------------------------------------------------
# REQUEST MODEL
# -----------------------------------------------------

class AnalysisRequest(BaseModel):
    value: str
    data_type: str

    @validator("data_type")
    def validate_type(cls, v):
        if v.upper() not in ALLOWED_TYPES:
            raise ValueError("Invalid data_type")
        return v.upper()

# -----------------------------------------------------
# CACHE STORE
# -----------------------------------------------------

CACHE_STORE = {}

def cache_key(value: str, data_type: str, analysis_type: str):
    raw = f"{value}_{data_type}_{analysis_type}"
    return hashlib.md5(raw.encode()).hexdigest()

# -----------------------------------------------------
# LOAD STATIC DATA
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

ZONE_PATH = os.path.join(DATA_DIR, "ZONE_REDUCED.gpkg")
BUILDINGS_PATH = os.path.join(DATA_DIR, "BUILDINGS_FINAL.gpkg")

print("Loading zoning dataset...")
ZONE_DATA = gpd.read_file(ZONE_PATH).to_crs(3857)

print("Loading building height dataset...")
BUILDING_DATA = gpd.read_file(BUILDINGS_PATH).to_crs(3857)

if "HEIGHT_M" not in BUILDING_DATA.columns:
    raise ValueError("HEIGHT_M column missing in building dataset")

BUILDING_DATA = BUILDING_DATA[BUILDING_DATA["HEIGHT_M"] > 5]

print("Startup complete.")

# -----------------------------------------------------
# CENTRAL LOCATION RESOLVER
# -----------------------------------------------------

def resolve_location(search_value: str, data_type: str):

    base_url = "https://mapapi.geodata.gov.hk/gs/api/v1.0.0"
    url = f"{base_url}/lus/{data_type.lower()}/SearchNumber?text={search_value.replace(' ','%20')}"

    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError(f"Failed to resolve {data_type}")

    data = r.json()

    if "candidates" not in data or len(data["candidates"]) == 0:
        raise ValueError("Location not found")

    best = max(data["candidates"], key=lambda x: x.get("score", 0))

    x2326 = best["location"]["x"]
    y2326 = best["location"]["y"]

    lon, lat = Transformer.from_crs(
        2326, 4326, always_xy=True
    ).transform(x2326, y2326)

    return lon, lat

# -----------------------------------------------------
# IMAGE RESPONSE
# -----------------------------------------------------

def image_response(buffer: BytesIO):
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

# -----------------------------------------------------
# GENERIC WRAPPER
# -----------------------------------------------------

def run_analysis(value: str, data_type: str, analysis_type: str, func, *args):

    logging.info(f"{analysis_type.upper()} request for {value} ({data_type})")
    start = time.time()

    key = cache_key(value, data_type, analysis_type)

    if key in CACHE_STORE:
        logging.info(f"{analysis_type.upper()} cache hit")
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
def walking(req: AnalysisRequest):
    try:
        lon, lat = resolve_location(req.value, req.data_type)
        img = run_analysis(req.value, req.data_type, "walking", generate_walking, lon, lat)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/driving")
def driving(req: AnalysisRequest):
    try:
        lon, lat = resolve_location(req.value, req.data_type)
        img = run_analysis(req.value, req.data_type, "driving", generate_driving, lon, lat, ZONE_DATA)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transport")
def transport(req: AnalysisRequest):
    try:
        lon, lat = resolve_location(req.value, req.data_type)
        img = run_analysis(req.value, req.data_type, "transport", generate_transport, lon, lat)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context")
def context(req: AnalysisRequest):
    try:
        lon, lat = resolve_location(req.value, req.data_type)
        img = run_analysis(req.value, req.data_type, "context", generate_context, lon, lat, ZONE_DATA)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/view")
def view(req: AnalysisRequest):
    try:
        lon, lat = resolve_location(req.value, req.data_type)
        img = run_analysis(req.value, req.data_type, "view", generate_view, lon, lat, BUILDING_DATA)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/noise")
def noise(req: AnalysisRequest):
    try:
        lon, lat = resolve_location(req.value, req.data_type)
        img = run_analysis(req.value, req.data_type, "noise", generate_noise, lon, lat)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# PDF REPORT
# -----------------------------------------------------

def generate_pdf_report(value: str, data_type: str):

    lon, lat = resolve_location(value, data_type)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    elements = []

    view_img = generate_view(lon, lat, BUILDING_DATA)
    noise_img = generate_noise(lon, lat)

    view_img.seek(0)
    noise_img.seek(0)

    elements.append(RLImage(view_img, width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(RLImage(noise_img, width=6*inch, height=4*inch))

    doc.build(elements)

    buffer.seek(0)
    return buffer


@app.post("/report")
def report(req: AnalysisRequest):
    try:
        pdf_buffer = generate_pdf_report(req.value, req.data_type)

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=site_report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------

@app.get("/")
def root():
    return {"status": "Automated Site Analysis API Running"}
