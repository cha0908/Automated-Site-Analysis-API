from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import geopandas as gpd
import os

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
    version="1.0"
)

# -----------------------------------------------------
# LOAD STATIC DATA ONCE
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

ZONE_PATH = os.path.join(DATA_DIR, "ZONE_REDUCED.gpkg")
BUILDINGS_PATH = os.path.join(DATA_DIR, "BUILDINGS_FINAL .gpkg")

print("Loading building height dataset...")

BUILDING_DATA = gpd.read_file(BUILDINGS_PATH).to_crs(3857)

# Validate required column
if "HEIGHT_M" not in BUILDING_DATA.columns:
    raise ValueError(
        f"HEIGHT_M column not found. Available columns: {BUILDING_DATA.columns}"
    )

# Filter valid buildings
BUILDING_DATA = BUILDING_DATA[BUILDING_DATA["HEIGHT_M"] > 5]

print("Building dataset loaded successfully.")


# -----------------------------------------------------
# REQUEST MODEL
# -----------------------------------------------------

class LotRequest(BaseModel):
    lot_id: str


def image_response(buffer: BytesIO):
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


# -----------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------

@app.post("/walking")
def walking(req: LotRequest):
    try:
        img = generate_walking(req.lot_id)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/driving")
def driving(req: LotRequest):
    try:
        img = generate_driving(req.lot_id, ZONE_DATA)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transport")
def transport(req: LotRequest):
    try:
        img = generate_transport(req.lot_id)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context")
def context(req: LotRequest):
    try:
        img = generate_context(req.lot_id, ZONE_DATA)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/view")
def view(req: LotRequest):
    try:
        img = generate_view(req.lot_id, BUILDING_DATA)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/noise")
def noise(req: LotRequest):
    try:
        img = generate_noise(req.lot_id)
        return image_response(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------

@app.get("/")
def root():
    return {"status": "Automated Site Analysis API Running"}
