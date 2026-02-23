import requests
from pyproj import Transformer

BASE_URL = "https://mapapi.geodata.gov.hk/gs/api/v1.0.0"

ALLOWED_TYPES = [
    "LOT",
    "STT",
    "GLA",
    "LPP",
    "UN",
    "BUILDINGCSUID",
    "LOTCSUID",
    "PRN"
]

def resolve_location(data_type: str, value: str):

    data_type = data_type.upper()

    if data_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported data type: {data_type}")

    url = (
        f"{BASE_URL}/lus/{data_type.lower()}/SearchNumber"
        f"?text={value.replace(' ', '%20')}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError("Failed to resolve number.")

    data = response.json()

    if "candidates" not in data or len(data["candidates"]) == 0:
        raise ValueError("No matching result found.")

    best = max(data["candidates"], key=lambda x: x.get("score", 0))

    x2326 = best["location"]["x"]
    y2326 = best["location"]["y"]

    lon, lat = Transformer.from_crs(
        2326, 4326, always_xy=True
    ).transform(x2326, y2326)

    return lon, lat
