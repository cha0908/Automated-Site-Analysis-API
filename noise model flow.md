We are building a traffic-based environmental noise model.

The model needs:

flow
heavy_pct
speed
surface_correction
geometry

Below is the full structured breakdown.

1️⃣ TRAFFIC FLOW DATA (Primary Emission Input)
Dataset:

Traffic Flow Census – ATC_TRAFFIC_DATA
Provided by Transport Department

Format:

ZIP → contains Excel (.xlsx) files
Each Excel file = 1 ATC station

Structure:
2024.zip
 └── 2024/
      └── Current/
           ├── S6224.xlsx
           ├── SXXXX.xlsx
           ...
File Format:

Excel (.xlsx)

Tabular numeric data

Fields Extracted:

AADT (All-Day)

AM Peak Hour Flow

PM Peak Hour Flow

Prop. of Commercial Vehicles (%)

Used For:
flow = max(AM_peak, PM_peak)
heavy_pct = commercial_pct / 100
2️⃣ ATC STATION GEOMETRY (Spatial Linking)
Dataset:

Traffic Flow Census (CSDI Portal)
Provided by Transport Department

API Type:

OGC WFS (GeoJSON)

API Endpoint:
https://portal.csdi.gov.hk/server/services/common/td_rcd_1638950691670_3837/MapServer/WFSServer?service=wfs&request=GetFeature&typenames=ATC_STATION_PT&outputFormat=geojson&srsName=EPSG:4326&count=10000
Format:

GeoJSON

Geometry Type:

Point

Fields:

ATC_STATION_NO

geometry (lon/lat)

Used For:

Spatial snapping of OSM roads to nearest station

Assigning flow + heavy_pct to road links

3️⃣ SPEED DATA (Strategic / Major Roads)
Dataset:

Traffic Data of Strategic / Major Roads
Provided by Transport Department

Components:
A) Road Network Segments

Format: CSV
Contains:

segment_id

geometry reference

road name

B) Traffic Speeds (Processed Data)

Format: XML
Contains:

segment_id

traffic_speed (km/h)

timestamp

Used For:

Override speed for strategic roads:

if road intersects strategic_segment:
    speed = traffic_speed
else:
    speed = OSM maxspeed
4️⃣ LOW NOISE ROAD SURFACE (LNRS)
Dataset:

Low Noise Road Surface
Provided by Environmental Protection Department

API Type:

OGC WFS (GeoJSON)

API Endpoint:
https://portal.csdi.gov.hk/server/services/common/epd_rcd_1696984809000_78322/MapServer/WFSServer?service=wfs&request=GetFeature&typenames=noise_lnrs&outputFormat=geojson&count=10000
Format:

GeoJSON

Geometry Type:

MultiLineString

Fields:

road

materials

geometry

Used For:

Apply surface correction:

lnrs_corr = -3 dB  (if road intersects LNRS)
5️⃣ BASE ROAD NETWORK
Source:

OSM (via OSMnx)

Format:

GeoDataFrame (LineString)

Fields:

geometry

highway type

maxspeed

name

Used As:

Base geometry for noise propagation.

📊 FINAL DATA FLOW PIPELINE
OSM Roads (LineString)
        ↓
Snap to ATC Station (Point WFS)
        ↓
Attach flow + heavy_pct (Excel ZIP)
        ↓
Override speed (Strategic Roads XML)
        ↓
Apply LNRS correction (WFS)
        ↓
Compute L_source per link
        ↓
Spatial noise propagation
🧠 FILE FORMAT SUMMARY
Dataset	Format	API Type
ATC Traffic Data	ZIP → Excel (.xlsx)	Download
ATC Station Geometry	GeoJSON	WFS
Strategic Road Segments	CSV	Static
Strategic Road Speeds	XML	Static/API
LNRS	GeoJSON	WFS
OSM Roads	GeoDataFrame	OSMnx
🎯 FINAL EMISSION FORMULA

For each road segment:

L_source = traffic_emission(flow, heavy_pct, speed) + lnrs_corr

Where:

flow       → ATC ZIP (Peak Hour)
heavy_pct  → ATC ZIP (% commercial)
speed      → Strategic XML OR OSM
lnrs_corr  → LNRS WFS
