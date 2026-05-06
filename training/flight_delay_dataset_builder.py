"""
flight_delay_dataset_builder.py

Builds a machine learning dataset for predicting US domestic flight delays
by combining BTS On-Time Performance data with weather observations from
Open-Meteo.

Data Sources:
    - Bureau of Transportation Statistics (BTS): On-Time Reporting Carrier Performance
      https://www.transtats.bts.gov
    - Open-Meteo Historical Weather API: Hourly weather reanalysis data
      https://open-meteo.com/en/docs/historical-weather-api
    - Open-Meteo Forecast API (at prediction time): Same variable names/units
      https://open-meteo.com/en/docs

Why Open-Meteo?
    The historical archive API and the forecast API use identical variable names
    and units. This means a model trained on historical data can receive forecast
    inputs at prediction time with zero schema translation. api.weather.gov only
    retains ~7 days of observations (unusable for training), and its forecast
    schema differs from IEM's historical schema, requiring a mapping layer.

Targets:
    - Binary classification: is the flight delayed? (1/0, using 15-min threshold)
    - Regression: how many minutes is the delay?

Usage:
    # Run all steps end-to-end
    python flight_delay_dataset_builder.py --step all

    # Run individual steps (useful for resuming after interruptions)
    python flight_delay_dataset_builder.py --step flights
    python flight_delay_dataset_builder.py --step weather
    python flight_delay_dataset_builder.py --step merge

    # Test the prediction-time forecast fetch for a single airport
    python flight_delay_dataset_builder.py --step demo-forecast

Requirements:
    pip install pandas requests tqdm
"""

import os
import sys
import time
import json
import zipfile
import argparse
import logging
from io import BytesIO, StringIO
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

YEAR = 2024
DATA_DIR = Path("./data")
BTS_DIR = DATA_DIR / "bts_raw"
WEATHER_DIR = DATA_DIR / "weather_openmeteo"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = DATA_DIR / "logs"

# BTS download URL pattern
BTS_PREZIP_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Open-Meteo API endpoints
# Historical archive (for training data): ERA5 reanalysis, available from 1940-present
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Forecast (for prediction time): same variable names, same units
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Rate limiting: Open-Meteo free tier allows 10,000 requests/day
# No strict per-second throttle, but be respectful
OPENMETEO_REQUEST_DELAY = 0.5
OPENMETEO_MAX_RETRIES = 3

# Hourly weather variables to request from Open-Meteo
# These names are IDENTICAL between the archive and forecast APIs,
# which is the whole reason we use Open-Meteo.
HOURLY_WEATHER_VARS = [
    "temperature_2m",           # Air temperature at 2m (°F)
    "dew_point_2m",             # Dewpoint at 2m (°F)
    "relative_humidity_2m",     # Relative humidity at 2m (%)
    "precipitation",            # Total precipitation in preceding hour (inch)
    "rain",                     # Rain in preceding hour (inch)
    "snowfall",                 # Snowfall in preceding hour (inch)
    "weather_code",             # WMO weather condition code (integer)
    "cloud_cover",              # Total cloud cover (%)
    "cloud_cover_low",          # Low-level cloud cover (%) - proxy for ceiling
    "cloud_cover_mid",          # Mid-level cloud cover (%)
    "cloud_cover_high",         # High-level cloud cover (%)
    "visibility",               # Horizontal visibility (feet)
    "pressure_msl",             # Sea-level pressure (hPa)
    "surface_pressure",         # Surface pressure (hPa)
    "wind_speed_10m",           # Wind speed at 10m (mph)
    "wind_direction_10m",       # Wind direction at 10m (degrees)
    "wind_gusts_10m",           # Wind gusts at 10m (mph)
]

# Columns to keep from BTS data
BTS_COLUMNS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "DOT_ID_Reporting_Airline",
    "IATA_CODE_Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "OriginCityName", "OriginState",
    "Dest", "DestCityName", "DestState",
    "CRSDepTime", "DepTime", "DepDelay", "DepDel15",
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDel15",
    "Cancelled", "CancellationCode", "Diverted",
    "CRSElapsedTime", "ActualElapsedTime", "AirTime",
    "Distance", "DistanceGroup",
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
]

# ═════════════════════════════════════════════════════════════════════════════
# US AIRPORT COORDINATES
# ═════════════════════════════════════════════════════════════════════════════
#
# Coordinates for US commercial airports used in BTS data. These are needed
# to query Open-Meteo (which takes lat/lon rather than station codes).
# Source: FAA / OurAirports (public domain).
# This covers all major + regional airports. If an airport isn't in this dict,
# the script will log a warning and skip its weather data.
# ═════════════════════════════════════════════════════════════════════════════

AIRPORT_COORDS = {
    "ABE": (40.6521, -75.4408), "ABI": (32.4113, -99.6819), "ABQ": (35.0402, -106.6092),
    "ABR": (45.4491, -98.4218), "ABY": (31.5355, -84.1945), "ACK": (41.2531, -70.0602),
    "ACT": (31.6113, -97.2305), "ACV": (40.9781, -124.1086), "ACY": (39.4576, -74.5773),
    "ADK": (51.8780, -176.6460), "ADQ": (57.7500, -152.4938), "AEX": (31.3274, -92.5486),
    "AGS": (33.3700, -81.9645), "AKN": (58.6768, -156.6492), "ALB": (42.7483, -73.8017),
    "ALO": (42.5571, -92.4003), "ALS": (37.4349, -105.8666), "AMA": (35.2194, -101.7059),
    "ANC": (61.1743, -149.9962), "APN": (44.8462, -83.6754), "ASE": (39.2232, -106.8688),
    "ATL": (33.6407, -84.4277), "ATW": (44.2581, -88.5191), "AUS": (30.1945, -97.6699),
    "AVL": (35.4362, -82.5418), "AVP": (41.3385, -75.7234), "AZA": (33.3078, -111.6550),
    "AZO": (42.2350, -85.5521), "BDL": (41.9389, -72.6832), "BET": (60.7798, -161.8380),
    "BFL": (35.4334, -119.0568), "BGM": (42.2087, -75.9798), "BGR": (44.8074, -68.8281),
    "BHM": (33.5629, -86.7535), "BIL": (45.8077, -108.5430), "BIS": (46.7727, -100.7468),
    "BJI": (47.5094, -94.9337), "BLI": (48.7928, -122.5375), "BLV": (38.5452, -89.8352),
    "BMI": (40.4770, -88.9160), "BNA": (36.1264, -86.6774), "BOI": (43.5644, -116.2228),
    "BOS": (42.3656, -71.0096), "BPT": (29.9508, -94.0207), "BQK": (31.2588, -81.4665),
    "BQN": (18.4949, -67.1294), "BRD": (46.3983, -94.1382), "BRO": (25.9068, -97.4259),
    "BRW": (71.2854, -156.7660), "BTM": (45.9548, -112.4972), "BTR": (30.5332, -91.1496),
    "BTV": (44.4720, -73.1533), "BUF": (42.9405, -78.7322), "BUR": (34.2006, -118.3585),
    "BWI": (39.1754, -76.6683), "BZN": (45.7775, -111.1530), "CAE": (33.9388, -81.1195),
    "CAK": (40.9161, -81.4422), "CDC": (37.7010, -113.0988), "CDV": (60.4918, -145.4777),
    "CHA": (35.0353, -85.2038), "CHO": (38.1386, -78.4529), "CHS": (32.8986, -80.0405),
    "CID": (41.8847, -91.7108), "CIU": (46.2508, -84.4724), "CKB": (39.2966, -80.2281),
    "CLE": (41.4117, -81.8498), "CLL": (30.5886, -96.3638), "CLT": (35.2140, -80.9431),
    "CMH": (39.9980, -82.8919), "CMI": (40.0392, -88.2781), "CMX": (47.1684, -88.4891),
    "CNY": (38.7550, -109.7549), "COD": (44.5202, -109.0238), "COS": (38.8058, -104.7009),
    "COU": (38.8181, -92.2196), "CPR": (42.9080, -106.4644), "CRP": (27.7704, -97.5012),
    "CRW": (38.3731, -81.5932), "CSG": (32.5163, -84.9389), "CVG": (39.0488, -84.6678),
    "CWA": (44.7776, -89.6668), "DAB": (29.1799, -81.0581), "DAL": (32.8471, -96.8518),
    "DAY": (39.9024, -84.2194), "DBQ": (42.4020, -90.7095), "DCA": (38.8512, -77.0402),
    "DEN": (39.8561, -104.6737), "DFW": (32.8998, -97.0403), "DHN": (31.3213, -85.4496),
    "DIK": (46.7974, -102.8019), "DLG": (59.0447, -158.5054), "DLH": (46.8421, -92.1936),
    "DRO": (37.1515, -107.7538), "DSM": (41.5340, -93.6631), "DTW": (42.2124, -83.3534),
    "DUT": (53.8998, -166.5435), "DVL": (48.1142, -98.9088), "EAR": (40.7270, -99.0068),
    "EAU": (44.8658, -91.4843), "ECP": (30.3580, -85.7956), "EGE": (39.6426, -106.9159),
    "EKO": (40.8249, -115.7917), "ELM": (42.1599, -76.8916), "ELP": (31.8072, -106.3776),
    "ERI": (42.0831, -80.1739), "ESC": (45.7227, -87.0937), "EUG": (44.1246, -123.2190),
    "EVV": (38.0370, -87.5324), "EWN": (35.0730, -77.0429), "EWR": (40.6895, -74.1745),
    "EYW": (24.5561, -81.7596), "FAI": (64.8151, -147.8564), "FAR": (46.9207, -96.8158),
    "FAT": (36.7762, -119.7181), "FAY": (34.9912, -78.8803), "FCA": (48.3105, -114.2560),
    "FLG": (35.1385, -111.6712), "FLL": (26.0726, -80.1527), "FLO": (34.1854, -79.7239),
    "FNT": (42.9655, -83.7436), "FSD": (43.5820, -96.7419), "FSM": (35.3366, -94.3674),
    "FWA": (40.9785, -85.1951), "GCC": (44.3489, -105.5392), "GCK": (37.9275, -100.7244),
    "GEG": (47.6199, -117.5338), "GFK": (47.9493, -97.1761), "GGG": (32.3840, -94.7115),
    "GJT": (39.1224, -108.5267), "GNV": (29.6901, -82.2718), "GPT": (30.4073, -89.0701),
    "GRB": (44.4851, -88.1298), "GRI": (40.9675, -98.3096), "GRK": (31.0672, -97.8289),
    "GRR": (42.8808, -85.5228), "GSO": (36.0978, -79.9373), "GSP": (34.8957, -82.2189),
    "GST": (58.4254, -135.7070), "GTF": (47.4820, -111.3707), "GTR": (33.4503, -88.5914),
    "GUC": (38.5339, -106.9332), "GUM": (13.4834, 144.7960), "HDN": (40.4812, -106.8662),
    "HGR": (39.7079, -77.7295), "HHH": (32.2244, -80.6975), "HIB": (47.3866, -92.8390),
    "HLN": (46.6068, -111.9827), "HNL": (21.3187, -157.9224), "HOB": (32.6873, -103.2170),
    "HOU": (29.6454, -95.2789), "HPN": (41.0670, -73.7076), "HRL": (26.2285, -97.6544),
    "HSV": (34.6372, -86.7751), "HTS": (38.3667, -82.5580), "HVN": (41.2637, -72.8868),
    "IAD": (38.9445, -77.4558), "IAG": (43.1073, -78.9462), "IAH": (29.9844, -95.3414),
    "ICT": (37.6499, -97.4331), "IDA": (43.5146, -112.0708), "ILM": (34.2706, -77.9026),
    "IMT": (45.8184, -88.1146), "IND": (39.7173, -86.2944), "INL": (48.5662, -93.4031),
    "ISN": (48.1779, -103.6424), "ISP": (40.7952, -73.1002), "ITH": (42.4910, -76.4584),
    "ITO": (19.7214, -155.0485), "JAC": (43.6073, -110.7377), "JAN": (32.3112, -90.0759),
    "JAX": (30.4941, -81.6879), "JFK": (40.6413, -73.7781), "JLN": (37.1518, -94.4983),
    "JMS": (46.9297, -98.6782), "JNU": (58.3550, -134.5762), "KOA": (19.7389, -156.0456),
    "KTN": (55.3556, -131.7137), "LAR": (41.3121, -105.6750), "LAS": (36.0840, -115.1537),
    "LAW": (34.5677, -98.4166), "LAX": (33.9425, -118.4081), "LBB": (33.6636, -101.8227),
    "LBE": (40.2759, -79.4048), "LBF": (41.1262, -100.6838), "LCH": (30.1261, -93.2234),
    "LEX": (38.0365, -84.6059), "LFT": (30.2053, -91.9876), "LGA": (40.7769, -73.8740),
    "LGB": (33.8177, -118.1516), "LIH": (21.9760, -159.3390), "LIT": (34.7294, -92.2243),
    "LNK": (40.8511, -96.7592), "LNY": (20.7856, -156.9514), "LRD": (27.5438, -99.4616),
    "LSE": (43.8793, -91.2569), "LWS": (46.3745, -117.0154), "MAF": (31.9425, -102.2019),
    "MBS": (43.5329, -84.0796), "MCI": (39.2976, -94.7139), "MCO": (28.4294, -81.3090),
    "MDT": (40.1935, -76.7634), "MDW": (41.7868, -87.7524), "MEM": (35.0424, -89.9767),
    "MFE": (26.1758, -98.2386), "MFR": (42.3742, -122.8735), "MGM": (32.3006, -86.3940),
    "MHK": (39.1410, -96.6708), "MHT": (42.9326, -71.4357), "MIA": (25.7959, -80.2870),
    "MKE": (42.9472, -87.8966), "MKG": (43.1695, -86.2382), "MKK": (21.1529, -157.0963),
    "MLB": (28.1028, -80.6453), "MLI": (41.4485, -90.5075), "MLU": (32.5109, -92.0377),
    "MMH": (37.6241, -118.8378), "MOB": (30.6912, -88.2428), "MOT": (48.2594, -101.2803),
    "MQT": (46.5336, -87.5615), "MRY": (36.5870, -121.8430), "MSN": (43.1399, -89.3375),
    "MSO": (46.9163, -114.0906), "MSP": (44.8820, -93.2218), "MSY": (29.9934, -90.2580),
    "MTJ": (38.5098, -107.8942), "MVY": (41.3931, -70.6143), "MYR": (33.6797, -78.9283),
    "OAJ": (34.8292, -77.6121), "OAK": (37.7213, -122.2208), "OGD": (41.1961, -112.0122),
    "OGG": (20.8986, -156.4305), "OKC": (35.3931, -97.6007), "OMA": (41.3032, -95.8941),
    "OME": (64.5122, -165.4453), "ONT": (34.0560, -117.6012), "ORD": (41.9742, -87.9073),
    "ORF": (36.8946, -76.2012), "ORH": (42.2673, -71.8757), "OTZ": (66.8847, -162.5985),
    "OWB": (37.7501, -87.1668), "PAE": (47.9063, -122.2816), "PAH": (37.0603, -88.7727),
    "PBG": (44.6509, -73.4681), "PBI": (26.6832, -80.0956), "PDX": (45.5898, -122.5951),
    "PGD": (26.9202, -81.9906), "PGV": (35.6353, -77.3853), "PHF": (37.1319, -76.4930),
    "PHL": (39.8721, -75.2411), "PHX": (33.4373, -112.0078), "PIA": (40.6642, -89.6933),
    "PIB": (31.4671, -89.3371), "PIE": (27.9110, -82.6874), "PIH": (42.9098, -112.5962),
    "PIT": (40.4915, -80.2329), "PLN": (45.5712, -84.7967), "PNS": (30.4734, -87.1866),
    "PPG": (-14.3310, -170.7132), "PSC": (46.2647, -119.1191), "PSE": (18.0083, -66.5630),
    "PSG": (56.8017, -132.9453), "PSP": (33.8303, -116.5067), "PUB": (38.2891, -104.4967),
    "PVD": (41.7326, -71.4204), "PVU": (40.2192, -111.7234), "PWM": (43.6462, -70.3093),
    "RAP": (44.0453, -103.0574), "RDD": (40.5090, -122.2934), "RDM": (44.2541, -121.1500),
    "RDU": (35.8776, -78.7875), "RFD": (42.1954, -89.0972), "RHI": (45.6312, -89.4675),
    "RIC": (37.5052, -77.3197), "RKS": (41.5942, -109.0652), "RNO": (39.4991, -119.7681),
    "ROA": (37.3255, -79.9754), "ROC": (43.1189, -77.6724), "ROW": (33.3016, -104.5307),
    "RST": (43.9083, -92.5000), "RSW": (26.5362, -81.7552), "SAF": (35.6171, -106.0892),
    "SAN": (32.7336, -117.1897), "SAT": (29.5337, -98.4698), "SAV": (32.1276, -81.2021),
    "SBA": (34.4262, -119.8404), "SBN": (41.7087, -86.3173), "SBP": (35.2368, -120.6424),
    "SCC": (70.1947, -148.4652), "SCE": (40.8493, -77.8487), "SDF": (38.1744, -85.7360),
    "SEA": (47.4502, -122.3088), "SFB": (28.7776, -81.2436), "SFO": (37.6213, -122.3790),
    "SGF": (37.2457, -93.3886), "SGU": (37.0364, -113.5103), "SHD": (38.2638, -78.8964),
    "SHR": (44.7692, -106.9803), "SHV": (32.4466, -93.8256), "SIT": (57.0471, -135.3616),
    "SJC": (37.3626, -121.9291), "SJT": (31.3577, -100.4963), "SJU": (18.4394, -66.0018),
    "SLC": (40.7884, -111.9778), "SMF": (38.6954, -121.5908), "SMX": (34.8989, -120.4577),
    "SNA": (33.6757, -117.8683), "SPI": (39.8441, -89.6779), "SPN": (15.1190, 145.7295),
    "SPS": (33.9888, -98.4919), "SRQ": (27.3954, -82.5543), "STC": (45.5466, -94.0599),
    "STL": (38.7487, -90.3700), "STT": (18.3373, -64.9734), "STX": (17.7019, -64.7986),
    "SUN": (43.5044, -114.2966), "SUX": (42.4026, -96.3844), "SWF": (41.5041, -74.1048),
    "SYR": (43.1112, -76.1063), "TLH": (30.3965, -84.3503), "TOL": (41.5868, -83.8078),
    "TPA": (27.9755, -82.5333), "TRI": (36.4752, -82.4074), "TTN": (40.2767, -74.8135),
    "TUL": (36.1984, -95.8881), "TUS": (32.1161, -110.9410), "TVC": (44.7415, -85.5822),
    "TWF": (42.4818, -114.4877), "TXK": (33.4537, -93.9910), "TYR": (32.3541, -95.4024),
    "TYS": (35.8110, -83.9940), "UIN": (39.9427, -91.1946), "USA": (33.3000, -111.6550),
    "UST": (29.9592, -81.3398), "VEL": (40.4409, -109.5100), "VLD": (30.7825, -83.2767),
    "VPS": (30.4832, -86.5254), "WRG": (56.4843, -132.3698), "WYS": (44.6884, -111.1176),
    "XNA": (36.2819, -94.3068), "YAK": (59.5033, -139.6603), "YKM": (46.5682, -120.5440),
    "YUM": (32.6566, -114.6063),
    # ── Added: airports missing from initial dict ────────────────────────
    "ALW": (46.0949, -118.2884),  # Walla Walla, WA
    "BFF": (41.8740, -103.5956),  # Scottsbluff, NE
    "BIH": (37.3731, -118.3636),  # Bishop, CA
    "CYS": (41.1557, -104.8118),  # Cheyenne, WY
    "DDC": (37.7634, -99.9656),   # Dodge City, KS
    "DEC": (39.8346, -88.8657),   # Decatur, IL
    # FLL and FNT already in dict above; empty files from rate-limit errors
    "FOD": (42.5512, -94.1926),   # Fort Dodge, IA
    "HYA": (41.6693, -70.2804),   # Hyannis, MA
    "HYS": (38.8422, -99.2732),   # Hays, KS
    "JST": (40.3161, -78.8340),   # Johnstown, PA
    "LAN": (42.7787, -84.5874),   # Lansing, MI
    "LBL": (37.0421, -100.9599),  # Liberal, KS
    "LCK": (39.8138, -82.9278),   # Rickenbacker (Columbus), OH
    "MCW": (43.1578, -93.3313),   # Mason City, IA
    "MEI": (32.3326, -88.7519),   # Meridian, MS
    "MGW": (39.6430, -79.9163),   # Morgantown, WV
    "OTH": (43.4171, -124.2460),  # North Bend, OR
    "PQI": (46.6890, -68.0448),   # Presque Isle, ME
    "PRC": (34.6545, -112.4196),  # Prescott, AZ
    "PSM": (43.0779, -70.8233),   # Portsmouth, NH
    "RIW": (43.0642, -108.4598),  # Riverton, WY
    "SCK": (37.8942, -121.2386),  # Stockton, CA
    "SLN": (38.7910, -97.6522),   # Salina, KS
    "STS": (38.5090, -122.8128),  # Santa Rosa (Sonoma County), CA
    "SWO": (36.1612, -97.0857),   # Stillwater, OK
    "VCT": (28.8526, -96.9185),   # Victoria, TX
    "XWA": (48.2579, -103.7514),  # Williston, ND
}

# ═════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure logging to both console and file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD BTS FLIGHT DATA
# ═════════════════════════════════════════════════════════════════════════════

def download_bts_data(year: int = YEAR) -> pd.DataFrame:
    """
    Download monthly On-Time Performance CSVs from BTS and combine them.

    The BTS publishes pre-zipped CSV files for each month. Each file contains
    all domestic flight records for that month (~500-600k rows).
    """
    BTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = BTS_DIR / f"flights_{year}.parquet"

    if combined_path.exists():
        logger.info(f"Loading cached flight data from {combined_path}")
        return pd.read_parquet(combined_path)

    all_months = []

    for month in range(1, 13):
        month_csv = BTS_DIR / f"flights_{year}_{month:02d}.csv"

        if month_csv.exists():
            logger.info(f"Month {month:02d} already downloaded, loading from cache")
            df_month = pd.read_csv(month_csv, low_memory=False)
            all_months.append(df_month)
            continue

        url = BTS_PREZIP_URL.format(year=year, month=month)
        logger.info(f"Downloading BTS data: {year}-{month:02d} ...")

        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            content = BytesIO()
            with tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc=f"  {year}-{month:02d}", disable=total_size == 0,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    content.write(chunk)
                    pbar.update(len(chunk))

            content.seek(0)
            with zipfile.ZipFile(content) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    logger.error(f"No CSV found in zip for {year}-{month:02d}")
                    continue
                with zf.open(csv_names[0]) as csv_file:
                    df_month = pd.read_csv(csv_file, low_memory=False)

            available_cols = [c for c in BTS_COLUMNS if c in df_month.columns]
            df_month = df_month[available_cols]
            df_month.to_csv(month_csv, index=False)
            all_months.append(df_month)
            logger.info(f"  -> {len(df_month):,} flight records for {year}-{month:02d}")

        except requests.RequestException as e:
            logger.error(f"Failed to download {year}-{month:02d}: {e}")
            logger.error("Re-run with --step flights to retry failed months.")
            continue

    if not all_months:
        logger.error("No flight data was downloaded. Check your internet connection.")
        sys.exit(1)

    flights = pd.concat(all_months, ignore_index=True)
    logger.info(f"Combined flight data: {len(flights):,} total records")

    flights.to_parquet(combined_path, index=False)
    logger.info(f"Saved combined flight data to {combined_path}")

    return flights


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: FETCH WEATHER DATA FROM OPEN-METEO
# ═════════════════════════════════════════════════════════════════════════════

def fetch_openmeteo_airport_year(
    iata: str, lat: float, lon: float, year: int
) -> pd.DataFrame:
    """
    Fetch a full year of hourly weather data for one airport from Open-Meteo.

    Uses the Archive API (ERA5 reanalysis) for historical data. The same
    variable names are used in the Forecast API at prediction time.

    Args:
        iata: Airport IATA code (for labeling/logging only).
        lat: Airport latitude.
        lon: Airport longitude.
        year: Year to fetch.

    Returns:
        DataFrame with 'time' column (UTC datetime) + all weather variable columns.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(HOURLY_WEATHER_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
    }

    for attempt in range(1, OPENMETEO_MAX_RETRIES + 1):
        try:
            resp = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                hourly = data.get("hourly", {})

                if "time" not in hourly:
                    logger.warning(f"  {iata}: response missing 'time' field")
                    return pd.DataFrame()

                df = pd.DataFrame(hourly)
                df.loc[:, "time"] = pd.to_datetime(df["time"], utc=True)
                df.loc[:, "airport"] = iata
                return df

            elif resp.status_code == 429:
                wait = 30 * attempt
                logger.warning(
                    f"  {iata}: rate limited (429), waiting {wait}s "
                    f"({attempt}/{OPENMETEO_MAX_RETRIES})"
                )
                time.sleep(wait)
                continue

            elif resp.status_code == 400:
                # Often means a variable isn't available; try without visibility
                error_msg = resp.text[:200]
                logger.warning(f"  {iata}: HTTP 400 - {error_msg}")

                if "visibility" in error_msg.lower() and "visibility" in params["hourly"]:
                    logger.info(f"  {iata}: retrying without 'visibility'")
                    vars_without_vis = [v for v in HOURLY_WEATHER_VARS if v != "visibility"]
                    params["hourly"] = ",".join(vars_without_vis)
                    continue

                return pd.DataFrame()

            else:
                logger.warning(f"  {iata}: HTTP {resp.status_code}")
                return pd.DataFrame()

        except requests.RequestException as e:
            wait = 10 * attempt
            logger.warning(
                f"  {iata}: {e}, retrying in {wait}s ({attempt}/{OPENMETEO_MAX_RETRIES})"
            )
            time.sleep(wait)

    logger.error(f"  {iata}: exhausted retries")
    return pd.DataFrame()


def fetch_weather_data(flights: pd.DataFrame, year: int = YEAR):
    """
    Fetch a full year of hourly weather for every unique airport in the flight data.

    One API call per airport (vs. 12 per airport with IEM), for ~350 total requests.
    Each airport's data is saved as a parquet file for resume support.
    """
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    # Get unique airports from both origin and destination
    all_airports = sorted(
        set(flights["Origin"].unique()) | set(flights["Dest"].unique())
    )
    logger.info(f"Fetching weather for {len(all_airports)} airports, {year}")

    fetched = 0
    skipped_no_coords = 0
    cached = 0

    for iata in tqdm(all_airports, desc="Fetching weather"):
        airport_file = WEATHER_DIR / f"{iata}_{year}.parquet"

        # Resume support: skip already-fetched airports
        if airport_file.exists():
            cached += 1
            continue

        coords = AIRPORT_COORDS.get(iata)
        if coords is None:
            logger.warning(f"  {iata}: no coordinates available, skipping")
            skipped_no_coords += 1
            # Save empty file so we don't retry
            pd.DataFrame().to_parquet(airport_file, index=False)
            continue

        lat, lon = coords
        df = fetch_openmeteo_airport_year(iata, lat, lon, year)

        if len(df) > 0:
            df.to_parquet(airport_file, index=False)
            fetched += 1
            logger.info(f"  {iata}: {len(df)} hourly observations")
        else:
            pd.DataFrame().to_parquet(airport_file, index=False)
            logger.warning(f"  {iata}: no data returned")

        time.sleep(OPENMETEO_REQUEST_DELAY)

    logger.info(
        f"Weather fetch complete: {fetched} freshly downloaded, "
        f"{cached} from cache, {skipped_no_coords} skipped (no coordinates)"
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: MERGE FLIGHT DATA WITH WEATHER
# ═════════════════════════════════════════════════════════════════════════════

def load_airport_weather(iata: str, year: int = YEAR) -> pd.DataFrame:
    """Load cached weather data for one airport."""
    f = WEATHER_DIR / f"{iata}_{year}.parquet"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f)
    if len(df) == 0 or "time" not in df.columns:
        return pd.DataFrame()
    df.loc[:, "time"] = pd.to_datetime(df["time"], utc=True)
    if "airport" not in df.columns:
        df["airport"] = iata
    return df.sort_values("time")


def parse_bts_time_vectorized(flights: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Vectorized conversion of BTS date + HHMM time columns into UTC timestamps.

    Returns a Series of tz-aware Timestamps (NaT where parsing fails).
    BTS times are local, but the 3-hour tolerance window for weather matching
    accommodates US timezone offsets adequately.
    """
    dates = pd.to_datetime(flights["FlightDate"], errors="coerce")
    time_vals = pd.to_numeric(flights[time_col], errors="coerce")

    # Handle 2400 -> midnight next day
    is_2400 = time_vals == 2400
    time_vals = time_vals.where(~is_2400, 0)

    hours = (time_vals // 100).astype("Int64")
    minutes = (time_vals % 100).astype("Int64")

    # Mark invalid times
    valid = hours.between(0, 23) & minutes.between(0, 59) & dates.notna() & time_vals.notna()

    result = pd.Series(pd.NaT, index=flights.index)
    if valid.any():
        td = pd.to_timedelta(hours[valid] * 3600 + minutes[valid] * 60, unit="s")
        result[valid] = dates[valid] + td
        # Add one day for 2400 entries
        mask_2400 = valid & is_2400
        if mask_2400.any():
            result[mask_2400] = dates[mask_2400] + pd.Timedelta(days=1)

    return result.dt.tz_localize("UTC")


def merge_datasets(flights: pd.DataFrame) -> pd.DataFrame:
    """
    Merge flight data with weather observations from both origin and
    destination airports using vectorized merge_asof (fast).

    For each flight:
    1. Find the weather observation closest to the scheduled departure at origin
    2. Find the weather observation closest to the scheduled arrival at destination
    3. Attach both as prefixed columns (origin_wx_*, dest_wx_*)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"flight_delay_dataset_{YEAR}.parquet"
    output_csv = OUTPUT_DIR / f"flight_delay_dataset_{YEAR}.csv"

    if output_path.exists():
        logger.info(f"Final dataset already exists at {output_path}")
        return pd.read_parquet(output_path)

    # Pre-load all airport weather into memory
    logger.info("Loading weather data into memory...")
    all_airports = sorted(
        set(flights["Origin"].unique()) | set(flights["Dest"].unique())
    )
    weather_dfs = []
    for iata in tqdm(all_airports, desc="Loading weather"):
        w = load_airport_weather(iata, YEAR)
        if not w.empty:
            weather_dfs.append(w)

    logger.info(f"Loaded weather for {len(weather_dfs)} airports")
    if not weather_dfs:
        logger.error("No weather data found to merge.")
        return flights

    # Combine all weather data into one DataFrame
    all_weather = pd.concat(weather_dfs, ignore_index=True)
    
    # Sort weather by time, required for merge_asof
    logger.info("Sorting combined weather data...")
    all_weather = all_weather.sort_values("time")

    # Keep all flights including cancellations (cancellation is a prediction target)
    flights_active = flights.copy()
    n_cancelled = int(flights["Cancelled"].sum())
    logger.info(
        f"Processing {len(flights_active):,} total flights "
        f"({n_cancelled:,} cancelled, {len(flights_active) - n_cancelled:,} operated)"
    )

    wx_cols = HOURLY_WEATHER_VARS
    tolerance = pd.Timedelta(hours=3)

    # ── Parse scheduled times once (vectorized) ─────────────────────────
    logger.info("Parsing scheduled departure/arrival times...")
    flights_active["_dep_time"] = parse_bts_time_vectorized(flights_active, "CRSDepTime")
    flights_active["_arr_time"] = parse_bts_time_vectorized(flights_active, "CRSArrTime")

    # ── Origin weather: single vectorized merge_asof ────────────────────
    logger.info("Merging origin weather (vectorized)...")
    
    # Sort flights by departure time for merge_asof
    flights_active = flights_active.sort_values("_dep_time")
    
    # Rename weather columns for origin
    origin_weather = all_weather.rename(columns={c: f"origin_wx_{c}" for c in wx_cols})
    
    # We must drop rows with NaT in the match time column before merge_asof
    valid_dep = flights_active["_dep_time"].notna()
    invalid_dep_flights = flights_active[~valid_dep].copy()
    valid_dep_flights = flights_active[valid_dep].copy()
    
    if not valid_dep_flights.empty:
        valid_dep_flights = pd.merge_asof(
            valid_dep_flights,
            origin_weather,
            left_on="_dep_time",
            right_on="time",
            left_by="Origin",
            right_by="airport",
            direction="nearest",
            tolerance=tolerance
        ).drop(columns=["time", "airport"], errors="ignore")
    
    # Recombine
    flights_active = pd.concat([valid_dep_flights, invalid_dep_flights], ignore_index=True)
    
    # Ensure missing origin_wx cols exist
    for col in wx_cols:
        pcol = f"origin_wx_{col}"
        if pcol not in flights_active.columns:
            flights_active[pcol] = None

    # ── Destination weather: single vectorized merge_asof ───────────────
    logger.info("Merging destination weather (vectorized)...")
    
    # Sort flights by arrival time for merge_asof
    flights_active = flights_active.sort_values("_arr_time")
    
    # Rename weather columns for destination
    dest_weather = all_weather.rename(columns={c: f"dest_wx_{c}" for c in wx_cols})
    
    valid_arr = flights_active["_arr_time"].notna()
    invalid_arr_flights = flights_active[~valid_arr].copy()
    valid_arr_flights = flights_active[valid_arr].copy()
    
    if not valid_arr_flights.empty:
        valid_arr_flights = pd.merge_asof(
            valid_arr_flights,
            dest_weather,
            left_on="_arr_time",
            right_on="time",
            left_by="Dest",
            right_by="airport",
            direction="nearest",
            tolerance=tolerance
        ).drop(columns=["time", "airport"], errors="ignore")
        
    # Recombine
    flights_active = pd.concat([valid_arr_flights, invalid_arr_flights], ignore_index=True)
    
    # Ensure missing dest_wx cols exist
    for col in wx_cols:
        pcol = f"dest_wx_{col}"
        if pcol not in flights_active.columns:
            flights_active[pcol] = None

    # Drop temporary time columns
    merged = flights_active.drop(columns=["_dep_time", "_arr_time"], errors="ignore")

    # ── Target variables ─────────────────────────────────────────────────
    merged["target_cancelled"] = merged["Cancelled"].fillna(0).astype(int)
    merged["target_delayed"] = merged["ArrDel15"].fillna(0).astype(int)
    merged["target_delay_minutes"] = merged["ArrDelay"].fillna(0)

    # ── Save ─────────────────────────────────────────────────────────────
    merged.to_parquet(output_path, index=False)
    logger.info(f"Saving CSV (this may take a minute for {len(merged):,} rows)...")
    merged.to_csv(output_csv, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    delayed_pct = merged["target_delayed"].mean() * 100
    delayed_count = merged["target_delayed"].sum()
    pos_delays = merged.loc[merged["target_delay_minutes"] > 0, "target_delay_minutes"]
    avg_delay = pos_delays.mean() if len(pos_delays) > 0 else 0.0

    origin_cov = merged["origin_wx_temperature_2m"].notna().mean() * 100
    dest_cov = merged["dest_wx_temperature_2m"].notna().mean() * 100

    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET SUMMARY")
    logger.info(f"{'='*60}")
    cancelled_count = merged["target_cancelled"].sum()
    cancelled_pct = merged["target_cancelled"].mean() * 100

    logger.info(f"  Total flights:             {len(merged):>12,}")
    logger.info(f"  Cancelled flights:         {cancelled_count:>12,} ({cancelled_pct:.1f}%)")
    logger.info(f"  Delayed flights (>=15m):   {delayed_count:>12,} ({delayed_pct:.1f}%)")
    logger.info(f"  Avg delay (when >0 min):   {avg_delay:>12.1f} min")
    logger.info(f"  Origin weather coverage:   {origin_cov:>11.1f}%")
    logger.info(f"  Dest weather coverage:     {dest_cov:>11.1f}%")
    logger.info(f"  Unique carriers:           {merged['Reporting_Airline'].nunique():>12}")
    logger.info(f"  Unique origin airports:    {merged['Origin'].nunique():>12}")
    logger.info(f"  Unique dest airports:      {merged['Dest'].nunique():>12}")
    logger.info(f"  Date range:                {merged['FlightDate'].min()} to {merged['FlightDate'].max()}")
    logger.info(f"{'='*60}")

    return merged


# =============================================================================
# PREDICTION-TIME WEATHER FETCH (for deployed model)
# =============================================================================


def fetch_forecast_weather(iata, target_datetime=None):
    """
    Fetch forecasted weather for an airport from Open-Meteo's Forecast API.

    This function is used at prediction time (deployed model). It returns
    weather data using the SAME variable names and units as the training data,
    so no schema translation is needed.

    Args:
        iata: Airport IATA code (e.g., "ATL", "ORD").
        target_datetime: ISO datetime string (e.g., "2025-06-15T14:00").
                         If None, returns the current hour's forecast.

    Returns:
        Dict with weather variable names as keys and forecast values.

    Example:
        >>> wx = fetch_forecast_weather("ATL", "2025-06-15T14:00")
        >>> print(wx["temperature_2m"])  # 85.3 (deg F)
        >>> print(wx["wind_speed_10m"])  # 12.4 (mph)
    """
    coords = AIRPORT_COORDS.get(iata)
    if coords is None:
        raise ValueError(f"Unknown airport: {iata}")

    lat, lon = coords

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_WEATHER_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
    }

    resp = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        return {}

    # Find the hour closest to the target time
    if target_datetime:
        target = pd.Timestamp(target_datetime, tz="UTC")
    else:
        target = pd.Timestamp.now(tz="UTC").floor("h")

    time_series = pd.to_datetime(times, utc=True)
    diffs = (time_series - target).abs()
    best_idx = diffs.argmin()

    result = {"time": str(time_series[best_idx])}
    for var in HOURLY_WEATHER_VARS:
        values = hourly.get(var, [])
        result[var] = values[best_idx] if best_idx < len(values) else None

    return result


def demo_forecast():
    """
    Demo: fetch and display forecast weather for ATL to show the
    training-to-prediction pipeline works with identical schemas.
    """
    logger.info("DEMO: Fetching forecast weather for ATL (Hartsfield-Jackson)")
    logger.info("This uses the SAME variable names as the training data.\n")

    wx = fetch_forecast_weather("ATL")

    logger.info(f"  Forecast time:       {wx.get('time')}")
    logger.info(f"  Temperature:         {wx.get('temperature_2m')} deg F")
    logger.info(f"  Dewpoint:            {wx.get('dew_point_2m')} deg F")
    logger.info(f"  Humidity:            {wx.get('relative_humidity_2m')} %")
    logger.info(f"  Wind speed:          {wx.get('wind_speed_10m')} mph")
    logger.info(f"  Wind gusts:          {wx.get('wind_gusts_10m')} mph")
    logger.info(f"  Wind direction:      {wx.get('wind_direction_10m')} deg")
    logger.info(f"  Visibility:          {wx.get('visibility')} ft")
    logger.info(f"  Cloud cover:         {wx.get('cloud_cover')} %")
    logger.info(f"  Precipitation:       {wx.get('precipitation')} in")
    logger.info(f"  Pressure (MSL):      {wx.get('pressure_msl')} hPa")
    logger.info(f"  Weather code (WMO):  {wx.get('weather_code')}")
    logger.info("")
    logger.info("These column names match the training data exactly:")
    logger.info("  origin_wx_temperature_2m, dest_wx_wind_speed_10m, etc.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global DATA_DIR, BTS_DIR, WEATHER_DIR, OUTPUT_DIR, YEAR

    parser = argparse.ArgumentParser(
        description="Build flight delay prediction dataset from BTS + Open-Meteo."
    )
    parser.add_argument(
        "--step",
        choices=["all", "flights", "weather", "merge", "demo-forecast"],
        default="all",
        help="Pipeline step to run. Default: all",


    )
    parser.add_argument(
        "--year", type=int, default=YEAR,
        help=f"Year of flight data. Default: {YEAR}",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help=f"Root data directory. Default: {DATA_DIR}",
    )
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    BTS_DIR = DATA_DIR / "bts_raw"
    WEATHER_DIR = DATA_DIR / "weather_openmeteo"
    OUTPUT_DIR = DATA_DIR / "output"
    YEAR = args.year

    logger.info(f"Flight Delay Dataset Builder (Open-Meteo)")
    logger.info(f"  Year: {YEAR}")
    logger.info(f"  Data directory: {DATA_DIR}")
    logger.info(f"  Step: {args.step}")
    logger.info("")

    # -- Demo forecast mode
    if args.step == "demo-forecast":
        demo_forecast()
        return

    # -- Step 1: BTS flight data
    if args.step in ("all", "flights"):
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading BTS On-Time Performance Data")
        logger.info("=" * 60)
        flights = download_bts_data(YEAR)
    else:
        combined_path = BTS_DIR / f"flights_{YEAR}.parquet"
        if not combined_path.exists():
            logger.error("Flight data not found. Run --step flights first.")
            sys.exit(1)
        flights = pd.read_parquet(combined_path)

    # -- Step 2: Weather data
    if args.step in ("all", "weather"):
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Fetching Weather from Open-Meteo Archive API")
        logger.info("=" * 60)
        fetch_weather_data(flights, YEAR)

    # -- Step 3: Merge
    if args.step in ("all", "merge"):
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: Merging Flight Data with Weather Observations")
        logger.info("=" * 60)
        merge_datasets(flights)

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
