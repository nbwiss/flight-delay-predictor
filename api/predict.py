"""
Flight Delay Prediction API
Vercel Python Serverless Function

Loads V2 models (A: cancel, B: delay) + V3 model (C: minutes).
Fetches live weather from Open-Meteo forecast API.
Returns predictions + SHAP factors + Gemini LLM analysis.
"""

import os
import json
import math
import traceback
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import joblib
import requests
import xgboost as xgb

# ═══════════════════════════════════════════════════════════════════════
# PATHS — resolve relative to this file for Vercel serverless
# ═════════════════════════════════════════════════════════��═════════════

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
V2_DIR = BASE_DIR / "models" / "v2"
V3_DIR = BASE_DIR / "models" / "v3"

# ═══════════════════════════════════════════════════════════════════════
# LAZY-LOADED GLOBALS (load once per cold start)
# ═══════════════════════════════════════════════════════════════════════

_models = {}


def _load_models():
    """Load all model artifacts once per cold start."""
    if _models:
        return

    # V2 artifacts (Models A + B)
    _models["model_a"] = joblib.load(V2_DIR / "model_a_cancellation.joblib")
    _models["model_b"] = joblib.load(V2_DIR / "model_b_delay.joblib")
    _models["scaler"] = joblib.load(V2_DIR / "scaler.joblib")
    _models["ord_encoder"] = joblib.load(V2_DIR / "ord_encoder.joblib")
    _models["medians"] = joblib.load(V2_DIR / "medians.joblib")
    _models["num_cols"] = joblib.load(V2_DIR / "num_cols.joblib")
    _models["cat_cols"] = joblib.load(V2_DIR / "cat_cols.joblib")
    _models["feature_names"] = joblib.load(V2_DIR / "feature_names.joblib")

    # V3 model (Model C) is no longer loaded to save memory.

    # Static data (airport coordinates only — no historical route lookup)
    with open(DATA_DIR / "airport_coords.json") as f:
        _models["airport_coords"] = json.load(f)

    print(f"Models loaded. Features: {len(_models['feature_names'])}")


def haversine_miles(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between two points in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def estimate_elapsed_minutes(distance_miles):
    """Estimate flight time from distance. Accounts for taxi + cruise."""
    # ~30 min taxi/climb/descent overhead + cruise at ~500 mph
    if distance_miles <= 0:
        return 60
    return 30 + (distance_miles / 500) * 60


# ═══════════════════════════════════════════════════════════════════════
# OPEN-METEO WEATHER FETCH
# ═══════════════════════════════════════════════════════════════════════

WEATHER_VARS = [
    "temperature_2m", "dew_point_2m", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "weather_code",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility", "pressure_msl", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
]


def fetch_weather(iata: str, target_dt: str) -> dict:
    """
    Fetch weather for an airport at a target datetime.
    - Past dates  -> Open-Meteo Historical Archive API (ERA5 reanalysis, same
                     source as training data).
    - Today/future -> Open-Meteo Forecast API (up to ~16 days out).

    Both endpoints use timezone=auto so returned times are in the airport's
    local timezone, matching the local departure/arrival times the user enters.
    """
    coords = _models["airport_coords"].get(iata)
    if not coords:
        return {v: None for v in WEATHER_VARS}

    target = datetime.fromisoformat(target_dt)
    target_date = target.date()
    today = datetime.utcnow().date()

    # Common params shared by both APIs
    base_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(WEATHER_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }

    is_historical = target_date < today

    if is_historical:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = dict(base_params)
        params["start_date"] = target_date.isoformat()
        params["end_date"] = target_date.isoformat()
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = base_params

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            return {v: None for v in WEATHER_VARS}

        # Find closest hour to target (both naive local-time datetimes)
        best_idx = 0
        best_diff = float("inf")
        for i, t in enumerate(times):
            dt = datetime.fromisoformat(t)
            diff = abs((dt - target).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        result = {}
        for var in WEATHER_VARS:
            vals = hourly.get(var, [])
            result[var] = vals[best_idx] if best_idx < len(vals) else None
        return result

    except Exception as e:
        api_type = "archive" if is_historical else "forecast"
        print(f"Weather fetch failed for {iata} ({api_type}): {e}")
        return {v: None for v in WEATHER_VARS}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

# US holidays for 2024-2027 (approximate, covers common travel dates)
HOLIDAYS = {
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-07-04",
    "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-07-04",
    "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-07-03",
    "2026-09-07", "2026-10-12", "2026-11-11", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-05-31", "2027-07-05",
    "2027-09-06", "2027-10-11", "2027-11-11", "2027-11-25", "2027-12-24",
}


def get_time_block(hour: int) -> str:
    """Map hour to time block (matching training pipeline)."""
    if hour < 6:
        return "early_morning"
    elif hour < 9:
        return "morning"
    elif hour < 12:
        return "late_morning"
    elif hour < 15:
        return "afternoon"
    elif hour < 18:
        return "late_afternoon"
    elif hour < 21:
        return "evening"
    else:
        return "night"


def derive_weather_features(wx: dict, prefix: str) -> dict:
    """Compute derived weather features matching training pipeline."""
    feats = {}

    # Raw weather vars with prefix
    for var in WEATHER_VARS:
        feats[f"{prefix}{var}"] = wx.get(var)

    temp = wx.get("temperature_2m")
    dew = wx.get("dew_point_2m")
    rain_val = wx.get("rain") or 0
    snow_val = wx.get("snowfall") or 0
    code = wx.get("weather_code") or 0
    low_cloud = wx.get("cloud_cover_low") or 0
    vis = wx.get("visibility")

    # Derived features
    feats[f"{prefix}temp_dew_spread"] = (temp - dew) if (temp is not None and dew is not None) else None
    feats[f"{prefix}total_precip"] = rain_val + snow_val
    feats[f"{prefix}adverse_wx"] = 1 if code >= 50 else 0
    feats[f"{prefix}tstorm"] = 1 if code >= 95 else 0
    feats[f"{prefix}low_ceiling"] = 1 if low_cloud >= 80 else 0
    feats[f"{prefix}low_vis"] = 1 if (vis is not None and vis < 5280) else 0  # less than 1 mile

    return feats


def build_feature_vector(
    carrier: str, flight_num: int, origin: str,
    dep_date: str, dep_time: str,
    dest: str, distance: float, elapsed: float, arr_time: int,
    origin_wx: dict, dest_wx: dict,
) -> dict:
    """Assemble all 61 features matching the training schema."""
    dt = datetime.fromisoformat(f"{dep_date}T{dep_time}")
    dep_hour = dt.hour
    dep_minute = dt.minute

    # Compute arrival hour from CRSArrTime (HHMM format)
    arr_hour = arr_time // 100 if arr_time else dep_hour + int(elapsed / 60) if elapsed else dep_hour

    # Date features
    day_of_week = dt.isoweekday()  # 1=Monday, 7=Sunday
    is_weekend = 1 if day_of_week >= 6 else 0

    # Check holiday (date itself or +/- 1 day)
    date_str = dep_date
    prev_day = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    is_holiday = 1 if (date_str in HOLIDAYS or prev_day in HOLIDAYS or next_day in HOLIDAYS) else 0

    features = {
        "Month": dt.month,
        "Quarter": (dt.month - 1) // 3 + 1,
        "DayofMonth": dt.day,
        "DayOfWeek": day_of_week,
        "CRSElapsedTime": elapsed,
        "Distance": distance,
        "dep_hour": dep_hour,
        "dep_minute": dep_minute,
        "arr_hour": arr_hour,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "time_block": get_time_block(dep_hour),
        "Origin": origin,
        "Dest": dest,
        "Reporting_Airline": carrier,
    }

    # Weather features
    features.update(derive_weather_features(origin_wx, "origin_wx_"))
    features.update(derive_weather_features(dest_wx, "dest_wx_"))

    return features


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING + INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def preprocess_v2(features: dict) -> np.ndarray:
    """Apply V2 preprocessing: median imputation, ordinal encode, scale."""
    num_cols = _models["num_cols"]
    cat_cols = _models["cat_cols"]
    feature_names = _models["feature_names"]
    medians = _models["medians"]
    scaler = _models["scaler"]
    encoder = _models["ord_encoder"]

    # Numeric values with median imputation
    num_vals = []
    for col in num_cols:
        val = features.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = float(medians.get(col, 0)) if hasattr(medians, 'get') else 0
        num_vals.append(float(val))

    num_arr = np.array([num_vals])
    num_scaled = scaler.transform(num_arr)

    # Categorical values
    cat_vals = [[features.get(col, "UNKNOWN") for col in cat_cols]]
    try:
        cat_encoded = encoder.transform(cat_vals)
    except Exception:
        # Handle unseen categories: set to 0
        cat_encoded = np.zeros((1, len(cat_cols)))

    # Combine in feature_names order
    combined = np.zeros((1, len(feature_names)))
    num_idx = 0
    cat_idx = 0
    for i, fname in enumerate(feature_names):
        if fname in cat_cols:
            col_pos = cat_cols.index(fname)
            combined[0, i] = cat_encoded[0, col_pos]
        elif fname in num_cols:
            col_pos = num_cols.index(fname)
            combined[0, i] = num_scaled[0, col_pos]

    return combined



def predict(features: dict) -> dict:
    """Run the cascade: Model A (cancel) -> Model B (delay) -> Model C (minutes)."""

    # V2 preprocessing for Models A and B
    X_v2 = preprocess_v2(features)

    # Model A: Cancellation probability
    model_a = _models["model_a"]
    cancel_prob = float(model_a.predict_proba(X_v2)[0, 1])

    # Model B: Delay probability
    model_b = _models["model_b"]
    delay_prob = float(model_b.predict_proba(X_v2)[0, 1])



    # SHAP factors for delay model (Model B)
    feature_names = _models["feature_names"]
    try:
        booster = model_b
        if hasattr(booster, 'get_booster'):
            booster = booster.get_booster()
            import shap
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(X_v2)
            shap_list = []
            for idx in np.argsort(-np.abs(shap_values[0]))[:8]:
                fname = feature_names[idx]
                shap_list.append({
                    "feature": fname,
                    "display_name": _friendly_name(fname),
                    "value": round(float(shap_values[0][idx]), 4),
                })
        else:
            shap_list = []
    except Exception as e:
        print(f"SHAP computation skipped: {e}")
        shap_list = []

    return {
        "cancel_probability": round(cancel_prob, 4),
        "delay_probability": round(delay_prob, 4),
        "shap_factors": shap_list,
    }


FRIENDLY_NAMES = {
    "dep_hour": "Departure Hour",
    "arr_hour": "Arrival Hour",
    "dep_minute": "Departure Minute",
    "DayOfWeek": "Day of Week",
    "DayofMonth": "Day of Month",
    "Month": "Month",
    "Quarter": "Quarter",
    "Distance": "Flight Distance",
    "CRSElapsedTime": "Scheduled Flight Time",
    "is_weekend": "Weekend",
    "is_holiday": "Holiday",
    "time_block": "Time of Day Block",
    "Origin": "Origin Airport",
    "Dest": "Destination Airport",
    "Reporting_Airline": "Airline",
    "origin_wx_temperature_2m": "Origin Temperature",
    "origin_wx_wind_speed_10m": "Origin Wind Speed",
    "origin_wx_wind_gusts_10m": "Origin Wind Gusts",
    "origin_wx_visibility": "Origin Visibility",
    "origin_wx_precipitation": "Origin Precipitation",
    "origin_wx_cloud_cover_low": "Origin Low Clouds",
    "origin_wx_weather_code": "Origin Weather Code",
    "origin_wx_temp_dew_spread": "Origin Fog Risk",
    "origin_wx_adverse_wx": "Origin Adverse Weather",
    "origin_wx_tstorm": "Origin Thunderstorm",
    "origin_wx_low_ceiling": "Origin Low Ceiling",
    "origin_wx_low_vis": "Origin Low Visibility",
    "origin_wx_snowfall": "Origin Snowfall",
    "dest_wx_temperature_2m": "Dest Temperature",
    "dest_wx_wind_speed_10m": "Dest Wind Speed",
    "dest_wx_wind_gusts_10m": "Dest Wind Gusts",
    "dest_wx_visibility": "Dest Visibility",
    "dest_wx_precipitation": "Dest Precipitation",
    "dest_wx_cloud_cover_low": "Dest Low Clouds",
    "dest_wx_weather_code": "Dest Weather Code",
    "dest_wx_temp_dew_spread": "Dest Fog Risk",
    "dest_wx_adverse_wx": "Dest Adverse Weather",
    "dest_wx_tstorm": "Dest Thunderstorm",
    "dest_wx_low_ceiling": "Dest Low Ceiling",
    "dest_wx_low_vis": "Dest Low Visibility",
    "dest_wx_snowfall": "Dest Snowfall",
}


def _friendly_name(feature: str) -> str:
    return FRIENDLY_NAMES.get(feature, feature.replace("_", " ").title())


# ═══════════════════════════════════════════════════════════════════════
# GEMINI LLM SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def llm_synthesize(prediction: dict, flight_info: dict, origin_wx: dict, dest_wx: dict) -> str:
    """
    Use Gemini 2.5 Flash to synthesize prediction results into
    a human-readable analysis. Falls back to template if no API key.
    """
    if not GEMINI_API_KEY:
        return _template_analysis(prediction, flight_info, origin_wx, dest_wx)

    prompt = f"""You are a flight delay analyst. Given the ML model outputs below, write a 2-3 sentence
plain-English analysis for the passenger. Be specific about the top contributing factors.
Do not use bullet points. Be concise.

Flight: {flight_info['carrier']}{flight_info['flight_num']} from {flight_info['origin']} to {flight_info.get('dest', '?')}
Date/Time: {flight_info['dep_date']} at {flight_info['dep_time']}

Model Predictions:
- Cancellation probability: {prediction['cancel_probability']*100:.1f}%
- Delay probability (>=15 min): {prediction['delay_probability']*100:.1f}%

Origin weather: temp {origin_wx.get('temperature_2m','?')}F, wind {origin_wx.get('wind_speed_10m','?')}mph, gusts {origin_wx.get('wind_gusts_10m','?')}mph, visibility {origin_wx.get('visibility','?')}ft, precip {origin_wx.get('precipitation','?')}in
Dest weather: temp {dest_wx.get('temperature_2m','?')}F, wind {dest_wx.get('wind_speed_10m','?')}mph, gusts {dest_wx.get('wind_gusts_10m','?')}mph, visibility {dest_wx.get('visibility','?')}ft, precip {dest_wx.get('precipitation','?')}in

Top SHAP factors: {', '.join(f['display_name'] + ' (' + ('+' if f['value']>0 else '') + str(f['value']) + ')' for f in prediction.get('shap_factors', [])[:5])}

Write a brief, helpful analysis:"""

    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
    except Exception as e:
        print(f"Gemini API failed: {e}")
        return _template_analysis(prediction, flight_info, origin_wx, dest_wx)


def _template_analysis(prediction: dict, flight_info: dict, origin_wx: dict, dest_wx: dict) -> str:
    """Fallback template when Gemini API is unavailable."""
    cancel_pct = prediction["cancel_probability"] * 100
    delay_pct = prediction["delay_probability"] * 100

    parts = []

    if cancel_pct > 30:
        parts.append(f"This flight has an elevated cancellation risk of {cancel_pct:.0f}%.")
    elif cancel_pct > 10:
        parts.append(f"Cancellation risk is moderate at {cancel_pct:.0f}%.")
    else:
        parts.append(f"Cancellation risk is low at {cancel_pct:.0f}%.")

    if delay_pct > 50:
        parts.append(f"There is a {delay_pct:.0f}% chance of a significant delay (15+ minutes).")
    elif delay_pct > 25:
        parts.append(f"Moderate delay risk at {delay_pct:.0f}%.")
    else:
        parts.append(f"Low delay risk at {delay_pct:.0f}%. The flight is likely to arrive close to schedule.")

    # Weather commentary
    wind_o = origin_wx.get("wind_gusts_10m") or 0
    wind_d = dest_wx.get("wind_gusts_10m") or 0
    vis_o = origin_wx.get("visibility") or 99999
    vis_d = dest_wx.get("visibility") or 99999
    precip_o = (origin_wx.get("precipitation") or 0)
    precip_d = (dest_wx.get("precipitation") or 0)

    wx_notes = []
    if wind_o > 25 or wind_d > 25:
        wx_notes.append("high wind gusts")
    if vis_o < 5280 or vis_d < 5280:
        wx_notes.append("reduced visibility")
    if precip_o > 0.1 or precip_d > 0.1:
        wx_notes.append("precipitation")

    if wx_notes:
        parts.append(f"Key weather factors include {', '.join(wx_notes)}.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# VERCEL HANDLER
# ═══════════════════════════════════════════════════════════════════════

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            _load_models()

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            carrier = body["carrier"]
            flight_num = int(body["flight_num"])
            origin = body["origin"].upper()
            dest = body["dest"].upper()
            dep_date = body["dep_date"]
            dep_time = body["dep_time"]

            # Validate date is today or future (no past-flight predictions)
            from datetime import date as _date
            dep_date_obj = datetime.strptime(dep_date, "%Y-%m-%d").date()
            if dep_date_obj < _date.today():
                self._send_json(400, {"error": "Only today's flights or upcoming flights can be predicted. Past dates are not supported."})
                return

            # Validate airports exist in our coordinates
            coords = _models["airport_coords"]
            if origin not in coords:
                self._send_json(400, {"error": f"Unknown origin airport: {origin}"})
                return
            if dest not in coords:
                self._send_json(400, {"error": f"Unknown destination airport: {dest}"})
                return

            # Compute distance from airport coordinates (haversine)
            o = coords[origin]
            d = coords[dest]
            distance = round(haversine_miles(o["lat"], o["lon"], d["lat"], d["lon"]), 1)

            # Estimate flight elapsed time from distance
            elapsed = round(estimate_elapsed_minutes(distance))

            # Compute arrival datetime for weather fetch
            dep_dt = datetime.fromisoformat(f"{dep_date}T{dep_time}")
            arr_dt = dep_dt + timedelta(minutes=elapsed)
            arr_hour = arr_dt.hour
            arr_time_hhmm = arr_hour * 100 + arr_dt.minute

            # Fetch live weather forecasts from Open-Meteo
            origin_wx = fetch_weather(origin, dep_dt.isoformat())
            dest_wx = fetch_weather(dest, arr_dt.isoformat())

            # Build features
            features = build_feature_vector(
                carrier, flight_num, origin,
                dep_date, dep_time,
                dest, distance, elapsed, arr_time_hhmm,
                origin_wx, dest_wx,
            )

            # Predict
            prediction = predict(features)
            prediction["destination"] = dest
            prediction["distance"] = distance
            prediction["elapsed_time"] = elapsed

            # LLM synthesis
            flight_info = {
                "carrier": carrier, "flight_num": flight_num,
                "origin": origin, "dest": dest,
                "dep_date": dep_date, "dep_time": dep_time,
            }
            prediction["llm_analysis"] = llm_synthesize(prediction, flight_info, origin_wx, dest_wx)

            # Weather summary for frontend
            prediction["origin_weather"] = {
                "temp": origin_wx.get("temperature_2m"),
                "wind": origin_wx.get("wind_speed_10m"),
                "visibility": origin_wx.get("visibility"),
            }
            prediction["dest_weather"] = {
                "temp": dest_wx.get("temperature_2m"),
                "wind": dest_wx.get("wind_speed_10m"),
                "visibility": dest_wx.get("visibility"),
            }

            self._send_json(200, prediction)

        except KeyError as e:
            self._send_json(400, {"error": f"Missing field: {e}"})
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        self._send_json(200, {"status": "ok", "message": "POST flight data to this endpoint"})

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
