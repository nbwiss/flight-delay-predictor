"""
Flask wrapper for local development and Railway/Render deployment.
Serves the static frontend + the prediction API.

Usage:
    pip install flask
    python app.py

For Railway: just push, the Procfile handles it.
For local dev: python app.py -> open http://localhost:5000
"""

import os
import json
import math
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import joblib
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="public", static_url_path="")

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
V2_DIR = BASE_DIR / "models" / "v2"
V3_DIR = BASE_DIR / "models" / "v3"

# ─── Import inference logic from api/predict.py ──────────────────────
# We reuse all the functions but swap the handler for Flask routes.
import sys
sys.path.insert(0, str(BASE_DIR / "api"))
from predict import (
    _models, _load_models, fetch_weather, build_feature_vector,
    predict as run_prediction, llm_synthesize, WEATHER_VARS,
    haversine_miles, estimate_elapsed_minutes,
)


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def api_predict():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    try:
        _load_models()
        body = request.get_json()

        carrier = body["carrier"]
        flight_num = int(body["flight_num"])
        origin = body["origin"].upper()
        dest = body["dest"].upper()
        dep_date = body["dep_date"]
        dep_time = body["dep_time"]

        # Validate airports
        coords = _models["airport_coords"]
        if origin not in coords:
            return jsonify({"error": f"Unknown origin airport: {origin}"}), 400
        if dest not in coords:
            return jsonify({"error": f"Unknown destination airport: {dest}"}), 400

        # Compute distance and elapsed time from coordinates
        o = coords[origin]
        d = coords[dest]
        distance = round(haversine_miles(o["lat"], o["lon"], d["lat"], d["lon"]), 1)
        elapsed = round(estimate_elapsed_minutes(distance))

        dep_dt = datetime.fromisoformat(f"{dep_date}T{dep_time}")
        arr_dt = dep_dt + timedelta(minutes=elapsed)
        arr_time_hhmm = arr_dt.hour * 100 + arr_dt.minute

        origin_wx = fetch_weather(origin, dep_dt.isoformat())
        dest_wx = fetch_weather(dest, arr_dt.isoformat())

        features = build_feature_vector(
            carrier, flight_num, origin,
            dep_date, dep_time,
            dest, distance, elapsed, arr_time_hhmm,
            origin_wx, dest_wx,
        )

        prediction = run_prediction(features)
        prediction["destination"] = dest
        prediction["distance"] = distance
        prediction["elapsed_time"] = elapsed

        flight_info = {
            "carrier": carrier, "flight_num": flight_num,
            "origin": origin, "dest": dest,
            "dep_date": dep_date, "dep_time": dep_time,
        }
        prediction["llm_analysis"] = llm_synthesize(prediction, flight_info, origin_wx, dest_wx)

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

        return jsonify(prediction)

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flight Delay Predictor on port {port}")
    print(f"Open http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
