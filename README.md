# Flight Delay Predictor

A deployed web application that predicts cancellation risk and delay probability for upcoming domestic US flights using machine learning and live weather data.

**Live app:** https://web-production-7b3d.up.railway.app

---

## What It Does

A user enters their airline, origin airport, destination airport, and scheduled departure date and time. The app:

1. Fetches live hourly weather forecasts for both airports from the [Open-Meteo Forecast API](https://open-meteo.com/)
2. Assembles a 61-feature vector (flight metadata + weather at origin and destination)
3. Runs two XGBoost classifiers in sequence — cancellation risk, then delay probability
4. Computes SHAP values to explain which factors drove the prediction
5. Passes results to Gemini 2.5 Flash to generate a plain-English summary

Results are displayed as a risk badge, probability tiles, a SHAP bar chart, and an LLM-generated explanation.

---

## Models

Three XGBoost models were trained on ~7.08 million domestic US flights from 2024 (BTS On-Time Performance data) with matched hourly weather observations from the Open-Meteo Historical Archive API. The deployed app uses a V2/V3 hybrid:

| Model | Task | Version Deployed | Test AUC |
|-------|------|-----------------|----------|
| Model A | Cancellation classifier | V2 (ordinal encoding) | 0.939 |
| Model B | Delay classifier (≥15 min late) | V2 (ordinal encoding) | 0.753 |
| Model C | Delay duration regressor | Removed — poor tail prediction | — |

Training used a time-based split: Jan–Oct 2024 train, Nov 2024 validation, Dec 2024 test.

---

## Repo Structure

```
flight-delay-app/
├── public/index.html          # Frontend (single-page, inline CSS + JS)
├── api/predict.py             # Inference engine — feature assembly, model inference, SHAP, LLM
├── app.py                     # Flask server (Railway / gunicorn)
├── data/
│   ├── airport_coords.json    # 370 airports with lat/lon
│   ├── airports.json          # 344 airport IATA codes
│   └── carriers.json          # 15 airline codes
├── models/
│   ├── v2/                    # Models A + B artifacts (deployed)
│   └── v3/                    # Model C artifacts (not deployed)
├── training/                  # Dataset construction + model training code
│   ├── flight_delay_dataset_builder.py   # Fetches BTS + weather, builds parquet
│   ├── flight_delay_pipeline_v2.py       # V2 training (ordinal encoding)
│   ├── flight_delay_pipeline_v2.ipynb    # V2 notebook
│   ├── flight_delay_pipeline_v3.py       # V3 training (target encoding, log transform)
│   ├── flight_delay_pipeline_v3.ipynb    # V3 notebook
│   ├── find_missing_airports.py          # Utility to check airport coverage
│   └── data/README.md                    # How to reconstruct the training dataset
├── requirements.txt
├── Procfile                   # Railway: gunicorn app:app
└── runtime.txt                # Python 3.11.9
```

---

## Data Sources

- **Flight data:** [BTS On-Time Reporting Carrier Performance](https://www.transtats.bts.gov/) — all 12 monthly archives for 2024, ~7.08M records
- **Weather data:** [Open-Meteo Historical Archive API](https://archive-api.open-meteo.com/v1/archive) — free, no API key, ERA5 reanalysis at hourly resolution for all 348 airports

The final training dataset (`flight_delay_dataset_2024.parquet`, ~358MB) is not committed to this repo due to size. See `training/data/README.md` for reconstruction instructions.

---

## Deployment

Hosted on [Railway](https://railway.app) via GitHub integration. The app runs as a gunicorn Flask server. Pushes to `main` trigger automatic redeployment.

Environment variables required in Railway:
- `GEMINI_API_KEY` — Google AI API key for the LLM summary (app falls back to template-based output if unset)

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML models | XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| LLM | Gemini 2.5 Flash (Google AI API) |
| Weather API | Open-Meteo (free, no key) |
| Backend | Python 3.11, Flask, gunicorn |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Deployment | Railway |
