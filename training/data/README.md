# Training Data

The training dataset is not committed to this repo due to size (~358MB compressed parquet, ~2.3GB CSV).

## Reconstructing the Dataset

Run `flight_delay_dataset_builder.py` from the `training/` directory. It will:

1. Download all 12 monthly BTS On-Time Performance archives for 2024 from `transtats.bts.gov`
2. Fetch hourly weather data for all ~348 airports from the Open-Meteo Historical Archive API (free, no API key)
3. Join flight records to weather observations at origin (departure time) and destination (arrival time)
4. Write the final dataset to `data/output/flight_delay_dataset_2024.parquet`

**Expected runtime:** several hours (weather API fetching is the bottleneck — ~348 API requests, one per airport)

**Output dimensions:** ~7.08 million rows, 74 columns

## Raw Data Sources

| Source | URL | Notes |
|--------|-----|-------|
| BTS On-Time Performance | https://www.transtats.bts.gov | Pre-zipped monthly CSV archives, free |
| Open-Meteo Historical Archive | https://archive-api.open-meteo.com/v1/archive | ERA5 reanalysis, free, no API key |

## Folder Structure (after building)

```
data/
├── bts_raw/                          # Monthly BTS CSV files (1.3GB)
│   ├── flights_2024_01.csv
│   └── ... (12 files)
├── weather_openmeteo/                # Per-airport hourly weather JSON cache (71MB)
├── output/
│   ├── flight_delay_dataset_2024.parquet   # Final training dataset (358MB)
│   └── flight_delay_dataset_2024.csv       # CSV version (2.3GB)
└── logs/                             # Build logs
```
