"""Run this from your fart directory: python find_missing_airports.py"""
import pandas as pd
import os

# Load flight data
flights = pd.read_parquet("data/bts_raw/flights_2024.parquet")
bts_airports = set(flights["Origin"].unique()) | set(flights["Dest"].unique())

# Check which ones have empty or missing weather files
weather_dir = "data/weather_openmeteo"
missing = []
for iata in sorted(bts_airports):
    f = os.path.join(weather_dir, f"{iata}_2024.parquet")
    if not os.path.exists(f):
        missing.append((iata, "no file"))
    else:
        df = pd.read_parquet(f)
        if len(df) == 0:
            missing.append((iata, "empty (skipped)"))

print(f"\nBTS airports: {len(bts_airports)}")
print(f"Missing/empty weather: {len(missing)}\n")
for iata, reason in missing:
    # Count how many flights involve this airport
    n = len(flights[(flights["Origin"] == iata) | (flights["Dest"] == iata)])
    print(f"  {iata:>4}  ({reason:>15})  -  {n:>7,} flights")
