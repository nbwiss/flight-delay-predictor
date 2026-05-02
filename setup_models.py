"""
Setup script: copies model artifacts from your Modelling Completed folder
into the app's models/ directory.

Run this ONCE before deploying or running locally:
    python setup_models.py --source "path/to/Modelling Completed"

If your folder structure is:
    Modelling Completed/
        v2model/
        v3model/

Then run:
    python setup_models.py --source "../Modelling Completed"
"""

import shutil
import argparse
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

V2_FILES = [
    "model_a_cancellation.joblib",
    "model_b_delay.joblib",
    "model_c_minutes.joblib",  # not used at inference, but kept for reference
    "scaler.joblib",
    "ord_encoder.joblib",
    "medians.joblib",
    "num_cols.joblib",
    "cat_cols.joblib",
    "feature_names.joblib",
]

V3_FILES = [
    "model_c_minutes.joblib",
    "target_encoder_c.joblib",
    "scaler.joblib",
    "feature_names.joblib",
]


def main():
    parser = argparse.ArgumentParser(description="Copy model artifacts into app directory")
    parser.add_argument("--source", required=True, help="Path to 'Modelling Completed' folder")
    args = parser.parse_args()

    src = Path(args.source).resolve()
    v2_src = src / "v2model"
    v3_src = src / "v3model"

    if not v2_src.exists():
        print(f"ERROR: {v2_src} not found")
        return
    if not v3_src.exists():
        print(f"ERROR: {v3_src} not found")
        return

    # Copy V2
    v2_dst = APP_DIR / "models" / "v2"
    v2_dst.mkdir(parents=True, exist_ok=True)
    for f in V2_FILES:
        src_f = v2_src / f
        if src_f.exists():
            shutil.copy2(src_f, v2_dst / f)
            print(f"  V2: {f} ({src_f.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  V2: {f} NOT FOUND (skipping)")

    # Copy V3
    v3_dst = APP_DIR / "models" / "v3"
    v3_dst.mkdir(parents=True, exist_ok=True)
    for f in V3_FILES:
        src_f = v3_src / f
        if src_f.exists():
            shutil.copy2(src_f, v3_dst / f)
            print(f"  V3: {f} ({src_f.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  V3: {f} NOT FOUND (skipping)")

    total_size = sum(
        f.stat().st_size for f in (APP_DIR / "models").rglob("*.joblib")
    ) / 1024 / 1024
    print(f"\nTotal model payload: {total_size:.1f} MB")
    print("Done! Run `python app.py` to start locally.")


if __name__ == "__main__":
    main()
