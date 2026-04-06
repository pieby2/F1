"""
Offline Retraining Utility.

Use this script locally (where RAM is free) to train models up to a specific season/round.
It will package the trained models into a `models_upload.zip` file which you can upload via the web dashboard.
"""
import argparse
import os
import shutil
import zipfile
from pathlib import Path

import joblib

from src.inference.retrain_service import WalkForwardPredictor


def main():
    parser = argparse.ArgumentParser(description="Offline Retrain and Package Models")
    parser.add_argument("--season", type=int, required=True, help="Target season (models will be trained on data BEFORE this season/round)")
    parser.add_argument("--round", type=int, required=True, help="Target round")
    parser.add_argument("--out", type=str, default="models_upload.zip", help="Output zip filename")
    args = parser.parse_args()

    print(f"🚀 Starting offline retraining for {args.season} R{args.round:02d} cutoff...")
    
    # Use WalkForwardPredictor logic to get the models
    predictor = WalkForwardPredictor()
    dataset = predictor._ensure_dataset()

    train_mask = (
        (dataset["year"] < args.season)
        | ((dataset["year"] == args.season) & (dataset["round"] < args.round))
    )

    if train_mask.sum() < 20:
        print("❌ Error: Insufficient training data. Need to ingest data first.")
        return

    print(f"📊 Training on {train_mask.sum()} historical rows...")
    models = predictor._train_all_models(dataset, train_mask)

    # Temporary directory to hold joblib files before zipping
    tmp_dir = Path("_tmp_models_export")
    tmp_dir.mkdir(exist_ok=True)

    print("💾 Saving models to disk...")
    for model_name, info in models.items():
        joblib.dump(info, tmp_dir / f"{model_name}_ensemble.joblib")

    print(f"📦 Packaging into {args.out}...")
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in tmp_dir.glob("*.joblib"):
            zipf.write(file, arcname=file.name)

    # Cleanup temp directory
    shutil.rmtree(tmp_dir)

    print(f"✅ Success! You can now upload '{args.out}' to the web dashboard.")

if __name__ == "__main__":
    main()
