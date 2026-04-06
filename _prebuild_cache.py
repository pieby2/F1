"""Pre-build the walk-forward dataset cache."""
import time

from src.inference.retrain_service import WalkForwardPredictor

t0 = time.time()
wf = WalkForwardPredictor()
ds = wf._ensure_dataset()
print(f"Dataset shape: {ds.shape}")
print(f"Years: {sorted(ds['year'].dropna().unique())}")
cols = [c for c in ds.columns if c.startswith("wx_") or c.startswith("circuit_") or c.startswith("sprint")]
print(f"New feature columns: {cols[:20]}")
has_targets = "points_target" in ds.columns
print(f"Has race targets: {has_targets}")
if has_targets:
    print(f"  points_target non-null: {ds['points_target'].notna().sum()}")
    print(f"  dnf_target non-null: {ds['dnf_target'].notna().sum()}")
print(f"Total time: {time.time()-t0:.1f}s")
