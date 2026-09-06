import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, ID_COLS, LABEL_COL
from src.data import load_raw, build_combined_frame, assert_frame_sane
from src.features import build_features, leakage_assertions

t0 = time.time()
print("Loading raw data...")
train, test = load_raw()
print(f"train={train.shape} test={test.shape}  ({time.time()-t0:.1f}s)")

print("Building time-sorted combined frame...")
df = build_combined_frame(train, test)
assert_frame_sane(df)
print(f"combined={df.shape}  ({time.time()-t0:.1f}s)")

print("Building features...")
df = build_features(df)
print(f"featurized={df.shape}  ({time.time()-t0:.1f}s)")

print("Running leakage assertions...")
leakage_assertions(df)
print("OK.")

# pre-submission-relevant assertion: no raw ID / label leakage into what will become
# the model feature matrix later (checked again at train time, but cheap to check now)
for c in ID_COLS:
    assert c in df.columns, f"expected raw id column {c} present as join key"
print(f"Raw ID columns present as join keys only (not yet excluded from df): {ID_COLS}")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
out_path = PROCESSED_DIR / "features.pkl"
df.to_pickle(out_path)
print(f"Saved {out_path}  ({time.time()-t0:.1f}s total)")
print(f"Final shape: {df.shape}, columns: {len(df.columns)}")
