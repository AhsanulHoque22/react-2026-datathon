"""Which features still separate fraud in the July regime?

The signal-decay audit showed every engineered signal weakens in July, but
the base rate holds at ~1.56% -- so July fraud exists, it just is not where
our features are looking. This ranks every numeric feature by its UNIVARIATE
average precision, computed separately for June and July, to find what still
carries signal in the regime the test set actually resembles.

No model training -- just per-feature AP, so it runs in seconds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import PROCESSED_DIR, LABEL_COL, TIME_COL
from src.model import get_feature_columns

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
feature_cols, cat_cols = get_feature_columns(df)
numeric_cols = [c for c in feature_cols if c not in cat_cols]

lab = df[~df["is_test"]]
june = lab[(lab[TIME_COL] >= "2026-06-01") & (lab[TIME_COL] < "2026-07-01")]
july = lab[lab[TIME_COL] >= "2026-07-01"]
print(f"June rows={len(june)} frauds={int(june[LABEL_COL].sum())} | "
      f"July rows={len(july)} frauds={int(july[LABEL_COL].sum())}\n")


def univariate_ap(frame, col):
    """AP of the feature as a score, and of its negation; take the better,
    so a feature predictive in either direction is credited."""
    y = frame[LABEL_COL].astype(int).values
    x = frame[col].values.astype("float64")
    if np.all(np.isnan(x)):
        return np.nan, np.nan
    filled = np.nan_to_num(x, nan=np.nanmedian(x))
    if np.std(filled) == 0:
        return np.nan, np.nan
    ap_pos = average_precision_score(y, filled)
    ap_neg = average_precision_score(y, -filled)
    return max(ap_pos, ap_neg), (1 if ap_pos >= ap_neg else -1)


rows = []
for c in numeric_cols:
    ap_jun, dir_jun = univariate_ap(june, c)
    ap_jul, dir_jul = univariate_ap(july, c)
    rows.append(dict(feature=c, ap_june=ap_jun, ap_july=ap_jul,
                     ratio=(ap_jul / ap_jun if ap_jun and ap_jun > 0 else np.nan),
                     dir_flip=(dir_jun != dir_jul)))

res = pd.DataFrame(rows).dropna(subset=["ap_july"]).sort_values("ap_july", ascending=False)
base_jun = june[LABEL_COL].mean()
base_jul = july[LABEL_COL].mean()
print(f"Base AP (random ranker): June={base_jun:.4f}  July={base_jul:.4f}\n")

pd.set_option("display.width", 200)
print("=== TOP 20 features by JULY univariate AP (what still works) ===")
print(res.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== BIGGEST DECAYERS (strong in June, dead in July) ===")
decay = res[res["ap_june"] > 3 * base_jun].sort_values("ratio")
print(decay.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== FEATURES THAT HOLD UP BEST (july/june ratio, among useful ones) ===")
holds = res[(res["ap_june"] > 2 * base_jun) & (res["ap_july"] > 2 * base_jul)].sort_values("ratio", ascending=False)
print(holds.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

res.to_csv(PROCESSED_DIR / "univariate_ap_june_july.csv", index=False)
print(f"\nSaved {PROCESSED_DIR / 'univariate_ap_june_july.csv'}")
