"""Insurance baseline: plain LightGBM, no feature engineering, no raw IDs.
Uses only directly-given, non-ID raw fields. Submitted first so there's a
scored fallback on the board before deeper feature work risks anything.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import SEED, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV, TRAIN_CSV, TEST_CSV, LABEL_COL

t0 = time.time()
train = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
test = pd.read_csv(TEST_CSV, parse_dates=["timestamp"])

cat_cols = ["merchant_category", "device_type", "location", "payment_method", "transaction_type"]
num_cols = ["amount_bdt", "account_age_days"]
# no raw ID columns (customer_id/merchant_id/device_id/transaction_id) -- banned
feature_cols = num_cols + cat_cols

X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_test[c] = X_test[c].astype("category")
y_train = train[LABEL_COL].astype(int)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / n_pos

params = dict(
    objective="binary", metric="average_precision",
    scale_pos_weight=scale_pos_weight, seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1, learning_rate=0.05,
    num_leaves=31, min_data_in_leaf=100,
)
train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
booster = lgb.train(params, train_set, num_boost_round=300)

preds = booster.predict(X_test)
assert np.all((preds >= 0) & (preds <= 1)), "predictions out of [0,1] range"

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test["transaction_id"], "fraud": preds})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

assert list(sub["transaction_id"]) == list(sample_sub["transaction_id"]), "row order mismatch vs sample_submission"
assert sub["fraud"].between(0, 1).all()
assert not sub["fraud"].isna().any()

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
out_path = SUBMISSIONS_DIR / "submission_baseline.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path}  shape={sub.shape}  ({time.time()-t0:.1f}s)")
print(sub.head())
