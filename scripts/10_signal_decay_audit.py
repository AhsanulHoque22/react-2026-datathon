"""Why does fold 3 (Jul 2-15) collapse to ~0.50 when folds 0-2 score ~0.78,
and why does the leaderboard match fold 3 rather than the mean?

Hypothesis: a regime change near the end of train. If individual signals
lose their predictive lift in the final weeks, the model is largely trained
on a regime that no longer applies -- which would explain why recency
weighting didn't help (the new regime is barely represented in train at all)
and why absolute-threshold features transfer worse than relative ones.

Pure pandas, no model training. Measures each signal's fraud lift per month.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, LABEL_COL, TIME_COL

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
lab = df[~df["is_test"]].copy()
lab["month"] = lab[TIME_COL].dt.to_period("M")

base_by_month = lab.groupby("month")[LABEL_COL].agg(["mean", "size", "sum"])
print("=== Base fraud rate by month ===")
print(base_by_month.rename(columns={"mean": "fraud_rate", "size": "rows", "sum": "frauds"}))

# Each signal: a boolean rule. Lift = P(fraud|signal) / P(fraud) within that month.
signals = {
    "night (hour<5)": lab["hour_of_day"] < 5,
    "amount > 3000": lab["amount_bdt"] > 3000,
    "amount > 10000": lab["amount_bdt"] > 10000,
    "new device for customer": lab["is_new_device_for_customer"] == 1,
    "new merchant for customer": lab["is_new_merchant_for_customer"] == 1,
    "new location for customer": lab["is_new_location_for_customer"] == 1,
    "burst (<300s since last)": lab["cust_seconds_since_last"] < 300,
    "cust robust_z > 5": lab["cust_amt_robust_z"] > 5,
    "cust log_amt_z > 3": lab["cust_log_amt_zscore"] > 3,
    "device shared >3 customers": lab["device_id_nunique_customer_for_device_prior"] > 3,
}

print("\n=== Fraud rate WITHIN each signal, by month (raw rate, not lift) ===")
rows = []
for name, mask in signals.items():
    sub = lab[mask]
    if len(sub) == 0:
        continue
    by_month = sub.groupby("month")[LABEL_COL].mean()
    rows.append(pd.Series(by_month, name=name))
sig_df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print((sig_df * 100).round(2).to_string())

print("\n=== Lift (signal fraud rate / month base rate) ===")
lift = sig_df.div(base_by_month["mean"], axis=1)
print(lift.round(2).to_string())

print("\n=== Coverage: what % of that month's rows fire each signal ===")
cov_rows = []
for name, mask in signals.items():
    cov = lab.assign(_m=mask).groupby("month")["_m"].mean()
    cov_rows.append(pd.Series(cov, name=name))
print((pd.DataFrame(cov_rows) * 100).round(2).to_string())

print("\n=== Recall: what % of that month's frauds does each signal catch ===")
rec_rows = []
for name, mask in signals.items():
    fr = lab[lab[LABEL_COL] == 1]
    rec = fr.assign(_m=mask.loc[fr.index]).groupby("month")["_m"].mean()
    rec_rows.append(pd.Series(rec, name=name))
print((pd.DataFrame(rec_rows) * 100).round(2).to_string())
