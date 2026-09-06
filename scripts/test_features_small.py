"""Quick correctness check of src/features.py on a tiny synthetic frame."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features import build_features, leakage_assertions

rows = [
    # customer, merchant, device, timestamp, amount
    ("C1", "M1", "D1", "2026-01-01 00:00:00", 100.0),
    ("C1", "M1", "D1", "2026-01-01 01:00:00", 200.0),
    ("C1", "M2", "D1", "2026-01-01 02:00:00", 300.0),
    ("C2", "M1", "D2", "2026-01-01 00:30:00", 50.0),
    ("C1", "M1", "D2", "2026-01-01 03:00:00", 9999.0),  # new device for C1, big amount
]
df = pd.DataFrame(rows, columns=["customer_id", "merchant_id", "device_id", TIME_COL := "timestamp", "amount_bdt"])
df["transaction_id"] = [f"T{i}" for i in range(len(df))]
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df["merchant_category"] = "grocery"
df["device_type"] = "mobile_app"
df["location"] = "LOC_1"
df["payment_method"] = "mobile_wallet"
df["transaction_type"] = "purchase"
df = df.sort_values([TIME_COL, "transaction_id"]).reset_index(drop=True)

out = build_features(df)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
cols = [
    "transaction_id", "customer_id", "device_id", "amount_bdt",
    "cust_history_count", "cust_amt_mean_prior", "cust_amt_std_prior", "cust_amt_zscore",
    "cust_seconds_since_last", "is_new_device_for_customer", "customer_id_nunique_device_for_customer_prior",
    "cust_cnt_1h", "cust_cnt_24h",
]
print(out[cols].to_string())

leakage_assertions(out)
print("\nAll assertions passed.")

# Manual spot-check: row 4 (C1's 4th txn, new device D2) should have
# cust_history_count == 3, mean of [100,200,300] == 200, and is_new_device == 1
row = out.iloc[4]
assert row["cust_history_count"] == 3
assert abs(row["cust_amt_mean_prior"] - 200.0) < 1e-9
assert row["is_new_device_for_customer"] == 1
print("Manual spot-checks passed.")
