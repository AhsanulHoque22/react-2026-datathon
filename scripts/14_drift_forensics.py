"""Forensic audit of the June vs July regime shift in train.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import TRAIN_CSV, TIME_COL, LABEL_COL

print("Loading train.csv...")
df = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])
df["month"] = df[TIME_COL].dt.month

# Filter specifically to June (month 6) and July (month 7)
sub = df[df["month"].isin([6, 7])].copy()

june = sub[sub["month"] == 6]
july = sub[sub["month"] == 7]

print(f"Total rows: {len(df):,}")
print(f"June: rows={len(june):,}, fraud_count={int(june[LABEL_COL].sum()):,}, fraud_rate={june[LABEL_COL].mean()*100:.2f}%")
print(f"July: rows={len(july):,}, fraud_count={int(july[LABEL_COL].sum()):,}, fraud_rate={july[LABEL_COL].mean()*100:.2f}%")

print("\n" + "="*70)
print("1. FRAUD RATE BY CATEGORICAL ATTRIBUTES (JUNE vs JULY)")
print("="*70)

cat_cols = ["payment_method", "device_type", "transaction_type", "merchant_category", "location"]
for col in cat_cols:
    pt = sub.groupby(["month", col])[LABEL_COL].agg(["count", "mean"]).unstack(level=0)
    pt.columns = ["June_Count", "July_Count", "June_FraudRate%", "July_FraudRate%"]
    pt["June_FraudRate%"] = (pt["June_FraudRate%"] * 100).round(2)
    pt["July_FraudRate%"] = (pt["July_FraudRate%"] * 100).round(2)
    pt["Rate_Delta%"] = (pt["July_FraudRate%"] - pt["June_FraudRate%"]).round(2)
    pt = pt.sort_values("July_FraudRate%", ascending=False)
    print(f"\n--- Attribute: {col} ---")
    print(pt.head(10).to_string())

print("\n" + "="*70)
print("2. TRANSACTION AMOUNT DISTRIBUTION (FRAUD CASES ONLY)")
print("="*70)

june_fraud = june[june[LABEL_COL] == 1]["amount_bdt"]
july_fraud = july[july[LABEL_COL] == 1]["amount_bdt"]

print("June Fraud Amounts (quantiles):")
print(june_fraud.quantile([0.05, 0.25, 0.50, 0.75, 0.90, 0.99]).round(2))

print("\nJuly Fraud Amounts (quantiles):")
print(july_fraud.quantile([0.05, 0.25, 0.50, 0.75, 0.90, 0.99]).round(2))

print("\n" + "="*70)
print("3. AMOUNT BUCKETS FRAUD PREVALENCE (JUNE vs JULY)")
print("="*70)

bins = [0, 100, 500, 1000, 3000, 5000, 10000, 20000, np.inf]
labels = ["<100", "100-500", "500-1k", "1k-3k", "3k-5k", "5k-10k", "10k-20k", "20k+"]
sub["amt_bucket"] = pd.cut(sub["amount_bdt"], bins=bins, labels=labels)

amt_pt = sub.groupby(["month", "amt_bucket"], observed=False)[LABEL_COL].agg(["count", "mean"]).unstack(level=0)
amt_pt.columns = ["June_Count", "July_Count", "June_FraudRate%", "July_FraudRate%"]
amt_pt["June_FraudRate%"] = (amt_pt["June_FraudRate%"] * 100).round(2)
amt_pt["July_FraudRate%"] = (amt_pt["July_FraudRate%"] * 100).round(2)
amt_pt["Rate_Delta%"] = (amt_pt["July_FraudRate%"] - amt_pt["June_FraudRate%"]).round(2)
print(amt_pt.to_string())

print("\n" + "="*70)
print("4. TIME OF DAY (HOUR) SHIFT (FRAUD CASES ONLY)")
print("="*70)

sub["hour"] = sub[TIME_COL].dt.hour
sub["is_night"] = (sub["hour"] >= 0) & (sub["hour"] < 6)
night_pt = sub.groupby(["month", "is_night"])[LABEL_COL].agg(["count", "mean"]).unstack(level=0)
night_pt.columns = ["June_Count", "July_Count", "June_FraudRate%", "July_FraudRate%"]
night_pt["June_FraudRate%"] = (night_pt["June_FraudRate%"] * 100).round(2)
night_pt["July_FraudRate%"] = (night_pt["July_FraudRate%"] * 100).round(2)
print(night_pt.to_string())
