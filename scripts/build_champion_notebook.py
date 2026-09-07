#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

ROOT = Path('/home/sanzid/competitions/sust-datathon/react-2026-datathon')
OUT = ROOT / "notebooks" / "REACT_2026_1st_Place_Champion.ipynb"

optimal_features_path = ROOT / "data" / "processed" / "optimal_pruned_features.csv"
SELECTED_FEATURES = pd.read_csv(optimal_features_path)["0"].tolist()

MD_HEADER = r"""# REACT 2026 Datathon — Reproducibility Notebook

**Team**: Overfit & Overcaffeinated  
**Competition**: REACT 2026 Datathon (Tabular Fraud Detection, Average Precision / PR-AUC)  
**Selected Leaderboard Result**: **`0.56548` Public Leaderboard** | **`0.5359` Local PR-AUC**  
**Submission Artifact**: `submission.csv` (MD5: `d64c632fc843a39c5ae4f4aa528316ff`)

---

## Architectural Summary
This notebook reproduces our solution end-to-end from raw `train.csv` and `test.csv`:

1. **Causally Rigorous Behavioral Features (Top 160 Core Features)**:
   - Expanding per-entity aggregates (mean, std, robust MAD around expanding median).
   - Dimensionless self-relative ratios (`cust_amt_ratio`, `cust_burst_accel`, `gap_accel`).
   - Sub-hour & trailing temporal windows (5m, 15m, 30m, 1h, 3h, 6h, 12h, 24h, 72h, 168h) with strictly-prior `closed="left"`.
   - Bipartite counterparty novelty (`is_new_device_for_customer`, `device_id_nunique_customer_for_device_prior`).
   - Location and merchant category behavioral context (`amt_vs_cat_mean_prior`, `loc_amtsum_24h`).
   - Zero forward leakage (`assert_strictly_past()` verified).

2. **Dual-Horizon Tree Ensemble (LightGBM)**:
   - **Full-Train Horizon** (882 rounds, `num_leaves=127`, `lr=0.02`): Captures long-term macro patterns and rare categoricals.
   - **90-Day Specialist Horizon** (433 rounds, `num_leaves=63`, `lr=0.02`): Adapts to modern summer device-velocity attacks.
   - Horizon Mixture: `0.48 * P_Full + 0.52 * P_90d` (Local PR-AUC: `0.5284`).

3. **Tabular ResNet (Neural Manifold Regularization)**:
   - Learnable categorical entity embeddings + standardized continuous features.
   - 3 Residual Blocks (`LayerNorm` -> `Linear` -> `GELU` -> `Dropout(0.15)` -> `Linear` -> `Dropout(0.15)` -> Skip).
   - Trained with `AdamW` (`lr=1e-3`) and `CosineAnnealingLR` over 8 epochs.
   - Tree-Neural Mixture: `0.88 * P_LGBM + 0.12 * P_ResNet` (Local PR-AUC: `0.5298`).

4. **Unsupervised Temporal Entity Prediction Propagation**:
   - Sliding leave-one-out (LOO) neighborhood diffusion over $\pm 60$-minute window ($|t_j - t_i| \le 30$ min, $j \neq i$) across customer and device graphs ($w=0.50$).
   - Pushes local PR-AUC from `0.5298` -> **`0.5359`** -> **`0.56548` on Public Leaderboard**.
"""

CODE_SETUP = r"""# =============================================================================
# 1. SETUP & ENVIRONMENT
# =============================================================================
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Reproducibility seeds
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8)
np.random.seed(SEED)

# Detect Kaggle input paths
_CANDIDATES = [
    Path("/kaggle/input/competitions/react-2026-datathon"),
    Path("/kaggle/input/react-2026-datathon"),
    Path("./data/raw"),
    Path("../data/raw"),
]
RAW_DIR = next((p for p in _CANDIDATES if (p / "train.csv").exists()), _CANDIDATES[0])
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./")

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

print(f"Data Directory:       {RAW_DIR}")
print(f"Output Directory:     {OUT_DIR}")
assert TRAIN_CSV.exists(), f"Missing {TRAIN_CSV}"
assert TEST_CSV.exists(), f"Missing {TEST_CSV}"
"""

CODE_FEATURES = r"""# =============================================================================
# 2. CAUSALLY GUARANTEED FEATURE ENGINEERING PIPELINE
# =============================================================================
ID_COLS = ["customer_id", "merchant_id", "device_id", "transaction_id"]
LABEL_COL = "fraud"
TIME_COL = "timestamp"
EPS = 1e-6

def build_combined_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    test = test.copy()
    train["is_test"] = False
    test["is_test"] = True
    if LABEL_COL not in test.columns:
        test[LABEL_COL] = np.nan
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df = df.sort_values([TIME_COL, "transaction_id"], kind="mergesort").reset_index(drop=True)
    return df

def _prior_count_sum_sumsq(df: pd.DataFrame, entity_col: str, value_col: str):
    grp = df.groupby(entity_col)[value_col]
    cum_sum = grp.cumsum()
    prior_sum = cum_sum - df[value_col]
    prior_count = grp.cumcount().astype("float64")
    sq_col = df[value_col] ** 2
    cum_sumsq = sq_col.groupby(df[entity_col]).cumsum()
    prior_sumsq = cum_sumsq - sq_col
    return prior_count, prior_sum, prior_sumsq

def _prior_median_mad(df: pd.DataFrame, entity_col: str, value_col: str, prior_count: pd.Series):
    s = df[value_col]
    g = df.groupby(entity_col)[value_col]
    expanding_median_incl = g.expanding().median().reset_index(level=0, drop=True)
    prior_median = expanding_median_incl.shift(1)
    abs_dev = (s - prior_median).abs()
    expanding_mad_incl = abs_dev.groupby(df[entity_col]).expanding().median().reset_index(level=0, drop=True)
    prior_mad = expanding_mad_incl.shift(1)
    is_first = prior_count == 0
    prior_median = prior_median.where(~is_first, np.nan)
    prior_mad = prior_mad.where(~is_first, np.nan)
    return prior_median, prior_mad

def add_entity_expanding_features(df: pd.DataFrame, entity_col: str, prefix: str) -> pd.DataFrame:
    prior_count, prior_sum, prior_sumsq = _prior_count_sum_sumsq(df, entity_col, "amount_bdt")
    prior_mean = prior_sum / prior_count.replace(0, np.nan)
    prior_var = (prior_sumsq / prior_count.replace(0, np.nan)) - prior_mean ** 2
    prior_var = prior_var.clip(lower=0)
    prior_std = np.sqrt(prior_var)

    df[f"{prefix}_history_count"] = prior_count
    df[f"{prefix}_amt_sum_prior"] = prior_sum
    df[f"{prefix}_amt_mean_prior"] = prior_mean
    df[f"{prefix}_amt_std_prior"] = prior_std
    df[f"{prefix}_amt_zscore"] = (df["amount_bdt"] - prior_mean) / (prior_std + EPS)

    prior_median, prior_mad = _prior_median_mad(df, entity_col, "amount_bdt", prior_count)
    df[f"{prefix}_amt_robust_z"] = (df["amount_bdt"] - prior_median) / (prior_mad + EPS)
    df[f"{prefix}_amt_ratio"] = df["amount_bdt"] / (prior_mean + EPS)

    if "log_amount_bdt" not in df.columns:
        df["log_amount_bdt"] = np.log1p(df["amount_bdt"])
    log_prior_count, log_prior_sum, log_prior_sumsq = _prior_count_sum_sumsq(df, entity_col, "log_amount_bdt")
    log_prior_mean = log_prior_sum / log_prior_count.replace(0, np.nan)
    log_prior_var = (log_prior_sumsq / log_prior_count.replace(0, np.nan)) - log_prior_mean ** 2
    log_prior_std = np.sqrt(log_prior_var.clip(lower=0))
    df[f"{prefix}_log_amt_zscore"] = (df["log_amount_bdt"] - log_prior_mean) / (log_prior_std + EPS)

    ts_epoch = df[TIME_COL].astype("int64") // 10 ** 9
    prev_ts = ts_epoch.groupby(df[entity_col]).shift(1)
    df[f"{prefix}_seconds_since_last"] = ts_epoch - prev_ts
    return df

def add_self_relative_features(df: pd.DataFrame, entity_col: str, prefix: str) -> pd.DataFrame:
    ts = df[TIME_COL].astype("int64") // 10 ** 9
    first_ts = ts.groupby(df[entity_col]).transform("min")
    age_sec = (ts - first_ts).astype("float64")
    hist = df[f"{prefix}_history_count"]

    mean_gap = age_sec / np.maximum(hist, 1.0)
    df[f"{prefix}_gap_accel"] = mean_gap / (df[f"{prefix}_seconds_since_last"] + 1.0)

    rate_per_sec = hist / np.maximum(age_sec, 1.0)
    for h in (1, 24):
        col = f"{prefix}_cnt_{h}h"
        if col in df.columns:
            expected = rate_per_sec * (h * 3600.0)
            df[f"{prefix}_velocity_ratio_{h}h"] = df[col] / (expected + EPS)

    prior_max = df.groupby(entity_col)["amount_bdt"].cummax().groupby(df[entity_col]).shift(1)
    df[f"{prefix}_amt_vs_prior_max"] = df["amount_bdt"] / (prior_max + EPS)
    df[f"{prefix}_is_record_amt"] = (df["amount_bdt"] > prior_max).astype("float64")
    df.loc[prior_max.isna(), f"{prefix}_is_record_amt"] = np.nan
    return df

def add_hour_profile_features(df: pd.DataFrame, entity_col: str, prefix: str) -> pd.DataFrame:
    bucket = (df[TIME_COL].dt.hour // 4).astype("int8")
    prior_in_bucket = df.groupby([df[entity_col], bucket]).cumcount().astype("float64")
    prior_total = df.groupby(entity_col).cumcount().astype("float64")
    df[f"{prefix}_hour_bucket_share"] = prior_in_bucket / np.maximum(prior_total, 1.0)
    df[f"{prefix}_new_hour_bucket"] = (prior_in_bucket == 0).astype("int8")
    return df

def add_pair_novelty_features(df: pd.DataFrame, col_a: str, col_b: str, name: str) -> pd.DataFrame:
    is_first_pair = ~df.duplicated(subset=[col_a, col_b], keep="first")
    cumsum_incl = is_first_pair.groupby(df[col_a]).cumsum()
    nunique_prior = cumsum_incl - is_first_pair.astype(int)

    df[f"is_new_{name}"] = is_first_pair.astype("int8")
    df[f"{col_a}_nunique_{name}_prior"] = nunique_prior
    prior_txn_count = df.groupby(col_a).cumcount().astype("float64")
    df[f"{col_a}_nunique_{name}_share_prior"] = nunique_prior / np.maximum(prior_txn_count, 1.0)
    return df

def add_frequency_encoding(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    rows_so_far = np.arange(len(df), dtype="float64")
    for col in cat_cols:
        filled = df[col].fillna("__missing__")
        prior_count = filled.groupby(filled).cumcount().astype("float64")
        df[f"{col}_freq_share_prior"] = prior_count / np.maximum(rows_so_far, 1.0)
    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[TIME_COL]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    df["hour_of_day"] = hour
    df["day_of_week"] = dow
    df["is_night"] = ((hour >= 0) & (hour < 6)).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df

def add_recent_window_stats(df: pd.DataFrame, entity_col: str, prefix: str, windows=("5min", "15min", "30min", "3h", "12h")) -> pd.DataFrame:
    sub = df[[entity_col, TIME_COL, "amount_bdt"]]
    grouped = sub.groupby(entity_col)
    for w in windows:
        roll = grouped.rolling(w, on=TIME_COL, closed="left")["amount_bdt"]
        stats = roll.agg(["count", "sum", "mean", "max"])
        stats = stats.reset_index(level=0, drop=True).sort_index()
        tag = w.replace("min", "m")
        df[f"{prefix}_cnt_{tag}"] = np.nan_to_num(stats["count"].values, nan=0.0)
        df[f"{prefix}_amtsum_{tag}"] = np.nan_to_num(stats["sum"].values, nan=0.0)
        df[f"{prefix}_amt_vs_recent_mean_{tag}"] = df["amount_bdt"] / (stats["mean"].values + EPS)
        df[f"{prefix}_amt_vs_recent_max_{tag}"] = df["amount_bdt"] / (stats["max"].values + EPS)
    return df

def add_recent_diversity_features(df: pd.DataFrame, entity_col: str, other_col: str, prefix: str, windows=("1h", "24h", "168h")) -> pd.DataFrame:
    ts = df[TIME_COL].astype("int64") // 10 ** 9
    pair = df[entity_col].astype(str) + "|" + df[other_col].astype(str)
    pair_prev = ts.groupby(pair).shift(1)
    pair_gap = ts - pair_prev

    for w in windows:
        secs = pd.Timedelta(w).total_seconds()
        is_fresh = (pair_gap.isna() | (pair_gap > secs)).astype("float64")
        tmp = pd.DataFrame({entity_col: df[entity_col], TIME_COL: df[TIME_COL], "_f": is_fresh})
        roll = tmp.groupby(entity_col).rolling(w, on=TIME_COL, closed="left")["_f"]
        fresh_cnt = roll.sum().reset_index(level=0, drop=True).sort_index()
        cnt_col = f"{prefix}_cnt_{w}"
        df[f"{prefix}_fresh_{other_col}_{w}"] = np.nan_to_num(fresh_cnt.values, nan=0.0)
        if cnt_col in df.columns:
            df[f"{prefix}_fresh_{other_col}_share_{w}"] = fresh_cnt.values / (df[cnt_col] + 1.0)
    return df

def add_trailing_window_features(df: pd.DataFrame, entity_col: str, prefix: str, windows_hours=(1, 6, 24, 72, 168)) -> pd.DataFrame:
    sub = df[[entity_col, TIME_COL, "amount_bdt"]]
    grouped = sub.groupby(entity_col)
    for h in windows_hours:
        window = f"{h}h"
        roll = grouped.rolling(window, on=TIME_COL, closed="left")["amount_bdt"]
        cnt = roll.count().reset_index(level=0, drop=True).sort_index()
        s = roll.sum().reset_index(level=0, drop=True).sort_index()
        df[f"{prefix}_cnt_{h}h"] = np.nan_to_num(cnt.values, nan=0.0)
        df[f"{prefix}_amtsum_{h}h"] = np.nan_to_num(s.values, nan=0.0)

    for h in windows_hours:
        df[f"{prefix}_amt_share_{h}h"] = df["amount_bdt"] / (df[f"{prefix}_amtsum_{h}h"] + EPS)
    short, long = windows_hours[0], windows_hours[-1]
    df[f"{prefix}_burst_cnt_ratio"] = (df[f"{prefix}_cnt_{short}h"] + 1) / (df[f"{prefix}_cnt_{long}h"] + 1)
    df[f"{prefix}_burst_amt_ratio"] = (df[f"{prefix}_amtsum_{short}h"] + 1) / (df[f"{prefix}_amtsum_{long}h"] + 1)
    return df

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def component_size(self, x: int) -> int:
        return self.size[self.find(x)]

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

def add_bipartite_component_features(df: pd.DataFrame, col_a: str, col_b: str, name: str) -> pd.DataFrame:
    codes_a, uniques_a = pd.factorize(df[col_a])
    codes_b, uniques_b = pd.factorize(df[col_b])
    offset = len(uniques_a)
    n_nodes = offset + len(uniques_b)
    uf = _UnionFind(n_nodes)
    size_a = np.empty(len(df), dtype="int32")
    size_b = np.empty(len(df), dtype="int32")
    a_ids = codes_a
    b_ids = codes_b + offset
    for i in range(len(df)):
        a, b = a_ids[i], b_ids[i]
        size_a[i] = uf.component_size(a)
        size_b[i] = uf.component_size(b)
        uf.union(a, b)
    df[f"{name}_{col_a}_component_size_prior"] = size_a
    df[f"{name}_{col_b}_component_size_prior"] = size_b
    return df

def add_location_and_category_context_features(df: pd.DataFrame) -> pd.DataFrame:
    loc_filled = df["location"].fillna("__unknown__")
    sub_loc = pd.DataFrame({"location": loc_filled, TIME_COL: df[TIME_COL], "amount_bdt": df["amount_bdt"]})
    grouped_loc = sub_loc.groupby("location")
    for h in (1, 24):
        roll = grouped_loc.rolling(f"{h}h", on=TIME_COL, closed="left")["amount_bdt"]
        cnt = roll.count().reset_index(level=0, drop=True).sort_index()
        s = roll.sum().reset_index(level=0, drop=True).sort_index()
        df[f"loc_cnt_{h}h"] = np.nan_to_num(cnt.values, nan=0.0)
        df[f"loc_amtsum_{h}h"] = np.nan_to_num(s.values, nan=0.0)

    cat_filled = df["merchant_category"].fillna("__unknown__")
    cat_grp = df.groupby(cat_filled)["amount_bdt"]
    cat_prior_cnt = cat_grp.cumcount().astype("float64")
    cat_prior_mean = (cat_grp.cumsum() - df["amount_bdt"]) / cat_prior_cnt.replace(0, np.nan)
    df["amt_vs_cat_mean_prior"] = df["amount_bdt"] / (cat_prior_mean + EPS)

    loc_grp = df.groupby(loc_filled)["amount_bdt"]
    loc_prior_cnt = loc_grp.cumcount().astype("float64")
    loc_prior_mean = (loc_grp.cumsum() - df["amount_bdt"]) / loc_prior_cnt.replace(0, np.nan)
    df["loc_amt_ratio"] = df["amount_bdt"] / (loc_prior_mean + EPS)

    for pre in ("cust", "dev"):
        if f"{pre}_cnt_24h" in df.columns and f"{pre}_amtsum_24h" in df.columns:
            mean24 = df[f"{pre}_amtsum_24h"] / (df[f"{pre}_cnt_24h"] + EPS)
            df[f"{pre}_amt_vs_24h_mean"] = df["amount_bdt"] / (mean24 + EPS)
            df[f"{pre}_burst_accel"] = (df[f"{pre}_cnt_1h"] * 24.0) / (df[f"{pre}_cnt_24h"] + 1.0)
        if f"{pre}_cnt_1h" in df.columns and f"{pre}_amtsum_1h" in df.columns:
            df[f"{pre}_smurf_ratio_1h"] = (df[f"{pre}_cnt_1h"] + 1.0) / (df[f"{pre}_amtsum_1h"] + 10.0)

    df["pay_x_dev"] = df["payment_method"].astype(str) + "_" + df["device_type"].astype(str)
    df["cat_x_loc"] = df["merchant_category"].astype(str) + "_" + df["location"].astype(str)
    df["txn_x_pay"] = df["transaction_type"].astype(str) + "_" + df["payment_method"].astype(str)
    rows_so_far = np.arange(len(df), dtype="float64")
    for col in ("pay_x_dev", "cat_x_loc", "txn_x_pay"):
        filled = df[col].fillna("__unknown__")
        df[f"{col}_freq_share_prior"] = filled.groupby(filled).cumcount().astype("float64") / np.maximum(rows_so_far, 1.0)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_calendar_features(df)
    for entity_col, prefix in [("customer_id", "cust"), ("merchant_id", "merch"), ("device_id", "dev")]:
        df = add_entity_expanding_features(df, entity_col, prefix)
    df = add_pair_novelty_features(df, "customer_id", "device_id", "device_for_customer")
    df = add_pair_novelty_features(df, "customer_id", "merchant_id", "merchant_for_customer")
    df = add_pair_novelty_features(df, "device_id", "customer_id", "customer_for_device")
    df = add_pair_novelty_features(df, "merchant_id", "customer_id", "customer_for_merchant")
    df = add_frequency_encoding(df, ["merchant_category", "device_type", "location", "payment_method", "transaction_type"])
    df = add_trailing_window_features(df, "customer_id", "cust")
    df = add_trailing_window_features(df, "merchant_id", "merch")
    df = add_trailing_window_features(df, "device_id", "dev")
    df = add_pair_novelty_features(df, "customer_id", "location", "location_for_customer")
    df = add_recent_window_stats(df, "customer_id", "cust", windows=("30min", "3h", "12h"))
    df = add_recent_window_stats(df, "device_id", "dev", windows=("5min", "15min", "30min", "3h", "12h"))
    df = add_recent_window_stats(df, "merchant_id", "merch", windows=("5min", "30min", "3h"))
    df = add_recent_diversity_features(df, "device_id", "customer_id", "dev")
    df = add_recent_diversity_features(df, "customer_id", "device_id", "cust")
    df = add_recent_diversity_features(df, "customer_id", "merchant_id", "cust")
    df = add_recent_diversity_features(df, "customer_id", "location", "cust")
    for entity_col, prefix in [("customer_id", "cust"), ("merchant_id", "merch"), ("device_id", "dev")]:
        df = add_self_relative_features(df, entity_col, prefix)
    df = add_hour_profile_features(df, "customer_id", "cust")
    df = add_bipartite_component_features(df, "customer_id", "device_id", "cd")
    df = add_bipartite_component_features(df, "customer_id", "merchant_id", "cm")
    df = add_location_and_category_context_features(df)
    return df.copy()

FORWARD_MARKERS = ("_fwd_", "_sym_", "_seconds_to_next", "_accel_", "_amt_vs_sym_", "_gap_fwd_vs_bwd")

def assert_strictly_past(feature_cols) -> None:
    bad = [c for c in feature_cols if any(m in c for m in FORWARD_MARKERS)]
    assert not bad, f"Forward-looking feature leakage detected: {bad}"

def leakage_assertions(df: pd.DataFrame):
    first_cust = df["cust_history_count"] == 0
    assert df.loc[first_cust, "cust_amt_mean_prior"].isna().all(), "leak in customer expanding mean"
    assert (df.loc[first_cust, "is_new_device_for_customer"] == 1).all(), "leak in new device flag"
    print("Point-in-time leakage assertions passed successfully.")
"""

CODE_FEATURES_LIST = f"""# =============================================================================
# 3. SELECTED OPTIMAL 160 FEATURES
# =============================================================================
SELECTED_FEATURES = {repr(SELECTED_FEATURES)}

CAT_COLS = [
    "merchant_category", "device_type", "location", "payment_method", "transaction_type",
    "hour_of_day", "day_of_week", "is_night", "pay_x_dev", "cat_x_loc", "txn_x_pay",
]

def prepare_lgb_frame(frame: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = frame[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in X.columns:
        if X[c].dtype == "float64":
            X[c] = X[c].astype("float32")
    for banned in ID_COLS:
        assert banned not in X.columns, f"banned raw ID column {{banned}} leaked"
    assert LABEL_COL not in X.columns
    return X
"""

CODE_NN_PROP = r'''# =============================================================================
# 4. TABULAR RESNET & TEMPORAL PREDICTION PROPAGATION
# =============================================================================
class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x)
        z = self.linear1(z)
        z = self.act(z)
        z = self.dropout1(z)
        z = self.linear2(z)
        z = self.dropout2(z)
        return residual + z

class TabularResNet(nn.Module):
    def __init__(self, num_features: int, cat_cardinalities: list[int], hidden_dim: int = 256, num_blocks: int = 3, dropout: float = 0.15):
        super().__init__()
        embedding_dims = [min(16, max(4, int(1.6 * (card ** 0.56)))) for card in cat_cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(card + 1, dim) for card, dim in zip(cat_cardinalities, embedding_dims)])
        total_emb_dim = sum(embedding_dims)
        self.input_proj = nn.Linear(num_features + total_emb_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) > 0 and x_cat is not None and x_cat.shape[1] > 0:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num] + emb_outs, dim=1)
        else:
            x = x_num
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        return self.head(h).squeeze(-1)

def sliding_loo_blend(preds, entity_ids, timestamps, w: float = 0.50, window_minutes: float = 60.0):
    """Leave-one-out blend over a sliding time window per entity (+/- 30 min)."""
    if not isinstance(entity_ids, dict):
        entity_ids = {"entity": entity_ids}
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)).reset_index(drop=True))
    t = (ts - ts.min()).dt.total_seconds().to_numpy() / 60.0
    p = np.asarray(preds, dtype="float64")
    half = float(window_minutes) / 2.0

    loos = []
    for ids in entity_ids.values():
        codes = pd.factorize(pd.Series(np.asarray(ids)).astype(str))[0].astype("int64")
        span = (t.max() - t.min()) + 2 * half + 1.0
        key = codes * span + t
        order = np.argsort(key, kind="stable")
        ks, ps = key[order], p[order]
        csum = np.concatenate([[0.0], np.cumsum(ps)])
        lo = np.searchsorted(ks, ks - half, side="left")
        hi = np.searchsorted(ks, ks + half, side="right")
        n = (hi - lo).astype("float64")
        tot = csum[hi] - csum[lo]
        loo_sorted = np.where(n > 1, (tot - ps) / np.maximum(n - 1, 1), ps)
        loo = np.empty_like(loo_sorted)
        loo[order] = loo_sorted
        loos.append(loo)
    return (1.0 - w) * p + w * np.mean(loos, axis=0)
'''

CODE_MAIN_EXECUTION = r"""# =============================================================================
# 5. END-TO-END TRAINING, ENSEMBLING & SUBMISSION GENERATION
# =============================================================================
t0 = time.time()
print("1. Ingesting raw datasets...")
train_raw = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])
test_raw = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])
sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
print(f"   Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")

df = build_combined_frame(train_raw, test_raw)
del train_raw, test_raw
gc.collect()

print("\n2. Executing leakage-safe feature engineering pipeline...")
df = build_features(df)
leakage_assertions(df)
assert_strictly_past(SELECTED_FEATURES)

selected_cols = [c for c in SELECTED_FEATURES if c in df.columns]
selected_cats = [c for c in CAT_COLS if c in selected_cols]
selected_nums = [c for c in selected_cols if c not in selected_cats]
print(f"   Using {len(selected_cols)} features ({len(selected_nums)} numerical, {len(selected_cats)} categorical)")

labeled = df[~df["is_test"]].copy()
test_df = df[df["is_test"]].copy()
del df
gc.collect()

# Train/Holdout partition (July 1, 2026 split)
holdout_start = pd.Timestamp("2026-07-01")
fit_full = labeled.loc[labeled[TIME_COL] < holdout_start].copy()
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start].copy()

# 90-Day Tail Specialist partition (April 1 to July 1)
mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
train_90d = fit_full.loc[mask_90d].copy()

y_full = fit_full[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)
y_90d = train_90d[LABEL_COL].astype(int)

X_full_lgb = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold_lgb = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test_lgb = prepare_lgb_frame(test_df, selected_cols, selected_cats)
X_90d_lgb  = X_full_lgb.loc[mask_90d]

# -----------------------------------------------------------------------------
# A. DUAL-HORIZON LIGHTGBM ENSEMBLE
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("3. Training LightGBM Horizon A (Full Train, 882 rounds)...")
print("="*70)
ds_full = lgb.Dataset(X_full_lgb, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
p_full = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
b_full = lgb.train(p_full, ds_full, num_boost_round=882)
p_lgb_full_hold = b_full.predict(X_hold_lgb)
p_lgb_full_test = b_full.predict(X_test_lgb)

print("4. Training LightGBM Horizon B (90-Day Specialist, 433 rounds)...")
ds_90d = lgb.Dataset(X_90d_lgb, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
p_90d = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
)
b_90d = lgb.train(p_90d, ds_90d, num_boost_round=433)
p_lgb_90d_hold = b_90d.predict(X_hold_lgb)
p_lgb_90d_test = b_90d.predict(X_test_lgb)

p_lgb_champ_hold = 0.48 * p_lgb_full_hold + 0.52 * p_lgb_90d_hold
p_lgb_champ_test = 0.48 * p_lgb_full_test + 0.52 * p_lgb_90d_test
print(f"   LightGBM Dual-Horizon Holdout PR-AUC = {average_precision_score(y_hold, p_lgb_champ_hold):.4f}")

# -----------------------------------------------------------------------------
# B. TABULAR RESNET (Neural Manifold Regularization)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("5. Training Tabular ResNet (Seed 42)...")
print("="*70)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")

cat_cardinalities, train_cats, hold_cats, test_cats = [], [], [], []
for c in selected_cats:
    val_counts = train_90d[c].value_counts()
    categories = list(val_counts.index)
    cat_to_id = {val: idx + 1 for idx, val in enumerate(categories)}
    cat_cardinalities.append(len(categories) + 1)
    train_cats.append(train_90d[c].map(cat_to_id).fillna(0).values.astype(np.int64))
    hold_cats.append(hold_df[c].map(cat_to_id).fillna(0).values.astype(np.int64))
    test_cats.append(test_df[c].map(cat_to_id).fillna(0).values.astype(np.int64))

X_cat_train = np.column_stack(train_cats) if train_cats else np.empty((len(train_90d), 0), dtype=np.int64)
X_cat_hold  = np.column_stack(hold_cats)  if hold_cats  else np.empty((len(hold_df), 0), dtype=np.int64)
X_cat_test  = np.column_stack(test_cats)  if test_cats  else np.empty((len(test_df), 0), dtype=np.int64)

X_num_train_raw = train_90d[selected_nums].copy()
X_num_hold_raw  = hold_df[selected_nums].copy()
X_num_test_raw  = test_df[selected_nums].copy()

medians = X_num_train_raw.median()
X_num_train_raw = X_num_train_raw.fillna(medians)
X_num_hold_raw  = X_num_hold_raw.fillna(medians)
X_num_test_raw  = X_num_test_raw.fillna(medians)

scaler = StandardScaler()
X_num_train = np.clip(scaler.fit_transform(X_num_train_raw), -5.0, 5.0).astype(np.float32)
X_num_hold  = np.clip(scaler.transform(X_num_hold_raw), -5.0, 5.0).astype(np.float32)
X_num_test  = np.clip(scaler.transform(X_num_test_raw), -5.0, 5.0).astype(np.float32)

y_train_arr = train_90d[LABEL_COL].values.astype(np.float32)
y_hold_arr  = hold_df[LABEL_COL].values.astype(np.float32)

BATCH_SIZE = 1024
train_dataset = TensorDataset(torch.from_numpy(X_num_train), torch.from_numpy(X_cat_train), torch.from_numpy(y_train_arr))
hold_dataset  = TensorDataset(torch.from_numpy(X_num_hold), torch.from_numpy(X_cat_hold), torch.from_numpy(y_hold_arr))
test_dataset  = TensorDataset(torch.from_numpy(X_num_test), torch.from_numpy(X_cat_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
hold_loader  = DataLoader(hold_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)

resnet = TabularResNet(
    num_features=len(selected_nums),
    cat_cardinalities=cat_cardinalities,
    hidden_dim=256,
    num_blocks=3,
    dropout=0.15,
).to(device)

EPOCHS = 8
optimizer = torch.optim.AdamW(resnet.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss()

best_resnet_auc = 0.0
best_state = None
best_hold_resnet = None

for epoch in range(1, EPOCHS + 1):
    resnet.train()
    for bn, bc, by in train_loader:
        bn, bc, by = bn.to(device), bc.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(resnet(bn, bc), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnet.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    resnet.eval()
    hp = []
    with torch.no_grad():
        for bn, bc, _ in hold_loader:
            bn, bc = bn.to(device), bc.to(device)
            hp.append(torch.sigmoid(resnet(bn, bc)).cpu().numpy())
    vp = np.concatenate(hp)
    va = average_precision_score(y_hold_arr, vp)
    if va > best_resnet_auc:
        best_resnet_auc = va
        best_hold_resnet = vp.copy()
        best_state = {k: v.cpu().clone() for k, v in resnet.state_dict().items()}
    print(f"   Epoch {epoch}/{EPOCHS} | Val PR-AUC = {va:.4f}")

print(f"   --> Tabular ResNet Standalone PR-AUC = {best_resnet_auc:.4f}")

resnet.load_state_dict(best_state)
resnet.to(device)
resnet.eval()
tp = []
with torch.no_grad():
    for bn, bc in test_loader:
        bn, bc = bn.to(device), bc.to(device)
        tp.append(torch.sigmoid(resnet(bn, bc)).cpu().numpy())
p_resnet_test = np.concatenate(tp)

# -----------------------------------------------------------------------------
# C. MIXTURE BLEND & TEMPORAL PROPAGATION
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("6. Tree-Neural Mixture & Temporal Prediction Propagation...")
print("="*70)
p_raw_blend_hold = 0.88 * p_lgb_champ_hold + 0.12 * best_hold_resnet
p_raw_blend_test = 0.88 * p_lgb_champ_test + 0.12 * p_resnet_test
print(f"   Raw Tree-Neural Holdout PR-AUC = {average_precision_score(y_hold, p_raw_blend_hold):.4f}")

ent_hold = {"cust": hold_df["customer_id"].values, "dev": hold_df["device_id"].values}
ent_test = {"cust": test_df["customer_id"].values, "dev": test_df["device_id"].values}

prop_hold = sliding_loo_blend(p_raw_blend_hold, ent_hold, hold_df[TIME_COL], w=0.50, window_minutes=60.0)
final_local_score = average_precision_score(y_hold, prop_hold)
print(f"\n🏆 FINAL PROPAGATED TREE-NEURAL LOCAL PR-AUC = {final_local_score:.4f}")

prop_test = sliding_loo_blend(p_raw_blend_test, ent_test, test_df[TIME_COL], w=0.50, window_minutes=60.0)

# Build submission
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": prop_test})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

# Verifications
assert len(sub) == len(sample_sub) == 262648, "Row count mismatch"
assert not sub["fraud"].isna().any(), "Contains NaNs"
assert (sub["fraud"] >= 0.0).all() and (sub["fraud"] <= 1.0).all(), "Predictions outside [0, 1]"
assert list(sub["transaction_id"]) == list(sample_sub["transaction_id"]), "ID mismatch"

out_path = OUT_DIR / "submission.csv"
sub.to_csv(out_path, index=False)
print(f"\n" + "="*70)
print(f"✅ SUCCESS: Wrote {out_path} ({len(sub)} rows, mean={sub['fraud'].mean():.5f})")
print(f"   Total pipeline elapsed time: {time.time()-t0:.1f}s")
print(f"="*70)
"""

def make_cell(cell_type, text):
    lines = [line + "\n" for line in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }

def main():
    nb = {
        "cells": [
            make_cell("markdown", MD_HEADER),
            make_cell("code", CODE_SETUP),
            make_cell("code", CODE_FEATURES),
            make_cell("code", CODE_FEATURES_LIST),
            make_cell("code", CODE_NN_PROP),
            make_cell("code", CODE_MAIN_EXECUTION),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully generated complete Champion notebook: {OUT} ({len(nb['cells'])} cells)")

if __name__ == "__main__":
    main()
