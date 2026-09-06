"""REACT 2026 Datathon -- full pipeline, generated from src/ by
scripts/build_kaggle_kernel.py. Do not edit here; edit src/ and regenerate.

Leakage-safe behavioural feature engineering + LightGBM, trained on a
chronological split. Every engineered feature for a row at time t uses only
rows strictly before t (ties broken by transaction_id). No raw entity IDs
reach the model, and nothing touches the fraud label except the label itself.
"""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_score, recall_score

# Attaching the competition via the API mounts it under input/competitions/,
# while the notebook UI mounts it directly under input/. Accept either.
_CANDIDATES = [
    Path("/kaggle/input/competitions/react-2026-datathon"),
    Path("/kaggle/input/react-2026-datathon"),
]
RAW_DIR = next((p for p in _CANDIDATES if (p / "train.csv").exists()), _CANDIDATES[0])
OUT_DIR = Path("/kaggle/working")
TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"
print(f"Using data dir: {RAW_DIR}")

# ===================== src/config.py =====================
SEED = 42



# Raw ID / label columns that must NEVER enter the model's feature matrix directly
# (PLAN.md "Banned" section). Kept only as join keys during feature engineering.
ID_COLS = ["customer_id", "merchant_id", "device_id", "transaction_id"]
LABEL_COL = "fraud"
TIME_COL = "timestamp"

# Walk-forward CV fold windows (PLAN.md "Validation" section).
# Each fold's test window; train = all rows strictly before the window's start.
CV_FOLDS = [
    ("2026-05-21", "2026-06-04"),
    ("2026-06-04", "2026-06-18"),
    ("2026-06-18", "2026-07-02"),
    ("2026-07-02", "2026-07-15"),
]

FEVAL_SUBSAMPLE_SIZE = 120_000

# ===================== src/data.py =====================
def load_raw():
    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])
    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])
    return train, test


def build_combined_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Concatenate train+test, sort by (timestamp, transaction_id), tag split."""
    train = train.copy()
    test = test.copy()
    train["is_test"] = False
    test["is_test"] = True
    if LABEL_COL not in test.columns:
        test[LABEL_COL] = np.nan

    df = pd.concat([train, test], axis=0, ignore_index=True)
    df = df.sort_values([TIME_COL, "transaction_id"], kind="mergesort").reset_index(drop=True)
    return df


def assert_frame_sane(df: pd.DataFrame):
    assert df[TIME_COL].is_monotonic_increasing, "frame not sorted by timestamp after tie-break sort"
    # tie-break key must also be non-decreasing within any timestamp tie
    dup_ts = df[TIME_COL].duplicated(keep=False)
    if dup_ts.any():
        sub = df.loc[dup_ts, [TIME_COL, "transaction_id"]]
        for _, g in sub.groupby(TIME_COL):
            assert g["transaction_id"].is_monotonic_increasing, "tie-break key not sorted within a timestamp tie"

# ===================== src/features.py =====================
EPS = 1e-6


def _prior_count_sum_sumsq(df: pd.DataFrame, entity_col: str, value_col: str):
    """Vectorized strictly-prior count/sum/sumsq of value_col within entity_col groups."""
    grp = df.groupby(entity_col)[value_col]
    cum_sum = grp.cumsum()
    prior_sum = cum_sum - df[value_col]
    prior_count = grp.cumcount().astype("float64")

    sq_col = df[value_col] ** 2
    cum_sumsq = sq_col.groupby(df[entity_col]).cumsum()
    prior_sumsq = cum_sumsq - sq_col
    return prior_count, prior_sum, prior_sumsq


def _prior_median_mad(df: pd.DataFrame, entity_col: str, value_col: str, prior_count: pd.Series):
    """Prior expanding median/MAD per entity. Not cumsum-vectorizable exactly;
    uses an expanding().median()/mad-like pass per group, which is still fast
    at this data scale (grouped expanding, not row-by-row apply)."""
    s = df[value_col]
    g = df.groupby(entity_col)[value_col]
    expanding_median_incl = g.expanding().median().reset_index(level=0, drop=True)
    prior_median = expanding_median_incl.shift(1)
    # MAD of prior values around the prior median (approximate: uses prior_median as center)
    abs_dev = (s - prior_median).abs()
    expanding_mad_incl = abs_dev.groupby(df[entity_col]).expanding().median().reset_index(level=0, drop=True)
    prior_mad = expanding_mad_incl.shift(1)
    # first occurrence per entity has no prior median/mad -> NaN (cold start, left as NaN)
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

    # z-score deviation from own baseline (std-based)
    df[f"{prefix}_amt_zscore"] = (df["amount_bdt"] - prior_mean) / (prior_std + EPS)

    # robust MAD-based deviation
    prior_median, prior_mad = _prior_median_mad(df, entity_col, "amount_bdt", prior_count)
    df[f"{prefix}_amt_robust_z"] = (df["amount_bdt"] - prior_median) / (prior_mad + EPS)

    # amount ratio to the entity's own prior mean. Distinct from the z-scores
    # above: those divide by std/MAD (spread), this divides by the mean
    # (level). Measured as the single strongest surviving feature in the July
    # regime (univariate AP 0.2448 vs 0.2393 for robust-z, 0.1607 for raw
    # amount) -- see scripts/12_july_signal_hunt.py.
    #
    # No log variant: LightGBM splits on thresholds, so it is invariant to
    # monotonic transforms of a single feature. log1p(ratio) would produce
    # identical splits and identical AP, while stealing a slot from
    # feature_fraction sampling. (The log *z-scores* below are NOT redundant
    # -- their mean/std are computed in log space, which reorders rows.)
    df[f"{prefix}_amt_ratio"] = df["amount_bdt"] / (prior_mean + EPS)

    # log-amount z-score: amount is heavily right-skewed (fraud amounts run
    # to ~5x the legit max), so a z-score on log1p(amount) is better-behaved
    # than the linear-scale z-score above -- kept alongside it, not instead.
    if "log_amount_bdt" not in df.columns:
        df["log_amount_bdt"] = np.log1p(df["amount_bdt"])
    log_prior_count, log_prior_sum, log_prior_sumsq = _prior_count_sum_sumsq(df, entity_col, "log_amount_bdt")
    log_prior_mean = log_prior_sum / log_prior_count.replace(0, np.nan)
    log_prior_var = (log_prior_sumsq / log_prior_count.replace(0, np.nan)) - log_prior_mean ** 2
    log_prior_std = np.sqrt(log_prior_var.clip(lower=0))
    df[f"{prefix}_log_amt_zscore"] = (df["log_amount_bdt"] - log_prior_mean) / (log_prior_std + EPS)

    # time since this entity's previous transaction
    ts_epoch = df[TIME_COL].astype("int64") // 10 ** 9
    prev_ts = ts_epoch.groupby(df[entity_col]).shift(1)
    df[f"{prefix}_seconds_since_last"] = ts_epoch - prev_ts
    # (no log variant -- monotonic transform, identical tree splits)

    return df


def add_pair_novelty_features(df: pd.DataFrame, col_a: str, col_b: str, name: str) -> pd.DataFrame:
    """For each (col_a, col_b) pair, add:
    - is_first_pair: True at the row this exact pair first appears (chronologically)
    - nunique of col_b seen so far by col_a (prior, i.e. before this row)
    """
    is_first_pair = ~df.duplicated(subset=[col_a, col_b], keep="first")
    cumsum_incl = is_first_pair.groupby(df[col_a]).cumsum()
    nunique_prior = cumsum_incl - is_first_pair.astype(int)

    df[f"is_new_{name}"] = is_first_pair.astype("int8")
    df[f"{col_a}_nunique_{name}_prior"] = nunique_prior

    # Stationary companion: distinct counterparties as a SHARE of this
    # entity's own prior transactions. The raw count saturates over time
    # (measured: "device shared with >3 customers" fires on 5.7% of rows in
    # January but 91.7% by July, at which point its fraud lift is ~0.9 --
    # i.e. the feature has decayed into noise). The share stays comparable
    # across periods: "this device sees a new customer on most of its
    # transactions" means the same thing in January and July.
    prior_txn_count = df.groupby(col_a).cumcount().astype("float64")
    df[f"{col_a}_nunique_{name}_share_prior"] = nunique_prior / np.maximum(prior_txn_count, 1.0)
    return df


def add_frequency_encoding(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    """Strictly-past expanding frequency of each category value, as a
    PROPORTION of all rows seen so far rather than a raw running count.

    The raw count was non-stationary by construction: it only grows, so
    "transaction_type_freq_prior > 50000" is really a disguised timestamp,
    and test-period values sit far outside the range seen in training
    (measured PSI 9.27 for transaction_type, 5.63 payment_method, 5.40
    device_type -- among the worst drifters in the whole feature set).
    The share-of-traffic version carries the same information about how
    common a category is, but is scale-invariant across periods.
    """
    rows_so_far = np.arange(len(df), dtype="float64")  # rows strictly before row i
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


def add_trailing_window_features(df: pd.DataFrame, entity_col: str, prefix: str, windows_hours=(1, 6, 24, 72, 168)) -> pd.DataFrame:
    """Trailing rolling count/sum of amount_bdt in the last N hours, strictly prior
    (closed='left' excludes the current row's own timestamp).

    This family is the priority: it is the ONLY feature class that gets
    STRONGER in the July regime the test set resembles (univariate AP
    dev_amtsum_24h 0.0669 June -> 0.1315 July, cust_amtsum_24h 1.73x,
    dev_amtsum_1h 1.39x) while every other family roughly halves. So it gets
    more horizons (1h/6h/24h/3d/7d) plus derived ratios, rather than the
    original two windows.
    """
    sub = df[[entity_col, TIME_COL, "amount_bdt"]]
    grouped = sub.groupby(entity_col)
    for h in windows_hours:
        window = f"{h}h"
        roll = grouped.rolling(window, on=TIME_COL, closed="left")["amount_bdt"]
        cnt = roll.count().reset_index(level=0, drop=True).sort_index()
        s = roll.sum().reset_index(level=0, drop=True).sort_index()
        df[f"{prefix}_cnt_{h}h"] = cnt.values
        df[f"{prefix}_amtsum_{h}h"] = s.values

    # This transaction's amount as a share of the entity's recent volume, and
    # burst ratios (short window vs long window). Both are scale-invariant
    # ratios within the family that survives the regime change.
    for h in windows_hours:
        df[f"{prefix}_amt_share_{h}h"] = df["amount_bdt"] / (df[f"{prefix}_amtsum_{h}h"] + EPS)
    short, long = windows_hours[0], windows_hours[-1]
    df[f"{prefix}_burst_cnt_ratio"] = (df[f"{prefix}_cnt_{short}h"] + 1) / (df[f"{prefix}_cnt_{long}h"] + 1)
    df[f"{prefix}_burst_amt_ratio"] = (df[f"{prefix}_amtsum_{short}h"] + 1) / (df[f"{prefix}_amtsum_{long}h"] + 1)
    return df


class _UnionFind:
    """Weighted union-find with path compression. Used instead of a single
    static `scipy.sparse.csgraph.connected_components` call (PLAN.md's
    originally-suggested tool): that function only answers "what are the
    components of this *fixed* graph" -- it has no notion of point-in-time.
    Naively running it once on the full combined graph would leak edges
    formed by *later* transactions into early rows' features. This
    incremental structure gives the exact per-row equivalent instead:
    look up an entity's current component size (read), THEN add today's
    edge (write) -- so every row's feature reflects strictly-prior edges
    only, at row-level precision rather than a coarser daily/weekly
    snapshot. Still the same underlying idea PLAN.md scoped (bipartite
    connected components on customer-device / customer-merchant graphs),
    just applied incrementally rather than as one static scipy call.

    Kept despite scripts/02_cv_eval.py showing no measurable CV benefit
    (mean 0.6923 -> 0.6929, inside the noise floor) -- team decision to
    retain it for the write-up / in case a different fold split shows more
    signal, rather than delete unproven-but-harmless code.
    """

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
    """Strictly-past component size of col_a's and col_b's node in the
    incrementally-built col_a<->col_b bipartite graph. E.g. for
    (customer_id, device_id): "how large is the cluster of
    customers+devices this customer already belongs to, before this
    transaction" and the same for the device's own prior cluster --
    directly answers the problem description's "does this device suddenly
    belong to many customers" / "do groups form suspicious clusters"."""
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_calendar_features(df)

    for entity_col, prefix in [
        ("customer_id", "cust"),
        ("merchant_id", "merch"),
        ("device_id", "dev"),
    ]:
        df = add_entity_expanding_features(df, entity_col, prefix)

    df = add_pair_novelty_features(df, "customer_id", "device_id", "device_for_customer")
    df = add_pair_novelty_features(df, "customer_id", "merchant_id", "merchant_for_customer")
    df = add_pair_novelty_features(df, "device_id", "customer_id", "customer_for_device")
    df = add_pair_novelty_features(df, "merchant_id", "customer_id", "customer_for_merchant")

    df = add_frequency_encoding(
        df, ["merchant_category", "device_type", "location", "payment_method", "transaction_type"]
    )

    df = add_trailing_window_features(df, "customer_id", "cust")
    df = add_trailing_window_features(df, "merchant_id", "merch")
    df = add_trailing_window_features(df, "device_id", "dev")

    # is-new-location-for-customer: same novelty pattern as device/merchant,
    # confirmed compliant and worthwhile signal by an independent teardown.
    df = add_pair_novelty_features(df, "customer_id", "location", "location_for_customer")

    # Stretch goal (PLAN.md): second-order relationship/cluster features.
    df = add_bipartite_component_features(df, "customer_id", "device_id", "cd")
    df = add_bipartite_component_features(df, "customer_id", "merchant_id", "cm")

    return df


def leakage_assertions(df: pd.DataFrame):
    """Cheap sanity checks: first-occurrence rows must show NaN/zero prior stats,
    never their own current-row value baked in."""
    first_cust = df["cust_history_count"] == 0
    assert df.loc[first_cust, "cust_amt_mean_prior"].isna().all(), (
        "leak: a customer's first-occurrence row has a non-NaN prior mean"
    )
    assert (df.loc[first_cust, "is_new_device_for_customer"] == 1).all(), (
        "leak: a customer's first transaction should always be a new device pairing"
    )

# ===================== src/model.py =====================
CAT_COLS = ["merchant_category", "device_type", "location", "payment_method", "transaction_type"]
# log_amount_bdt is a helper column for the log-space z-scores, not a model
# feature: it is a monotonic transform of amount_bdt, so it yields identical
# tree splits and identical AP while consuming a feature_fraction slot.
EXCLUDE_COLS = set(ID_COLS) | {LABEL_COL, TIME_COL, "is_test", "log_amount_bdt"}


def get_feature_columns(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    return feature_cols, cat_cols


def prepare_lgb_frame(df: pd.DataFrame, feature_cols, cat_cols) -> pd.DataFrame:
    """Return a copy of df[feature_cols] with categoricals cast to pandas 'category'
    dtype (LightGBM native categorical handling) and raw ID / label columns absent
    (banned per PLAN.md — asserted, not just assumed)."""
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    # float32 halves memory (this box has 7GB RAM and the frame grew to 124
    # columns). LightGBM histogram-bins features anyway, so the reduced
    # precision does not change the splits it can find.
    for c in X.columns:
        if X[c].dtype == "float64":
            X[c] = X[c].astype("float32")
    for banned in ID_COLS:
        assert banned not in X.columns, f"banned raw ID column {banned} leaked into feature matrix"
    assert LABEL_COL not in X.columns
    return X


def make_pr_auc_feval(y_valid: np.ndarray, seed: int = SEED, subsample_size: int = FEVAL_SUBSAMPLE_SIZE):
    """Fixed (not resampled per round) stratified subsample of the validation
    fold, used to keep average_precision_score cheap across boosting rounds."""
    n = len(y_valid)
    if n <= subsample_size:
        idx = np.arange(n)
    else:
        rng = np.random.RandomState(seed)
        pos_idx = np.where(y_valid == 1)[0]
        neg_idx = np.where(y_valid == 0)[0]
        frac = subsample_size / n
        n_pos = max(1, int(round(len(pos_idx) * frac)))
        n_neg = subsample_size - n_pos
        pos_sample = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        neg_sample = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([pos_sample, neg_sample])

    def feval(preds, train_data):
        labels = train_data.get_label()
        score = average_precision_score(labels[idx], preds[idx])
        return "pr_auc", score, True

    return feval


def train_lgb(X_train, y_train, X_valid, y_valid, cat_cols, seed: int = SEED):
    # NO scale_pos_weight. PLAN.md originally prescribed neg/pos (~55x) for the
    # 1.76% imbalance, and every hyperparameter sweep held it fixed, so it went
    # untested until late. Measured on fold 3, less weighting is monotonically
    # better: spw=55 -> 0.5045, spw=10 -> 0.5069, spw=7.4 -> 0.5091,
    # spw=1 -> 0.5116 +/- 0.0020 (3 seeds). PR-AUC is a RANK metric; upweighting
    # positives distorts the ranking it is scored on. +0.0071 = 3.5x seed std.
    params = dict(
        objective="binary",
        metric="None",
        seed=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        verbosity=-1,
        # Re-swept on Kaggle under spw=1 (the original sweep ran under spw=55
        # and every conclusion it drew was void). lr=0.02/leaves=127 scored
        # 0.5230 +/- 0.0006 on the Jul01-15 tail vs 0.5195 +/- 0.0012 for the
        # old lr=0.05/leaves=63 -- and also wins on fold2, so it is not
        # trading away the easy regime. Removing the class weighting shifted
        # the optimum toward a slower rate with many more trees (~940 vs ~140).
        learning_rate=0.02,
        num_leaves=127,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        min_data_in_leaf=50,
    )

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)

    feval = make_pr_auc_feval(y_valid.values if hasattr(y_valid, "values") else y_valid, seed=seed)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        feval=feval,
        # patience 200 to match the sweep that selected lr=0.02: at this slower
        # rate the curve improves in longer, flatter stretches and 100 rounds
        # of patience cuts it off early.
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False), lgb.log_evaluation(period=0)],
    )
    return booster


def precision_recall_at_k(y_true, y_score, k_frac):
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-y_score)
    top_k_idx = order[:k]
    y_pred = np.zeros(n, dtype=int)
    y_pred[top_k_idx] = 1
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return prec, rec

SEEDS = [0, 1, 2, 3, 4]
FINAL_PARAMS = dict(
    objective="binary", metric="None", verbosity=-1, learning_rate=0.02,
    num_leaves=127, feature_fraction=0.85, bagging_fraction=0.85,
    bagging_freq=1, min_data_in_leaf=50,
)


# --------- recency asymmetry + ensemble blending experiments ---------------
def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])
    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])
    df = build_combined_frame(train, test)
    assert_frame_sane(df)
    del train, test
    gc.collect()
    df = build_features(df)
    feature_cols, cat_cols = get_feature_columns(df)
    labeled = df[~df["is_test"]]
    del df
    gc.collect()
    print(f"featurized, {len(feature_cols)} features ({time.time()-t0:.0f}s)", flush=True)

    # EXPERIMENT A -- recency weighting, tested FAIRLY this time.
    # The earlier test validated on Jul 2-15 while training on data ending
    # Jul 2, i.e. entirely BEFORE the regime change -- there was no
    # post-change data to weight toward, so it could not have helped. Here
    # the split is Jul 8: training now contains ~1 week of post-change data,
    # which mirrors the real submission (trains through Jul 15, predicts
    # Jul 16+). This is the only configuration where recency weighting has
    # a mechanism to work.
    split = pd.Timestamp("2026-07-08")
    tr = labeled.loc[labeled[TIME_COL] < split]
    va = labeled.loc[labeled[TIME_COL] >= split]
    X_tr, y_tr = prepare_lgb_frame(tr, feature_cols, cat_cols), tr[LABEL_COL].astype(int)
    X_va, y_va = prepare_lgb_frame(va, feature_cols, cat_cols), va[LABEL_COL].astype(int)
    print(f"\nEXP A split Jul08: train={len(tr)} valid={len(va)} "
          f"frauds={int(y_va.sum())}", flush=True)

    days_before = (tr[TIME_COL].max() - tr[TIME_COL]).dt.total_seconds().values / 86400.0

    def run_weighted(half_life, tag):
        aps = []
        for seed in (0, 1, 2):
            w = None if half_life is None else np.exp(-np.log(2) / half_life * days_before)
            params = dict(objective="binary", metric="None", verbosity=-1,
                          learning_rate=0.05, num_leaves=63, feature_fraction=0.85,
                          bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
                          seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=w, categorical_feature=cat_cols, free_raw_data=False)
            ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=ds_tr, free_raw_data=False)
            bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],
                            feval=make_pr_auc_feval(y_va.values, seed=seed),
                            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)])
            aps.append(average_precision_score(y_va, bst.predict(X_va, num_iteration=bst.best_iteration)))
        print(f"  {tag:32s} {np.mean(aps):.4f} +/- {np.std(aps):.4f}  [{time.time()-t0:.0f}s]", flush=True)
        return float(np.mean(aps))

    print("EXPERIMENT A: recency weighting with post-change data in train")
    run_weighted(None, "no weighting")
    for hl in (7, 14, 30, 60):
        run_weighted(hl, f"exp decay half-life={hl}d")

    # EXPERIMENT B -- ensemble blending. Research says rank-averaging beats
    # probability-averaging for rank metrics because it ignores calibration
    # differences between models. Worth checking directly rather than assuming.
    print("\nEXPERIMENT B: ensemble blending (rank vs probability averaging)")
    CONFIGS = [dict(learning_rate=0.05, num_leaves=63, feature_fraction=0.85),
               dict(learning_rate=0.02, num_leaves=127, feature_fraction=0.7),
               dict(learning_rate=0.1, num_leaves=31, feature_fraction=0.95)]
    preds, singles = [], []
    for i, cfg in enumerate(CONFIGS):
        params = dict(objective="binary", metric="None", verbosity=-1,
                      bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
                      seed=i, bagging_seed=i, feature_fraction_seed=i, **cfg)
        ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=ds_tr, free_raw_data=False)
        bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],
                        feval=make_pr_auc_feval(y_va.values, seed=i),
                        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)])
        p = bst.predict(X_va, num_iteration=bst.best_iteration)
        preds.append(p)
        ap = average_precision_score(y_va, p)
        singles.append(ap)
        print(f"  model {i} {cfg} -> {ap:.4f}", flush=True)

    P = np.vstack(preds)
    prob_avg = P.mean(axis=0)
    ranks = np.vstack([pd.Series(p).rank(pct=True).values for p in preds])
    rank_avg = ranks.mean(axis=0)
    print(f"  best single         : {max(singles):.4f}")
    print(f"  probability average : {average_precision_score(y_va, prob_avg):.4f}")
    print(f"  RANK average        : {average_precision_score(y_va, rank_avg):.4f}", flush=True)


if __name__ == "__main__":
    main()
