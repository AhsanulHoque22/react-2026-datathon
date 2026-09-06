"""Leakage-safe behavioral feature engineering (PLAN.md "Feature engineering").

Core rule enforced throughout: every feature for row i must depend only on
rows strictly before row i in time (ties broken by transaction_id). All
expanding stats are computed via cumsum/cumcount tricks that exclude the
current row's own value, not `.expanding()` + `.shift(1)` (equivalent, but
vectorized and far faster at ~1M rows x tens of thousands of groups).

Banned per PLAN.md: raw ID columns as model features, and any per-entity
fraud-rate/target encoding. Nothing here touches the `fraud` column.
"""
import numpy as np
import pandas as pd

from src.config import TIME_COL

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

    # time since this entity's previous transaction
    ts_epoch = df[TIME_COL].astype("int64") // 10 ** 9
    prev_ts = ts_epoch.groupby(df[entity_col]).shift(1)
    df[f"{prefix}_seconds_since_last"] = ts_epoch - prev_ts

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
    return df


def add_frequency_encoding(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    """Strictly-past expanding frequency (count, not rate) of each category value."""
    for col in cat_cols:
        filled = df[col].fillna("__missing__")
        df[f"{col}_freq_prior"] = filled.groupby(filled).cumcount().astype("float64")
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


def add_trailing_window_features(df: pd.DataFrame, entity_col: str, prefix: str, windows_hours=(1, 24)) -> pd.DataFrame:
    """Trailing rolling count/sum of amount_bdt in the last N hours, strictly prior
    (closed='left' excludes the current row's own timestamp)."""
    sub = df[[entity_col, TIME_COL, "amount_bdt"]]
    grouped = sub.groupby(entity_col)
    for h in windows_hours:
        window = f"{h}h"
        roll = grouped.rolling(window, on=TIME_COL, closed="left")["amount_bdt"]
        cnt = roll.count().reset_index(level=0, drop=True).sort_index()
        s = roll.sum().reset_index(level=0, drop=True).sort_index()
        df[f"{prefix}_cnt_{h}h"] = cnt.values
        df[f"{prefix}_amtsum_{h}h"] = s.values
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

    df = add_trailing_window_features(df, "customer_id", "cust", windows_hours=(1, 24))
    df = add_trailing_window_features(df, "merchant_id", "merch", windows_hours=(1, 24))
    df = add_trailing_window_features(df, "device_id", "dev", windows_hours=(1, 24))

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
