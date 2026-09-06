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


def add_self_relative_features(df: pd.DataFrame, entity_col: str, prefix: str) -> pd.DataFrame:
    """Everything here is "this entity vs its OWN norm", which is the only
    pattern that survives the July regime change.

    The univariate audit (scripts/12) found the sole feature family that gets
    STRONGER in July is short-horizon volume (dev_amtsum_24h 1.96x,
    cust_amtsum_24h 1.73x), and the single best July feature is
    cust_amt_ratio -- amount over the entity's own prior mean. Both are
    self-relative. Absolute levels and cumulative counts decay. So these
    normalise the winning family by each entity's own baseline rather than
    adding more absolute quantities.
    """
    ts = df[TIME_COL].astype("int64") // 10 ** 9
    first_ts = ts.groupby(df[entity_col]).transform("min")
    age_sec = (ts - first_ts).astype("float64")
    hist = df[f"{prefix}_history_count"]

    # Expected inter-arrival gap for this entity, from its own history.
    mean_gap = age_sec / np.maximum(hist, 1.0)
    # >1 means this transaction came sooner than this entity's own norm.
    df[f"{prefix}_gap_accel"] = mean_gap / (df[f"{prefix}_seconds_since_last"] + 1.0)

    # Burst relative to the entity's own long-run rate, rather than in
    # absolute counts -- "unusually busy FOR ITSELF right now".
    rate_per_sec = hist / np.maximum(age_sec, 1.0)
    for h in (1, 24):
        col = f"{prefix}_cnt_{h}h"
        if col in df.columns:
            expected = rate_per_sec * (h * 3600.0)
            df[f"{prefix}_velocity_ratio_{h}h"] = df[col] / (expected + EPS)

    # Largest amount this entity has ever transacted, and whether this one
    # breaks that record. A new personal maximum is a different signal from
    # "large relative to the mean".
    # Shift WITHIN the group: an entity's rows are scattered through the
    # time-sorted frame, so a plain .shift(1) would pull in whatever
    # unrelated transaction happened to precede it globally.
    prior_max = df.groupby(entity_col)["amount_bdt"].cummax().groupby(df[entity_col]).shift(1)
    df[f"{prefix}_amt_vs_prior_max"] = df["amount_bdt"] / (prior_max + EPS)
    df[f"{prefix}_is_record_amt"] = (df["amount_bdt"] > prior_max).astype("float64")
    df.loc[prior_max.isna(), f"{prefix}_is_record_amt"] = np.nan
    return df


def add_hour_profile_features(df: pd.DataFrame, entity_col: str, prefix: str) -> pd.DataFrame:
    """Is this an unusual hour for THIS entity? The brief asks for exactly
    this ("is this an unusual hour or day for this customer?"), and our
    existing calendar features only encode the global hour, not whether the
    hour is unusual for the individual."""
    bucket = (df[TIME_COL].dt.hour // 4).astype("int8")  # 6 x 4-hour buckets
    prior_in_bucket = df.groupby([df[entity_col], bucket]).cumcount().astype("float64")
    prior_total = df.groupby(entity_col).cumcount().astype("float64")
    df[f"{prefix}_hour_bucket_share"] = prior_in_bucket / np.maximum(prior_total, 1.0)
    df[f"{prefix}_new_hour_bucket"] = (prior_in_bucket == 0).astype("int8")
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


def add_recent_window_stats(df: pd.DataFrame, entity_col: str, prefix: str, windows=("5min", "15min", "30min", "3h", "12h")) -> pd.DataFrame:
    """Sub-hour and intermediate horizons, plus amount-relative-to-recent.

    Phase 3.1/3.2 of the fold-3 plan. Short-window device velocity is the one
    family that gets STRONGER in July (dev_amtsum_6h 1.82x, dev_amtsum_24h
    1.96x June->July), so the burst region between 5 minutes and 12 hours is
    where the surviving signal lives -- and we previously only sampled
    1h/6h/24h/72h/168h, skipping sub-hour entirely.

    The ratios matter more than the levels: absolute amount thresholds
    transfer badly across the regime change (median fraud amount fell from
    ~2,524 to ~1,064 BDT), while "large relative to this entity's own recent
    behaviour" is scale-free.
    """
    sub = df[[entity_col, TIME_COL, "amount_bdt"]]
    grouped = sub.groupby(entity_col)
    for w in windows:
        roll = grouped.rolling(w, on=TIME_COL, closed="left")["amount_bdt"]
        stats = roll.agg(["count", "sum", "mean", "max"])
        stats = stats.reset_index(level=0, drop=True).sort_index()
        tag = w.replace("min", "m")
        # An empty window means ZERO prior activity, which is real information.
        # pandas rolling returns NaN there, which would tell LightGBM "unknown"
        # and lump quiet entities in with genuinely-missing values.
        df[f"{prefix}_cnt_{tag}"] = np.nan_to_num(stats["count"].values, nan=0.0)
        df[f"{prefix}_amtsum_{tag}"] = np.nan_to_num(stats["sum"].values, nan=0.0)
        # scale-free: this amount against its own recent context
        df[f"{prefix}_amt_vs_recent_mean_{tag}"] = df["amount_bdt"] / (stats["mean"].values + EPS)
        df[f"{prefix}_amt_vs_recent_max_{tag}"] = df["amount_bdt"] / (stats["max"].values + EPS)
    return df


def add_recent_diversity_features(df: pd.DataFrame, entity_col: str, other_col: str, prefix: str,
                                  windows=("1h", "24h", "168h")) -> pd.DataFrame:
    """How many counterparties has this entity seen RECENTLY that it had not
    seen just before -- i.e. current suspicious expansion.

    Phase 3.3. The lifetime version of this saturates and dies: "device shared
    with >3 customers" fires on 5.7% of rows in January but 91.7% by July, at
    which point its fraud lift is ~0.9 (noise). A recent-window version cannot
    saturate, because it forgets.

    Implemented via pair recency rather than a rolling nunique (which pandas
    cannot do efficiently over ~1M rows): flag rows where this exact
    (entity, counterparty) pair had not transacted within the window, then
    roll a sum of those flags. That counts "transactions from counterparties
    this entity had not just seen" -- the expansion signal we want.
    """
    ts = df[TIME_COL].astype("int64") // 10 ** 9
    pair = df[entity_col].astype(str) + "|" + df[other_col].astype(str)
    pair_prev = ts.groupby(pair).shift(1)
    pair_gap = ts - pair_prev  # NaN => never seen before

    for w in windows:
        secs = pd.Timedelta(w).total_seconds()
        is_fresh = (pair_gap.isna() | (pair_gap > secs)).astype("float64")
        tmp = pd.DataFrame({entity_col: df[entity_col], TIME_COL: df[TIME_COL], "_f": is_fresh})
        roll = tmp.groupby(entity_col).rolling(w, on=TIME_COL, closed="left")["_f"]
        fresh_cnt = roll.sum().reset_index(level=0, drop=True).sort_index()
        cnt_col = f"{prefix}_cnt_{w}"
        df[f"{prefix}_fresh_{other_col}_{w}"] = np.nan_to_num(fresh_cnt.values, nan=0.0)
        if cnt_col in df.columns:
            # share: is this entity's recent traffic mostly NEW counterparties?
            df[f"{prefix}_fresh_{other_col}_share_{w}"] = fresh_cnt.values / (df[cnt_col] + 1.0)
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
        # empty window == zero prior activity (see add_recent_window_stats)
        df[f"{prefix}_cnt_{h}h"] = np.nan_to_num(cnt.values, nan=0.0)
        df[f"{prefix}_amtsum_{h}h"] = np.nan_to_num(s.values, nan=0.0)

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


def add_location_and_category_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Location velocity surges, category-relative amounts, and smurfing ratios.
    
    In July, fraud concentrated geographically (LOC_033 fraud rate jumped 4.2%->8.7%,
    LOC_016 jumped 2.3%->5.2%) and median fraud amount collapsed from 1,636 to 1,064 BDT.
    These features capture:
    1. Geographic velocity surges (rolling 1h and 24h count/sum by location).
    2. Category-relative amount scaling (evaluates amounts against category baseline).
    3. Smurfing burst ratios (high count with low dollar volume).
    """
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
    cat_cum_sum = cat_grp.cumsum()
    cat_prior_sum = cat_cum_sum - df["amount_bdt"]
    cat_prior_cnt = cat_grp.cumcount().astype("float64")
    cat_prior_mean = cat_prior_sum / cat_prior_cnt.replace(0, np.nan)
    df["amt_vs_cat_mean_prior"] = df["amount_bdt"] / (cat_prior_mean + EPS)

    # Location-level amount ratio
    loc_grp = df.groupby(loc_filled)["amount_bdt"]
    loc_cum_sum = loc_grp.cumsum()
    loc_prior_sum = loc_cum_sum - df["amount_bdt"]
    loc_prior_cnt = loc_grp.cumcount().astype("float64")
    loc_prior_mean = loc_prior_sum / loc_prior_cnt.replace(0, np.nan)
    df["loc_amt_ratio"] = df["amount_bdt"] / (loc_prior_mean + EPS)

    # Rolling 24h spending pace ratios & velocity acceleration
    if "cust_cnt_24h" in df.columns and "cust_amtsum_24h" in df.columns:
        cust_24h_mean = df["cust_amtsum_24h"] / (df["cust_cnt_24h"] + EPS)
        df["cust_amt_vs_24h_mean"] = df["amount_bdt"] / (cust_24h_mean + EPS)
        df["cust_burst_accel"] = (df["cust_cnt_1h"] * 24.0) / (df["cust_cnt_24h"] + 1.0)
    if "dev_cnt_24h" in df.columns and "dev_amtsum_24h" in df.columns:
        dev_24h_mean = df["dev_amtsum_24h"] / (df["dev_cnt_24h"] + EPS)
        df["dev_amt_vs_24h_mean"] = df["amount_bdt"] / (dev_24h_mean + EPS)
        df["dev_burst_accel"] = (df["dev_cnt_1h"] * 24.0) / (df["dev_cnt_24h"] + 1.0)

    # Categorical interaction crosses (coarse attributes)
    df["pay_x_dev"] = df["payment_method"].astype(str) + "_" + df["device_type"].astype(str)
    df["cat_x_loc"] = df["merchant_category"].astype(str) + "_" + df["location"].astype(str)
    df["txn_x_pay"] = df["transaction_type"].astype(str) + "_" + df["payment_method"].astype(str)

    rows_so_far = np.arange(len(df), dtype="float64")
    for col in ["pay_x_dev", "cat_x_loc", "txn_x_pay"]:
        filled = df[col].fillna("__unknown__")
        prior_cnt = filled.groupby(filled).cumcount().astype("float64")
        df[f"{col}_freq_share_prior"] = prior_cnt / np.maximum(rows_so_far, 1.0)

    if "cust_cnt_1h" in df.columns and "cust_amtsum_1h" in df.columns:
        df["cust_smurf_ratio_1h"] = (df["cust_cnt_1h"] + 1.0) / (df["cust_amtsum_1h"] + 10.0)
    if "dev_cnt_1h" in df.columns and "dev_amtsum_1h" in df.columns:
        df["dev_smurf_ratio_1h"] = (df["dev_cnt_1h"] + 1.0) / (df["dev_amtsum_1h"] + 10.0)

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

    # Phase 3.1/3.2: sub-hour and intermediate horizons + amount-vs-recent
    # ratios. Customer and device get the full ladder (both are in the family
    # that strengthens in July); merchant gets fewer, to bound runtime.
    # Window ladder matched to each entity's traffic density. A 5-minute
    # CUSTOMER window is empty 99.6% of the time (customers average ~25
    # transactions across 8 months), so those columns are almost pure NaN and
    # only dilute feature sampling. Devices and merchants carry enough traffic
    # for sub-hour horizons -- and device short-windows are precisely the
    # family the July audit found strengthening.
    df = add_recent_window_stats(df, "customer_id", "cust", windows=("30min", "3h", "12h"))
    df = add_recent_window_stats(df, "device_id", "dev", windows=("5min", "15min", "30min", "3h", "12h"))
    df = add_recent_window_stats(df, "merchant_id", "merch", windows=("5min", "30min", "3h"))

    # Phase 3.3: recent counterparty expansion (the non-saturating version of
    # the fan-out features whose lifetime form decays into noise by July).
    df = add_recent_diversity_features(df, "device_id", "customer_id", "dev")
    df = add_recent_diversity_features(df, "customer_id", "device_id", "cust")
    df = add_recent_diversity_features(df, "customer_id", "merchant_id", "cust")
    df = add_recent_diversity_features(df, "customer_id", "location", "cust")

    # Self-relative features -- must run AFTER the trailing windows, since
    # the velocity ratios normalise those counts by each entity's own rate.
    for entity_col, prefix in [("customer_id", "cust"), ("merchant_id", "merch"), ("device_id", "dev")]:
        df = add_self_relative_features(df, entity_col, prefix)
    df = add_hour_profile_features(df, "customer_id", "cust")
    df = add_location_and_category_context_features(df)

    # Stretch goal (PLAN.md): second-order relationship/cluster features.
    df = add_bipartite_component_features(df, "customer_id", "device_id", "cd")
    df = add_bipartite_component_features(df, "customer_id", "merchant_id", "cm")

    # Defragment: ~130 individual column inserts leave the block manager
    # badly fragmented, which slows every downstream slice.
    return df.copy()


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
