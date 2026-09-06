"""Load train/test and build one time-sorted combined frame.

PLAN.md correctness requirements implemented here:
- Deterministic tie-break sort by (timestamp, transaction_id) before any
  expanding/rolling operation (train alone has ~20K duplicate timestamps).
- train + test concatenated into one frame (non-label facts only) so entity
  history threads correctly across the train->test boundary. The `fraud`
  label is NaN for test rows and must never be used to build features.
"""
import numpy as np
import pandas as pd

from src.config import TRAIN_CSV, TEST_CSV, TIME_COL, LABEL_COL


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
