"""Guard: the strictly-before-t rule, checked mechanically."""
import numpy as np
import pandas as pd
from src.features import assert_strictly_past, add_location_and_category_context_features


def demo():
    # the guard must catch every forward-feature family we built
    for bad in ["cust_fwd_cnt_60m", "cust_sym_cnt_60m", "dev_seconds_to_next",
                "cust_accel_15m", "cust_amt_vs_sym_60m", "dev_gap_fwd_vs_bwd"]:
        try:
            assert_strictly_past(["amount_bdt", bad])
        except AssertionError:
            continue
        raise SystemExit(f"guard FAILED to catch {bad}")
    # and must pass a clean prior-only set
    assert_strictly_past(["amount_bdt", "cust_cnt_1h", "cust_amt_robust_z",
                          "loc_cnt_24h", "amt_vs_cat_mean_prior"])

    # the ported context features must themselves be strictly past: a row's
    # value must not change when a LATER row is altered
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:30", "2026-01-01 01:00"])
    base = pd.DataFrame({"timestamp": ts, "location": ["L1"] * 3,
                         "merchant_category": ["c"] * 3, "payment_method": ["p"] * 3,
                         "device_type": ["d"] * 3, "transaction_type": ["t"] * 3,
                         "amount_bdt": [10.0, 20.0, 30.0]})
    a = add_location_and_category_context_features(base.copy())
    b2 = base.copy()
    b2.loc[2, "amount_bdt"] = 9999.0          # change only the LAST row
    b = add_location_and_category_context_features(b2)
    for c in ("loc_cnt_1h", "loc_amtsum_1h", "amt_vs_cat_mean_prior", "loc_amt_ratio"):
        assert np.allclose(np.nan_to_num(a[c].values[:2]), np.nan_to_num(b[c].values[:2])), \
            f"{c} on earlier rows moved when a later row changed -> not strictly past"
    # first row of a location has no prior window
    assert a["loc_cnt_1h"].iloc[0] == 0
    print("compliance guard + ported context features: all checks pass")


if __name__ == "__main__":
    demo()


def demo_dtypes():
    """The crosses must survive prepare_lgb_frame rather than failing at
    LightGBM Dataset construction an hour into a run."""
    from src.model import prepare_lgb_frame, get_feature_columns, CAT_COLS
    from src.features import add_location_and_category_context_features
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:30", "2026-01-01 01:00"])
    df = pd.DataFrame({"timestamp": ts, "location": ["L1", "L1", "L2"],
                       "merchant_category": ["c", "c", "d"], "payment_method": ["p", "q", "p"],
                       "device_type": ["d1", "d1", "d2"], "transaction_type": ["t", "t", "u"],
                       "amount_bdt": [10.0, 20.0, 30.0], "fraud": [0, 1, 0], "is_test": [False] * 3,
                       "customer_id": ["a", "b", "c"], "merchant_id": ["m", "m", "n"],
                       "device_id": ["x", "y", "z"], "transaction_id": ["1", "2", "3"]})
    out = add_location_and_category_context_features(df)
    for c in ("pay_x_dev", "cat_x_loc", "txn_x_pay"):
        assert c in CAT_COLS, f"{c} must be declared categorical"
    cols, cats = get_feature_columns(out)
    X = prepare_lgb_frame(out, cols, cats)
    objs = [c for c in X.columns if X[c].dtype == object]
    assert not objs, f"object dtypes survived: {objs}"

    # and the guard must fire when something slips through undeclared
    out["rogue_string_col"] = "oops"
    cols2, cats2 = get_feature_columns(out)
    try:
        prepare_lgb_frame(out, cols2, cats2)
    except AssertionError as e:
        assert "rogue_string_col" in str(e)
        print("dtype guard: all checks pass")
        return
    raise SystemExit("dtype guard FAILED to catch an object column")


def demo_backward():
    """The backward family must be strictly past: a later row must never move
    an earlier row's values, and the guard must accept the column names."""
    from src.features import add_backward_rich_features, assert_strictly_past
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10", "2026-01-01 00:20",
                         "2026-01-01 09:00", "2026-01-01 00:05"])
    df = pd.DataFrame({"timestamp": ts, "customer_id": ["c1", "c1", "c1", "c1", "c2"],
                       "amount_bdt": [10.0, 20.0, 30.0, 40.0, 99.0]})
    out = add_backward_rich_features(df, "customer_id", "cust", windows_min=(15,))
    # row 0 has no prior; row 2 (00:20) sees rows 0 and 1 within 15min? 00:20-15m
    # = 00:05, so only row 1 (00:10) qualifies -- row 0 at 00:00 is outside.
    assert out["cust_bwd_cnt_15m"][0] == 0
    assert out["cust_bwd_cnt_15m"][1] == 1, out["cust_bwd_cnt_15m"][1]
    assert out["cust_bwd_cnt_15m"][2] == 1, out["cust_bwd_cnt_15m"][2]
    assert np.isclose(out["cust_bwd_amtsum_15m"][2], 20.0)
    assert out["cust_bwd_cnt_15m"][4] == 0, "c2 must not see c1"
    assert out["cust_bwd_cnt_15m"][3] == 0, "row 3 is hours later, empty window"

    # strictly past: perturbing the LAST row cannot change any earlier row
    d2 = df.copy(); d2.loc[3, "amount_bdt"] = 9e6
    out2 = add_backward_rich_features(d2, "customer_id", "cust", windows_min=(15,))
    for c in ("cust_bwd_cnt_15m", "cust_bwd_amtsum_15m", "cust_amt_vs_bwd_15m"):
        assert np.allclose(np.nan_to_num(out[c].values[:3]),
                           np.nan_to_num(out2[c].values[:3])), f"{c} not strictly past"
    assert_strictly_past([c for c in out.columns if c.startswith("cust_")])
    print("add_backward_rich_features: all checks pass")
