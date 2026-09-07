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
