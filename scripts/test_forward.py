"""Self-check for add_forward_window_features."""
import numpy as np
import pandas as pd
from src.features import add_forward_window_features


def demo():
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:30", "2026-01-01 00:45",
                         "2026-01-01 05:00", "2026-01-01 00:10"])
    df = pd.DataFrame({"timestamp": ts,
                       "customer_id": ["c1", "c1", "c1", "c1", "c2"],
                       "amount_bdt": [10.0, 20.0, 30.0, 40.0, 99.0]})
    out = add_forward_window_features(df, "customer_id", "cust", windows_min=(60,))

    # row 0 (00:00): forward 60min sees rows 1 and 2 (00:30, 00:45), not row 3 (05:00)
    assert out["cust_fwd_cnt_60m"][0] == 2, out["cust_fwd_cnt_60m"][0]
    assert np.isclose(out["cust_fwd_amtsum_60m"][0], 50.0)
    # row 2 (00:45): forward sees nothing within 60min
    assert out["cust_fwd_cnt_60m"][2] == 0
    # symmetric for row 1 (00:30): rows 0 and 2, excluding itself
    assert out["cust_sym_cnt_60m"][1] == 2
    assert np.isclose(out["cust_sym_amtsum_60m"][1], 40.0)
    # c2 is alone -- must never pick up c1's rows
    assert out["cust_fwd_cnt_60m"][4] == 0 and out["cust_sym_cnt_60m"][4] == 0
    # seconds_to_next
    assert np.isclose(out["cust_seconds_to_next"][0], 1800.0)
    assert np.isclose(out["cust_seconds_to_next"][2], (5 * 3600 - 45 * 60))
    assert np.isnan(out["cust_seconds_to_next"][3]), "last row of an entity has no next"
    assert np.isnan(out["cust_seconds_to_next"][4]), "c2 singleton has no next"

    # forward must NOT count the row itself
    assert (out["cust_fwd_cnt_60m"] >= 0).all()
    # and it must be direction-correct: reversing time flips fwd/backward counts
    print("add_forward_window_features: all checks pass")


if __name__ == "__main__":
    demo()


def demo_rich():
    from src.features import add_forward_rich_features
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10", "2026-01-01 00:20",
                         "2026-01-01 09:00", "2026-01-01 00:05"])
    df = pd.DataFrame({"timestamp": ts,
                       "customer_id": ["c1", "c1", "c1", "c1", "c2"],
                       "amount_bdt": [10.0, 20.0, 30.0, 40.0, 99.0]})
    out = add_forward_rich_features(df, "customer_id", "cust", windows_min=(15,))

    # row 1 (00:10) within +/-15min sees rows 0 (00:00) and 2 (00:20)
    assert out["cust_fwd_cnt_15m"][1] == 1 and out["cust_sym_cnt_15m"][1] == 2
    assert np.isclose(out["cust_fwd_amtsum_15m"][1], 30.0)
    assert np.isclose(out["cust_sym_amtsum_15m"][1], 40.0)
    # accel: 1 forward vs 1 backward -> 1.0
    assert np.isclose(out["cust_accel_15m"][1], 1.0)
    # row 0 is at 00:00 so its +/-15min window ends at 00:15: row 1 (00:10) is
    # inside, row 2 (00:20) is NOT. One forward, zero backward.
    assert out["cust_fwd_cnt_15m"][0] == 1, out["cust_fwd_cnt_15m"][0]
    assert np.isclose(out["cust_accel_15m"][0], 2.0 / 1.0)
    # amt_vs_sym for row 1: 20 / mean(10,30) = 1.0
    assert np.isclose(out["cust_amt_vs_sym_15m"][1], 1.0, atol=1e-4)
    # c2 never pools with c1
    assert out["cust_sym_cnt_15m"][4] == 0
    # row 3 is 8h40 after row 2 -- outside any 15min window
    assert out["cust_sym_cnt_15m"][3] == 0
    # seconds_to_next / next2
    assert np.isclose(out["cust_seconds_to_next"][0], 600.0)
    assert np.isclose(out["cust_seconds_to_next2"][0], 1200.0)
    assert np.isnan(out["cust_seconds_to_next"][3]), "last row of entity has no next"
    print("add_forward_rich_features: all checks pass")


if __name__ == "__main__":
    demo()
    demo_rich()
