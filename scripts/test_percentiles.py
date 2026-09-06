"""Self-check for add_time_local_percentiles: correctness + no same-day leakage."""
import numpy as np
import pandas as pd
from src.features import add_time_local_percentiles


def demo():
    # 4 days x 5 rows. Values chosen so the expected percentile is hand-checkable.
    days = pd.to_datetime(["2026-01-01"] * 5 + ["2026-01-02"] * 5 +
                          ["2026-01-03"] * 5 + ["2026-01-04"] * 5)
    v = np.array([0., 1., 2., 3., 4.,      # day 0: no prior -> all NaN
                  0., 1., 2., 3., 4.,      # day 1: ref = day 0
                  10., 10., 10., 10., 10., # day 2: ref = days 0-1, all above -> 1.0
                  -5., 2.5, 100., np.nan, 0.])
    df = pd.DataFrame({"timestamp": days, "x": v})
    out = add_time_local_percentiles(df, ["x"], window_days=14, suffix="_pct")
    p = out["x_pct"].to_numpy()

    assert np.isnan(p[:5]).all(), "day 0 has no prior day, must be NaN"
    # day 1 ref is day 0 = [0,1,2,3,4]; searchsorted 'left' -> strictly-less fraction
    assert np.allclose(p[5:10], [0.0, 0.2, 0.4, 0.6, 0.8]), p[5:10]
    # day 2: every value above the whole reference
    assert np.allclose(p[10:15], 1.0), p[10:15]
    # day 3 ref = days 0,1,2 = [0,1,2,3,4]*2 + [10]*5 (15 values)
    assert p[15] == 0.0, "below the reference minimum"
    assert np.isclose(p[16], 6 / 15), p[16]          # 6 values strictly < 2.5
    assert p[17] == 1.0, "above the reference maximum"
    assert np.isnan(p[18]), "NaN input must stay NaN, not become the 100th pct"
    assert np.isclose(p[19], 0.0), p[19]

    # The leakage check that matters: a row's own day must never enter its
    # reference. Move one day's values to absurd numbers and the OTHER days'
    # percentiles must not budge.
    df2 = df.copy()
    df2.loc[10:14, "x"] = 1e9
    p2 = add_time_local_percentiles(df2, ["x"], window_days=14, suffix="_pct")["x_pct"].to_numpy()
    assert np.allclose(p2[10:15], 1.0), "day 2 still above its own (unchanged) reference"
    assert np.array_equal(np.isnan(p[:10]), np.isnan(p2[:10])) and np.allclose(
        p[5:10], p2[5:10]), "earlier days must be untouched by a later day's values"

    # Window must actually expire: with window_days=1, day 3 sees only day 2.
    p3 = add_time_local_percentiles(df, ["x"], window_days=1, suffix="_pct")["x_pct"].to_numpy()
    assert p3[16] == 0.0 and p3[17] == 1.0, "day 3 vs day 2 ([10]*5) only"
    print("add_time_local_percentiles: all checks pass")


if __name__ == "__main__":
    demo()
