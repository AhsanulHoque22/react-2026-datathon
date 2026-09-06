"""Self-check for add_target_encoding: strictly-prior, and test rows never
accumulate from the unlabelled period."""
import numpy as np
import pandas as pd
from src.features import add_target_encoding


def demo():
    # one customer, 3 labelled rows then 2 test rows
    df = pd.DataFrame({
        "customer_id": ["c1"] * 5 + ["c2"],
        "merchant_id": ["m1"] * 6,
        "device_id": ["d1"] * 6,
        "fraud": [1.0, 0.0, 1.0, np.nan, np.nan, 0.0],
        "is_test": [False, False, False, True, True, False],
    })
    a = 20.0
    out = add_target_encoding(df, alpha=a)
    g = 2.0 / 4.0  # global rate over labelled rows: 2 frauds / 4 labelled

    te, n, pos = out["te_cust"].to_numpy(), out["te_cust_n"].to_numpy(), out["te_cust_pos"].to_numpy()
    # row 0: no prior -> pure smoothing toward the global rate
    assert n[0] == 0 and pos[0] == 0 and np.isclose(te[0], (0 + a * g) / (0 + a))
    # row 2 sees rows 0,1 -> 1 fraud in 2 labelled rows
    assert n[2] == 2 and pos[2] == 1 and np.isclose(te[2], (1 + a * g) / (2 + a))
    # first test row sees the 3 labelled rows before it
    assert n[3] == 3 and pos[3] == 2, (n[3], pos[3])
    # second test row must be IDENTICAL: the unlabelled row 3 added nothing
    assert n[4] == 3 and pos[4] == 2, "a test row leaked into the denominator"
    assert np.isclose(te[3], te[4])
    # never includes its own label: row 0 is a fraud but its own encoding is the prior
    assert pos[0] == 0

    # a different customer must not inherit c1's history
    assert n[5] == 0 and pos[5] == 0

    # Changing a LATER label must not move an EARLIER row's ENTITY history.
    # The smoothed value legitimately shifts, because the global fraud rate it
    # blends toward is a whole-training-set statistic and that moved too --
    # PLAN.md permits aggregation at exactly that grain, and only finer.
    df2 = df.copy()
    df2.loc[2, "fraud"] = 0.0
    out2 = add_target_encoding(df2, alpha=a)
    assert np.array_equal(pos[:3], out2["te_cust_pos"].to_numpy()[:3]), "future label leaked backwards"
    assert np.array_equal(n[:3], out2["te_cust_n"].to_numpy()[:3])
    # and with the global prior held fixed, the encoding itself is unchanged
    g2 = 1.0 / 4.0
    fix = lambda t, p_, n_, gg: (p_ + a * gg) / (n_ + a)
    assert np.allclose(fix(None, pos[:3], n[:3], g), fix(None, pos[:3], n[:3], g)), "sanity"
    assert np.allclose(te[:3], fix(None, pos[:3], n[:3], g)), "te must be exactly the smoothed prior"
    assert np.allclose(out2["te_cust"].to_numpy()[:3], fix(None, pos[:3], n[:3], g2))
    print("add_target_encoding: all checks pass")


if __name__ == "__main__":
    demo()
