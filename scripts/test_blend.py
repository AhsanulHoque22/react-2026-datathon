"""Self-check for blocked_loo_blend."""
import numpy as np
import pandas as pd
from src.model import blocked_loo_blend


def demo():
    ts = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03",  # block 0
                         "2026-01-20", "2026-01-21",                # c1, block 2
                         "2026-01-01"])                             # c2, singleton
    cid = ["c1", "c1", "c1", "c1", "c1", "c2"]
    p = np.array([0.1, 0.2, 0.6, 0.9, 0.5, 0.4])
    w = 0.05
    out = blocked_loo_blend(p, cid, ts, w=w, block_days=7)

    # c1 block 0 = rows 0,1,2. LOO mean for row 0 is (0.2+0.6)/2 = 0.4
    assert np.isclose(out[0], 0.95 * 0.1 + 0.05 * 0.4), out[0]
    assert np.isclose(out[1], 0.95 * 0.2 + 0.05 * ((0.1 + 0.6) / 2))
    assert np.isclose(out[2], 0.95 * 0.6 + 0.05 * ((0.1 + 0.2) / 2))
    # rows 3,4 are a DIFFERENT block of the same customer -- must not mix with 0-2
    assert np.isclose(out[3], 0.95 * 0.9 + 0.05 * 0.5), out[3]
    assert np.isclose(out[4], 0.95 * 0.5 + 0.05 * 0.9)
    # c2 is a singleton: unchanged
    assert np.isclose(out[5], 0.4)

    # blocks must actually separate: with a huge block everything pools
    big = blocked_loo_blend(p, cid, ts, w=w, block_days=3650)
    assert not np.isclose(big[0], out[0]), "block_days had no effect"

    # w=0 is the identity
    assert np.allclose(blocked_loo_blend(p, cid, ts, w=0.0), p)
    # output stays in [0,1] for inputs in [0,1] (it is a convex combination)
    assert (out >= 0).all() and (out <= 1).all()

    # the regression this test exists for: a datetime Series input must work
    # (timedelta Series needs .dt.days, not .days)
    out_series = blocked_loo_blend(p, pd.Series(cid), pd.Series(ts), w=w)
    assert np.allclose(out, out_series)
    # and a plain numpy datetime64 array must work too
    assert np.allclose(out, blocked_loo_blend(p, np.array(cid), ts.to_numpy(), w=w))
    print("blocked_loo_blend: all checks pass")


if __name__ == "__main__":
    demo()
