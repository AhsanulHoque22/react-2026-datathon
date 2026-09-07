"""Self-check for blocked_loo_blend (hour-based blocks)."""
import numpy as np
import pandas as pd
from src.model import blocked_loo_blend


def demo():
    # c1 has three transactions inside one 24h block, then two in a later block.
    ts = pd.to_datetime([
        "2026-01-01 01:00", "2026-01-01 09:00", "2026-01-01 20:00",   # c1, block 0
        "2026-01-05 03:00", "2026-01-05 10:00",                        # c1, block 4
        "2026-01-01 05:00",                                            # c2, singleton
    ])
    cid = ["c1", "c1", "c1", "c1", "c1", "c2"]
    p = np.array([0.1, 0.2, 0.6, 0.9, 0.5, 0.4])
    w = 0.1
    out = blocked_loo_blend(p, {"c": cid}, ts, w=w, block_hours=24)

    assert np.isclose(out[0], 0.9 * 0.1 + 0.1 * ((0.2 + 0.6) / 2)), out[0]
    assert np.isclose(out[1], 0.9 * 0.2 + 0.1 * ((0.1 + 0.6) / 2))
    assert np.isclose(out[2], 0.9 * 0.6 + 0.1 * ((0.1 + 0.2) / 2))
    # a later block of the SAME customer must not pool with the earlier one
    assert np.isclose(out[3], 0.9 * 0.9 + 0.1 * 0.5), out[3]
    assert np.isclose(out[4], 0.9 * 0.5 + 0.1 * 0.9)
    # c2 is alone in its block: unchanged
    assert np.isclose(out[5], 0.4)

    # blocks must bite: one huge block pools everything, so row 0 moves
    big = blocked_loo_blend(p, {"c": cid}, ts, w=w, block_hours=24 * 3650)
    assert not np.isclose(big[0], out[0]), "block size had no effect"
    # and a sub-day block splits the first group apart
    tiny = blocked_loo_blend(p, {"c": cid}, ts, w=w, block_hours=4)
    assert np.isclose(tiny[0], 0.1), "4h block should leave row 0 a singleton"

    assert np.allclose(blocked_loo_blend(p, {"c": cid}, ts, w=0.0), p)
    assert (out >= 0).all() and (out <= 1).all()

    # regression: timedelta Series needs .dt, and every input shape must agree
    assert np.allclose(out, blocked_loo_blend(p, {"c": pd.Series(cid)}, pd.Series(ts), w=w))
    assert np.allclose(out, blocked_loo_blend(p, {"c": np.array(cid)}, ts.to_numpy(), w=w))
    assert np.allclose(out, blocked_loo_blend(p, cid, ts, w=w))          # bare array
    two = blocked_loo_blend(p, {"a": cid, "b": ["x"] * 6}, ts, w=w)
    assert not np.allclose(two, out), "second entity had no effect"
    print("blocked_loo_blend: all checks pass")


if __name__ == "__main__":
    demo()


def demo_sliding():
    from src.model import sliding_loo_blend
    ts = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:20", "2026-01-01 11:30",
                         "2026-01-01 10:10"])
    cid = ["c1", "c1", "c1", "c2"]
    p = np.array([0.1, 0.5, 0.9, 0.4])
    # +/-30min window => half-width 15min. Rows 0 and 1 are 20min apart, so
    # NEITHER sees the other; everything is a singleton.
    out = sliding_loo_blend(p, {"c": cid}, ts, w=0.5, window_minutes=30)
    assert np.allclose(out, p), out
    # +/-60min => half 30min: rows 0,1 see each other; row 2 is 70min from row 1
    out = sliding_loo_blend(p, {"c": cid}, ts, w=0.5, window_minutes=60)
    assert np.isclose(out[0], 0.5 * 0.1 + 0.5 * 0.5), out[0]
    assert np.isclose(out[1], 0.5 * 0.5 + 0.5 * 0.1), out[1]
    assert np.isclose(out[2], 0.9), "row 2 is outside the window of both"
    assert np.isclose(out[3], 0.4), "c2 must never pool with c1"
    # w=0 identity, and a huge window pools the whole entity
    assert np.allclose(sliding_loo_blend(p, {"c": cid}, ts, w=0.0, window_minutes=60), p)
    big = sliding_loo_blend(p, {"c": cid}, ts, w=0.5, window_minutes=10000)
    assert np.isclose(big[0], 0.5 * 0.1 + 0.5 * ((0.5 + 0.9) / 2)), big[0]
    # order independence: shuffling rows must not change each row's result
    idx = np.array([2, 0, 3, 1])
    sh = sliding_loo_blend(p[idx], {"c": [cid[i] for i in idx]}, ts[idx], w=0.5, window_minutes=60)
    assert np.allclose(sh, out[idx]), "result depends on input row order"
    print("sliding_loo_blend: all checks pass")


if __name__ == "__main__":
    demo_sliding()
