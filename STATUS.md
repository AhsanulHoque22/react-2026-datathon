# REACT 2026 Datathon — Team Status Dashboard

Last updated: **2026-09-06 16:15 Dhaka** · deadline **2026-09-07 23:59:59 Dhaka** (~31.7h left)

> Modeling plan and rationale: [`PLAN.md`](./PLAN.md) · Onboarding tasks: [`TEAM_TASKS.md`](./TEAM_TASKS.md)

---

## 🔒 Current best — HOLD

| | |
|---|---|
| **Public LB score** | **0.53948** |
| **Position** | 6th of 13 |
| **Submitted** | 2026-09-06 14:07 Dhaka |
| **Artifact** | `submissions/BEST_0.53948_2329b1b.csv` (md5 `334ce114e8f761538871e691ce0227f4`) |
| **Code** | commit `2329b1b`, git tag **`best-0.53948`** |
| **Reproduce** | `git checkout best-0.53948 && python scripts/01_build_features.py && python scripts/04_train_final_and_submit.py` |

Exact configuration that produced it: 124-column feature set, LightGBM
`lr=0.05, num_leaves=127→63, min_data_in_leaf=50, feature_fraction=0.85,
bagging_fraction=0.85`, early-stopping patience 100, **no `scale_pos_weight`**,
predictions averaged over 5 seeds, rounds picked from a Jul 1–15 held-out probe.

> ⚠️ `HEAD` has moved past this (now `lr=0.02, leaves=127`, patience 200). The
> tag is the recovery point. Nothing gets submitted without explicit approval.

## Leaderboard

| # | Team | Score |
|---|---|---|
| 1 | Eldians | 0.55283 |
| 2 | AL_Masaar | 0.55094 |
| 3 | DataX | 0.54896 |
| 4 | PHANTOM TROUPE | 0.54707 |
| 5 | Paradox | 0.54157 |
| **6** | **Overfit & Overcaffeinated** | **0.53948** |
| 7 | Hotath Atonko | 0.52932 |
| 8 | Sabr | 0.52748 |
| 9 | Scuba Scubaa | 0.52130 |
| 10 | Team Optima | 0.51967 |

Gap to 1st: **0.0134**. The whole field sits in 0.497–0.553, so this is a
tight race where ~0.01 is worth several places.

## Submission budget

**0 remaining today.** Kaggle resets at **00:00 UTC = 06:00 Dhaka**, not local
midnight (our 14:07 Dhaka submission is logged 08:07 UTC). Next 5 slots open
**06:00 Sep 7**, leaving ~18h of competition after that. Today's 5 went:
baseline, full pipeline, improved features, the class-weight fix, plus one
teammate run (0.46350).

## Score history

| Score | Change |
|---|---|
| 0.16644 | Insurance baseline — raw fields only |
| 0.51309 | Full behavioural pipeline |
| 0.51055 | +log-amount z-score, location novelty, graph features (tied w/ prev — within noise) |
| **0.53948** | **Removed `scale_pos_weight` + 5-seed averaging** |

## The one big win, and why it was hidden

`scale_pos_weight` was prescribed in PLAN.md from round 1 for the 1.76%
imbalance, and **every** hyperparameter sweep varied leaves/learning-rate/
regularization while holding it fixed — so it went untested for most of the
competition. PR-AUC is a **rank** metric; upweighting positives 55× distorts
the ranking it is scored on. Monotonic evidence on fold 3:

```
spw=55 → 0.5045    spw=10 → 0.5069    spw=7.4 → 0.5091    spw=1 → 0.5116 ± 0.0020
```

Worth **+0.026 on the leaderboard** — far more than the +0.007 validation
delta suggested, because a two-month test window punishes ranking distortion
much harder than a two-week fold does.

## Measurement discipline (learned the hard way)

**Seed-noise std on fold 3 = 0.0020.** Measured only after seven consecutive
experiments — CatBoost, the first hyperparameter sweep, recency weighting,
graph features, stationarity fixes, ratio features, expanded windows — all
produced deltas of 0.0015–0.002, i.e. **at or below the noise floor and never
actually measurable**. Conclusions were being drawn from noise.

**Rule going forward:** no result counts without either a multi-seed mean or
an effect comfortably above 0.002.

**Also: select on fold 3, not the 4-fold mean.** Fold 3 (Jul 2–15) predicted
the leaderboard almost exactly (0.5133 vs 0.51055 at the time). The mean
(~0.73) is dominated by three easy-regime folds that look nothing like the
test window.

## Settled — do not revisit

| Thing | Verdict |
|---|---|
| CatBoost (alone or blended) | Lost on every fold; no blend weight beat LightGBM alone |
| Recency weighting | Dead. Retested *fairly* (post-change data in train): hurts monotonically — none 0.5449, 60d 0.5401, 30d 0.5361, 14d 0.5334, 7d 0.5308 |
| Ensembling / rank-averaging | Nothing: best single 0.5467, prob-avg 0.5473, rank-avg 0.5474 — all inside noise |
| Isotonic calibration | Cut — cannot help a rank metric, can hurt via tie-collapsing |
| Graph/cluster features | Net-neutral; kept by team decision, not because they help |
| Log-transform duplicates | Removed — trees are invariant to monotonic transforms; they only stole `feature_fraction` slots |

## In flight (Kaggle, all compute runs there now)

- **`react-2026-pipeline`** — regenerating a candidate with `lr=0.02, leaves=127`
  (+0.0035 on the tail window, 0.5230 ± 0.0006 vs 0.5195 ± 0.0012). **Will not
  be submitted without approval.**
- **`react-2026-sweep`** (stage 2) — `min_data_in_leaf` × `feature_fraction` at
  the new lr/leaves, which stage 1 held fixed and which matter more now that
  we train ~940 rounds instead of ~140.

Kaggle kernels are generated from `src/` by `scripts/build_kaggle_kernel.py`,
so they can't drift from local source. Gotcha recorded: the API mounts
competition data at `/kaggle/input/competitions/<slug>/`, not
`/kaggle/input/<slug>/`.

## Open decision — yours

**Per-entity target encoding** (`te_customer`, `te_device`, `te_merchant`:
smoothed, past-only historical fraud rates). The one documented higher-scoring
approach we've seen had these as its 4th/6th/11th most important features, and
it is the only signal our model structurally cannot reach — we withhold both
raw IDs and any label-derived entity statistic.

- Organizer bans target encoding *"using **test.csv**"* — a train-only,
  past-only encoding isn't covered by that clause.
- But: *"targeting specific **entity IDs** rather than behavioural patterns…
  may be flagged during reproducibility review."* That's a human judgment call.
- Our own PLAN.md ban is **broader than the organizer's text** (we also banned
  "per-subgroup"), a deliberately conservative choice we made ourselves.

Plausibly worth part of the 0.0134 gap to 1st; risk is forfeiting a top-15
slot at reproducibility review. **I can measure the exact payoff without
submitting anything** — building a feature locally is not a rules violation,
only submitting a model that uses it would be.

## Next

Holding at 0.53948 until midnight. Then ~8h of work on your plans, scored
**locally only** — no submissions without your say-so.
