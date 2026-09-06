# REACT 2026 Datathon — Team Status Dashboard

Last updated: **2026-09-06 18:30 Dhaka** · deadline **2026-09-07 23:59:59 Dhaka** (~29.5h left)

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
| Graph/cluster features | Net-neutral across every test; **dropped** in the final arm C on parsimony (removing them cost −0.0004, i.e. nothing) |
| Log-transform duplicates | Removed — trees are invariant to monotonic transforms; they only stole `feature_fraction` slots |

## Local scores — current standing (all Kaggle-run, 3 seeds each)

Selection is on the **Jul 1–15 tail / fold 3**, never the 4-fold mean.
Seed-noise std is **0.0020**, so anything under ~0.004 is not a result.

| Configuration | fold3 | tail (Jul 1–15) | fold2 (guard) |
|---|---|---|---|
| Previous best (`lr=0.05/63`) — **the 0.53948 submission** | 0.5133 | 0.5204 | 0.7919 |
| Swept (`lr=0.02/127`) | 0.5158 | 0.5233 | 0.7932 |
| Swept + 17 self-relative features | — | 0.5233 | 0.7950 |
| **Arm C — swept + all Phase-3 feats − graph** | — | **0.5252** | 0.7930 |

### Aggregation run (`react-2026-final`) — 3 arms × 4 windows × 3 seeds

| Arm | Jun25–Jul02 | Jul02–Jul08 | Jul08–Jul16 | fold2-guard | recent mean |
|---|---|---|---|---|---|
| A — prev best (`lr.05/63`, old feats) | 0.7704 | 0.4862 | 0.5419 | 0.7908 | 0.5995 |
| B — new feats (`lr.02/127`) | 0.7708 | 0.4935 | 0.5474 | 0.7935 | 0.6039 |
| **C — B minus graph features** | 0.7701 | **0.4943** | **0.5484** | 0.7930 | **0.6043** |

All ± are ≤0.003. Both B and C clear the acceptance rule against A
(+0.0044 / +0.0047 recent-mean, no window worse than −0.0004). **B vs C is
+0.0004 — noise**; C was taken because it is the smaller model, not because
graph features were shown to hurt. That is consistent with every prior graph
measurement: net-neutral. They are now dropped on parsimony.

Winner refit on the full window: **held-out Jul 1–15 = 0.5252, 967 rounds**.
Archived as `submissions/CANDIDATE_C_nograph_0.5252.csv` — verified 262,648
rows, order matches `sample_submission.csv`, 0 NaN, range [0.00016, 0.99995].
Rank overlap with the 0.53948 submission is 77% at top-1k and 91% at top-3k
(global Spearman is only 0.49, but that is tail reshuffling among near-zero
rows and does not touch PR-AUC). **Not submitted.**

Net gain over the live submission's configuration: **+0.0048 on the Jul 1–15
tail**, ~2.4× the 0.0020 seed-noise std.

**Caveat on attribution:** arm C differs from A in *two* ways at once —
hyperparameters and the 85 new Phase-3 features — so the +0.0047 is the
combined effect, not the features' own. Decomposed against the isolated sweep
result (tail 0.5204 → 0.5233), hyperparameters carry ~+0.0029 and the feature
block ~+0.0019. The feature half alone is under the noise floor; it is only
credible because it holds the same sign across three independent recent
windows, which is what the acceptance rule was written to catch.

### Confirmed wins
- **Swept hyperparameters** `lr=0.02, num_leaves=127`: tail 0.5204 → **0.5233**,
  fold3 0.5133 → **0.5158**. Candidate archived as
  `submissions/CANDIDATE_swept_lr002_leaves127.csv` (verified: 262,648 rows,
  order matches sample, no NaN, all in [0,1]). **Not submitted.**

### Negative results (do not retry)
- **Self-relative features** (gap acceleration, velocity ratios, personal-record
  amounts, customer hour-bucket profile) — 17 features, **+0.0003 on the tail**,
  i.e. pure noise, and variance rose. The +0.0018 on fold2 is real but lands on
  the easy regime that does not predict the leaderboard. Fails the acceptance
  rule (needs ≥0.004 on recent windows).

## Working the fold-3 improvement plan

Two of its prescriptions were **already superseded** and are deliberately not
followed:
- *Phase 0.1* fixes `lr=0.05/leaves=63` as standard — the Kaggle re-sweep since
  measured `lr=0.02/leaves=127` better on both the tail and fold2.
- *Phase 5*'s recency specialist is **already done exactly as specified** (Jul 8
  split so training contains post-change data, no weighting, 3 seeds) and is
  negative: no weighting 0.5449 beats every half-life, monotonically. That also
  removes the specialist half of Phase 6's blend.

Implemented: Phase 3.1 (sub-hour/intermediate horizons, **matched to entity
traffic density** — a 5-minute *customer* window is empty 99.6% of the time),
3.2 (amount vs recent mean/max), 3.3 (recent counterparty expansion, the
non-saturating replacement for lifetime fan-out).

**Bug found while implementing:** pandas rolling `.count()` returns NaN for an
empty window, so "zero transactions in the last hour" — real information — was
encoded as *unknown*, including in the pre-existing `cnt_1h`/`amtsum_1h`
features. Counts/sums now zero-fill; ratios correctly stay NaN.

### Stage-2 sweep (`react-2026-sweep`) — `min_data_in_leaf` x `feature_fraction`

12 configs x 2 windows x 3 seeds at the fixed `lr=0.02 / num_leaves=127`.
**The incumbent `mdl=50, ff=0.85` is already the grid optimum**, so nothing
changes.

| min_data_in_leaf | ff=0.50 | ff=0.70 | ff=0.85 |
|---|---|---|---|
| 20 | 0.5212 | 0.5225 | 0.5219 |
| **50** | 0.5217 | 0.5216 | **0.5230** |
| 200 | 0.5217 | 0.5214 | 0.5225 |
| 500 | 0.5215 | 0.5212 | 0.5202 |

(Jul 1-15 tail, 3-seed means, per-cell std 0.0002-0.0009.)

The entire grid spans **0.0028**, barely above one seed-std, and the top four
cells are separated by 0.0005. Read honestly this is a **flat surface, not a
ranking** — `mdl=50/ff=0.85` "winning" is not distinguishable from
`mdl=20/ff=0.70`. The useful conclusion is the negative one: these two knobs
have nothing left to give, and the +0.002-0.004 they were hoped to add does
not exist. Only `mdl=500` is clearly bad (over-regularised: 453 rounds at
ff=0.85 vs ~940).

### Forecast-horizon decay (`react-2026-horizon`)

Validation window held fixed at Jul 1-16; the training cutoff walks backwards
so the train-to-prediction gap grows. 3 seeds per row.

| train ends | gap (days) | PR-AUC | best_iter |
|---|---|---|---|
| 2026-07-01 | 0 | 0.5256 | 911 |
| 2026-06-24 | 7 | 0.5259 | 768 |
| 2026-06-17 | 14 | 0.5243 | 676 |
| 2026-06-01 | 30 | 0.5209 | 1097 |
| 2026-05-17 | 45 | 0.5144 | 805 |
| 2026-05-02 | 60 | 0.5068 | 756 |

**Two findings, and the one we went looking for is disconfirmed.**

1. `best_iter` does **not** decay with the gap: 911 -> 768 -> 676 -> 1097 ->
   805 -> 756, a noisy band of 676-1097 with no trend. The hypothesis that
   ~967 rounds is fitted to short-horizon structure and should be cut is
   **wrong**. The final refit keeps its round count unchanged.
2. Accuracy *does* decay, and steeply: **-0.019 AP over 60 days** of staleness,
   almost all of it after day 14 (-0.0013 to day 14, then -0.017). The model
   ages fast.

Finding 2 has no fix available to us. It measures the cost of *stale training
data*, and we already train to the last row of `train.csv` -- there is no
fresher data to add. What it does mean is that the Jul 1-15 probe (gap ~0-15d)
**overstates** what late test rows will score: the test window runs ~62 days
past training, so its later half sits in the -0.015 to -0.019 region. Expect
the leaderboard to reward this model less than the local probe implies.

It also explains why recency weighting kept failing. Down-weighting old rows
does not make the model younger; it just discards signal while leaving the
staleness untouched.

### Prediction propagation (`react-2026-prop`, `prop2`)

Fraud clusters by entity, and a per-row model cannot express that. Blending
each row's score with a leave-one-out mean of its entity's other scored rows
uses only model outputs and entity ids -- no labels, nothing fitted to
test.csv.

Clustering is real (train rows, P(sibling fraud | fraud) vs the 0.0176 base):

| entity | conditional | lift |
|---|---|---|
| device_id | 0.0682 | 3.87x |
| customer_id | 0.0464 | 2.63x |
| merchant_id | 0.0134 | 0.76x |

**Bug worth remembering:** the first kernel divided fraud-fraud pairs by
`n(n-1)`, which is the *joint* probability, and printed it as the conditional.
Against the marginal that made 2.6x clustering read as "lift 0.0x" and nearly
got the whole idea discarded.

`prop2`, 4 windows x 3 seeds, customer-level:

| weight | recent mean | delta | windows up | worst | fold2 guard |
|---|---|---|---|---|---|
| 0.05 | 0.6092 | **+0.0049** | 3/3 | +0.0028 | +0.0062 |
| 0.10 | 0.6092 | **+0.0049** | 3/3 | +0.0028 | +0.0064 |
| 0.15 | 0.6076 | +0.0034 | 3/3 | +0.0022 | +0.0042 |

Cleanest ACCEPT since `scale_pos_weight`: every window up, guard up, no
negative cell anywhere. Device rejects (2/3 up, worst -0.0018) -- 28 rows per
entity pools in too much unrelated traffic. Merchant is dead, as its 0.76x
lift predicts.

**Not adopted yet, because the validation regime does not match test:**

| window | rows/customer | singletons | median day-span |
|---|---|---|---|
| val Jul02-08 | 1.8 | 62.6% | 0 days |
| val Jul08-16 | 2.0 | 57.8% | 0 days |
| **TEST (62d)** | **7.5** | **18.6%** | **36 days** |

The gain was measured where propagation is a no-op for ~60% of rows and pools
same-day siblings. On test it would touch 81% of rows and pool transactions a
month apart. `react-2026-prop3` re-validates on 60-day windows with test-like
group sizes, and tries a 7-day-blocked variant as the fallback.

## In flight

- `react-2026-te` -- target-encoding payoff (measurement only; still banned).
- `react-2026-drift` -- time-local percentile normalisation.
- `react-2026-obj` -- ranking objectives vs logloss.
- `react-2026-prop3` -- does propagation survive test-like entity groups.


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

Every queued experiment has now reported. **Final local score: 0.5252** on the
Jul 1-15 tail (arm C), vs 0.5204 for the configuration currently sitting at
0.53948 on the leaderboard — **+0.0048, ~2.4x the seed-noise std**, with the
sign holding across three independent recent windows.

The candidate is built, verified and waiting: `CANDIDATE_C_nograph_0.5252.csv`.
**Nothing has been submitted and nothing will be without your explicit say-so.**

What is left is genuinely thin. The three big knobs are now measured out:
hyperparameters are at a flat optimum, the feature block gave what it had, and
horizon decay is not fixable from our side. The one materially different lever
we have never pulled is the per-entity target encoding in the open decision
above -- still your call, and still measurable without submitting anything.
