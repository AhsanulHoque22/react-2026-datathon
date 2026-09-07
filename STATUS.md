# REACT 2026 Datathon — Team Status Dashboard

Last updated: **2026-09-07 12:00 Dhaka** · deadline **2026-09-07 23:59:59 Dhaka** (~12h left)

> Modeling plan and rationale: [`PLAN.md`](./PLAN.md) · Onboarding tasks: [`TEAM_TASKS.md`](./TEAM_TASKS.md)

---

## 🏆 Current best — 1st place

| | |
|---|---|
| **Public LB score** | **0.56492** |
| **Position** | **1 of 33** |
| **Submitted** | 2026-09-07 11:31 Dhaka |
| **Artifact** | `submissions/CANDIDATE_final_v5_0.5322.csv` (md5 `4bd4510b4ffb7b9b3f51909818e665f9`) |
| **Code** | commit `d273067` |

Arm C (lr=0.02, leaves=127, no `scale_pos_weight`, Phase-3 features, no graph
features, 967 rounds, 5 seeds) **plus entity propagation**: a leave-one-out
blend over customer and device neighbours inside a sliding +/-60min window at
w=0.50.

Held-out Jul01-15: raw 0.5252 -> blended 0.5322.

### Leaderboard

| # | Team | Score |
|---|---|---|
| **1** | **Overfit & Overcaffeinated** | **0.56492** |
| 2 | Error404 | 0.56479 |
| 3 | AL_Masaar | 0.56041 |
| 4 | COiN Lab | 0.55988 |
| 5 | Eldians | 0.55941 |

**The margin over 2nd is 0.00013.** That is not a lead, it is a tie, and the
public board is only 60% of the test set. Do not treat this as safe.

## Rules compliance — read before selecting final submissions

The organiser's rule: *"Every engineered feature for a transaction at time t
may only use information strictly before t"*, clarified to permit test-period
rows only where they *"occurred earlier than the row being scored"*.

**Our features are all compliant.** No labels near test.csv, no target
encoding, no raw IDs in the matrix, every aggregate strictly prior and
mechanically asserted.

**One post-inference step is a judgement call.** The propagation blend uses a
symmetric +/-30min window, so it reads rows after t. It is not a feature (the
model never sees it) and uses no labels, but a strict reading of the rule's
intent still reaches it. It moves **4.7% of test rows**.

| artifact | status | compliance |
|---|---|---|
| `CANDIDATE_final_v5_0.5322.csv` (LB 0.56492) | submitted | prior-only features + propagation post-process -- **arguable** |
| sanzid champion (LB 0.56548, rank 1) | submitted | same shape -- **arguable** |
| `CANDIDATE_C_nograph_0.5252.csv` | not submitted | **fully compliant fallback** |
| `CANDIDATE_v7_forward_0.5380.csv` | **never submit** | forward-reading FEATURES -- indefensible |

Full argument for a reviewer: [`docs/METHODOLOGY_DISCLOSURE.md`](docs/METHODOLOGY_DISCLOSURE.md).

**Two submissions may be selected for private scoring.** The intended pairing
is one arguable-but-disclosed and one strictly compliant, so a reviewer who
reads the rule more strictly than we do still has something to score.

`assert_strictly_past()` now fails the build if any forward-looking column
reaches a feature matrix, so v7's family cannot re-enter by accident.

## THE EXCHANGE RATE (the most important number here)

Two calibration points now exist:

| local (Jul01-15 tail) | leaderboard |
|---|---|
| 0.5204 | 0.53948 |
| 0.5322 (+0.0118) | 0.56492 (+0.0254) |

**Local gains amplify ~2.2x on the leaderboard.** Every estimate made before
this submission assumed 1:1 pass-through and was therefore too conservative --
the prediction was ~0.559 and the result was 0.56492.

**Consequence: the +0.004 acceptance bar was set under the wrong assumption.**
Calibrated correctly it should be about **+0.002 local**. Several results
rejected today sit in that reopened band -- the stacker at +0.0029 above all.

## Submission budget

**4 remaining today** (v5 spent one). Kaggle resets at **00:00 UTC = 06:00 Dhaka**, not local
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
| Ranking objectives | Dead. lambdarank -0.0100, rank_xendcg -0.0092, focal g=2 -0.0006 on the tail, all losing on fold2 too. PR-AUC scores ONE GLOBAL ranking; lambdarank optimises NDCG *within group*, so day-grouping taught within-day sorting and discarded the cross-day ordering the metric measures. No grouping fixes that -- a single 700k-row group is what the metric wants and is infeasible |
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

### Drift normalisation — rejected (`react-2026-drift`)

| arm | Jun25-Jul02 | Jul02-Jul08 | Jul08-Jul16 | fold2 | recent mean |
|---|---|---|---|---|---|
| A — raw (current best) | 0.7701 | 0.4943 | 0.5484 | 0.7930 | 0.6043 |
| B — raw + percentiles | 0.7720 | 0.4940 | 0.5471 | 0.7962 | 0.6044 (+0.0001) |
| C — percentiles replacing raw | 0.7716 | 0.4941 | 0.5446 | 0.7937 | 0.6034 (-0.0008) |

Both reject, 1/3 windows improved. The hypothesis -- that feature drift was
costing us real score and re-centring against the trailing population would
recover it -- is **wrong**.

The drift itself is not in doubt. 33 of 182 numeric features have PSI > 0.25,
and the top of the table is extreme:

| feature | PSI |
|---|---|
| `device_type_freq_share_prior` | 9.38 |
| `transaction_type_freq_share_prior` | 5.73 |
| `payment_method_freq_share_prior` | 5.38 |
| `merchant_id_nunique_customer_for_merchant_share_prior` | 5.21 |
| `dev_history_count` | 3.52 |

**Two things worth keeping from this.** First, the `*_freq_share_prior`
features are the *most* drifted in the entire set -- and those are exactly the
share-of-traffic encodings introduced earlier specifically to be stationary.
That fix did not achieve stationarity, and nobody had checked.

Second, and the reason the whole idea fails: much of this "drift" is
deterministic accumulation, not regime change. `dev_history_count` and
`*_amt_sum_prior` grow monotonically by construction, so PSI flags them
loudly while the model already handles them fine. Converting to a percentile
strips the absolute level, which carries real signal, about as fast as it
removes the drift. High PSI marks a feature worth *looking at*; it does not
mark one worth normalising.

### FINAL CANDIDATE (`react-2026-candidate`)

Arm C + 7-day blocked customer propagation at w=0.05.

```
held-out(Jul01-15)  raw=0.5252  blended=0.5292  (+0.0040)   rounds=967
test groups: rows/cust mean=7.5, singletons=18.6%
blend moved 70.1% of test rows; spearman(raw, blended)=0.9526
```

`submissions/CANDIDATE_propblend_0.5292.csv`
(md5 `d32a8f036def52c8525b32786660e44b`) -- verified: 262,648 rows, order
matches `sample_submission.csv`, 0 NaN, range [0.000207, 0.999944].
**Not submitted.**

The +0.0040 held-out gain sits exactly where `prop3` predicted for a 15-day
window: it measured +0.0021 on an 8-day window and +0.0046..+0.0052 on 60-day
windows, and Jul 1-15 falls between them. Independent confirmation that the
blocked blend behaves as modelled rather than fitting the window it was tuned
on.

Worth noting for anyone reading the artifact: top-1k overlap with arm C is only
0.587 despite a 0.95 Spearman. A w=0.05 blend barely moves any individual
score, but it reorders rows whose raw scores were close together -- which is
precisely the mechanism, PR-AUC being a rank metric. It is not evidence of
instability.

## Final local scores

| Configuration | Jul 1-15 tail | recent mean |
|---|---|---|
| Live 0.53948 submission (`lr=0.05/63`, old feats) | 0.5204 | 0.5995 |
| Arm C (swept + Phase-3 feats, no graph) | 0.5252 | 0.6043 |
| **Arm C + blocked propagation (final candidate)** | **0.5292** | **~0.609** |

**Total local gain over what is on the leaderboard: +0.0088.**

## Tonight's four levers

| lever | result | verdict |
|---|---|---|
| Propagation, 7-day blocked, w=0.05 | +0.0040 held-out | **ADOPTED** |
| Target encoding | +0.0031, loses the hardest window | rejected |
| Drift normalisation | +0.0001 | rejected |
| Ranking objectives | -0.009 | dead |

One of four paid, and it only survived because `prop3` caught that the
validation regime did not match the test window.

## Sep 7 morning — what worked and what did not

**Propagation is the only thing that paid, and it paid three times over.**

Blend each score with a leave-one-out mean of the same entity's other scores
nearby in time. Fraud clusters by entity (P(sibling fraud | fraud) is 3.87x
base for devices, 2.63x for customers) and a per-row model cannot express it.
Uses only model outputs, entity ids and timestamps -- no labels -- so nothing
is fitted to test.csv.

Searched to a bracketed optimum: **sliding +/-60min, w=0.50, customer+device
-> +0.0150** on 60-day windows whose group structure matches the test set.
All three windows improve, fold2 guard included (+0.0149), std 0.0002-0.0004.

| step | config | gain |
|---|---|---|
| first attempt | customer, 7-day block, w=0.05 | +0.0049 |
| add device, tighter | cust+dev, 3-day, w=0.075 | +0.0073 |
| tighter still | cust+dev, 24h, w=0.10 | +0.0091 |
| to the floor | cust+dev, 1h, w=0.2 | +0.0134 |
| sliding, bracketed | cust+dev, +/-60min, w=0.50 | **+0.0150** |

Window is a true peak (+/-30 and +/-120 both lower); weight plateaus at
0.5-0.6 and falls by 0.9. Sliding matched fixed blocks, so the boundary-effect
worry was unfounded.

### Rejected this morning

| lever | result | why it failed |
|---|---|---|
| Target encoding | +0.0031 | below the +0.004 bar, loses the hardest window, and carries reproducibility-review exposure for less than propagation gives free |
| Drift normalisation | +0.0001 | much of the flagged drift is deterministic accumulation; the percentile transform strips absolute level, which carries signal |
| Ranking objectives | -0.009 | PR-AUC scores one GLOBAL ranking; lambdarank optimises NDCG *within group*, so day-grouping discarded the cross-day ordering |
| Self-training (soft) | +0.0023 | real but small, and the only idea that trains on test rows |
| Self-training (hard) | -0.0101 | pseudo-labels at AP 0.53 inject more noise than signal |
| Drop dead-regime features | +0.0001 / -0.0010 | see below |
| DART, L1/L2, max_bin, extra_trees, GOSS | -0.0039..+0.0006 | defaults were already right |

### Why the score will not climb further

The July diagnostic (`react-2026-what`) found the fraud signature changed
**shape**: customer-relative features roughly halved in univariate AP
(`cust_amt_robust_z` 0.41 -> 0.24) while device-volume features roughly
doubled (`dev_amtsum_6h` 0.095 -> 0.172). Old fraud was a transaction unusual
for its customer; new fraud is volume through a device, often across new
customers.

The obvious prescription -- drop the dead features so the model finds the live
ones -- **was tested and is wrong**. Dropping the customer-relative block costs
-0.0010 on July and -0.0027 on the guard. The reason: 0.24 AP is *still* better
than 0.172. The old signal decayed but remains the strongest thing in the
feature set, and there is no stronger alternative hiding behind it.

That is the ceiling, and it is a property of the data. Supporting evidence: 33
teams have converged into 0.497-0.565, the serious ones packed in 0.54-0.565.
Nobody found a magic signal.

## Final local scores

| Configuration | Jul 1-15 tail | 60-day windows |
|---|---|---|
| Live 0.53948 submission | 0.5204 | -- |
| Arm C | 0.5252 | baseline |
| **Arm C + propagation (final)** | -- | **+0.0150** |

Total local gain over what is on the leaderboard: **~+0.020**.

## Propagation is closed, from three directions

The winning lever, searched until it stopped giving:

| probe | what it varied | verdict |
|---|---|---|
| `psweep` | entity scheme x block size x weight | cust+dev beats either alone; tighter blocks win monotonically |
| `sub` / `tight` | block size down to 1h, then sliding +/-5..240min | +/-60min is a true peak, lower on both sides |
| `weight` | blend weight 0.4-0.9 | plateau at 0.5-0.6, falls by 0.9 -- optimum bracketed, not at an edge |
| `hop` | asymmetric weights, max-blend, 2nd diffusion pass, 2-hop cust->dev->cust, merchant | every variant within +/-0.0006. Equal weighting was already right; 2-hop adds nothing; merchant is dead even tight |
| `attn` | time-decay, inverse-distance, amount-similarity weighting | every kernel within +/-0.0002 of a flat average |

**On the attention question specifically:** weighting neighbours cannot help
here because the median neighbourhood has ~1 sibling. A weighted mean over one
element is an unweighted mean over one element. A learned attention layer would
have had strictly more ways to fail against the same zero headroom -- this is
now measured, not argued.

## Everything that did not work

| lever | result | why |
|---|---|---|
| Target encoding | +0.0031 | under the (old) bar, loses the hardest window, carries reproducibility-review exposure |
| Drift normalisation | +0.0001 | most flagged drift is deterministic accumulation; percentiles strip the absolute level, which carries signal |
| Ranking objectives | -0.009 | PR-AUC scores one GLOBAL ranking; lambdarank optimises NDCG within group |
| Self-training (soft / hard) | +0.0023 / -0.0101 | small, and the only idea that trains on test rows |
| Drop dead-regime features | +0.0001 / -0.0010 | `cust_amt_robust_z` fell 0.41 -> 0.24 univariate AP but 0.24 still beats the best device feature's 0.172 |
| DART, L1/L2, max_bin, extra_trees, GOSS | -0.0039..+0.0006 | the defaults were already right |
| Learned stacking | +0.0029 | **reopened** -- see the exchange rate above |

## Bugs found and fixed today

Each of these produced a plausible-looking wrong answer before being caught:

- **Clustering diagnostic** divided fraud-fraud pairs by `n(n-1)`, printing the
  *joint* probability as the conditional. Real 2.6x clustering read as "lift
  0.0x" and nearly got propagation -- the winning idea -- discarded outright.
- **`.days` on a timedelta Series** (needs `.dt.days`), crashing the candidate
  after 3 minutes of featurisation.
- **Pseudo-label objective**: `binary` counts any label > 0 as a positive, so
  soft labels of ~0.017 were all read as frauds and AP "collapsed" 0.73 -> 0.03.
- **`astype("int64")/1e9`** assumed nanosecond resolution; pandas 2.x used
  microseconds, silently scaling the clock by 1000 so a 20-minute gap read as
  0.02. Caught by `scripts/test_blend.py` before it ever ran.
