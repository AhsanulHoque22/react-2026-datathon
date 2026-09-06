# REACT 2026 Datathon — Team Status Dashboard

Last updated: **2026-09-06 10:56 Dhaka** (~37h to the 2026-09-07 23:59:59 hard deadline)

> Full modeling plan, rules cross-checks, and rationale: [`PLAN.md`](./PLAN.md).
> This doc is the short "what's the state of things" view for the team — check
> `PLAN.md` for the *why* behind any decision below.

## Leaderboard so far

| # | Submission | Public PR-AUC | Notes |
|---|---|---|---|
| 1 | `submission_baseline.csv` | **0.16644** | Raw fields only (amount, account age, 5 low-card categoricals), no behavioral features — insurance submission |
| 2 | `submission_final.csv` | **0.51309** | Full behavioral pipeline (see below). 3.1x the baseline. Matches our walk-forward CV estimate on the most recent fold (~0.50) closely — good sign there's no leakage inflating validation |

**Submission budget**: 2 of 5 used today (Sep 6). **3 remaining today are being held — no more Kaggle submissions today per team decision**, even if local CV shows an improvement. A fresh 5/day unlocks after midnight (Sep 7 Dhaka), for up to 10 total across the contest.

## What's done

- **Rules verified verbatim** against the live Kaggle Overview/Data/Rules pages (not paraphrased) — metric is PR-AUC (`average_precision_score`), chronological train/test split, 5 submissions/day/team, top-15 needs a reproducible Kaggle Notebook (Version History, not just current code).
- **Official pre-launch rulebook PDF cross-checked**: shortlisted teams' notebook + summary is due **2026-09-08 10:00 Dhaka**, only ~10h after the leaderboard closes — the reproducibility notebook needs to be built incrementally, not assembled at the last minute.
- **Data pipeline**: time-sorted `(timestamp, transaction_id)`-keyed combined train+test frame with a deterministic tie-break sort (train alone has ~20K duplicate timestamps that would otherwise silently corrupt history-based features).
- **Feature engineering** (leakage-safe, verified on a hand-built synthetic example before running on real data):
  - Per-entity (customer/merchant/device) strictly-past expanding stats: history count, amount mean/std, MAD-based robust z-score, time-since-last-transaction
  - Fan-out / novelty pairs: is-new-device-for-customer, is-new-merchant-for-customer, and the reverse (device's customer fan-out, merchant's customer fan-out)
  - Frequency encoding of low-cardinality categoricals (count-based, never fraud-rate-based — see Banned rule in PLAN.md)
  - Calendar + cyclic (sin/cos) time features
  - Trailing 1h/24h rolling windows (currently customer; merchant/device just added, CV in progress)
  - All built via vectorized cumsum/cumcount tricks — full 994K-row frame builds in ~65s, not the slow groupby-apply path
- **Walk-forward CV harness**: 4 chronological folds, PR-AUC feval on a fixed subsample, has-history vs. cold-start split, fold-to-fold spread reporting.
- **Confirmed the organizer's drift warning is real**: fold 3 (closest to the train/test boundary) scored 0.495 PR-AUC vs. 0.73–0.77 on earlier folds. Weekly fraud *rate* is stable (~1.5–1.9% throughout) — so this is a change in fraud *behavior*, not volume, exactly as warned.
- **Tested the recency-weighting contingency** (exponential decay at 14/30/60-day half-lives, and a recent-90-day-only training window) against that hard fold — **no measurable benefit** (0.495–0.497 across every variant, within noise). Conclusion: full-history training stays the approach; the drift isn't fixable by reweighting old rows, it's a genuine harder-to-predict segment.
- **Final model v1** trained on all of train.csv, iteration count picked via a held-out tail, full pre-submission assertions (no raw IDs, no label leakage, no thresholded probabilities, row order matches `sample_submission.csv` exactly) — all passed and independently spot-checked by hand before submitting.
- **Feature importance sanity check**: top drivers are `amount_bdt`, time-since-last, `is_new_device_for_customer`, robust z-score, device/merchant fan-out counts — all genuinely behavioral, nothing resembling entity-ID targeting.
- Code committed and pushed to `main` (`796ea34`): `src/{config,data,features,model}.py`, `scripts/00`–`04`.

## In progress

- ~~CV re-run with merchant/device trailing 1h/24h windows~~ **Done.** Result:
  small net positive (mean PR-AUC 0.6882 → 0.6923), no regressions of
  concern. The drift fold (fold 3) barely moved (0.4952 → 0.4973, within
  noise) — confirming again that fold's difficulty is a genuine behavior
  shift, not something more features fix. Keeping the extended windows.
- ~~Targeted hyperparameter search on the drift fold~~ **Done — negative
  result, current defaults kept.** Tested 5 configs
  (`scripts/05_hparam_search.py`): the best alternative (num_leaves=127,
  min_data_in_leaf=30) beat the current config by only +0.0006 PR-AUC —
  smaller than the ~0.002 noise floor already seen in the recency-weighting
  experiment, so not a real signal. Every *more*-regularized variant was
  clearly worse (−0.006 to −0.010), confirming again that fold 3's
  difficulty is genuine behavior drift, not overfitting a regularization
  knob could fix. No full-CV re-validation run, since there was no
  candidate worth validating.

## What's next

1. ~~Decide on the optional CatBoost blend~~ **Done — negative result, not
   adopted.** CatBoost (one-hot encoded categoricals, deliberately *not*
   CatBoost's native `cat_features` — see `scripts/06_catboost_cv.py`
   docstring for why: its default CTR handling is a per-category target
   encoding, which is exactly what our Banned rule forbids) scored worse
   than LightGBM on **every single fold** (−0.005 to −0.016). A blend-weight
   sweep on the drift fold confirmed no weight beats LightGBM alone
   (best blend +0.0004, inside the noise floor; every weight below 0.9
   was worse). Single LightGBM stays the model — PLAN.md's original
   "no 3-framework ensemble" leaning is now empirically confirmed, not
   just a time-saving assumption.
2. ~~Stretch goal: second-order relationship/cluster features~~ **Done —
   net-neutral alone** (mean 0.6923 → 0.6929, inside noise; implemented via
   an incremental Union-Find rather than a single static
   `scipy.sparse.csgraph.connected_components` call, since that function has
   no point-in-time notion and would leak later edges into earlier rows —
   see `src/features.py` docstring). Kept in the pipeline by team decision
   despite the neutral result.
3. **A teammate shared an independent EDA/teardown doc** analyzing the raw
   signal structure. Its headline finding — per-entity historical fraud-rate
   / target encoding (`te_dev`, `te_mer`, `te_cust`) — is the single most
   important input to its higher CV score (0.768 mean, best fold 0.812),
   but it's exactly the mechanism our round-4 council unanimously rejected
   as violating the organizer's "targeting specific entity IDs rather than
   behavioral patterns" rule. **Not adopted** — same reasoning as before,
   now with concrete evidence of the tradeoff (see `PLAN.md` for the
   full writeup). Its CV also uses wider, looser folds (~117K rows each)
   than ours, which independently inflates comparability.
4. **Two compliant ideas from that same doc were adopted and validated —
   large, real improvement**: log1p(amount) z-score per entity (amount is
   heavily right-skewed; log-scale deviation is better-behaved than the
   linear-scale one we already had) and `is_new_location_for_customer`
   (same novelty pattern as device/merchant, extended to location). Combined
   with the retained graph features:

   | Fold | Before | Now | Δ |
   |---|---|---|---|
   | 0 | 0.7455 | 0.7748 | +0.029 |
   | 1 | 0.7716 | 0.7883 | +0.017 |
   | 2 | 0.7596 | 0.7810 | +0.021 |
   | 3 (drift) | 0.4947 | 0.5029 | +0.008 |
   | **mean** | 0.6929 | **0.7117** | **+0.019** |

   Consistent gain across every fold, well beyond the ~0.002-0.006 noise
   floor — a real improvement, not noise. This is now the best-validated
   local pipeline, well ahead of what's actually submitted (0.51309 public).
5. Retrained and **submitted** (2 remaining today): scored **0.51055** —
   slightly *lower* than the previous best (0.51309), despite CV going up
   (0.6923 → 0.7117). Investigated why via adversarial validation
   (`scripts/08_adversarial_validation.py`, run per a team request to
   research broader improvements, not just the drift fold): a train-vs-test
   classifier achieves **AUC=1.0000** (perfectly separable), driven almost
   entirely by the two bipartite graph component-size features — PSI
   ~10-11, importance orders of magnitude above every other feature.
   Root cause: Union-Find component sizes only grow over time, so test-period
   values sit at a scale never seen in training — classic unbounded-feature
   covariate shift. Testing a fix (drop vs. scale-invariant normalization)
   now, before further changes.
6. Keep building the Kaggle Notebook version incrementally (not just at the end) given the tight ~10h post-close reproducibility window — see `TEAM_TASKS.md` task 2.
7. Team can now also work in parallel: `TEAM_TASKS.md` has step-by-step commands for (1) reproducing the pipeline locally and (2) packaging the Kaggle Notebook.

## Key risks being tracked

- **Temporal drift** — confirmed real (see above). Actual test set (Jul 16–Sep 15) is even further from train's end than the fold that showed the drop, so expect the true leaderboard score to sit nearer the drift-affected estimate than the earlier, easier folds.
- **Submission budget discipline** — only validate via local CV; a submission is spent only on a change we're confident about, not to "see what happens."
- **Reproducibility deadline** — Sep 8 10:00 AM is close behind the Sep 7 23:59:59 leaderboard close; don't leave notebook packaging for the end.
