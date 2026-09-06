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
- Targeted hyperparameter search on the drift fold (fold 3) — running now.

## What's next

1. Finish the hyperparameter search above; adopt only a config that helps without hurting the easier folds.
2. Decide on the optional CatBoost blend (PLAN.md: opportunistic, only if hours allow with LightGBM already solid).
3. Stretch goal (only after the above): second-order relationship/cluster features via `scipy.sparse.csgraph.connected_components` — not started yet, lowest priority per the council's scope-creep triage.
4. Keep building the Kaggle Notebook version incrementally (not just at the end) given the tight ~10h post-close reproducibility window.
5. Retrain + validate the final model with whatever the best-confirmed feature/hyperparameter set turns out to be; no Kaggle submission until the team says so.

## Key risks being tracked

- **Temporal drift** — confirmed real (see above). Actual test set (Jul 16–Sep 15) is even further from train's end than the fold that showed the drop, so expect the true leaderboard score to sit nearer the drift-affected estimate than the earlier, easier folds.
- **Submission budget discipline** — only validate via local CV; a submission is spent only on a change we're confident about, not to "see what happens."
- **Reproducibility deadline** — Sep 8 10:00 AM is close behind the Sep 7 23:59:59 leaderboard close; don't leave notebook packaging for the end.
