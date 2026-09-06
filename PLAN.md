# REACT 2026 — modeling plan (round 5 — final verbatim-verified check)

Original plan (pre-competition) assumed an IEEE-CIS-style analog with masked
columns and a hidden entity ID needing reconstruction — wrong on both the
entity-ID point and the timeline once the competition opened (round 2).
Round 3 was a hardening pass (brutal-critique council → targeted research →
refinement council). **Correction on top of round 3**: the "~48h / 2 days"
runway used throughout rounds 2–3 was wrong — it came from Kaggle's rounded
"2 days to go" display, never from the exact deadline timestamp, despite the
plan itself flagging "re-verify verbatim" as an open task. Pulled the exact
`title` attribute from the countdown element via JS: **start 2026-09-06
08:00:00 +06, close 2026-09-07 23:59:59 +06 — a 39h59m59s window, i.e.
~40 hours total, not 48.** The original pre-competition plan's "~40h"
estimate was right; the "correction" to ~47/46h in rounds 2–3 was the actual
error. Timeline rebuilt below from the real remaining hours.

## Ground truth (confirmed from kaggle.com/competitions/react-2026-datathon + data)

- **Metric: PR-AUC** (`sklearn.metrics.average_precision_score`), not ROC-AUC.
  Fraud rate 1.76% (731,942 train rows / 262,648 test rows; ~4,600 fraud rows
  expected in test at this rate — small enough that fold-to-fold PR-AUC has
  real variance, see Validation below).
- **Deadline: exact timestamps, not the rounded UI display** — start
  2026-09-06 08:00:00 +06 (Dhaka), close 2026-09-07 23:59:59 +06. Total
  contest window ≈ 40 hours. No buffer past 23:59:59 on the 7th — that is a
  hard stop, not a display rounding.
- Train: 2026-01-01→07-15. Test: 2026-07-16→09-15, strictly after train, zero
  overlap. Leaderboard 60% public / 40% private, up to 2 selected submissions.
  **5 submissions/day cap** (competition Rules page) — track daily count,
  don't burn it on near-duplicate probes.
- `customer_id`/`merchant_id`/`device_id` are **real IDs given directly** —
  NOT masked/proxy columns needing reconstruction. Raw overlap (ID appears
  anywhere in train) is 87%/83%/84%. **The overlap number that actually
  matters is stricter**: an entity only has usable history if it has ≥1
  transaction *strictly before* the row being scored — recompute this inside
  the feature pipeline itself (one `groupby().cumcount()`-style check you're
  already running), don't just trust the raw-overlap figure as the cold-start
  rate.
- Missingness <1%, only in `merchant_category`/`device_type`/`location`.
  `location` is a coarse categorical code (e.g. `LOC_028`), **not
  lat/long coordinates** — geo-distance/haversine features are inapplicable
  to this dataset, confirmed against the data dictionary, not assumed.
- **Event-level facts, cross-checked against the organizer's official
  pre-launch rulebook PDF (round 4 — a source the user surfaced, verified as
  genuine, not the problem-specific rules which remain the Kaggle page)**:
  online round closes 2026-09-07 23:59:59 Dhaka (matches the DOM-verified
  timestamp above — independent confirmation, not a duplicate assumption).
  **Shortlisted teams' notebook + 1–2 page summary is due 2026-09-08 10:00
  Dhaka — only ~10h after the leaderboard closes.** The notebook must be
  privately shared (collaborator, not public) and must point to a specific
  **Kaggle Notebook Version History** entry that reproduces the scoring
  submission — not just the current/latest notebook state. Team size is
  confirmed 2–4. Implication: the reproducibility artifact needs to be built
  incrementally inside an actual Kaggle Notebook (or ported into one and
  versioned) well before the close, not assembled from scratch in the ~10h
  grace window — see the timeline below.
- **Organizer's rules — verbatim-confirmed in round 5, not a paraphrase**.
  Re-fetched Overview/Description, Overview/Evaluation, Data, and Rules
  pages via `get_page_text` (full DOM text, not screenshots this time) and
  diffed every quote in this document against the live text word-for-word.
  Zero discrepancies found. Also checked the Discussion tab: empty, no
  organizer clarifications posted to reconcile. Confirmed exact quotes:
  - *"Every engineered feature for a transaction at time t may only use
    information strictly before t."* Separately: *"Feature computation that
    uses only the raw, non-target columns of test.csv in a strictly-past-only
    way is fine (e.g., a device's known transaction history can include
    test-period rows that occurred earlier than the row being scored); using
    the fraud label anywhere near test.csv is not, since it does not exist
    for you."*
  - *"Validation leakage — random K-fold cross-validation is not appropriate
    for this task... Use time-based, walk-forward, expanding-window, or
    purged/embargoed validation instead."*
  - *"Attempting to reverse-engineer or scrape the organizer's private
    fraud-generation logic, or targeting specific entity IDs rather than
    behavioral patterns, undermines the spirit of the competition and may be
    flagged during reproducibility review."*
  - *"No external data of any kind... Top 15 private leaderboard teams must
    submit a notebook and short summary for reproducibility verification to
    qualify for the onsite final."* *"Maximum 5 submissions per day per
    team."* *"Final score: 60% online phase + 40% onsite presentation."*
  - *"Relationship structure — do groups of customers, devices, and
    merchants form suspicious clusters that no single transaction reveals on
    its own?"*
  - Official metric confirmed verbatim: *"PR-AUC (Average Precision)...
    This is exactly `sklearn.metrics.average_precision_score`."* Fraud rate
    stated by the organizer as *"roughly 1.5–2%"* — matches our measured
    1.76% train rate. Train/test date ranges confirmed verbatim against the
    Data page, matching this document exactly.

## Decided approach

- **Model**: single LightGBM. CatBoost only opportunistically if hours allow
  with a solid LightGBM already frozen. No 3-framework ensemble.
- **Training objective**: `objective=binary` + `scale_pos_weight` for the
  1.76% imbalance. `metric=custom` + a `feval` computing
  `average_precision_score` for early stopping and fold scoring — do not
  rely on logloss or ROC-AUC as the selection signal.
  - **Cost control (new, round 3)**: `average_precision_score` re-sorts the
    full validation fold every boosting round; across ~1M rows × 4 folds ×
    hyperparameter attempts this is a real tax LightGBM has no built-in
    "eval every N rounds" option for. Mitigation, lazy version only:
    evaluate the feval on a **fixed random stratified subsample** of each
    fold's validation set (e.g. ~100–150K rows, sampled once per fold, not
    re-sampled each round) and cap `num_boost_round`/`early_stopping_rounds`
    to sane values (e.g. 3000 / 100). Do not build a custom training loop or
    adaptive eval-frequency scaffolding — not worth the hours.
- **Feature engineering** (core, required from hour 0):
  - Per-entity (customer/merchant/device) strictly-past expanding aggregates:
    rolling txn count & amount sum in trailing windows, time-since-last-txn,
    expanding mean/std of `amount_bdt`, nunique distinct counterparties seen
    so far.
  - Deviation-from-own-baseline: z-score of `amount_bdt` vs that customer's
    own expanding mean/std, **plus a MAD-based robust variant (round 4)** —
    `(amount - expanding_median) / (expanding_MAD + eps)` — since a plain
    std-based z-score is itself inflated by the very outliers (fraud) it's
    trying to flag; the robust version is a one-line addition alongside the
    existing z-score, not a replacement, so the model can use whichever
    generalizes better.
  - First-order relationship/fan-out counts: distinct customers per device so
    far, distinct devices per customer so far, is-new-device-for-customer
    flag, is-new-merchant-for-customer flag.
  - Frequency (count-based, not fraud-rate-based) encoding of
    `merchant_category`/`device_type`/`location`/`payment_method`/`transaction_type`.
  - Calendar features: hour-of-day, day-of-week, is-night, **plus cyclic
    sin/cos encoding of hour-of-day and day-of-week (round 4)** — one line
    each, lets the model see that hour 23 and hour 0 are adjacent.
  - **Minimum-history-count gating (round 4)**: alongside every expanding
    mean/std/z-score feature, also emit the entity's prior observation count
    (`customer_history_count`, `merchant_history_count`,
    `device_history_count` — largely already implied by the existing count
    aggregates, made explicit here). This lets the tree discount a baseline
    computed from one or two prior transactions instead of treating it as
    equally reliable as a baseline from hundreds.
  - All aggregates computed on one time-sorted train+test frame (non-label
    facts only) so history threads correctly across the train→test boundary.
  - **Mandatory correctness checks**:
    - Every expanding/rolling aggregate must use `.shift(1)` (or equivalent)
      so a row never sees its own value in its own baseline. Assert each
      entity's first-occurrence row shows NaN/global-prior, never its own
      row's value.
    - **Deterministic tie-break for the sort (new, round 3)**: train alone
      has ~20K duplicate timestamps out of 731,942 rows (`nunique` timestamp
      count confirmed via EDA). Sort every frame by `(timestamp,
      transaction_id)` — a stable, arbitrary-but-fixed secondary key — before
      any expanding/rolling/cumulative operation. Without this, rows sharing
      a timestamp get an undefined, run-to-run-unstable ordering, which
      silently corrupts whichever aggregate depends on "what came before."
    - **Pre-submission assertions checklist (round 4, cheap, run before every
      submission)**: timestamps are non-decreasing after the tie-break sort;
      no column named `fraud` (or derived from it) is present in the
      inference-time feature matrix; no raw ID column
      (`customer_id`/`merchant_id`/`device_id`/`transaction_id`) is in the
      model's feature list; `submission.csv` row order and `transaction_id`
      values exactly match `sample_submission.csv`; **every `fraud` value is
      a float in `[0, 1]`, never thresholded to a hard 0/1 label (round 5 —
      confirmed verbatim: "must be a probability in [0, 1], not a hard
      label")**. Six `assert` lines, not a framework.
  - **Explicit ban (new, round 3): never pass `customer_id`/`merchant_id`/
    `device_id`/`transaction_id` into the model's feature matrix directly**
    (as native LightGBM categoricals or otherwise) — only their *derived*
    aggregates. Two independent reasons, not just one: (1) it is the literal
    reading of "targeting specific entity IDs rather than behavioral
    patterns"; (2) LightGBM's own docs confirm high-cardinality categoricals
    (ours: 38.7K/4.2K/19.1K uniques) are a known overfitting risk independent
    of this competition's rules. Drop these columns right before `.fit()`;
    keep them only as join keys upstream.
  - **Why count-based features aren't leakage even though they correlate
    with fraud (new, round 3 — document this paragraph in the submitted
    notebook)**: a feature like "device fan-out spiked this week" will
    correlate with real fraud activity, because that correlation *is* the
    detection signal the organizer explicitly asked for — not an artifact.
    It passes the leakage test because it never touches the `fraud` column
    and only uses information strictly before `t`; it is exactly the
    "behavioral pattern" the rules distinguish from "targeting an ID."
  - **Cold-start handling** (13–17% raw-overlap gap, tighter once measured
    per the stricter definition above; new-device/new-merchant is a classic
    fraud signature, so this is not a footnote): leave expanding/deviation
    features as NaN for first-occurrence rows and let LightGBM's native
    NaN-routing handle the split; keep the is-new-* flags. Every walk-forward
    fold must report **PR-AUC split by has-history vs. cold-start rows** —
    one groupby.
  - **Stretch goal only, not core**: second-order relationship/cluster
    features (shared-device components, customer-device-merchant triangle
    structure). **Implementation constraint (new, round 3)**: use
    `scipy.sparse.csgraph.connected_components` on a sparse adjacency matrix
    — scipy is already a transitive dependency via scikit-learn, so this is
    a function-call swap, not a new dependency — never NetworkX, which does
    not scale past roughly 100K nodes on commodity RAM and our entity counts
    (38.7K customers, 19.1K devices, 4.2K merchants) sit right at that edge
    once combined into a bipartite graph. Attempt only after the core
    pipeline is solid and submitted.
- **Validation**: time-ordered expanding-window walk-forward CV, ~4 folds
  tiling the last ~8 weeks of train (fold test-windows [05-21→06-04],
  [06-04→06-18], [06-18→07-02], [07-02→07-15], train = all rows strictly
  before each window's start). Report per-fold PR-AUC, not just a mean:
  - Has-history vs. cold-start split (already planned).
  - **Fold-to-fold spread — min/max/std across the ~4 folds (new, round 3,
    replaces any bootstrap-CI plan)**: with only ~4,600 total fraud rows
    expected in test, PR-AUC has real sampling variance. When choosing
    between "best CV" and "one materially different model" for the final 2
    submissions, compare their fold-score *spreads*, not single point
    averages — this is free, since the folds are already being scored
    individually. Do not build bootstrap-resampling or closed-form CI
    machinery; the existing per-fold numbers already answer the question at
    near-zero extra cost.
  - **Per-fold PR-AUC trend over time (new, round 3, grounded in the Bank
    Account Fraud / NeurIPS 2022 dataset — see Why)**: plot the 4 fold
    scores against their time order (one matplotlib call, reusing numbers
    already computed). The organizer explicitly warns fraud patterns can
    drift; if the plot shows a decay trend, treat it as license to consider
    recency-weighted training (e.g. `sample_weight` favoring recent rows) —
    but only *if* the plot actually shows decay. Do not pre-build a
    drift-mitigation pipeline against a problem that hasn't been observed.
  - Adversarial validation kept, as a train/test time-drift sanity check.
  - **Precision/recall-at-top-k diagnostics (round 4, reporting only, no
    model change)**: alongside PR-AUC, log precision@0.5% and recall@1% per
    fold. Doesn't change optimization (still PR-AUC) but gives the team an
    interpretable "if we flagged the top N transactions, how many would be
    real fraud" number for the write-up and for sanity-checking that CV
    improvements are real, not curve-shape artifacts.
  - **Explicitly rejected as scope creep (round 4)**: an 8-time-window ×
    7-entity/pair-type velocity matrix, two-hop/multi-hop graph traversal
    beyond the already-scoped first-order fan-out + stretch-goal
    `scipy.sparse.csgraph` component features, and haversine geo-distance
    (the `location` field has no coordinates — see Ground truth). All three
    were in the external report; the council's unanimous read was
    combinatorial feature-engineering effort disproportionate to ~38h left,
    for gains the existing plan already substantially captures or that the
    data doesn't support. Recency-weighted training with tuned half-lives
    stays exactly as already planned — a contingency *if* the per-fold trend
    plot shows decay, not pre-built.
- **Class imbalance**: `scale_pos_weight`. No SMOTE.
- **Calibration: cut entirely (new, round 3 — reverses the round-2 plan)**.
  `average_precision_score` is a rank metric; isotonic regression is *not*
  strictly monotonic (confirmed via a scikit-learn maintainer bug report —
  it can introduce ties among originally-distinct scores), so it can only
  leave PR-AUC unchanged or silently hurt it via tie-collapsing — never
  improve it. There is no scenario where spending time on this pays off for
  this metric. Not scheduled, not a stretch goal, not attempted.
- **Banned** (mechanical rule — this exact paragraph goes in the submitted
  notebook): *A feature is safe iff it can be computed with the `fraud`
  column deleted from existence. Count-based, time-delta, and
  non-label-numeric aggregates (frequency, fan-out, time-since-last,
  deviation-from-own-baseline) are always fine, including when grouped by
  `customer_id`/`merchant_id`/`device_id` — grouping by an ID is not the same
  as targeting it. Banned: (1) any feature whose aggregation touches the
  `fraud` column at a grain finer than the whole training set — i.e. any
  per-entity or per-subgroup fraud-rate/target encoding, and the related
  IEEE-CIS "replace prediction with within-ID average including train
  labels" post-processing trick; (2) passing raw `customer_id`/
  `merchant_id`/`device_id`/`transaction_id` into the model's feature matrix
  directly, in any encoding. Zero exceptions.*
  - **Round 4 — explicitly reconsidered and reaffirmed**: an external
    strategy report (user-supplied, unverified source) proposed smoothed
    per-entity historical fraud-rate features (`(f_g + α·p0)/(n_g + α)` using
    only strictly-past labels for that customer/device/merchant/pair),
    arguing it's leakage-safe. Put to a fresh 5-advisor council specifically
    on this question; **unanimous reject, independently reasoned**: this is
    not a leakage question (which the strictly-past framing does answer) —
    it's a mechanism-identity question, and the organizer wrote a *separate*,
    human-judged rule for exactly this ("targeting specific entity IDs rather
    than behavioral patterns"). A smoothed per-entity fraud rate is a
    fingerprint of "this entity has historically been fraudulent," not a
    behavioral pattern, no matter how it's regularized — a reviewer reading a
    feature-importance plot with a per-customer fraud-rate column at the top
    would read it as ID-targeting regardless of the smoothing math. The
    asymmetry is decisive: modest PR-AUC upside (existing velocity/deviation
    features already proxy much of the same signal) against total loss of a
    top-15 slot on reproducibility review. **Not added, not built behind a
    flag.** The existing ban stands as originally written.
- **Reproducibility hygiene (new, round 3 — cheap, required by the rules,
  previously missing entirely)**: `pip freeze > requirements.txt` once the
  environment is final (currently unpinned); one global `SEED` constant used
  everywhere a `random_state`/`seed` parameter exists (LightGBM, numpy,
  train/val splitting). ~25 minutes total, no further infra (no Docker, no
  DVC — disproportionate for a ~40h student datathon).
- **Team logistics (new, round 3 — a shared doc, not tooling)**: assign
  explicit ownership (who owns leakage-critical feature code + CV harness —
  keep this to one person for a clean bus factor on the highest-risk code;
  others own EDA / hyperparameter search / second model / write-up). Track
  the 5-submissions/day cap in a shared note — note the contest spans two
  Kaggle calendar days (Sep 6 and Sep 7 Dhaka time), so the cap effectively
  resets once at midnight. See the timeline below for the replanning
  checkpoint: if walk-forward CV and the two insurance-submission
  leaderboard scores disagree materially, stop and re-diagnose before
  continuing to iterate blind.

## Execution timeline (~39.2 hrs remaining, re-anchored round 5)

Data ingestion, initial EDA (shape/fraud-rate/missingness/cardinality/
overlap), and rules verification are **already done** as of this revision —
the table below starts from that point, not from a cold start. Times are
`T+` hours from **2026-09-06 08:50 Dhaka** (re-anchored to the actual
current time for this final check; the previous anchor was 08:35, a ~15-min
drift now folded in) with wall-clock in parentheses; the hard deadline is
**2026-09-07 23:59:59 Dhaka**, no exceptions — the last row's end time is
the deadline itself, not a rounded duration.

| T+ (wall clock) | Task |
|---|---|
| 0.0–0.5h (08:50–09:20) | Pin `requirements.txt`, set global `SEED`, agree task ownership |
| 0.5–1.5h (09:20–10:20) | Build the time-sorted `(timestamp, transaction_id)`-keyed train+test frame |
| 1.5–2.5h (10:20–11:20) | **Submit insurance baseline** — plain LightGBM, no feature engineering, no raw IDs |
| 2.5–12.5h (11:20–21:20) | Full behavioral feature engineering (shifted expanding aggregates, deviation-from-baseline, fan-out, frequency, calendar) + tie-break sort + leakage assertions + cold-start NaN routing |
| 12.5–14.5h (21:20–23:20, Sep 6) | Adversarial validation + walk-forward CV harness (feval on subsampled fold, has-history/cold-start split, per-fold spread + trend plot) |
| **14.5h (~23:20, Sep 6)** | **Replanning checkpoint**: does CV agree with the two leaderboard scores so far? If not, stop and diagnose before continuing |
| 14.5–30.5h (23:20 Sep 6 → 15:20 Sep 7) | Model iteration scored only on walk-forward PR-AUC (custom feval) |
| 30.5–34.7h (15:20–19:30 Sep 7) | Optional: CatBoost blend, stretch-goal second-order graph features (scipy csgraph) if hours remain and core is solid |
| 34.7–38.2h (19:30–23:00 Sep 7) | Pick final 2 submissions: best-CV + one materially different model, compared by fold-score spread, not single averages |
| 38.2–39.2h (23:00–23:59:59 Sep 7) | Lock and stop; final commit of the Kaggle Notebook version that generated the scoring submission |
| — (23:59:59 Sep 7 → 10:00 Sep 8) | **~10h grace window, not extra build time (from the official rulebook)**: if shortlisted, share the already-versioned Kaggle Notebook privately with organizers + finish the 1–2 page method summary. This window is for packaging, not for writing new code — the notebook must already reproduce the scoring submission by 23:59:59, so treat notebook cleanliness as continuous throughout the run, not a last-hour task. |

## Why

Research base: IEEE-CIS Fraud Detection (Kaggle, 2019) for the *behavioral
aggregation by entity* technique, explicitly not its label-averaging
post-processing trick (banned here). **Bank Account Fraud / BAF (NeurIPS
2022, Feedzai)** — added this round — is a closer academic analog than
IEEE-CIS for the temporal-drift concern specifically: a privacy-preserving
fraud benchmark using the same "first N months train, later months test"
split as literal standard practice in the real-world fraud domain, with
dataset variants built to test model degradation as "features that were
useful to detect fraud for a time may become obsolete as fraudsters adapt
their behaviour to evade detection" — directly grounding the organizer's own
warning and this plan's per-fold trend check. Isotonic-calibration's
tie-breaking flaw was confirmed via a scikit-learn maintainer bug report
(GH #16321), not assumed. `scipy.sparse.csgraph.connected_components` and
LightGBM's categorical-cardinality guidance came from targeted research into
this round's identified gaps.

Plan finalized via two additional 5-advisor LLM council rounds this pass: a
brutal-critique round with no constraint to be constructive (it found the
isotonic-calibration flaw, the missing tie-break rule, the missing raw-ID
ban, the unbudgeted feval cost, and the reproducibility gap, among others),
followed by a refinement round that converged — independently, across all
five advisors — on the same short list of cheap must-fix items and the same
verdict to cut isotonic calibration outright rather than patch it.

**Timeline correction**: the ~48h figure used throughout rounds 2–3 was
never re-verified against the actual deadline timestamp — it was inferred
from Kaggle's rounded "2 days to go" display. Caught when the user
questioned the math against the real total (40h, 30 min elapsed at the
time). Re-checked directly against the countdown element's exact `title`
attribute via JS on the live page rather than trusting the rounded display
a second time. Lesson applied: "re-verify verbatim" items in this plan mean
pulling the exact underlying value, not re-reading the same rounded UI text
more carefully.

**Round 4**: the user supplied an external, unverified AI-generated fraud-
detection strategy report and asked for it to be pressure-tested via
council rather than folded in uncritically. It cited a presigned S3 link to
"DATATHON_RULEBOOK.pdf" — downloaded and read directly rather than trusted
on the report's say-so; confirmed genuine (IEEE SEU SB's own pre-launch
event rulebook), and it independently corroborates the Sep 7 23:59:59 close
already found via the DOM, plus adds facts absent from the Kaggle page
itself (Sep 8 10:00 notebook+summary deadline, Kaggle Notebook Version
History requirement, team size 2–4). On the report's substantive ML
suggestions: a fresh 5-advisor council unanimously rejected its central
recommendation (per-entity historical fraud-rate encoding) as a
mechanism-identity violation of the organizer's entity-targeting rule, not
a leakage question — and separately triaged the report's other ideas into
cheap accepted wins (MAD z-score, cyclic time encoding, min-history-count
gating, precision/recall@k, assertions checklist) versus scope creep
rejected outright (multi-window/multi-entity velocity matrix, multi-hop
graph traversal, haversine distance — the last inapplicable since
`location` isn't coordinate data).

**Round 5 (final check)**: re-fetched every organizer-facing page
(Overview/Description, Evaluation, Data, Rules, Discussion) via full-page
text extraction and diffed every quote used across all five rounds against
the live text — zero discrepancies. Read the plan front-to-back for internal
contradictions: found one stale line (the Ground-truth section still framed
verbatim rule-checking as an open task after round 3, when it should have
been marked done once actually performed) and one small gap (the
pre-submission assertions checklist didn't explicitly check the
probability-not-hard-label submission format, even though the rest of the
plan assumes it) — both fixed above. Everything else — the banned-feature
rule, the feature list, the validation design, the timeline's phase
ordering, the cold-start/tie-break/leakage mechanics — checked internally
consistent with no contradictions. Timeline re-anchored to the exact current
time (2026-09-06 08:50 Dhaka, ~39.2h remaining) rather than compounding
drift from the round-3 anchor.
