# REACT 2026 — modeling plan

Full write-up with reasoning: see the case file artifact (ask Ahsanul for the link).

## Decided approach

- **Model**: single LightGBM. No 3-framework ensemble — not enough hours for a small team to tune XGBoost+CatBoost+LightGBM in parallel. Add CatBoost only if LightGBM is solid with hours to spare.
- **Feature engineering**: time-safe aggregates first (rolling counts, frequency encoding, time-since-last-event). Entity-ID reconstruction + group-aggregation ("magic feature") only if EDA finds reconstructable identity columns by hour 10 — hard cutoff, not guaranteed.
- **Validation**: chronological GroupKFold, never random k-fold. Run adversarial validation (train-vs-test classifier) to confirm the split behaves like the real public/private split before trusting it.
- **Class imbalance**: `scale_pos_weight` / `class_weight='balanced'`. No SMOTE — reliably loses AUC and hurts calibration on this kind of fraud data.
- **Calibration**: isotonic regression on held-out folds, as a 10-minute check after the model is frozen. Keep only if it doesn't cost validation score. Not a scheduled phase.
- **Banned**: cross-entity pseudo-labeling (matching test rows to train rows via reconstructed identity to borrow labels). Sits against the rule barring "reverse-engineering how the dataset was generated" — zero hours, no exceptions.

## Execution timeline (~40 hrs, Dhaka time)

| Hours | Task |
|---|---|
| H0–3 | Ingest data, confirm eval metric, check class balance/time column |
| H3–4 | **Submit insurance baseline** — plain LightGBM, no feature engineering |
| H4–10 | Adversarial validation + hunt for reconstructable entity ID (hard cutoff) |
| H10–26 | Feature engineering + model iteration, scored on chronological CV only |
| H26–34 | Optional CatBoost blend + calibration check |
| H34–37 | Pick final 2 submissions: best-CV + one materially different model |
| H37–40 | Lock and stop — no new features/models this close to the deadline |

## Why

Research base: IEEE-CIS Fraud Detection (Kaggle, 2019) — closest known analog (masked tabular columns, time-ordered, entity-based fraud, probability/AUC output, public/private split). Plan finalized via a 5-advisor LLM council + peer review; the round caught that none of the advisors specified concrete time-based CV mechanics, which is why validation is spelled out explicitly above rather than left implicit.
