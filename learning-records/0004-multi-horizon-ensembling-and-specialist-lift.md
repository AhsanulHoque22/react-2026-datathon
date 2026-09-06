# Multi-Horizon Specialist Ensembling Overcomes July Regime Drift

While recency sample-weighting on a single model degraded long-term entity baseline features, ensembling separate time-horizon models preserves both:
1. Full Train (Jan 01 -> Jul 01, PR-AUC 0.5269) captures lifetime customer spending distributions and historical baseline z-scores.
2. 90-Day Specialist (Apr 01 -> Jul 01, PR-AUC 0.5259) focuses purely on recent daytime, location-spike, and smurfing behavioral shifts.

The two models exhibit low rank correlation (r = 0.614). Blending them via rank-averaging (60% Full + 40% 90-Day) produced PR-AUC = 0.5282, establishing a new all-time high local score for the competition.
