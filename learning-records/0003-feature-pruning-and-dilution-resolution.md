# Feature Dilution Resolved via Systematic Gain Pruning

When feature count expanded to 227 columns, nearly 60 features contributed near-zero gain, causing feature_fraction dilution and early stopping at only 139-232 rounds (PR-AUC 0.5193). Systematic sweep across feature brackets proved that the Top 160 feature core eliminated dilution, restored deep tree growth to 882 rounds, and pushed held-out PR-AUC to 0.5269 (the highest local tail score across the repo).
