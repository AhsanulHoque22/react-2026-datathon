# Tabular Fraud Machine Learning Glossary

Key terminology and conventions used in this project.

## Terms

**Binary Classification**:
A predictive task where the target variable has exactly two classes: 0 (negative/legitimate) and 1 (positive/fraud).

**Class Imbalance**:
A situation where one outcome occurs far less frequently than another (e.g. 1.7% fraud vs 98.3% non-fraud).
_Avoid_: Unbalanced labels, skewed numbers

**Data Leakage**:
The introduction of information from outside the training dataset (or from the future, or from target labels) into the model during training or feature creation.
_Avoid_: Sneaking labels, lookahead cheat

**PR-AUC (Precision-Recall Area Under Curve)**:
A metric that evaluates binary classifier performance by calculating the average precision across all decision thresholds; ideal for heavily imbalanced classes.
_Avoid_: ROC-AUC, accuracy

**Strictly-Prior Expanding Statistics**:
Summary features calculated over an entity's historical records strictly prior to the current transaction's timestamp.
_Avoid_: Cumulative averages with current row included

**Walk-Forward Cross Validation**:
A temporal validation strategy where folds move sequentially forward in time, training only on strictly earlier periods and validating on the immediately following period.
_Avoid_: Random K-fold, train_test_split
