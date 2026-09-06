from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SUBMISSIONS_DIR = ROOT / "submissions"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

# Raw ID / label columns that must NEVER enter the model's feature matrix directly
# (PLAN.md "Banned" section). Kept only as join keys during feature engineering.
ID_COLS = ["customer_id", "merchant_id", "device_id", "transaction_id"]
LABEL_COL = "fraud"
TIME_COL = "timestamp"

# Walk-forward CV fold windows (PLAN.md "Validation" section).
# Each fold's test window; train = all rows strictly before the window's start.
CV_FOLDS = [
    ("2026-05-21", "2026-06-04"),
    ("2026-06-04", "2026-06-18"),
    ("2026-06-18", "2026-07-02"),
    ("2026-07-02", "2026-07-15"),
]

FEVAL_SUBSAMPLE_SIZE = 120_000
