"""Central configuration for the retail intelligence pipeline."""

# --- Paths ---
RAW_DATA_PATH = "data/online_retail_II.xlsx"
CACHE_PATH = "data/raw_cache.pkl"  # local cache of the parsed Excel sheet (slow to read)
SHEET_NAME = "Year 2010-2011"
LOG_FILE = "logs/pipeline.log"
PLOT_DIR = "plots"
OUTPUT_DIR = "outputs"
MODEL_PATH = "models/retail_churn_v1.pkl"
MODEL_VERSION = "v1.1"

# --- Sampling ---
# 1.0 = use every customer. If < 1.0, whole CUSTOMERS are sampled (never individual
# rows), so each sampled customer keeps their complete purchase history.
SAMPLE_FRAC = 1.0
RANDOM_STATE = 42

# --- Churn definition ---
# A customer is "churned" if they place no order in the CHURN_WINDOW_DAYS after the cutoff.
CHURN_WINDOW_DAYS = 90
TRAIN_CUTOFF = "2011-09-01"
# Earlier cutoff used only for out-of-time validation. Its label window
# (Jun 1 -> Aug 30) closes before TRAIN_CUTOFF, so its labels are fully known at test time.
OOT_TRAIN_CUTOFF = "2011-06-01"

# --- Evaluation protocol ---
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 2

# --- Serving ---
RISK_THRESHOLD_CRITICAL = 0.7  # NOTE: probabilities are not calibrated; revisit using the calibration plot
