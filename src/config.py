"""Shared config for paths and business assumptions."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORTS = os.path.join(PROJECT_ROOT, "outputs", "reports")

# Expected raw files (Filenames used by the official Kaggle downloads)
# Accepted raw filenames (official Kaggle download names)
TELCO_FILES = ("Telco-Customer-Churn.csv", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
RETAIL_GLOBS = ("OnlineRetail*.xlsx", "OnlineRetail*.csv", "online_retail*.xlsx")

# ------------------------------------------------------------------
# Business assumptions (documented, kept in config so they are auditable)
# ------------------------------------------------------------------
MARGIN = 0.30                    # gross margin on revenue (for CLV)
DISCOUNT_RATE = 0.10             # value discounting, kept simple
RETENTION_SAVE_RATE = 0.25       # a targeted risky customer is "saved" 25% of the time
RETENTION_OFFER_COST = 75.0      # cost of a retention offer (telecom, USD)
MAX_OFFER_RATE_PER_YEAR = 4      # max purchases we credit a retail customer per year
ANALYSIS_HORIZON_YEARS = 3       # CLV horizon for retail heuristics


def ensure_dirs():
    for p in (DATA_RAW, DATA_PROCESSED, FIGURES, REPORTS):
        os.makedirs(p, exist_ok=True)
