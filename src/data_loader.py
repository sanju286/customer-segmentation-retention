"""Dataset loaders with resilient path / format handling."""
import glob
import os

import pandas as pd

import config as cfg


def load_telco() -> pd.DataFrame:
    p = None
    for f in cfg.TELCO_FILES:
        cand = os.path.join(cfg.DATA_RAW, f)
        if os.path.exists(cand):
            p = cand
            break
    if p is None:
        raise SystemExit("No Telco file in data/raw/. See src/01_data_acquisition.py output.")
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_retail() -> pd.DataFrame:
    for pattern in cfg.RETAIL_GLOBS:
        hits = glob.glob(os.path.join(cfg.DATA_RAW, pattern))
        if hits:
            path = sorted(hits, key=os.path.getsize, reverse=True)[0]
            if path.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(path)
                except UnicodeDecodeError:
                    # classic UCI export uses Latin-1 (£ currency symbol)
                    df = pd.read_csv(path, encoding="latin-1")
            else:
                try:
                    df = pd.read_excel(path, engine="openpyxl")
                except ImportError as e:
                    raise SystemExit(
                        "Reading .xlsx requires openpyxl -> run: pip install openpyxl") from e
            return df
    raise SystemExit("No Online Retail file in data/raw/. See src/01_data_acquisition.py output.")
