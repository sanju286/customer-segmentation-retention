"""
01 — Data acquisition & sanity checks.

Downloads the two Kaggle datasets if a Kaggle credential / internet is
available, otherwise validates manually-downloaded files in data/raw/.

Datasets:
  - blastchar/telco-customer-churn  -> Telco-Customer-Churn.csv (7043 rows)
  - vijayuv/onlineretail            -> OnlineRetail.xlsx / .csv (~540k lines)

Usage:  python src/01_data_acquisition.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def try_kagglehub():
    """Use kagglehub (pip install kagglehub) if credentials are present."""
    try:
        import kagglehub
    except ImportError:
        print("[skip] kagglehub not installed -> install with: pip install kagglehub")
        return False
    try:
        cfg.ensure_dirs()
        telco_dir = kagglehub.dataset_download("blastchar/telco-customer-churn")
        retail_dir = kagglehub.dataset_download("vijayuv/onlineretail")
        for d in (telco_dir, retail_dir):
            for root, _, files in os.walk(d):
                for f in files:
                    src = os.path.join(root, f)
                    dst = os.path.join(cfg.DATA_RAW, f)
                    if not os.path.exists(dst) and os.stat(src).st_size < 200e6:
                        with open(src, "rb") as fi, open(dst, "wb") as fo:
                            fo.write(fi.read())
        return True
    except Exception as e:
        print(f"[warn] kagglehub download failed: {e}")
        return False


def find_telco():
    for f in cfg.TELCO_FILES:
        p = os.path.join(cfg.DATA_RAW, f)
        if os.path.exists(p):
            return p
    return None


def validate():
    ok = find_telco() is not None
    retail = None
    if ok:
        p = find_telco()
        print(f"[ok] Telco: {p} ({os.path.getsize(p)/1e6:.1f} MB)")
    else:
        print(f"[missing] {os.path.join(cfg.DATA_RAW, cfg.TELCO_FILES[0])}")
        print("""  Download manually:
    1. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    2. Place 'Telco-Customer-Churn.csv' in data/raw/""")

    for g in cfg.RETAIL_GLOBS:
        hit = [os.path.join(cfg.DATA_RAW, f)
               for f in os.listdir(cfg.DATA_RAW)
               if f.lower().endswith(g.split("*", 1)[1])]
        if hit:
            retail = sorted(hit, key=os.path.getsize, reverse=True)[0]
    if retail:
        print(f"[ok] Retail: {retail} ({os.path.getsize(retail)/1e6:.1f} MB)")
    else:
        print("[missing] Online Retail file (xlsx or csv)")
        print("""  Download manually:
    1. https://www.kaggle.com/datasets/vijayuv/onlineretail
    2. Place 'OnlineRetail.xlsx' (or the csv) in data/raw/  (file may be zipped -> unzip first)""")
    cfg.ensure_dirs()
    return ok and retail is not None


if __name__ == "__main__":
    if not validate():
        print("\nAttempting kagglehub download ...")
        try_kagglehub()
        if not validate():
            sys.exit(1)
    print("\nData acquisition complete.")
