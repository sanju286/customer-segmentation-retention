"""
04 — Customer Lifetime Value estimates, unifying both tracks.

Track A (Telco): contract-based expected remaining tenure -> CLV already computed
                 in 03 within add_business_layer(); this script derives the
                 segment-level economics (revenue-weighted churn economics).
Track B (Retail): simple, assumption-explicit CLV |
      CLV = AOV x buys/yr x horizon yrs x margin, with churn proxy = 1/repeat rate.

Output: outputs/reports/clv_summary.csv, repeat-churn retail flag, heatmap figure.

Usage:  python src/04_clv_estimation.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from data_loader import load_retail
from importlib import import_module

r02 = import_module("02_rfm_segmentation")

sns.set_theme(style="whitegrid", context="notebook")


def retail_clv_and_churn(tx: pd.DataFrame) -> pd.DataFrame:
    rfm = pd.read_csv(os.path.join(cfg.DATA_PROCESSED, "rfm_customers.csv"))

    # churn proxy: no purchase in the trailing 180 days
    churn_proxy_active = (tx.groupby("CustomerID")
                            .agg(last=("InvoiceDate", "max")).reset_index())
    snapshot = tx["InvoiceDate"].max()
    churn_proxy_active["churn_proxy"] = (
        (snapshot - churn_proxy_active["last"]).dt.days > 180).astype(int)

    c = cfg
    rfm = rfm.merge(churn_proxy_active[["CustomerID", "churn_proxy"]], on="CustomerID")
    rfm["buys_per_yr"] = np.minimum(
        rfm["Frequency"] / c.ANALYSIS_HORIZON_YEARS * 2, c.MAX_OFFER_RATE_PER_YEAR)
    rfm["CLV"] = (rfm["AOV"] * rfm["buys_per_yr"] * c.ANALYSIS_HORIZON_YEARS * c.MARGIN).round(2)
    rfm.to_csv(os.path.join(cfg.DATA_PROCESSED, "rfm_customers.csv"), index=False)

    summary = (rfm.groupby("RFM_Segment")
               .agg(customers=("CustomerID", "count"),
                    clv_avg=("CLV", "mean"),
                    clv_total=("CLV", "sum"),
                    churn_proxy_rate=("churn_proxy", "mean"),
                    avg_recency=("Recency", "mean"),
                    avg_rfm_score=("RFM_score", "mean"))
               .sort_index())
    summary["expected_loss"] = (summary["clv_avg"] * summary["churn_proxy_rate"]
                               * summary["customers"])
    summary.round(2).to_csv(os.path.join(cfg.REPORTS, "clv_summary.csv"))

    # value-at-risk map: recent & high value vs fading
    show = rfm.sample(min(len(rfm), 4000), random_state=0)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(show["Recency"], show["Monetary"],
                    c=np.log1p(show["CLV"]), cmap="RdYlGn", alpha=0.4, s=18)
    fig.colorbar(sc, ax=ax, label="log CLV")
    med_r, med_m = rfm["Recency"].median(), rfm["Monetary"].median()
    ax.axvline(med_r, color="gray", ls="--", lw=1)
    ax.axhline(med_m, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set(xlabel="Recency (days since last order)", ylabel="Monetary (log)",
           title="Retail value-at-risk map — right side: high value, fading")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES, "retail_value_risk_map.png"), dpi=140)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    cfg.ensure_dirs()
    tx = r02.clean_transactions(load_retail())
    print("Estimating retail CLV + churn proxy ...")
    s = retail_clv_and_churn(tx)
    print("\nRetail CLV summary by RFM segment:")
    print(s.round(1).to_string())

    # Telco CLV side (computed in 03)
    telco_path = os.path.join(cfg.DATA_PROCESSED, "telco_scored.csv")
    if os.path.exists(telco_path):
        t = pd.read_csv(telco_path)
        ts = t.groupby("PriorityBucket").agg(
            customers=("customerID", "count"), clv_avg=("CLV", "mean"),
            exp_loss_sum=("ExpectedLoss", "sum")).sort_index()
        ts.round(0).to_csv(os.path.join(cfg.REPORTS, "telco_clv_summary.csv"))
        print("\nTelco CLV summary by priority bucket:")
        print(ts.round(0).to_string())
    else:
        print("\n[note] telco_scored.csv not found -> run 03_churn_modeling.py first")
    print("\nDone.")
