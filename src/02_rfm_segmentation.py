"""
02 — RFM segmentation on the Online Retail transaction dataset.

Logic:
  * Clean: drop NaN CustomerID, negative quantities/returns, cancelled invoices.
  * RFM per customer, snapshot = day after last invoice in the data.
      Recency  = days since last purchase
      Frequency = number of distinct invoices
      Monetary  = total spend
  * Segment via R & F quintile scores + KMeans (k=5) as a data-driven overlay.

Output: data/processed/rfm_customers.csv, figures.

Usage:  python src/02_rfm_segmentation.py
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

sns.set_theme(style="whitegrid", context="notebook")


# ------------------------------------------------------------------ cleaning
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # normalise column names across "Online Retail" / "Online Retail II"
    ren = {"Invoice": "InvoiceNo", "Invoice Date": "InvoiceDate",
           "Customer ID": "CustomerID", "Price": "UnitPrice"}
    df = df.rename(columns=ren)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)

    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df = df.dropna(subset=["CustomerID", "InvoiceDate", "UnitPrice"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # 'C' prefix = cancelled order, keep returns out of RFM spend
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df = df[(df["Revenue"] > 0) & (df["Revenue"] < 1e5)]  # strip extreme outliers
    print(f"Cleaning: {n0:,} -> {len(df):,} rows "
          f"({df['CustomerID'].nunique():,} customers, "
          f"{df['InvoiceDate'].max() - df['InvoiceDate'].min()} span)")
    return df


# ------------------------------------------------------------------ RFM
def build_rfm(tx: pd.DataFrame) -> pd.DataFrame:
    snapshot = tx["InvoiceDate"].max() + pd.Timedelta(days=1)

    g = tx.groupby("CustomerID")
    rfm = pd.DataFrame({
        "Recency": (snapshot - g["InvoiceDate"].max()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["Revenue"].sum(),
        "FirstPurchase": g["InvoiceDate"].min(),
        "AOV": g["Revenue"].sum() / g["InvoiceNo"].nunique(),
    }).reset_index()

    # R score: 4 = freshest ; F, M: 4 = biggest (quintiles)
    rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[3, 2, 1, 0]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[0, 1, 2, 3]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[0, 1, 2, 3]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] * 10 + rfm["F_score"]  # classic concat style

    # Human segment mapping (business-friendly)
    seg = np.select(
        [
            (rfm.R_score >= 2) & (rfm.F_score >= 2),                 # recent and frequent
            (rfm.R_score >= 2) & (rfm.F_score < 2),                  # recent, new-ish
            (rfm.R_score < 2) & (rfm.F_score >= 2),                  # gone but was loyal
            (rfm.R_score < 2) & (rfm.F_score < 2),                   # gone and low value
        ],
        ["01 Champions & Loyal", "02 Potential Loyalists / New",
         "03 At Risk / Can't Lose", "04 Hibernating / Lost"],
        default="04 Hibernating / Lost",
    )
    rfm["RFM_Segment"] = seg
    rfm["HighValue"] = (rfm["M_score"] >= 2).astype(int)
    rfm.to_csv(os.path.join(cfg.DATA_PROCESSED, "rfm_customers.csv"), index=False)
    return rfm


def kmeans_overlay(rfm: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    rfm["KMeans_Cluster"] = km.fit_predict(Z)

    # label clusters by monetary centroid rank
    order = rfm.groupby("KMeans_Cluster")["Monetary"].median().sort_values().index.tolist()
    names = {c: f"Cluster {chr(65 + i)} (.{'$'*0}{'low' if i == 0 else 'mid' if i < k-1 else 'high'}-value)"
             for i, c in enumerate(order)}
    rfm["Cluster_Spend_Label"] = rfm["KMeans_Cluster"].map(names)
    return rfm


# ------------------------------------------------------------------ figures
def plot_rfm(rfm: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, log in zip(axes, ["Recency", "Frequency", "Monetary"], [False, True, True]):
        sns.histplot(rfm[col], bins=40, ax=ax, color="#2b6cb0")
        if log:
            ax.set_xscale("log")
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES, "rfm_distributions.png"), dpi=140)
    plt.close(fig)

    agg = (rfm.groupby("RFM_Segment")
              .agg(Customers=("CustomerID", "count"),
                   Revenue=("Monetary", "sum"),
                   AvgRecency=("Recency", "mean"),
                   AvgFrequency=("Frequency", "mean"),
                   AvgMonetary=("Monetary", "mean"))
              .sort_index())
    agg["RevenueSharePct"] = 100 * agg["Revenue"] / agg["Revenue"].sum()
    agg.round(2).to_csv(os.path.join(cfg.REPORTS, "rfm_segment_summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2f855a", "#3182ce", "#dd6b20", "#c53030"]
    axes[0].bar(range(len(agg)), agg["Customers"], color=colors)
    axes[0].set_xticks(range(len(agg)))
    axes[0].set_xticklabels([i[3:] for i in agg.index], rotation=20, ha="right")
    axes[0].set_title("Customers per RFM segment")
    axes[1].bar(range(len(agg)), agg["RevenueSharePct"], color=colors)
    axes[1].set_xticks(range(len(agg)))
    axes[1].set_xticklabels([i[3:] for i in agg.index], rotation=20, ha="right")
    axes[1].set_title("Revenue share per RFM segment (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES, "rfm_segment_revenue.png"), dpi=140)
    plt.close(fig)
    return agg


if __name__ == "__main__":
    cfg.ensure_dirs()
    tx = load_retail()
    tx = clean_transactions(tx)
    rfm = build_rfm(tx)
    rfm = kmeans_overlay(rfm)
    rfm.to_csv(os.path.join(cfg.DATA_PROCESSED, "rfm_customers.csv"), index=False)

    summary = plot_rfm(rfm)
    print("\nRFM segment summary:")
    print(summary.round(1).to_string())
    print("\nSaved outputs: rfm_customers.csv, rfm_segment_summary.csv, 2 figures")
