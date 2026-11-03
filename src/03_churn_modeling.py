"""
03 — Churn prediction on Telco, business-first design.

Stages:
  A. Preprocess (coerce TotalCharges, one-hot encode, stratified split).
  B. Train Logistic Regression (baseline) + XGBoost; evaluate on AUC / PR-AUC.
  C. Convert probabilities -> business outputs:
       * CLV per customer (margin-adjusted expected remaining value on contract horizon)
       * Expected loss = P(churn) x CLV  -> the retention-priority metric
       * Contract-eligibility gate: month-to-month customers are realistically
         addressable; locked-in annual customers are 'do-not-spend' targets.
       * Campaign simulation: expected saved value vs random targeting & cost.

Output: outputs/data/telco_scored.csv, figures, model comparison table.

Usage:  python src/03_churn_modeling.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from data_loader import load_telco

sns.set_theme(style="whitegrid", context="notebook")
RNG = 42


# ------------------------------------------------------------------ A. data
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["ChurnFlag"] = (df["Churn"].str.strip() == "Yes").astype(int)

    num = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat = [c for c in df.columns
           if c not in num + ["customerID", "Churn", "ChurnFlag", "TotalCharges"]
           and df[c].dtype == object]
    X = df[num + cat]
    y = df["ChurnFlag"].values

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    Xtr, Xte, ytr, yte, id_tr, id_te = train_test_split(
        X, y, df["customerID"], test_size=0.25, stratify=y, random_state=RNG)
    return pre, X, y, df, Xtr, Xte, ytr, yte, id_tr, id_te


def build_models():
    return {
        "LogisticRegression": Pipeline([("pre", None),
                                        ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "XGBoost": Pipeline([("pre", None),
                             ("clf", __import__("xgboost", fromlist=["xgb"]).XGBClassifier(
                                 n_estimators=400, max_depth=4, learning_rate=0.08,
                                 subsample=0.9, colsample_bytree=0.8,
                                 eval_metric="auc", random_state=RNG, n_jobs=-1))]),
    }


# ------------------------------------------------------------------ B. evaluate
def evaluate(model, name, Xte, yte, results, curves):
    p = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    apr = average_precision_score(yte, p)
    results[name] = {"ROC_AUC": round(float(auc), 4), "PR_AUC": round(float(apr), 4),
                     "positive_rate": round(float(yte.mean()), 4)}
    fpr, tpr, _ = roc_curve(yte, p)
    curves[name] = (fpr, tpr, p)
    print(f"  {name:<20} ROC-AUC {auc:.3f} | PR-AUC {apr:.3f} | baseline P(churn) {yte.mean():.3f}")
    return p


def cumulative_capture(y_true, y_score):
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    total = y_true.sum()
    cum = np.cumsum(y_sorted) / total
    return np.concatenate([[0], cum])


# ------------------------------------------------------------------ C. business layer
def add_business_layer(df):
    """CLV is margin x expected annuitised value. Keep assumptions explicit & conservative."""
    c = cfg
    df = df.copy()
    df["churn_prob"] = np.clip(df["churn_prob"], 0.002, 0.995)

    # expected remaining lifetime: half-life intuition -> remaining tenure ~ ln2 / hazard
    # capped: month-to-month 3yr, contract terms biased by tenure position
    cap = np.where(df["Contract"] == "Two year", 6.0,
          np.where(df["Contract"] == "One year", 4.0, 3.0))
    expected_years = np.minimum(1.0 / df["churn_prob"], c.ANALYSIS_HORIZON_YEARS + 3.0)
    expected_years = np.minimum(expected_years, cap)
    df["ExpectedRemainingYears"] = expected_years.round(2)

    df["CLV"] = (df["MonthlyCharges"] * 12 * c.MARGIN * df["ExpectedRemainingYears"]).round(2)
    df["ExpectedLoss"] = (df["churn_prob"] * df["CLV"]).round(2)

    # eligibility gate: only month-to-month + basic tenure is realistically addressable
    addressable = df["Contract"].isin(["Month-to-month", "One year"])
    df["RetentionEligible"] = addressable.astype(int)
    df["PriorityBucket"] = "C — do not spend (safe or locked in)"
    df.loc[(df["ExpectedLoss"] >= df["ExpectedLoss"].quantile(0.50)) & addressable,
           "PriorityBucket"] = "B — monitor, soft-touch offers"
    df.loc[(df["ExpectedLoss"] >= df["ExpectedLoss"].quantile(0.80)) & addressable,
           "PriorityBucket"] = "A — immediate retention action"
    return df


def campaign_simulation(df):
    """Compare risk-ranked targeting vs random + who NOT to target."""
    r = df[df["RetentionEligible"] == 1].copy()
    r["will_churn"] = df.loc[r.index, "ChurnFlag"]
    month_to_month = r  # addressable pool
    n = len(month_to_month)

    ranked = month_to_month.sort_values("churn_prob", ascending=False)
    total_churners = month_to_month["will_churn"].sum()
    base_rate = month_to_month["will_churn"].mean()
    rows = []
    for frac in (0.10, 0.20, 0.30):
        take = ranked.head(int(frac * n))
        caught = take["will_churn"].sum()
        random_caught = base_rate * int(frac * n)  # churners a RANDOM call list of same size finds
        saved = caught * cfg.RETENTION_SAVE_RATE
        gain = saved * take["CLV"].mean() - len(take) * cfg.RETENTION_OFFER_COST
        rows.append({
            "TargetSharePct": frac * 100,
            "Rank_churnersCaught": int(caught),
            "Rank_churnersCaughtPct": round(100 * caught / total_churners, 1),
            "Random_churnersCaughtPct": round(100 * random_caught / total_churners, 1),
            "SavedValue_USD": round(saved * take["CLV"].mean(), 0),
            "OfferCost_USD": round(len(take) * cfg.RETENTION_OFFER_COST, 0),
            "NetValue_USD": round(gain, 0),
        })
    sim = pd.DataFrame(rows)
    sim.to_csv(os.path.join(cfg.REPORTS, "retention_campaign_simulation.csv"), index=False)

    wasted = df[(df["churn_prob"] < 0.15) & (df["Contract"] == "Two year")]
    print("\nCampaign simulation (month-to-month + 1yr contract pool only):")
    print(sim.to_string(index=False))
    print(f"\n'Do not target' group: {len(wasted):,} happy locked-in 2yr customers "
          f"({100*len(wasted)/len(df):.1f}% of base) - near-zero churn risk + unable to leave")
    return sim, wasted


# ------------------------------------------------------------------ figures
def plot_all(curves, yte, results, df):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (fpr, tpr, _) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['ROC_AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Churn model ROC — Telco test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES, "churn_roc.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    base = df["ChurnFlag"].mean()
    xs = np.linspace(0, 1, 201)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, label="random")
    for name, (_, _, p) in curves.items():
        caps = cumulative_capture(yte, p)
        share = np.linspace(0, 1, len(caps))
        axes[0].plot(xs, np.interp(xs, share, caps), label=f"{name} (top-10% = "
                     f"{np.interp(0.1, share, caps):.0%})")
    axes[0].set(xlabel="Share of customers targeted", ylabel="Share of churners captured",
                title="Cumulative gains — ranked vs random targeting")
    axes[0].legend()

    pt = df.groupby("PriorityBucket").agg(
        customers=("customerID", "count"),
        expected_loss=("ExpectedLoss", "sum")).sort_index()
    pt["expected_loss"] /= 1000
    axes[1].barh(pt.index, pt["expected_loss"], color=["#c53030", "#dd6b20", "#48bb78"])
    axes[1].set(xlabel="Total expected loss ($K)", title="Retention priority buckets — value at risk")
    for i, v in enumerate(pt["expected_loss"]):
        axes[1].text(v, i, f"  ${v:,.0f}K", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES, "churn_business_output.png"), dpi=140)
    plt.close(fig)


def feature_importance(fitted_xgb, X):
    try:
        names = fitted_xgb.named_steps["pre"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        imp = fitted_xgb.named_steps["clf"].feature_importances_
        fi = pd.Series(imp, index=names).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fi[::-1].plot.barh(ax=ax, color="#2b6cb0")
        ax.set(title="Top churn drivers (XGBoost gain-based importance)")
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.FIGURES, "churn_top_features.png"), dpi=140)
        plt.close(fig)
        fi.round(3).to_csv(os.path.join(cfg.REPORTS, "churn_feature_importance.csv"))
        return fi
    except Exception as e:
        print(f"[warn] feature importance skipped: {e}")
        return None


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    cfg.ensure_dirs()
    print("Loading Telco data ...")
    raw = load_telco()
    pre, X, y, df, Xtr, Xte, ytr, yte, id_tr, id_te = preprocess(raw)
    print(f"  {len(df):,} customers | churn rate {y.mean():.1%} | "
          f"train {len(Xtr):,} / test {len(Xte):,}")

    results, curves, preds = {}, {}, {}
    for name, model in build_models().items():
        print(f"\nTraining {name} ...")
        model.set_params(pre=pre)
        model.fit(Xtr, ytr)
        p = evaluate(model, name, Xte, yte, results, curves)
        preds[name] = p
        if name == "XGBoost":
            fitted_xgb = model
            feature_importance(model, X)

    pd.DataFrame(results).T.to_csv(os.path.join(cfg.REPORTS, "churn_model_metrics.csv"))

    # score the FULL base with the best model (XGBoost), business layer added
    print("\nScoring full base + business layer ...")
    all_scores = fitted_xgb.predict_proba(X)[:, 1]
    out = df[["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
              "Contract", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
              "ChurnFlag"]].copy()
    out["churn_prob"] = np.round(all_scores, 4)
    out = add_business_layer(out)
    out.to_csv(os.path.join(cfg.DATA_PROCESSED, "telco_scored.csv"), index=False)

    plot_all(curves, yte, results, out)
    campaign_simulation(out)
    print("\nSaved: telco_scored.csv, churn_model_metrics.csv, campaign simulation, 3 figures")
