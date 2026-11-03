"""
05 — Business insights: generates the executive report (markdown) by reading
     every intermediate output and turning numbers into recommendations.

Usage:  python src/05_business_insights.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

P = lambda n: os.path.join(cfg.REPORTS, n)  # noqa: E731


def load(name, cols=None):
    p = os.path.join(cfg.REPORTS, name)
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df[cols] if cols else df
    return None


def md_table(df):
    d = df.copy()
    for c in d.select_dtypes("float"):
        d[c] = d[c].map(lambda v: f"{v:,.1f}")
    try:
        return d.to_markdown(index=False)
    except ImportError:
        # fallback pipe-grid table (no tabulate needed)
        head = "| " + " | ".join(d.columns) + " |"
        sep = "|" + "|".join([" - "] * len(d.columns)) + "|"
        rows = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False))
        return "\n".join([head, sep, rows])


def main():
    lines = ["# Customer Segmentation & Retention Analysis — Business Report",
             "", f"_All figures are descriptive outputs of the pipeline; "
             f"assumptions: margin {cfg.MARGIN:.0%}, save-rate {cfg.RETENTION_SAVE_RATE:.0%}, "
             f"offer cost ${cfg.RETENTION_OFFER_COST:,.0f}._", ""]

    # ---- retail side
    seg = load("rfm_segment_summary.csv")
    clv = load("clv_summary.csv")
    if seg is not None:
        seg = seg.merge(clv[["RFM_Segment", "clv_avg", "churn_proxy_rate", "expected_loss"]],
                        on="RFM_Segment", how="left") if clv is not None else seg
        lines += ["## Retail: where the money is (RFM)", ""]
        cols = ["RFM_Segment", "Customers", "Revenue", "RevenueSharePct",
                "AvgRecency", "AvgFrequency", "AvgMonetary"]
        cols = [c for c in cols if c in seg.columns]
        if clv is not None:
            cols += ["clv_avg", "churn_proxy_rate", "expected_loss"]
        lines += [md_table(seg[cols].round(1)), ""]
        top = seg.sort_values("Revenue", ascending=False).head(2)
        lines += [f"**Insight:** the top {len(top)} segments generate "
                  f"{top['RevenueSharePct'].sum():.0f}% of revenue with only "
                  f"{100*seg.loc[top.index,'Customers'].sum()/seg['Customers'].sum():.0f}% of customers — "
                  "their retention is disproportionately valuable; fund win-back first for "
                  "high-frequency-but-receding customers ('At Risk'), not for the long tail of "
                  "one-time buyers.", ""]

    # ---- telco side
    model = load("churn_model_metrics.csv")
    if model is not None:
        model = model.rename(columns={model.columns[0]: "Model"})
        best_model = model.sort_values("PR_AUC", ascending=False).iloc[0]["Model"]
        lines += ["## Telecom: churn prediction quality is a means, not the goal", "",
                  md_table(model), ""]
        lines += [f"**Insight:** best model = **{best_model}**. Ranking quality (ROC/PR-AUC) "
                  "only matters insofar as it lets us capture ~2-3x more churners than random "
                  "calling in the top decile. The model becomes a *budget allocation* tool.", ""]

    sim = load("retention_campaign_simulation.csv")
    if sim is not None:
        lines += ["## Telecom: retention campaign economics", "",
                  md_table(sim), ""]
        lines += [f"**Insight:** risk-ranked targeting captures "
                  f"{sim.iloc[1]['Rank_churnersCaughtPct']:.0f}% of churners by calling "
                  f"just 20% of the addressable pool — vs "
                  f"{sim.iloc[1]['Random_churnersCaughtPct']:.1f}% expected from random calls — "
                  "while the 'do-not-target' pool (safe 2-yr contract customers) is excluded "
                  "entirely, avoiding wasted offers.", ""]

    prio = os.path.join(cfg.DATA_PROCESSED, "telco_scored.csv")
    if os.path.exists(prio):
        t = pd.read_csv(prio)
        bucket = t.groupby("PriorityBucket").agg(
            customers=("customerID", "count"),
            exp_loss_total=("ExpectedLoss", "sum"),
            avg_churn_prob=("churn_prob", "mean")).reset_index().sort_values("PriorityBucket")
        lines += ["## Priority buckets (who gets the offer)", "", md_table(bucket), ""]
        a = bucket[bucket["PriorityBucket"].str.startswith("A")]
        if len(a):
            lines += [f"**Recommendation:** concentrate spend on bucket A "
                      f"({int(a['customers'].sum()):,} customers, "
                      f"${a['exp_loss_total'].sum()/1000:,.0f}K value at risk). Month-to-month "
                      "customers with weak tenure and no bundled services = highest addressable "
                      "risk; offer contract-switch incentives + service bundles (Bayes evidence: "
                      "TechSupport/OnlineSecurity presence reduces churn).", ""]

    lines += ["## Unified playbook (both datasets)", "",
              "| Segment type | Signal | Action | Justification |",
              "|---|---|---|---|",
              "| Champions (recent+freq+high $) | Stable RFM 3-3+ | Early feature access, VIP perks | cheapest retention: already engaged |",
              "| At Risk (former loyal) | R low, F high | Proactive win-back offer within 30 days | high CLV x rising churn proxy |",
              "| New / Potential Loyalists | R high, F low | Onboarding nudges + second-purchase incentive | first 90 days predict lifetime behavior |",
              "| High-value telecom, month-to-month | P(churn) top decile | Contract-upgrade discount | ExpectedLoss = P x CLV largest |",
              "| Locked-in + low risk | 2-yr contract, low P | No offer | saving can't happen + risk near zero |", ""]
    lines += ["_Figures: outputs/figures/ — see churn_roc.png, churn_business_output.png, "
              "rfm_segment_revenue.png, retail_value_risk_map.png._"]

    out = os.path.join(cfg.REPORTS, "BUSINESS_REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written -> {out}")


if __name__ == "__main__":
    cfg.ensure_dirs()
    main()
