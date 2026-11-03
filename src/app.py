"""
Interactive site: Customer Segmentation & Retention Analysis
Run:  streamlit run src/app.py
Requires the pipeline outputs (data/processed/*.csv) -> python src/00_run_all.py
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

st.set_page_config(page_title="Retention Intelligence", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")

SEG_COLORS = ["#2f855a", "#3182ce", "#dd6b20", "#c53030"]


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    out = {}
    p_rfm = os.path.join(cfg.DATA_PROCESSED, "rfm_customers.csv")
    p_tel = os.path.join(cfg.DATA_PROCESSED, "telco_scored.csv")
    if os.path.exists(p_rfm):
        out["rfm"] = pd.read_csv(p_rfm)
    if os.path.exists(p_tel):
        out["telco"] = pd.read_csv(p_tel)
    return out


D = load()
if "rfm" not in D or "telco" not in D:
    st.error("Missing pipeline outputs. Run `python src/00_run_all.py` first.")
    st.stop()

rfm, telco = D["rfm"], D["telco"]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Retention Intelligence")
page = st.sidebar.radio("Navigate", [
    "① Executive overview",
    "② Retail — RFM segmentation",
    "③ Telecom — churn risk",
    "④ Retention campaign simulator",
    "⑤ Unified playbook",
], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(f"Margin **{cfg.MARGIN:.0%}** · save-rate **{cfg.RETENTION_SAVE_RATE:.0%}** "
                   f"· offer cost **${cfg.RETENTION_OFFER_COST:,.0f}**")
st.sidebar.caption("Datasets: Kaggle Telco Customer Churn & UCI Online Retail")

# ================================================================== ① OVERVIEW
if page.startswith("①"):
    st.title("Executive overview")
    st.caption("Who is about to leave, what they are worth, and where the retention budget goes.")

    k1, k2, k3, k4 = st.columns(4)
    tel_churn = telco["ChurnFlag"].mean()
    risk_pool = telco[(telco["RetentionEligible"] == 1) & (telco["churn_prob"] >= 0.5)]
    k1.metric("Retail customers", f"{len(rfm):,}")
    k2.metric("Retail revenue top-2 segments",
              f"{rfm.groupby('RFM_Segment')['Monetary'].sum().nlargest(2).sum()/1e6:.1f}M £",
              f"{100*rfm.groupby('RFM_Segment')['Monetary'].sum().nlargest(2).sum()/rfm['Monetary'].sum():.0f}% of total")
    k3.metric("Telecom churn rate", f"{tel_churn:.1%}", "7,043 customers")
    k4.metric("Value at risk (telco)", f"${risk_pool['ExpectedLoss'].sum()/1e3:,.0f}K",
              f"{len(risk_pool):,} high-risk addressable")

    c1, c2 = st.columns(2)
    with c1:
        seg = rfm.groupby("RFM_Segment").agg(
            Customers=("CustomerID", "count"), Revenue=("Monetary", "sum")).reset_index()
        seg["RevenueSharePct"] = 100 * seg["Revenue"] / seg["Revenue"].sum()
        fig = go.Figure()
        fig.add_bar(x=seg["RFM_Segment"], y=seg["Customers"], name="Customers", marker_color="#3182ce")
        fig.add_scatter(x=seg["RFM_Segment"], y=seg["RevenueSharePct"], name="Revenue share %",
                        yaxis="y2", marker_color="#c53030")
        fig.update_layout(title="Retail: revenue is concentrated in few segments",
                          yaxis=dict(title="Customers"), yaxis2=dict(overlaying="y", side="right",
                          title="Revenue share (%)"), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pct = telco["churn_prob"].rank(pct=True)
        fig = px.histogram(telco, x="churn_prob", color=np.where(pct >= 0.8, "A — top decile risk",
                           np.where(telco["RetentionEligible"] == 1, "B — addressable", "C — do-not-spend")),
                           nbins=40, color_discrete_map={"A — top decile risk": "#c53030",
                           "B — addressable": "#dd6b20", "C — do-not-spend": "#48bb78"})
        fig.update_layout(title="Telecom: predicted churn probability distribution",
                          xaxis_title="P(churn)", yaxis_title="Customers",
                          legend_title="Bucket")
        st.plotly_chart(fig, use_container_width=True)

    st.info("**The one-line story:** ~33% of retail customers drive ~70% of revenue, and in telecom "
            "calling the risk-ranked top 20% of addressable customers catches ~50% of all churners — "
            "2.5× better than random outreach. Everything below is the evidence.")

# ================================================================== ② RETAIL RFM
elif page.startswith("②"):
    st.title("Retail — RFM segmentation")
    st.caption("2,996 customers · 168k clean transactions (Of ~540k raw; returns/cancelled/anonymous removed)")

    segs = st.multiselect("Filter segments", sorted(rfm["RFM_Segment"].unique()),
                          default=sorted(rfm["RFM_Segment"].unique()))
    f = rfm[rfm["RFM_Segment"].isin(segs)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(f):,}")
    k2.metric("Total spend", f"£{f['Monetary'].sum()/1e6:.2f}M")
    k3.metric("Avg recency", f"{f['Recency'].mean():.0f} days")
    k4.metric("Avg CLV (3-yr, margin-adj)", f"£{f['CLV'].mean():,.0f}")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.scatter(f.sample(min(len(f), 4000), random_state=0),
                         x="Recency", y="Monetary", color="RFM_Segment",
                         color_discrete_sequence=SEG_COLORS, hover_data=["Frequency", "CLV"],
                         log_y=True, opacity=0.6,
                         title="Customer map — recency vs monetary (log)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        agg_ = f.groupby("RFM_Segment")["CLV"].mean().reset_index()
        fig = px.bar(agg_, x="RFM_Segment", y="CLV", color="RFM_Segment",
                     color_discrete_sequence=SEG_COLORS,
                     title="Average CLV by segment (£)")
        st.plotly_chart(fig, use_container_width=True)

    show = (f.groupby("RFM_Segment")
            .agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"),
                 AvgRecency=("Recency", "mean"), AvgFrequency=("Frequency", "mean"),
                 AvgMonetary=("Monetary", "mean"), AvgCLV=("CLV", "mean"))
            .sort_index().reset_index())
    st.dataframe(show, use_container_width=True, hide_index=True)

# ================================================================== ③ TELECOM CHURN
elif page.startswith("③"):
    st.title("Telecom — churn risk & priority buckets")
    st.caption("Scored on the full 7,043-customer base · Expected Loss = P(churn) × CLV")

    bucket = st.selectbox("Priority bucket", [
        "All", "A — immediate retention action", "B — monitor, soft-touch offers",
        "C — do not spend (safe or locked in)"])
    f = telco if bucket == "All" else telco[telco["PriorityBucket"] == bucket]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(f):,}")
    k2.metric("Avg P(churn)", f"{f['churn_prob'].mean():.1%}")
    k3.metric("Avg CLV", f"${f['CLV'].mean():,.0f}")
    k4.metric("Total expected loss", f"${f['ExpectedLoss'].sum()/1e3:,.0f}K")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(f.sample(min(len(f), 4000), random_state=0), x="tenure",
                         y="MonthlyCharges", color="PriorityBucket", size="ExpectedLoss",
                         color_discrete_map={"A — immediate retention action": "#c53030",
                         "B — monitor, soft-touch offers": "#dd6b20",
                         "C — do not spend (safe or locked in)": "#48bb78"},
                         hover_data=["Contract", "churn_prob"], opacity=0.6,
                         title="Risk landscape — bubble size = Expected Loss ($)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fi_path = os.path.join(cfg.REPORTS, "churn_feature_importance.csv")
        if os.path.exists(fi_path):
            fi = pd.read_csv(fi_path, header=None, names=["feature", "importance"])
            fig = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                         orientation="h", title="Top churn drivers (XGBoost)")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest expected-loss customers (the call list)")
    cols = ["customerID", "Contract", "PaymentMethod", "tenure", "MonthlyCharges",
            "churn_prob", "CLV", "ExpectedLoss", "PriorityBucket"]
    top = f.sort_values("ExpectedLoss", ascending=False).head(500)[cols]
    st.dataframe(top, use_container_width=True, hide_index=True, height=380)
    st.download_button("Download call list (CSV)", top.to_csv(index=False).encode(),
                       "retention_call_list.csv", "text/csv")

# ================================================================== ④ SIMULATOR
elif page.startswith("④"):
    st.title("Retention campaign simulator")
    st.caption("Re-ranks the *addressable* telecom pool (month-to-month + 1-yr contracts) live.")

    c1, c2, c3 = st.columns(3)
    share = c1.slider("Outreach depth (% of pool)", 5, 50, 20, 5) / 100
    save_rate = c2.slider("Save rate (share of targets retained)", 0, 100, 25, 5) / 100
    offer_cost = c3.slider("Offer cost per customer ($)", 10, 300, 75, 5)

    st.caption("Assumed value per saved customer = CLV of the targeted group (margin-adjusted).")
    ranked = telco[telco["RetentionEligible"] == 1].sort_values("churn_prob", ascending=False)
    n = len(ranked)
    take_n = int(share * n)
    take = ranked.head(take_n)
    caught = take["ChurnFlag"].sum()
    total_churners = ranked["ChurnFlag"].sum()
    random_caught = ranked["ChurnFlag"].mean() * take_n
    saved_val = caught * save_rate * take["CLV"].mean()
    cost = take_n * offer_cost
    net = saved_val - cost

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers contacted", f"{take_n:,}")
    k2.metric("Churners caught", f"{int(caught):,}",
              f"{100*caught/total_churners:.0f}% of all churners")
    k3.metric("vs random targeting", f"{100*caught/total_churners:.0f}%",
              f"+{100*(caught-random_caught)/total_churners:.0f} pts", delta_color="off")
    k4.metric("Saved value", f"${saved_val/1e3:,.0f}K")
    k5.metric("Net campaign value", f"${net/1e3:,.0f}K", "profitable" if net > 0 else "at a loss")

    # sweep chart
    depths, nets, caps = [], [], []
    for s in np.arange(0.05, 0.51, 0.01):
        t = ranked.head(int(s * n))
        c_ = t["ChurnFlag"].sum()
        caps.append(100 * c_ / total_churners)
        nets.append(c_ * save_rate * t["CLV"].mean() - int(s * n) * offer_cost)
        depths.append(s * 100)
    fig = go.Figure()
    fig.add_scatter(x=depths, y=nets, name="Net value ($)", marker_color="#c53030")
    fig.add_vline(x=share * 100, line_dash="dot", annotation_text="your setting")
    fig.update_layout(title="Campaign net value vs outreach depth",
                      xaxis_title="Outreach depth (%)", yaxis_title="Net value ($)")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_scatter(x=depths, y=caps, name="Risk-ranked", marker_color="#2b6cb0")
    fig2.add_scatter(x=depths, y=depths * ranked["ChurnFlag"].mean() * n / total_churners * 100,
                     name="Random", line=dict(dash="dash"), marker_color="gray")
    fig2.update_layout(title="Cumulative churner capture", xaxis_title="Outreach depth (%)",
                       yaxis_title="% of churners captured")
    st.plotly_chart(fig2, use_container_width=True)

# ================================================================== ⑤ PLAYBOOK
else:
    st.title("Unified playbook")
    rows = [
        ("Champions (recent + frequent + high spend)", "Stable RFM 3-3+",
         "Early access, VIP perks", "Cheapest retention — already engaged"),
        ("At Risk (former loyal)", "R low, F high", "Proactive win-back within 30 days",
         "High CLV × rising churn proxy — ~£189K at risk"),
        ("Potential Loyalists (new)", "R high, F low", "Onboarding nudges + 2nd purchase incentive",
         "Early behavior predicts lifetime value"),
        ("High-value month-to-month telecom", "P(churn) top decile", "Contract-upgrade discount",
         "Largest Expected Loss ($451K bucket A)"),
        ("Locked-in low-risk (2-yr contract)", "Low P(churn)", "No offer",
         "Can't leave + won't leave — spend is wasted"),
    ]
    pb = pd.DataFrame(rows, columns=["Segment", "Signal", "Action", "Justification"])
    st.dataframe(pb, use_container_width=True, hide_index=True)

    st.subheader("Why 'do-not-target' matters")
    wasted = telco[(telco["churn_prob"] < 0.15) & (telco["Contract"] == "Two year")]
    k1, k2 = st.columns(2)
    k1.metric("Excluded customers", f"{len(wasted):,}", f"{100*len(wasted)/len(telco):.0f}% of base")
    k2.metric("Offer spend avoided",
              f"${len(wasted)*cfg.RETENTION_OFFER_COST/1e3:,.0f}K",
              "if nothing else changes, that money goes to bucket A instead")
    st.info("A churn model that merely predicts well is a cost center. A churn model that **ranks "
            "customers by expected loss and gates spend by eligibility** is a budget allocation tool — "
            "that is the thesis of this project.", icon="🎯")
