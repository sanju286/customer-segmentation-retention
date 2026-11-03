# Customer Segmentation & Retention Analysis

RFM segmentation + churn prediction + CLV-driven retention targeting, built on two
real-world Kaggle datasets:

| Track | Dataset | What it demonstrates |
|---|---|---|
| A — Overseas retail (B2C transactions) | [vijayuv/onlineretail](https://www.kaggle.com/datasets/vijayuv/onlineretail) (~540k transactions) | RFM analysis, KMeans overlay, churn proxy, retail CLV |
| B — Telecom (subscription) | [blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers) | Churn modeling (Logistic Regression vs XGBoost), Expected Loss = P(churn) × CLV, campaign ROI simulation |

The thesis of the project: **models are only valuable if they change where a business
spends its retention budget.** Every model output is therefore converted into a
decision artifact — priority buckets, a churner-capture table vs random targeting,
and an explicit *do-not-target* group.

## Project structure

```
customer-segmentation-retention/
├── data/
│   ├── raw/                  # put Kaggle downloads here
│   └── processed/            # scored customer tables (output)
├── outputs/
│   ├── figures/              # charts (output)
│   └── reports/              # segment summaries, metrics, BUSINESS_REPORT.md (output)
└── src/
    ├── 00_run_all.py          # full pipeline
    ├── 01_data_acquisition.py # download/validates Kaggle data
    ├── 02_rfm_segmentation.py # cleaning -> RFM scores -> segments (+KMeans)
    ├── 03_churn_modeling.py   # churn models -> scoring -> CLV -> campaign simulation
    ├── 04_clv_estimation.py   # CLV both tracks + value-at-risk map
    ├── 05_business_insights.py# executive markdown report
    ├── config.py              # paths + business assumptions (margin, save rate, offer cost)
    └── data_loader.py
```

## Interactive site

```bash
pip install streamlit plotly
streamlit run src/app.py            # opens at http://localhost:8501
```

Five live pages (all charts are Plotly — hover, zoom, filter):

1. **Executive overview** — KPI cards + concentration & risk charts
2. **Retail RFM** — segment filter, recency×monetary customer map, CLV by segment
3. **Telecom churn risk** — priority-bucket filter, risk landscape, top churn drivers,
   **downloadable retention call list** (top 500 by Expected Loss)
4. **Campaign simulator** — sliders for outreach depth / save rate / offer cost;
   net value and churner-capture recompute live
5. **Unified playbook** — segment → action → justification, plus the 
   "do-not-target" savings estimate

## Getting started

```bash
pip install -r requirements.txt

# Option 1 — automatic (needs Kaggle account once: https://www.kaggle.com/settings
#             -> API -> Create New Token for kagglehub credentials)
python src/01_data_acquisition.py

# Option 2 — manual
#   download Telco-Customer-Churn.csv  and  OnlineRetail.xlsx (unzip)
#   into data/raw/

python src/00_run_all.py          # runs steps 01..05 end to end
```

Open **`outputs/reports/BUSINESS_REPORT.md`** last — it's the executive synthesis.

## Key business logic (auditable in `config.py`)

- **CLV** = margin × annuitized revenue. Telecom: `MonthlyCharges × 12 × margin ×
  expected remaining years` where expected remaining years is derived from churn
  probability and capped by contract type (2-yr contracts can't churn before expiry).
- **Expected Loss** = `P(churn) × CLV` — the retention priority metric.
- **Eligibility gate**: month-to-month and 1-yr customers are realistically
  addressable; happy 2-yr customers are *do-not-spend* (near-zero risk AND locked in).
- **Campaign simulation**: top-10/20/30% risk-ranked targeting vs random calling,
  using documented for save-rate (25%) and offer cost ($75) assumptions.

## Headline outputs to highlight in a portfolio/interview

1. `rfm_segment_revenue.png` — Pareto: a few segments carry most revenue.
2. `churn_business_output.png` — cumulative gains (rank > random) + value-at-risk by bucket.
3. `retail_value_risk_map.png` — high-value customers fading out (right tail).
4. `outputs/reports/retention_campaign_simulation.csv` — the ROI table.
5. `outputs/reports/BUSINESS_REPORT.md` — the recommendation writeup.

## Caveats / extension ideas

- Retail churn is a *proxy* (no purchase in 180 days); a probabilistic
  BG/NBD-GG model (`lifetimes` library) would refine CLV.
- Telecom data is single-snapshot — time-based features (usage trend) would lift AUC.
- Campaign economics are simulated on assumptions; an A/B uplift test is the real
  thing to run in production.
