# Customer Segmentation & Retention Analysis — Business Report

_All figures are descriptive outputs of the pipeline; assumptions: margin 30%, save-rate 25%, offer cost $75._

## Retail: where the money is (RFM)

| RFM_Segment                  |   Customers |         Revenue |   RevenueSharePct |   AvgRecency |   AvgFrequency |   AvgMonetary |   clv_avg |   churn_proxy_rate |   expected_loss |
|:-----------------------------|------------:|----------------:|------------------:|-------------:|---------------:|--------------:|----------:|-------------------:|----------------:|
| 01 Champions & Loyal         |        1000 |      2.5806e+06 |              69.6 |         48.5 |            5   |        2580.6 |    1036.9 |                0   |               0 |
| 02 Potential Loyalists / New |         498 | 221018          |               6   |         59.5 |            1   |         443.8 |     266.3 |                0   |               0 |
| 03 At Risk / Can't Lose      |         498 | 553316          |              14.9 |        204.6 |            2.8 |        1111.1 |     638.9 |                0.6 |          189110 |
| 04 Hibernating / Lost        |        1000 | 352087          |               9.5 |        285.6 |            1   |         352.1 |     211.2 |                0.8 |          173016 |

**Insight:** the top 2 segments generate 85% of revenue with only 50% of customers — their retention is disproportionately valuable; fund win-back first for high-frequency-but-receding customers ('At Risk'), not for the long tail of one-time buyers.

## Telecom: churn prediction quality is a means, not the goal

| Model              |   ROC_AUC |   PR_AUC |   positive_rate |
|:-------------------|----------:|---------:|----------------:|
| LogisticRegression |       0.8 |      0.6 |             0.3 |
| XGBoost            |       0.8 |      0.6 |             0.3 |

**Insight:** best model = **LogisticRegression**. Ranking quality (ROC/PR-AUC) only matters insofar as it lets us capture ~2-3x more churners than random calling in the top decile. The model becomes a *budget allocation* tool.

## Telecom: retention campaign economics

|   TargetSharePct |   Rank_churnersCaught |   Rank_churnersCaughtPct |   Random_churnersCaughtPct |   SavedValue_USD |   OfferCost_USD |   NetValue_USD |
|-----------------:|----------------------:|-------------------------:|---------------------------:|-----------------:|----------------:|---------------:|
|               10 |                   485 |                     26.6 |                         10 |            42252 |           40050 |           2202 |
|               20 |                   900 |                     49.4 |                         20 |            85192 |           80175 |           5017 |
|               30 |                  1224 |                     67.2 |                         30 |           125637 |          120300 |           5337 |

**Insight:** risk-ranked targeting captures 49% of churners by calling just 20% of the addressable pool — vs 20.0% expected from random calls — while the 'do-not-target' pool (safe 2-yr contract customers) is excluded entirely, avoiding wasted offers.

## Priority buckets (who gets the offer)

| PriorityBucket                       |   customers |   exp_loss_total |   avg_churn_prob |
|:-------------------------------------|------------:|-----------------:|-----------------:|
| A — immediate retention action       |        1338 |           450911 |              0.6 |
| B — monitor, soft-touch offers       |        2005 |           403196 |              0.4 |
| C — do not spend (safe or locked in) |        3700 |           155511 |              0.1 |

**Recommendation:** concentrate spend on bucket A (1,338 customers, $451K value at risk). Month-to-month customers with weak tenure and no bundled services = highest addressable risk; offer contract-switch incentives + service bundles (Bayes evidence: TechSupport/OnlineSecurity presence reduces churn).

## Unified playbook (both datasets)

| Segment type | Signal | Action | Justification |
|---|---|---|---|
| Champions (recent+freq+high $) | Stable RFM 3-3+ | Early feature access, VIP perks | cheapest retention: already engaged |
| At Risk (former loyal) | R low, F high | Proactive win-back offer within 30 days | high CLV x rising churn proxy |
| New / Potential Loyalists | R high, F low | Onboarding nudges + second-purchase incentive | first 90 days predict lifetime behavior |
| High-value telecom, month-to-month | P(churn) top decile | Contract-upgrade discount | ExpectedLoss = P x CLV largest |
| Locked-in + low risk | 2-yr contract, low P | No offer | saving can't happen + risk near zero |

_Figures: outputs/figures/ — see churn_roc.png, churn_business_output.png, rfm_segment_revenue.png, retail_value_risk_map.png._