# Survival Analysis – Telecom Churn

**Homework 3 | Norayr Harutyunyan**

Parametric survival analysis on a telecom subscriber dataset to model churn risk, estimate Customer Lifetime Value (CLV), and guide retention strategy.

---

## Overview

This project applies **Accelerated Failure Time (AFT)** models to understand what drives subscriber churn in a telecom company. Three distributions are compared (Weibull, Log-Normal, Log-Logistic), a final model is selected on AIC/BIC, CLV is computed per customer, and an annual retention budget is estimated.

---

## Repository Structure

```
.
├── data/
│   ├── telco.csv            # Raw dataset (1,000 subscribers, 15 variables)
│   └── clv_output.csv       # Per-customer CLV and 12-month survival output
├── plots/                   # All generated visualisations
├── notebook.ipynb           # Full analysis notebook (report + code)
├── analysis.py              # Standalone Python analysis script
├── requirements.txt
└── README.md
```

---

## Dataset Variables

| Variable | Description |
|---|---|
| `ID` | Subscriber ID |
| `region` | Geographic zone (Zone 1/2/3) |
| `tenure` | Months as a subscriber (survival time) |
| `age` | Subscriber age |
| `marital` | Married / Unmarried |
| `address` | Years at current address |
| `income` | Annual income (thousands USD) |
| `ed` | Education level |
| `retire` | Retired (Yes/No) |
| `gender` | Male/Female |
| `voice` | Voice add-on (Yes/No) |
| `internet` | Internet service (Yes/No) |
| `forward` | Call forwarding (Yes/No) |
| `custcat` | Customer category (Basic/Plus/E-service/Total) |
| `churn` | Churned (Yes = event, No = censored) |

---

## Methods

- **Kaplan–Meier** non-parametric baseline curves by segment
- **AFT Models**: Weibull, Log-Normal, Log-Logistic (via `lifelines`)
- **Model selection**: AIC / BIC / Log-likelihood comparison
- **Feature selection**: Keep features with p < 0.05
- **CLV**: `monthly_ARPU × predicted_median_tenure`
- **Retention budget**: 10% of cumulative CLV of bottom-quartile survival subscribers

---

## Key Results

| Model | AIC | BIC | Concordance |
|---|---|---|---|
| **Log-Normal** ✔ | **2944** | **2938** | **0.78** |
| Log-Logistic | 2956 | 2930 | 0.77 |
| Weibull | 2964 | 2938 | 0.76 |

**Significant predictors of longer survival**: customer category (E/Plus/Total service), age, residential stability.  
**Significant predictors of faster churn**: internet service, unmarried status, voice add-on.

**Most valuable segment**: Married Plus/Total-service subscribers aged 50+ with stable residential history.  
**Recommended annual retention budget**: ~$14,700 (targeting 131 at-risk active subscribers).

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis script
python analysis.py

# Or open the notebook
jupyter lab notebook.ipynb
```

---

## Report Summary

The Log-Normal AFT model is chosen for its best fit and interpretable, non-monotone hazard structure. Customer category is the single strongest predictor of churn: subscribers on bundled or premium plans stay 2–3× longer. Counterintuitively, internet and voice add-on subscribers churn faster, suggesting unmet expectations from multi-service bundles. Retention efforts should prioritise Basic-service subscribers at early tenure stages and offer proactive outreach to single, younger customers.
