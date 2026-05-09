"""
Homework 3 – Survival Analysis
Karen Hovhannisyan
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from lifelines import (
    WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter,
    ExponentialFitter, WeibullFitter, KaplanMeierFitter
)
from lifelines.statistics import logrank_test
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

# ── 0. Settings ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
PALETTE = sns.color_palette("tab10")

# ── 1. Load & Prepare Data ─────────────────────────────────────────────────────
df = pd.read_csv("data/telco.csv")
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Binary target: churn event (1 = churned, 0 = censored/still active)
df["event"] = (df["churn"] == "Yes").astype(int)
df["duration"] = df["tenure"]

# Encode categoricals for modelling
cat_cols = ["region", "marital", "ed", "retire", "gender",
            "voice", "internet", "forward", "custcat"]
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df_enc.columns = [c.replace(" ", "_").replace("-", "_") for c in df_enc.columns]

FEATURES = [c for c in df_enc.columns
            if c not in ("ID", "tenure", "churn", "event", "duration")]

print(f"Features for modelling: {FEATURES}\n")

# ── 2. Kaplan-Meier Overview ───────────────────────────────────────────────────
kmf = KaplanMeierFitter()
kmf.fit(df["duration"], event_observed=df["event"], label="All customers")

fig, ax = plt.subplots(figsize=(8, 5))
kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[0])
ax.set_title("Kaplan–Meier Survival Curve – All Customers")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Survival Probability")
ax.axvline(12, ls="--", color="grey", lw=1, label="12 months")
ax.legend()
plt.tight_layout()
plt.savefig("plots/01_km_overall.png")
plt.close()
print("Saved plots/01_km_overall.png")

# KM by custcat
fig, ax = plt.subplots(figsize=(9, 5))
for i, seg in enumerate(df["custcat"].unique()):
    mask = df["custcat"] == seg
    kmf_seg = KaplanMeierFitter()
    kmf_seg.fit(df.loc[mask, "duration"], df.loc[mask, "event"], label=seg)
    kmf_seg.plot_survival_function(ax=ax, ci_show=False, color=PALETTE[i])
ax.set_title("Kaplan–Meier by Customer Category")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Survival Probability")
plt.tight_layout()
plt.savefig("plots/02_km_custcat.png")
plt.close()
print("Saved plots/02_km_custcat.png")

# ── 3. AFT Models ─────────────────────────────────────────────────────────────
covariates = df_enc[FEATURES + ["duration", "event"]].copy()

aft_models = {
    "Weibull":      WeibullAFTFitter(),
    "LogNormal":    LogNormalAFTFitter(),
    "LogLogistic":  LogLogisticAFTFitter(),
}

fitted = {}
metrics = {}

for name, model in aft_models.items():
    model.fit(covariates, duration_col="duration", event_col="event")
    fitted[name] = model
    metrics[name] = {
        "AIC":  model.AIC_,
        "BIC":  model.BIC_,
        "log-likelihood": model.log_likelihood_,
    }
    print(f"\n{'='*60}")
    print(f"  {name} AFT Model")
    print(f"{'='*60}")
    print(f"  AIC: {model.AIC_:.2f}   BIC: {model.BIC_:.2f}   LogLik: {model.log_likelihood_:.2f}")

metrics_df = pd.DataFrame(metrics).T.sort_values("AIC")
print("\n── Model Comparison ──────────────────────────────────────")
print(metrics_df.to_string())

# ── 4. Visualise All AFT Survival Curves in One Plot ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = {"Weibull": PALETTE[0], "LogNormal": PALETTE[1], "LogLogistic": PALETTE[2]}

# Plot baseline (mean covariate) survival for each model
t_range = np.linspace(1, 72, 200)
for name, model in fitted.items():
    # Predict median survival function on mean observation
    mean_row = covariates[FEATURES].mean(numeric_only=True).to_frame().T
    sf = model.predict_survival_function(mean_row, times=t_range)
    ax.plot(t_range, sf.values.flatten(), label=name, color=colors[name], lw=2)

# Also overlay KM
kmf.plot_survival_function(ax=ax, ci_show=False, color="black",
                            linestyle="--", label="Kaplan–Meier")
ax.set_title("AFT Survival Curves – All Models (Mean Covariate Profile)")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Survival Probability")
ax.legend()
plt.tight_layout()
plt.savefig("plots/03_aft_all_curves.png")
plt.close()
print("\nSaved plots/03_aft_all_curves.png")

# ── 5. Select Best Model & Keep Significant Features ──────────────────────────
best_name = metrics_df.index[0]
best_model = fitted[best_name]
print(f"\n✔ Best model by AIC: {best_name}")

# Extract p-values from lambda_ sub-model (location parameter)
summary = best_model.summary
sig_features = summary[summary["p"] < 0.05].index.get_level_values(-1).tolist()
# Remove intercept
sig_features = [f for f in sig_features if f not in ("Intercept", "intercept")]
print(f"Significant features (p<0.05): {sig_features}")

# Refit final model with significant features only
if sig_features:
    final_covariates = df_enc[sig_features + ["duration", "event"]].copy()
else:
    final_covariates = covariates  # fallback

final_model_cls = {
    "Weibull": WeibullAFTFitter,
    "LogNormal": LogNormalAFTFitter,
    "LogLogistic": LogLogisticAFTFitter,
}[best_name]

final_model = final_model_cls()
final_model.fit(final_covariates, duration_col="duration", event_col="event")

print(f"\n── Final {best_name} AFT Model (significant features only) ──────────")
print(f"  AIC: {final_model.AIC_:.2f}   BIC: {final_model.BIC_:.2f}")
final_model.print_summary()

# Coefficient plot
fig, ax = plt.subplots(figsize=(9, 6))
sm = final_model.summary.reset_index()
# Keep only the location parameter rows (mu_ for LogNormal, lambda_ for Weibull/LogLogistic)
loc_param = sm["param"].iloc[0]  # first param name is the location param
coef_df = sm[sm["param"] == loc_param].copy()
coef_df = coef_df[coef_df["covariate"] != "Intercept"]
coef_df = coef_df.sort_values("coef")
coef_df = coef_df.set_index("covariate")
colors_coef = ["#d62728" if c < 0 else "#2ca02c" for c in coef_df["coef"]]
ax.barh(list(coef_df.index), coef_df["coef"].tolist(), color=colors_coef, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_title(f"Final {best_name} AFT – Coefficients (log acceleration)")
ax.set_xlabel("Coefficient")
plt.tight_layout()
plt.savefig("plots/04_final_model_coefs.png")
plt.close()
print("Saved plots/04_final_model_coefs.png")

# ── 6. Customer Lifetime Value (CLV) ─────────────────────────────────────────
# CLV = ARPU × E[T]  where E[T] = predicted median survival time
# We use a rough ARPU based on custcat as proxy for monthly revenue

ARPU = {"Basic service": 25, "Plus service": 45,
        "E-service": 35, "Total service": 65}  # USD/month

df2 = df.copy()
df2["monthly_revenue"] = df2["custcat"].map(ARPU)

# Predicted median survival
pred_cols = sig_features if sig_features else FEATURES
pred_input = df_enc[sig_features].copy() if sig_features else df_enc[FEATURES].copy()
df2["expected_tenure"] = np.clip(final_model.predict_median(pred_input).values, 1, None)
df2["CLV"] = df2["monthly_revenue"] * df2["expected_tenure"]

print("\n── CLV Summary ────────────────────────────────────────────")
print(df2["CLV"].describe())

# CLV by segment
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# by custcat
order = df2.groupby("custcat")["CLV"].median().sort_values(ascending=False).index
sns.boxplot(data=df2, x="custcat", y="CLV", order=order,
            palette="tab10", ax=axes[0])
axes[0].set_title("CLV by Customer Category")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=15)

# by churn
sns.boxplot(data=df2, x="churn", y="CLV", palette=["#2ca02c", "#d62728"], ax=axes[1])
axes[1].set_title("CLV by Churn Status")
axes[1].set_xlabel("")
plt.tight_layout()
plt.savefig("plots/05_clv_segments.png")
plt.close()
print("Saved plots/05_clv_segments.png")

# CLV by region & marital
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df2, x="region", y="CLV", palette="Set2", ax=axes[0])
axes[0].set_title("CLV by Region")
sns.boxplot(data=df2, x="marital", y="CLV", palette="Set3", ax=axes[1])
axes[1].set_title("CLV by Marital Status")
plt.tight_layout()
plt.savefig("plots/06_clv_segments2.png")
plt.close()
print("Saved plots/06_clv_segments2.png")

# ── 7. Retention Budget Estimate ─────────────────────────────────────────────
sf_12 = final_model.predict_survival_function(pred_input, times=[12]).T
sf_12.columns = ["sf_12m"]
df2["sf_12m"] = sf_12["sf_12m"].values
# Use bottom-quartile survival probability as "at-risk" threshold
sf_threshold = df2["sf_12m"].quantile(0.25)
at_risk = df2[(df2["sf_12m"] <= sf_threshold) & (df2["churn"] == "No")]
print(f"\nAt-risk subscribers (sf@12m ≤ {sf_threshold:.2f}, bottom 25%, not yet churned): {len(at_risk)}")
print(f"Their median CLV: ${at_risk['CLV'].median():.0f}")
annual_retention_budget = at_risk["CLV"].sum() * 0.10   # 10% of at-risk CLV
print(f"Suggested annual retention budget (10% of at-risk CLV): ${annual_retention_budget:,.0f}")

# Save CLV table
df2[["ID", "custcat", "region", "marital", "churn",
     "monthly_revenue", "expected_tenure", "CLV", "sf_12m"]].to_csv(
    "data/clv_output.csv", index=False)
print("\nSaved data/clv_output.csv")

# ── 8. Final Summary Stats ────────────────────────────────────────────────────
print("\n── CLV by Customer Category ─────────────────────────────")
print(df2.groupby("custcat")["CLV"].agg(["mean", "median", "count"]).round(1))

print("\n── High-Value Segment Definition ────────────────────────")
top_clv = df2[df2["CLV"] > df2["CLV"].quantile(0.75)]
print(f"Top 25% by CLV (n={len(top_clv)})")
print(top_clv["custcat"].value_counts())
print(top_clv["marital"].value_counts())

print("\n✅  analysis.py complete")
