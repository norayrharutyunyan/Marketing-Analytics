

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from helper_functions import (
    bass_cumulative,
    bass_incremental,
    peak_adoption_year,
    r_squared,
    build_forecast_table,
    check_curve_shape,
    plot_cumulative_fit,
    plot_incremental_fit,
    plot_forecast,
)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


mifi_data = pd.read_csv("data/mifi_raw.csv")

YEARS      = mifi_data["Year"].values.astype(int)
n_obs      = mifi_data["Est_Annual_Adopters_M"].values.astype(float)
N_obs      = np.cumsum(n_obs)
t_obs      = np.arange(len(n_obs))
YEAR_START = int(YEARS[0])

print("MiFi Look-alike Data Loaded:")
print(mifi_data.to_string(index=False))

shape = check_curve_shape(YEARS, n_obs)
print(f"\nCurve shape validation:")
print(f"  Peak year    : {shape['peak_year']}")
print(f"  Rising arm   : {shape['points_before']} points")
print(f"  Declining arm: {shape['points_after']} points")
print(f"  Internal peak: {shape['is_internal']}")


p0 = [0.05, 0.40, N_obs.max() * 1.5]
lb = [1e-5, 1e-5, N_obs.max()      ]
ub = [0.99, 0.99, N_obs.max() * 20 ]

popt, pcov = curve_fit(
    bass_cumulative, t_obs, N_obs,
    p0=p0, bounds=(lb, ub), maxfev=100_000
)
p_fit, q_fit, M_fit = popt
perr  = np.sqrt(np.diag(pcov))

N_pred    = bass_cumulative(t_obs, *popt)
R2        = r_squared(N_obs, N_pred)
year_peak = peak_adoption_year(p_fit, q_fit, YEAR_START)

print("\n── Bass Parameters (MiFi Look-alike) ───────────────────")
print(f"  p  = {p_fit:.4f}  +/- {perr[0]:.4f}")
print(f"  q  = {q_fit:.4f}  +/- {perr[1]:.4f}")
print(f"  M  = {M_fit:.4f}M  +/- {perr[2]:.4f}M")
print(f"  R² = {R2:.4f}")
print(f"  Peak year = {year_peak:.1f}")

fitted_df = pd.DataFrame({
    "Year"                       : YEARS,
    "Observed_New_Adopters_M"    : n_obs,
    "Observed_Cumulative_M"      : N_obs,
    "Fitted_Cumulative_M"        : N_pred.round(4),
    "Residual"                   : (N_obs - N_pred).round(4),
})
fitted_df.to_csv("data/mifi_bass_fitted.csv", index=False)
print("\nSaved -> data/mifi_bass_fitted.csv")


p_m3 = p_fit
q_m3 = q_fit
M_m3 = 50.0

T_M3_START = 2024
T_M3_END   = 2035

year_star_m3 = peak_adoption_year(p_m3, q_m3, T_M3_START)

m3_table = build_forecast_table(p_m3, q_m3, M_m3, T_M3_START, T_M3_END)
m3_table.to_csv("data/nighthawk_m3_forecast.csv", index=False)
print("Saved -> data/nighthawk_m3_forecast.csv")

print(f"\n── Nighthawk M3 Forecast Summary ───────────────────────")
print(f"  p  = {p_m3:.4f}  (transferred)")
print(f"  q  = {q_m3:.4f}  (transferred)")
print(f"  M  = {M_m3:.0f}M devices")
print(f"  Peak year = {year_star_m3:.1f}")
print()
display_years = [2024, 2026, 2028, int(round(year_star_m3)), 2032, 2035]
print(m3_table[m3_table["Year"].isin(display_years)].to_string(index=False))


fig1, ax1 = plt.subplots(figsize=(8, 5))
plot_cumulative_fit(
    YEARS, N_obs, p_fit, q_fit, M_fit, R2,
    title="MiFi — Cumulative Adopters: Observed vs Bass Fit",
    ylabel="Cumulative devices (millions)",
    ax=ax1
)
fig1.tight_layout()
fig1.savefig("img/mifi_cumulative_fit.png", bbox_inches="tight")
plt.close(fig1)
print("\nSaved -> img/mifi_cumulative_fit.png")

fig2, ax2 = plt.subplots(figsize=(8, 5))
plot_incremental_fit(
    YEARS, n_obs, p_fit, q_fit, M_fit, year_peak,
    title="MiFi — New Adopters per Year: Observed vs Bass Fit",
    ylabel="New devices per year (millions)",
    ax=ax2
)
fig2.tight_layout()
fig2.savefig("img/mifi_incremental_fit.png", bbox_inches="tight")
plt.close(fig2)
print("Saved -> img/mifi_incremental_fit.png")

t_m3   = np.arange(0, T_M3_END - T_M3_START + 1)
N_m3   = bass_cumulative(t_m3, p_m3, q_m3, M_m3)
n_m3   = np.diff(N_m3, prepend=0)
n_m3[0] = N_m3[0]

fig3, _ = plot_forecast(
    years_fore = T_M3_START + t_m3,
    N_fore     = N_m3,
    n_fore     = n_m3,
    year_peak  = year_star_m3,
    M          = M_m3,
    title_cum  = "Nighthawk M3 — Cumulative Adoption Forecast (Global)",
    title_inc  = "Nighthawk M3 — New Adopters per Year (Global)",
)
fig3.suptitle(
    f"Netgear Nighthawk M3 — Bass Diffusion Forecast\n"
    f"p={p_m3:.4f}, q={q_m3:.4f}, M={M_m3:.0f}M devices",
    fontsize=11, fontweight="bold", y=1.03
)
fig3.savefig("img/nighthawk_m3_forecast.png", bbox_inches="tight")
plt.close(fig3)
print("Saved -> img/nighthawk_m3_forecast.png")

print("\nAll outputs generated successfully.")
