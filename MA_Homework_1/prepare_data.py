

import pandas as pd
import numpy as np

RAW_FILE  = "data/MiFi Dataset for Bass Model Analysis.csv"
OUT_FILE  = "data/mifi_raw.csv"

df_raw = pd.read_csv(RAW_FILE)
print("Raw file loaded:")
print(df_raw.to_string(index=False))

df_raw["Year"] = df_raw["Year"].astype(str).str.extract(r"(\d{4})").astype(int)

df_raw["MiFi_Revenue_USD_M"] = (
    df_raw["MiFi Revenue (Millions USD)"]
    .astype(str)
    .str.replace(r"[~\$,]", "", regex=True)
    .astype(float)
)

df_raw["Pct_Total_Revenue"] = (
    df_raw["% of Total Revenue"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .astype(float) / 100
)

df_raw.rename(
    columns={"Est. Annual Adopters (Proxy)": "Est_Annual_Adopters_M"},
    inplace=True
)

df_clean = df_raw[[
    "Year",
    "MiFi_Revenue_USD_M",
    "Pct_Total_Revenue",
    "Est_Annual_Adopters_M",
]].copy()

df_clean.to_csv(OUT_FILE, index=False)
print(f"\nClean file saved -> {OUT_FILE}")
print(df_clean.to_string(index=False))

n_obs = df_clean["Est_Annual_Adopters_M"].values
N_obs = n_obs.cumsum()

print(f"\nDescriptive statistics (incremental adopters, millions):")
print(f"  Mean   : {n_obs.mean():.3f}M")
print(f"  Std    : {n_obs.std():.3f}M")
print(f"  Min    : {n_obs.min():.3f}M  ({df_clean['Year'].values[n_obs.argmin()]})")
print(f"  Max    : {n_obs.max():.3f}M  ({df_clean['Year'].values[n_obs.argmax()]})")
print(f"  Total cumulative by 2014: {N_obs[-1]:.3f}M devices")
