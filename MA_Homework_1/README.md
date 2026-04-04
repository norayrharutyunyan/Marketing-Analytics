# Bass Diffusion Model — Netgear Nighthawk M3
**Author:** Norayr Harutyunyan | **Date:** March 6, 2026 | **Course:** Marketing Analytics

## Overview

This project applies the **Bass Diffusion Model** to forecast the global adoption of the **Netgear Nighthawk M3** (5G mobile hotspot, TIME Best Inventions 2025), using the original **MiFi portable hotspot device (2009–2014)** as a look-alike innovation to estimate diffusion parameters.

**Key results:**
- p = 0.1088, q = 0.4965, R² = 0.9976 (from MiFi look-alike fit)
- Nighthawk M3 peak adoption forecast: **~2027**
- Projected market penetration by 2032: **~96%** of 50M addressable devices

---

## Directory Structure

```
/
├── README.md                              ← This file
├── bass_model.py                          ← Main analysis: fits model, generates all outputs
├── prepare_data.py                        ← Cleans raw MiFi CSV, saves mifi_raw.csv
├── helper_functions.py                    ← Reusable Bass model functions and plot helpers
├── bass_model_nighthawk.ipynb             ← Jupyter notebook (full analysis with text)
│
├── data/
│   ├── MiFi_Dataset_for_Bass_Model_Analysis.csv  ← Raw MiFi revenue data (Novatel Wireless)
│   ├── mifi_raw.csv                              ← Cleaned MiFi dataset (output of prepare_data.py)
│   ├── mifi_bass_fitted.csv                      ← Observed vs Bass-fitted values (MiFi)
│   └── nighthawk_m3_forecast.csv                 ← Full M3 adoption forecast 2024–2035
│
├── img/
│   ├── mifi_cumulative_fit.png            ← Figure 1: MiFi cumulative adopters vs Bass fit
│   ├── mifi_incremental_fit.png           ← Figure 2: MiFi new adopters per year vs Bass fit
│   └── nighthawk_m3_forecast.png          ← Figure 3: Nighthawk M3 forecast (cumulative + incremental)
│
└── report/
    └── report.pdf                         ← Final rendered PDF report
```

---

## Setup and Usage

### Requirements

```bash
pip install numpy pandas matplotlib scipy reportlab
```

### Run the analysis

**Step 1 — Prepare data** (cleans the raw CSV):
```bash
python prepare_data.py
```

**Step 2 — Run the Bass model** (fits model, generates plots and forecast tables):
```bash
python bass_model.py
```

**Step 3 — Open the Jupyter notebook for the full interactive analysis with embedded text:
```bash
jupyter notebook bass_model_nighthawk.ipynb
```

---

## Data Sources

| File | Description | Source |
|------|-------------|--------|
| `MiFi_Dataset_for_Bass_Model_Analysis.csv` | MiFi annual revenue and estimated adopter proxy, 2009–2014 | Novatel Wireless Annual Reports (SEC EDGAR) |
| `mifi_raw.csv` | Cleaned version of the above | Derived |
| `mifi_bass_fitted.csv` | Observed cumulative adopters vs Bass model fit with residuals | Computed |
| `nighthawk_m3_forecast.csv` | Bass model adoption forecast for Nighthawk M3, 2024–2035 | Computed |

---

## Code Files

| File | Description |
|------|-------------|
| `helper_functions.py` | Core Bass model functions (`bass_cumulative`, `bass_incremental`, `peak_adoption_year`, `r_squared`, `build_forecast_table`, `check_curve_shape`) and plot helpers |
| `prepare_data.py` | Reads raw MiFi CSV, cleans year/revenue/percentage columns, saves `mifi_raw.csv` |
| `bass_model.py` | Main script: loads `mifi_raw.csv`, fits Bass model via NLS, saves fitted values and M3 forecast, generates all three plots |

---

## Model Summary

| Parameter | MiFi Look-alike | Nighthawk M3 Forecast |
|-----------|----------------|----------------------|
| p (innovation) | 0.1088 ± 0.0120 | 0.1088 (transferred) |
| q (imitation) | 0.4965 ± 0.1748 | 0.4965 (transferred) |
| M (market potential) | 14.31M devices | 50M devices (Fermi) |
| Peak adoption year | 2011.5 | 2026.5 |
| R² | 0.9976 | — |

