

def bass_cumulative(t, p, q, M):
    t = np.asarray(t, dtype=float)
    e = np.exp(-(p + q) * t)
    return M * (1.0 - e) / (1.0 + (q / p) * e)


def bass_incremental(t, p, q, M):
    t     = np.asarray(t, dtype=float)
    N_t   = bass_cumulative(t,     p, q, M)
    N_tm1 = bass_cumulative(t - 1, p, q, M)
    N_tm1[t == 0] = 0.0
    return N_t - N_tm1


def peak_adoption_year(p, q, year_start):
    t_star = np.log(q / p) / (p + q)
    return year_start + t_star



def r_squared(y_obs, y_pred):
    y_obs  = np.asarray(y_obs,  dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    return 1.0 - ss_res / ss_tot



def build_forecast_table(p, q, M, year_start, year_end):
    t      = np.arange(0, year_end - year_start + 1)
    years  = year_start + t
    N      = bass_cumulative(t, p, q, M)
    n      = np.diff(N, prepend=0)
    n[0]   = N[0]

    return pd.DataFrame({
        "Year"                : years,
        "Cumulative_Adopters" : np.round(N, 4),
        "New_Adopters"        : np.round(n, 4),
        "Penetration_%"       : np.round(N / M * 100, 2),
    })



def check_curve_shape(years, n_obs):
    years = np.asarray(years)
    n_obs = np.asarray(n_obs, dtype=float)
    peak_idx = int(np.argmax(n_obs))

    return {
        "peak_year"     : int(years[peak_idx]),
        "peak_idx"      : peak_idx,
        "points_before" : peak_idx,
        "points_after"  : len(n_obs) - 1 - peak_idx,
        "is_internal"   : (years[0] < years[peak_idx] < years[-1]),
    }



def plot_cumulative_fit(years, N_obs, p, q, M, R2,
                        title="Cumulative Adopters",
                        ylabel="Cumulative adopters",
                        ax=None, color_obs="#2166ac", color_fit="#e31a1c"):
    if ax is None:
        _, ax = plt.subplots()
    t_smooth = np.linspace(0, len(years) - 1, 300)
    year_start = years[0]
    ax.scatter(years, N_obs, color=color_obs, s=70, zorder=5,
               label="Observed")
    ax.plot(year_start + t_smooth,
            bass_cumulative(t_smooth, p, q, M),
            color=color_fit, linewidth=2,
            label=f"Bass fit (R\u00b2={R2:.4f})")
    ax.axhline(M, color="#33a02c", linestyle=":", linewidth=1.2,
               label=f"M = {M:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_incremental_fit(years, n_obs, p, q, M, year_peak,
                         title="New Adopters per Year",
                         ylabel="New adopters",
                         ax=None):
    if ax is None:
        _, ax = plt.subplots()
    t_smooth   = np.linspace(0, len(years) - 1, 300)
    year_start = years[0]
    ax.bar(years, n_obs, color="#a6cee3", alpha=0.85, label="Observed")
    ax.plot(year_start + t_smooth,
            bass_incremental(t_smooth, p, q, M),
            color="#e31a1c", linewidth=2, label="Bass fit")
    ax.axvline(year_peak, color="gray", linestyle="--", linewidth=1.2,
               label=f"Peak: {year_peak:.1f}")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_forecast(years_fore, N_fore, n_fore, year_peak, M,
                  title_cum="Cumulative Forecast",
                  title_inc="New Adopters Forecast",
                  color="#33a02c", color_inc="#ff7f00"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(years_fore, N_fore, color=color, linewidth=2.5)
    ax1.fill_between(years_fore, N_fore, alpha=0.12, color=color)
    ax1.axvline(year_peak, color="gray", linestyle="--", linewidth=1.2,
                label=f"Peak: {year_peak:.1f}")
    ax1.axhline(M, color="#e31a1c", linestyle=":", linewidth=1.2,
                label=f"M = {M:.0f}M")
    ax1.set_title(title_cum)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative devices (millions)")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(years_fore, n_fore, color=color_inc, linewidth=2.5)
    ax2.fill_between(years_fore, n_fore, alpha=0.12, color=color_inc)
    ax2.axvline(year_peak, color="gray", linestyle="--", linewidth=1.2,
                label=f"Peak: {year_peak:.1f}")
    ax2.set_title(title_inc)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("New devices per year (millions)")
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, (ax1, ax2)
