"""
Consolidated, final versions of all figures built during the analysis.
Each function is standalone - call only the ones you need.

Input files expected (adjust paths below):
    MASTER_XLSX     - output of 03_build_master_dataset.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from scipy import stats
import pymannkendall as mk

MASTER_XLSX = "./data/processed/ssim_q_weather_time_vs10.xlsx"  # <-- adjust
MASTER_SHEET = "clean_complete_cases"
OUT_DIR = "./figures"

C_NIEDER = "#1b4f72"
C_VERNAGT = "#922b21"

# Peak-day-timing figure specifically uses the project's newer categorical
# duo (CVD-validated: worst adjacent normal-vision floor 34.1, deutan 27.9).
C_VERNAGT_PEAK = "#4a3aa7"
C_NIEDER_PEAK = "#008300"

PERIOD_BINS = [(1974, 1979), (1980, 1984), (1985, 1989), (1990, 1994), (1995, 1999),
               (2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2023)]


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    })


MONTH_ABBR_DE = {1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
                  7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez"}


def doy_anomaly(dd, col, window=9):
    """Daily anomaly relative to the (9d-smoothed) day-of-year climatology -
    isolates day-to-day weather-driven fluctuations from the seasonal cycle
    that Q and T/RR all naturally share."""
    clim = dd.groupby("doy")[col].mean()
    clim_smooth = clim.rolling(window, center=True, min_periods=3).mean()
    return dd[col] - dd["doy"].map(clim_smooth)


def _month_ticks(vmin, vmax):
    """Day-of-year (non-leap reference year) of the 1st of each month that
    falls within [vmin, vmax], plus its German abbreviation - used to label
    day-of-year axes with month names instead of raw numbers."""
    ticks, labels = [], []
    for m in range(1, 13):
        doy = pd.Timestamp(2001, m, 1).dayofyear
        if vmin - 3 <= doy <= vmax + 3:
            ticks.append(doy)
            labels.append(MONTH_ABBR_DE[m])
    return ticks, labels


def set_month_xaxis(ax):
    vmin, vmax = ax.get_xlim()
    ticks, labels = _month_ticks(vmin, vmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def set_month_yaxis(ax):
    vmin, vmax = ax.get_ylim()
    ticks, labels = _month_ticks(vmin, vmax)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def load_master():
    df = pd.read_excel(MASTER_XLSX, sheet_name=MASTER_SHEET)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


# --------------------------------------------------------------- Figure 10 --
def fig_peak_day_running_window(df, smoothing_window=9):
    """For each year, find the (N-day-smoothed) day-of-year of maximum
    discharge, then apply 5/10/15-year rolling means to that peak-day series
    to check whether an apparent seasonal-timing shift is robust to the
    smoothing-window choice."""
    d = df.copy()
    d["doy"] = d["time"].dt.dayofyear
    d["year"] = d["time"].dt.year

    def peak_day_per_year(col):
        rows = []
        for y, sub in d.groupby("year"):
            s = sub.dropna(subset=[col]).sort_values("doy")
            if len(s) < 100:
                continue
            med = s.set_index("doy")[col]
            smooth = med.rolling(smoothing_window, center=True, min_periods=max(3, smoothing_window // 2)).mean()
            rows.append((y, smooth.idxmax()))
        return pd.DataFrame(rows, columns=["year", "peak_day"]).set_index("year")

    def shades(base_hex, n):
        base = np.array(mcolors.to_rgb(base_hex))
        white = np.array([1, 1, 1])
        fracs = np.linspace(0.62, 0.0, n)
        return [tuple(base * (1 - f) + white * f) for f in fracs]

    panels = [("Q_Vernagt", "Vernagtbach", C_VERNAGT_PEAK, "a"), ("Q_Niedertalbach", "Rofenache – Pegel Vent", C_NIEDER_PEAK, "b")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    for ax, (col, title, base_color, lab) in zip(axes, panels):
        pk = peak_day_per_year(col)
        ax.plot(pk.index, pk["peak_day"], color="0.8", marker="o", markersize=3, linewidth=0.7,
                label="Jahreswert", zorder=1)
        for w, c in zip([5, 10, 15], shades(base_color, 3)):
            rm = pk["peak_day"].rolling(w, center=True, min_periods=max(3, w // 2)).mean()
            ax.plot(rm.index, rm.values, color=c, linewidth=1.6, label=f"{w}-Jahres-Mittel")
            valid = rm.dropna()
            slope, _, _, p, _ = stats.linregress(valid.index, valid.values)
            print(f"  {title} ({smoothing_window}d smoothing, {w}y rolling mean): "
                  f"{slope * 10:+.2f} d/Dek (p={p:.4f}), n={len(pk)} Jahre")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Jahr")
        ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="upper left", frameon=False)
        set_month_yaxis(ax)
    axes[0].set_ylabel("Monat mit\nmaximalem Abfluss (9d-geglättet)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig10_peak_day_running_window.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 9 --
def fig_small_multiples(df, col, color, title, outfile):
    """One panel per 5-year period, median + IQR, shared y-axis for fair
    comparison. This shows the actual shape of the seasonal hydrograph
    instead of a single derived peak-day number - if there are two competing
    humps, they are directly visible here."""
    d = df.copy()
    d["doy"] = d["time"].dt.dayofyear
    d["year"] = d["time"].dt.year

    def period_label(y):
        for a, b in PERIOD_BINS:
            if a <= y <= b:
                return f"{a}-{b}"
        return None

    d["periode"] = d["year"].apply(period_label)
    ymax = d[col].quantile(0.998)

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.3), dpi=300, sharex=True, sharey=True)
    for ax, (a, b) in zip(axes.flat, PERIOD_BINS):
        lab = f"{a}-{b}"
        sub = d[d["periode"] == lab]
        if sub[col].notna().sum() < 20:
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, color="0.6")
            ax.set_title(lab, fontsize=8)
            continue
        g = sub.groupby("doy")[col]
        med, q25, q75 = g.median(), g.quantile(0.25), g.quantile(0.75)
        med = med.rolling(5, center=True, min_periods=3).mean()
        q25 = q25.rolling(5, center=True, min_periods=3).mean()
        q75 = q75.rolling(5, center=True, min_periods=3).mean()
        ax.fill_between(med.index, q25, q75, color=color, alpha=0.25, linewidth=0)
        ax.plot(med.index, med.values, color=color, linewidth=1)
        ax.set_title(lab, fontsize=8)
        ax.set_ylim(0, ymax)
    for ax in axes[-1, :]:
        ax.set_xlabel("")
        set_month_xaxis(ax)
        ax.tick_params(axis="x", labelrotation=45)
    for ax in axes[:, 0]:
        ax.set_ylabel("Q [m3/s]", fontsize=7)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{outfile}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 7 --
def fig_seasonal_by_period_gradient(df):
    """All 10 five-year-period median curves (lightly smoothed) overlaid in
    one panel per catchment, colour-graded early (dark) to late (light).
    Lets you see directly whether the shape/timing drifts gradually across
    periods or jumps around - a single "average of the smoothed period
    curves" plot would hide exactly that."""
    d = df.copy()
    d["doy"] = d["time"].dt.dayofyear
    d["year"] = d["time"].dt.year

    def period_label(y):
        for a, b in PERIOD_BINS:
            if a <= y <= b:
                return f"{a}-{b}"
        return None

    d["periode"] = d["year"].apply(period_label)
    period_labels = [f"{a}-{b}" for a, b in PERIOD_BINS]

    # climate-stripes convention: early period = blue (cool/past), late = red (warm/present)
    cmap = plt.get_cmap("coolwarm", len(period_labels))
    grad_colors = [cmap(i) for i in range(len(period_labels))]

    panels = [("Q_Vernagt", "Vernagtbach", "a"),
              ("Q_Niedertalbach", "Rofenache – Pegel Vent", "b")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300, sharex=True)
    for ax, (col, title, lab) in zip(axes, panels):
        for pl, c in zip(period_labels, grad_colors):
            sub = d[d["periode"] == pl].dropna(subset=[col])
            if len(sub) == 0:
                continue
            med = sub.groupby("doy")[col].median()
            med = med.reindex(range(med.index.min(), med.index.max() + 1))
            med = med.rolling(5, center=True, min_periods=3).mean()
            ax.plot(med.index, med.values, color=c, linewidth=1.2, label=pl)
        ax.set_title(title, fontsize=10)
        ax.text(0.0, 1.1, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
        ax.set_xlabel("")
        set_month_xaxis(ax)
    axes[0].set_ylabel(r"Median $Q$ [m$^3$ s$^{-1}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(f"{OUT_DIR}/fig7_seasonal_by_period_gradient.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- Figure 9b (new) --
def fig_hump_overlay(df):
    """Direct overlay of the early-period vs. late-period median seasonal
    curve for both catchments, normalized to each period's own peak so the
    SHAPE (single peak vs. two humps, and which hump dominates) is
    comparable across periods regardless of absolute discharge magnitude."""
    d = df.copy()
    d["doy"] = d["time"].dt.dayofyear
    d["year"] = d["time"].dt.year

    early, late = (1974, 1988), (2009, 2023)
    panels = [("Q_Vernagt", "Vernagtbach", C_VERNAGT_PEAK, "a"),
              ("Q_Niedertalbach", "Rofenache – Pegel Vent", C_NIEDER_PEAK, "b")]

    def shades(base_hex, n):
        base = np.array(mcolors.to_rgb(base_hex))
        white = np.array([1, 1, 1])
        fracs = np.linspace(0.55, 0.0, n)
        return [tuple(base * (1 - f) + white * f) for f in fracs]

    smooth_win = 5  # light smoothing only, for display - not used in any of the trend stats above

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=300, sharey=True)
    for ax, (col, title, base_color, lab) in zip(axes, panels):
        light_c, dark_c = shades(base_color, 2)
        for (y0, y1), c, alpha, label in [(early, light_c, 0.9, f"{early[0]}-{early[1]}"),
                                           (late, dark_c, 1.0, f"{late[0]}-{late[1]}")]:
            sub = d[(d["year"] >= y0) & (d["year"] <= y1)].dropna(subset=[col])
            g = sub.groupby("doy")[col]
            med, q25, q75 = g.median(), g.quantile(0.25), g.quantile(0.75)
            norm = med.max()
            med, q25, q75 = med / norm, q25 / norm, q75 / norm
            med_s = med.rolling(smooth_win, center=True, min_periods=3).mean()
            q25_s = q25.rolling(smooth_win, center=True, min_periods=3).mean()
            q75_s = q75.rolling(smooth_win, center=True, min_periods=3).mean()
            ax.fill_between(med_s.index, q25_s, q75_s, color=c, alpha=0.25, linewidth=0)
            ax.plot(med_s.index, med_s.values, color=c, linewidth=1.8, alpha=alpha, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        set_month_xaxis(ax)
        ax.text(0.0, 1.1, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("Median Q, normiert\nauf Perioden-Maximum")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig9b_hump_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 11 --
def fig_flow_timing_cumulative(df):
    """Robust alternative to the discrete peak-day: for each year, the day by
    which 25%/50%/75% of that season's cumulative discharge has passed. Needs
    no smoothing window and never has to pick a winner between near-tied
    local maxima, unlike the discrete peak-day approach (fig10) - which we
    found flips sign for Vernagtbach depending on smoothing-window choice
    because its melt season often has two similarly-high humps competing for
    "the" peak day."""
    d = df.copy()
    d["doy"] = d["time"].dt.dayofyear
    d["year"] = d["time"].dt.year

    def flow_dates_per_year(col):
        rows = []
        for y, sub in d.groupby("year"):
            s = sub.dropna(subset=[col]).sort_values("doy")
            if len(s) < 100:
                continue
            cum = s[col].cumsum() / s[col].sum()
            d25 = s["doy"].values[np.searchsorted(cum.values, 0.25)]
            d50 = s["doy"].values[np.searchsorted(cum.values, 0.50)]
            d75 = s["doy"].values[np.searchsorted(cum.values, 0.75)]
            rows.append((y, d25, d50, d75))
        return pd.DataFrame(rows, columns=["year", "d25", "d50", "d75"]).set_index("year")

    def shades(base_hex, n):
        base = np.array(mcolors.to_rgb(base_hex))
        white = np.array([1, 1, 1])
        fracs = np.linspace(0.55, 0.0, n)
        return [tuple(base * (1 - f) + white * f) for f in fracs]

    panels = [("Q_Vernagt", "Vernagtbach", C_VERNAGT_PEAK, "a"), ("Q_Niedertalbach", "Rofenache – Pegel Vent", C_NIEDER_PEAK, "b")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300, sharey=True)
    for ax, (col, title, base_color, lab) in zip(axes, panels):
        fd = flow_dates_per_year(col)
        cols3 = shades(base_color, 3)
        for m, label, c in [("d25", "D25 (25% des Saisonabflusses)", cols3[0]),
                             ("d50", "D50 (50% - Median-Datum)", cols3[1]),
                             ("d75", "D75 (75% des Saisonabflusses)", cols3[2])]:
            ax.scatter(fd.index, fd[m], s=10, alpha=0.5, color=c, linewidths=0)
            slope, intercept, r, p, se = stats.linregress(fd.index, fd[m])
            xs = np.array([fd.index.min(), fd.index.max()])
            ax.plot(xs, slope * xs + intercept, color=c, linewidth=1.8,
                    label=f"{label}: {slope*10:+.1f} d/Dek, p={p:.4f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Jahr")
        ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="upper left", frameon=False, fontsize=6.5)
        set_month_yaxis(ax)
    axes[0].set_ylabel("Monat (kumulatives\nAbfluss-Datum)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig11_flow_timing_cumulative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 12 --
def fig_cumulative_fraction_curves(df):
    """Generalizes fig11's D25/D50/D75 to the full quantile curve: for each
    year, the day-of-year at which each 5%-step of that season's cumulative
    discharge was reached (x-axis = fraction, y-axis = day). Averaged per
    5-year period (raw per-year curves are too crowded at 50 lines) and
    colour-graded blue (early) to red (late), same convention as fig7."""
    d = df.copy()
    d["doy"] = d["time"].dt.dayofyear
    d["year"] = d["time"].dt.year

    def period_label(y):
        for a, b in PERIOD_BINS:
            if a <= y <= b:
                return f"{a}-{b}"
        return None

    d["periode"] = d["year"].apply(period_label)
    fractions = np.arange(0.05, 1.0001, 0.05)

    def fraction_days_per_year(col):
        rows = []
        for y, sub in d.groupby("year"):
            s = sub.dropna(subset=[col]).sort_values("doy")
            if len(s) < 100:
                continue
            cum = s[col].cumsum().values / s[col].sum()
            days = [s["doy"].values[min(np.searchsorted(cum, f), len(cum) - 1)] for f in fractions]
            rows.append([y] + days)
        return pd.DataFrame(rows, columns=["year"] + [f"{int(round(f*100))}" for f in fractions]).set_index("year")

    period_labels = [f"{a}-{b}" for a, b in PERIOD_BINS]
    cmap = plt.get_cmap("coolwarm", len(period_labels))
    grad_colors = [cmap(i) for i in range(len(period_labels))]

    panels = [("Q_Vernagt", "Vernagtbach", "a"),
              ("Q_Niedertalbach", "Rofenache – Pegel Vent", "b")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300, sharex=True, sharey=True)
    for ax, (col, title, lab) in zip(axes, panels):
        fd = fraction_days_per_year(col)
        fd["periode"] = [period_label(y) for y in fd.index]
        for pl, c in zip(period_labels, grad_colors):
            sub = fd[fd["periode"] == pl]
            if len(sub) == 0:
                continue
            mean_days = sub[[f"{int(round(f*100))}" for f in fractions]].mean()
            ax.plot(mean_days.values, fractions * 100, color=c, linewidth=1.6, marker="o", markersize=2.5, label=pl)
        ax.set_title(title, fontsize=10)
        ax.text(0.0, 1.1, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
        ax.set_xlabel("")
        set_month_xaxis(ax)
    axes[0].set_ylabel("Kumulativer Saisonabfluss [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(f"{OUT_DIR}/fig12_cumulative_fraction_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- Figure 12b --
def fig_cumulative_fraction_curves_by_year(df):
    """Same as fig12, but one thin line per individual year (n=50) instead of
    period averages - no averaging step at all, each year's 100% mark is
    that year's own seasonal discharge total. Colour = year on a continuous
    blue-to-red scale (colourbar instead of a 50-entry legend)."""
    d = df.copy()
    d["doy"] = d["time"].dt.dayofyear
    d["year"] = d["time"].dt.year
    fractions = np.arange(0.05, 1.0001, 0.05)

    def fraction_days_per_year(col):
        rows = []
        for y, sub in d.groupby("year"):
            s = sub.dropna(subset=[col]).sort_values("doy")
            if len(s) < 100:
                continue
            cum = s[col].cumsum().values / s[col].sum()
            days = [s["doy"].values[min(np.searchsorted(cum, f), len(cum) - 1)] for f in fractions]
            rows.append([y] + days)
        return pd.DataFrame(rows, columns=["year"] + [f"{int(round(f*100))}" for f in fractions]).set_index("year")

    panels = [("Q_Vernagt", "Vernagtbach", "a"), ("Q_Niedertalbach", "Rofenache – Pegel Vent", "b")]
    year_min, year_max = int(d["year"].min()), int(d["year"].max())
    norm = mcolors.Normalize(vmin=year_min, vmax=year_max)
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300, sharex=True, sharey=True)
    for ax, (col, title, lab) in zip(axes, panels):
        fd = fraction_days_per_year(col)
        cols = [f"{int(round(f*100))}" for f in fractions]
        for y, row in fd.iterrows():
            ax.plot(row[cols].values, fractions * 100, color=cmap(norm(y)), linewidth=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.text(0.0, 1.08, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
        ax.set_xlabel("")
        set_month_xaxis(ax)
    axes[0].set_ylabel("Kumulativer Saisonabfluss [%]")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.15, aspect=40)
    cbar.set_label("Jahr")
    fig.savefig(f"{OUT_DIR}/fig12b_cumulative_fraction_curves_by_year.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 13 --
def fig_parde_coefficients(df):
    """Pardé/regime coefficient k_i = mean(Q_month_i) / mean(Q_season), per
    year, averaged per 5-year period. Both gauges only have data May-Sep (no
    winter measurements - presumably frozen/ungauged), so this is a
    season-only adaptation of the classical 12-month Pardé coefficient: it
    shows the RELATIVE distribution of flow across the measured months, not
    an absolute-magnitude trend. k=1 would mean an average month."""
    d = df.copy()
    d["month"] = d["time"].dt.month
    d["year"] = d["time"].dt.year

    def period_label(y):
        for a, b in PERIOD_BINS:
            if a <= y <= b:
                return f"{a}-{b}"
        return None

    d["periode"] = d["year"].apply(period_label)
    months = [5, 6, 7, 8, 9]
    month_names = [MONTH_ABBR_DE[m] for m in months]

    def parde_per_year(col):
        rows = []
        for y, sub in d.groupby("year"):
            sub = sub.dropna(subset=[col])
            month_means = sub.groupby("month")[col].mean()
            if not all(m in month_means.index for m in months):
                continue
            season_mean = month_means.reindex(months).mean()
            k = month_means.reindex(months) / season_mean
            rows.append([y] + list(k.values))
        return pd.DataFrame(rows, columns=["year"] + [str(m) for m in months]).set_index("year")

    period_labels = [f"{a}-{b}" for a, b in PERIOD_BINS]
    cmap = plt.get_cmap("coolwarm", len(period_labels))
    grad_colors = [cmap(i) for i in range(len(period_labels))]

    panels = [("Q_Vernagt", "Vernagtbach", "a"), ("Q_Niedertalbach", "Rofenache – Pegel Vent", "b")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=300, sharey=True)
    for ax, (col, title, lab) in zip(axes, panels):
        pk = parde_per_year(col)
        pk["periode"] = [period_label(y) for y in pk.index]
        for pl, c in zip(period_labels, grad_colors):
            sub = pk[pk["periode"] == pl]
            if len(sub) == 0:
                continue
            mean_k = sub[[str(m) for m in months]].mean()
            ax.plot(month_names, mean_k.values, color=c, linewidth=1.6, marker="o", markersize=4, label=pl)
        ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(title, fontsize=10)
        ax.text(0.0, 1.1, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
    axes[0].set_ylabel(r"Pardé-Koeffizient $k_i$"+"\n(Monatsmittel / Saisonmittel)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(f"{OUT_DIR}/fig13_parde_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 14 --
def fig_seasonal_magnitude_trend(df):
    """Has summer discharge magnitude (not just timing) changed? Seasonal
    mean Q per year (May-Sep), scatter + Mann-Kendall/Sen's-slope trend per
    catchment. Mann-Kendall + Sen's slope (not OLS linear regression) is the
    standard trend test in hydrology/climatology (WMO-recommended): a
    rank-based, non-parametric test that doesn't assume normally distributed,
    linear residuals and is far less sensitive to single outlier years than
    OLS - the more defensible "is this monotonically increasing/decreasing"
    statistic, the direct magnitude-trend analogue of the Pardé coefficient
    used above for the seasonal-shape question."""
    d = df.copy()
    d["year"] = d["time"].dt.year

    panels = [("Q_Vernagt", "Vernagtbach", C_VERNAGT_PEAK, "a"), ("Q_Niedertalbach", "Rofenache – Pegel Vent", C_NIEDER_PEAK, "b")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    for ax, (col, title, c, lab) in zip(axes, panels):
        sub = d.dropna(subset=[col])
        counts = sub.groupby("year")[col].count()
        seas = sub.groupby("year")[col].mean()
        seas = seas[counts >= 100]
        ax.scatter(seas.index, seas.values, s=14, alpha=0.6, color=c, linewidths=0)
        result = mk.original_test(seas.values)
        xs = np.array([seas.index.min(), seas.index.max()])
        ys = result.intercept + result.slope * (xs - seas.index.min())
        ax.plot(xs, ys, color=c, linewidth=1.8,
                label=f"Sen's slope: {result.slope*10:+.2f} m³/s/Dek\nMann-Kendall p={result.p:.4f}, Tau={result.Tau:+.2f} ({result.trend})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Jahr")
        ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    axes[0].set_ylabel(r"Saisonmittel $Q$ [m$^3$ s$^{-1}$] (Mai–Sep)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig14_seasonal_magnitude_trend.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- Figure 14c --
def fig_seasonal_magnitude_bars(df):
    """Same seasonal-mean-Q-per-year data as fig14, but as bars with error
    bars showing the 25th-75th percentile spread of DAILY Q within that
    year's season - fig14's scatter+trend-line only shows the one mean point
    per year, not how much the daily values varied within a season. Bars
    coloured continuously blue (1974) to red (2023), same convention as the
    other per-year figures."""
    d = df.copy()
    d["year"] = d["time"].dt.year

    panels = [("Q_Vernagt", "Vernagtbach", "a"), ("Q_Niedertalbach", "Rofenache – Pegel Vent", "b")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), dpi=300, sharex=True)
    for ax, (col, title, lab) in zip(axes, panels):
        sub = d.dropna(subset=[col])
        counts = sub.groupby("year")[col].count()
        mean_q = sub.groupby("year")[col].mean()
        q25 = sub.groupby("year")[col].quantile(0.25)
        q75 = sub.groupby("year")[col].quantile(0.75)
        valid = counts[counts >= 100].index
        mean_q, q25, q75 = mean_q.loc[valid], q25.loc[valid], q75.loc[valid]

        norm = mcolors.Normalize(vmin=valid.min(), vmax=valid.max())
        cmap = plt.get_cmap("coolwarm")
        colors = [cmap(norm(y)) for y in valid]
        yerr = np.array([mean_q.values - q25.values, q75.values - mean_q.values])
        yerr = np.clip(yerr, 0, None)
        ax.bar(valid, mean_q.values, color=colors, width=0.8)
        ax.errorbar(valid, mean_q.values, yerr=yerr, fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2)
        ax.set_title(title, fontsize=10)
        ax.text(0.0, 1.08, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
        ax.set_ylabel(r"$Q$ [m$^3$ s$^{-1}$]" + "\n(Saisonmittel, Fehlerbalken = 25.-75. Perzentil Tageswerte)", fontsize=8)
    axes[-1].set_xlabel("Jahr")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig14c_seasonal_magnitude_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- Figure 14b --
def fig_monthly_magnitude_by_period(df):
    """Un-normalized companion to fig13: absolute monthly mean Q [m3/s] per
    5-year period, same blue-to-red gradient - shows the magnitude behind
    the Pardé-coefficient shape changes (is a month's share falling because
    IT dropped, or because OTHER months grew faster?)."""
    d = df.copy()
    d["month"] = d["time"].dt.month
    d["year"] = d["time"].dt.year

    def period_label(y):
        for a, b in PERIOD_BINS:
            if a <= y <= b:
                return f"{a}-{b}"
        return None

    d["periode"] = d["year"].apply(period_label)
    months = [5, 6, 7, 8, 9]
    month_names = [MONTH_ABBR_DE[m] for m in months]
    period_labels = [f"{a}-{b}" for a, b in PERIOD_BINS]
    cmap = plt.get_cmap("coolwarm", len(period_labels))
    grad_colors = [cmap(i) for i in range(len(period_labels))]

    panels = [("Q_Vernagt", "Vernagtbach", "a"), ("Q_Niedertalbach", "Rofenache – Pegel Vent", "b")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=300)
    for ax, (col, title, lab) in zip(axes, panels):
        sub = d.dropna(subset=[col])
        for pl, c in zip(period_labels, grad_colors):
            psub = sub[sub["periode"] == pl]
            if len(psub) == 0:
                continue
            month_means = psub.groupby("month")[col].mean().reindex(months)
            ax.plot(month_names, month_means.values, color=c, linewidth=1.6, marker="o", markersize=4, label=pl)
        ax.set_title(title, fontsize=10)
        ax.text(0.0, 1.1, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
    axes[0].set_ylabel(r"Monatsmittel $Q$ [m$^3$ s$^{-1}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(f"{OUT_DIR}/fig14b_monthly_magnitude_by_period.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 15 --
def fig_monthly_circular_scatter(df):
    """Polar scatter: angle = calendar month (Jan at top, clockwise), radius
    = that year's mean Q for the month, colour = year (blue 1974 to red
    2023). Each month fills the full sector between its own gridline and the
    next month's gridline (not a narrow fan centered on the line), and the
    view is cropped to the Apr-Oct half-circle since the other half is
    always empty (no data outside May-Sep)."""
    d = df.copy()
    d["month"] = d["time"].dt.month
    d["year"] = d["time"].dt.year
    months = [5, 6, 7, 8, 9]

    panels = [("Q_Vernagt", "Vernagtbach", "a"), ("Q_Niedertalbach", "Rofenache – Pegel Vent", "b")]
    year_min, year_max = int(d["year"].min()), int(d["year"].max())
    norm = mcolors.Normalize(vmin=year_min, vmax=year_max)
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=300, subplot_kw={"projection": "polar"})
    for ax, (col, title, lab) in zip(axes, panels):
        sub = d.dropna(subset=[col])
        month_means = sub.groupby(["year", "month"])[col].mean().reset_index()
        for m in months:
            msub = month_means[month_means["month"] == m].sort_values("year")
            sector_start = 2 * np.pi * (m - 1) / 12
            sector_width = 2 * np.pi / 12
            # spread across 5%-95% of the sector so points don't sit on the gridlines
            fracs = np.linspace(0.05, 0.95, len(msub))
            thetas = sector_start + fracs * sector_width
            colors = [cmap(norm(y)) for y in msub["year"]]
            ax.scatter(thetas, msub[col].values, c=colors, s=14, alpha=0.85, linewidths=0)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        # set_thetagrids resets thetamin/thetamax internally, so it must come
        # first - otherwise the Apr-Oct crop below silently gets undone.
        ax.set_thetagrids([2 * np.pi * (m - 1) / 12 * 180 / np.pi for m in range(1, 13)],
                           labels=[MONTH_ABBR_DE[m] for m in range(1, 13)], fontsize=8)
        ax.set_thetamin(90)   # Apr
        ax.set_thetamax(270)  # Oct
        ax.set_title(title, fontsize=10, pad=28)
        ax.text(-0.05, 1.15, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.set_rlabel_position(0)
        ax.tick_params(axis="y", labelsize=6.5)
        # polar axes always reserve a full-circle bounding box even when
        # cropped to a half-circle via thetamin/thetamax - shift each axes
        # up so the (invisible) top half moves off-canvas instead of leaving
        # a blank gap between the title and the visible plot.
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + pos.height * 0.40, pos.width, pos.height])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40, shrink=0.6)
    cbar.set_label("Jahr")
    fig.suptitle(r"Monatsmittel $Q$ [m$^3$ s$^{-1}$] nach Kalendermonat (Radius), je Jahr (Farbe)", fontsize=9, y=1.35)
    fig.savefig(f"{OUT_DIR}/fig15_monthly_circular_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 16 --
def fig_climate_trends(df):
    """Has it gotten warmer/wetter in summer? Seasonal (May-Sep) mean
    temperature and total precipitation per year at the two SPARTACUS grid
    points closest to each gauge (Vernagtkessel for Vernagtbach, Vent for
    Rofenache) - SPARTACUS chosen over the PANGAEA/ACINN station series
    because it's gap-free and methodologically consistent across the whole
    1974-2023 period (no sensor swaps/relocations to correct for). Same
    Mann-Kendall/Sen's-slope trend test as fig14."""
    d = df.copy()
    d["year"] = d["time"].dt.year
    d["month"] = d["time"].dt.month
    seas = d[d["month"].isin([5, 6, 7, 8, 9])]

    panels = [
        ("Tmean_SPARTACUS_Vernagtkessel", "mean", "Temperatur Vernagtkessel", C_VERNAGT_PEAK,
         r"Saisonmittel $T$ [°C]", "°C", "a"),
        ("Tmean_SPARTACUS_Vent", "mean", "Temperatur Vent", C_NIEDER_PEAK,
         r"Saisonmittel $T$ [°C]", "°C", "b"),
        ("RR_SPARTACUS_Vernagtkessel", "sum", "Niederschlag Vernagtkessel", C_VERNAGT_PEAK,
         r"Saisonsumme $RR$ [mm]", "mm", "c"),
        ("RR_SPARTACUS_Vent", "sum", "Niederschlag Vent", C_NIEDER_PEAK,
         r"Saisonsumme $RR$ [mm]", "mm", "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    for ax, (col, agg, title, c, ylabel, unit, lab) in zip(axes.flat, panels):
        yearly = seas.groupby("year")[col].agg(agg)
        ax.scatter(yearly.index, yearly.values, s=14, alpha=0.6, color=c, linewidths=0)
        result = mk.original_test(yearly.values)
        xs = np.array([yearly.index.min(), yearly.index.max()])
        ys = result.intercept + result.slope * (xs - yearly.index.min())
        ax.plot(xs, ys, color=c, linewidth=1.8,
                label=f"Sen's slope: {result.slope*10:+.2f} {unit}/Dek\nMann-Kendall p={result.p:.4f} ({result.trend})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Jahr")
        ax.set_ylabel(ylabel)
        ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig16_climate_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 17 --
def fig_discharge_temp_sensitivity(df):
    """Discharge sensitivity to temperature: day-of-year-deseasonalized
    anomalies of Q and T (9d-smoothed climatology subtracted, so this isolates
    day-to-day weather-driven fluctuations from the seasonal cycle both
    naturally share), regressed against each other, split into an early and
    late half to check whether the sensitivity itself has changed."""
    d = df.copy()
    d["year"] = d["time"].dt.year
    d["doy"] = d["time"].dt.dayofyear
    d["month"] = d["time"].dt.month
    seas = d[d["month"].isin([5, 6, 7, 8, 9])].copy()

    halves = [(1974, 1998, "1974-1998", "#4a3aa7"), (1999, 2023, "1999-2023", "#c0392b")]
    panels = [("Q_Vernagt", "Tmean_SPARTACUS_Vernagtkessel", "Vernagtbach", "a"),
              ("Q_Niedertalbach", "Tmean_SPARTACUS_Vent", "Rofenache – Pegel Vent", "b")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=300)
    for ax, (qcol, tcol, title, lab) in zip(axes, panels):
        sub = seas.dropna(subset=[qcol, tcol]).copy()
        sub["Q_a"] = doy_anomaly(sub, qcol)
        sub["T_a"] = doy_anomaly(sub, tcol)
        sub = sub.dropna(subset=["Q_a", "T_a"])
        for y0, y1, hlabel, c in halves:
            s2 = sub[(sub["year"] >= y0) & (sub["year"] <= y1)]
            ax.scatter(s2["T_a"], s2["Q_a"], s=5, alpha=0.15, color=c, linewidths=0)
            slope, intercept, r, p, se = stats.linregress(s2["T_a"], s2["Q_a"])
            xs = np.array([s2["T_a"].min(), s2["T_a"].max()])
            ax.plot(xs, slope * xs + intercept, color=c, linewidth=2,
                    label=f"{hlabel}: {slope:+.3f} m³/s/°C, r={r:+.2f}, p={p:.1e}")
        ax.axhline(0, color="0.7", linewidth=0.6)
        ax.axvline(0, color="0.7", linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Temperatur-Anomalie [°C] (Tageswert - Klimatologie)")
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    axes[0].set_ylabel(r"Abfluss-Anomalie $Q$ [m$^3$ s$^{-1}$] (Tageswert - Klimatologie)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig17_discharge_temp_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 18 --
def fig_spartacus_station_validation(df):
    """How well does the SPARTACUS grid point match the actual station it
    stands in for? fig16/fig17 use SPARTACUS (gap-free, methodologically
    consistent) rather than the station series - this checks that choice.
    Scatter + 1:1 line + bias/correlation, daily values, for both variables
    at both locations, over whatever years the station series covers."""
    d = df.copy()
    d["year"] = d["time"].dt.year

    panels = [
        ("Tmean_SPARTACUS_Vernagtkessel", "Tmean_Vernagt_Pangea", "Temperatur Vernagtkessel", "°C", "a"),
        ("Tmean_SPARTACUS_Vent", "Tmean_Vent_Pangea", "Temperatur Vent", "°C", "b"),
        ("RR_SPARTACUS_Vernagtkessel", "RR_Vernagt_Pangea", "Niederschlag Vernagtkessel", "mm/d", "c"),
        ("RR_SPARTACUS_Vent", "RR_Vent_Pangea", "Niederschlag Vent", "mm/d", "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.6), dpi=300)
    for ax, (spartacus_col, station_col, title, unit, lab) in zip(axes.flat, panels):
        sub = d.dropna(subset=[spartacus_col, station_col])
        bias = (sub[spartacus_col] - sub[station_col]).mean()
        r, p = stats.pearsonr(sub[spartacus_col], sub[station_col])
        y0, y1 = sub["year"].min(), sub["year"].max()
        ax.scatter(sub[station_col], sub[spartacus_col], s=4, alpha=0.12, color="0.3", linewidths=0)
        lims = [min(sub[station_col].min(), sub[spartacus_col].min()),
                max(sub[station_col].max(), sub[spartacus_col].max())]
        ax.plot(lims, lims, color="#c0392b", linewidth=1.2, linestyle="--", label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"Station [{unit}]")
        ax.set_ylabel(f"SPARTACUS [{unit}]")
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.text(0.97, 0.06,
                f"n={len(sub)} ({y0}-{y1})\nBias (SPARTACUS-Station): {bias:+.2f} {unit}\nr={r:.3f}",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig18_spartacus_station_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 19 --
def fig_data_availability(df):
    """Gantt-style data availability chart, one row per variable, coloured
    bars where a valid (non-NaN) daily value exists. The master sheet only
    contains May-Sep rows by design (Q is only meaningful/measured in the
    melt season), so this necessarily shows a comb pattern of short yearly
    segments rather than continuous bars - gaps WITHIN a segment are real
    missing data, the regular winter gaps between segments are not (out of
    scope for this sheet, not a data problem)."""
    d = df[["time"]].copy()
    full_range = pd.date_range(df["time"].min(), df["time"].max(), freq="D")

    def segments(col):
        s = df.set_index("time")[col].reindex(full_range)
        valid = s.notna().values
        segs = []
        start = None
        for i, v in enumerate(valid):
            if v and start is None:
                start = i
            elif not v and start is not None:
                segs.append((mdates.date2num(full_range[start]), mdates.date2num(full_range[i - 1]) - mdates.date2num(full_range[start]) + 1))
                start = None
        if start is not None:
            segs.append((mdates.date2num(full_range[start]), mdates.date2num(full_range[-1]) - mdates.date2num(full_range[start]) + 1))
        return segs

    rows = [
        ("Q_Vernagt", "Q Vernagtbach", C_VERNAGT_PEAK),
        ("Tmean_Vernagt_Pangea", "Temperatur Vernagt (Station)", C_VERNAGT_PEAK),
        ("RR_Vernagt_Pangea", "Niederschlag Vernagt (Station)", C_VERNAGT_PEAK),
        ("Tmean_SPARTACUS_Vernagtkessel", "Temperatur Vernagtkessel (SPARTACUS)", C_VERNAGT_PEAK),
        ("Q_Niedertalbach", "Q Rofenache (Pegel Vent)", C_NIEDER_PEAK),
        ("W_Niedertalbach", "Pegelstand Rofenache", C_NIEDER_PEAK),
        ("Tmean_Vent_Pangea", "Temperatur Vent (Station)", C_NIEDER_PEAK),
        ("RR_Vent_Pangea", "Niederschlag Vent (Station)", C_NIEDER_PEAK),
        ("TxTxTx_Vent_Pangea", "Tx Vent (Station)", C_NIEDER_PEAK),
        ("TnTnTn_Vent_Pangea", "Tn Vent (Station)", C_NIEDER_PEAK),
        ("Tmean_SPARTACUS_Vent", "Temperatur Vent (SPARTACUS)", C_NIEDER_PEAK),
        ("bmu", "Wetterlagenklasse (bmu)", "#6b6b6b"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    for i, (col, label, color) in enumerate(rows):
        y = len(rows) - i
        ax.broken_barh(segments(col), (y - 0.35, 0.7), facecolors=color)
    ax.set_yticks([len(rows) - i for i in range(len(rows))])
    ax.set_yticklabels([label for _, label, _ in rows], fontsize=8)
    ax.set_ylim(0.3, len(rows) + 0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Jahr")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Datenverfügbarkeit im Master-Datensatz (Mai–Sep je Jahr)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig19_data_availability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _monthly_sensitivity_grid(df, driver_cols, driver_label, driver_unit, outfile):
    """Shared builder for fig20 (temperature) and fig21 (precipitation):
    2 rows (Vernagtbach, Rofenache) x 5 columns (May-Sep), each panel a
    scatter of the driver's daily anomaly vs. Q's daily anomaly (both
    deseasonalized against their own 9d-smoothed day-of-year climatology),
    coloured continuously by year, with a per-panel regression line and
    slope/r/p annotated - the monthly-faceted, year-coloured layout mirrors
    the team's earlier exploratory version of this plot."""
    d = df.copy()
    d["year"] = d["time"].dt.year
    d["doy"] = d["time"].dt.dayofyear
    d["month"] = d["time"].dt.month
    months = [5, 6, 7, 8, 9]
    seas = d[d["month"].isin(months)].copy()

    rows = [("Q_Vernagt", driver_cols[0], "Vernagtbach"),
            ("Q_Niedertalbach", driver_cols[1], "Rofenache – Pegel Vent")]

    year_min, year_max = int(d["year"].min()), int(d["year"].max())
    norm = mcolors.Normalize(vmin=year_min, vmax=year_max)
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), dpi=300, sharex="col")
    for row_i, (qcol, dcol, rowtitle) in enumerate(rows):
        sub = seas.dropna(subset=[qcol, dcol]).copy()
        sub["Q_a"] = doy_anomaly(sub, qcol)
        sub["D_a"] = doy_anomaly(sub, dcol)
        sub = sub.dropna(subset=["Q_a", "D_a"])
        for col_i, m in enumerate(months):
            ax = axes[row_i, col_i]
            msub = sub[sub["month"] == m]
            ax.scatter(msub["D_a"], msub["Q_a"], c=msub["year"], cmap=cmap, norm=norm, s=6, alpha=0.5, linewidths=0)
            slope, intercept, r, p, se = stats.linregress(msub["D_a"], msub["Q_a"])
            xs = np.array([msub["D_a"].min(), msub["D_a"].max()])
            ax.plot(xs, slope * xs + intercept, color="0.15", linewidth=1.4, linestyle="--")
            ax.axhline(0, color="0.75", linewidth=0.5)
            ax.axvline(0, color="0.75", linewidth=0.5)
            ax.text(0.03, 0.96, f"n={len(msub)}\n{slope:+.2f} m³/s/{driver_unit}\np={p:.1e}",
                    transform=ax.transAxes, fontsize=6.5, va="top")
            if row_i == 0:
                ax.set_title(MONTH_ABBR_DE[m], fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"{rowtitle}\nQ-Anomalie [m³/s]", fontsize=8)
            if row_i == 1:
                ax.set_xlabel(f"{driver_label}-Anomalie", fontsize=7.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.01, aspect=40)
    cbar.set_label("Jahr")
    fig.suptitle(f"Monatliche {driver_label}sensitivität des Abflusses (Tages-Anomalien, deseasonalisiert)", fontsize=10, y=0.995)
    fig.savefig(f"{OUT_DIR}/{outfile}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure 20 --
def fig_monthly_temp_sensitivity(df):
    _monthly_sensitivity_grid(df, ["Tmean_SPARTACUS_Vernagtkessel", "Tmean_SPARTACUS_Vent"],
                               "Temperatur", "°C", "fig20_monthly_temp_sensitivity.png")


# --------------------------------------------------------------- Figure 21 --
def fig_monthly_precip_sensitivity(df):
    _monthly_sensitivity_grid(df, ["RR_SPARTACUS_Vernagtkessel", "RR_SPARTACUS_Vent"],
                               "Niederschlag", "mm", "fig21_monthly_precip_sensitivity.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    set_style()
    df = load_master()
    fig_peak_day_running_window(df)
    fig_flow_timing_cumulative(df)
    fig_small_multiples(df, "Q_Vernagt", C_VERNAGT_PEAK, "Vernagtbach - Median und IQR je Periode", "fig9_smallmult_vernagt.png")
    fig_small_multiples(df, "Q_Niedertalbach", C_NIEDER_PEAK, "Rofenache – Pegel Vent - Median und IQR je Periode", "fig9_smallmult_rofenache.png")
    fig_seasonal_by_period_gradient(df)
    fig_hump_overlay(df)
    fig_cumulative_fraction_curves(df)
    fig_cumulative_fraction_curves_by_year(df)
    fig_parde_coefficients(df)
    fig_seasonal_magnitude_trend(df)
    fig_seasonal_magnitude_bars(df)
    fig_monthly_magnitude_by_period(df)
    fig_monthly_circular_scatter(df)
    fig_climate_trends(df)
    fig_discharge_temp_sensitivity(df)
    fig_spartacus_station_validation(df)
    fig_data_availability(df)
    fig_monthly_temp_sensitivity(df)
    fig_monthly_precip_sensitivity(df)
