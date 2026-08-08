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
from scipy import stats

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
    axes[0].set_ylabel("Tag des Jahres mit\nmaximalem Abfluss (9d-geglättet)")
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
        ax.fill_between(med.index, q25, q75, color=color, alpha=0.25, linewidth=0)
        ax.plot(med.index, med.values, color=color, linewidth=1)
        ax.set_title(lab, fontsize=8)
        ax.set_ylim(0, ymax)
    for ax in axes[-1, :]:
        ax.set_xlabel("Tag d. J.", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Q [m3/s]", fontsize=7)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{outfile}", dpi=300, bbox_inches="tight")
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300, sharey=True)
    for ax, (col, title, base_color, lab) in zip(axes, panels):
        for (y0, y1), ls, alpha, label in [(early, "--", 0.9, f"{early[0]}-{early[1]}"),
                                            (late, "-", 1.0, f"{late[0]}-{late[1]}")]:
            sub = d[(d["year"] >= y0) & (d["year"] <= y1)].dropna(subset=[col])
            med = sub.groupby("doy")[col].median()
            med_norm = med / med.max()
            ax.plot(med.index, med_norm.values, color=base_color, linewidth=1.6, linestyle=ls,
                    alpha=alpha, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Tag des Jahres")
        ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
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
    axes[0].set_ylabel("Tag des Jahres (kumulatives\nAbfluss-Datum)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig11_flow_timing_cumulative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    set_style()
    df = load_master()
    fig_peak_day_running_window(df)
    fig_flow_timing_cumulative(df)
    fig_small_multiples(df, "Q_Vernagt", C_VERNAGT_PEAK, "Vernagtbach - Median und IQR je Periode", "fig9_smallmult_vernagt.png")
    fig_small_multiples(df, "Q_Niedertalbach", C_NIEDER_PEAK, "Rofenache – Pegel Vent - Median und IQR je Periode", "fig9_smallmult_rofenache.png")
    fig_hump_overlay(df)
