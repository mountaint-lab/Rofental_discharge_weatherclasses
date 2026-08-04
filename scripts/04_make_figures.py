"""
Consolidated, final versions of all figures built during the analysis.
Each function is standalone - call only the ones you need.

Input files expected (adjust paths below):
    MASTER_XLSX     - output of 03_build_master_dataset.py
    SMARTSTAKE_CSV  - smartstake_daily_processed_2021_2024.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors

MASTER_XLSX = "./data/processed/ssim_q_weather_time_vs03_corrected.xlsx"      # <-- adjust
MASTER_SHEET = "clean_complete_cases"
SMARTSTAKE_CSV = "./data/raw/smartstake_daily_processed_2021_2024.csv"  # <-- adjust
OUT_DIR = "./figures"

C_NIEDER = "#1b4f72"
C_VERNAGT = "#922b21"
C_BAR = "#3d6d9c"

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
    df["klasse"] = df["bmu"] + 1
    return df.sort_values("time").reset_index(drop=True)


def break_gaps(d: pd.DataFrame, gap_days=10) -> pd.DataFrame:
    """Insert NaN rows at large time gaps so plotted lines don't bridge them
    (e.g. the winter gap between two May-Sept seasons)."""
    d = d.copy()
    dt = d["time"].diff().dt.days
    gap_idx = d.index[dt > gap_days]
    insert_rows = []
    for i in gap_idx:
        row = {c: np.nan for c in d.columns}
        row["time"] = d.loc[i - 1, "time"] + pd.Timedelta(days=1)
        insert_rows.append(row)
    if insert_rows:
        d = pd.concat([d, pd.DataFrame(insert_rows)], ignore_index=True).sort_values("time").reset_index(drop=True)
    return d


# ---------------------------------------------------------------- Figure 1 --
def fig_discharge_overview(df):
    d1 = break_gaps(df[["time", "Q_Niedertalbach", "Q_Vernagt"]])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.2), sharex=True, dpi=300)
    axes[0].plot(d1["time"], d1["Q_Niedertalbach"], color=C_NIEDER, linewidth=0.5)
    axes[0].set_ylabel(r"$Q$ Niedertalbach [m$^3$ s$^{-1}$]")
    axes[0].text(0.005, 0.92, "a", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top")
    axes[1].plot(d1["time"], d1["Q_Vernagt"], color=C_VERNAGT, linewidth=0.5)
    axes[1].set_ylabel(r"$Q$ Vernagtbach [m$^3$ s$^{-1}$]")
    axes[1].set_xlabel("Jahr")
    axes[1].text(0.005, 0.92, "b", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="top")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.margins(x=0.01)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_discharge_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1c -
def fig_discharge_overlay_robust(df):
    """Both stations on one panel, normalised to their 1st-99th percentile
    range (robust to the single 1978 Niedertalbach outlier)."""
    def robust_norm(s, lo=1, hi=99):
        p_lo, p_hi = np.nanpercentile(s, [lo, hi])
        return (s - p_lo) / (p_hi - p_lo)

    d = df[["time", "Q_Niedertalbach", "Q_Vernagt"]].copy()
    d["Q_Niedertalbach_norm"] = robust_norm(d["Q_Niedertalbach"])
    d["Q_Vernagt_norm"] = robust_norm(d["Q_Vernagt"])
    d = break_gaps(d[["time", "Q_Niedertalbach_norm", "Q_Vernagt_norm"]])

    fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=300)
    ax.plot(d["time"], d["Q_Niedertalbach_norm"], color=C_NIEDER, linewidth=0.5, label="Niedertalbach", alpha=0.85)
    ax.plot(d["time"], d["Q_Vernagt_norm"], color=C_VERNAGT, linewidth=0.5, label="Vernagtbach", alpha=0.85)
    ax.axhline(0, color="0.7", linewidth=0.5)
    ax.axhline(1, color="0.7", linewidth=0.5)
    ax.set_ylabel("Normierter Abfluss\n(1.\u201399. Perzentil \u2192 0\u20131)")
    ax.set_xlabel("Jahr")
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1c_discharge_overlay_robust.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2 --
def fig_discharge_by_class(df):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=300)
    specs = [("Q_Niedertalbach_anom", "Niedertalbach (1974-2023)", C_NIEDER, "a"),
             ("Q_Vernagt_anom", "Vernagtbach (1974-2019)", C_VERNAGT, "b")]
    for ax, (col, title, color, lab) in zip(axes, specs):
        data_by_class = [df.loc[df["klasse"] == k, col].dropna().values for k in range(1, 10)]
        n_by_class = [len(x) for x in data_by_class]
        ax.boxplot(data_by_class, positions=range(1, 10), widths=0.55, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", linewidth=1.2),
                   boxprops=dict(facecolor=color, alpha=0.30, edgecolor=color, linewidth=0.8),
                   whiskerprops=dict(color=color, linewidth=0.8), capprops=dict(color=color, linewidth=0.8))
        ax.axhline(0, color="0.4", linewidth=0.6, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Wetterlagenklasse")
        ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ymax = ax.get_ylim()[1]
        ax.set_ylim(top=ymax * 1.14)
        for k, nn in zip(range(1, 10), n_by_class):
            ax.text(k, ax.get_ylim()[1] * 0.965, f"n={nn}", ha="center", va="top",
                    fontsize=6.3, color="0.45", rotation=90)
    axes[0].set_ylabel(r"Abfluss-Anomalie [m$^3$ s$^{-1}$]")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_discharge_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3 --
def fig_class_sizes(df):
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=300)
    counts = df["klasse"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=C_BAR, width=0.6, edgecolor="none")
    ax.set_xlabel("Wetterlagenklasse")
    ax.set_ylabel("Anzahl Tage")
    ax.set_xticks(range(1, 10))
    for k, v in counts.items():
        ax.text(k, v + 12, str(v), ha="center", fontsize=7.5)
    ax.set_ylim(top=counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_class_sizes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 7 --
def fig_seasonal_by_period_gradient(df):
    """Median discharge by day-of-year, one line per 5-year period, coloured
    on a gradient (early=dark purple, late=yellow) instead of a stacked area
    chart (which is hard to read with >5 overlapping series)."""
    d = df.copy()
    d["doy"] = d["time"].dt.dayofyear
    d["year"] = d["time"].dt.year

    def period_label(y):
        for a, b in PERIOD_BINS:
            if a <= y <= b:
                return f"{a}\u2013{b}"
        return None

    d["periode"] = d["year"].apply(period_label)
    period_labels = [f"{a}\u2013{b}" for a, b in PERIOD_BINS]
    cmap = plt.get_cmap("viridis", len(PERIOD_BINS))
    colors = {lab: cmap(i) for i, lab in enumerate(period_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300, sharex=True)
    for ax, col, title, lab in zip(axes, ["Q_Niedertalbach", "Q_Vernagt"], ["Niedertalbach", "Vernagtbach"], ["a", "b"]):
        for pl in period_labels:
            sub = d[d["periode"] == pl]
            if len(sub) == 0:
                continue
            med = sub.groupby("doy")[col].median()
            med = med.reindex(range(med.index.min(), med.index.max() + 1))
            ax.plot(med.index, med.values, color=colors[pl], linewidth=1.1, label=pl)
        ax.set_title(title, fontsize=10)
        ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.set_xlabel("Tag des Jahres")
    axes[0].set_ylabel(r"Median $Q$ [m$^3$ s$^{-1}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT_DIR}/fig7_seasonal_by_period_gradient.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 9 --
def fig_small_multiples(df, col, color, title, outfile):
    """One panel per 5-year period, median + IQR, shared y-axis for fair comparison."""
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


# --------------------------------------------------------------- Figure 10 --
def fig_peak_day_running_window(df):
    """For each year, find the (9-day-smoothed) day-of-year of maximum
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
            smooth = med.rolling(9, center=True, min_periods=5).mean()
            rows.append((y, smooth.idxmax()))
        return pd.DataFrame(rows, columns=["year", "peak_day"]).set_index("year")

    def shades(base_hex, n):
        base = np.array(mcolors.to_rgb(base_hex))
        white = np.array([1, 1, 1])
        fracs = np.linspace(0.62, 0.0, n)
        return [tuple(base * (1 - f) + white * f) for f in fracs]

    panels = [("Q_Vernagt", "Vernagtbach", C_VERNAGT, "a"), ("Q_Niedertalbach", "Niedertalbach", C_NIEDER, "b")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    for ax, (col, title, base_color, lab) in zip(axes, panels):
        pk = peak_day_per_year(col)
        ax.plot(pk.index, pk["peak_day"], color="0.8", marker="o", markersize=3, linewidth=0.7,
                label="Jahreswert", zorder=1)
        for w, c in zip([5, 10, 15], shades(base_color, 3)):
            rm = pk["peak_day"].rolling(w, center=True, min_periods=max(3, w // 2)).mean()
            ax.plot(rm.index, rm.values, color=c, linewidth=1.6, label=f"{w}-Jahres-Mittel")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Jahr")
        ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("Tag des Jahres mit\nmaximalem Abfluss")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig10_peak_day_running_window.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- Smartstake figures -
def fig_smartstake_ablation(sensor_outage_start="2024-11-07"):
    df = pd.read_csv(SMARTSTAKE_CSV, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=300)
    ax.plot(df["Timestamp"], df["ablation_diff"], color=C_NIEDER, linewidth=0.6)
    ax.axhline(0, color="0.5", linewidth=0.6)
    ax.axvspan(pd.Timestamp(sensor_outage_start, tz="UTC"), df["Timestamp"].max(), color="0.85", alpha=0.5, zorder=0)
    ax.text(pd.Timestamp(sensor_outage_start, tz="UTC") + pd.Timedelta(days=3), ax.get_ylim()[0],
            "Sensorausfall", fontsize=7, color="0.4", rotation=90, va="bottom", ha="center")
    ax.set_ylabel("Tagesmittel Ablation [m/Tag]")
    ax.set_xlabel("Jahr")
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_smartstake_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_smartstake_temp_split():
    """Ablation vs. air temperature, split at 0 degC: correlation is strong
    above freezing (melt regime) and essentially zero below (no melt regardless
    of how cold) - a cleaner statement than one overall correlation coefficient."""
    df = pd.read_csv(SMARTSTAKE_CSV, parse_dates=["Timestamp"])
    sub = df.dropna(subset=["Temperature [degC]"]).copy()
    warm = sub[sub["Temperature [degC]"] > 0]
    cold = sub[sub["Temperature [degC]"] <= 0]

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=300)
    r_cold = cold["Temperature [degC]"].corr(cold["ablation_diff"])
    r_warm = warm["Temperature [degC]"].corr(warm["ablation_diff"])
    ax.scatter(cold["Temperature [degC]"], cold["ablation_diff"], s=6, alpha=0.5, color=C_NIEDER,
               linewidths=0, label=f"T \u2264 0\u00b0C (n={len(cold)}), r={r_cold:.2f}")
    ax.scatter(warm["Temperature [degC]"], warm["ablation_diff"], s=6, alpha=0.5, color=C_VERNAGT,
               linewidths=0, label=f"T > 0\u00b0C (n={len(warm)}), r={r_warm:.2f}")
    for d, c in [(warm, C_VERNAGT), (cold, C_NIEDER)]:
        m, b0 = np.polyfit(d["Temperature [degC]"], d["ablation_diff"], 1)
        xs = np.linspace(d["Temperature [degC]"].min(), d["Temperature [degC]"].max(), 20)
        ax.plot(xs, m * xs + b0, color=c, linewidth=1.2)
    ax.axhline(0, color="0.6", linewidth=0.5)
    ax.axvline(0, color="0.6", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lufttemperatur [Grad C]")
    ax.set_ylabel("Ablation [m/Tag]")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_smartstake_temp_split.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    set_style()

    df = load_master()
    fig_discharge_overview(df)
    fig_discharge_overlay_robust(df)
    fig_discharge_by_class(df)
    fig_class_sizes(df)
    fig_seasonal_by_period_gradient(df)
    fig_small_multiples(df, "Q_Niedertalbach", C_NIEDER, "Niedertalbach - Median und IQR je Periode", "fig9_smallmult_nieder.png")
    fig_small_multiples(df, "Q_Vernagt", C_VERNAGT, "Vernagtbach - Median und IQR je Periode", "fig9_smallmult_vernagt.png")
    fig_peak_day_running_window(df)

    fig_smartstake_ablation()
    fig_smartstake_temp_split()

    print(f"All figures written to {OUT_DIR}/")
