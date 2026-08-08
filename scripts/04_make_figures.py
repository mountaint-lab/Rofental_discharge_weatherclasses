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

    panels = [("Q_Vernagt", "Vernagtbach", C_VERNAGT_PEAK, "a"), ("Q_Niedertalbach", "Niedertalbach", C_NIEDER_PEAK, "b")]
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


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    set_style()
    df = load_master()
    fig_peak_day_running_window(df)
