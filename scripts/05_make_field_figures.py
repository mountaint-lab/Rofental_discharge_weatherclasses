"""
Figures for the HEFEX-III Levelstick field campaign (Hintereisferner, 06.-10.08.2025).
Standalone from 04_make_figures.py (different data source, short high-frequency
field deployment instead of the 50-year daily master record) but reuses the same
house style conventions (blue-to-red = time gradient, green family = Rofenache).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

LEVELSTICK_XLSX = "/root/.claude/uploads/162f5734-736e-5c1e-81c8-f083b733fb54/adb97b11-HEFEX_levelsticks_trimmed.xlsx"
OUT_DIR = "./figures"

SITES = [
    ("Rofenache_deep", "Rofenache (deep)", "#0f6b3a"),
    ("Rofenache_low", "Rofenache (low)", "#7fbf8f"),
    ("supraglacial_stream", "Supraglazialer Bach (Q2)", "#c9820c"),
]


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    })


def load_sites():
    xl = pd.ExcelFile(LEVELSTICK_XLSX)
    data = {}
    for sheet, _, _ in SITES:
        df = xl.parse(sheet)
        df = df.rename(columns={"Time [Europe/Rome]": "time", "Pressure [hPa]": "pressure", "Temperature [°C]": "temp"})
        df["time"] = pd.to_datetime(df["time"])
        data[sheet] = df
    return data


# --------------------------------------------------------------- Figure F1 --
def fig_levelstick_timeseries(data):
    """Full campaign time series, pressure(=level proxy) and water temperature,
    all 3 sites overlaid, our colours (green family = Rofenache deep/low,
    amber = supraglacial stream)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=300, sharex=True)
    for sheet, label, color in SITES:
        df = data[sheet]
        ax1.plot(df["time"], df["pressure"], color=color, linewidth=0.8, label=label)
        ax2.plot(df["time"], df["temp"], color=color, linewidth=0.8, label=label)
    ax1.set_ylabel("Druck [hPa]\n(Wasserstand-Proxy)")
    ax1.axhline(0, color="0.7", linewidth=0.6)
    ax1.legend(loc="upper right", frameon=False, ncol=3)
    ax1.text(0.0, 1.08, "a", transform=ax1.transAxes, fontsize=11, fontweight="bold", va="top")
    ax2.set_ylabel("Wassertemperatur [°C]")
    ax2.set_xlabel("Datum (06.-10.08.2025)")
    ax2.axhline(0, color="0.7", linewidth=0.6)
    ax2.text(0.0, 1.08, "b", transform=ax2.transAxes, fontsize=11, fontweight="bold", va="top")
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
    fig.suptitle("HEFEX-III Levelstick-Kampagne, Hintereisferner (06.-10.08.2025)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/field_f1_levelstick_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure F2 --
def fig_levelstick_diurnal(data):
    """Diurnal cycle per site, common y-scale, one line per campaign day
    coloured blue (06.08.) to red (10.08.) - same time-gradient convention as
    the 50-year figures (fig7/fig12)."""
    all_days = sorted(set().union(*[set(df["time"].dt.date) for df in data.values()]))
    norm = mcolors.Normalize(vmin=0, vmax=len(all_days) - 1)
    cmap = plt.get_cmap("coolwarm")

    ymin = min(df["pressure"].min() for df in data.values())
    ymax = max(df["pressure"].max() for df in data.values())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=300, sharey=True)
    for ax, (sheet, label, _) in zip(axes, SITES):
        df = data[sheet].copy()
        df["date"] = df["time"].dt.date
        df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60
        for i, day in enumerate(all_days):
            sub = df[df["date"] == day].sort_values("hour")
            if len(sub) == 0:
                continue
            ax.plot(sub["hour"], sub["pressure"], color=cmap(norm(i)), linewidth=1.1,
                    label=day.strftime("%d.%m."))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Uhrzeit [h]")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.axhline(0, color="0.7", linewidth=0.6)
    axes[0].set_ylabel("Druck [hPa] (Wasserstand-Proxy)")
    axes[0].set_ylim(ymin - 1, ymax + 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=len(all_days), frameon=False, fontsize=8)
    fig.suptitle("Tagesgang je Standort - gemeinsamer Maßstab", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/field_f2_levelstick_diurnal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    set_style()
    data = load_sites()
    fig_levelstick_timeseries(data)
    fig_levelstick_diurnal(data)
