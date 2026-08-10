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
IMET_RAW_DIR = "/tmp/claude-0/-home-user-Rofental-discharge-weatherclasses/162f5734-736e-5c1e-81c8-f083b733fb54/scratchpad/imet_graz/imet_graz/raw"
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


# ------------------------------------------------------------- Figure F3 ---
# HEFEX3 fieldbook (Q1/Q2 launch site per date+time+IMET serial), transcribed
# directly from "HEFEX3 Fieldbook Graz.pdf" - needed because the raw UAV CSVs
# are organised by IMET logger serial number, not by launch site.
FIELDBOOK_LOG = [
    ("2025-08-06", "17:00", "Q1", 65805), ("2025-08-06", "17:00", "Q2", 65808),
    ("2025-08-06", "20:00", "Q1", 65805), ("2025-08-06", "20:00", "Q2", 65808),
    ("2025-08-07", "05:00", "Q1", 65805), ("2025-08-07", "05:00", "Q2", 65805),
    ("2025-08-07", "08:00", "Q1", 65805), ("2025-08-07", "08:00", "Q2", 65808),
    ("2025-08-07", "11:00", "Q1", 65805), ("2025-08-07", "11:00", "Q2", 65808),
    ("2025-08-07", "14:00", "Q1", 65805), ("2025-08-07", "14:00", "Q2", 65808),
    ("2025-08-07", "17:00", "Q1", 65805), ("2025-08-07", "17:00", "Q2", 65808),
    ("2025-08-07", "20:00", "Q1", 65805), ("2025-08-07", "20:00", "Q2", 65808),
    ("2025-08-07", "23:00", "Q1", 65805), ("2025-08-07", "23:00", "Q2", 65808),
    ("2025-08-08", "05:00", "Q1", 65805), ("2025-08-08", "05:00", "Q2", 65808),
    ("2025-08-08", "08:00", "Q1", 65805), ("2025-08-08", "08:00", "Q2", 65808),
    ("2025-08-08", "11:00", "Q1", 65805), ("2025-08-08", "11:00", "Q2", 65808),
    ("2025-08-08", "14:00", "Q1", 65805), ("2025-08-08", "14:00", "Q2", 65808),
    ("2025-08-08", "17:00", "Q1", 65805), ("2025-08-08", "17:00", "Q2", 65805),
    ("2025-08-08", "20:00", "Q1", 65805), ("2025-08-08", "20:00", "Q2", 61169),
    ("2025-08-08", "23:00", "Q1", 65805), ("2025-08-08", "23:00", "Q2", 61169),
    ("2025-08-09", "05:00", "Q1", 65805), ("2025-08-09", "05:00", "Q2", 61169),
    ("2025-08-09", "08:00", "Q1", 65805), ("2025-08-09", "08:00", "Q2", 61169),
    ("2025-08-09", "11:00", "Q1", 65805), ("2025-08-09", "11:00", "Q2", 61169),
    ("2025-08-09", "14:00", "Q1", 65805), ("2025-08-09", "14:00", "Q2", 65805),
    ("2025-08-09", "17:00", "Q1", 65805), ("2025-08-09", "17:00", "Q2", 65805),
    ("2025-08-09", "20:00", "Q1", 65805), ("2025-08-09", "20:00", "Q2", 65805),
    ("2025-08-09", "23:00", "Q1", 65805), ("2025-08-09", "23:00", "Q2", 65805),
    ("2025-08-10", "05:00", "Q1", 65805), ("2025-08-10", "05:00", "Q2", 65805),
    ("2025-08-10", "07:00", "Q1", 65805), ("2025-08-10", "07:00", "Q2", 65805),
    ("2025-08-10", "08:00", "Q1", 65805), ("2025-08-10", "08:00", "Q2", 65805),
]
IMET_FILE_PREFIX = {"2": 61169, "3": 65805, "4": 65808}


def load_q2_uav_flights():
    """Parses the raw per-IMET-logger UAV CSVs, splits into individual
    ascent-flight segments (>10min time gap = new flight), matches each
    segment against the fieldbook log (nearest-time, same IMET serial) to
    identify which were flown at Q2, and keeps only physically plausible
    ones (altitude span > 80m = a real ascent, air temperature within a
    sane -15..25 degC range) - several of the raw files are explicitly
    labelled by the field team as (possibly) broken sensor sessions, and a
    plausibility check on the actual values is more defensible than trusting
    the filename label alone."""
    import glob

    log_df = pd.DataFrame(FIELDBOOK_LOG, columns=["date", "time", "site", "imet"])
    log_df["local_dt"] = pd.to_datetime(log_df["date"] + " " + log_df["time"])

    flights = []
    for path in sorted(glob.glob(f"{IMET_RAW_DIR}/*.csv")):
        prefix = path.split("/")[-1].split("_")[0]
        imet_serial = IMET_FILE_PREFIX[prefix]
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["xq_dt"] = pd.to_datetime(df["XQ-iMet-XQ Date"] + " " + df["XQ-iMet-XQ Time"], errors="coerce", format="mixed")
        sub = df.dropna(subset=["xq_dt"])
        sub = sub[(sub["XQ-iMet-XQ Altitude"] > 1000) & (sub["XQ-iMet-XQ Altitude"] < 4500)]
        sub = sub.sort_values("xq_dt")
        if len(sub) == 0:
            continue
        gaps = sub["xq_dt"].diff() > pd.Timedelta(minutes=10)
        sub["flight_id"] = gaps.cumsum()
        for fid, g in sub.groupby("flight_id"):
            if len(g) < 30:
                continue
            local_start = g["xq_dt"].min() + pd.Timedelta(hours=2)  # UTC -> Europe/Rome (CEST)
            alt_span = g["XQ-iMet-XQ Altitude"].max() - g["XQ-iMet-XQ Altitude"].min()
            t_min, t_max = g["XQ-iMet-XQ Air Temperature"].min(), g["XQ-iMet-XQ Air Temperature"].max()
            plausible = (alt_span > 80) and (-15 < t_min) and (t_max < 25)
            cand = log_df[log_df["imet"] == imet_serial].copy()
            cand["dt_diff"] = (cand["local_dt"] - local_start).abs()
            best = cand.loc[cand["dt_diff"].idxmin()] if len(cand) else None
            site = best["site"] if best is not None and best["dt_diff"] < pd.Timedelta(minutes=90) else "?"
            if site == "Q2" and plausible:
                gg = g.copy()
                gg["local_start"] = local_start
                flights.append(gg)
    return flights


def fig_uav_q2_profiles():
    """UAV vertical profiles (temperature and relative humidity vs. pressure,
    i.e. altitude) at Q2 - the same site as the supraglacial-stream
    levelstick - coloured by time like field_f2, so the atmospheric profile
    and the stream response can be read side by side. Covers 06.08. 17:09 to
    09.08. 10:54 only: after that Q2 switched back to IMET 65805, the same
    logger used at Q1 throughout, and the two sites can no longer be told
    apart from the raw files alone (see 05_make_field_figures.py docstring)."""
    flights = load_q2_uav_flights()
    if not flights:
        print("No plausible Q2 UAV flights found - skipping fig_uav_q2_profiles.")
        return

    starts = sorted(f["local_start"].iloc[0] for f in flights)
    norm = mcolors.Normalize(vmin=0, vmax=len(starts) - 1)
    cmap = plt.get_cmap("coolwarm")
    order = {s: i for i, s in enumerate(starts)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), dpi=300)
    for g in flights:
        t0 = g["local_start"].iloc[0]
        color = cmap(norm(order[t0]))
        label = t0.strftime("%d.%m. %H:%M")
        ax1.plot(g["XQ-iMet-XQ Air Temperature"], g["XQ-iMet-XQ Pressure"], color=color, linewidth=0.8, label=label)
        ax2.plot(g["XQ-iMet-XQ Humidity"], g["XQ-iMet-XQ Pressure"], color=color, linewidth=0.8, label=label)
    for ax in (ax1, ax2):
        ax.invert_yaxis()
    ax1.set_xlabel("Temperatur [°C]")
    ax1.set_ylabel("Druck [hPa]")
    ax2.set_xlabel("Rel. Feuchte [%]")
    ax2.set_xlim(0, 100)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, fontsize=7)
    fig.suptitle("UAV-Vertikalprofile an Standort Q2 (supraglazialer Bach), 06.-09.08.2025", fontsize=10, y=1.12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/field_f3_uav_q2_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    set_style()
    data = load_sites()
    fig_levelstick_timeseries(data)
    fig_levelstick_diurnal(data)
    fig_uav_q2_profiles()
