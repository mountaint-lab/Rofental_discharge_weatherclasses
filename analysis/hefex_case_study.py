"""Plot HEFEX meteorology over the full campaign with the available Rofenache signals.

The meteorological panels cover 4--12 August 2025.  The lower panel preserves
that shared time axis but draws relative water-level signals only during their
actual observation window (6--10 August); no missing values are extrapolated.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
INPUT = PROJECT_ROOT / "data" / "hefex_iop1" / "HEFEX_Kampagne_Aug2025.xlsx"
OUT_DIR = PROJECT_ROOT / "outputs" / "hefex_case_study"
OUTPUT = OUT_DIR / "hefex_2025_wetter_energie_wasserstand_stuendlich"

START = pd.Timestamp("2025-08-04 00:00")
END = pd.Timestamp("2025-08-13 00:00")
CLASS_COLORS = {4: "#A7BEDD", 6: "#E7A06A"}
COLORS = {
    "Latschbloder": "#6A4C93",
    "Proviantdepot": "#C76D2C",
    "Vent": "#505050",
    "Mitte": "#1D5E7A",
    "Ufer": "#5F8F3E",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aws = pd.read_excel(INPUT, sheet_name="AWS_10min", dtype={"Zeit": str})
    daily = pd.read_excel(INPUT, sheet_name="Tagesdaten", dtype={"Datum": str})
    levels = pd.read_excel(INPUT, sheet_name="Levelsticks", dtype={"Zeit_MESZ": str})
    aws["Zeit"] = pd.to_datetime(aws["Zeit"], dayfirst=True)
    daily["Datum"] = pd.to_datetime(daily["Datum"], dayfirst=True)
    levels["Zeit_MESZ"] = pd.to_datetime(levels["Zeit_MESZ"], dayfirst=True)
    aws = aws.loc[aws["Zeit"].between(START, END, inclusive="left")].copy()
    daily = daily.loc[daily["Datum"].between(START.normalize(), (END - pd.Timedelta(days=1)).normalize())].copy()
    levels = levels.loc[levels["Zeit_MESZ"].between(START, END, inclusive="left")].copy()

    aws["Latschbloder_Rnetto"] = (
        aws["Latschbloder_SW_ein"] - aws["Latschbloder_SW_aus"]
        + aws["Latschbloder_LW_ein"] - aws["Latschbloder_LW_aus"]
    )
    aws["Proviantdepot_Rnetto"] = (
        aws["Proviantdepot_SW_ein"] - aws["Proviantdepot_SW_aus"]
        + aws["Proviantdepot_LW_ein"] - aws["Proviantdepot_LW_aus"]
    )
    hourly_aws = aws.set_index("Zeit").resample("1h").mean(numeric_only=True)
    level_indexed = levels.set_index("Zeit_MESZ")
    for column in ("Rofenache_Mitte_Wasserstand_hPa", "Rofenache_Ufer_Wasserstand_hPa"):
        baseline = level_indexed[column].dropna().iloc[:6].median()
        level_indexed[column] = level_indexed[column] - baseline
    hourly_levels = level_indexed.resample("1h").mean(numeric_only=True)
    return hourly_aws, daily, hourly_levels


def add_class_strip(axis: plt.Axes, daily: pd.DataFrame) -> None:
    blocks = daily.assign(block=daily["Wetterklasse"].ne(daily["Wetterklasse"].shift()).cumsum())
    for _, block in blocks.groupby("block", sort=False):
        number = int(block["Wetterklasse"].iloc[0])
        first = block["Datum"].min()
        last = block["Datum"].max() + pd.Timedelta(days=1)
        axis.axvspan(first, last, color=CLASS_COLORS[number], alpha=0.9)
        axis.text(first + (last - first) / 2, 0.5, f"Klasse {number}", ha="center", va="center", fontweight="bold")
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_ylabel("Wetterlage")
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.spines["bottom"].set_visible(False)


def add_day_lines(axis: plt.Axes) -> None:
    for day in pd.date_range(START + pd.Timedelta(days=1), END - pd.Timedelta(days=1), freq="D"):
        axis.axvline(day, color="#D2D2D2", linewidth=0.6, zorder=0)


def plot(hourly_aws: pd.DataFrame, daily: pd.DataFrame, hourly_levels: pd.DataFrame) -> None:
    set_style()
    fig = plt.figure(figsize=(9.3, 8.1))
    grid = fig.add_gridspec(4, 1, height_ratios=[0.33, 1.25, 1.25, 1.05], hspace=0.16)
    classes, temperature, radiation, water = [fig.add_subplot(grid[index]) for index in range(4)]

    add_class_strip(classes, daily)

    for column, label in (("Latschbloder_T_C", "Latschbloder"), ("Proviantdepot_T_C", "Proviantdepot")):
        temperature.plot(hourly_aws.index, hourly_aws[column], color=COLORS[label], linewidth=1.1, label=label)
    vent_times = daily["Datum"] + pd.Timedelta(hours=12)
    temperature.scatter(vent_times, daily["T_Vent_C"], color=COLORS["Vent"], s=18, marker="s", zorder=4, label="Vent (Tagesmittel)")
    temperature.set_ylabel("Lufttemperatur [°C]")
    temperature.set_title("HEFEX-Kampagne: stündliche meteorologische Entwicklung", loc="left", fontweight="bold")
    temperature.legend(frameon=False, ncol=3, loc="upper left")
    temperature.tick_params(axis="x", bottom=False, labelbottom=False)
    temperature.spines["bottom"].set_visible(False)

    for column, label in (("Latschbloder_Rnetto", "Latschbloder"), ("Proviantdepot_Rnetto", "Proviantdepot")):
        radiation.plot(hourly_aws.index, hourly_aws[column], color=COLORS[label], linewidth=1.0, label=label)
    radiation.axhline(0, color="#707070", linewidth=0.75)
    radiation.set_ylabel("Netto-Strahlung [W m$^{-2}$]")
    radiation.legend(frameon=False, ncol=2, loc="upper left")
    radiation.tick_params(axis="x", bottom=False, labelbottom=False)
    radiation.spines["bottom"].set_visible(False)

    water.plot(hourly_levels.index, hourly_levels["Rofenache_Mitte_Wasserstand_hPa"], color=COLORS["Mitte"], linewidth=1.2, label="Rofenache Mitte")
    water.plot(hourly_levels.index, hourly_levels["Rofenache_Ufer_Wasserstand_hPa"], color=COLORS["Ufer"], linewidth=1.2, label="Rofenache Ufer")
    water.axhline(0, color="#707070", linewidth=0.75)
    water.set_ylabel("relative Wasserstands-\nänderung [hPa]")
    water.set_xlabel("Zeit [MESZ]")
    water.legend(frameon=False, ncol=2, loc="upper left")

    for axis in (classes, temperature, radiation, water):
        axis.set_xlim(START, END)
    for axis in (temperature, radiation, water):
        add_day_lines(axis)
    water.xaxis.set_major_locator(mdates.DayLocator())
    water.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
    fig.text(
        0.5,
        0.01,
        "Lufttemperatur und Netto-Strahlung: Stundenmittel aus 10-Minuten-Werten. Wasserstände: Stundenmittel aus 5-Minuten-Drucksignalen, relativ zum jeweiligen Messbeginn; keine Abflusskalibrierung.",
        ha="center",
        fontsize=7.1,
        color="#555555",
    )
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.94)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Kampagnendatei fehlt: {INPUT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot(*read_data())


if __name__ == "__main__":
    main()
