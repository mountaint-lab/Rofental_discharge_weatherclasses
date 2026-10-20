"""Scientific climate-trend figures from the complete SPARTACUS time series.

Outputs are written as new files in ``figures/variants``. Existing figures
are never overwritten. Daily SPARTACUS files are used directly so seasonal
precipitation sums are not biased by gaps in the discharge-joined master file.
"""

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
CLIMATE_DIR = PROJECT_ROOT / "data" / "climat"
TEMPERATURE_FILE = CLIMATE_DIR / "SPARTACUS_TM24_Datensatz_19610801_20260806.csv"
PRECIPITATION_FILE = CLIMATE_DIR / "SPARTACUS_RR_Datensatz_19610801_20260806.csv"
OUT_DIR = SCRIPT_DIR / "figures" / "variants"

START_YEAR, END_YEAR = 1974, 2023
MONTHS = (5, 6, 7, 8, 9)
MONTH_NAMES = ("Mai", "Jun", "Jul", "Aug", "Sep")
EARLY = (1974, 1988)
RECENT = (2009, 2023)

BLUE = "#4169D8"
RED = "#C5162E"
GREY = "#BDBDBD"
BLACK = "#262626"
GRID = "#D9D9D9"

SITES = {
    "Vernagtkessel": {
        "lat": 46.864788,
        "lon": 10.8206415,
        "colour": "#7953A6",
        "dark": "#503377",
    },
    "Vent": {
        "lat": 46.857796,
        "lon": 10.912837,
        "colour": "#8FBC65",
        "dark": "#517A35",
    },
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.facecolor": "white",
        }
    )


def sen_mann_kendall(years, values) -> dict[str, float]:
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(y)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
        if x[j] != x[i]
    ]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))

    s_value = sum(
        int(np.sign(y[j] - y[i]))
        for i in range(n - 1)
        for j in range(i + 1, n)
    )
    _, tie_counts = np.unique(y, return_counts=True)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts if t > 1)
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if s_value > 0:
        z_value = (s_value - 1) / sqrt(variance)
    elif s_value < 0:
        z_value = (s_value + 1) / sqrt(variance)
    else:
        z_value = 0.0
    p_value = erfc(abs(z_value) / sqrt(2.0))
    return {"slope": slope, "intercept": intercept, "p": p_value, "n": n}


def p_text(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0,001"
    return f"p = {p_value:.3f}".replace(".", ",")


def load_spartacus() -> pd.DataFrame:
    temperature = pd.read_csv(TEMPERATURE_FILE).rename(columns={"TM24 [degC]": "temperature"})
    precipitation = pd.read_csv(PRECIPITATION_FILE).rename(columns={"RR [kg m-2]": "precipitation"})
    keys = ["time", "lat", "lon"]
    data = temperature.merge(precipitation, on=keys, how="inner", validate="one_to_one")
    data["time"] = pd.to_datetime(data["time"], utc=True)
    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    data = data[
        data["year"].between(START_YEAR, END_YEAR) & data["month"].isin(MONTHS)
    ].copy()

    frames = []
    for site, properties in SITES.items():
        distance = np.hypot(
            data["lat"] - properties["lat"], data["lon"] - properties["lon"]
        )
        location = data[distance < 1e-5].copy()
        location["site"] = site
        frames.append(location)
    result = pd.concat(frames, ignore_index=True)

    coverage = result.groupby(["site", "year"]).size()
    if coverage.min() < 150:
        raise ValueError("Unexpected gaps in the raw SPARTACUS May-September series")
    return result


def annual_values(data: pd.DataFrame) -> pd.DataFrame:
    annual = (
        data.groupby(["site", "year"])
        .agg(
            temperature=("temperature", "mean"),
            precipitation=("precipitation", "sum"),
            n_days=("time", "size"),
        )
        .reset_index()
    )
    return annual


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_annual_climate_trends(data: pd.DataFrame) -> pd.DataFrame:
    annual = annual_values(data)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex="col", dpi=180)
    summary_rows = []
    panel_letters = (("a", "b"), ("c", "d"))

    for column_index, (site, properties) in enumerate(SITES.items()):
        site_data = annual[annual["site"] == site].set_index("year")
        for row_index, (variable, unit, decimals) in enumerate(
            (("temperature", "°C", 2), ("precipitation", "mm", 1))
        ):
            ax = axes[row_index, column_index]
            series = site_data[variable]
            years = series.index.to_numpy(dtype=float)
            values = series.to_numpy(dtype=float)
            colour, dark = properties["colour"], properties["dark"]

            if variable == "temperature":
                ax.plot(years, values, color=GREY, linewidth=0.7, zorder=1)
                ax.scatter(
                    years,
                    values,
                    s=22,
                    facecolor="white",
                    edgecolor=colour,
                    linewidth=0.9,
                    zorder=3,
                )
            else:
                ax.bar(
                    years,
                    values,
                    width=0.78,
                    color=colour,
                    alpha=0.58,
                    edgecolor="none",
                    zorder=2,
                )

            rolling = series.rolling(10, min_periods=10).mean()
            ax.plot(
                rolling.index,
                rolling,
                color=dark,
                linewidth=2.1,
                zorder=4,
            )
            result = sen_mann_kendall(years, values)
            ax.plot(
                years,
                result["intercept"] + result["slope"] * years,
                color=BLACK,
                linestyle="--",
                linewidth=1.25,
                zorder=3,
            )
            slope_decade = result["slope"] * 10
            sign = "+" if slope_decade >= 0 else "−"
            slope = f"{abs(slope_decade):.{decimals}f}".replace(".", ",")
            ax.text(
                0.98,
                0.96,
                f"Sen: {sign}{slope} {unit} pro Dekade; {p_text(result['p'])}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.2},
            )
            letter = panel_letters[row_index][column_index]
            ax.text(-0.09, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
            ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55, zorder=0)
            ax.margins(x=0.02)
            if variable == "temperature":
                ax.set_title(site)
                ax.set_ylabel("Saisonmittel $T$ [°C]")
            else:
                ax.set_ylabel("Saisonsumme $RR$ [mm]")
                ax.set_xlabel("Jahr")

            summary_rows.append(
                {
                    "site": site,
                    "variable": variable,
                    "period": f"{START_YEAR}-{END_YEAR}",
                    "n_years": result["n"],
                    "sen_slope_per_decade": slope_decade,
                    "unit": unit,
                    "mann_kendall_p": result["p"],
                }
            )

    legend_handles = [
        Line2D([0], [0], marker="o", color=GREY, markerfacecolor="white", markeredgecolor="#666666", linewidth=0.7, markersize=4.5, label="Jahreswert"),
        Line2D([0], [0], color="#555555", linewidth=2.1, label="Gleitendes 10-Jahres-Mittel (rückblickend)"),
        Line2D([0], [0], color=BLACK, linestyle="--", linewidth=1.25, label="Sen-Trend"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.5, w_pad=2.0)
    save_figure(fig, "fig15a_climate_trends_spartacus_copy")
    return pd.DataFrame(summary_rows)


def monthly_period_summary(
    data: pd.DataFrame, site: str, variable: str, period: tuple[int, int]
) -> pd.DataFrame:
    sub = data[(data["site"] == site) & data["year"].between(*period)]
    aggregation = "mean" if variable == "temperature" else "sum"
    year_month = sub.groupby(["year", "month"])[variable].agg(aggregation)
    grouped = year_month.groupby("month")
    return pd.DataFrame(
        {
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
        }
    ).reindex(MONTHS)


def plot_monthly_early_recent(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.7), sharex="col", dpi=180)
    panel_letters = (("a", "b"), ("c", "d"))
    x = np.arange(len(MONTHS))

    for column_index, site in enumerate(SITES):
        for row_index, variable in enumerate(("temperature", "precipitation")):
            ax = axes[row_index, column_index]
            early = monthly_period_summary(data, site, variable, EARLY)
            recent = monthly_period_summary(data, site, variable, RECENT)

            if variable == "temperature":
                for summary, colour, label in (
                    (early, BLUE, f"{EARLY[0]}–{EARLY[1]}"),
                    (recent, RED, f"{RECENT[0]}–{RECENT[1]}"),
                ):
                    ax.fill_between(x, summary["q25"], summary["q75"], color=colour, alpha=0.15, linewidth=0)
                    ax.plot(x, summary["median"], color=colour, linewidth=1.7, marker="o", markersize=3.6, label=label)
                ax.set_ylabel("Monatsmittel $T$ [°C]")
                ax.set_title(site)
            else:
                width = 0.34
                for offset, summary, colour, label in (
                    (-width / 2, early, BLUE, f"{EARLY[0]}–{EARLY[1]}"),
                    (width / 2, recent, RED, f"{RECENT[0]}–{RECENT[1]}"),
                ):
                    lower = summary["median"] - summary["q25"]
                    upper = summary["q75"] - summary["median"]
                    ax.bar(x + offset, summary["median"], width=width, color=colour, alpha=0.78, edgecolor="none", label=label)
                    ax.errorbar(x + offset, summary["median"], yerr=np.vstack([lower, upper]), fmt="none", ecolor="#4A4A4A", elinewidth=0.75, capsize=2.0)
                ax.set_ylabel("Monatssumme $RR$ [mm]")
                ax.set_xlabel("Monat")

            letter = panel_letters[row_index][column_index]
            ax.text(-0.09, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
            ax.set_xticks(x, MONTH_NAMES)
            ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55)
            ax.margins(x=0.04)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.5, w_pad=2.0)
    save_figure(fig, "fig15b_climate_monthly_early_recent_copy")


def main() -> None:
    set_style()
    data = load_spartacus()
    statistics = plot_annual_climate_trends(data)
    plot_monthly_early_recent(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(OUT_DIR / "fig15_climate_trend_statistics_copy.csv", index=False)
    print(statistics.to_string(index=False))
    print(f"\nFigures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
