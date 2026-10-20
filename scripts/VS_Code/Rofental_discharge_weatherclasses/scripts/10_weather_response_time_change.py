"""Descriptive and adjusted weather-response diagnostics based on corrected dataset vs13.

The script complements, rather than replaces, the adjusted sensitivity figures
from 09_weather_adjusted_sensitivity.py. It creates four new figure copies:

1. the intuitive temperature-discharge scatter plot requested for orientation,
2. the monthly lag structure of the adjusted temperature response,
3. a descriptive overview of the nine BMU weather classes, and
4. early-to-recent changes in class frequency and temperature sensitivity.

All inferential uncertainty is estimated by resampling complete years.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "all_final" / "ssim_q_weather_time_vs13.xlsx"
OUT_DIR = SCRIPT_DIR / "figures" / "corrected_vs13" / "variants"

FIG_SCATTER = OUT_DIR / "fig18a_temperature_discharge_scatter_copy"
FIG_SCATTER_COMPARISON = OUT_DIR / "fig18a2_temperature_discharge_same_vs_previous_day_copy"
FIG_SCATTER_SAME_DAY = OUT_DIR / "fig18a3_temperature_discharge_same_day_copy"
FIG_LAG = OUT_DIR / "fig18b_temperature_lag_structure_copy"
FIG_OVERVIEW = OUT_DIR / "fig18c_weatherclass_overview_copy"
FIG_CHANGE = OUT_DIR / "fig18d_weatherclass_change_copy"
FIG_CHANGE_SAME_DAY = OUT_DIR / "fig18d2_weatherclass_change_same_day_copy"
FIG_CLASS_RESPONSE_VERNAGT = OUT_DIR / "fig18e1_weatherclass_temperature_response_vernagtbach_copy"
FIG_CLASS_RESPONSE_VENT = OUT_DIR / "fig18e2_weatherclass_temperature_response_rofenache_vent_copy"
FIG_CLASS_TIME = OUT_DIR / "fig18f_weatherclass_sensitivity_over_time_copy"
FIG_DECOMPOSITION = OUT_DIR / "fig18g_weatherclass_discharge_change_decomposition_copy"
FIG_CLASS_MEAN_Q = OUT_DIR / "fig18h_weatherclass_mean_discharge_temperature_order_copy"
RESULTS_FILE = OUT_DIR / "fig18_weather_response_time_change_statistics_copy.csv"

MONTHS = [5, 6, 7, 8, 9]
MONTH_NAMES = {5: "Mai", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
BMUS = list(range(9))
LAGS = list(range(7))
EARLY = (1974, 1988)
RECENT = (2009, 2023)
N_BOOTSTRAP = 2000
RANDOM_SEED = 20260810

PURPLE = "#7953A6"
PURPLE_DARK = "#503377"
GREEN = "#8FBC65"
GREEN_DARK = "#517A35"
BLUE = "#4472C4"
RED = "#C94C4C"
GREY = "#777777"
GRID = "#D9D9D9"
BLACK = "#262626"


def load_adjusted_module():
    path = SCRIPT_DIR / "09_weather_adjusted_sensitivity.py"
    spec = importlib.util.spec_from_file_location("weather_adjusted_sensitivity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_adjusted_module()
SITES = BASE.SITES


def set_style() -> None:
    BASE.set_style()
    plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9})


def exact_shift(frame: pd.DataFrame, column: str, days: int) -> pd.Series:
    shifted = frame.groupby("year")[column].shift(days)
    shifted_time = frame.groupby("year")["time"].shift(days)
    exact = (frame["time"] - shifted_time).dt.days == days
    return shifted.where(exact)


def prepare_site(raw: pd.DataFrame, site, climatology_window: int = 7) -> pd.DataFrame:
    columns = ["time", "year", "month", "doy", "bmu", site.q_column, site.temperature_column, site.precipitation_column]
    data = raw[columns].copy()
    data.columns = ["time", "year", "month", "doy", "bmu", "q", "temperature", "precipitation"]
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    for column in ["year", "month", "doy", "bmu", "q", "temperature", "precipitation"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[data["month"].isin(MONTHS)].sort_values("time").reset_index(drop=True)
    data["season_day"] = (
        data["time"] - pd.to_datetime(data["year"].astype("Int64").astype(str) + "-05-01", errors="coerce")
    ).dt.days
    data["q_specific"] = data["q"] * 86.4 / site.area_km2
    data["q_climatology"] = BASE.smooth_climatology(
        data["q_specific"], data["season_day"], window=climatology_window
    )
    data["temperature_climatology"] = BASE.smooth_climatology(
        data["temperature"], data["season_day"], window=climatology_window
    )
    data["q_anomaly"] = data["q_specific"] - data["q_climatology"]
    data["temperature_anomaly"] = data["temperature"] - data["temperature_climatology"]
    data["q_lag1"] = exact_shift(data, "q_anomaly", 1)
    for lag in LAGS:
        data[f"temperature_lag{lag}"] = data["temperature_anomaly"] if lag == 0 else exact_shift(data, "temperature_anomaly", lag)
    data["precipitation_3d"] = (
        data.groupby("year")["precipitation"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    data["precipitation_log3d"] = np.log1p(data["precipitation_3d"].clip(lower=0))
    data["q_z"] = BASE.zscore_by_month(data["q_anomaly"], data["month"])
    data["precipitation_z"] = BASE.zscore_by_month(data["precipitation_log3d"], data["month"])
    data["q_lag1_z"] = BASE.zscore_by_month(data["q_lag1"], data["month"])
    for lag in LAGS:
        data[f"temperature_lag{lag}_z"] = BASE.zscore_by_month(data[f"temperature_lag{lag}"], data["month"])
    data["year_z"] = (data["year"] - data["year"].mean()) / data["year"].std()
    data["bmu"] = data["bmu"].astype("Int64")
    data["site"] = site.name
    return data


def adjusted_design(frame: pd.DataFrame, temperature_column: str, include_bmu: bool = True) -> tuple[np.ndarray, np.ndarray]:
    blocks = [
        np.ones((len(frame), 1)),
        frame[[temperature_column, "precipitation_z", "q_lag1_z", "year_z"]].to_numpy(float),
    ]
    month_dummies = pd.get_dummies(frame["month"].astype(int), prefix="month", dtype=float)
    for month in MONTHS[1:]:
        if f"month_{month}" not in month_dummies:
            month_dummies[f"month_{month}"] = 0.0
    blocks.append(month_dummies[[f"month_{month}" for month in MONTHS[1:]]].to_numpy(float))
    if include_bmu:
        class_dummies = pd.get_dummies(frame["bmu"].astype(int), prefix="bmu", dtype=float)
        for bmu in BMUS[1:]:
            if f"bmu_{bmu}" not in class_dummies:
                class_dummies[f"bmu_{bmu}"] = 0.0
        blocks.append(class_dummies[[f"bmu_{bmu}" for bmu in BMUS[1:]]].to_numpy(float))
    return np.column_stack(blocks), frame["q_z"].to_numpy(float)


def bootstrap_beta(frame: pd.DataFrame, x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float, float, np.ndarray]:
    estimate = float(np.linalg.lstsq(x, y, rcond=None)[0][1])
    years = np.sort(frame["year"].unique())
    indices_by_year = [np.flatnonzero(frame["year"].to_numpy() == year) for year in years]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(years), size=len(years))
        indices = np.concatenate([indices_by_year[index] for index in sampled])
        boot.append(float(np.linalg.lstsq(x[indices], y[indices], rcond=None)[0][1]))
    boot_array = np.asarray(boot)
    low, high = np.percentile(boot_array, [2.5, 97.5])
    return estimate, float(low), float(high), boot_array


def bootstrap_year_mean(frame: pd.DataFrame, value: str, seed: int) -> tuple[float, float, float]:
    annual = frame.groupby("year")[value].mean().dropna()
    estimate = float(annual.mean())
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(annual.to_numpy(), len(annual), replace=True).mean() for _ in range(N_BOOTSTRAP)])
    low, high = np.percentile(boot, [2.5, 97.5])
    return estimate, float(low), float(high)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def scatter_figure(
    prepared: dict[str, pd.DataFrame],
    *,
    lag: int = 1,
    output: Path = FIG_SCATTER,
) -> list[dict]:
    set_style()
    fig, axes = plt.subplots(2, 5, figsize=(12.2, 5.8), dpi=180, sharex=True)
    year_norm = plt.Normalize(1974, 2023)
    cmap = plt.get_cmap("coolwarm")
    rows = []
    for row, site in enumerate(SITES):
        data = prepared[site.name]
        for column, month in enumerate(MONTHS):
            axis = axes[row, column]
            temperature_column = f"temperature_lag{lag}"
            subset = data[data["month"] == month].dropna(subset=[temperature_column, "q_anomaly", "year"])
            x = subset[temperature_column].to_numpy(float)
            y = subset["q_anomaly"].to_numpy(float)
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
            axis.scatter(x, y, c=subset["year"], cmap=cmap, norm=year_norm, s=6, alpha=0.35, linewidths=0)
            axis.plot(xx, intercept + slope * xx, color=BLACK, linewidth=1.25)
            axis.axhline(0, color=GREY, linewidth=0.65)
            axis.axvline(0, color=GREY, linewidth=0.65)
            axis.text(0.03, 0.94, f"b = {slope:+.2f}", transform=axis.transAxes, ha="left", va="top", fontsize=7.5)
            if row == 0:
                axis.set_title(MONTH_NAMES[month])
            if row == 1:
                temperature_label = "am selben Tag" if lag == 0 else "am Vortag"
                axis.set_xlabel(f"T-Anomalie {temperature_label} [°C]")
            if column == 0:
                axis.set_ylabel(f"{site.name}\nQ-Anomalie [mm d$^{{-1}}$]")
            axis.grid(color=GRID, linewidth=0.45)
            rows.append({"analysis": "descriptive_scatter", "site": site.name, "month": month, "bmu": np.nan, "lag": lag, "period": "1974–2023", "metric": "raw_slope_mm_d_per_degC", "estimate": slope, "ci_low": np.nan, "ci_high": np.nan, "n_days": len(subset)})
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=year_norm, cmap=cmap), ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("Jahr")
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.11, top=0.95, wspace=0.19, hspace=0.26)
    save_figure(fig, output)
    return rows


def binned_response(frame: pd.DataFrame, temperature_column: str, bins: np.ndarray, min_count: int = 20) -> pd.DataFrame:
    subset = frame.dropna(subset=[temperature_column, "q_anomaly"]).copy()
    subset["temperature_bin"] = pd.cut(subset[temperature_column], bins=bins, include_lowest=True)
    grouped = subset.groupby("temperature_bin", observed=True).agg(
        x=(temperature_column, "median"),
        q_median=("q_anomaly", "median"),
        q_low=("q_anomaly", lambda values: values.quantile(0.25)),
        q_high=("q_anomaly", lambda values: values.quantile(0.75)),
        n=("q_anomaly", "size"),
    ).reset_index(drop=True)
    return grouped[grouped["n"] >= min_count]


def clearer_lag_comparison_scatter(
    prepared: dict[str, pd.DataFrame],
    output: Path = FIG_SCATTER_COMPARISON,
) -> list[dict]:
    """Keep the intuitive scatter, but compare day 0 and day -1 explicitly."""
    set_style()
    fig, axes = plt.subplots(2, 5, figsize=(12.2, 5.8), dpi=180, sharex=True)
    bins = np.arange(-10, 10.01, 1.5)
    rows = []

    row_limits = {}
    for site in SITES:
        values = prepared[site.name]["q_anomaly"].dropna().to_numpy(float)
        lower, upper = np.nanpercentile(values, [0.5, 99.5])
        padding = 0.05 * (upper - lower)
        row_limits[site.name] = (lower - padding, upper + padding)

    for row, site in enumerate(SITES):
        data = prepared[site.name]
        for column, month in enumerate(MONTHS):
            axis = axes[row, column]
            subset = data[data["month"] == month].copy()
            same = subset.dropna(subset=["temperature_lag0", "q_anomaly"])
            axis.scatter(
                same["temperature_lag0"],
                same["q_anomaly"],
                s=5,
                color="#8F8F8F",
                alpha=0.16,
                linewidths=0,
                rasterized=True,
            )

            for lag, marker, linestyle, label in [
                (0, "o", "-", "Temperatur am selben Tag"),
                (1, "s", "--", "Temperatur am Vortag"),
            ]:
                temperature_column = f"temperature_lag{lag}"
                response = binned_response(subset, temperature_column, bins)
                axis.errorbar(
                    response["x"],
                    response["q_median"],
                    yerr=np.vstack(
                        [response["q_median"] - response["q_low"], response["q_high"] - response["q_median"]]
                    ),
                    color=site.dark,
                    marker=marker,
                    markerfacecolor=site.dark if lag == 0 else "white",
                    markeredgecolor=site.dark,
                    linestyle=linestyle,
                    linewidth=1.15,
                    elinewidth=0.7,
                    capsize=1.8,
                    markersize=3.8,
                    label=label,
                    zorder=3,
                )
                regression = subset.dropna(subset=[temperature_column, "q_anomaly"])
                slope, _ = np.polyfit(regression[temperature_column], regression["q_anomaly"], 1)
                rows.append(
                    {
                        "analysis": "descriptive_same_previous_day",
                        "site": site.name,
                        "month": month,
                        "bmu": np.nan,
                        "lag": lag,
                        "period": "1974–2023",
                        "metric": "raw_slope_mm_d_per_degC",
                        "estimate": slope,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_days": len(regression),
                    }
                )

            slopes = [item["estimate"] for item in rows if item["site"] == site.name and item["month"] == month and item["analysis"] == "descriptive_same_previous_day"]
            axis.text(0.03, 0.95, f"b(t) = {slopes[0]:+.2f}\nb(t-1) = {slopes[1]:+.2f}", transform=axis.transAxes, ha="left", va="top", fontsize=7.4)
            axis.axhline(0, color=GREY, linewidth=0.65)
            axis.axvline(0, color=GREY, linewidth=0.65)
            axis.set_ylim(*row_limits[site.name])
            axis.set_xlim(-10, 10)
            if row == 0:
                axis.set_title(MONTH_NAMES[month])
            if row == 1:
                axis.set_xlabel("T-Anomalie [°C]")
            if column == 0:
                axis.set_ylabel(f"{site.name}\nQ-Anomalie [mm d$^{{-1}}$]")
            axis.grid(color=GRID, linewidth=0.45)

    handles = [
        Line2D([0], [0], color=BLACK, marker="o", markerfacecolor=BLACK, linestyle="-", label="Temperatur am selben Tag"),
        Line2D([0], [0], color=BLACK, marker="s", markerfacecolor="white", linestyle="--", label="Temperatur am Vortag"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.88, wspace=0.19, hspace=0.25)
    save_figure(fig, output)
    return rows


def lag_analysis(prepared: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    for site_index, site in enumerate(SITES):
        data = prepared[site.name]
        for month_index, month in enumerate(MONTHS):
            for lag in LAGS:
                temp_col = f"temperature_lag{lag}_z"
                subset = data[data["month"] == month].dropna(subset=["q_z", temp_col, "precipitation_z", "q_lag1_z", "year_z", "bmu"]).reset_index(drop=True)
                x, y = adjusted_design(subset, temp_col, include_bmu=True)
                estimate, low, high, _ = bootstrap_beta(subset, x, y, RANDOM_SEED + site_index * 1000 + month_index * 20 + lag)
                rows.append({"analysis": "adjusted_temperature_lag", "site": site.name, "month": month, "bmu": np.nan, "lag": lag, "period": "1974–2023", "metric": "standardised_beta", "estimate": estimate, "ci_low": low, "ci_high": high, "n_days": len(subset)})
    result = pd.DataFrame(rows)

    set_style()
    limit = max(0.15, float(np.nanpercentile(np.abs(result["estimate"]), 98)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), dpi=180, sharex=True, sharey=True)
    image = None
    for panel, (axis, site) in enumerate(zip(axes, SITES)):
        subset = result[result["site"] == site.name]
        pivot = subset.pivot(index="month", columns="lag", values="estimate").reindex(index=MONTHS, columns=LAGS)
        image = axis.imshow(pivot.to_numpy(float), cmap="RdBu_r", norm=norm, aspect="auto")
        for y_index, month in enumerate(MONTHS):
            row = subset[subset["month"] == month].set_index("lag").reindex(LAGS)
            maximum_lag = int(row["estimate"].idxmax())
            for x_index, lag in enumerate(LAGS):
                significant = row.loc[lag, "ci_low"] > 0 or row.loc[lag, "ci_high"] < 0
                if significant:
                    axis.plot(x_index, y_index, marker="o", color=BLACK, markersize=2.2)
            axis.plot(maximum_lag, y_index, marker="o", markerfacecolor="none", markeredgecolor=BLACK, markeredgewidth=1.0, markersize=9)
        axis.set_title(site.name)
        axis.set_xticks(np.arange(len(LAGS)), [str(lag) for lag in LAGS])
        axis.set_yticks(np.arange(len(MONTHS)), [MONTH_NAMES[month] for month in MONTHS])
        axis.set_xlabel("Zeitversatz der Temperatur [Tage]")
        axis.tick_params(length=0)
        axis.text(-0.10, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Monat")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.025)
    colorbar.set_label("Bereinigter standardisierter Koeffizient β")
    fig.text(0.5, 0.015, "● 95-%-Bootstrapintervall ohne 0     ○ stärkster positiver Zusammenhang je Monat", ha="center", fontsize=7.5, color="#555555")
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.17, top=0.91, wspace=0.14)
    save_figure(fig, FIG_LAG)
    return result, rows


def overview_figure(
    raw: pd.DataFrame,
    prepared: dict[str, pd.DataFrame],
    output: Path = FIG_OVERVIEW,
    show_grid: bool = False,
    class_colours: np.ndarray | None = None,
    class_annotations: list[str] | None = None,
) -> list[dict]:
    rows = []
    valid = raw[raw["month"].isin(MONTHS) & raw["bmu"].notna()].copy()
    frequencies = valid["bmu"].astype(int).value_counts(normalize=True).reindex(BMUS) * 100

    summaries: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for site_index, site in enumerate(SITES):
        data = prepared[site.name]
        for metric, column in [("T-Anomalie", "temperature_anomaly"), ("3-Tage-Niederschlag", "precipitation_3d"), ("Q-Anomalie", "q_anomaly")]:
            values = []
            for bmu in BMUS:
                subset = data[data["bmu"].astype("Int64") == bmu].dropna(subset=[column])
                estimate, low, high = bootstrap_year_mean(subset, column, RANDOM_SEED + 5000 + site_index * 100 + bmu)
                values.append((estimate, low, high))
                rows.append({"analysis": "weatherclass_overview", "site": site.name, "month": np.nan, "bmu": bmu, "lag": np.nan, "period": "1974–2023", "metric": metric, "estimate": estimate, "ci_low": low, "ci_high": high, "n_days": len(subset)})
            summaries[(site.name, metric)] = values

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), dpi=180, sharex=True)
    x = np.arange(1, 10)
    colours = class_colours if class_colours is not None else "#A9A9A9"
    bars = axes[0, 0].bar(x, frequencies.to_numpy(float), color=colours, width=0.68)
    axes[0, 0].set_ylabel("Anteil der Saisontage [%]")
    if class_annotations is not None:
        for bar, annotation in zip(bars, class_annotations):
            red, green, blue, _ = bar.get_facecolor()
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                annotation,
                ha="center", va="center", rotation=90,
                fontsize=6.7, color="#333333" if luminance > 0.58 else "white", fontweight="bold",
            )

    panels = [
        (axes[0, 1], "T-Anomalie", "Mittlere T-Anomalie [°C]"),
        (axes[1, 0], "3-Tage-Niederschlag", "Mittlerer 3-Tage-Niederschlag [mm]"),
        (axes[1, 1], "Q-Anomalie", "Mittlere Q-Anomalie [mm d$^{-1}$]"),
    ]
    for axis, metric, ylabel in panels:
        for site in SITES:
            values = np.asarray(summaries[(site.name, metric)])
            axis.errorbar(x, values[:, 0], yerr=np.vstack([values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]]), color=site.dark, marker="o" if site == SITES[0] else "s", markerfacecolor=site.colour, linestyle="none", linewidth=1.15, markersize=4.4, capsize=2, label=site.name)
        axis.set_ylabel(ylabel)
        if metric in {"T-Anomalie", "Q-Anomalie"}:
            axis.axhline(0, color=GREY, linewidth=0.7)
    for index, axis in enumerate(axes.ravel()):
        if show_grid:
            axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.text(-0.10, 1.03, chr(ord("a") + index), transform=axis.transAxes, fontweight="bold", fontsize=11)
        axis.set_xticks(x)
    for axis in axes[1, :]:
        axis.set_xlabel("Wetterlagenklasse (BMU)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.89, wspace=0.23, hspace=0.25)
    save_figure(fig, output)
    return rows


def mean_discharge_temperature_order_figure(
    prepared: dict[str, pd.DataFrame],
    output: Path = FIG_CLASS_MEAN_Q,
) -> list[dict]:
    """Show mean raw discharge by class on one shared cold-to-warm class order."""
    temperature_by_site = []
    for site in SITES:
        temperature_by_site.append(
            prepared[site.name]
            .groupby("bmu")["temperature_anomaly"]
            .mean()
            .reindex(BMUS)
            .to_numpy(float)
        )
    mean_temperature = np.nanmean(np.vstack(temperature_by_site), axis=0)
    order = np.argsort(mean_temperature)
    normalise = plt.Normalize(float(np.nanmin(mean_temperature)), float(np.nanmax(mean_temperature)))
    colours = plt.get_cmap("coolwarm")(normalise(mean_temperature[order]))

    rows = []
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), dpi=180)
    x = np.arange(len(BMUS))
    for panel, (axis, site) in enumerate(zip(axes, SITES)):
        data = prepared[site.name]
        class_means = data.groupby("bmu")["q"].mean().reindex(BMUS).to_numpy(float)
        axis.bar(x, class_means[order], color=colours, width=0.66)
        axis.set_title(site.name)
        axis.set_xticks(x, [str(int(bmu) + 1) for bmu in order])
        axis.set_xlabel("SOM-Klasse (nach Temperatur kalt–warm geordnet)")
        axis.set_ylabel("Mittleres Q [m$^3$ s$^{-1}$]")
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.text(-0.10, 1.04, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold", fontsize=11)
        for rank, bmu in enumerate(order):
            rows.append(
                {
                    "analysis": "weatherclass_mean_discharge_temperature_order",
                    "site": site.name,
                    "month": np.nan,
                    "bmu": int(bmu),
                    "temperature_rank": int(rank + 1),
                    "lag": np.nan,
                    "period": "1974–2023",
                    "metric": "mean_discharge_m3_s",
                    "estimate": class_means[bmu],
                    "mean_temperature_anomaly_c": mean_temperature[bmu],
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_days": int(data.loc[data["bmu"].astype("Int64") == bmu, "q"].notna().sum()),
                }
            )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.88, wspace=0.22)
    save_figure(fig, output)
    return rows


def frequency_difference(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = raw[raw["month"].isin(MONTHS) & raw["bmu"].notna()].copy()
    data["bmu"] = data["bmu"].astype(int)
    period_frames = [data[data["year"].between(*EARLY)], data[data["year"].between(*RECENT)]]
    estimates = []
    bootstraps = []
    for period_index, frame in enumerate(period_frames):
        years = np.sort(frame["year"].unique())
        estimates.append(frame["bmu"].value_counts(normalize=True).reindex(BMUS, fill_value=0).to_numpy(float) * 100)
        rng = np.random.default_rng(RANDOM_SEED + 7000 + period_index)
        samples = []
        for _ in range(N_BOOTSTRAP):
            sampled_years = rng.choice(years, len(years), replace=True)
            sample = pd.concat([frame[frame["year"] == year] for year in sampled_years], ignore_index=True)
            samples.append(sample["bmu"].value_counts(normalize=True).reindex(BMUS, fill_value=0).to_numpy(float) * 100)
        bootstraps.append(np.asarray(samples))
    difference = estimates[1] - estimates[0]
    boot_difference = bootstraps[1] - bootstraps[0]
    low, high = np.percentile(boot_difference, [2.5, 97.5], axis=0)
    return estimates[0], estimates[1], difference, low, high


def change_figure(
    raw: pd.DataFrame,
    prepared: dict[str, pd.DataFrame],
    temperature_column: str = "temperature_lag1_z",
    lag: int = 1,
    output: Path = FIG_CHANGE,
    analysis_suffix: str = "",
) -> list[dict]:
    rows = []
    freq_early, freq_recent, freq_diff, freq_low, freq_high = frequency_difference(raw)
    if not analysis_suffix:
        for bmu in BMUS:
            rows.append({"analysis": "weatherclass_frequency_period", "site": "beide", "month": np.nan, "bmu": bmu, "lag": np.nan, "period": "1974–1988", "metric": "frequency_percent", "estimate": freq_early[bmu], "ci_low": np.nan, "ci_high": np.nan, "n_days": np.nan})
            rows.append({"analysis": "weatherclass_frequency_period", "site": "beide", "month": np.nan, "bmu": bmu, "lag": np.nan, "period": "2009–2023", "metric": "frequency_percent", "estimate": freq_recent[bmu], "ci_low": np.nan, "ci_high": np.nan, "n_days": np.nan})
            rows.append({"analysis": "weatherclass_frequency_change", "site": "beide", "month": np.nan, "bmu": bmu, "lag": np.nan, "period": "2009–2023 minus 1974–1988", "metric": "percentage_point_change", "estimate": freq_diff[bmu], "ci_low": freq_low[bmu], "ci_high": freq_high[bmu], "n_days": np.nan})

    sensitivity: dict[str, list[dict[str, float]]] = {}
    for site_index, site in enumerate(SITES):
        site_rows = []
        data = prepared[site.name]
        for bmu in BMUS:
            period_results = []
            for period_index, period in enumerate([EARLY, RECENT]):
                subset = data[data["year"].between(*period) & (data["bmu"].astype("Int64") == bmu)].dropna(subset=["q_z", temperature_column, "precipitation_z", "q_lag1_z", "year_z", "month"]).reset_index(drop=True)
                x, y = adjusted_design(subset, temperature_column, include_bmu=False)
                period_results.append(bootstrap_beta(subset, x, y, RANDOM_SEED + 8000 + site_index * 500 + bmu * 20 + period_index))
            early_estimate, _, _, early_boot = period_results[0]
            recent_estimate, _, _, recent_boot = period_results[1]
            early_low, early_high = period_results[0][1:3]
            recent_low, recent_high = period_results[1][1:3]
            delta = recent_estimate - early_estimate
            delta_boot = recent_boot - early_boot
            low, high = np.percentile(delta_boot, [2.5, 97.5])
            site_rows.append(
                {
                    "early": early_estimate,
                    "early_low": early_low,
                    "early_high": early_high,
                    "recent": recent_estimate,
                    "recent_low": recent_low,
                    "recent_high": recent_high,
                    "delta": delta,
                    "delta_low": float(low),
                    "delta_high": float(high),
                }
            )
            rows.append({"analysis": f"weatherclass_temperature_sensitivity_period{analysis_suffix}", "site": site.name, "month": np.nan, "bmu": bmu, "lag": lag, "period": "1974–1988", "metric": "standardised_beta", "estimate": early_estimate, "ci_low": early_low, "ci_high": early_high, "n_days": np.nan})
            rows.append({"analysis": f"weatherclass_temperature_sensitivity_period{analysis_suffix}", "site": site.name, "month": np.nan, "bmu": bmu, "lag": lag, "period": "2009–2023", "metric": "standardised_beta", "estimate": recent_estimate, "ci_low": recent_low, "ci_high": recent_high, "n_days": np.nan})
            rows.append({"analysis": f"weatherclass_temperature_sensitivity_change{analysis_suffix}", "site": site.name, "month": np.nan, "bmu": bmu, "lag": lag, "period": "2009–2023 minus 1974–1988", "metric": "delta_standardised_beta", "estimate": delta, "ci_low": low, "ci_high": high, "n_days": np.nan})
        sensitivity[site.name] = site_rows

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), dpi=180, sharex=True)
    x = np.arange(1, 10)
    width = 0.34
    axes[0].bar(x - width / 2, freq_early, color=BLUE, width=width)
    axes[0].bar(x + width / 2, freq_recent, color=RED, width=width)
    for index in range(len(BMUS)):
        if freq_low[index] > 0 or freq_high[index] < 0:
            axes[0].text(x[index], max(freq_early[index], freq_recent[index]) + 0.45, "*", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("Anteil der Saisontage [%]")
    axes[0].set_title("Auftretenshäufigkeit")

    for panel, (axis, site) in enumerate(zip(axes[1:], SITES), start=1):
        values = sensitivity[site.name]
        early = np.asarray([value["early"] for value in values])
        early_low = np.asarray([value["early_low"] for value in values])
        early_high = np.asarray([value["early_high"] for value in values])
        recent = np.asarray([value["recent"] for value in values])
        recent_low = np.asarray([value["recent_low"] for value in values])
        recent_high = np.asarray([value["recent_high"] for value in values])
        axis.errorbar(x - 0.10, early, yerr=np.vstack([early - early_low, early_high - early]), color=BLUE, marker="o", markerfacecolor="white", linestyle="none", linewidth=1.0, capsize=2.0)
        axis.errorbar(x + 0.10, recent, yerr=np.vstack([recent - recent_low, recent_high - recent]), color=RED, marker="s", markerfacecolor=RED, linestyle="none", linewidth=1.0, capsize=2.0)
        for index, value in enumerate(values):
            if value["delta_low"] > 0 or value["delta_high"] < 0:
                axis.text(x[index], max(value["early_high"], value["recent_high"]) + 0.025, "*", ha="center", va="bottom", fontsize=10)
        axis.set_title(site.name)
        temperature_label = "T(t)" if lag == 0 else "T(t−1)"
        axis.set_ylabel(f"Bereinigte {temperature_label}-Sensitivität β")
    for index, axis in enumerate(axes):
        axis.axhline(0, color=GREY, linewidth=0.8)
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.set_xticks(x)
        axis.set_xlabel("Wetterlagenklasse (BMU)")
        axis.text(-0.12, 1.04, chr(ord("a") + index), transform=axis.transAxes, fontweight="bold", fontsize=11)
    legend = [
        Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white", linestyle="none", label="1974–1988"),
        Line2D([0], [0], color=RED, marker="s", markerfacecolor=RED, linestyle="none", label="2009–2023"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.text(0.5, 0.015, "* Unterschied zwischen den Perioden: 95-%-Jahresblock-Bootstrapintervall ohne 0", ha="center", fontsize=7.6, color="#555555")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.83, wspace=0.28)
    save_figure(fig, output)
    return rows


def weatherclass_period_response(
    prepared: dict[str, pd.DataFrame],
    output_vernagt: Path = FIG_CLASS_RESPONSE_VERNAGT,
    output_vent: Path = FIG_CLASS_RESPONSE_VENT,
) -> list[dict]:
    """Plot one clear early/recent temperature-response panel per BMU class."""
    rows = []
    bins = np.arange(-10, 10.01, 1.5)
    periods = [(EARLY, BLUE, "o", "1974–1988"), (RECENT, RED, "s", "2009–2023")]

    for site_index, site in enumerate(SITES):
        set_style()
        data = prepared[site.name]
        q_values = data["q_anomaly"].dropna().to_numpy(float)
        q_low, q_high = np.nanpercentile(q_values, [0.5, 99.5])
        q_padding = 0.05 * (q_high - q_low)

        fig, axes = plt.subplots(3, 3, figsize=(9.2, 7.8), dpi=180, sharex=True, sharey=True)
        for bmu, axis in zip(BMUS, axes.ravel()):
            class_data = data[data["bmu"].astype("Int64") == bmu]
            for period_index, (period, colour, marker, label) in enumerate(periods):
                subset = class_data[class_data["year"].between(*period)].dropna(subset=["temperature_lag0", "q_anomaly"])
                axis.scatter(
                    subset["temperature_lag0"],
                    subset["q_anomaly"],
                    color=colour,
                    s=6,
                    alpha=0.11,
                    linewidths=0,
                    rasterized=True,
                )
                response = binned_response(subset, "temperature_lag0", bins, min_count=10)
                axis.errorbar(
                    response["x"],
                    response["q_median"],
                    yerr=np.vstack(
                        [response["q_median"] - response["q_low"], response["q_high"] - response["q_median"]]
                    ),
                    color=colour,
                    marker=marker,
                    markerfacecolor=colour if period == RECENT else "white",
                    markeredgecolor=colour,
                    linestyle="-",
                    linewidth=1.0,
                    elinewidth=0.65,
                    capsize=1.6,
                    markersize=3.5,
                    zorder=3,
                )
                slope, _ = np.polyfit(subset["temperature_lag0"], subset["q_anomaly"], 1)
                rows.append(
                    {
                        "analysis": "descriptive_weatherclass_period_response",
                        "site": site.name,
                        "month": np.nan,
                        "bmu": bmu,
                        "lag": 0,
                        "period": label,
                        "metric": "raw_slope_mm_d_per_degC",
                        "estimate": slope,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_days": len(subset),
                    }
                )
                text_y = 0.96 - period_index * 0.085
                axis.text(
                    0.03,
                    text_y,
                    f"b = {slope:+.2f}",
                    transform=axis.transAxes,
                    color=colour,
                    ha="left",
                    va="top",
                    fontsize=7.2,
                )
            axis.axhline(0, color=GREY, linewidth=0.65)
            axis.axvline(0, color=GREY, linewidth=0.65)
            axis.set_title(f"Wetterlage {bmu + 1}")
            axis.set_xlim(-10, 10)
            axis.set_ylim(q_low - q_padding, q_high + q_padding)
            axis.grid(color=GRID, linewidth=0.45)

        for axis in axes[-1, :]:
            axis.set_xlabel("T-Anomalie am selben Tag [°C]")
        for axis in axes[:, 0]:
            axis.set_ylabel("Q-Anomalie [mm d$^{-1}$]")
        handles = [
            Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white", label="1974–1988"),
            Line2D([0], [0], color=RED, marker="s", markerfacecolor=RED, label="2009–2023"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.91, wspace=0.12, hspace=0.20)
        output = output_vernagt if site_index == 0 else output_vent
        save_figure(fig, output)
    return rows


def rolling_weatherclass_sensitivity(prepared: dict[str, pd.DataFrame]) -> list[dict]:
    """Estimate adjusted same-day temperature sensitivity in rolling 15-year windows."""
    rows = []
    window = 15
    starts = list(range(1974, 2023 - window + 2))
    for site in SITES:
        data = prepared[site.name]
        for bmu in BMUS:
            for start in starts:
                end = start + window - 1
                subset = data[
                    data["year"].between(start, end) & (data["bmu"].astype("Int64") == bmu)
                ].dropna(
                    subset=["q_z", "temperature_lag0_z", "precipitation_z", "q_lag1_z", "year_z", "month"]
                ).reset_index(drop=True)
                if len(subset) < 120 or subset["year"].nunique() < 12:
                    estimate = np.nan
                else:
                    x, y = adjusted_design(subset, "temperature_lag0_z", include_bmu=False)
                    estimate = float(np.linalg.lstsq(x, y, rcond=None)[0][1])
                rows.append(
                    {
                        "analysis": "rolling_weatherclass_temperature_sensitivity",
                        "site": site.name,
                        "month": np.nan,
                        "bmu": bmu,
                        "lag": 0,
                        "period": f"{start}–{end}",
                        "window_centre": start + window // 2,
                        "metric": "standardised_beta",
                        "estimate": estimate,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_days": len(subset),
                    }
                )

    result = pd.DataFrame(rows)
    set_style()
    finite = result["estimate"].dropna().to_numpy(float)
    limit = max(0.20, float(np.nanpercentile(np.abs(finite), 98)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    centres = sorted(result["window_centre"].dropna().astype(int).unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), dpi=180, sharex=True, sharey=True)
    image = None
    for panel, (axis, site) in enumerate(zip(axes, SITES)):
        subset = result[result["site"] == site.name]
        pivot = subset.pivot(index="bmu", columns="window_centre", values="estimate").reindex(index=BMUS, columns=centres)
        image = axis.imshow(pivot.to_numpy(float), cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        axis.set_title(site.name)
        tick_positions = np.arange(0, len(centres) - 1, 5)
        axis.set_xticks(tick_positions, [str(centres[index]) for index in tick_positions])
        tick_labels = axis.get_xticklabels()
        if tick_labels:
            tick_labels[0].set_ha("left")
            tick_labels[-1].set_ha("right")
        axis.set_yticks(np.arange(len(BMUS)), [str(bmu + 1) for bmu in BMUS])
        axis.set_xlabel("Mitteljahr des 15-Jahres-Fensters")
        axis.tick_params(length=0)
        axis.text(-0.10, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Wetterlagenklasse (BMU)")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.027, pad=0.025)
    colorbar.set_label("Bereinigte standardisierte T-Sensitivität β")
    fig.text(0.5, 0.018, "Überlappende 15-Jahres-Fenster; Monat, Dreitagesniederschlag, Vorabfluss und Zeittrend kontrolliert", ha="center", fontsize=7.5, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.16, top=0.91, wspace=0.14)
    save_figure(fig, FIG_CLASS_TIME)
    return rows


def _period_class_statistics(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Return class frequencies, class means and the overall specific discharge."""
    valid = frame.dropna(subset=["bmu", "q_specific"]).copy()
    valid["bmu"] = valid["bmu"].astype(int)
    frequencies = (
        valid["bmu"].value_counts(normalize=True).reindex(BMUS, fill_value=0).to_numpy(float)
    )
    class_means = valid.groupby("bmu")["q_specific"].mean().reindex(BMUS).to_numpy(float)
    if np.isnan(class_means).any():
        raise ValueError("At least one weather class has no valid discharge values in a period.")
    return frequencies, class_means, float(valid["q_specific"].mean())


def _symmetric_change_decomposition(
    early_frame: pd.DataFrame,
    recent_frame: pd.DataFrame,
) -> dict[str, float]:
    """Symmetrically decompose the change in mean discharge into two exact parts."""
    early_frequency, early_mean, early_total = _period_class_statistics(early_frame)
    recent_frequency, recent_mean, recent_total = _period_class_statistics(recent_frame)
    frequency_effect = 0.5 * np.sum(
        (recent_frequency - early_frequency) * (recent_mean + early_mean)
    )
    within_class_effect = 0.5 * np.sum(
        (recent_mean - early_mean) * (recent_frequency + early_frequency)
    )
    total_change = recent_total - early_total
    return {
        "frequency_effect": float(frequency_effect),
        "within_class_effect": float(within_class_effect),
        "total_change": float(total_change),
        "early_mean": float(early_total),
        "recent_mean": float(recent_total),
        "closure_error": float(total_change - frequency_effect - within_class_effect),
    }


def _resample_complete_years(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    years = np.sort(frame["year"].dropna().unique())
    sampled_years = rng.choice(years, len(years), replace=True)
    return pd.concat([frame[frame["year"] == year] for year in sampled_years], ignore_index=True)


def discharge_change_decomposition(prepared: dict[str, pd.DataFrame]) -> list[dict]:
    """Separate the early-to-recent Q change into frequency and within-class effects."""
    component_keys = ["frequency_effect", "within_class_effect", "total_change"]
    component_labels = [
        "Veränderte Häufigkeit\nder Wetterlagen",
        "Veränderung innerhalb\nder Wetterlagen",
        "Beobachtete\nGesamtänderung",
    ]
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    rows: list[dict] = []

    for site_index, site in enumerate(SITES):
        data = prepared[site.name].dropna(subset=["year", "bmu", "q_specific"]).copy()
        early = data[data["year"].between(*EARLY)]
        recent = data[data["year"].between(*RECENT)]
        estimate = _symmetric_change_decomposition(early, recent)

        rng = np.random.default_rng(RANDOM_SEED + 11000 + site_index)
        bootstrap_values = []
        for _ in range(N_BOOTSTRAP):
            early_sample = _resample_complete_years(early, rng)
            recent_sample = _resample_complete_years(recent, rng)
            bootstrap_values.append(
                _symmetric_change_decomposition(early_sample, recent_sample)
            )

        site_summary: dict[str, dict[str, float]] = {}
        for key in component_keys:
            distribution = np.asarray([sample[key] for sample in bootstrap_values], dtype=float)
            low, high = np.percentile(distribution, [2.5, 97.5])
            site_summary[key] = {
                "estimate": estimate[key],
                "low": float(low),
                "high": float(high),
            }
            rows.append(
                {
                    "analysis": "weatherclass_discharge_change_decomposition",
                    "site": site.name,
                    "month": np.nan,
                    "bmu": np.nan,
                    "lag": np.nan,
                    "period": "2009–2023 minus 1974–1988",
                    "metric": key,
                    "estimate": estimate[key],
                    "ci_low": low,
                    "ci_high": high,
                    "n_days": len(early) + len(recent),
                }
            )
        for period_label, key in [("1974–1988", "early_mean"), ("2009–2023", "recent_mean")]:
            rows.append(
                {
                    "analysis": "weatherclass_discharge_change_decomposition",
                    "site": site.name,
                    "month": np.nan,
                    "bmu": np.nan,
                    "lag": np.nan,
                    "period": period_label,
                    "metric": "mean_specific_discharge_mm_d",
                    "estimate": estimate[key],
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_days": len(early) if key == "early_mean" else len(recent),
                }
            )
        if abs(estimate["closure_error"]) > 1e-10:
            raise RuntimeError(f"Decomposition does not close for {site.name}.")
        summaries[site.name] = site_summary

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), dpi=180, sharex=True, sharey=True)
    y = np.arange(len(component_keys))[::-1]
    markers = ["o", "s", "D"]
    for panel, (axis, site) in enumerate(zip(axes, SITES)):
        summary = summaries[site.name]
        for index, (key, marker) in enumerate(zip(component_keys, markers)):
            value = summary[key]["estimate"]
            low = summary[key]["low"]
            high = summary[key]["high"]
            face = "white" if key == "frequency_effect" else site.colour
            if key == "total_change":
                face = site.dark
            axis.errorbar(
                value,
                y[index],
                xerr=np.asarray([[value - low], [high - value]]),
                color=site.dark,
                marker=marker,
                markerfacecolor=face,
                markeredgecolor=site.dark,
                markersize=5.5,
                linewidth=1.15,
                capsize=2.5,
                linestyle="none",
                zorder=3,
            )
            offset = 7 if value >= 0 else -7
            axis.annotate(
                f"{value:+.2f}",
                (value, y[index]),
                xytext=(offset, 7),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="bottom",
                fontsize=7.5,
                color=BLACK,
            )
        axis.axvline(0, color=GREY, linewidth=0.8)
        axis.set_title(site.name)
        axis.set_yticks(y, component_labels)
        axis.set_xlabel("Beitrag zur Änderung von Q [mm d$^{-1}$]")
        axis.grid(axis="x", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.text(-0.10, 1.04, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold", fontsize=11)
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.18, top=0.88, wspace=0.16)
    save_figure(fig, FIG_DECOMPOSITION)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(DATA_FILE, sheet_name="clean_complete_cases")
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    prepared = {site.name: prepare_site(raw, site) for site in SITES}

    rows = []
    rows.extend(scatter_figure(prepared))
    rows.extend(scatter_figure(prepared, lag=0, output=FIG_SCATTER_SAME_DAY))
    rows.extend(clearer_lag_comparison_scatter(prepared))
    _, lag_rows = lag_analysis(prepared)
    rows.extend(lag_rows)
    rows.extend(overview_figure(raw, prepared))
    rows.extend(mean_discharge_temperature_order_figure(prepared))
    rows.extend(change_figure(raw, prepared))
    rows.extend(
        change_figure(
            raw,
            prepared,
            temperature_column="temperature_lag0_z",
            lag=0,
            output=FIG_CHANGE_SAME_DAY,
            analysis_suffix="_same_day",
        )
    )
    rows.extend(weatherclass_period_response(prepared))
    rows.extend(rolling_weatherclass_sensitivity(prepared))
    rows.extend(discharge_change_decomposition(prepared))
    pd.DataFrame(rows).to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")

    for stem in [
        FIG_SCATTER,
        FIG_SCATTER_COMPARISON,
        FIG_SCATTER_SAME_DAY,
        FIG_LAG,
        FIG_OVERVIEW,
        FIG_CHANGE,
        FIG_CHANGE_SAME_DAY,
        FIG_CLASS_RESPONSE_VERNAGT,
        FIG_CLASS_RESPONSE_VENT,
        FIG_CLASS_TIME,
        FIG_DECOMPOSITION,
        FIG_CLASS_MEAN_Q,
    ]:
        print(f"Saved: {stem.with_suffix('.png')}")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
