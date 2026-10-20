"""Weather-type-adjusted discharge sensitivity analysis using corrected dataset vs13.

This script creates new result copies and leaves the exploratory notebook and
its figures unchanged. Daily discharge is converted to specific discharge and
deseasonalised. The regression controls for antecedent discharge, long-term
change, month and the nine BMU weather types. Temperature and precipitation
sensitivities are estimated both overall and within each weather type.

Uncertainty is quantified with a year-block bootstrap: complete melt seasons
are resampled, preserving serial dependence within years. Coefficients shown in
the figures are standardised partial regression coefficients and are therefore
directly comparable between predictors and catchments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "all_final" / "ssim_q_weather_time_vs13.xlsx"
DATA_SHEET = "clean_complete_cases"
OUT_DIR = SCRIPT_DIR / "figures" / "corrected_vs13" / "variants"

FIG_MONTHLY_PNG = OUT_DIR / "fig17a_weather_adjusted_monthly_sensitivity_copy.png"
FIG_MONTHLY_PDF = OUT_DIR / "fig17a_weather_adjusted_monthly_sensitivity_copy.pdf"
FIG_HEATMAP_PNG = OUT_DIR / "fig17b_weatherclass_specific_sensitivity_copy.png"
FIG_HEATMAP_PDF = OUT_DIR / "fig17b_weatherclass_specific_sensitivity_copy.pdf"
FIG_EFFECT_PNG = OUT_DIR / "fig17c_adjusted_weatherclass_effect_copy.png"
FIG_EFFECT_PDF = OUT_DIR / "fig17c_adjusted_weatherclass_effect_copy.pdf"
RESULTS_FILE = OUT_DIR / "fig17_weather_adjusted_sensitivity_statistics_copy.csv"

MONTHS = [5, 6, 7, 8, 9]
MONTH_NAMES = {5: "Mai", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
BMU_VALUES = list(range(9))
N_BOOTSTRAP = 500
RANDOM_SEED = 20260809

PURPLE = "#7953A6"
PURPLE_DARK = "#503377"
GREEN = "#8FBC65"
GREEN_DARK = "#517A35"
BLACK = "#262626"
GREY = "#777777"
GRID = "#D9D9D9"


@dataclass(frozen=True)
class Site:
    name: str
    q_column: str
    temperature_column: str
    precipitation_column: str
    area_km2: float
    colour: str
    dark: str


SITES = [
    Site(
        name="Vernagtbach",
        q_column="Q_Vernagt",
        temperature_column="Tmean_SPARTACUS_Vernagtkessel",
        precipitation_column="RR_SPARTACUS_Vernagtkessel",
        area_km2=11.57,
        colour=PURPLE,
        dark=PURPLE_DARK,
    ),
    Site(
        name="Rofenache – Pegel Vent",
        q_column="Q_Rofenache_Vent",
        temperature_column="Tmean_SPARTACUS_Vent",
        precipitation_column="RR_SPARTACUS_Vent",
        area_km2=97.83,
        colour=GREEN,
        dark=GREEN_DARK,
    ),
]


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


def smooth_climatology(values: pd.Series, season_day: pd.Series, window: int = 15) -> pd.Series:
    maximum_day = int(season_day.max())
    climatology = values.groupby(season_day).mean().reindex(range(maximum_day + 1)).interpolate(limit_direction="both")
    climatology = climatology.rolling(window, center=True, min_periods=1).mean()
    return season_day.map(climatology)


def zscore_by_month(values: pd.Series, month: pd.Series) -> pd.Series:
    means = values.groupby(month).transform("mean")
    scales = values.groupby(month).transform("std")
    return (values - means) / scales.replace(0, np.nan)


def prepare_site(raw: pd.DataFrame, site: Site) -> pd.DataFrame:
    columns = ["time", "year", "month", "doy", "bmu", site.q_column, site.temperature_column, site.precipitation_column]
    data = raw[columns].copy()
    data.columns = ["time", "year", "month", "doy", "bmu", "q", "temperature", "precipitation"]
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    for column in ["year", "month", "doy", "bmu", "q", "temperature", "precipitation"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[data["month"].isin(MONTHS)].sort_values("time").copy()
    data["season_day"] = (
        data["time"]
        - pd.to_datetime(data["year"].astype("Int64").astype(str) + "-05-01", errors="coerce")
    ).dt.days

    # Q [m3 s-1] -> specific discharge [mm d-1].
    data["q_specific"] = data["q"] * 86.4 / site.area_km2
    data["q_climatology"] = smooth_climatology(data["q_specific"], data["season_day"])
    data["temperature_climatology"] = smooth_climatology(data["temperature"], data["season_day"])
    data["q_anomaly"] = data["q_specific"] - data["q_climatology"]
    data["temperature_anomaly"] = data["temperature"] - data["temperature_climatology"]

    # A one-day temperature lag follows the observed maximum in the exploratory
    # lag diagnostic. Precipitation is accumulated over the current and two
    # preceding days and log-transformed to limit leverage from extremes.
    data["temperature_lag1"] = data.groupby("year")["temperature_anomaly"].shift(1)
    data["q_lag1"] = data.groupby("year")["q_anomaly"].shift(1)
    data["precipitation_3d"] = (
        data.groupby("year")["precipitation"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    data["precipitation_log3d"] = np.log1p(data["precipitation_3d"].clip(lower=0))

    data["q_z"] = zscore_by_month(data["q_anomaly"], data["month"])
    data["temperature_z"] = zscore_by_month(data["temperature_lag1"], data["month"])
    data["precipitation_z"] = zscore_by_month(data["precipitation_log3d"], data["month"])
    data["q_lag1_z"] = zscore_by_month(data["q_lag1"], data["month"])
    data["year_z"] = (data["year"] - data["year"].mean()) / data["year"].std()
    data["bmu"] = data["bmu"].astype("Int64")
    data["site"] = site.name
    return data.dropna(
        subset=["q_z", "temperature_z", "precipitation_z", "q_lag1_z", "year_z", "bmu"]
    ).copy()


def least_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def year_block_bootstrap(
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    extractor,
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    beta = least_squares(x, y)
    estimate = np.asarray(extractor(beta), dtype=float)
    fitted = x @ beta
    denominator = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - np.sum((y - fitted) ** 2) / denominator if denominator > 0 else np.nan

    years = np.sort(frame["year"].unique())
    year_indices = [np.flatnonzero(frame["year"].to_numpy() == year) for year in years]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(year_indices), size=len(year_indices))
        indices = np.concatenate([year_indices[index] for index in sampled])
        try:
            beta_boot = least_squares(x[indices], y[indices])
            boot.append(np.asarray(extractor(beta_boot), dtype=float))
        except np.linalg.LinAlgError:
            continue
    boot_array = np.asarray(boot)
    lower = np.nanpercentile(boot_array, 2.5, axis=0)
    upper = np.nanpercentile(boot_array, 97.5, axis=0)
    return estimate, lower, upper, r_squared


def overall_month_design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    class_dummies = pd.get_dummies(frame["bmu"].astype(int), prefix="bmu", dtype=float)
    for bmu in BMU_VALUES[1:]:
        column = f"bmu_{bmu}"
        if column not in class_dummies:
            class_dummies[column] = 0.0
    x = np.column_stack(
        [
            np.ones(len(frame)),
            frame["temperature_z"],
            frame["precipitation_z"],
            frame["q_lag1_z"],
            frame["year_z"],
            class_dummies[[f"bmu_{bmu}" for bmu in BMU_VALUES[1:]]].to_numpy(),
        ]
    )
    return x.astype(float), frame["q_z"].to_numpy(dtype=float)


def class_month_design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    classes = frame["bmu"].astype(int).to_numpy()
    one_hot = np.column_stack([(classes == bmu).astype(float) for bmu in BMU_VALUES])
    temperature_interactions = one_hot * frame["temperature_z"].to_numpy()[:, None]
    precipitation_interactions = one_hot * frame["precipitation_z"].to_numpy()[:, None]
    x = np.column_stack(
        [
            one_hot,
            temperature_interactions,
            precipitation_interactions,
            frame["q_lag1_z"],
            frame["year_z"],
        ]
    )
    return x.astype(float), frame["q_z"].to_numpy(dtype=float)


def class_effect_design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    classes = frame["bmu"].astype(int).to_numpy()
    one_hot = np.column_stack([(classes == bmu).astype(float) for bmu in BMU_VALUES])
    month_dummies = pd.get_dummies(frame["month"].astype(int), prefix="month", dtype=float)
    for month in MONTHS[1:]:
        column = f"month_{month}"
        if column not in month_dummies:
            month_dummies[column] = 0.0
    x = np.column_stack(
        [
            one_hot,
            month_dummies[[f"month_{month}" for month in MONTHS[1:]]].to_numpy(),
            frame["temperature_z"],
            frame["precipitation_z"],
            frame["q_lag1_z"],
            frame["year_z"],
        ]
    )
    return x.astype(float), frame["q_z"].to_numpy(dtype=float)


def analyse_site(data: pd.DataFrame, site: Site, site_index: int) -> pd.DataFrame:
    rows: list[dict] = []

    for month_index, month in enumerate(MONTHS):
        subset = data[data["month"] == month].reset_index(drop=True)
        x, y = overall_month_design(subset)
        estimate, lower, upper, r_squared = year_block_bootstrap(
            subset,
            x,
            y,
            extractor=lambda beta: beta[1:3],
            seed=RANDOM_SEED + site_index * 100 + month_index,
        )
        for predictor_index, predictor in enumerate(["Temperatur (t−1)", "Niederschlag (3 Tage)"]):
            rows.append(
                {
                    "analysis": "monthly_adjusted",
                    "site": site.name,
                    "month": month,
                    "month_name": MONTH_NAMES[month],
                    "bmu": np.nan,
                    "weather_type": "alle",
                    "predictor": predictor,
                    "estimate": estimate[predictor_index],
                    "ci_low": lower[predictor_index],
                    "ci_high": upper[predictor_index],
                    "n_days": len(subset),
                    "n_years": subset["year"].nunique(),
                    "r_squared": r_squared,
                }
            )

        x_class, y_class = class_month_design(subset)
        estimate, lower, upper, r_squared = year_block_bootstrap(
            subset,
            x_class,
            y_class,
            extractor=lambda beta: np.r_[beta[9:18], beta[18:27]],
            seed=RANDOM_SEED + 1000 + site_index * 100 + month_index,
        )
        for predictor_index, predictor in enumerate(["Temperatur (t−1)", "Niederschlag (3 Tage)"]):
            offset = predictor_index * 9
            for class_index, bmu in enumerate(BMU_VALUES):
                class_count = int((subset["bmu"].astype(int) == bmu).sum())
                rows.append(
                    {
                        "analysis": "weatherclass_specific",
                        "site": site.name,
                        "month": month,
                        "month_name": MONTH_NAMES[month],
                        "bmu": bmu,
                        "weather_type": f"WL {bmu + 1}",
                        "predictor": predictor,
                        "estimate": estimate[offset + class_index],
                        "ci_low": lower[offset + class_index],
                        "ci_high": upper[offset + class_index],
                        "n_days": class_count,
                        "n_years": subset.loc[subset["bmu"].astype(int) == bmu, "year"].nunique(),
                        "r_squared": r_squared,
                    }
                )

    full = data.reset_index(drop=True)
    x_effect, y_effect = class_effect_design(full)
    estimate, lower, upper, r_squared = year_block_bootstrap(
        full,
        x_effect,
        y_effect,
        extractor=lambda beta: beta[:9] - np.mean(beta[:9]),
        seed=RANDOM_SEED + 2000 + site_index,
    )
    for class_index, bmu in enumerate(BMU_VALUES):
        class_subset = full[full["bmu"].astype(int) == bmu]
        rows.append(
            {
                "analysis": "adjusted_weatherclass_effect",
                "site": site.name,
                "month": np.nan,
                "month_name": "Mai–Sep",
                "bmu": bmu,
                "weather_type": f"WL {bmu + 1}",
                "predictor": "bereinigter Wetterlageneffekt",
                "estimate": estimate[class_index],
                "ci_low": lower[class_index],
                "ci_high": upper[class_index],
                "n_days": len(class_subset),
                "n_years": class_subset["year"].nunique(),
                "r_squared": r_squared,
            }
        )
    return pd.DataFrame(rows)


def add_significance(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    result["ci_excludes_zero"] = (result["ci_low"] > 0) | (result["ci_high"] < 0)
    return result


def plot_monthly_sensitivity(result: pd.DataFrame) -> None:
    subset = result[result["analysis"] == "monthly_adjusted"]
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.1), dpi=180, sharey=True)
    x = np.arange(len(MONTHS))
    for panel_index, (axis, site) in enumerate(zip(axes, SITES)):
        site_data = subset[subset["site"] == site.name]
        for predictor, marker, linestyle, fill in [
            ("Temperatur (t−1)", "o", "-", site.dark),
            ("Niederschlag (3 Tage)", "s", "--", "white"),
        ]:
            values = site_data[site_data["predictor"] == predictor].set_index("month").reindex(MONTHS)
            estimate = values["estimate"].to_numpy(float)
            lower = values["ci_low"].to_numpy(float)
            upper = values["ci_high"].to_numpy(float)
            axis.errorbar(
                x,
                estimate,
                yerr=np.vstack([np.maximum(0, estimate - lower), np.maximum(0, upper - estimate)]),
                color=site.dark,
                marker=marker,
                markerfacecolor=fill,
                markeredgecolor=site.dark,
                markersize=5.2,
                linestyle=linestyle,
                linewidth=1.35,
                elinewidth=0.9,
                capsize=2.5,
                label=predictor,
                zorder=3,
            )
        axis.axhline(0, color=GREY, linewidth=0.8, zorder=1)
        axis.set_xticks(x, [MONTH_NAMES[month] for month in MONTHS])
        axis.set_title(site.name)
        axis.grid(axis="y", color=GRID, linewidth=0.55)
        axis.set_axisbelow(True)
        axis.text(-0.10, 1.03, chr(ord("a") + panel_index), transform=axis.transAxes, fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Bereinigter standardisierter Koeffizient β")
    handles = [
        Line2D([0], [0], color=BLACK, marker="o", markerfacecolor=BLACK, label="Temperatur am Vortag"),
        Line2D([0], [0], color=BLACK, marker="s", markerfacecolor="white", linestyle="--", label="Niederschlagssumme über 3 Tage"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.82, wspace=0.16)
    fig.savefig(FIG_MONTHLY_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_MONTHLY_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_weatherclass_heatmaps(result: pd.DataFrame) -> None:
    subset = result[result["analysis"] == "weatherclass_specific"]
    values = subset["estimate"].to_numpy(float)
    limit = float(np.nanpercentile(np.abs(values), 97.5))
    limit = max(limit, 0.15)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, axes = plt.subplots(2, 2, figsize=(10.1, 6.5), dpi=180, sharex=True, sharey=True)
    predictors = ["Temperatur (t−1)", "Niederschlag (3 Tage)"]
    image = None
    panel_index = 0
    for row_index, predictor in enumerate(predictors):
        for column_index, site in enumerate(SITES):
            axis = axes[row_index, column_index]
            panel = subset[(subset["site"] == site.name) & (subset["predictor"] == predictor)]
            pivot = panel.pivot(index="month", columns="bmu", values="estimate").reindex(index=MONTHS, columns=BMU_VALUES)
            significant = panel.pivot(index="month", columns="bmu", values="ci_excludes_zero").reindex(index=MONTHS, columns=BMU_VALUES)
            counts = panel.pivot(index="month", columns="bmu", values="n_days").reindex(index=MONTHS, columns=BMU_VALUES)
            image = axis.imshow(pivot.to_numpy(float), cmap="RdBu_r", norm=norm, aspect="auto")
            for y in range(len(MONTHS)):
                for x in range(len(BMU_VALUES)):
                    if bool(significant.iloc[y, x]):
                        axis.plot(x, y, marker="o", markersize=2.5, color=BLACK)
                    if counts.iloc[y, x] < 80:
                        axis.plot(x + 0.27, y - 0.27, marker="x", markersize=2.7, color="#666666", markeredgewidth=0.7)
            if row_index == 0:
                axis.set_title(site.name)
            if column_index == 0:
                axis.set_ylabel("Temperatur" if row_index == 0 else "Niederschlag")
            axis.set_yticks(np.arange(len(MONTHS)), [MONTH_NAMES[month] for month in MONTHS])
            axis.set_xticks(np.arange(len(BMU_VALUES)), [str(bmu + 1) for bmu in BMU_VALUES])
            axis.tick_params(length=0)
            axis.text(-0.10, 1.03, chr(ord("a") + panel_index), transform=axis.transAxes, fontweight="bold", fontsize=11)
            panel_index += 1
    for axis in axes[-1, :]:
        axis.set_xlabel("Wetterlagenklasse (BMU)")
    colorbar = fig.colorbar(image, ax=axes, orientation="vertical", fraction=0.024, pad=0.025)
    colorbar.set_label("Standardisierter partieller Koeffizient β")
    fig.text(0.5, 0.025, "● 95-%-Bootstrapintervall ohne 0     × weniger als 80 Beobachtungstage", ha="center", fontsize=7.7, color="#555555")
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.10, top=0.95, wspace=0.12, hspace=0.18)
    fig.savefig(FIG_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_HEATMAP_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_adjusted_weatherclass_effect(result: pd.DataFrame) -> None:
    subset = result[result["analysis"] == "adjusted_weatherclass_effect"]
    fig, axis = plt.subplots(figsize=(7.2, 4.7), dpi=180)
    y = np.arange(len(BMU_VALUES))[::-1]
    offsets = {SITES[0].name: 0.10, SITES[1].name: -0.10}
    markers = {SITES[0].name: "o", SITES[1].name: "s"}
    for site in SITES:
        values = subset[subset["site"] == site.name].set_index("bmu").reindex(BMU_VALUES)
        estimate = values["estimate"].to_numpy(float)
        lower = values["ci_low"].to_numpy(float)
        upper = values["ci_high"].to_numpy(float)
        axis.errorbar(
            estimate,
            y + offsets[site.name],
            xerr=np.vstack([np.maximum(0, estimate - lower), np.maximum(0, upper - estimate)]),
            color=site.dark,
            marker=markers[site.name],
            markerfacecolor=site.colour,
            markeredgecolor=site.dark,
            markersize=5.3,
            linestyle="none",
            elinewidth=1.0,
            capsize=2.4,
            label=site.name,
        )
    axis.axvline(0, color=GREY, linewidth=0.8)
    axis.set_yticks(y, [f"Wetterlage {bmu + 1}" for bmu in BMU_VALUES])
    axis.set_xlabel("Bereinigter Wetterlageneffekt auf die Q-Anomalie [Standardabweichungen]")
    axis.grid(axis="x", color=GRID, linewidth=0.55)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.98)
    fig.savefig(FIG_EFFECT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_EFFECT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(DATA_FILE, sheet_name=DATA_SHEET)
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")

    all_results = []
    diagnostics = []
    for site_index, site in enumerate(SITES):
        prepared = prepare_site(raw, site)
        diagnostics.append(
            {
                "analysis": "data_diagnostic",
                "site": site.name,
                "month": np.nan,
                "month_name": "Mai–Sep",
                "bmu": np.nan,
                "weather_type": "alle",
                "predictor": "vollständige Modellzeilen",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_days": len(prepared),
                "n_years": prepared["year"].nunique(),
                "r_squared": np.nan,
            }
        )
        all_results.append(analyse_site(prepared, site, site_index))

    result = pd.concat([*all_results, pd.DataFrame(diagnostics)], ignore_index=True)
    result = add_significance(result)
    result.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")

    plot_monthly_sensitivity(result)
    plot_weatherclass_heatmaps(result)
    plot_adjusted_weatherclass_effect(result)

    print(f"Saved: {FIG_MONTHLY_PNG}")
    print(f"Saved: {FIG_HEATMAP_PNG}")
    print(f"Saved: {FIG_EFFECT_PNG}")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
