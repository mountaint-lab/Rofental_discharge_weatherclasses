"""Estimate and plot the weather-type-specific temperature response of runoff.

The main inference compares catchments with standardised coefficients. This
companion retains the class-wise, frequency-weighted construction but estimates
raw specific-discharge anomalies (mm d-1) on temperature anomalies (K).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PRIMARY_SCRIPT = SCRIPT_DIR / "_primary_som_response.py"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "weather_type_response"
OUT_DIR = Path(os.environ.get("PHYSICAL_RESPONSE_OUT_DIR", str(DEFAULT_OUT_DIR)))
OUTPUT_STEM = OUT_DIR / "abb20_veraenderung_bedingte_temperaturreaktion"
PERIODS = {"1974–1988": (1974, 1988), "2009–2023": (2009, 2023)}
N_BOOTSTRAP = 2_000
RANDOM_SEED = 20260829
PURPLE = "#7953A6"
GREEN = "#8FBC65"
BLUE = "#4472C4"
RED = "#C94C4C"
GREY = "#777777"


def load_primary_module():
    spec = importlib.util.spec_from_file_location("primary_som_p0_physical", PRIMARY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {PRIMARY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def year_weights(years: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(years), size=(N_BOOTSTRAP, len(years)))
    return np.stack([np.bincount(sample, minlength=len(years)) for sample in samples])


def design(frame: pd.DataFrame, months: list[int]) -> tuple[np.ndarray, np.ndarray]:
    month_dummies = pd.get_dummies(frame["month"].astype(int), prefix="month", dtype=float)
    for month in months[1:]:
        if f"month_{month}" not in month_dummies:
            month_dummies[f"month_{month}"] = 0.0
    x = np.column_stack(
        [
            np.ones(len(frame)),
            frame[["temperature_lag0", "precipitation_log3d", "q_lag1", "year_z"]].to_numpy(float),
            month_dummies[[f"month_{month}" for month in months[1:]]].to_numpy(float),
        ]
    )
    return x.astype(float), frame["q_anomaly"].to_numpy(float)


def slope_bootstrap(
    frame: pd.DataFrame, x: np.ndarray, y: np.ndarray, years: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    point = float(np.linalg.lstsq(x, y, rcond=None)[0][1])
    xtx = np.zeros((len(years), x.shape[1], x.shape[1]))
    xty = np.zeros((len(years), x.shape[1]))
    frame_years = frame["year"].to_numpy(int)
    for index, year in enumerate(years):
        mask = frame_years == year
        if mask.any():
            xtx[index] = x[mask].T @ x[mask]
            xty[index] = x[mask].T @ y[mask]
    boot_xtx = np.einsum("by,yij->bij", weights, xtx)
    boot_xty = np.einsum("by,yj->bj", weights, xty)
    bootstrap = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        try:
            bootstrap[index] = np.linalg.solve(boot_xtx[index], boot_xty[index])[1]
        except np.linalg.LinAlgError:
            bootstrap[index] = np.linalg.lstsq(boot_xtx[index], boot_xty[index], rcond=None)[0][1]
    return point, bootstrap


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    primary = load_primary_module()
    analysis = primary.load_analysis_module()
    raw = primary.load_corrected_primary_data()
    raw["time"] = pd.to_datetime(raw["time"], errors="raise")
    prepared = {site.name: analysis.prepare_site(raw, site) for site in analysis.SITES}
    years = {name: np.arange(low, high + 1) for name, (low, high) in PERIODS.items()}
    weights = {
        "1974–1988": year_weights(years["1974–1988"], RANDOM_SEED),
        "2009–2023": year_weights(years["2009–2023"], RANDOM_SEED + 1),
    }
    class_frequency = (
        raw.loc[raw["month"].isin(analysis.MONTHS), "bmu"]
        .value_counts(normalize=True)
        .reindex(analysis.BMUS, fill_value=0)
        .to_numpy(float)
    )

    class_rows, overall_rows = [], []
    required = ["q_anomaly", "temperature_lag0", "precipitation_log3d", "q_lag1", "year_z", "month"]
    for site in analysis.SITES:
        for period, bounds in PERIODS.items():
            points = np.empty(len(analysis.BMUS))
            bootstraps = np.empty((N_BOOTSTRAP, len(analysis.BMUS)))
            for bmu in analysis.BMUS:
                subset = prepared[site.name].loc[
                    lambda data: data["year"].between(*bounds) & (data["bmu"].astype("Int64") == bmu)
                ].dropna(subset=required).reset_index(drop=True)
                x, y = design(subset, analysis.MONTHS)
                point, bootstrap = slope_bootstrap(subset, x, y, years[period], weights[period])
                points[bmu], bootstraps[:, bmu] = point, bootstrap
                class_rows.append(
                    {
                        "site": site.name, "period": period, "class": bmu + 1,
                        "slope_mm_d_per_K": point,
                        "ci_low": np.percentile(bootstrap, 2.5),
                        "ci_high": np.percentile(bootstrap, 97.5),
                    }
                )
            weighted_point = float(class_frequency @ points)
            weighted_bootstrap = bootstraps @ class_frequency
            overall_rows.append(
                {
                    "site": site.name, "period": period, "slope_mm_d_per_K": weighted_point,
                    "ci_low": np.percentile(weighted_bootstrap, 2.5),
                    "ci_high": np.percentile(weighted_bootstrap, 97.5),
                    "class_weighting": "full-period weather-class frequency",
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall = pd.DataFrame(overall_rows)
    pd.DataFrame(class_rows).to_csv(OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_class_statistics.csv"), index=False)
    overall.to_csv(OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_statistics.csv"), index=False)

    class_results = pd.DataFrame(class_rows)
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), dpi=180)
    x = np.arange(1, 10)
    for panel, (axis, site) in enumerate(zip(axes, analysis.SITES)):
        site_rows = class_results.loc[class_results["site"].eq(site.name)]
        early = site_rows.loc[site_rows["period"].eq("1974–1988")].sort_values("class")
        recent = site_rows.loc[site_rows["period"].eq("2009–2023")].sort_values("class")
        axis.errorbar(
            x - 0.10, early["slope_mm_d_per_K"],
            yerr=np.vstack([
                early["slope_mm_d_per_K"] - early["ci_low"],
                early["ci_high"] - early["slope_mm_d_per_K"],
            ]),
            color=BLUE, marker="o", markerfacecolor="white", markeredgecolor=BLUE,
            linestyle="none", linewidth=1.0, capsize=2.2, markersize=5.2,
            label="1974–1988", zorder=3,
        )
        axis.errorbar(
            x + 0.10, recent["slope_mm_d_per_K"],
            yerr=np.vstack([
                recent["slope_mm_d_per_K"] - recent["ci_low"],
                recent["ci_high"] - recent["slope_mm_d_per_K"],
            ]),
            color=RED, marker="s", markerfacecolor=RED, markeredgecolor=RED,
            linestyle="none", linewidth=1.0, capsize=2.2, markersize=5.0,
            label="2009–2023", zorder=3,
        )
        axis.axhline(0, color=GREY, linewidth=0.75, zorder=1)
        axis.set_title(site.name)
        axis.set_xlim(0.45, 9.55)
        axis.set_xticks(x)
        axis.set_xlabel("Wetterlagenklasse")
        axis.set_ylabel("Bedingte Temperaturreaktion\n[mm d$^{-1}$ K$^{-1}$]")
        axis.text(-0.11, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.16, top=0.83, wspace=0.23)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
