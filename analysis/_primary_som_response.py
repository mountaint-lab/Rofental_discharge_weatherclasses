"""Weather-response figures based on the unweighted primary SOM (p = 0).

The corrected master workbook still contains the historical ablation-weighted
BMU assignment in its ``bmu`` column.  This script deliberately replaces that
column after a date-exact merge with the 24,000-update, unweighted SOM p=0
assignment used as the primary classification in the thesis.  The original
workbook and the legacy figure copies remain unchanged.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "all_final" / "ssim_q_weather_time_vs13.xlsx"
DEFAULT_CLASS_FILE = PROJECT_ROOT / "data" / "weather_types" / "primary_som_assignments.csv"
CLASS_FILE = Path(os.environ.get("SOM_CLASS_FILE", str(DEFAULT_CLASS_FILE)))
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "weather_type_response"
OUT_DIR = Path(os.environ.get("SOM_OUT_DIR", str(DEFAULT_OUT_DIR)))

FIG_SCATTER_SAME_DAY = OUT_DIR / "fig25a_temperature_discharge_same_day_p0"
FIG_SCATTER_LAG_COMPARISON = OUT_DIR / "fig25b_temperature_same_vs_previous_day_p0"
FIG_OVERVIEW = OUT_DIR / "fig26_weatherclass_overview_p0"
FIG_MEAN_Q = OUT_DIR / "fig27_weatherclass_mean_discharge_temperature_order_p0"
FIG_CHANGE = OUT_DIR / "fig28_weatherclass_frequency_sensitivity_change_p0"
FIG_RESPONSE_VERNAGT = OUT_DIR / "fig29_weatherclass_temperature_response_vernagtbach_p0"
FIG_RESPONSE_ROFENACHE = OUT_DIR / "fig30_weatherclass_temperature_response_rofenache_vent_p0"
RESULTS_FILE = OUT_DIR / "primary_som_p0_weather_response_statistics.csv"


def load_analysis_module():
    path = SCRIPT_DIR / "_weather_response_diagnostics.py"
    spec = importlib.util.spec_from_file_location("weather_response_time_change", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_corrected_primary_data() -> pd.DataFrame:
    raw = pd.read_excel(DATA_FILE, sheet_name="clean_complete_cases")
    raw["date"] = pd.to_datetime(raw["time"], errors="raise").dt.normalize()
    labels = pd.read_csv(CLASS_FILE)
    labels["date"] = pd.to_datetime(labels["time"], format="%Y-%m-%d", errors="raise").dt.normalize()
    labels = labels[["date", "bmu_proj"]].drop_duplicates("date")

    raw = raw.rename(columns={"bmu": "bmu_legacy_p15"}).merge(
        labels,
        on="date",
        how="left",
        validate="many_to_one",
    )
    if raw["bmu_proj"].isna().any():
        missing = raw.loc[raw["bmu_proj"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"Missing primary SOM labels for {len(missing)} dates: {missing[:5]}")
    raw["bmu"] = raw["bmu_proj"].astype(int)
    if sorted(raw["bmu"].unique()) != list(range(9)):
        raise RuntimeError("Primary SOM assignment does not contain all classes 0–8")
    return raw


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = load_analysis_module()
    raw = load_corrected_primary_data()
    prepared = {site.name: analysis.prepare_site(raw, site) for site in analysis.SITES}

    rows: list[dict] = []
    rows.extend(
        analysis.scatter_figure(
            prepared,
            lag=0,
            output=FIG_SCATTER_SAME_DAY,
        )
    )
    rows.extend(
        analysis.clearer_lag_comparison_scatter(
            prepared,
            output=FIG_SCATTER_LAG_COMPARISON,
        )
    )
    rows.extend(analysis.overview_figure(raw, prepared, output=FIG_OVERVIEW))
    rows.extend(analysis.mean_discharge_temperature_order_figure(prepared, output=FIG_MEAN_Q))
    rows.extend(
        analysis.change_figure(
            raw,
            prepared,
            temperature_column="temperature_lag0_z",
            lag=0,
            output=FIG_CHANGE,
        )
    )
    rows.extend(
        analysis.weatherclass_period_response(
            prepared,
            output_vernagt=FIG_RESPONSE_VERNAGT,
            output_vent=FIG_RESPONSE_ROFENACHE,
        )
    )

    result = pd.DataFrame(rows)
    result.insert(0, "classification", "SOM p=0, 3x3, 24000 updates")
    result.insert(1, "discharge_dataset", "ssim_q_weather_time_vs13.xlsx")
    result.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")

    print(f"Input rows: {len(raw):,}")
    print(f"Date range: {raw['date'].min().date()} to {raw['date'].max().date()}")
    print(f"Primary class counts: {raw['bmu'].value_counts().sort_index().to_dict()}")
    print(f"Legacy labels retained only as: bmu_legacy_p15")
    for stem in [
        FIG_SCATTER_SAME_DAY,
        FIG_SCATTER_LAG_COMPARISON,
        FIG_OVERVIEW,
        FIG_MEAN_Q,
        FIG_CHANGE,
        FIG_RESPONSE_VERNAGT,
        FIG_RESPONSE_ROFENACHE,
    ]:
        print(f"Saved: {stem.with_suffix('.png')}")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
