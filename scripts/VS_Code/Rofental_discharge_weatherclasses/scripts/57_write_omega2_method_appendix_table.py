"""Create a compact appendix table of omega-squared values by method and site."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
AUDIT_DIR = PROJECT_ROOT / "data" / "classification_audit"
SOURCE = AUDIT_DIR / "omega2_method_audit_summary.csv"
ALL_METHODS_SOURCE = AUDIT_DIR / "omega2_all_methods_lags.csv"
OUT_DIR = SCRIPT_DIR / "figures" / "final_vs13" / "04_statistik"

METHODS = [
    ("SOM p=0 (24k; primär)", "Referenzklassifikation"),
    ("K-Means k=4", "Sensitivität"),
    ("K-Means k=6", "Sensitivität"),
    ("K-Means k=9", "Sensitivität"),
    ("Z500-Richtungen (8)", "Physikalische Referenz"),
    ("SOM p=1,0 (24k; diagnostisch)", "Diagnostisch"),
]


def interval(estimate: float, low: float, high: float, *, scale: float = 100.0) -> str:
    return f"{estimate * scale:.1f} [{low * scale:.1f}; {high * scale:.1f}]"


def main() -> None:
    source = pd.read_csv(SOURCE)
    all_methods = pd.read_csv(ALL_METHODS_SOURCE)
    rows: list[dict[str, object]] = []
    for method, role in METHODS:
        selected = source.loc[source["method"].eq(method)]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one row for {method!r}, found {len(selected)}")
        row = selected.iloc[0]
        site_rows = all_methods.loc[
            all_methods["method"].eq(method) & all_methods["lag_days_weather_before_q"].eq(0)
        ].set_index("site")
        vernagt = site_rows.loc["Vernagtbach"]
        rofenache = site_rows.loc["Rofenache – Pegel Vent"]
        rows.append(
            {
                "Verfahren": method.replace(" (24k; primär)", "").replace(" (24k; diagnostisch)", ""),
                "Rolle": role,
                "Klassen k": int(row["k"]),
                "ω² Vernagtbach [%]": interval(vernagt["omega2"], vernagt["omega2_ci_low"], vernagt["omega2_ci_high"]),
                "ω² Rofenache [%]": interval(rofenache["omega2"], rofenache["omega2_ci_low"], rofenache["omega2_ci_high"]),
                "Δω² V − R [pp]": interval(row["delta_lag0"], row["delta_lag0_ci_low"], row["delta_lag0_ci_high"]),
            }
        )

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "tabelle_a3_omega2_klassifikationsvergleich.csv", index=False, encoding="utf-8-sig")
    # Avoid console-encoding issues on Windows; the UTF-8 CSV is the deliverable.
    print(f"Wrote {len(table)} rows to {OUT_DIR / 'tabelle_a3_omega2_klassifikationsvergleich.csv'}")


if __name__ == "__main__":
    main()
