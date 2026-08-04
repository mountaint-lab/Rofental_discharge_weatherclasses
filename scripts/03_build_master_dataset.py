"""
Build the corrected master dataset:
  1. Re-attach bmu/smu from the SOM weather-type classification CSV via an
     exact date join (fixes an earlier row-alignment bug where bmu/smu had
     been pasted in without a proper key, giving ~11% match rate = chance
     level for 9 classes).
  2. Recompute month/year/doy for every row directly from the timestamp
     (these were empty for the 2014-2023 extension in the original file).
  3. Recompute Q_Niedertalbach_anom / Q_Vernagt_anom fresh and consistently
     over the *entire* available record per station (day-of-year climatology,
     7-day smoothed). Note: this does not attempt to reproduce whatever
     method produced the original 1974-2013 anomaly values exactly (a
     reconstruction check did not match closely enough to be confident of
     the original method) - the whole column is recomputed with one
     documented method instead of mixing two unknown methods.

RR / Tmean / rain_flag remain empty for 2014-2023: there is no meteorological
data source for that period in this pipeline. This is a genuine data gap,
not something this script can fix.

Input:  a "clean_complete_cases" sheet with at least the columns
        time, bmu, smu, Q_Vernagt, Q_Niedertalbach, W_Niedertalbach, RR,
        Tmean, rain_flag
        + the SOM classification csv with columns time, bmu_proj, smu_proj
        (use the *final* weighting version - confirm with whoever produced it)
Output: corrected xlsx with the same sheet plus a documentation sheet.
"""
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

INPUT_XLSX = "./data/raw/ssim_q_weather_time_vs03.xlsx"                       # <-- adjust
INPUT_SHEET = "clean_complete_cases"
CLASSIFICATION_CSV = "./data/raw/w_1_5_n_3x3_m_ed_i_pca_clusters_time.csv"     # <-- adjust: final weighting version
OUTPUT_XLSX = "./data/processed/ssim_q_weather_time_vs03_corrected.xlsx"

COLUMN_ORDER = ["time", "bmu", "smu", "Q_Vernagt", "Q_Niedertalbach", "W_Niedertalbach",
                "RR", "Tmean", "rain_flag", "month", "Q_Vernagt_anom",
                "Q_Niedertalbach_anom", "year", "doy"]


def fix_bmu_smu(df: pd.DataFrame, classification_csv: str) -> pd.DataFrame:
    cls = pd.read_csv(classification_csv, sep=None, engine="python")
    # parse_dates during read_csv is silently skipped when engine="python" + sep=None,
    # and the source csv uses day-first dates (e.g. "01.05.1952") - parse explicitly.
    cls["time"] = pd.to_datetime(cls["time"], dayfirst=True).dt.normalize()
    cls = cls.set_index("time")
    df = df.copy()
    key = pd.to_datetime(df["time"]).dt.normalize()
    df["bmu"] = key.map(cls["bmu_proj"]).astype("Int64")
    df["smu"] = key.map(cls["smu_proj"]).astype("Int64")
    n_unmatched = df["bmu"].isna().sum()
    if n_unmatched:
        print(f"WARNING: {n_unmatched} rows had no matching date in the classification csv")
    return df


def add_derived_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df["doy"] = df["time"].dt.dayofyear
    return df


def add_anomaly(df: pd.DataFrame, col: str, anom_col: str) -> pd.DataFrame:
    """Anomaly = value - (7-day-smoothed day-of-year climatology), computed
    over the full available record for that column."""
    df = df.copy()
    clim = df.groupby("doy")[col].mean()
    clim_smooth = clim.rolling(7, center=True, min_periods=3).mean()
    df[anom_col] = df.apply(
        lambda r: r[col] - clim_smooth.loc[r["doy"]] if pd.notna(r[col]) else np.nan, axis=1
    )
    return df


def save_xlsx(df: pd.DataFrame, path: str):
    df = df[COLUMN_ORDER]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = INPUT_SHEET
    font_normal = Font(name="Arial", size=10)
    font_header = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    fill_header = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    for c, h in enumerate(COLUMN_ORDER, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font, cell.fill = font_header, fill_header
        cell.alignment = Alignment(horizontal="center")
    for ridx, row in enumerate(df.itertuples(index=False), start=2):
        for c, v in enumerate(row, start=1):
            v = None if pd.isna(v) else v
            cell = ws.cell(row=ridx, column=c, value=v)
            cell.font = font_normal
            if c == 1 and v is not None:
                cell.number_format = "yyyy-mm-dd hh:mm:ss"
    ws.freeze_panes = "A2"
    for c, h in enumerate(COLUMN_ORDER, start=1):
        ws.column_dimensions[get_column_letter(c)].width = max(12, len(h) + 4)

    notes = wb.create_sheet("Korrektur_Hinweise")
    b, n = Font(name="Arial", size=10, bold=True), Font(name="Arial", size=10)
    lines = [
        ("bmu/smu:", "Re-attached from the final SOM classification csv via exact date join."),
        ("month/year/doy:", "Freshly computed from time for every row."),
        ("Q_*_anom:", "Freshly and consistently recomputed over the full available "
                       "record per station (see script docstring for method + caveat)."),
        ("RR, Tmean, rain_flag:", "Still empty where no meteorological source exists "
                                   "for that period. Real data gap, not a bug."),
    ]
    for i, (k, v) in enumerate(lines, start=1):
        c1 = notes.cell(row=i, column=1, value=k); c1.font = b
        c2 = notes.cell(row=i, column=2, value=v); c2.font = n
        c2.alignment = Alignment(wrap_text=True, vertical="top")
    notes.column_dimensions["A"].width = 24
    notes.column_dimensions["B"].width = 100
    wb.save(path)


if __name__ == "__main__":
    import os
    os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
    df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)
    df = df[df["time"].notna()].copy()          # drop stray rows with no timestamp

    df = fix_bmu_smu(df, CLASSIFICATION_CSV)
    df = add_derived_time_fields(df)
    df = add_anomaly(df, "Q_Niedertalbach", "Q_Niedertalbach_anom")
    df = add_anomaly(df, "Q_Vernagt", "Q_Vernagt_anom")

    save_xlsx(df, OUTPUT_XLSX)
    print(f"Saved {len(df)} rows to {OUTPUT_XLSX}")
    print("Q_Niedertalbach_anom coverage:", df["Q_Niedertalbach_anom"].notna().sum(),
          df.loc[df["Q_Niedertalbach_anom"].notna(), "year"].min(), "-",
          df.loc[df["Q_Niedertalbach_anom"].notna(), "year"].max())
    print("Q_Vernagt_anom coverage:", df["Q_Vernagt_anom"].notna().sum(),
          df.loc[df["Q_Vernagt_anom"].notna(), "year"].min(), "-",
          df.loc[df["Q_Vernagt_anom"].notna(), "year"].max())
