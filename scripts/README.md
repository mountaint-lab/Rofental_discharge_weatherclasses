# Rofental discharge / weather-type analysis - exported scripts

Pipeline order:

1. **01_extract_niedertalbach_pdfs.py** - parses Hydrographisches Jahrbuch PDFs
   for Niedertalbach (station 201350) into `niedertalbach_parsed.pkl`.
2. **02_extract_vernagtbach_pdfs.py** - same for Vernagtbach (station 202689),
   handles the winter station closure via position-calibrated column parsing
   -> `vernagtbach_parsed.pkl`.
3. **03_build_master_dataset.py** - takes your existing base spreadsheet plus
   the final SOM weather-classification CSV, fixes the bmu/smu date-join bug,
   fills in month/year/doy, and recomputes the discharge anomaly columns over
   the full available record -> `ssim_q_weather_time_vs03_corrected.xlsx`.
4. **04_make_figures.py** - all final figures (discharge overview, boxplot by
   weather class, class sizes, seasonal shape by period, small multiples,
   peak-timing running-window robustness check, smartstake ablation checks).

## Open in VS Code

This folder is a self-contained VS Code project (`.vscode/` has run configs for
all four scripts under **Run and Debug**). Steps:

1. Open this `scripts/` folder in VS Code (`File > Open Folder...`).
2. Install the recommended extension if prompted (Python, by Microsoft).
3. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Scripts 01/02 also need `pdftotext` (poppler-utils) on PATH:
   - Ubuntu/Debian: `sudo apt install poppler-utils`
   - macOS: `brew install poppler`
   - Windows: install poppler and add its `bin` folder to PATH.
5. Select the `.venv` interpreter in VS Code (bottom-right status bar, or
   `Ctrl+Shift+P` -> "Python: Select Interpreter").
6. Drop your input files into `data/raw/` (see layout below), then run each
   script from **Run and Debug** in order 01 -> 04, or from the integrated
   terminal with `python3 01_extract_niedertalbach_pdfs.py` etc.

## Data layout

```
scripts/
  data/
    raw/                                    <- put your source files here
      pdfs_niedertalbach/OWF_201350_<year>.pdf
      pdfs_vernagtbach/OWF_202689_<year>.pdf
      ssim_q_weather_time_vs03.xlsx
      w_1_5_n_3x3_m_ed_i_pca_clusters_time.csv
      smartstake_daily_processed_2021_2024.csv
    interim/                                <- auto-created by 01/02 (pdftotext output + .pkl)
    processed/                              <- auto-created by 03 (corrected master .xlsx)
  figures/                                  <- auto-created by 04 (final .png figures)
```

Every script still has an `INPUT_DIR` / `INPUT_XLSX` / `OUTPUT_*` block near the
top pointing at these `data/` subfolders - adjust if your files live elsewhere.
Also adjust `YEARS` in scripts 01/02 to the years you actually have PDFs for.

## Testing status

All four scripts were syntax-checked and run end-to-end against synthetic
stand-in data built to match each script's expected input format (synthetic
Hydrographisches-Jahrbuch-style PDFs for 01/02, including a year with blank
winter columns for 02's station-closure handling; a synthetic base
spreadsheet + classification CSV for 03; and a synthetic master file +
smartstake CSV for 04). All four completed without errors, and 01/02's
built-in validation against the synthetic "MQ" row matched within rounding
tolerance (<0.001 m3/s), and 03's bmu/smu date join matched 100% of rows.
This confirms the parsing/join/plotting logic itself is sound - it does not
confirm the scripts reproduce a specific dataset's expected values, since no
real input files were available in the environment this was tested in. Run
them once against your real data and skim the printed validation output
before treating results as final.

## Known open items (see chat for full context)

- RR / Tmean / rain_flag have no source for 2014-2023 (real gap, not a bug).
- The recomputed anomaly method does not exactly reproduce whatever produced
  the original 1974-2013 anomaly values - documented in script 03's docstring.
- Vernagtbach data currently ends 2019; a request for 2020-2025 is pending.
