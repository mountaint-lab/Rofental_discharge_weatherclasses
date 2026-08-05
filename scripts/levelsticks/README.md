# HEFEX-III levelstick figures

`plot_hefex_levelsticks.py` reproduces the two HEFEX-III levelstick figures
for the Hintereisferner campaign (06.–10.08.2025). It creates:

- the full campaign time series for pressure proxy and water temperature;
- the diurnal comparison using a common y-axis and a continuous blue-to-red
  chronological colour scale.

## Private input data

The measurement workbook is deliberately **not** stored in this repository.
Place the locally available file at:

```text
private_data/HEFEX_levelsticks_trimmed.xlsx
```

The workbook contains the processed measurements, with the first 60 minutes
after deployment removed for each sensor. The raw CSV files are not required by
the script.

## Run

From this directory, with the project Python environment active:

```powershell
python plot_hefex_levelsticks.py
```

The script requires `pandas`, `openpyxl`, and `Pillow`. It writes these local
outputs into `figures/`:

```text
fig12_hefex_levelsticks.png
fig13_hefex_levelsticks_diurnal.png
```

`private_data/` and `figures/` are ignored by Git. This keeps both the
measurement data and generated figure files out of GitHub by default.
