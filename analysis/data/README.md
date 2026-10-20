# Data included in this repository

The repository contains only the compact data files needed to reproduce the
main runoff, climate-trend, and weather-response analyses. The files are kept
in their original working formats and paths so that the scripts can be run
without changing file locations.

| File | Purpose | Source and licence |
| --- | --- | --- |
| `all_final/ssim_q_weather_time_vs13.xlsx` | Prepared daily analysis dataset for May–September 1974–2023, including runoff anomalies and the meteorological variables used by the response analyses. | Derived for this study from the sources listed below. Please cite the original data providers when reusing individual variables. |
| `climat/SPARTACUS_TM24_Datensatz_19610801_20260806.csv` | Daily SPARTACUS air-temperature series used in `climate_trends.py`. | GeoSphere Austria, SPARTACUS daily climate data; [CC BY 4.0](https://data.hub.geosphere.at/en/dataset/spartacus-v2-1d-1km). |
| `climat/SPARTACUS_RR_Datensatz_19610801_20260806.csv` | Daily SPARTACUS precipitation series used in `climate_trends.py`. | GeoSphere Austria, SPARTACUS daily climate data; [CC BY 4.0](https://data.hub.geosphere.at/en/dataset/spartacus-v2-1d-1km). |
| `weather_types/primary_som_assignments.csv` | Daily class assignments of the primary unweighted 3 x 3 self-organizing map. | Derived from ERA5 500-hPa geopotential-height fields (Copernicus Climate Change Service information, 2020) for this study. |

The prepared analysis dataset also incorporates discharge and meteorological
records from the Rofental research basin. The eHYD data package and the
PANGAEA Rofental data series are available under Creative Commons Attribution
licences; see Strasser et al. (2017), https://doi.org/10.1594/PANGAEA.876120.

Raw ERA5 fields, high-frequency station measurements, HEFEX-III campaign data,
and large source archives are deliberately not redistributed here. They are not
needed for the included scripts and may require separate access or permission.
