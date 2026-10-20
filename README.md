# Rofental runoff analysis

Selected analysis scripts for the bachelor thesis *Synoptic control of summer runoff in the Rofental*. The code covers climate and runoff trends, weather-type-dependent runoff responses, the HEFEX-III case study, and the comparison of weather-type classifications.

## Scripts

The main scripts are in [`analysis/`](analysis/):

- `runoff_trends.py` – seasonal runoff characteristics and long-term trends;
- `climate_trends.py` – temperature and precipitation trends from SPARTACUS;
- `weather_type_response.py` – temperature response of runoff within weather types;
- `hefex_case_study.py` – meteorological and water-level observations from HEFEX-III;
- `classification_comparison.py` – summary table for the classification comparison.

Files starting with an underscore are helper modules used by `weather_type_response.py`.

## Data and execution

The repository does not contain measurement data, ERA5 fields, intermediate classification results, or figures. The expected input paths are listed in [`analysis/README.md`](analysis/README.md). With the original data available, install the packages in `requirements.txt` and run a script from the repository root, for example:

```bash
python analysis/weather_type_response.py
```

Results are written to `outputs/`, which is intentionally ignored by Git.

## Weather-type classification

The scripts analyse the primary unweighted 3×3 SOM classification. The full ERA5 preprocessing and SOM-training workflow is not included here because it relies on the separately developed `wsom_pad` package and the ERA5 input data. It can be added once authorship and reuse have been clarified.

Questions about reuse of the code or underlying data can be directed to moritz.thomaser@edu.uni-graz.at.
