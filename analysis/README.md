# Analysis scripts

## Main scripts

`runoff_trends.py` and `climate_trends.py` are independent analyses. Run `weather_type_response.py` for the class-wise temperature-response figure; it loads the internal helper modules automatically. `hefex_case_study.py` creates the campaign figure. `classification_comparison.py` writes the table summarising the precomputed omega-squared comparison.

## Input data

The scripts expect the following files relative to the repository root:

- `data/all_final/ssim_q_weather_time_vs13.xlsx` for runoff and weather-response analyses;
- `data/climat/SPARTACUS_TM24_Datensatz_19610801_20260806.csv` and `SPARTACUS_RR_Datensatz_19610801_20260806.csv` for climate trends;
- `classification/wsom_hefex-main/results/3x3_som_results_seasonal_filtered_24000updates/clusters_series/w_0.0_n_3x3_m_ed_i_pca_clusters_time.csv` for the primary SOM assignment;
- `data/hefex_iop1/HEFEX_Kampagne_Aug2025.xlsx` for the HEFEX case study;
- `data/classification_audit/omega2_method_audit_summary.csv` and `omega2_all_methods_lags.csv` for the classification comparison.

Outputs are stored in `outputs/`. Existing outputs may be overwritten when a script is rerun.
