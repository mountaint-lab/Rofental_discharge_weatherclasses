# Codeauswahl zur Bachelorarbeit: Abfluss und Wetterlagen im Rofental

Dieses Paket enthält eine Auswahl von Auswertungsskripten zur Bachelorarbeit über die klimatologische und synoptische Steuerung des sommerlichen Abflusses im Rofental. Es ist keine vollständige oder unabhängig ausführbare Reproduktionspipeline, beinhaltet jedoch die wichtigsten Auswertungen für die Bachelorarbeit.

## Fünf ausgewählte Skripte

Die Dateien liegen unter `scripts/VS_Code/Rofental_discharge_weatherclasses/scripts/`:

1. `05_results_trend_variants.py`: Abflussentwicklung, Trends und saisonale Darstellungen.
2. `06_climate_trend_figures.py`: Temperatur- und Niederschlagsentwicklung auf Basis von SPARTACUS-Daten.
3. `23_plot_paired_temperature_contrast.py`: klassenweise Temperatur-Abfluss-Beziehungen und deren Veränderung zwischen den Vergleichsperioden.
4. `53_plot_hefex_campaign_hourly_waterlevel.py`: stündliche Darstellung meteorologischer Größen und normierter Wasserstandssignale der HEFEX-III-Kampagne.
5. `57_write_omega2_method_appendix_table.py`: Zusammenstellung bereits berechneter ω²-Ergebnisse für den Klassifikationsvergleich; das Skript berechnet die zugrunde liegende Analyse nicht selbst.

Die Dateien `09_weather_adjusted_sensitivity.py`, `10_weather_response_time_change.py` und `12_primary_som_p0_response_vs13.py` sind Hilfsskripte, die für die Temperaturauswertung benötigt werden. Einzelne Auswertungen und Abbildungen der Arbeit, insbesondere SOM-Training, Erstellung der Klassifikationen und des gemeinsamen Datensatzes, sind in dieser Auswahl **nicht** vollständig enthalten.

## Daten und Einschränkungen

Messdaten, auf Anfrage erhaltene Daten, der zusammengeführte Datensatz, Klassifikationsdateien und vorab berechnete Ergebnisse sind nicht Bestandteil dieses Pakets. Ihre Weitergabe und Nutzung unterliegen den Bedingungen der jeweiligen Datengeber. Eine vollständige Reproduktion der Ergebnisse ist allein mit diesem Paket nicht möglich.

Die Skripte erwarten – sofern die entsprechenden Daten vorliegen und verwendet werden dürfen – unter anderem folgende Pfade relativ zu diesem Ordner:

- `data/all_final/ssim_q_weather_time_vs13.xlsx`: gemeinsamer Datensatz; die verwendeten Tabellenblätter sind in den Skripten angegeben.
- `data/climat/`: SPARTACUS-Temperatur- und Niederschlagsdateien mit den in Skript 06 genannten Namen.
- `classification/wsom_hefex-main/results/3x3_som_results_seasonal_filtered_24000updates/clusters_series/w_0.0_n_3x3_m_ed_i_pca_clusters_time.csv`: Zuordnung der Referenz-SOM; Skript 12 unterstützt alternativ die Umgebungsvariable `SOM_CLASS_FILE`.
- `data/hefex_iop1/HEFEX_Kampagne_Aug2025.xlsx`: Eingangsdaten der HEFEX-Darstellung.
- `data/classification_audit/omega2_method_audit_summary.csv` und `omega2_all_methods_lags.csv`: zuvor berechnete Eingangstabellen für Skript 57.

Bei Bedarf müssen Dateipfade, Tabellenblätter und Eingabespalten an die verfügbaren Daten angepasst werden. Die Ausgabe erfolgt in Unterordnern von `scripts/VS_Code/Rofental_discharge_weatherclasses/scripts/figures/`. Die Skripte können dort Dateien mit festen Namen überschreiben; bei erneuter Ausführung Ausgaben vorher sichern.

## Umgebung

Für die ausgewählten Skripte werden Python 3 und die Pakete in `requirements.txt` benötigt. Beispiel:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/VS_Code/Rofental_discharge_weatherclasses/scripts/23_plot_paired_temperature_contrast.py
```

Das Beispiel setzt die oben genannten Datendateien und Klassenzuordnungen voraus. Die Paketversionen sind nicht fixiert; die genaue Originalumgebung wurde nicht vollständig rekonstruiert. Ohne die Originaldaten wurden die Auswertungen nicht erneut end-to-end ausgeführt.

## Herkunft und Weitergabe

Dies ist eine vorläufige Auswahl aus dem Arbeitsverzeichnis. Rechte an gemeinsam entwickeltem Code und an verwendeten Daten sind vor einer öffentlichen Veröffentlichung bei moritz.thomaser@edu.uni-graz.at anzufragen.
