# Orientierung zu den Auswertungsskripten

Die Skripte befinden sich im Unterordner
`VS_Code/Rofental_discharge_weatherclasses/scripts/`. Sie wurden aus dem
Arbeitsverzeichnis übernommen und erzeugen Abbildungen beziehungsweise
Ergebnistabellen für die Arbeit. Die Eingabedaten liegen nicht im Repository.

## Empfohlene Reihenfolge

Die drei Hilfsskripte bilden eine kleine Kette für die Auswertung der
Temperaturreaktion:

1. `09_weather_adjusted_sensitivity.py` bereitet die wetterlagenbereinigte
   Sensitivitätsanalyse vor.
2. `10_weather_response_time_change.py` verwendet die Funktionen aus Skript
   09 für ergänzende Lag- und Klassenanalysen.
3. `12_primary_som_p0_response_vs13.py` ersetzt die historische
   Klassenzuordnung im Arbeitsdatensatz durch die primäre, ungewichtete
   3×3-SOM-Zuordnung.
4. `23_plot_paired_temperature_contrast.py` baut darauf auf und erstellt die
   Darstellung der Temperaturreaktion in physikalischen Einheiten.

Die Skripte `05_results_trend_variants.py`, `06_climate_trend_figures.py`,
`53_plot_hefex_campaign_hourly_waterlevel.py` und
`57_write_omega2_method_appendix_table.py` können unabhängig davon verwendet
werden, sobald die jeweils benötigten Dateien unter `data/` liegen.

## Ausführung

Die Pfade werden ausgehend vom Speicherort des jeweiligen Skripts aufgebaut.
Das Repository muss daher in seiner vorliegenden Ordnerstruktur verwendet
werden. Ein typischer Aufruf lautet:

```bash
python scripts/VS_Code/Rofental_discharge_weatherclasses/scripts/06_climate_trend_figures.py
```

Die Skripte schreiben Ergebnisse in Unterordner von `scripts/.../figures/`.
Diese Ordner und alle Datenbestände werden nicht versioniert. Vor einem
erneuten Lauf sollten vorhandene Abbildungen gesichert werden, falls sie
behalten werden sollen.
