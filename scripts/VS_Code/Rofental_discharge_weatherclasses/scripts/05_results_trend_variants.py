"""Alternative result figures for the seasonal discharge analysis.

The script deliberately writes only to ``figures/variants`` and therefore
does not overwrite any existing figure.  It uses the established blue-to-red
time colours (early to recent) rather than the study-area site palette.
"""

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "all_final" / "ssim_q_weather_time_vs13.xlsx"
DATA_SHEET = "clean_complete_cases"
OUT_DIR = SCRIPT_DIR / "figures" / "corrected_vs13" / "variants"

SEASON_MONTHS = (5, 6, 7, 8, 9)
SEASON_LENGTH = 153  # 1 May through 30 September
EARLY = (1974, 1988)
RECENT = (2009, 2023)
MIN_HYDROGRAPH_N = 5

BLUE = "#4169D8"
RED = "#C5162E"
TREND = "#262626"
GRID = "#D9D9D9"
VERNAGT_SITE = "#7953A6"
VERNAGT_DARK = "#503377"
ROFENACHE_SITE = "#8FBC65"
ROFENACHE_DARK = "#517A35"

STATIONS = (
    ("Q_Vernagt", "Vernagtbach", "a"),
    ("Q_Rofenache_Vent", "Rofenache – Pegel Vent", "b"),
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.facecolor": "white",
        }
    )


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name=DATA_SHEET)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df = df[df["month"].isin(SEASON_MONTHS)].copy()
    may_first = pd.to_datetime(df["year"].astype(str) + "-05-01")
    df["season_day"] = (df["time"].dt.normalize() - may_first).dt.days
    return df.sort_values("time").reset_index(drop=True)


def sen_mann_kendall(years, values) -> dict[str, float]:
    """Return the Theil-Sen slope and a Hamed-Rao modified MK p-value.

    The modified variance accounts for significant positive serial
    correlation in the detrended annual series (Hamed and Rao, 1998).
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(y)
    if n < 3:
        return {"slope": np.nan, "intercept": np.nan, "p": np.nan, "n": n}

    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
        if x[j] != x[i]
    ]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))

    s_value = sum(
        int(np.sign(y[j] - y[i]))
        for i in range(n - 1)
        for j in range(i + 1, n)
    )
    _, tie_counts = np.unique(y, return_counts=True)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts if t > 1)
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    detrended = y - slope * x
    threshold = 1.96 / sqrt(n)
    serial_term = 0.0
    for lag in range(1, n - 2):
        leading, lagged = detrended[:-lag], detrended[lag:]
        if np.std(leading) == 0 or np.std(lagged) == 0:
            continue
        autocorrelation = float(np.corrcoef(leading, lagged)[0, 1])
        if np.isfinite(autocorrelation) and autocorrelation > threshold:
            serial_term += (
                (n - lag)
                * (n - lag - 1)
                * (n - lag - 2)
                * autocorrelation
            )
    effective_sample_factor = 1.0 + (
        2.0 * serial_term / (n * (n - 1) * (n - 2))
    )
    variance *= effective_sample_factor
    if variance <= 0:
        p_value = 1.0
    else:
        if s_value > 0:
            z_value = (s_value - 1) / sqrt(variance)
        elif s_value < 0:
            z_value = (s_value + 1) / sqrt(variance)
        else:
            z_value = 0.0
        p_value = erfc(abs(z_value) / sqrt(2.0))
    return {"slope": slope, "intercept": intercept, "p": p_value, "n": n}


def significance_text(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0,001"
    return f"p = {p_value:.3f}".replace(".", ",")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_annual_magnitude_with_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Annual May-September means, daily IQR, and robust trend."""
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True, dpi=180)
    cmap = plt.get_cmap("coolwarm")
    summary_rows: list[dict[str, float | str]] = []

    for ax, (column, title, panel) in zip(axes, STATIONS):
        grouped = df.groupby("year")[column]
        annual = pd.DataFrame(
            {
                "mean": grouped.mean(),
                "q25": grouped.quantile(0.25),
                "q75": grouped.quantile(0.75),
                "n": grouped.count(),
            }
        )
        annual = annual[annual["n"] >= 120].dropna()
        years = annual.index.to_numpy(dtype=float)
        colours = cmap((years - years.min()) / (years.max() - years.min()))
        yerr = np.vstack(
            [annual["mean"] - annual["q25"], annual["q75"] - annual["mean"]]
        )
        ax.bar(years, annual["mean"], width=0.82, color=colours, edgecolor="none", zorder=2)
        ax.errorbar(
            years,
            annual["mean"],
            yerr=yerr,
            fmt="none",
            ecolor="#454545",
            elinewidth=0.85,
            capsize=2.0,
            zorder=3,
        )

        result = sen_mann_kendall(years, annual["mean"].to_numpy())
        trend_y = result["intercept"] + result["slope"] * years
        ax.plot(years, trend_y, color=TREND, linewidth=1.8, zorder=4)
        slope_decade = result["slope"] * 10
        sign = "+" if slope_decade >= 0 else "−"
        slope_value = f"{abs(slope_decade):.2f}".replace(".", ",")
        label = (
            f"Sen-Trend: {sign}{slope_value} "
            r"$\mathrm{m^3\,s^{-1}}$ pro Dekade; "
            f"{significance_text(result['p'])}"
        )
        ax.text(
            0.99,
            0.94,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
        )
        ax.text(-0.035, 1.02, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_title(title, pad=5)
        ax.set_ylabel(
            r"$Q$ [$\mathrm{m^3\,s^{-1}}$]" "\n"
            "(Saisonmittel; Fehlerbalken =\n25.–75. Perzentil der Tageswerte)"
        )
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55, zorder=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.015)
        summary_rows.append(
            {
                "station": title,
                "metric": "May-September mean discharge",
                "n_years": result["n"],
                "sen_slope_per_year": result["slope"],
                "sen_slope_per_decade": slope_decade,
                "mann_kendall_p": result["p"],
            }
        )

    axes[-1].set_xlabel("Jahr")
    axes[-1].set_xticks(np.arange(1980, 2025, 10))
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.1)
    save_figure(fig, "fig14a_annual_seasonal_mean_with_trend_copy")
    return pd.DataFrame(summary_rows)


def plot_annual_magnitude_site_colours(df: pd.DataFrame) -> None:
    """Same annual trend figure, but with one established colour per site."""
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True, dpi=180)
    site_colours = {
        "Q_Vernagt": (VERNAGT_SITE, VERNAGT_DARK),
        "Q_Rofenache_Vent": (ROFENACHE_SITE, ROFENACHE_DARK),
    }

    for ax, (column, title, panel) in zip(axes, STATIONS):
        grouped = df.groupby("year")[column]
        annual = pd.DataFrame(
            {
                "mean": grouped.mean(),
                "q25": grouped.quantile(0.25),
                "q75": grouped.quantile(0.75),
                "n": grouped.count(),
            }
        )
        annual = annual[annual["n"] >= 120].dropna()
        years = annual.index.to_numpy(dtype=float)
        yerr = np.vstack(
            [annual["mean"] - annual["q25"], annual["q75"] - annual["mean"]]
        )
        bar_colour, trend_colour = site_colours[column]
        ax.bar(
            years,
            annual["mean"],
            width=0.82,
            color=bar_colour,
            alpha=0.84,
            edgecolor="none",
            zorder=2,
        )
        ax.errorbar(
            years,
            annual["mean"],
            yerr=yerr,
            fmt="none",
            ecolor="#454545",
            elinewidth=0.85,
            capsize=2.0,
            zorder=3,
        )

        result = sen_mann_kendall(years, annual["mean"].to_numpy())
        trend_y = result["intercept"] + result["slope"] * years
        ax.plot(years, trend_y, color=trend_colour, linewidth=2.0, zorder=4)
        slope_decade = result["slope"] * 10
        sign = "+" if slope_decade >= 0 else "−"
        slope_value = f"{abs(slope_decade):.2f}".replace(".", ",")
        label = (
            f"Sen-Trend: {sign}{slope_value} "
            r"$\mathrm{m^3\,s^{-1}}$ pro Dekade; "
            f"{significance_text(result['p'])}"
        )
        ax.text(
            0.99,
            0.94,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
        )
        ax.text(-0.035, 1.02, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_title(title, pad=5)
        ax.set_ylabel(
            r"$Q$ [$\mathrm{m^3\,s^{-1}}$]" "\n"
            "(Saisonmittel; Fehlerbalken =\n25.–75. Perzentil der Tageswerte)"
        )
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55, zorder=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.015)

    axes[-1].set_xlabel("Jahr")
    axes[-1].set_xticks(np.arange(1980, 2025, 10))
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.1)
    save_figure(fig, "fig14a2_annual_seasonal_mean_site_colours_with_trend_copy")


def plot_peak_water_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """Explore whether Vernagtbach shows a sustained post-maximum decline.

    Panel a retains the annual May-September means and adds a trailing
    ten-year mean. Panel b shows robust Sen slopes for consecutive trailing
    15-year windows; a post-peak phase would require persistently negative
    window slopes rather than a single high-flow year.
    """
    grouped = df.groupby("year")["Q_Vernagt"].agg(["mean", "count"])
    annual = grouped[grouped["count"] >= 120]["mean"].dropna()
    years = annual.index.to_numpy(dtype=float)
    values = annual.to_numpy(dtype=float)
    rolling_10 = annual.rolling(10, min_periods=10).mean()
    overall = sen_mann_kendall(years, values)

    window_rows = []
    for end_year in range(int(years.min()) + 14, int(years.max()) + 1):
        window = annual.loc[end_year - 14 : end_year]
        if len(window) != 15:
            continue
        result = sen_mann_kendall(window.index.to_numpy(), window.to_numpy())
        window_rows.append(
            {
                "start_year": end_year - 14,
                "end_year": end_year,
                "slope_decade": result["slope"] * 10,
                "p": result["p"],
            }
        )
    windows = pd.DataFrame(window_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=180)
    ax = axes[0]
    ax.plot(years, values, color="#BDBDBD", linewidth=0.75, zorder=1)
    ax.scatter(
        years,
        values,
        s=22,
        facecolor="white",
        edgecolor=VERNAGT_SITE,
        linewidth=0.9,
        zorder=3,
        label="Jährliches Saisonmittel",
    )
    ax.plot(
        rolling_10.index,
        rolling_10,
        color=VERNAGT_DARK,
        linewidth=2.2,
        zorder=4,
        label="Gleitendes 10-Jahres-Mittel (rückblickend)",
    )
    trend_y = overall["intercept"] + overall["slope"] * years
    ax.plot(
        years,
        trend_y,
        color=TREND,
        linewidth=1.3,
        linestyle="--",
        label="Sen-Trend 1974–2023",
        zorder=2,
    )
    ax.set_title("Saisonmittel und langfristige Entwicklung")
    ax.set_xlabel("Jahr")
    ax.set_ylabel(r"$Q$ Mai–September [$\mathrm{m^3\,s^{-1}}$]")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55)
    ax.legend(frameon=False, loc="upper left", fontsize=7.4)
    ax.text(-0.08, 1.03, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1]
    ax.axhline(0, color="#777777", linewidth=0.9, zorder=1)
    ax.plot(
        windows["end_year"],
        windows["slope_decade"],
        color=VERNAGT_SITE,
        linewidth=1.4,
        zorder=2,
    )
    significant = windows["p"] < 0.05
    ax.scatter(
        windows.loc[~significant, "end_year"],
        windows.loc[~significant, "slope_decade"],
        s=22,
        facecolor="white",
        edgecolor=VERNAGT_SITE,
        linewidth=0.9,
        zorder=3,
        label=r"$p \geq 0{,}05$",
    )
    ax.scatter(
        windows.loc[significant, "end_year"],
        windows.loc[significant, "slope_decade"],
        s=24,
        facecolor=VERNAGT_DARK,
        edgecolor=VERNAGT_DARK,
        linewidth=0.8,
        zorder=4,
        label="$p < 0{,}05$",
    )
    latest = windows.iloc[-1]
    latest_sign = "+" if latest["slope_decade"] >= 0 else "−"
    latest_slope = f"{abs(latest['slope_decade']):.2f}".replace(".", ",")
    latest_p = significance_text(float(latest["p"]))
    ax.text(
        0.98,
        0.96,
        f"2009–2023: {latest_sign}{latest_slope} "
        r"$\mathrm{m^3\,s^{-1}}$ pro Dekade; "
        f"{latest_p}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.2},
    )
    ax.set_title("Sen-Trend in rückblickenden 15-Jahres-Fenstern")
    ax.set_xlabel("Endjahr des Zeitfensters")
    ax.set_ylabel(r"Sen-Steigung [$\mathrm{m^3\,s^{-1}}$ pro Dekade]")
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55)
    ax.legend(frameon=False, loc="lower left", fontsize=7.4)
    ax.text(-0.08, 1.03, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "fig14e_peak_water_diagnostic_vernagtbach_copy")
    return pd.DataFrame(
        [
            {
                "station": "Vernagtbach",
                "metric": "latest 15-year Sen slope (2009-2023)",
                "n_years": 15,
                "sen_slope_per_decade": latest["slope_decade"],
                "mann_kendall_p": latest["p"],
            }
        ]
    )


def plot_polar_monthly_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute monthly discharge by year in an expanded polar layout.

    Individual annual points receive a small deterministic angular jitter to
    reduce overplotting. The jitter has no temporal meaning; period medians
    remain exactly on the five monthly axes.
    """
    monthly = (
        df.groupby(["year", "month"])[["Q_Vernagt", "Q_Rofenache_Vent"]]
        .agg(["mean", "count"])
    )
    monthly.columns = ["_".join(parts) for parts in monthly.columns]
    monthly = monthly.reset_index()

    years = np.arange(int(df["year"].min()), int(df["year"].max()) + 1)
    norm = plt.Normalize(years.min(), years.max())
    cmap = plt.get_cmap("coolwarm")
    month_angles = dict(zip(SEASON_MONTHS, np.linspace(0, np.pi, len(SEASON_MONTHS))))
    month_names = ["Mai", "Jun", "Jul", "Aug", "Sep"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 6.1),
        subplot_kw={"projection": "polar"},
        dpi=180,
    )
    trend_rows: list[dict[str, float | str]] = []
    random_generator = np.random.default_rng(20260809)

    for ax, (column, title, panel) in zip(axes, STATIONS):
        mean_column = f"{column}_mean"
        count_column = f"{column}_count"
        valid = monthly[monthly[count_column] >= 20].copy()

        theta = valid["month"].map(month_angles).to_numpy(dtype=float)
        theta_display = theta + random_generator.uniform(
            low=np.deg2rad(-4.5), high=np.deg2rad(4.5), size=len(theta)
        )
        radii = valid[mean_column].to_numpy(dtype=float)
        point_colours = cmap(norm(valid["year"].to_numpy(dtype=float)))
        ax.scatter(
            theta_display,
            radii,
            c=point_colours,
            s=22,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.30,
            zorder=5,
        )

        for period, colour, label in (
            (EARLY, BLUE, f"Median {EARLY[0]}–{EARLY[1]}"),
            (RECENT, RED, f"Median {RECENT[0]}–{RECENT[1]}"),
        ):
            sub = valid[valid["year"].between(*period)]
            profile = sub.groupby("month")[mean_column].median().reindex(SEASON_MONTHS)
            ax.plot(
                list(month_angles.values()),
                profile.to_numpy(dtype=float),
                color=colour,
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                label=label,
                zorder=3,
            )

        for month in SEASON_MONTHS:
            sub = valid[valid["month"] == month]
            result = sen_mann_kendall(sub["year"], sub[mean_column])
            trend_rows.append(
                {
                    "station": title,
                    "metric": f"monthly mean discharge, month={month}",
                    "n_years": result["n"],
                    "sen_slope_per_decade": result["slope"] * 10,
                    "mann_kendall_p": result["p"],
                }
            )

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(list(month_angles.values()), month_names)
        ax.set_rlabel_position(270)
        ax.tick_params(axis="x", pad=7)
        ax.yaxis.grid(True, color="#BDBDBD", linewidth=0.65, alpha=0.68)
        ax.xaxis.grid(True, color="#D2D2D2", linewidth=0.55, alpha=0.50)
        ax.spines["polar"].set_color("#333333")
        ax.spines["polar"].set_linewidth(0.9)
        ax.set_title(title, pad=16)
        ax.text(
            -0.07,
            1.04,
            panel,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
            va="top",
        )
        ax.text(
            0.50,
            0.985,
            r"Monatsmittel $Q$ [$\mathrm{m^3\,s^{-1}}$]",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#444444",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colourbar = fig.colorbar(
        scalar,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.10,
        aspect=42,
    )
    colourbar.set_label("Jahr")
    colourbar.set_ticks([1980, 1990, 2000, 2010, 2020])
    fig.subplots_adjust(left=0.035, right=0.985, top=0.84, bottom=0.18, wspace=0.22)
    save_figure(fig, "fig14f2_polar_monthly_magnitude_expanded_copy")
    return pd.DataFrame(trend_rows)


def period_hydrograph(df: pd.DataFrame, column: str, years: tuple[int, int]) -> pd.DataFrame:
    sub = df[df["year"].between(*years)].dropna(subset=[column])
    grouped = sub.groupby("season_day")[column]
    result = pd.DataFrame(
        {
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
            "n": grouped.count(),
        }
    ).reindex(range(SEASON_LENGTH))
    result.loc[result["n"] < MIN_HYDROGRAPH_N, ["median", "q25", "q75"]] = np.nan
    return result


def plot_absolute_early_recent_hydrographs(df: pd.DataFrame) -> None:
    """Absolute early-versus-recent seasonal hydrographs with daily IQR."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=180)
    synthetic_dates = pd.date_range("2001-05-01", periods=SEASON_LENGTH, freq="D")

    for ax, (column, title, panel) in zip(axes, STATIONS):
        for years, colour, label in (
            (EARLY, BLUE, f"{EARLY[0]}–{EARLY[1]}"),
            (RECENT, RED, f"{RECENT[0]}–{RECENT[1]}"),
        ):
            hydro = period_hydrograph(df, column, years)
            ax.fill_between(
                synthetic_dates,
                hydro["q25"].to_numpy(dtype=float),
                hydro["q75"].to_numpy(dtype=float),
                color=colour,
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                synthetic_dates,
                hydro["median"].to_numpy(dtype=float),
                color=colour,
                linewidth=1.55,
                label=label,
            )
        ax.text(-0.08, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_title(title)
        ax.set_xlabel("Monat")
        ax.set_ylabel(r"Median $Q$ [$\mathrm{m^3\,s^{-1}}$]")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55)
        ax.set_ylim(bottom=0)
        ax.margins(x=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.2)
    save_figure(fig, "fig14b_absolute_hydrographs_early_recent_copy")


def annual_parde(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for year, sub in df.groupby("year"):
        values = sub[column].dropna()
        if len(values) < 120:
            continue
        seasonal_mean = values.mean()
        if not np.isfinite(seasonal_mean) or seasonal_mean <= 0:
            continue
        monthly = sub.groupby("month")[column].mean()
        for month, monthly_mean in monthly.items():
            rows.append({"year": year, "month": month, "parde": monthly_mean / seasonal_mean})
    return pd.DataFrame(rows)


def plot_parde_early_recent(df: pd.DataFrame) -> None:
    """Reduced Pardé figure: two robust 15-year periods instead of ten lines."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True, dpi=180)
    month_labels = ["Mai", "Jun", "Jul", "Aug", "Sep"]

    for ax, (column, title, panel) in zip(axes, STATIONS):
        parde = annual_parde(df, column)
        for years, colour, label in (
            (EARLY, BLUE, f"{EARLY[0]}–{EARLY[1]}"),
            (RECENT, RED, f"{RECENT[0]}–{RECENT[1]}"),
        ):
            sub = parde[parde["year"].between(*years)]
            grouped = sub.groupby("month")["parde"]
            median = grouped.median().reindex(SEASON_MONTHS)
            q25 = grouped.quantile(0.25).reindex(SEASON_MONTHS)
            q75 = grouped.quantile(0.75).reindex(SEASON_MONTHS)
            ax.fill_between(SEASON_MONTHS, q25, q75, color=colour, alpha=0.15, linewidth=0)
            ax.plot(
                SEASON_MONTHS,
                median,
                color=colour,
                marker="o",
                markersize=3.8,
                linewidth=1.5,
                label=label,
            )
        ax.axhline(1, color="#A8A8A8", linestyle="--", linewidth=0.8)
        ax.text(-0.08, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_title(title)
        ax.set_xticks(SEASON_MONTHS, month_labels)
        ax.set_xlabel("Monat")
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Pardé-Koeffizient $K_j$\n(Monatsmittel / Saisonmittel)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    save_figure(fig, "fig14c_parde_early_recent_copy")


def plot_parde_five_year_corrected(df: pd.DataFrame) -> None:
    """Pardé profiles for ten consecutive, equal five-year periods.

    This preserves the informative multi-period design from the earlier
    illustration while recomputing every value from the corrected vs13
    discharge series.  The periods are deliberately aligned to the analysis
    start: 1974–1978, 1979–1983, ..., 2019–2023.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35), sharey=True, dpi=180)
    month_labels = ["Mai", "Jun", "Jul", "Aug", "Sep"]
    periods = [(start, start + 4) for start in range(1974, 2024, 5)]
    colours = plt.get_cmap("coolwarm")(np.linspace(0.06, 0.94, len(periods)))

    for ax, (column, title, panel) in zip(axes, STATIONS):
        parde = annual_parde(df, column)
        for (start, end), colour in zip(periods, colours):
            profile = (
                parde[parde["year"].between(start, end)]
                .groupby("month")["parde"]
                .median()
                .reindex(SEASON_MONTHS)
            )
            ax.plot(
                SEASON_MONTHS,
                profile,
                color=colour,
                marker="o",
                markersize=3.3,
                linewidth=1.35,
                alpha=0.90,
                label=f"{start}–{end}",
            )
        ax.axhline(1, color="#A8A8A8", linestyle="--", linewidth=0.8)
        ax.text(-0.08, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_title(title)
        ax.set_xticks(SEASON_MONTHS, month_labels)
        ax.set_xlabel("Monat")
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Pardé-Koeffizient $K_j$\n(Monatsmittel / Saisonmittel)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.2)
    save_figure(fig, "fig14c_parde_five_year_corrected")


def flow_quantile_day(series: pd.Series, fraction: float) -> float:
    cumulative = series.cumsum().to_numpy(dtype=float)
    target = fraction * cumulative[-1]
    idx = int(np.searchsorted(cumulative, target))
    if idx == 0:
        return 0.0
    before, after = cumulative[idx - 1], cumulative[idx]
    if after == before:
        return float(idx)
    return (idx - 1) + (target - before) / (after - before)


def annual_flow_timing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for year, sub in df.groupby("year"):
        s = sub.set_index("season_day")[column].reindex(range(SEASON_LENGTH))
        coverage = s.notna().mean()
        if coverage < 0.90:
            continue
        # Only short internal gaps are interpolated; seasons with unresolved
        # gaps are excluded to avoid shifting the cumulative-flow dates.
        filled = s.interpolate(limit=3, limit_area="inside")
        if filled.isna().any() or filled.sum() <= 0:
            continue
        rows.append(
            {
                "year": year,
                "q25_day": flow_quantile_day(filled, 0.25),
                "q50_day": flow_quantile_day(filled, 0.50),
                "q75_day": flow_quantile_day(filled, 0.75),
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def season_day_label(value: float) -> str:
    date = pd.Timestamp("2001-05-01") + pd.Timedelta(days=float(value))
    return date.strftime("%d.%m.")


def plot_half_flow_date_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Annual half-flow date with robust temporal trend."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True, dpi=180)
    cmap = plt.get_cmap("coolwarm")
    summary_rows: list[dict[str, float | str]] = []

    for ax, (column, title, panel) in zip(axes, STATIONS):
        timing = annual_flow_timing(df, column)
        years = timing["year"].to_numpy(dtype=float)
        q50 = timing["q50_day"].to_numpy(dtype=float)
        colours = cmap((years - years.min()) / (years.max() - years.min()))
        ax.vlines(
            years,
            timing["q25_day"],
            timing["q75_day"],
            color="#BDBDBD",
            linewidth=0.75,
            zorder=1,
        )
        ax.scatter(years, q50, c=colours, s=23, edgecolor="white", linewidth=0.35, zorder=3)

        result = sen_mann_kendall(years, q50)
        x_line = np.array([years.min(), years.max()])
        y_line = result["intercept"] + result["slope"] * x_line
        ax.plot(x_line, y_line, color=TREND, linewidth=1.7, zorder=4)
        slope_decade = result["slope"] * 10
        sign = "+" if slope_decade >= 0 else "−"
        slope_value = f"{abs(slope_decade):.1f}".replace(".", ",")
        label = (
            f"Sen-Trend: {sign}{slope_value} Tage pro Dekade; "
            f"{significance_text(result['p'])}; n = {result['n']}"
        )
        ax.text(
            0.98,
            0.96,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.2},
        )
        ax.text(-0.08, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_title(title)
        ax.set_xlabel("Jahr")
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55)
        ax.margins(x=0.03)
        summary_rows.append(
            {
                "station": title,
                "metric": "50% seasonal-flow date",
                "n_years": result["n"],
                "sen_slope_days_per_year": result["slope"],
                "sen_slope_days_per_decade": slope_decade,
                "mann_kendall_p": result["p"],
            }
        )

    axes[0].set_ylabel("Datum des kumulierten Saisonabflusses")
    y_min = min(ax.get_ylim()[0] for ax in axes)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    ticks = np.arange(np.floor(y_min / 15) * 15, np.ceil(y_max / 15) * 15 + 1, 15)
    axes[0].set_yticks(ticks, [season_day_label(v) for v in ticks])
    legend_elements = [
        Line2D([0], [0], color="#BDBDBD", linewidth=1.0, label="25.–75. %"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7B7B7B", markeredgecolor="white", markersize=5, label="50 %"),
        Line2D([0], [0], color=TREND, linewidth=1.7, label="Sen-Trend"),
    ]
    fig.legend(legend_elements, [h.get_label() for h in legend_elements], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    save_figure(fig, "fig14d_half_flow_date_trend_copy")
    return pd.DataFrame(summary_rows)


def main() -> None:
    set_style()
    data = load_data()
    trend_summary = pd.concat(
        [
            plot_annual_magnitude_with_trend(data),
            plot_half_flow_date_trend(data),
            plot_peak_water_diagnostic(data),
            plot_polar_monthly_magnitude(data),
        ],
        ignore_index=True,
    )
    plot_annual_magnitude_site_colours(data)
    plot_absolute_early_recent_hydrographs(data)
    plot_parde_early_recent(data)
    plot_parde_five_year_corrected(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trend_summary.to_csv(OUT_DIR / "fig14_trend_statistics_copy.csv", index=False)
    print(trend_summary.to_string(index=False))
    print(f"\nFigures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
