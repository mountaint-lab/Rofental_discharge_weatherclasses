"""Create HEFEX-III levelstick figures from a locally stored trimmed workbook.

The source data stay outside version control in ``private_data/``. The first
60 minutes after deployment were already excluded in the input workbook, so
sensor-settling values cannot re-enter the figures.
"""

from datetime import timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pandas as pd


BASE = Path(__file__).resolve().parent
INPUT = BASE / "private_data" / "HEFEX_levelsticks_trimmed.xlsx"
OUTPUT = BASE / "figures"
SITES = {
    "Rofenache_deep": "Rofenache (deep)",
    "Rofenache_low": "Rofenache (low)",
    "supraglacial_stream": "supraglazialer Bach (Q2)",
}
SITE_COLORS = {
    "Rofenache_deep": "#24577A",
    "Rofenache_low": "#C97900",
    "supraglacial_stream": "#6C47A6",
}
# Soft chronological scale: blue (06.08.) through warm sand to orange-red (10.08.).
TIME_COLORS = ["#4B78A8", "#74A8C8", "#D8D3BC", "#E8A35B", "#C95442"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Use Arial on Windows; Pillow's built-in font is a portable fallback."""
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def blend_color(fraction: float) -> tuple[int, int, int]:
    """Interpolate the custom chronological palette at 0 <= fraction <= 1."""
    fraction = max(0.0, min(1.0, fraction))
    position = fraction * (len(TIME_COLORS) - 1)
    index = min(int(position), len(TIME_COLORS) - 2)
    remainder = position - index
    first, second = rgb(TIME_COLORS[index]), rgb(TIME_COLORS[index + 1])
    return tuple(round(first[i] + (second[i] - first[i]) * remainder) for i in range(3))


def read_trimmed_data() -> dict[str, pd.DataFrame]:
    """Read the three processed station series and standardise column names."""
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing private input workbook: {INPUT}\n"
            "Copy HEFEX_levelsticks_trimmed.xlsx into private_data/ before running the script."
        )
    frames = {}
    for sheet in SITES:
        frame = pd.read_excel(INPUT, sheet_name=sheet).rename(
            columns={
                "Time [Europe/Rome]": "time",
                "Pressure [hPa]": "pressure_hpa",
                "Temperature [°C]": "temperature_c",
            }
        )
        frame["time"] = pd.to_datetime(frame["time"])
        frames[sheet] = frame.sort_values("time").reset_index(drop=True)
    return frames


def mapped(value: float, src_min: float, src_max: float, dest_min: float, dest_max: float) -> float:
    return dest_min + (value - src_min) * (dest_max - dest_min) / (src_max - src_min)


def text_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill="black") -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def vertical_label(image: Image.Image, xy: tuple[float, float], text: str, fnt) -> None:
    """Paste a centred, readable vertical axis label."""
    probe = ImageDraw.Draw(image)
    box = probe.multiline_textbbox((0, 0), text, font=fnt, spacing=6, align="center")
    label = Image.new("RGBA", (round(box[2] - box[0] + 12), round(box[3] - box[1] + 12)), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.multiline_text((6, 6), text, font=fnt, fill="black", spacing=6, align="center")
    label = label.rotate(90, expand=True)
    image.paste(label, (round(xy[0] - label.width / 2), round(xy[1] - label.height / 2)), label)


def draw_axes(draw, bounds, y_min, y_max, y_ticks, x_labels=True, y_labels=True, x_ticks=True) -> None:
    left, top, right, bottom = bounds
    draw.line((left, top, left, bottom), fill="black", width=3)
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    for tick in y_ticks:
        y = mapped(tick, y_min, y_max, bottom, top)
        draw.line((left - 12, y, left, y), fill="black", width=3)
        if y_labels:
            label = f"{tick:g}"
            box = draw.textbbox((0, 0), label, font=font(30))
            draw.text((left - 25 - (box[2] - box[0]), y - (box[3] - box[1]) / 2), label, font=font(30), fill="black")
    if x_ticks:
        for tick in (0, 6, 12, 18, 24):
            x = mapped(tick, 0, 24, left, right)
            draw.line((x, bottom, x, bottom + 12), fill="black", width=3)
            if x_labels:
                text_centered(draw, (x, bottom + 46), str(tick), font(28))


def plot_series(draw, frame, column, bounds, x_min, x_max, y_min, y_max, color, width=4) -> None:
    left, top, right, bottom = bounds
    points = []
    for row in frame.itertuples(index=False):
        x = mapped(row.time.timestamp(), x_min.timestamp(), x_max.timestamp(), left, right)
        y = mapped(getattr(row, column), y_min, y_max, bottom, top)
        points.append((x, y))
    draw.line(points, fill=rgb(color), width=width, joint="curve")


def save_campaign_timeseries(data: dict[str, pd.DataFrame]) -> None:
    """Create Fig. 12 with the initial sensor-settling values excluded."""
    image = Image.new("RGB", (3750, 2400), "white")
    draw = ImageDraw.Draw(image)
    time_start = min(frame.time.min() for frame in data.values())
    time_end = max(frame.time.max() for frame in data.values())
    pressure_bounds = (270, 210, 3690, 1180)
    temperature_bounds = (270, 1370, 3690, 2250)

    text_centered(draw, (1980, 75), "HEFEX-III Levelstick-Kampagne, Hintereisferner (06.–10.08.2025)", font(49))
    draw_axes(draw, pressure_bounds, -3, 38, list(range(0, 36, 5)), x_labels=False, x_ticks=False)
    draw_axes(draw, temperature_bounds, -0.35, 1.5, [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], x_ticks=False)
    for bounds, y_min, y_max in ((pressure_bounds, -3, 38), (temperature_bounds, -0.35, 1.5)):
        zero = mapped(0, y_min, y_max, bounds[3], bounds[1])
        draw.line((bounds[0], zero, bounds[2], zero), fill="#B8B8B8", width=2)
    for sheet in SITES:
        plot_series(draw, data[sheet], "pressure_hpa", pressure_bounds, time_start, time_end, -3, 38, SITE_COLORS[sheet])
        plot_series(draw, data[sheet], "temperature_c", temperature_bounds, time_start, time_end, -0.35, 1.5, SITE_COLORS[sheet])

    draw.text((285, 250), "a", font=font(45, True), fill="black")
    draw.text((285, 1410), "b", font=font(45, True), fill="black")
    vertical_label(image, (75, 695), "Druck [hPa]\n(Wasserstand-Proxy)", font(33))
    vertical_label(image, (75, 1770), "Wassertemperatur [°C]", font(33))
    for day in pd.date_range(time_start.normalize() + timedelta(days=1), time_end.normalize(), freq="D"):
        x = mapped(day.timestamp(), time_start.timestamp(), time_end.timestamp(), temperature_bounds[0], temperature_bounds[2])
        draw.line((x, temperature_bounds[3], x, temperature_bounds[3] + 12), fill="black", width=3)
        text_centered(draw, (x, temperature_bounds[3] + 68), day.strftime("%d.%m."), font(29))
    text_centered(draw, (1980, 2360), "Datum (06.–10.08.2025)", font(34))

    legend_x, legend_y = 1830, 255
    for sheet, label in SITES.items():
        draw.line((legend_x, legend_y + 23, legend_x + 85, legend_y + 23), fill=rgb(SITE_COLORS[sheet]), width=5)
        draw.text((legend_x + 115, legend_y), label, font=font(30), fill="black")
        legend_x += 570 if sheet != "Rofenache_low" else 520
    image.save(OUTPUT / "fig12_hefex_levelsticks.png", dpi=(300, 300))


def save_diurnal_comparison(data: dict[str, pd.DataFrame]) -> None:
    """Create Fig. 13 with a common y-axis and continuous blue-to-red colours."""
    image = Image.new("RGB", (3960, 1950), "white")
    draw = ImageDraw.Draw(image)
    text_centered(draw, (1980, 75), "Tagesgang je Standort – gemeinsamer Maßstab", font(54))
    start = min(frame.time.min() for frame in data.values())
    end = max(frame.time.max() for frame in data.values())
    panels = [(210, 210, 1260, 1360), (1460, 210, 2510, 1360), (2710, 210, 3760, 1360)]
    y_min, y_max = -2, 38
    y_ticks = [0, 5, 10, 15, 20, 25, 30, 35]

    for panel, (sheet, label) in zip(panels, SITES.items()):
        draw_axes(draw, panel, y_min, y_max, y_ticks)
        zero = mapped(0, y_min, y_max, panel[3], panel[1])
        draw.line((panel[0], zero, panel[2], zero), fill="#B8B8B8", width=2)
        text_centered(draw, ((panel[0] + panel[2]) / 2, 155), label, font(39))
        text_centered(draw, ((panel[0] + panel[2]) / 2, 1480), "Uhrzeit [h]", font(33))
        for _, day in data[sheet].groupby(data[sheet].time.dt.normalize(), sort=True):
            previous = None
            for row in day.sort_values("time").itertuples(index=False):
                hour = row.time.hour + row.time.minute / 60 + row.time.second / 3600
                point = (mapped(hour, 0, 24, panel[0], panel[2]), mapped(row.pressure_hpa, y_min, y_max, panel[3], panel[1]))
                if previous is not None:
                    fraction = (row.time - start).total_seconds() / (end - start).total_seconds()
                    draw.line((previous, point), fill=blend_color(fraction), width=5)
                previous = point

    vertical_label(image, (70, 770), "Druck [hPa] (Wasserstand-Proxy)", font(32))
    bar_left, bar_right, bar_top, bar_bottom = 820, 3140, 1640, 1700
    for x in range(bar_left, bar_right):
        draw.line((x, bar_top, x, bar_bottom), fill=blend_color((x - bar_left) / (bar_right - bar_left - 1)), width=1)
    draw.rectangle((bar_left, bar_top, bar_right, bar_bottom), outline="black", width=3)
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    for index, day in enumerate(dates):
        x = bar_left + index / (len(dates) - 1) * (bar_right - bar_left)
        draw.line((x, bar_bottom, x, bar_bottom + 16), fill="black", width=3)
        text_centered(draw, (x, bar_bottom + 55), day.strftime("%d.%m."), font(29))
    text_centered(draw, (1980, 1840), "Kampagnenzeit (06.–10.08.2025), kontinuierlicher Farbverlauf", font(33))
    image.save(OUTPUT / "fig13_hefex_levelsticks_diurnal.png", dpi=(300, 300))


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    frames = read_trimmed_data()
    save_campaign_timeseries(frames)
    save_diurnal_comparison(frames)
