"""
Extract daily Abfluss (m3/s) from Hydrographisches Jahrbuch von Oesterreich PDF
pages for Messstelle 202689 "Vernagt / Vernagtbach".

Unlike Niedertalbach, this station is "eingewintert" (shut down) for part of
each year, so several month-columns in the printed table are entirely blank
for months with no season overlap. A simple "count tokens per row" parser
breaks in that case (values shift into the wrong month). This script instead
calibrates each month-column's approximate character position from the table
header, then assigns every printed value to its nearest column position -
which works regardless of which months happen to be blank that year.

Requirements: `pdftotext` (poppler-utils) available on PATH.

Validated against the printed monthly MQ values for 2014-2019: max deviation
0.005 m3/s (rounding tolerance).
"""
import re
import calendar
import subprocess
import pickle
from pathlib import Path

INPUT_DIR = Path("./data/raw/pdfs_vernagtbach")        # <-- adjust: folder with the source PDFs
LAYOUT_DIR = Path("./data/interim/_layout_vernagtbach")  # intermediate text files, auto-created
OUTPUT_PKL = Path("./data/interim/vernagtbach_parsed.pkl")
YEARS = list(range(2014, 2020))                # <-- adjust to the years you have

MONTH_ORDER = ["Jan", "Feb", "Mrz", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
DAY_RE = re.compile(r"^\s*(\d{1,2})\.\s")


def pdf_to_layout_text(pdf_path: Path, out_path: Path):
    subprocess.run(["pdftotext", "-layout", str(pdf_path), str(out_path)], check=True)


def get_header_positions(lines):
    """Character end-position of each month label in the 'Mon Jan Feb ... Dez' row."""
    for line in lines:
        if re.search(r"\bMon\b", line) and "Jan" in line and "Dez" in line:
            positions = {}
            for m in re.finditer(r"Jan|Feb|Mrz|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez", line):
                positions[m.group()] = m.end()
            if len(positions) == 12:
                return [positions[mo] for mo in MONTH_ORDER]
    raise ValueError("month header row not found")


def cluster_positions(all_end_positions, tol=4):
    """Group nearby x-positions (character columns) into one cluster per data column."""
    pts = sorted(set(all_end_positions))
    clusters = []
    for p in pts:
        if clusters and p - clusters[-1][-1] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [sum(c) / len(c) for c in clusters]


def best_offset(header_pos, clusters):
    """Find the header->data-column offset (in characters) that best aligns clusters."""
    best = None
    for offset in range(-2, 20):
        predicted = [h + offset for h in header_pos]
        cost = sum(min(abs(c - p) for p in predicted) for c in clusters)
        if best is None or cost < best[0]:
            best = (cost, offset)
    return best[1]


def parse_year(year: int, all_Q: dict, monthly_MQ_official: dict):
    pdf_path = INPUT_DIR / f"OWF_202689_{year}.pdf"
    txt_path = LAYOUT_DIR / f"{year}.txt"
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_to_layout_text(pdf_path, txt_path)

    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()

    idx_header_table = next(i for i, l in enumerate(lines) if "T\u00e4gliche Abfl\u00fcsse" in l)
    idx_extreme = next(i for i, l in enumerate(lines) if re.search(r"Extremwerte in m.?/s", l))
    header_pos = get_header_positions(lines[:idx_header_table + 1])

    day_lines = []
    for line in lines[idx_header_table + 1:idx_extreme]:
        m = DAY_RE.match(line)
        if m:
            day_lines.append((int(m.group(1)), line))

    all_ends = [mm.end() for _, line in day_lines for mm in NUM_RE.finditer(line) if mm.start() >= 8]
    clusters = cluster_positions(all_ends, tol=4)
    offset = best_offset(header_pos, clusters)
    predicted = [h + offset for h in header_pos]

    unmatched = 0
    for day, line in day_lines:
        for mm in NUM_RE.finditer(line):
            if mm.start() < 8:
                continue
            end = mm.end()
            diffs = sorted((abs(end - p), i) for i, p in enumerate(predicted))
            best_diff, month_idx = diffs[0]
            if best_diff > 5:
                unmatched += 1
                continue
            month = month_idx + 1
            _, last = calendar.monthrange(year, month)
            if day > last:
                unmatched += 1
                continue
            all_Q[(year, month, day)] = float(mm.group())

    active_months = sorted({k[1] for k in all_Q if k[0] == year})
    mq_vals = None
    for i in range(idx_extreme, min(idx_extreme + 30, len(lines))):
        if re.match(r"^\s*MQ\s", lines[i]):
            toks = NUM_RE.findall(lines[i])
            mq_vals = [float(t) for t in toks[:len(active_months)]]
            break
    monthly_MQ_official[year] = (active_months, mq_vals)

    print(f"Year {year}: months with data={active_months}, unmatched tokens={unmatched}")


def validate(all_Q, monthly_MQ_official):
    print("\n--- Validation: computed monthly mean Q vs official MQ ---")
    max_diff = 0.0
    for year, (active_months, mq_vals) in monthly_MQ_official.items():
        if not mq_vals:
            continue
        for month, official in zip(active_months, mq_vals):
            vals = [all_Q[(year, month, d)] for d in range(1, 32) if (year, month, d) in all_Q]
            if not vals:
                continue
            computed = sum(vals) / len(vals)
            diff = abs(computed - official)
            max_diff = max(max_diff, diff)
            if diff > 0.01:
                print(f"  MISMATCH {year}-{month:02d}: computed={computed:.4f} official={official:.4f}")
    print(f"Max abs deviation across all year-months: {max_diff:.4f} m3/s")


if __name__ == "__main__":
    all_Q, monthly_MQ_official = {}, {}
    for year in YEARS:
        parse_year(year, all_Q, monthly_MQ_official)
    validate(all_Q, monthly_MQ_official)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump({"Q": all_Q}, f)
    print(f"\nSaved {len(all_Q)} Q-records to {OUTPUT_PKL}")
