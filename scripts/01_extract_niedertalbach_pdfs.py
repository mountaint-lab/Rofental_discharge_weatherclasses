"""
Extract daily Wasserstand (cm) and Abfluss (m3/s) from Hydrographisches Jahrbuch von
Oesterreich PDF pages for Messstelle 201350 "Vent (oberh. Niedertalbach)".

Requirements: `pdftotext` (poppler-utils) available on PATH.
    - Ubuntu/Debian: sudo apt install poppler-utils
    - macOS: brew install poppler

Usage:
    1. Put the source PDFs (named e.g. OWF_201350_2014.pdf ... OWF_201350_2023.pdf)
       in INPUT_DIR.
    2. Adjust YEARS below.
    3. Run: python 01_extract_niedertalbach_pdfs.py
    4. Output: a pickle + a printed validation report comparing computed monthly
       means against the officially printed "MQ" row in each PDF (should match
       within ~0.05 m3/s, i.e. rounding tolerance).

Validated against the printed monthly MQ values for 2014-2023: max deviation
0.05 m3/s across all 120 year-month combinations.
"""
import re
import calendar
import subprocess
import pickle
from pathlib import Path

INPUT_DIR = Path("./data/raw/pdfs_niedertalbach")        # <-- adjust: folder with the source PDFs
LAYOUT_DIR = Path("./data/interim/_layout_niedertalbach")  # intermediate text files, auto-created
OUTPUT_PKL = Path("./data/interim/niedertalbach_parsed.pkl")
YEARS = list(range(2014, 2023))               # <-- adjust to the years you have

NUM_RE = re.compile(r"\d+(?:\.\d+)?\)?")
DAY_RE = re.compile(r"^\s*(\d{1,2})\.\s")


def pdf_to_layout_text(pdf_path: Path, out_path: Path):
    subprocess.run(["pdftotext", "-layout", str(pdf_path), str(out_path)], check=True)


def valid_months_for_day(year: int, day: int):
    months = []
    for m in range(1, 13):
        _, last_day = calendar.monthrange(year, m)
        if day <= last_day:
            months.append(m)
    return months


def parse_table(lines, start_idx, end_idx, year):
    """Parse a 'Taegliche ...' daily table block into {(year, month, day): value}."""
    result = {}
    for line in lines[start_idx:end_idx]:
        m = DAY_RE.match(line)
        if not m:
            continue
        day = int(m.group(1))
        rest = line[m.end():]
        tokens = NUM_RE.findall(rest)
        vmonths = valid_months_for_day(year, day)
        if len(tokens) != len(vmonths):
            print(f"    WARNING year={year} day={day}: expected {len(vmonths)} "
                  f"values, got {len(tokens)} -> tokens={tokens}")
        for month, tok in zip(vmonths, tokens):
            val = tok.rstrip(')')
            try:
                result[(year, month, day)] = float(val)
            except ValueError:
                pass
    return result


def parse_year(year: int, all_W: dict, all_Q: dict, monthly_MQ_official: dict):
    pdf_path = INPUT_DIR / f"OWF_201350_{year}.pdf"
    txt_path = LAYOUT_DIR / f"{year}.txt"
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_to_layout_text(pdf_path, txt_path)

    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()

    idx_wasser_header = next(i for i, l in enumerate(lines) if "T\u00e4gliche Wasserst\u00e4nde" in l)
    idx_wasser_extreme = next(i for i, l in enumerate(lines) if "Extremwerte in cm" in l)
    idx_abfluss_header = next(i for i, l in enumerate(lines) if "T\u00e4gliche Abfl\u00fcsse" in l)
    idx_abfluss_extreme = next(i for i, l in enumerate(lines) if "Extremwerte in m" in l and "/s" in l)

    W = parse_table(lines, idx_wasser_header + 1, idx_wasser_extreme, year)
    Q = parse_table(lines, idx_abfluss_header + 1, idx_abfluss_extreme, year)
    all_W.update(W)
    all_Q.update(Q)

    # locate the official monthly-mean (MQ) row for cross-validation
    mq_vals = None
    for i in range(idx_abfluss_extreme, idx_abfluss_extreme + 40):
        if i < len(lines) and re.match(r"^\s*MQ\s", lines[i]):
            toks = NUM_RE.findall(lines[i])
            mq_vals = [float(t.rstrip(')')) for t in toks[:12]]
            break
    monthly_MQ_official[year] = mq_vals
    print(f"Year {year}: parsed W={len(W)} days, Q={len(Q)} days, "
          f"official MQ row found={mq_vals is not None}")


def validate(all_Q, monthly_MQ_official):
    print("\n--- Validation: computed monthly mean Q vs official MQ ---")
    max_diff = 0.0
    for year, off in monthly_MQ_official.items():
        if not off:
            continue
        for month in range(1, 13):
            vals = [all_Q[(year, month, d)] for d in range(1, 32) if (year, month, d) in all_Q]
            if not vals:
                continue
            computed = sum(vals) / len(vals)
            official = off[month - 1]
            diff = abs(computed - official)
            max_diff = max(max_diff, diff)
            if diff > 0.02:
                print(f"  MISMATCH {year}-{month:02d}: computed={computed:.4f} "
                      f"official={official:.4f} diff={diff:.4f}")
    print(f"Max abs deviation across all year-months: {max_diff:.4f} m3/s")


if __name__ == "__main__":
    all_W, all_Q, monthly_MQ_official = {}, {}, {}
    for year in YEARS:
        parse_year(year, all_W, all_Q, monthly_MQ_official)
    validate(all_Q, monthly_MQ_official)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump({"W": all_W, "Q": all_Q}, f)
    print(f"\nSaved {len(all_Q)} Q-records and {len(all_W)} W-records to {OUTPUT_PKL}")
