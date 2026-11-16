"""Convert the raw laptop CSV (values with units) into a numeric CSV.

Raw:   data/laptops.csv        e.g. "16GB", "512 GB", "165hz", "80Wh", "1.87kg"
Clean: data/laptops_clean.csv  plain numbers, units fixed per column (see UNITS)

Run from anywhere:  python src/clean_data.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Paths are resolved relative to the project root, not the working directory.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "laptops.csv"
CLEAN_PATH = DATA_DIR / "laptops_clean.csv"

# Unit of every numeric column in the clean file.
UNITS: dict[str, str] = {
    "Price": "PLN",
    "System": "ordinal code (0 = None, 1 = Windows 11 Home, 2 = Windows 11 Pro)",
    "RAM": "GB",
    "Memory": "GB",
    "CPU_score": "PassMark CPU Mark (multithread)",
    "GPU_score": "PassMark Average G3D Mark",
    "Display": "Hz",
    "Battery": "Wh",
    "Weight": "kg",
}

SYSTEM_CODES: dict[str, int] = {"None": 0, "Windows 11 Home": 1, "Windows 11 Pro": 2}

# Accepted unit suffixes per column and the factor converting them to UNITS.
_UNIT_FACTORS: dict[str, dict[str, float]] = {
    "RAM": {"gb": 1.0},
    "Memory": {"gb": 1.0, "tb": 1024.0},
    "Display": {"hz": 1.0},
    "Battery": {"wh": 1.0},
    "Weight": {"kg": 1.0, "g": 0.001},
}

# Short labels for charts and tables, keyed by the full product name.
# Kept in code (not in the raw CSV) so the raw data stays exactly as collected.
SHORT_NAMES: dict[str, str] = {
    "Laptop HP 255R G10 Ryzen 5 7535U / 8 GB / 512 GB / W11 (B9YQ0ET)":
        "HP 255R G10",
    "Laptop Lenovo ThinkPad E16 G3 Ultra 7 258V / 32 GB / 1 TB / W11 Pro (22AY0089PB)":
        "Lenovo ThinkPad E16 G3",
    "Laptop Lenovo Legion 5 15AHP11 Ryzen 7 250 / 16 GB / 512 GB / RTX 5070 / 165 Hz / Windows 11 Pro (83Q7003JPB)":
        "Lenovo Legion 5 RTX 5070",
    "Laptop Lenovo Legion Pro 5 16ADR10 Ryzen 9 8945HX / 32 GB / 1 TB / RTX 5070 / 240 Hz / Windows 11 Pro (83LT001WPB)":
        "Lenovo Legion Pro 5 RTX 5070",
    "Laptop Asus ROG Strix G16 G615 Ultra 9 290HX Plus / 32 GB / 1 TB / RTX 5080 / 300 Hz (G615LW-U9R321)":
        "Asus ROG Strix G16 RTX 5080",
    "Laptop Gigabyte Gaming A18 3VH Ryzen 7 260 / 16 GB / 512 GB / RTX 5060 / 165 Hz / Windows 11 Home (3VHK3EEC93SD)":
        "Gigabyte Gaming A18 RTX 5060",
    "Laptop MSI Katana 15 HX B14WFK-020XPL Core i7-14650HX / 16 GB / 512 GB / RTX 5060 / 165 Hz / Windows 11 Home":
        "MSI Katana 15 HX RTX 5060",
    "Laptop Lenovo Legion 5 15AHP11 Ryzen 7 250 / 16 GB / 512 GB / RTX 5050 / 165 Hz / Windows 11 Pro (83Q7002FPB)":
        "Lenovo Legion 5 RTX 5050",
    "Laptop Asus Vivobook 16 X1607CA Core Ultra 5 225H / 16 GB / 512 GB / 60 Hz / Windows 11 Home (X1607CA-MB058)":
        "Asus Vivobook 16",
    "Laptop Dell Pro 15 Essential PV15250 i5-1334U / 16 GB / 512 GB / W11 Pro / 120 Hz (PV15250_RPLU_005_P)":
        "Dell Pro 15 Essential",
    "Laptop Lenovo LOQ 15AHP9 Ryzen 7 7445HS / 16 GB / 512 GB / RTX 4050 / 144 Hz / Windows 11 Pro (83DX00EVPB)":
        "Lenovo LOQ 15 RTX 4050",
}

_VALUE_WITH_UNIT = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z]+)\s*$")


def parse_with_unit(value: str, column: str) -> float:
    """Parse a string such as ``"512 GB"`` into a number in the column's unit.

    Raises:
        ValueError: if the format or the unit is not recognised.
    """
    match = _VALUE_WITH_UNIT.match(str(value))
    if match is None:
        raise ValueError(f"Column {column}: cannot parse value {value!r}.")
    number, unit = match.groups()
    factors = _UNIT_FACTORS[column]
    if unit.lower() not in factors:
        raise ValueError(
            f"Column {column}: unknown unit {unit!r} in {value!r}; "
            f"expected one of {sorted(factors)}."
        )
    return float(number.replace(",", ".")) * factors[unit.lower()]


def clean_laptops(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw laptop table.

    - adds a short ``ID`` (L01, L02, ...) used as the index,
    - adds ``Short_name`` (from SHORT_NAMES) right after ``Name``,
    - strips whitespace from text columns,
    - encodes ``System`` as 0/1/2,
    - removes units from RAM, Memory, Display, Battery and Weight.

    The raw DataFrame is not modified.
    """
    df = raw.copy()
    for col in ("Name", "System", "CPU_name", "GPU_name"):
        df[col] = df[col].astype(str).str.strip()

    missing_short = [n for n in df["Name"] if n not in SHORT_NAMES]
    if missing_short:
        raise ValueError(f"No short name defined for: {missing_short}.")
    df.insert(df.columns.get_loc("Name") + 1, "Short_name", df["Name"].map(SHORT_NAMES))

    unknown = sorted(set(df["System"]) - set(SYSTEM_CODES))
    if unknown:
        raise ValueError(f"Column System: unknown value(s) {unknown}.")
    df["System"] = df["System"].map(SYSTEM_CODES)

    for col in _UNIT_FACTORS:
        df[col] = [parse_with_unit(v, col) for v in df[col]]
    for col in ("Price", "CPU_score", "GPU_score"):
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Whole numbers stay integers in the CSV; Weight stays a float.
    for col in ("RAM", "Memory", "Display", "Battery"):
        df[col] = df[col].round().astype(int)

    df.index = pd.Index([f"L{i:02d}" for i in range(1, len(df) + 1)], name="ID")
    return df


def main() -> None:
    """Read the raw CSV, clean it and write the clean CSV."""
    # keep_default_na=False: otherwise pandas reads the text "None" as a missing value.
    raw = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)
    clean = clean_laptops(raw)
    clean.to_csv(CLEAN_PATH, lineterminator="\n")
    print(f"Wrote {len(clean)} rows to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
