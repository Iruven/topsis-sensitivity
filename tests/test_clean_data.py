"""Tests for src/clean_data.py.

Expected values follow directly from the unit conventions in data/README.md
(e.g. 1 TB = 1024 GB, 1000 g = 1 kg, System: None/Home/Pro = 0/1/2).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import clean_data
from clean_data import clean_laptops, parse_with_unit

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def make_raw(**overrides: list[str]) -> pd.DataFrame:
    """A small raw table (two laptops, all values as strings, as read by main())."""
    data = {
        "Name": ["  Laptop A  ", "Laptop B"],
        "Price": ["2499", "15690"],
        "System": ["Windows 11 Home", "None"],
        "RAM": ["8GB", "32 GB"],
        "Memory": ["512 GB", "1TB"],
        "CPU_name": ["CPU A", "CPU B"],
        "CPU_score": ["15445", "57942"],
        "GPU_name": ["GPU A", "GPU B"],
        "GPU_score": ["3197", "26356"],
        "Display": ["60hz", "300Hz"],
        "Battery": ["41Wh", "90wh"],
        "Weight": ["1.66kg", "2650g"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# --- parse_with_unit ------------------------------------------------------------


@pytest.mark.parametrize(
    "value, column, expected",
    [
        ("16GB", "RAM", 16.0),
        ("512 GB", "Memory", 512.0),
        ("1TB", "Memory", 1024.0),
        ("2 tb", "Memory", 2048.0),
        ("165hz", "Display", 165.0),
        ("80Wh", "Battery", 80.0),
        ("1.87kg", "Weight", 1.87),
        ("1,87kg", "Weight", 1.87),
        ("1870g", "Weight", 1.87),
        ("  16GB  ", "RAM", 16.0),
    ],
    ids=[
        "gb", "gb_with_space", "tb_to_gb", "tb_lowercase", "hz", "wh",
        "kg", "decimal_comma", "grams_to_kg", "surrounding_spaces",
    ],
)
def test_parse_with_unit(value, column, expected):
    assert parse_with_unit(value, column) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value, column, message",
    [
        ("16MB", "RAM", "unknown unit"),
        ("1TB", "RAM", "unknown unit"),  # TB is accepted for Memory only
        ("16", "RAM", "cannot parse"),
        ("GB", "RAM", "cannot parse"),
        ("16 GB extra", "RAM", "cannot parse"),
        ("", "Weight", "cannot parse"),
    ],
    ids=["wrong_unit", "unit_of_other_column", "no_unit", "no_number",
         "trailing_text", "empty"],
)
def test_parse_with_unit_rejects_bad_values(value, column, message):
    with pytest.raises(ValueError, match=message):
        parse_with_unit(value, column)


# --- clean_laptops --------------------------------------------------------------


def test_clean_laptops_converts_all_columns():
    clean = clean_laptops(make_raw())
    assert clean["System"].tolist() == [1, 0]
    assert clean["RAM"].tolist() == [8, 32]
    assert clean["Memory"].tolist() == [512, 1024]
    assert clean["Display"].tolist() == [60, 300]
    assert clean["Battery"].tolist() == [41, 90]
    assert clean["Weight"].tolist() == pytest.approx([1.66, 2.65])
    assert clean["Price"].tolist() == [2499, 15690]
    assert clean["CPU_score"].tolist() == [15445, 57942]
    assert clean["GPU_score"].tolist() == [3197, 26356]


def test_clean_laptops_column_types():
    clean = clean_laptops(make_raw())
    for col in ("System", "RAM", "Memory", "Display", "Battery", "Price",
                "CPU_score", "GPU_score"):
        assert pd.api.types.is_integer_dtype(clean[col]), col
    assert pd.api.types.is_float_dtype(clean["Weight"])


def test_clean_laptops_adds_ids_as_index():
    clean = clean_laptops(make_raw())
    assert clean.index.tolist() == ["L01", "L02"]
    assert clean.index.name == "ID"


def test_clean_laptops_strips_text():
    assert clean_laptops(make_raw())["Name"].tolist() == ["Laptop A", "Laptop B"]


def test_clean_laptops_all_system_codes():
    raw = make_raw(System=["Windows 11 Pro", " Windows 11 Home "])
    assert clean_laptops(raw)["System"].tolist() == [2, 1]


def test_clean_laptops_does_not_modify_input():
    raw = make_raw()
    before = raw.copy()
    clean_laptops(raw)
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"System": ["Windows 10", "None"]}, "unknown value"),
        ({"RAM": ["8GB", "lots"]}, "cannot parse"),
        ({"Weight": ["1.66kg", "2.65lb"]}, "unknown unit"),
        ({"Price": ["2499", "n/a"]}, "Unable to parse"),
    ],
    ids=["unknown_system", "unparsable_ram", "unknown_weight_unit", "non_numeric_price"],
)
def test_clean_laptops_rejects_bad_data(overrides, message):
    with pytest.raises(ValueError, match=message):
        clean_laptops(make_raw(**overrides))


# --- main() and the real data files ------------------------------------------------


def test_main_writes_clean_csv_and_keeps_none_system(tmp_path, monkeypatch):
    # The text "None" must be read as a System value, not as a missing value.
    raw_path = tmp_path / "raw.csv"
    clean_path = tmp_path / "clean.csv"
    make_raw().to_csv(raw_path, index=False)
    monkeypatch.setattr(clean_data, "RAW_PATH", raw_path)
    monkeypatch.setattr(clean_data, "CLEAN_PATH", clean_path)

    clean_data.main()

    written = pd.read_csv(clean_path, index_col="ID")
    assert written.index.tolist() == ["L01", "L02"]
    assert written["System"].tolist() == [1, 0]
    assert written["Memory"].tolist() == [512, 1024]


def test_committed_clean_csv_matches_raw_csv():
    # Fails if data/laptops.csv was edited without re-running clean_data.py.
    raw = pd.read_csv(DATA_DIR / "laptops.csv", dtype=str, keep_default_na=False)
    expected = clean_laptops(raw)
    committed = pd.read_csv(DATA_DIR / "laptops_clean.csv", index_col="ID")
    pd.testing.assert_frame_equal(committed, expected, check_dtype=False)
