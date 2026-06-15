import datetime
import math

import numpy as np
import xarray as xr

import erlab.utils.formatting
from erlab.utils.formatting import format_darr_html, format_html_table, format_value


def test_format_html_table_basic() -> None:
    rows = [["Header1", "Header2"], ["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
    expected = (
        "<div><style>\n.erlab-table td,\n.erlab-table th {\n    text-align: left;\n}\n\n"  # noqa: E501
        '.erlab-table th {\n    font-weight: bold;\n}\n</style><table class="erlab-table">'  # noqa: E501
        "<thead><tr><th>Header1</th><th>Header2</th></tr></thead><tbody><tr><td>Row1Col1</td>"
        "<td>Row1Col2</td></tr><tr><td>Row2Col1</td><td>Row2Col2</td></tr></tbody></table></div>"
    )
    assert format_html_table(rows, header_rows=1) == expected


def test_format_html_table_no_headers() -> None:
    rows = [["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
    expected = (
        "<div><style>\n.erlab-table td,\n.erlab-table th {\n    text-align: left;\n}\n\n"  # noqa: E501
        '.erlab-table th {\n    font-weight: bold;\n}\n</style><table class="erlab-table">'  # noqa: E501
        "<tr><td>Row1Col1</td><td>Row1Col2</td></tr><tr><td>Row2Col1</td><td>Row2Col2</td></tr>"
        "</table></div>"
    )
    assert format_html_table(rows) == expected


def test_format_html_table_header_cols() -> None:
    rows = [["Header1", "Header2"], ["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
    expected = (
        "<div><style>\n.erlab-table td,\n.erlab-table th {\n    text-align: left;\n}\n\n"  # noqa: E501
        '.erlab-table th {\n    font-weight: bold;\n}\n</style><table class="erlab-table">'  # noqa: E501
        "<thead><tr><th>Header1</th><th>Header2</th></tr></thead><tbody><tr><th>Row1Col1</th>"
        "<td>Row1Col2</td></tr><tr><th>Row2Col1</th><td>Row2Col2</td></tr></tbody></table></div>"
    )
    assert format_html_table(rows, header_rows=1, header_cols=1) == expected


def test_format_html_table_no_thead() -> None:
    rows = [["Header1", "Header2"], ["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
    expected = (
        "<div><style>\n.erlab-table td,\n.erlab-table th {\n    text-align: left;\n}\n\n"  # noqa: E501
        '.erlab-table th {\n    font-weight: bold;\n}\n</style><table class="erlab-table">'  # noqa: E501
        "<tr><th>Header1</th><th>Header2</th></tr><tr><td>Row1Col1</td><td>Row1Col2</td></tr>"
        "<tr><td>Row2Col1</td><td>Row2Col2</td></tr></table></div>"
    )
    assert format_html_table(rows, header_rows=1, use_thead=False) == expected


def test_format_darr_html_can_skip_coordinate_value_loading(monkeypatch) -> None:
    def fail_if_values_are_loaded(_value):
        raise AssertionError("metadata-only formatting should not load values")

    monkeypatch.setattr(
        erlab.utils.formatting, "_format_array_values", fail_if_values_are_loaded
    )

    data = xr.DataArray(
        np.zeros((2, 3)),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(3)},
    )

    html = format_darr_html(data, load_values=False)

    assert "int64 [2]" in html
    assert "int64 [3]" in html


def test_format_darr_html_falls_back_when_coordinate_formatting_fails(
    monkeypatch,
) -> None:
    def fail_array_format(_value):
        raise RuntimeError("failed to format coordinate values")

    monkeypatch.setattr(
        erlab.utils.formatting, "_format_array_values", fail_array_format
    )

    data = xr.DataArray(
        np.zeros(3),
        dims=("x",),
        coords={"x": np.arange(3)},
    )

    html = format_darr_html(data)

    assert '["0",  ... , "2"]' in html


def test_format_coord_values_falls_back_when_coordinate_values_fail() -> None:
    class BrokenCoord:
        dtype = np.dtype("float64")
        shape = (2,)

        @property
        def values(self):
            raise RuntimeError("failed to load coordinate values")

    assert (
        erlab.utils.formatting._format_coord_values(BrokenCoord(), load_values=True)
        == "float64 [2]"
    )


def test_format_value_numpy_array_len2() -> None:
    val = np.array([0.12, 0.25])
    expected = "[0.12, 0.25]"
    assert format_value(val) == expected


def test_format_value_numpy_array_evenly_spaced() -> None:
    val = np.array([0.1, 0.15, 0.2])
    expected = "0.1→0.2 (0.05, 3)"
    assert format_value(val) == expected


def test_format_value_numpy_array_monotonic() -> None:
    val = np.array([1.0, 2.0, 2.1])
    expected = "1→2.1 (3)"
    assert format_value(val) == expected


def test_format_value_numpy_array_non_monotonic() -> None:
    val = np.array([1.0, 2.1, 2.0])
    expected = "1~2.1 (3)"
    assert format_value(val) == expected


def test_format_value_list() -> None:
    val = [1, 1, 2, 2, 2, 3, 3, 3, 3]
    expected = "[1]×2, [2]×3, [3]×4"
    assert format_value(val) == expected


def test_format_value_float() -> None:
    val = math.pi
    expected = "3.1416"
    assert format_value(val) == expected


def test_format_value_integer_float() -> None:
    val = 42.0
    expected = "42"
    assert format_value(val) == expected


def test_format_value_integer() -> None:
    val = 42
    expected = "42"
    assert format_value(val) == expected


def test_format_value_date() -> None:
    val = datetime.date(2024, 1, 1)
    expected = "2024-01-01"
    assert format_value(val) == expected


def test_format_value_datetime() -> None:
    val = datetime.datetime(2024, 1, 1, 12, 0, 0, 0)
    expected = "2024-01-01 12:00:00"
    assert format_value(val) == expected


def test_format_value_datetime64() -> None:
    val = np.datetime64("2024-01-01T12:00:00")
    expected = "2024-01-01 12:00:00"
    assert format_value(val) == expected


def test_format_value_datetime64_array() -> None:
    val = np.array(
        [
            np.datetime64("2024-01-01T12:00:00"),
            np.datetime64("2024-01-02T12:00:00"),
        ]
    )
    expected = "[2024-01-01 12:00:00, 2024-01-02 12:00:00]"
    assert format_value(val) == expected


def test_format_value_pandas_timestamp() -> None:
    import pandas as pd

    val = pd.Timestamp("2024-01-01 12:00:00")
    expected = "2024-01-01 12:00:00"
    assert format_value(val) == expected
