"""Utilities related to representing data in a human-readable format."""

from __future__ import annotations

__all__ = ["format_darr_html", "format_html_table", "format_nbytes", "format_value"]

import datetime
import itertools
import typing

import numpy as np
import numpy.typing as npt

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    import xarray as xr

STYLE_SHEET = """
.erlab-table td,
.erlab-table th {
    text-align: left;
}

.erlab-table th {
    font-weight: bold;
}
"""

_DEFAULT_ACCENT_COLOR: str = "#0078d7"
"""Accent color in HTML strings."""


def format_html_table(
    rows: Iterable[Iterable[str]],
    header_cols: int = 0,
    header_rows: int = 0,
    use_thead: bool = True,
) -> str:
    """Create a simple HTML table from a dictionary."""
    table = "<div>"
    table += f"<style>{STYLE_SHEET}</style>"
    table += '<table class="erlab-table">'
    if header_rows > 0 and use_thead:
        table += "<thead>"
    for i, row in enumerate(rows):
        table += "<tr>"
        for j, cell in enumerate(row):
            tag = "th" if i < header_rows or j < header_cols else "td"
            table += f"<{tag}>{cell}</{tag}>"
        table += "</tr>"
        if i == header_rows - 1 and use_thead:
            table += "</thead><tbody>"
    if header_rows > 0 and use_thead:
        table += "</tbody>"
    table += "</table>"
    table += "</div>"
    return table


def format_nbytes(value: float | str, fmt: str = "%.1f", sep: str = " ") -> str:
    """Format the given number of bytes in a human-readable format.

    Parameters
    ----------
    value
        The number of bytes to format.
    fmt
        The format string to use when formatting the number of bytes.
    sep
        The separator to use between the formatted number of bytes and the unit.
    """
    suffixes = ("kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB")

    base = 1000
    nbytes = float(value) if isinstance(value, str) else value

    abs_bytes = abs(nbytes)

    if abs_bytes == 1:
        return f"{nbytes}{sep}Byte"

    if abs_bytes < base:
        return f"{nbytes}{sep}Bytes"

    for i, s in enumerate(suffixes, 2):
        unit = base**i
        if abs_bytes < unit:
            suffix = s
            break

    ret: str = (fmt % (base * (nbytes / unit))) + sep + suffix
    return ret


def format_value(
    val: object, precision: int = 4, use_unicode_minus: bool = False
) -> str:
    """Format the given value based on its type.

    This method is used when formatting the cells of the summary dataframe.

    Parameters
    ----------
    val
        The value to be formatted.
    precision
        The number of decimal places to use when formatting floating-point numbers. If
        the magnitude of the value is smaller than the precision, the number will be
        printed in scientific notation.
    use_unicode_minus
        Whether to replace the  Unicode hyphen-minus sign "-" (U+002D) with the
        better-looking Unicode minus sign "−" (U+2212) in the formatted value.

    Returns
    -------
    str or object
        The formatted value.

    Note
    ----
    This function formats the given value based on its type. It supports formatting for
    various types including numpy arrays, lists of strings, floating-point numbers,
    integers, and datetime objects.

    - For numpy arrays:
        - If the array has a size of 1, the value is recursively formatted using
          `format_value(val.item())`.
        - If the array can be squeezed to a 1-dimensional array, the following are
          applied.

            - If the array is evenly spaced, the start, end, step, and length values are
              formatted and returned as a string in the format "start→end (step,
              length)".
            - If the array is monotonic increasing or decreasing but not evenly spaced,
              the start, end, and length values are formatted and returned as a string
              in the format "start→end (length)".
            - If all elements are equal, the value is recursively formatted using
              `format_value(val[0])`.
            - If the array is not monotonic, the minimum and maximum values are
              formatted and returned as a string in the format "min~max".
            - If the array has two elements, the two elements are formatted and
              returned.

        - For arrays with more dimensions, the minimum and maximum values are formatted
          and returned as a string in the format "min~max".

    - For lists:
        The list is grouped by consecutive equal elements, and the count of each element
        is formatted and returned as a string in the format "[element]×count".

    - For floating-point numbers:
        - If the number is an integer, it is formatted as an integer using
          `format_value(np.int64(val))`.
        - Otherwise, it is formatted as a floating-point number with specified decimal
          places and returned as a string.

    - For integers:
        The integer is returned as a string.

    - For datetime objects:
        The datetime object is formatted as a string in the format "%Y-%m-%d %H:%M:%S".

    - For other types:
        The value is returned as is.

    Examples
    --------
    >>> format_value(np.array([0.1, 0.15, 0.2]))
    '0.1→0.2 (0.05, 3)'

    >>> format_value(np.array([1.0, 2.0, 2.1]))
    '1→2.1 (3)'

    >>> format_value(np.array([1.0, 2.1, 2.0]))
    '1~2.1 (3)'

    >>> format_value([1, 1, 2, 2, 2, 3, 3, 3, 3])
    '[1]×2, [2]×3, [3]×4'

    >>> format_value(3.14159)
    '3.1416'

    >>> format_value(42.0)
    '42'

    >>> format_value(42)
    '42'

    >>> format_value(datetime.datetime(2024, 1, 1, 12, 0, 0, 0))
    '2024-01-01 12:00:00'
    """

    def _format(val: object) -> str:
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return _format(val.item())

            val = val.squeeze()

            if val.ndim == 1:
                if len(val) == 2:
                    return f"[{_format(val[0])}, {_format(val[1])}]"

                if erlab.utils.array.is_uniform_spaced(val):
                    if val[0] == val[-1]:
                        return _format(val[0])
                    start, end, step = tuple(
                        _format(v) for v in (val[0], val[-1], val[1] - val[0])
                    )
                    formatted = f"{start}→{end} ({step}, {len(val)})"
                    if use_unicode_minus:
                        return formatted.replace("-", "−")
                    return formatted

                if erlab.utils.array.is_monotonic(val):
                    if val[0] == val[-1]:
                        return _format(val[0])

                    return f"{_format(val[0])}→{_format(val[-1])} ({len(val)})"

            mn, mx = tuple(_format(v) for v in (np.nanmin(val), np.nanmax(val)))
            return f"{mn}~{mx} ({len(val)})"

        if isinstance(val, list):
            return ", ".join(
                [f"[{k}]×{len(tuple(g))}" for k, g in itertools.groupby(val)]
            )

        if np.issubdtype(type(val), np.floating):
            val = typing.cast("np.floating", val)
            if val.is_integer():
                return _format(np.int64(val))
            if np.abs(val) < 10 ** (-precision):
                formatted = np.format_float_scientific(
                    val, precision=precision, trim="-"
                )
            else:
                formatted = np.format_float_positional(
                    val, precision=precision, trim="-"
                )
            if use_unicode_minus:
                return formatted.replace("-", "−")
            return formatted

        if np.issubdtype(type(val), np.integer):
            if use_unicode_minus:
                return str(val).replace("-", "−")
            return str(val)

        if isinstance(val, np.generic):
            # Convert to native Python type
            return _format(val.item())

        if isinstance(val, datetime.datetime):
            return val.isoformat(sep=" ", timespec="seconds")

        return str(val)

    try:
        return _format(val)
    except Exception:
        return str(val)


def _format_dim_name(s: Hashable) -> str:
    return f"<b>{s}</b>"


def _format_dim_sizes(darr: xr.DataArray, prefix: str) -> str:
    out = f"<p>{prefix}("

    dims_list = []
    for d in darr.dims:
        dim_label = _format_dim_name(d) if d in darr.coords else str(d)
        dims_list.append(f"{dim_label}: {darr.sizes[d]}")

    out += ", ".join(dims_list)
    out += r")</p>"
    return out


def _format_coord_dims(coord: xr.DataArray) -> str:
    dims = tuple(str(d) for d in coord.variable.dims)

    if len(dims) > 1:
        return f"({', '.join(dims)})&emsp;"

    if len(dims) == 1 and dims[0] != coord.name:
        return f"({dims[0]})&emsp;"

    return ""


def _format_array_values(val: npt.NDArray) -> str:
    if val.size == 1:
        return format_value(val.item())

    val = val.squeeze()

    if val.ndim == 1:
        if len(val) == 2:
            return format_value(val)

        if erlab.utils.array.is_uniform_spaced(val):
            if val[0] == val[-1]:
                return format_value(val[0])

            start, end, step = tuple(
                format_value(v) for v in (val[0], val[-1], val[1] - val[0])
            )
            return f"{start} : {step} : {end}"

        if erlab.utils.array.is_monotonic(val):
            if val[0] == val[-1]:
                return format_value(val[0])

            if len(val) == 2:
                return f"[{format_value(val[0])}, {format_value(val[1])}]"

            return f"{format_value(val[0])} to {format_value(val[-1])}"

    mn, mx = tuple(format_value(v) for v in (np.nanmin(val), np.nanmax(val)))
    return f"min {mn} max {mx}"


def _format_coord_key(key: Hashable, is_dim: bool) -> str:
    style = f"color: {_DEFAULT_ACCENT_COLOR}; "
    if is_dim:
        style += "font-weight: bold; "
    return f"<span style='{style}'>{key}</span>&emsp;"


def _format_attr_key(key: Hashable) -> str:
    style = f"color: {_DEFAULT_ACCENT_COLOR};"
    return f"<span style='{style}'>{key}</span>&emsp;"


def format_darr_html(
    darr: xr.DataArray,
    *,
    show_size: bool = True,
    additional_info: Iterable[str] | None = None,
) -> str:
    """Make a simple HTML representation of a DataArray.

    Parameters
    ----------
    darr
        The DataArray to represent.
    show_size
        Whether to include the size of the DataArray in the representation.
    additional_info
        Additional information to include in the representation. Each item in the list
        is added as a separate paragraph.

    Returns
    -------
    str
        The HTML representation of the DataArray.
    """
    out = ""
    if additional_info is None:
        additional_info = []

    name = ""
    if darr.name is not None and darr.name != "":
        name = f"'{darr.name}'&emsp;"

    out += _format_dim_sizes(darr, name)

    if show_size:
        out += rf"<p>Size {format_nbytes(darr.nbytes)}</p>"

    for info in additional_info:
        out += rf"<p>{info}</p>"

    out += r"Coordinates:"
    coord_rows: list[list[str]] = []
    for key, coord in darr.coords.items():
        is_dim: bool = key in darr.dims

        try:
            value_repr = _format_array_values(coord.values)
        except Exception:
            value_repr = f'["{coord.values[0]!s}",  ... , "{coord.values[-1]!s}"]'

        coord_rows.append(
            [
                _format_coord_key(key, is_dim),
                _format_coord_dims(coord),
                value_repr,
            ]
        )
    out += format_html_table(coord_rows)

    out += r"<br>Attributes:"
    attr_rows: list[list[str]] = []
    for key, attr in darr.attrs.items():
        attr_rows.append([_format_attr_key(key), format_value(attr)])
    out += format_html_table(attr_rows)

    return out
