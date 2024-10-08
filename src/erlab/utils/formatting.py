"""Utilites related to representing data in a human-readable format."""

__all__ = ["format_html_table", "format_value"]
import datetime
import itertools
from collections.abc import Iterable
from typing import cast

import numpy as np

from erlab.utils.array import is_monotonic, is_uniform_spaced

STYLE_SHEET = """
.erlab-table td,
.erlab-table th {
    text-align: left;
}

.erlab-table th {
    font-weight: bold;
}
"""


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


def format_value(
    val: object, precision: int = 4, use_unicode_minus: bool = True
) -> str:
    """Format the given value based on its type.

    This method is used when formatting the cells of the summary dataframe.

    Parameters
    ----------
    val
        The value to be formatted.
    precision
        The number of decimal places to use when formatting floating-point numbers.
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

    The function also tries to replace the Unicode hyphen-minus sign "-" (U+002D) with
    the better-looking Unicode minus sign "−" (U+2212) in most cases.

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

        - For arrays with more dimensions, the array is returned as is.

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

            if val.squeeze().ndim == 1:
                val = val.squeeze()

                if is_uniform_spaced(val):
                    start, end, step = tuple(
                        _format(v) for v in (val[0], val[-1], val[1] - val[0])
                    )
                    formatted = f"{start}→{end} ({step}, {len(val)})"
                    if use_unicode_minus:
                        return formatted.replace("-", "−")
                    return formatted

                if is_monotonic(val):
                    if val[0] == val[-1]:
                        return _format(val[0])

                    return f"{_format(val[0])}→{_format(val[-1])} ({len(val)})"

                mn, mx = tuple(_format(v) for v in (np.min(val), np.max(val)))
                return f"{mn}~{mx} ({len(val)})"

            return str(val)

        if isinstance(val, list):
            return ", ".join(
                [f"[{k}]×{len(tuple(g))}" for k, g in itertools.groupby(val)]
            )

        if np.issubdtype(type(val), np.floating):
            val = cast(np.floating, val)
            if val.is_integer():
                return _format(np.int64(val))
            formatted = np.format_float_positional(val, precision=precision, trim="-")
            if use_unicode_minus:
                return formatted.replace("-", "−")
            return formatted

        if np.issubdtype(type(val), np.integer):
            if use_unicode_minus:
                return str(val).replace("-", "−")
            return str(val)

        if isinstance(val, datetime.datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")

        return str(val)

    return _format(val)
