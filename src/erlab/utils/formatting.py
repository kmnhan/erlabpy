"""Utilites related to representing data in a human-readable format."""

from collections.abc import Iterable

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
