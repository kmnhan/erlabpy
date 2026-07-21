"""Fill loaded data metadata from spreadsheet rows.

Spreadsheet metadata sources match a loaded path's bare file name directly against a
spreadsheet row, falling back to its file number when needed. Selected columns are then
mapped to scalar coordinates or attributes on data returned by :func:`erlab.io.load`.

.. currentmodule:: erlab.io.metadata

.. autosummary::

   SpreadsheetMetadataSource
   ExcelMetadataSource
   GoogleSheetsMetadataSource
"""

from erlab.io.metadata._core import ExcelMetadataSource, SpreadsheetMetadataSource
from erlab.io.metadata._google import GoogleSheetsMetadataSource

__all__ = [
    "ExcelMetadataSource",
    "GoogleSheetsMetadataSource",
    "SpreadsheetMetadataSource",
]
