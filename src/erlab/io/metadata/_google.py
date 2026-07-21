"""Read metadata from publicly readable Google Sheets."""

from __future__ import annotations

__all__ = ["GoogleSheetsMetadataSource"]

import concurrent.futures
import csv
import importlib.util
import io
import json
import re
import typing
from urllib.parse import parse_qs, quote, urlsplit

from erlab.io.metadata._core import SpreadsheetMetadataSource, _parse_spreadsheet_value

if typing.TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def _parse_share_url(share_url: object) -> tuple[str, int | None]:
    if not isinstance(share_url, str) or not share_url:
        raise ValueError("share_url must be a non-empty string")

    parsed = urlsplit(share_url)
    if parsed.scheme != "https" or parsed.hostname != "docs.google.com":
        raise ValueError("share_url must be a Google Sheets HTTPS URL")

    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 3 or path_parts[:2] != ["spreadsheets", "d"]:
        raise ValueError("share_url does not contain a Google Sheets spreadsheet ID")
    spreadsheet_id = path_parts[2]
    if not spreadsheet_id:
        raise ValueError("share_url does not contain a Google Sheets spreadsheet ID")

    query_parameters = parse_qs(parsed.query)
    fragment_parameters = parse_qs(parsed.fragment)

    gid_value = fragment_parameters.get("gid", query_parameters.get("gid", [None]))[0]
    if gid_value is None:
        gid = None
    else:
        try:
            gid = int(gid_value)
        except ValueError as exc:
            raise ValueError("share_url contains an invalid worksheet gid") from exc
        if gid < 0:
            raise ValueError("share_url contains an invalid worksheet gid")

    return spreadsheet_id, gid


_SHEET_ITEM_PATTERN = re.compile(
    r"items\.push\(\{\s*name:\s*"
    r'(?P<name>"(?:\\.|[^"\\])*")'
    r'.*?gid:\s*"(?P<gid>\d+)"',
    re.DOTALL,
)
_JAVASCRIPT_HEX_ESCAPE_PATTERN = re.compile(r"\\x(?P<value>[0-9A-Fa-f]{2})")


def _decode_javascript_string(value: str) -> str:
    value = _JAVASCRIPT_HEX_ESCAPE_PATTERN.sub(
        lambda match: rf"\u00{match.group('value')}", value
    )
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Google Sheets returned invalid worksheet metadata") from exc
    return typing.cast("str", decoded)


def _request_public_sheet(url: str, *, params: dict[str, str], timeout: float) -> Any:
    if not importlib.util.find_spec("requests"):
        raise ImportError(
            "`requests` needs to be installed to read Google Sheets metadata"
        )
    import requests

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError("Google Sheets request failed") from exc

    if response.status_code in {401, 403}:
        raise RuntimeError("The spreadsheet is not publicly readable")
    if response.status_code == 429:
        raise RuntimeError("Google Sheets request was rate-limited")

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Google Sheets returned HTTP {response.status_code}"
        ) from exc
    return response


def _read_public_sheet_properties(
    *, spreadsheet_id: str, timeout: float
) -> list[tuple[str, int]]:
    """Read worksheet names and IDs from Google's lightweight view-only page.

    Google does not expose a documented keyless worksheet-list endpoint. Keep the
    parser deliberately narrow and fail closed if the page structure changes so an
    invalid worksheet can never silently fall back to the first one.
    """
    encoded_spreadsheet_id = quote(spreadsheet_id, safe="")
    url = f"https://docs.google.com/spreadsheets/d/{encoded_spreadsheet_id}/htmlview"
    response = _request_public_sheet(url, params={}, timeout=timeout)

    properties = [
        (_decode_javascript_string(match.group("name")), int(match.group("gid")))
        for match in _SHEET_ITEM_PATTERN.finditer(response.text)
    ]
    if not properties:
        raise RuntimeError(
            "Google Sheets did not return a worksheet list; verify that the "
            "spreadsheet is publicly readable"
        )
    return properties


def _read_public_sheet_values(
    *,
    spreadsheet_id: str,
    gid: int | None,
    sheet_name: str | None,
    row_range: tuple[int, int] | None,
    header_only: bool,
    timeout: float,
) -> list[list[Any]]:
    if gid is not None and sheet_name is not None:
        raise ValueError("Only one Google Sheets worksheet selector may be provided")

    params = {"tqx": "out:csv", "headers": "1"}
    if gid is not None:
        params["gid"] = str(gid)
    elif sheet_name is not None:
        # Google documents `sheet=sheet_name` as an alternative to `gid=N`:
        # https://developers.google.com/chart/interactive/docs/spreadsheets
        params["sheet"] = sheet_name
    if header_only:
        params["headers"] = "0"
        params["range"] = "1:1"
    elif row_range is not None:
        start, end = row_range
        # Google documents `range=5:7` as absolute spreadsheet rows 5 through 7:
        # https://developers.google.com/chart/interactive/docs/spreadsheets#query-source-ranges
        # Query OFFSET instead counts rows in the data table, which may omit blanks.
        params["headers"] = "0"
        params["range"] = f"{start}:{end}"

    encoded_spreadsheet_id = quote(spreadsheet_id, safe="")
    url = f"https://docs.google.com/spreadsheets/d/{encoded_spreadsheet_id}/gviz/tq"
    response = _request_public_sheet(url, params=params, timeout=timeout)

    try:
        rows = csv.reader(
            io.StringIO(response.text.removeprefix("\ufeff")), strict=True
        )
        return [[_parse_spreadsheet_value(value) for value in row] for row in rows]
    except csv.Error as exc:
        raise RuntimeError("Google Sheets returned invalid CSV") from exc


class GoogleSheetsMetadataSource(SpreadsheetMetadataSource):
    """Metadata source backed by a publicly readable Google Sheets link.

    Parameters
    ----------
    share_url
        Shareable link copied from Google Sheets. The linked worksheet is used
        automatically.
    sheet_name
        Visible tab name of another worksheet to use instead of the worksheet linked by
        `share_url`. The name is validated against the workbook's worksheet list before
        any concurrently fetched values are accepted. If the link has no worksheet and
        this is omitted, the first worksheet is used.
    timeout
        HTTP request timeout in seconds.

    Other Parameters
    ----------------
    file_name_column, coordinate_mapping, attribute_mapping, overwrite, row_range
        See :class:`erlab.io.metadata.SpreadsheetMetadataSource`.

    Notes
    -----
    The spreadsheet must be shared publicly or with anyone who has the link. Numeric
    and boolean CSV values are converted to Python scalars; other values retain their
    displayed text. Worksheet names and headers are cached, while row content is read
    again for every metadata lookup. Independent requests needed for an uncached lookup
    run concurrently. Worksheet names are still validated so that an invalid
    `sheet_name` cannot silently fall back to the first worksheet.

    .. versionadded:: 3.25.0
    """

    _cache_row_index = False

    def __init__(
        self,
        share_url: str,
        *,
        sheet_name: str | None = None,
        timeout: float = 10.0,
        file_name_column: str | None = None,
        coordinate_mapping: Mapping[str, str] | None = None,
        attribute_mapping: Mapping[str, str] | None = None,
        overwrite: bool = False,
        row_range: tuple[int, int] | None = None,
    ) -> None:
        spreadsheet_id, gid = _parse_share_url(share_url)
        if sheet_name is not None and not isinstance(sheet_name, str):
            raise TypeError("Google Sheets sheet_name must be a worksheet name")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a positive number")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.share_url = share_url
        self.spreadsheet_id = spreadsheet_id
        self.gid = gid
        self.timeout = float(timeout)
        self._worksheet_properties: tuple[tuple[str, int], ...] | None = None
        super().__init__(
            file_name_column=file_name_column,
            coordinate_mapping=coordinate_mapping,
            attribute_mapping=attribute_mapping,
            sheet_name=sheet_name,
            overwrite=overwrite,
            row_range=row_range,
        )

    @property
    def source_name(self) -> str:
        return f"Google Sheets metadata source {self.share_url!r}"

    def _clear_source_cache(self) -> None:
        self._worksheet_properties = None

    def _get_worksheet_properties(self) -> tuple[tuple[str, int], ...]:
        if self._worksheet_properties is None:
            self._set_worksheet_properties(
                _read_public_sheet_properties(
                    spreadsheet_id=self.spreadsheet_id,
                    timeout=self.timeout,
                )
            )
        if self._worksheet_properties is None:  # pragma: no cover - set above
            raise RuntimeError("Google Sheets worksheet cache is inconsistent")
        return self._worksheet_properties

    def _set_worksheet_properties(self, values: list[tuple[str, int]]) -> None:
        properties = tuple(values)
        names = [name for name, _ in properties]
        gids = [gid for _, gid in properties]
        if len(set(names)) != len(names) or len(set(gids)) != len(gids):
            raise RuntimeError("Google Sheets returned duplicate worksheet metadata")
        self._worksheet_properties = properties

    def _read_sheet_names(self) -> list[str]:
        return [name for name, _ in self._get_worksheet_properties()]

    def get_selected_sheet_name(self) -> str:
        """Return the visible name of the worksheet selected by the link or name."""
        with self._lock:
            selected_gid = self._selected_gid()
            for name, gid in self._get_worksheet_properties():
                if gid == selected_gid:
                    return name
        raise RuntimeError("Google Sheets worksheet selection is inconsistent")

    def _selected_gid(self) -> int:
        properties = self._get_worksheet_properties()
        if self.sheet_name is not None:
            sheet_name = typing.cast("str", self.sheet_name)
            for name, gid in properties:
                if name == sheet_name:
                    return gid
            available = ", ".join(repr(name) for name, _ in properties)
            raise ValueError(
                f"Worksheet {sheet_name!r} was not found in the Google Sheets "
                f"spreadsheet. Available worksheets: {available}"
            )
        if self.gid is not None:
            if any(gid == self.gid for _, gid in properties):
                return self.gid
            raise ValueError(
                f"Worksheet gid {self.gid} from share_url was not found in the "
                "Google Sheets spreadsheet"
            )
        return properties[0][1]

    def _configured_selector(self) -> tuple[int | None, str | None]:
        if self.sheet_name is not None:
            return None, typing.cast("str", self.sheet_name)
        return self.gid, None

    def _read_values(
        self,
        *,
        gid: int | None,
        sheet_name: str | None,
        row_range: tuple[int, int] | None,
        header_only: bool,
    ) -> list[list[Any]]:
        return _read_public_sheet_values(
            spreadsheet_id=self.spreadsheet_id,
            gid=gid,
            sheet_name=sheet_name,
            row_range=row_range,
            header_only=header_only,
            timeout=self.timeout,
        )

    def _read_header(self) -> list[Any]:
        if self._worksheet_properties is not None:
            rows = self._read_values(
                gid=self._selected_gid(),
                sheet_name=None,
                row_range=None,
                header_only=True,
            )
        else:
            gid, sheet_name = self._configured_selector()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                properties_future = executor.submit(
                    _read_public_sheet_properties,
                    spreadsheet_id=self.spreadsheet_id,
                    timeout=self.timeout,
                )
                rows_future = executor.submit(
                    self._read_values,
                    gid=gid,
                    sheet_name=sheet_name,
                    row_range=None,
                    header_only=True,
                )
                properties = properties_future.result()
                rows = rows_future.result()
            self._set_worksheet_properties(properties)
            self._selected_gid()
        if not rows:
            return []
        return rows[0]

    def _read_rows(self) -> list[list[Any]]:
        read_properties = self._worksheet_properties is None
        read_header = self.row_range is not None and self._headers is None
        header_rows: list[list[Any]] | None = None
        if not read_properties and not read_header:
            rows = self._read_values(
                gid=self._selected_gid(),
                sheet_name=None,
                row_range=self.row_range,
                header_only=False,
            )
        else:
            if read_properties:
                gid, sheet_name = self._configured_selector()
            else:
                gid, sheet_name = self._selected_gid(), None
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1 + int(read_properties) + int(read_header)
            ) as executor:
                properties_future = (
                    executor.submit(
                        _read_public_sheet_properties,
                        spreadsheet_id=self.spreadsheet_id,
                        timeout=self.timeout,
                    )
                    if read_properties
                    else None
                )
                header_future = (
                    executor.submit(
                        self._read_values,
                        gid=gid,
                        sheet_name=sheet_name,
                        row_range=None,
                        header_only=True,
                    )
                    if read_header
                    else None
                )
                rows_future = executor.submit(
                    self._read_values,
                    gid=gid,
                    sheet_name=sheet_name,
                    row_range=self.row_range,
                    header_only=False,
                )
                properties = (
                    properties_future.result()
                    if properties_future is not None
                    else None
                )
                header_rows = (
                    header_future.result() if header_future is not None else None
                )
                rows = rows_future.result()
            if properties is not None:
                self._set_worksheet_properties(properties)
                self._selected_gid()

        if self.row_range is None:
            if self._headers is not None:
                if rows:
                    return [list(self._headers), *rows[1:]]
                return [list(self._headers)]
            return rows
        if header_rows is not None:
            header = header_rows[0] if header_rows else []
        elif self._headers is not None:
            header = list(self._headers)
        else:  # pragma: no cover - read_header is true in this state
            raise RuntimeError("Google Sheets header cache is inconsistent")
        return [header, *rows]

    def refresh(self) -> None:
        """Refresh the cached worksheet validation and column names."""
        with self._lock:
            previous_properties = self._worksheet_properties
            try:
                super().refresh()
            except Exception:
                self._worksheet_properties = previous_properties
                raise
