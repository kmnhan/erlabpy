"""Private helpers for ALS beamline control system files."""

import csv
import json
import os
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import xarray as xr

_BCSHeader = dict[str, typing.Any]
_BCSRow = dict[str, str]
_FloatArray = npt.NDArray[np.float64]
_PayloadKind = typing.Literal["image", "text"]


class _TextPayload(typing.NamedTuple):
    axis_name: str
    value_names: tuple[str, ...]
    axis: _FloatArray
    values: _FloatArray


def _unique_name(name: str, existing: set[str]) -> str:
    if name not in existing:
        existing.add(name)
        return name

    candidate = f"{name} raw"
    if candidate not in existing:
        existing.add(candidate)
        return candidate

    index = 1
    while f"{candidate} {index}" in existing:
        index += 1

    unique = f"{candidate} {index}"
    existing.add(unique)
    return unique


def load_bcs(path: str | os.PathLike) -> xr.DataArray | xr.DataTree:
    """Load a beamline control system scan.

    Parameters
    ----------
    path
        Path to the BCS text file. Payload files referenced by the data table are
        resolved relative to the text file and, for image payloads, when needed from a
        sibling directory named ``"<text file stem> Images"``.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        A data array containing the compiled payload stack for BCS files with one
        payload column. If a file contains multiple payload columns, each payload stream
        is loaded into a separate child node of a data tree.

    .. versionchanged:: 3.22.0

        Added support for plain tabular text payloads referenced by BCS data tables.
    """
    path = pathlib.Path(path)

    header, columns, rows = _parse_bcs_table(path)

    payload_columns = _find_payload_columns(columns, rows)
    if not payload_columns:
        raise ValueError(f"{path} contains no BCS payload columns")

    payload_kinds = {column: _payload_kind(column, rows) for column in payload_columns}
    numeric_columns = _numeric_columns(columns, rows, payload_columns)
    scan_dim, scan_values, goal_column = _scan_axis(header, numeric_columns, rows)
    constant_values, varying_columns = _split_numeric_columns(
        numeric_columns, goal_column
    )

    data: dict[str, xr.DataArray] = {}
    for payload_column in payload_columns:
        kind = payload_kinds[payload_column]
        if kind == "image":
            data[payload_column] = _load_image_payload_column(
                path,
                payload_column,
                rows,
                scan_dim,
                scan_values,
                varying_columns,
                constant_values,
                header,
            )
        else:
            data[payload_column] = _load_text_payload_column(
                path,
                payload_column,
                rows,
                scan_dim,
                scan_values,
                varying_columns,
                constant_values,
                header,
            )

    if len(payload_columns) == 1:
        return data[payload_columns[0]]

    existing: set[str] = set()
    return xr.DataTree.from_dict(
        {
            _unique_name(payload_column.replace("/", "_"), existing): darr.to_dataset()
            for payload_column, darr in data.items()
        }
    )


def _parse_bcs_table(
    path: pathlib.Path,
) -> tuple[_BCSHeader, list[str], list[_BCSRow]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    try:
        header_start = next(
            i for i, line in enumerate(lines) if line.strip() == "HEADER"
        )
        data_start = next(i for i, line in enumerate(lines) if line.strip() == "DATA")
    except StopIteration as err:
        raise ValueError(f"{path} is not a valid BCS data file") from err

    if data_start <= header_start or data_start + 1 >= len(lines):
        raise ValueError(f"{path} is missing BCS data table")

    header = typing.cast(
        "_BCSHeader",
        json.loads("\n".join(lines[header_start + 1 : data_start]).strip()),
    )
    reader = csv.DictReader(lines[data_start + 1 :], delimiter="\t")
    columns = list(reader.fieldnames or ())
    rows = typing.cast("list[_BCSRow]", list(reader))
    if not rows:
        raise ValueError(f"{path} contains no BCS data rows")

    return header, columns, rows


def _find_payload_columns(columns: list[str], rows: list[_BCSRow]) -> list[str]:
    return [
        column
        for column in columns
        if any(_payload_suffix(row.get(column)) is not None for row in rows)
    ]


def _payload_suffix(value: str | None) -> str | None:
    if value is None:
        return None

    suffix = pathlib.PurePosixPath(value.strip().replace("\\", "/")).suffix.casefold()
    return suffix if suffix in {".png", ".txt"} else None


def _payload_kind(column: str, rows: list[_BCSRow]) -> _PayloadKind:
    suffixes = {
        suffix
        for row in rows
        if (suffix := _payload_suffix(row.get(column))) is not None
    }
    if suffixes == {".png"}:
        return "image"
    if suffixes == {".txt"}:
        return "text"
    raise ValueError(f"BCS payload column {column!r} mixes payload types")


def _numeric_columns(
    columns: list[str], rows: list[_BCSRow], payload_columns: list[str]
) -> dict[str, _FloatArray]:
    numeric_columns: dict[str, _FloatArray] = {}
    for column in columns:
        if column in payload_columns:
            continue

        values: list[float] = []
        for row in rows:
            value = row.get(column)
            if value is None or value == "":
                break
            try:
                values.append(float(value))
            except ValueError:
                break
        else:
            numeric_columns[column] = np.asarray(values, dtype=np.float64)
    return numeric_columns


def _scan_axis(
    header: _BCSHeader, numeric_columns: dict[str, _FloatArray], rows: list[_BCSRow]
) -> tuple[str, _FloatArray, str | None]:
    scan_type = header.get("Scan Type", "")
    scan_info = header.get(scan_type, {})
    scan_motor = scan_info.get("X Motor") if isinstance(scan_info, dict) else None
    scan_dim = str(scan_motor) if scan_motor else "step"
    goal_column: str | None = f"{scan_dim} Goal"
    if goal_column in numeric_columns:
        scan_values = numeric_columns[goal_column]
    elif scan_dim in numeric_columns:
        scan_values = numeric_columns[scan_dim]
        goal_column = scan_dim
    else:
        scan_values = np.arange(len(rows), dtype=np.float64)
        goal_column = None

    return scan_dim, scan_values, goal_column


def _split_numeric_columns(
    numeric_columns: dict[str, _FloatArray], goal_column: str | None
) -> tuple[dict[str, float], dict[str, _FloatArray]]:
    constant_values: dict[str, float] = {}
    varying_columns: dict[str, _FloatArray] = {}
    for column, column_values in numeric_columns.items():
        if column == goal_column:
            continue
        if np.all(column_values == column_values[0]):
            constant_values[column] = float(column_values[0])
        else:
            varying_columns[column] = column_values

    return constant_values, varying_columns


def _resolve_payload_path(
    bcs_path: pathlib.Path, payload_ref: str, kind: _PayloadKind
) -> pathlib.Path:
    normalized_ref = pathlib.Path(payload_ref.strip().replace("\\", "/"))
    if normalized_ref.is_absolute():
        candidates: tuple[pathlib.Path, ...] = (normalized_ref,)
    else:
        stripped_parts = tuple(
            part for part in normalized_ref.parts if part not in {".", ".."}
        )
        candidates = (
            bcs_path.parent / normalized_ref,
            bcs_path.parent.joinpath(*stripped_parts),
        )
        if kind == "image":
            candidates = (
                *candidates,
                bcs_path.parent / f"{bcs_path.stem} Images" / normalized_ref.name,
            )

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate

    payload_type = "image" if kind == "image" else "text payload"
    raise FileNotFoundError(
        f"Could not find BCS {payload_type} {payload_ref!r} for {bcs_path.name}"
    )


def _load_image_payload_column(
    bcs_path: pathlib.Path,
    payload_column: str,
    rows: list[_BCSRow],
    scan_dim: str,
    scan_values: _FloatArray,
    varying_columns: dict[str, _FloatArray],
    constant_values: dict[str, float],
    header: _BCSHeader,
) -> xr.DataArray:
    try:
        from PIL import Image
    except ImportError as err:
        raise ImportError(
            "PIL is required to load BCS files with image columns. "
            "Please install `erlab[io]` or Pillow to use this feature."
        ) from err

    images: list[npt.NDArray[np.generic]] = []
    for row in rows:
        image_ref = row[payload_column]
        image_path = _resolve_payload_path(bcs_path, image_ref, "image")

        with Image.open(image_path) as image:
            images.append(np.asarray(image).copy())

    try:
        image_stack = np.stack(images, axis=0)
    except ValueError as err:
        raise ValueError("All BCS images must have the same shape") from err

    image_shape = image_stack.shape[1:]
    if len(image_shape) == 2:
        image_stack = np.moveaxis(image_stack, 0, -1)
        dims: tuple[str, ...] = ("y", "x", scan_dim)
    elif len(image_shape) == 3:
        image_stack = np.moveaxis(image_stack, 0, -1)
        dims = ("y", "x", "channel", scan_dim)
    else:
        raise ValueError("BCS images must be two-dimensional or RGB/RGBA images")

    coords: dict[str, typing.Any] = {
        scan_dim: scan_values,
        "y": np.arange(image_shape[0]),
        "x": np.arange(image_shape[1]),
    }
    if len(image_shape) == 3:
        coords["channel"] = np.arange(image_shape[2])

    coords, attrs = _attach_bcs_metadata(
        coords, header, scan_dim, varying_columns, constant_values
    )
    return xr.DataArray(
        image_stack,
        dims=dims,
        coords=coords,
        name=payload_column,
        attrs=attrs,
    )


def _load_text_payload_column(
    bcs_path: pathlib.Path,
    payload_column: str,
    rows: list[_BCSRow],
    scan_dim: str,
    scan_values: _FloatArray,
    varying_columns: dict[str, _FloatArray],
    constant_values: dict[str, float],
    header: _BCSHeader,
) -> xr.DataArray:
    payloads = [
        _load_text_payload(_resolve_payload_path(bcs_path, row[payload_column], "text"))
        for row in rows
    ]
    first = payloads[0]
    _validate_text_payloads(payload_column, payloads)

    if first.axis_name == scan_dim:
        raise ValueError(
            f"BCS text payload axis {first.axis_name!r} conflicts with the scan "
            f"dimension for payload column {payload_column!r}"
        )

    if len(first.value_names) == 1:
        values = np.stack([payload.values[:, 0] for payload in payloads], axis=-1)
        dims: tuple[str, ...] = (first.axis_name, scan_dim)
        coords: dict[str, typing.Any] = {
            first.axis_name: first.axis,
            scan_dim: scan_values,
        }
    else:
        values = np.stack([payload.values for payload in payloads], axis=-1)
        coords = {
            first.axis_name: first.axis,
            scan_dim: scan_values,
        }
        column_dim = _unique_name("column", set(coords))
        coords[column_dim] = np.asarray(first.value_names, dtype=object)
        dims = (first.axis_name, column_dim, scan_dim)

    coords, attrs = _attach_bcs_metadata(
        coords, header, scan_dim, varying_columns, constant_values
    )
    if len(first.value_names) == 1:
        attrs[_unique_name("BCS value column", set(attrs))] = first.value_names[0]

    return xr.DataArray(
        values,
        dims=dims,
        coords=coords,
        name=payload_column,
        attrs=attrs,
    )


def _load_text_payload(path: pathlib.Path) -> _TextPayload:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = [
            row
            for row in csv.reader(file, delimiter="\t")
            if any(cell.strip() for cell in row)
        ]

    if not rows:
        raise ValueError(f"BCS text payload {path} is empty")

    header = tuple(rows[0])
    if len(header) < 2:
        raise ValueError(
            f"BCS text payload {path} must contain an axis column and at least one "
            "value column"
        )

    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"BCS text payload {path} contains no data rows")

    if any(len(row) != len(header) for row in data_rows):
        raise ValueError(f"BCS text payload {path} contains ragged rows")

    try:
        values = np.asarray(data_rows, dtype=np.float64)
    except ValueError as err:
        raise ValueError(f"BCS text payload {path} contains non-numeric data") from err

    return _TextPayload(
        axis_name=header[0],
        value_names=header[1:],
        axis=values[:, 0],
        values=values[:, 1:],
    )


def _validate_text_payloads(payload_column: str, payloads: list[_TextPayload]) -> None:
    first = payloads[0]
    for payload in payloads[1:]:
        if payload.axis_name != first.axis_name:
            raise ValueError(
                f"BCS text payloads in column {payload_column!r} must use the same "
                "axis label"
            )
        if payload.value_names != first.value_names:
            raise ValueError(
                f"BCS text payloads in column {payload_column!r} must use the same "
                "value column labels"
            )
        if payload.axis.shape != first.axis.shape or not np.array_equal(
            payload.axis, first.axis
        ):
            raise ValueError(
                f"BCS text payload axes differ for payload column {payload_column!r}"
            )


def _attach_bcs_metadata(
    coords: dict[str, typing.Any],
    header: _BCSHeader,
    scan_dim: str,
    varying_columns: dict[str, _FloatArray],
    constant_values: dict[str, float],
) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    existing_coords = set(coords)

    for column, coord_values in varying_columns.items():
        coords[_unique_name(column, existing_coords)] = (scan_dim, coord_values)

    for column, coord_values in constant_values.items():
        coords[_unique_name(column, existing_coords)] = coord_values

    motors = header.get("Motors", {})
    if isinstance(motors, dict):
        for motor_name, motor_value in motors.items():
            if motor_name not in existing_coords:
                coords[motor_name] = motor_value
                existing_coords.add(motor_name)

    attrs: dict[str, typing.Any] = {}
    existing_attrs = set(attrs)
    for attr_name, attr_value in header.items():
        if attr_name in {"General", "Motors"}:
            continue
        if attr_name not in existing_coords:
            attrs[_unique_name(attr_name, existing_attrs)] = attr_value

    return coords, attrs
