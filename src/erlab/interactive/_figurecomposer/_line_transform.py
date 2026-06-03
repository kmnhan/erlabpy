"""Shared 1D profile transform controls, rendering, and generated code."""

from __future__ import annotations

import math
import typing

import numpy as np
from qtpy import QtWidgets

from erlab.interactive._figurecomposer._text import (
    _float_tuple_from_text,
    _format_tuple,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer._state import FigureOperationState
    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_NORMALIZE_TEXT = {
    "none": "None",
    "max": "Each profile by maximum",
    "mean": "Each profile by mean",
}


def line_normalize_text(mode: str) -> str:
    return _NORMALIZE_TEXT.get(mode, _NORMALIZE_TEXT["none"])


def line_normalize_from_text(
    text: str,
) -> typing.Literal["none", "max", "mean"]:
    for mode, label in _NORMALIZE_TEXT.items():
        if text == label:
            return typing.cast("typing.Literal['none', 'max', 'mean']", mode)
    return "none"


def add_line_transform_controls(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    *,
    object_prefix: str,
    offset_coord_options: Callable[[FigureOperationState], Sequence[str]],
) -> None:
    """Add normalized profile transform controls to an operation editor."""
    normalize_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_normalize
    )
    normalize_combo = tool._combo(
        [
            line_normalize_text("none"),
            line_normalize_text("max"),
            line_normalize_text("mean"),
        ],
        None if normalize_mixed else line_normalize_text(operation.line_normalize),
        lambda text: tool._update_current_operation(
            line_normalize=line_normalize_from_text(text)
        ),
        parent=page,
        mixed=normalize_mixed,
    )
    normalize_combo.setObjectName(f"{object_prefix}NormalizeCombo")
    tool._add_form_row(
        layout,
        "Normalize",
        normalize_combo,
        "Normalize each extracted 1D profile independently before "
        "scale/offset. This does not normalize the source image or all "
        "profiles together.",
    )

    scales_text, scales_mixed = tool._batch_text(
        operation, lambda target: target.line_scales, _format_tuple
    )
    scales_edit = tool._line_edit(scales_text, parent=page)
    tool._apply_mixed_line_edit(scales_edit, scales_mixed)
    scales_edit.setObjectName(f"{object_prefix}ScalesEdit")
    tool._connect_line_edit_finished(
        scales_edit,
        lambda text: tool._update_current_operation(
            line_scales=_float_tuple_from_text(text)
        ),
    )
    tool._add_form_row(
        layout,
        "Scales",
        scales_edit,
        "Scale applied to profile data values.\n"
        "Use one value or comma-separated per-profile values.",
    )

    offset_source_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_offset_source
    )
    offset_source_combo = tool._combo(
        ["manual", "index", "coordinate", "associated"],
        None if offset_source_mixed else operation.line_offset_source,
        lambda source: update_current_line_offset_source(tool, source),
        parent=page,
        mixed=offset_source_mixed,
    )
    offset_source_combo.setObjectName(f"{object_prefix}OffsetSourceCombo")
    tool._add_form_row(
        layout,
        "Offset source",
        offset_source_combo,
        "Where offsets come from.\n"
        "manual: use Offsets.\n"
        "index/coordinate/associated: derive from profile order or coordinates.",
    )

    if operation.line_offset_source == "associated":
        offset_coord_values = ["", *offset_coord_options(operation)]
        offset_coord_options_match = tool._batch_options_match(
            operation, lambda target: ["", *offset_coord_options(target)]
        )
        offset_coord_mixed = tool._batch_is_mixed(
            operation, lambda target: target.line_offset_coord
        )
        if (
            operation.line_offset_coord is not None
            and operation.line_offset_coord not in offset_coord_values
        ):
            offset_coord_values.append(operation.line_offset_coord)
        offset_coord_combo = tool._combo(
            offset_coord_values,
            None if offset_coord_mixed else operation.line_offset_coord or "",
            lambda text: tool._update_current_operation(line_offset_coord=text or None),
            parent=page,
            mixed=offset_coord_mixed,
            enabled=offset_coord_options_match,
        )
        offset_coord_combo.setObjectName(f"{object_prefix}OffsetCoordinateCombo")
        tool._add_form_row(
            layout,
            "Offset coordinate",
            offset_coord_combo,
            "Associated coordinate used when Offset source is associated."
            + (
                "\nDisabled while selected steps have different valid choices."
                if not offset_coord_options_match
                else ""
            ),
        )

    if operation.line_offset_source != "manual":
        offset_scale_mixed = tool._batch_is_mixed(
            operation, lambda target: target.line_offset_scale
        )
        offset_scale_spin = QtWidgets.QDoubleSpinBox(page)
        offset_scale_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        offset_scale_spin.setDecimals(6)
        offset_scale_spin.setSingleStep(0.1)
        offset_scale_spin.setKeyboardTracking(False)
        offset_scale_spin.setValue(operation.line_offset_scale)
        offset_scale_spin.setObjectName(f"{object_prefix}OffsetScaleEdit")
        tool._connect_value_signal(
            offset_scale_spin,
            offset_scale_spin.valueChanged,
            float,
            lambda value: tool._update_current_operation(line_offset_scale=value),
        )
        offset_scale_tooltip = "Multiplier applied to offsets from the selected source."
        if offset_scale_mixed:
            offset_scale_tooltip += "\nSelected steps have multiple values."
        tool._add_form_row(
            layout,
            "Offset scale",
            tool._mixed_value_widget(
                offset_scale_spin, mixed=offset_scale_mixed, parent=page
            ),
            offset_scale_tooltip,
        )

    if operation.line_offset_source == "manual":
        offsets_text, offsets_mixed = tool._batch_text(
            operation, lambda target: target.line_offsets, _format_tuple
        )
        offsets_edit = tool._line_edit(offsets_text, parent=page)
        tool._apply_mixed_line_edit(offsets_edit, offsets_mixed)
        offsets_edit.setObjectName(f"{object_prefix}OffsetsEdit")
        tool._connect_line_edit_finished(
            offsets_edit,
            lambda text: tool._update_current_operation(
                line_offsets=_float_tuple_from_text(text)
            ),
        )
        tool._add_form_row(
            layout,
            "Offsets",
            offsets_edit,
            "Offset applied to profile data values.\n"
            "Use one value or comma-separated per-profile values.",
        )


def update_current_line_offset_source(tool: FigureComposerTool, source: str) -> None:
    updates: dict[str, typing.Any] = {"line_offset_source": source}
    if source == "manual":
        updates["line_offset_scale"] = 1.0
    tool._update_current_operation_rebuild(**updates)


def line_transform_active(operation: FigureOperationState) -> bool:
    return (
        operation.line_normalize != "none"
        or bool(operation.line_scales)
        or line_uses_offsets(operation)
    )


def transform_profiles(
    operation: FigureOperationState, profiles: Sequence[xr.DataArray]
) -> list[xr.DataArray]:
    if not line_transform_active(operation):
        return list(profiles)
    normalized = [
        normalize_line_data(profile, operation.line_normalize) for profile in profiles
    ]
    scales = line_transform_values(operation.line_scales, len(normalized), default=1.0)
    offsets = line_offsets_for_profiles(operation, normalized)
    return [
        offset + scale * profile
        for profile, scale, offset in zip(normalized, scales, offsets, strict=True)
    ]


def normalize_line_data(
    line_data: xr.DataArray, mode: typing.Literal["none", "max", "mean"]
) -> xr.DataArray:
    if mode == "none":
        return line_data
    if mode == "max":
        scale = float(line_data.max(skipna=True))
    else:
        scale = float(line_data.mean(skipna=True))
    if math.isfinite(scale) and scale != 0.0:
        return line_data / scale
    return line_data


def line_offsets_for_profiles(
    operation: FigureOperationState,
    profiles: Sequence[xr.DataArray],
) -> tuple[float, ...]:
    count = len(profiles)
    if operation.line_offset_source == "manual":
        return line_transform_values(operation.line_offsets, count, default=0.0)
    if operation.line_offset_source == "index":
        offsets = tuple(float(index) for index in range(count))
    else:
        coord_name = line_offset_coordinate_name(operation)
        offsets = tuple(
            profile_scalar_coord_value(profile, coord_name) for profile in profiles
        )
    if operation.line_offset_scale == 1.0:
        return offsets
    return tuple(operation.line_offset_scale * offset for offset in offsets)


def line_offset_coordinate_name(operation: FigureOperationState) -> str:
    if operation.line_offset_source == "coordinate":
        if operation.line_iter_dim is None:
            raise ValueError("Coordinate offsets require One profile per")
        return operation.line_iter_dim
    if operation.line_offset_coord is None:
        raise ValueError("Associated-coordinate offsets require a coordinate")
    return operation.line_offset_coord


def profile_scalar_coord_value(profile: xr.DataArray, coord_name: str) -> float:
    coord = profile.coords.get(coord_name)
    if coord is None:
        raise ValueError(f"Profile has no coordinate named {coord_name!r}")
    values = np.asarray(coord.values).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Profile coordinate {coord_name!r} is not scalar")
    return float(values[0])


def line_transform_values(
    values: Sequence[float], count: int, *, default: float
) -> tuple[float, ...]:
    if count < 1:
        return ()
    if not values:
        return (default,) * count
    if len(values) == 1:
        return (float(values[0]),) * count
    if len(values) != count:
        raise ValueError(
            "Profile scales and offsets must be one value or one per profile"
        )
    return tuple(float(value) for value in values)


def line_uses_offsets(operation: FigureOperationState) -> bool:
    return operation.line_offset_source != "manual" or bool(operation.line_offsets)


def profile_transform_code_lines(
    operation: FigureOperationState, *, profiles_name: str = "profiles"
) -> list[str]:
    if not line_transform_active(operation):
        return []

    lines: list[str] = []
    loop_names = ["profile"]
    loop_values = [profiles_name]
    scale_name = _profile_scale_code(lines, operation, loop_names, loop_values)
    offset_name = _profile_offset_code(
        lines, operation, loop_names, loop_values, profiles_name=profiles_name
    )
    value_expr = _profile_transform_expression(
        operation, scale_name=scale_name, offset_name=offset_name
    )
    lines.append(f"{profiles_name} = [")
    lines.append(f"    {value_expr}")
    if len(loop_names) == 1:
        lines.append(f"    for profile in {profiles_name}")
    else:
        lines.append("    for " + ", ".join(loop_names) + " in zip(")
        lines.extend(f"        {value}," for value in loop_values)
        lines.append("        strict=True,")
        lines.append("    )")
    lines.append("]")
    return lines


def _profile_scale_code(
    lines: list[str],
    operation: FigureOperationState,
    loop_names: list[str],
    loop_values: list[str],
) -> str | None:
    if not operation.line_scales:
        return None
    if len(operation.line_scales) == 1:
        lines.append(f"profile_scale = {operation.line_scales[0]!r}")
        return "profile_scale"
    lines.append(f"profile_scales = {list(operation.line_scales)!r}")
    loop_names.append("scale")
    loop_values.append("profile_scales")
    return "scale"


def _profile_offset_code(
    lines: list[str],
    operation: FigureOperationState,
    loop_names: list[str],
    loop_values: list[str],
    *,
    profiles_name: str,
) -> str | None:
    if not line_uses_offsets(operation):
        return None
    if operation.line_offset_source == "manual":
        if len(operation.line_offsets) == 1:
            lines.append(f"profile_offset = {operation.line_offsets[0]!r}")
            return "profile_offset"
        lines.append(f"profile_offsets = {list(operation.line_offsets)!r}")
    elif operation.line_offset_source == "index":
        lines.extend(_offset_list_code("float(index)", operation, profiles_name))
    else:
        coord_name = line_offset_coordinate_name(operation)
        lines.extend(
            _offset_list_code(
                f"float(profile[{coord_name!r}])", operation, profiles_name
            )
        )
    loop_names.append("offset")
    loop_values.append("profile_offsets")
    return "offset"


def _offset_list_code(
    value_expr: str, operation: FigureOperationState, profiles_name: str
) -> list[str]:
    if operation.line_offset_scale != 1.0:
        lines = [f"profile_offset_scale = {operation.line_offset_scale!r}"]
        value_expr = f"profile_offset_scale * {value_expr}"
    else:
        lines = []
    if operation.line_offset_source == "index":
        lines.append(
            f"profile_offsets = [{value_expr} for index in range(len({profiles_name}))]"
        )
    else:
        lines.append(f"profile_offsets = [{value_expr} for profile in {profiles_name}]")
    return lines


def _profile_transform_expression(
    operation: FigureOperationState,
    *,
    scale_name: str | None,
    offset_name: str | None,
) -> str:
    value_expr = _profile_normalize_expression(operation)
    if scale_name is not None:
        if operation.line_normalize == "none":
            value_expr = f"{scale_name} * {value_expr}"
        else:
            value_expr = f"{scale_name} * ({value_expr})"
    if offset_name is not None:
        value_expr = f"{offset_name} + {value_expr}"
    return value_expr


def _profile_normalize_expression(operation: FigureOperationState) -> str:
    if operation.line_normalize == "max":
        return "profile / profile.max(skipna=True)"
    if operation.line_normalize == "mean":
        return "profile / profile.mean(skipna=True)"
    return "profile"
