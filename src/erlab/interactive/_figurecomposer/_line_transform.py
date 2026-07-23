"""Shared 1D profile transform controls, rendering, and generated code."""

from __future__ import annotations

import dataclasses
import math
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer._model._state import FigureOperationState


_NORMALIZE_TEXT = {
    "none": "None",
    "max": "Each profile by maximum",
    "mean": "Each profile by mean",
}


@dataclasses.dataclass(frozen=True)
class LineTransformPlan:
    """Semantic inputs used to transform a sequence of line profiles."""

    iter_dim: str | None
    normalize: typing.Literal["none", "max", "mean"]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    offset_source: typing.Literal["manual", "index", "coordinate", "associated"]
    offset_coord: str | None
    offset_scale: float

    @classmethod
    def from_operation(cls, operation: FigureOperationState) -> LineTransformPlan:
        return cls(
            iter_dim=operation.line_iter_dim,
            normalize=operation.line_normalize,
            scales=operation.line_scales,
            offsets=operation.line_offsets,
            offset_source=operation.line_offset_source,
            offset_coord=operation.line_offset_coord,
            offset_scale=operation.line_offset_scale,
        )


def line_normalize_text(mode: str) -> str:
    return _NORMALIZE_TEXT.get(mode, _NORMALIZE_TEXT["none"])


def line_normalize_from_text(
    text: str,
) -> typing.Literal["none", "max", "mean"]:
    for mode, label in _NORMALIZE_TEXT.items():
        if text == label:
            return typing.cast("typing.Literal['none', 'max', 'mean']", mode)
    return "none"


def line_transform_active(operation: FigureOperationState) -> bool:
    return line_transform_plan_active(LineTransformPlan.from_operation(operation))


def line_transform_plan_active(plan: LineTransformPlan) -> bool:
    return (
        plan.normalize != "none"
        or bool(plan.scales)
        or line_transform_plan_uses_offsets(plan)
    )


def transform_profiles(
    operation: FigureOperationState, profiles: Sequence[xr.DataArray]
) -> list[xr.DataArray]:
    return transform_profiles_from_plan(
        LineTransformPlan.from_operation(operation),
        profiles,
    )


def transform_profiles_from_plan(
    plan: LineTransformPlan, profiles: Sequence[xr.DataArray]
) -> list[xr.DataArray]:
    if not line_transform_plan_active(plan):
        return list(profiles)
    scales = line_transform_values(plan.scales, len(profiles), default=1.0)
    offsets = line_offsets_for_plan(plan, profiles)
    return [
        offset + scale * normalize_line_data(profile, plan.normalize)
        for profile, scale, offset in zip(profiles, scales, offsets, strict=True)
    ]


def normalize_line_data(
    line_data: xr.DataArray, mode: typing.Literal["none", "max", "mean"]
) -> xr.DataArray:
    if mode == "none":
        return line_data
    scale = line_normalization_scale(line_data, mode)
    return line_data / scale


def line_normalization_scale(
    line_data: xr.DataArray, mode: typing.Literal["max", "mean"]
) -> float:
    if mode == "max":
        scale = float(line_data.max(skipna=True))
    else:
        scale = float(line_data.mean(skipna=True))
    if math.isfinite(scale) and scale != 0.0:
        return scale
    raise ValueError(
        f"Cannot normalize profile by {mode}: normalization factor is {scale!r}"
    )


def validate_line_normalization(
    operation: FigureOperationState, profiles: Sequence[xr.DataArray]
) -> None:
    if operation.line_normalize == "none":
        return
    for profile in profiles:
        line_normalization_scale(profile, operation.line_normalize)


def line_offsets_for_profiles(
    operation: FigureOperationState,
    profiles: Sequence[xr.DataArray],
) -> tuple[float, ...]:
    return line_offsets_for_plan(LineTransformPlan.from_operation(operation), profiles)


def line_offsets_for_plan(
    plan: LineTransformPlan,
    profiles: Sequence[xr.DataArray],
) -> tuple[float, ...]:
    count = len(profiles)
    if plan.offset_source == "manual":
        return line_transform_values(plan.offsets, count, default=0.0)
    if plan.offset_source == "index":
        offsets = tuple(float(index) for index in range(count))
    else:
        coord_name = line_offset_coordinate_name_from_plan(plan)
        offsets = tuple(
            profile_scalar_coord_value(profile, coord_name) for profile in profiles
        )
    if plan.offset_scale == 1.0:
        return offsets
    return tuple(plan.offset_scale * offset for offset in offsets)


def line_offset_coordinate_name(operation: FigureOperationState) -> str:
    return line_offset_coordinate_name_from_plan(
        LineTransformPlan.from_operation(operation)
    )


def line_offset_coordinate_name_from_plan(plan: LineTransformPlan) -> str:
    if plan.offset_source == "coordinate":
        if plan.iter_dim is None:
            raise ValueError("Coordinate offsets require One profile per")
        return plan.iter_dim
    if plan.offset_coord is None:
        raise ValueError("Associated-coordinate offsets require a coordinate")
    return plan.offset_coord


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
    return line_transform_plan_uses_offsets(LineTransformPlan.from_operation(operation))


def line_transform_plan_uses_offsets(plan: LineTransformPlan) -> bool:
    return plan.offset_source != "manual" or bool(plan.offsets)


def profile_transform_code_lines(
    operation: FigureOperationState,
    *,
    profiles: Sequence[xr.DataArray] | None = None,
    profiles_name: str = "profiles",
    input_name: str | None = None,
) -> list[str]:
    if not line_transform_active(operation):
        return []
    if profiles is not None:
        validate_line_normalization(operation, profiles)

    source_name = input_name or profiles_name
    lines: list[str] = []
    loop_names = ["profile"]
    loop_values = [source_name]
    scale_name = _profile_scale_code(operation, loop_names, loop_values)
    offset_name = _profile_offset_code(
        operation, loop_names, loop_values, profiles_name=source_name
    )
    value_expr = _profile_transform_expression(
        operation, scale_name=scale_name, offset_name=offset_name
    )
    lines.append(f"{profiles_name} = [")
    lines.append(f"    {value_expr}")
    if len(loop_names) == 1:
        lines.append(f"    for profile in {source_name}")
    else:
        lines.append("    for " + ", ".join(loop_names) + " in zip(")
        lines.extend(f"        {value}," for value in loop_values)
        lines.append("        strict=True,")
        lines.append("    )")
    lines.append("]")
    return lines


def profile_stack_transform_code(
    operation: FigureOperationState,
    *,
    data_name: str,
    line_data: xr.DataArray,
) -> str | None:
    if not line_transform_active(operation):
        return None
    if operation.line_iter_dim is None or operation.line_iter_dim not in line_data.dims:
        return None
    profile_dims = tuple(
        str(dim) for dim in line_data.dims if dim != operation.line_iter_dim
    )
    value_expr = _profile_stack_normalize_expression(
        operation, data_name=data_name, profile_dims=profile_dims
    )
    if value_expr is None:
        return None
    scale_expr = _profile_stack_scale_expression(operation)
    if operation.line_scales and scale_expr is None:
        return None
    offset_expr = _profile_stack_offset_expression(operation, data_name=data_name)
    if line_uses_offsets(operation) and offset_expr is None:
        return None
    return _profile_transform_expression(
        operation,
        value_expr=value_expr,
        scale_name=scale_expr,
        offset_name=offset_expr,
    )


def _profile_stack_normalize_expression(
    operation: FigureOperationState, *, data_name: str, profile_dims: Sequence[str]
) -> str | None:
    if operation.line_normalize == "none":
        return data_name
    if len(profile_dims) != 1:
        return None
    reducer = "max" if operation.line_normalize == "max" else "mean"
    return f"{data_name} / {data_name}.{reducer}({profile_dims[0]!r}, skipna=True)"


def _profile_stack_scale_expression(operation: FigureOperationState) -> str | None:
    if not operation.line_scales:
        return None
    if len(operation.line_scales) == 1:
        return repr(operation.line_scales[0])
    return None


def _profile_stack_offset_expression(
    operation: FigureOperationState, *, data_name: str
) -> str | None:
    if not line_uses_offsets(operation):
        return None
    if operation.line_offset_source == "manual":
        if len(operation.line_offsets) == 1:
            return repr(operation.line_offsets[0])
        return None
    if operation.line_offset_source == "index":
        return None
    coord_name = line_offset_coordinate_name(operation)
    offset_expr = f"{data_name}[{coord_name!r}]"
    if operation.line_offset_scale != 1.0:
        offset_expr = f"{operation.line_offset_scale!r} * {offset_expr}"
    return offset_expr


def _profile_scale_code(
    operation: FigureOperationState,
    loop_names: list[str],
    loop_values: list[str],
) -> str | None:
    if not operation.line_scales:
        return None
    if len(operation.line_scales) == 1:
        return repr(operation.line_scales[0])
    loop_names.append("scale")
    loop_values.append(repr(list(operation.line_scales)))
    return "scale"


def _profile_offset_code(
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
            return repr(operation.line_offsets[0])
        offset_values = repr(list(operation.line_offsets))
    elif operation.line_offset_source == "index":
        offset_values = _offset_list_code("float(index)", operation, profiles_name)
    else:
        coord_name = line_offset_coordinate_name(operation)
        offset_values = _offset_list_code(
            f"float(profile[{coord_name!r}])", operation, profiles_name
        )
    loop_names.append("offset")
    loop_values.append(offset_values)
    return "offset"


def _offset_list_code(
    value_expr: str, operation: FigureOperationState, profiles_name: str
) -> str:
    if operation.line_offset_scale != 1.0:
        value_expr = f"{operation.line_offset_scale!r} * {value_expr}"
    if operation.line_offset_source == "index":
        return f"[{value_expr} for index in range(len({profiles_name}))]"
    return f"[{value_expr} for profile in {profiles_name}]"


def _profile_transform_expression(
    operation: FigureOperationState,
    *,
    value_expr: str | None = None,
    scale_name: str | None,
    offset_name: str | None,
) -> str:
    if value_expr is None:
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
