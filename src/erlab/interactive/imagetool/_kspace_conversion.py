"""Shared momentum-conversion helpers for ImageTool and ktool."""

from __future__ import annotations

import contextlib
import dataclasses
import html
import math
import typing
import warnings

import numpy as np
import psutil

import erlab
from erlab.accessors.kspace import IncompleteDataError
from erlab.constants import AxesConfiguration
from erlab.interactive.imagetool import provenance

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import xarray as xr
    from qtpy import QtWidgets

    from erlab.interactive.imagetool.viewer import ImageSlicerArea

_MISSING_WORK_FUNCTION_WARNING_RE = (
    r"^Work function not found in data attributes, assuming 4\.5 eV$"
)
_MISSING_INNER_POTENTIAL_WARNING_RE = (
    r"^Inner potential not found in data attributes, assuming 10 eV$"
)
KSPACE_CONVERSION_GROUP_KIND = "kspace_conversion"
_FOCUS_CONFIGURATION = "configuration"
_FOCUS_WORK_FUNCTION = "work_function"
_FOCUS_INNER_POTENTIAL = "inner_potential"
_FOCUS_NORMAL = "normal_emission"
_FOCUS_CONVERT = "bounds_resolution"

_KSPACE_SETUP_TYPES = (
    provenance.KspaceConfigurationOperation,
    provenance.KspaceWorkFunctionOperation,
    provenance.KspaceInnerPotentialOperation,
    provenance.KspaceSetNormalOperation,
)
_KSPACE_CONVERSION_TYPES = (*_KSPACE_SETUP_TYPES, provenance.KspaceConvertOperation)
_GIB = 1024**3


@dataclasses.dataclass(frozen=True)
class KspaceMemoryBudget:
    """Current physical-memory budget used by interactive conversion guards."""

    total_bytes: int
    available_bytes: int
    reserve_bytes: int
    safe_budget_bytes: int


@dataclasses.dataclass(frozen=True)
class KspaceConversionEstimate:
    """Estimated destination size and memory use for one momentum conversion."""

    input_dims: tuple[str, ...]
    output_dims: tuple[str, ...]
    axis_sizes: dict[str, int]
    output_sizes: dict[str, int]
    bounds: dict[str, tuple[float, float]]
    resolution: dict[str, float]
    total_points: int
    final_bytes: int
    peak_bytes: int
    memory: KspaceMemoryBudget

    @property
    def is_safe(self) -> bool:
        return self.final_bytes <= self.memory.available_bytes


class KspaceConversionMemoryError(MemoryError):
    """Raised when an interactive momentum conversion exceeds the memory budget."""

    def __init__(self, estimate: KspaceConversionEstimate) -> None:
        self.estimate = estimate
        super().__init__(
            "The requested momentum grid is too large for currently available memory."
        )


@contextlib.contextmanager
def ignore_missing_kspace_parameter_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_MISSING_WORK_FUNCTION_WARNING_RE,
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=_MISSING_INNER_POTENTIAL_WARNING_RE,
            category=UserWarning,
        )
        yield


def configuration_text(configuration: AxesConfiguration | int) -> str:
    configuration = AxesConfiguration(int(configuration))
    return f"Configuration {int(configuration)} ({configuration.name})"


def initial_normal_emission_from_slicer_area(
    slicer_area: ImageSlicerArea,
) -> tuple[tuple[float, float] | None, float | None]:
    """Return ktool-compatible normal-emission seed values from an ImageTool."""
    data = slicer_area.data
    dim_values = {
        str(dim): float(value)
        for dim, value in zip(data.dims, slicer_area.current_values, strict=True)
    }
    beta_value = dim_values.get("beta")
    if beta_value is None and "beta" in data.coords:
        beta_coord = data["beta"]
        if beta_coord.size == 1:
            beta_value = float(beta_coord.values)
    if "alpha" not in dim_values or beta_value is None:
        return None, None

    initial_normal_emission: tuple[float, float] = (dim_values["alpha"], beta_value)
    initial_delta: float | None = None
    guideline_dims = tuple(
        str(data.dims[axis]) for axis in slicer_area.main_image.display_axis
    )
    if slicer_area.main_image.is_guidelines_visible and set(guideline_dims) == {
        "alpha",
        "beta",
    }:
        guideline_values: dict[str, float] = {}
        for axis, value in zip(
            slicer_area.main_image.display_axis,
            slicer_area.main_image._guideline_offset,
            strict=True,
        ):
            dim = str(data.dims[axis])
            if axis in slicer_area.array_slicer._nonuniform_axes_set:
                value = float(
                    np.interp(
                        value,
                        slicer_area.array_slicer.coords_uniform[axis],
                        slicer_area.array_slicer.coords[axis],
                    )
                )
            guideline_values[dim] = float(value)
        initial_normal_emission = (
            guideline_values["alpha"],
            guideline_values["beta"],
        )
        initial_delta = -slicer_area.main_image._guideline_angle

    return initial_normal_emission, initial_delta


def kspace_work_function(data: xr.DataArray) -> float:
    with ignore_missing_kspace_parameter_warnings():
        return float(data.kspace.work_function)


def kspace_inner_potential(data: xr.DataArray) -> float:
    with ignore_missing_kspace_parameter_warnings():
        return float(data.kspace.inner_potential)


def rounded_spin_value(spin: QtWidgets.QDoubleSpinBox) -> float:
    return float(np.round(spin.value(), spin.decimals()))


def system_memory_budget() -> KspaceMemoryBudget:
    """Return the adaptive physical-memory budget for interactive conversions."""
    memory = psutil.virtual_memory()
    total = int(memory.total)
    available = int(memory.available)
    reserve = min(max(2 * _GIB, int(0.2 * total)), int(0.5 * available))
    return KspaceMemoryBudget(
        total_bytes=total,
        available_bytes=available,
        reserve_bytes=reserve,
        safe_budget_bytes=max(0, available - reserve),
    )


def _validated_bounds(
    axis: str,
    lims: tuple[float, float],
) -> tuple[float, float]:
    lower, upper = float(lims[0]), float(lims[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"Momentum bounds for {axis!r} must be finite.")
    if upper < lower:
        raise ValueError(f"Momentum bounds for {axis!r} must be increasing.")
    return lower, upper


def _validated_resolution(axis: str, resolution: float) -> float:
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError(f"Momentum resolution for {axis!r} must be positive.")
    return resolution


def _momentum_axis_size(lims: tuple[float, float], resolution: float) -> int:
    return max(1, round((lims[1] - lims[0]) / resolution + 1))


def _conversion_dimension_mapping(
    data: xr.DataArray,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    output_dims: list[str]
    if data.kspace._has_hv:
        output_dims = [data.kspace.slit_axis, "kz"]
    elif data.kspace._has_beta:
        output_dims = ["kx", "ky"]
    else:
        output_dims = [data.kspace.slit_axis]

    input_dims: list[str] = []
    for dim in tuple(output_dims):
        if dim == data.kspace.slit_axis and "alpha" in data.dims:
            input_dims.append("alpha")
        elif dim == data.kspace.other_axis and "beta" in data.dims:
            input_dims.append("beta")
        elif dim == "kz" and "hv" in data.dims:
            input_dims.append("hv")

    if data.kspace._has_eV and "eV" in data.dims:
        input_dims.append("eV")
        output_dims.append("eV")

    return tuple(input_dims), tuple(output_dims)


def estimate_kspace_conversion(
    data: xr.DataArray,
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
    resolution: dict[str, float] | None = None,
    memory: KspaceMemoryBudget | None = None,
) -> KspaceConversionEstimate:
    """Estimate interactive momentum-conversion allocation size without converting."""
    if bounds is None:
        bounds = {}
    if resolution is None:
        resolution = {}
    if memory is None:
        memory = system_memory_budget()

    effective_bounds: dict[str, tuple[float, float]] = {}
    effective_resolution: dict[str, float] = {}
    axis_sizes: dict[str, int] = {}
    estimated_bounds = data.kspace.estimate_bounds()
    for axis, estimated_lims in estimated_bounds.items():
        if axis not in data.kspace.momentum_axes:
            continue

        lims = _validated_bounds(axis, bounds.get(axis, estimated_lims))
        res = data.kspace.estimate_resolution(axis, lims, from_numpoints=False)
        if axis == "kz":
            res_n = data.kspace.estimate_resolution(axis, lims, from_numpoints=True)
            res = min(res, res_n)
        res = _validated_resolution(axis, resolution.get(axis, res))

        effective_bounds[axis] = lims
        effective_resolution[axis] = res
        axis_sizes[axis] = _momentum_axis_size(lims, res)

    input_dims, output_dims = _conversion_dimension_mapping(data)
    output_sizes: dict[str, int] = {}
    for dim in data.dims:
        if str(dim) not in input_dims:
            output_sizes[str(dim)] = int(data.sizes[dim])
    for dim in output_dims:
        if dim in axis_sizes:
            output_sizes[dim] = axis_sizes[dim]
        elif dim in data.sizes:
            output_sizes[dim] = int(data.sizes[dim])

    total_points = math.prod(output_sizes.values()) if output_sizes else 1
    final_bytes = int(total_points * np.dtype(np.float64).itemsize)
    peak_bytes = int(final_bytes * (len(input_dims) + 3))
    return KspaceConversionEstimate(
        input_dims=input_dims,
        output_dims=output_dims,
        axis_sizes=axis_sizes,
        output_sizes=output_sizes,
        bounds=effective_bounds,
        resolution=effective_resolution,
        total_points=total_points,
        final_bytes=final_bytes,
        peak_bytes=peak_bytes,
        memory=memory,
    )


def validate_kspace_conversion_memory(
    data: xr.DataArray,
    *,
    bounds: dict[str, tuple[float, float]] | None,
    resolution: dict[str, float] | None,
) -> KspaceConversionEstimate:
    """Return an estimate or raise if the interactive conversion is unsafe."""
    estimate = estimate_kspace_conversion(
        data,
        bounds=bounds,
        resolution=resolution,
    )
    if not estimate.is_safe:
        raise KspaceConversionMemoryError(estimate)
    return estimate


def _format_bytes(value: int) -> str:
    return erlab.utils.formatting.format_nbytes(value)


def _format_sizes(sizes: Mapping[str, int]) -> str:
    if not sizes:
        return "scalar"
    return " × ".join(f"{axis}={size}" for axis, size in sizes.items())


def _format_numeric_mapping(
    mapping: Mapping[str, float | tuple[float, float]],
) -> str:
    parts: list[str] = []
    for axis, value in mapping.items():
        if isinstance(value, tuple):
            parts.append(f"{axis}=({value[0]:.6g}, {value[1]:.6g})")
        else:
            parts.append(f"{axis}={value:.6g}")
    return ", ".join(parts) if parts else "automatic"


def kspace_conversion_estimate_text(
    estimate: KspaceConversionEstimate,
    *,
    preview: bool = False,
) -> str:
    """Return concise inline feedback for a conversion estimate."""
    estimate_summary = (
        f"Output: {_format_sizes(estimate.output_sizes)}\n"
        f"Final array: {_format_bytes(estimate.final_bytes)}\n"
        f"Available memory: {_format_bytes(estimate.memory.available_bytes)}"
    )
    if estimate.is_safe:
        return estimate_summary
    if preview:
        return (
            "Preview unavailable.\n"
            f"{estimate_summary}\n"
            "Increase resolution or reduce bounds."
        )
    return (
        "Conversion unavailable.\n"
        f"{estimate_summary}\n"
        "Increase resolution or reduce bounds."
    )


def kspace_conversion_memory_dialog_title() -> str:
    return "Conversion Cannot Be Completed Safely"


def kspace_conversion_memory_dialog_text() -> str:
    return "The requested momentum grid is too large for currently available memory."


def kspace_conversion_memory_dialog_info(
    estimate: KspaceConversionEstimate,
) -> str:
    return (
        f"Final array estimate: {_format_bytes(estimate.final_bytes)}. "
        f"Available physical memory: {_format_bytes(estimate.memory.available_bytes)}. "
        "Increase the resolution value or reduce the bounds."
    )


def kspace_conversion_memory_dialog_details(
    estimate: KspaceConversionEstimate,
) -> str:
    lines = [
        f"Axis sizes: {_format_sizes(estimate.axis_sizes)}",
        f"Output sizes: {_format_sizes(estimate.output_sizes)}",
        f"Bounds: {_format_numeric_mapping(estimate.bounds)}",
        f"Resolution: {_format_numeric_mapping(estimate.resolution)}",
        f"Final array estimate: {_format_bytes(estimate.final_bytes)}",
        f"Available physical memory: {_format_bytes(estimate.memory.available_bytes)}",
        f"Peak estimate: {_format_bytes(estimate.peak_bytes)}",
        f"Reserve: {_format_bytes(estimate.memory.reserve_bytes)}",
        "Swap is intentionally not treated as usable memory.",
    ]
    return "<br>".join(html.escape(line) for line in lines)


def normal_emission_angles(
    data: xr.DataArray,
    offsets: Mapping[str, float],
) -> tuple[float, float]:
    if "xi" not in data.coords:
        raise IncompleteDataError("coord", "xi")

    angle_params = {
        "delta": offsets["delta"],
        "xi": float(data["xi"].values),
        "xi0": offsets["xi"],
    }

    match data.kspace.configuration:
        case AxesConfiguration.Type1 | AxesConfiguration.Type2:
            angle_params["beta0"] = offsets["beta"]
        case _:
            if "chi" not in data.coords:
                raise IncompleteDataError("coord", "chi")
            angle_params["chi"] = float(data["chi"].values)
            angle_params["chi0"] = offsets["chi"]

    alpha, beta = erlab.analysis.kspace._normal_emission_from_angle_params(
        data.kspace.configuration, angle_params
    )
    return (
        float(np.round(alpha / data.kspace.alpha_scale, 5)),
        float(np.round(beta / data.kspace.beta_scale, 5)),
    )


def apply_kspace_parameters(
    data: xr.DataArray,
    *,
    source_data: xr.DataArray,
    work_function: float,
    inner_potential: float | None,
    force_work_function: bool = False,
    force_inner_potential: bool = False,
    offsets: Mapping[str, float] | None = None,
    normal_emission: tuple[float, float] | None = None,
    delta: float | None = None,
    alpha_scale: float | None = None,
    beta_scale: float | None = None,
) -> xr.DataArray:
    if offsets is not None:
        data.kspace.offsets = offsets
    elif normal_emission is not None:
        data.kspace.set_normal(
            normal_emission[0],
            normal_emission[1],
            delta=delta,
            alpha_scale=alpha_scale,
            beta_scale=beta_scale,
        )

    if data.kspace._has_hv and inner_potential is not None:
        with ignore_missing_kspace_parameter_warnings():
            current_v0 = source_data.kspace.inner_potential
        if (
            force_inner_potential
            or "inner_potential" not in source_data.attrs
            or not np.isclose(inner_potential, current_v0)
        ):
            data.kspace.inner_potential = inner_potential

    with ignore_missing_kspace_parameter_warnings():
        current_wf = source_data.kspace.work_function
    if (
        force_work_function
        or "sample_workfunction" not in source_data.attrs
        or not np.isclose(work_function, current_wf)
    ):
        data.kspace.work_function = work_function
    return data


def kspace_conversion_operations(
    source_data: xr.DataArray,
    *,
    target_configuration: AxesConfiguration | int,
    source_configuration: AxesConfiguration | int | None = None,
    work_function: float,
    inner_potential: float | None,
    normal_emission: tuple[float, float],
    delta: float | None,
    bounds: dict[str, tuple[float, float]] | None,
    resolution: dict[str, float] | None,
    force_scalars: bool,
    alpha_scale: float | None = None,
    beta_scale: float | None = None,
) -> tuple[provenance.ToolProvenanceOperation, ...]:
    operations: list[provenance.ToolProvenanceOperation] = []
    focuses: list[str] = []
    target_configuration = AxesConfiguration(int(target_configuration))
    if source_configuration is None:
        source_configuration = source_data.kspace.configuration
    if int(target_configuration) != int(source_configuration):
        operations.append(
            provenance.KspaceConfigurationOperation(
                configuration=int(target_configuration)
            )
        )
        focuses.append(_FOCUS_CONFIGURATION)
        configured_data = source_data.kspace.as_configuration(target_configuration)
    else:
        configured_data = source_data

    if configured_data.kspace._has_hv and inner_potential is not None:
        with ignore_missing_kspace_parameter_warnings():
            current_v0 = configured_data.kspace.inner_potential
        if force_scalars or not np.isclose(inner_potential, current_v0):
            operations.append(
                provenance.KspaceInnerPotentialOperation(
                    inner_potential=float(inner_potential)
                )
            )
            focuses.append(_FOCUS_INNER_POTENTIAL)

    with ignore_missing_kspace_parameter_warnings():
        current_wf = configured_data.kspace.work_function
    if force_scalars or not np.isclose(work_function, current_wf):
        operations.append(
            provenance.KspaceWorkFunctionOperation(work_function=float(work_function))
        )
        focuses.append(_FOCUS_WORK_FUNCTION)

    if alpha_scale is None:
        alpha_scale = configured_data.kspace.alpha_scale
    if beta_scale is None:
        beta_scale = configured_data.kspace.beta_scale

    operations.append(
        provenance.KspaceSetNormalOperation(
            alpha=float(normal_emission[0]),
            beta=float(normal_emission[1]),
            delta=None if delta is None else float(delta),
            alpha_scale=None if np.isclose(alpha_scale, 1.0) else float(alpha_scale),
            beta_scale=None if np.isclose(beta_scale, 1.0) else float(beta_scale),
        )
    )
    focuses.append(_FOCUS_NORMAL)
    operations.append(
        provenance.KspaceConvertOperation(bounds=bounds, resolution=resolution)
    )
    focuses.append(_FOCUS_CONVERT)
    return provenance.stamp_operation_group(
        operations,
        kind=KSPACE_CONVERSION_GROUP_KIND,
        focuses=focuses,
    )


def _focus_for_operation(
    operation: provenance.ToolProvenanceOperation,
) -> str | None:
    if isinstance(operation, provenance.KspaceConfigurationOperation):
        return _FOCUS_CONFIGURATION
    if isinstance(operation, provenance.KspaceWorkFunctionOperation):
        return _FOCUS_WORK_FUNCTION
    if isinstance(operation, provenance.KspaceInnerPotentialOperation):
        return _FOCUS_INNER_POTENTIAL
    if isinstance(operation, provenance.KspaceSetNormalOperation):
        return _FOCUS_NORMAL
    if isinstance(operation, provenance.KspaceConvertOperation):
        return _FOCUS_CONVERT
    return None


def _complete_kspace_conversion_group(
    operations: Sequence[provenance.ToolProvenanceOperation],
) -> bool:
    if not operations or not isinstance(
        operations[-1],
        provenance.KspaceConvertOperation,
    ):
        return False
    setup_operations = operations[:-1]
    if not all(
        isinstance(operation, _KSPACE_SETUP_TYPES) for operation in setup_operations
    ):
        return False
    if (
        sum(
            isinstance(operation, provenance.KspaceSetNormalOperation)
            for operation in setup_operations
        )
        != 1
    ):
        return False
    for operation_type in _KSPACE_SETUP_TYPES:
        if (
            sum(isinstance(operation, operation_type) for operation in setup_operations)
            > 1
        ):
            return False
    return all(
        isinstance(operation, _KSPACE_CONVERSION_TYPES) for operation in operations
    )


def is_kspace_conversion_group(
    operations: Sequence[provenance.ToolProvenanceOperation],
    operation_index: int,
) -> tuple[int, int] | None:
    """Return the contiguous momentum-conversion operation range for an operation."""
    group = provenance.operation_group_range(
        operations,
        operation_index,
        kind=KSPACE_CONVERSION_GROUP_KIND,
    )
    if group is None:
        return None
    if not _complete_kspace_conversion_group(operations[group[0] : group[1]]):
        return None
    return group


def stamp_kspace_conversion_groups(
    operations: Sequence[provenance.ToolProvenanceOperation],
) -> tuple[provenance.ToolProvenanceOperation, ...]:
    """Stamp complete ungrouped kspace conversion runs in a console operation chain."""
    output = list(operations)
    index = 0
    while index < len(output):
        operation = output[index]
        if operation.group is not None or not isinstance(
            operation,
            _KSPACE_CONVERSION_TYPES,
        ):
            index += 1
            continue
        start = index
        while (
            index < len(output)
            and output[index].group is None
            and isinstance(
                output[index],
                _KSPACE_CONVERSION_TYPES,
            )
        ):
            if isinstance(output[index], provenance.KspaceConvertOperation):
                index += 1
                break
            index += 1
        stop = index
        if _complete_kspace_conversion_group(output[start:stop]):
            stamped = provenance.stamp_operation_group(
                output[start:stop],
                kind=KSPACE_CONVERSION_GROUP_KIND,
                focuses=tuple(
                    _focus_for_operation(group_operation)
                    for group_operation in output[start:stop]
                ),
            )
            output[start:stop] = stamped
        else:
            index = max(stop, start + 1)
    return tuple(output)


def incomplete_kspace_conversion_edit_reason(
    operation: provenance.ToolProvenanceOperation,
) -> str | None:
    """Return the edit tooltip/message for standalone primitive kspace rows."""
    if not isinstance(operation, _KSPACE_CONVERSION_TYPES):
        return None
    return (
        "This kspace step is structured and replayable, but it is not part of a "
        "complete editable momentum-conversion group. To edit it in the conversion "
        "dialog, include the full operation set: `Set normal emission` and "
        "`Convert to momentum`. Configuration, work-function, and inner-potential "
        "steps are optional setup rows."
    )
