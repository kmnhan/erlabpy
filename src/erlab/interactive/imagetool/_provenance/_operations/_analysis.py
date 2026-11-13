"""Filtering, reduction, and image-analysis provenance operations."""

from __future__ import annotations

import contextlib
import inspect
import keyword
import os
import typing
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from numbers import Integral

import numpy as np
import pydantic
import xarray as xr

import erlab
from erlab.interactive.imagetool._provenance._code import (
    _DATASET_MARKER,
    _FIT_DATASET_MARKER,
    _TUPLE_MARKER,
    _provenance_value_code,
)
from erlab.interactive.imagetool._provenance._model import (
    ConsoleCall,
    ConsoleOperationPattern,
    DerivationEntry,
    ProvenanceHashable,
    ProvenanceHashablePair,
    ProvenanceHashableTuple,
    ProvenanceIntMapping,
    ProvenanceKeywords,
    ProvenanceMapping,
    ProvenanceNumericVector,
    ToolProvenanceOperation,
    _coerce_float_mapping_field,
    _console_mapping_values,
    _console_values_equal,
    _ensure_float_tuple,
    _format_derivation_value,
    _is_identifier_string_mapping,
    decode_provenance_value,
    encode_provenance_value,
)


def _require_integer_mapping(value: typing.Any, *, field_name: str) -> typing.Any:
    """Reject lossy integer coercion before provenance mapping normalization."""
    decoded = decode_provenance_value(value)
    if not isinstance(decoded, Mapping):
        return value
    if any(
        isinstance(item, bool) or not isinstance(item, Integral)
        for item in decoded.values()
    ):
        raise TypeError(f"{field_name} must be integers")
    return value


class ConcatOperation(ToolProvenanceOperation):
    """Concatenate the current array and ordered, separately bound input arrays."""

    xarray_option_names: typing.ClassVar[tuple[str, ...]] = ()
    op: typing.Literal["concat"] = "concat"
    live_applicable: typing.ClassVar[bool] = False
    input_count: int = pydantic.Field(strict=True, ge=2)
    dim: str = pydantic.Field(strict=True, min_length=1)
    coords: typing.Literal["minimal", "different", "all"] = "minimal"
    compat: typing.Literal[
        "identical", "equals", "broadcast_equals", "no_conflicts", "override"
    ] = "override"
    join: typing.Literal["outer", "inner", "left", "right", "exact", "override"] = (
        "outer"
    )
    combine_attrs: typing.Literal[
        "drop", "identical", "no_conflicts", "drop_conflicts", "override"
    ] = "override"

    @pydantic.model_validator(mode="after")
    def _validate_concat_options(self) -> typing.Self:
        if self.coords == "different" and self.compat == "override":
            raise ValueError("coords='different' cannot be used with compat='override'")
        return self

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "dim": self.dim,
            "coords": self.coords,
            "compat": self.compat,
            "join": self.join,
            "combine_attrs": self.combine_attrs,
        }

    def input_slots(self) -> tuple[str, ...]:
        return tuple(f"other_{index}" for index in range(self.input_count - 1))

    def apply_with_inputs(
        self,
        data: xr.DataArray,
        inputs: Mapping[str, xr.DataArray],
        *,
        authorization: object | None = None,
    ) -> xr.DataArray:
        del authorization
        return xr.concat(
            [data, *(inputs[slot] for slot in self.input_slots())], **self.kwargs
        )

    def derivation_label(self) -> str:
        return f"Concatenate({_format_derivation_value(self.kwargs)})"

    def expression_code_with_inputs(
        self,
        input_name: str,
        inputs: Mapping[str, str],
        *,
        source_name: str | None = None,
    ) -> str:
        del source_name
        input_names = [input_name, *(inputs[slot] for slot in self.input_slots())]
        kwargs = ", ".join(
            f"{name}={_provenance_value_code(value)}"
            for name, value in self.kwargs.items()
        )
        return f"xr.concat([{', '.join(input_names)}], {kwargs})"


class DenoiseOperation(ToolProvenanceOperation):
    """Record the inputs for machine learning denoising."""

    op: typing.Literal["denoise"] = "denoise"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.image.ml._denoise.denoise",
            fields=("resize",),
            defaults={
                "resize": (300, 300),
                "along": "alpha",
                "model_path": None,
                "tta": True,
                "restore_scale": True,
                "restore_size": True,
                "num_threads": 0,
            },
        ),
    )
    resize: int | tuple[int, int] = (300, 300)
    along: str = "alpha"
    model_path: str | None = None
    tta: pydantic.StrictBool = True
    restore_scale: pydantic.StrictBool = True
    restore_size: pydantic.StrictBool = True
    num_threads: typing.Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] = 0

    @pydantic.field_validator("resize", mode="before")
    @classmethod
    def _validate_resize(cls, value: typing.Any) -> int | tuple[int, int]:
        from erlab.analysis.image.ml._denoise import _parse_shape

        shape = _parse_shape(value)
        return int(value) if isinstance(value, Integral) else shape

    @pydantic.field_validator("model_path", mode="before")
    @classmethod
    def _validate_model_path(cls, value: typing.Any) -> str | None:
        return None if value is None else os.fsdecode(value)

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "resize": self.resize,
            "along": self.along,
            "model_path": self.model_path,
            "tta": self.tta,
            "restore_scale": self.restore_scale,
            "restore_size": self.restore_size,
            "num_threads": self.num_threads,
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.image.ml.denoise(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Denoise({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.image.ml.denoise,
            [f"|{input_name}|"],
            self.kwargs,
            module="era.image.ml",
            name="denoise",
            remove_defaults=False,
        )


_ROTATION_DEFAULTS = {
    "axes": (0, 1),
    "center": (0.0, 0.0),
    "reshape": True,
    "order": 1,
    "mode": "constant",
    "cval": np.nan,
    "prefilter": True,
}


class _RotationPlaneOperation(ToolProvenanceOperation):
    batch_available: typing.ClassVar[bool] = True
    axes: ProvenanceHashablePair = (0, 1)
    center: tuple[float, float] | dict[Hashable, float] = (0.0, 0.0)
    reshape: bool = True
    order: int = 1
    mode: str = "constant"
    cval: float = pydantic.Field(default=np.nan, exclude_if=np.isnan)
    prefilter: bool = True

    @pydantic.field_validator("center", mode="before")
    @classmethod
    def _validate_center(cls, value: typing.Any) -> typing.Any:
        decoded = decode_provenance_value(value)
        if isinstance(decoded, Mapping):
            return _coerce_float_mapping_field(decoded)
        return _ensure_float_tuple(
            typing.cast("Sequence[typing.Any]", decoded), expected_len=2
        )

    @pydantic.field_serializer("center", when_used="json")
    def _serialize_center(self, value: typing.Any) -> typing.Any:
        return encode_provenance_value(value)

    @property
    def _plane_kwargs(self) -> dict[str, typing.Any]:
        kwargs: dict[str, typing.Any] = {
            "axes": self.axes,
            "center": self.center,
            "reshape": self.reshape,
            "order": self.order,
        }
        if self.mode != "constant":
            kwargs["mode"] = self.mode
        if not np.isnan(self.cval):
            kwargs["cval"] = self.cval
        if not self.prefilter:
            kwargs["prefilter"] = False
        return kwargs

    def _rotation_code(
        self,
        func: Callable[..., xr.DataArray],
        input_name: str,
        kwargs: dict[str, typing.Any],
    ) -> str:
        if self.axes != (0, 1):
            kwargs["axes"] = f"|{_provenance_value_code(self.axes)}|"
        if "cval" in kwargs:
            kwargs["cval"] = f"|{_provenance_value_code(self.cval)}|"
        if isinstance(self.center, Mapping):
            kwargs["center"] = f"|{_provenance_value_code(self.center)}|"
        return erlab.interactive.utils.generate_code(
            func, [f"|{input_name}|"], kwargs, module="era.transform"
        )


class RotateOperation(_RotationPlaneOperation):
    op: typing.Literal["rotate"] = "rotate"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.transform.rotate",
            fields=("angle", "axes", "center"),
            defaults=_ROTATION_DEFAULTS,
        ),
    )
    angle: float

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {"angle": self.angle, **self._plane_kwargs}

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.rotate(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Rotate({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return self._rotation_code(
            erlab.analysis.transform.rotate, input_name, self.kwargs
        )


ShiftMode = typing.Literal[
    "reflect",
    "grid-mirror",
    "constant",
    "grid-constant",
    "nearest",
    "mirror",
    "grid-wrap",
    "wrap",
]
SHIFT_MODES: tuple[ShiftMode, ...] = (
    "reflect",
    "grid-mirror",
    "constant",
    "grid-constant",
    "nearest",
    "mirror",
    "grid-wrap",
    "wrap",
)


class _BaseShiftOperation(ToolProvenanceOperation):
    """Common interpolation options for scalar and bound-input shifts."""

    batch_available: typing.ClassVar[bool] = False
    negate: bool = False
    along: ProvenanceHashable
    shift_coords: bool = False
    keep_dim_order: bool = True
    assume_sorted: bool = False
    order: int = pydantic.Field(default=1, ge=0, le=5)
    mode: ShiftMode = "constant"
    cval: float = np.nan
    prefilter: bool = False

    @pydantic.field_validator("cval", mode="before")
    @classmethod
    def _restore_nan_fill_value(cls, value: typing.Any) -> typing.Any:
        # Older tool-state JSON encoded the default NaN fill value as null.
        return np.nan if value is None else value

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "along": self.along,
            "shift_coords": self.shift_coords,
            "keep_dim_order": self.keep_dim_order,
            "assume_sorted": self.assume_sorted,
            "order": self.order,
            "mode": self.mode,
            "cval": self.cval,
            "prefilter": self.prefilter,
        }

    @property
    def code_kwargs(self) -> dict[str, typing.Any]:
        kwargs: dict[str, typing.Any] = {"along": self.along}
        defaults = {
            "shift_coords": False,
            "keep_dim_order": True,
            "assume_sorted": False,
            "order": 1,
            "mode": "constant",
            "prefilter": False,
        }
        for name, default in defaults.items():
            value = getattr(self, name)
            if value != default:
                kwargs[name] = value
        if self.mode == "constant":
            if not np.isnan(self.cval):
                kwargs["cval"] = f"|{_provenance_value_code(self.cval)}|"
        elif self.mode == "grid-constant" and (np.isnan(self.cval) or self.cval != 0.0):
            kwargs["cval"] = f"|{_provenance_value_code(self.cval)}|"
        return kwargs


class ShiftOperation(_BaseShiftOperation):
    """Shift data by one recorded scalar value along one dimension."""

    op: typing.Literal["shift"] = "shift"
    shift: float

    @property
    def effective_shift(self) -> float:
        return -self.shift if self.negate else self.shift

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.shift(
            data,
            self.effective_shift,
            **self.kwargs,
        )

    def derivation_label(self) -> str:
        label_kwargs = {
            "shift": self.shift,
            "negate": self.negate,
            **self.kwargs,
        }
        return f"Shift({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.shift,
            [f"|{input_name}|", self.effective_shift],
            self.code_kwargs,
            module="era.transform",
        )


class ShiftFromInputOperation(_BaseShiftOperation):
    """Shift data by a separately recorded provenance input."""

    op: typing.Literal["shift_from_input"] = "shift_from_input"
    live_applicable: typing.ClassVar[bool] = False

    def input_slots(self) -> tuple[str, ...]:
        return ("shift",)

    def apply_with_inputs(
        self,
        data: xr.DataArray,
        inputs: Mapping[str, xr.DataArray],
        *,
        authorization: object | None = None,
    ) -> xr.DataArray:
        del authorization
        shift = inputs["shift"]
        return erlab.analysis.transform.shift(
            data,
            -shift if self.negate else shift,
            **self.kwargs,
        )

    def derivation_label(self) -> str:
        label_kwargs = {"shift": "input", "negate": self.negate, **self.kwargs}
        return f"Shift({_format_derivation_value(label_kwargs)})"

    def expression_code_with_inputs(
        self,
        input_name: str,
        inputs: Mapping[str, str],
        *,
        source_name: str | None = None,
    ) -> str:
        del source_name
        shift_name = inputs["shift"]
        shift_expression = f"-{shift_name}" if self.negate else shift_name
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.shift,
            [f"|{input_name}|", f"|{shift_expression}|"],
            self.code_kwargs,
            module="era.transform",
        )


def _format_qsel_dims_arg(dims: ProvenanceHashableTuple) -> str:
    return (
        erlab.interactive.utils._parse_single_arg(dims[0])
        if len(dims) == 1 and isinstance(dims[0], str)
        else erlab.interactive.utils._parse_single_arg(dims)
    )


class AverageOperation(ToolProvenanceOperation):
    op: typing.Literal["average"] = "average"
    dims: ProvenanceHashableTuple

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method is not None
            or call.accessor_path != ("qsel", "average")
            or len(call.args) > 1
        ):
            return None
        kwargs = dict(call.kwargs)
        if call.args:
            if "dim" in kwargs:
                return None
            dim = call.args[0]
        else:
            dim = kwargs.pop("dim", None)
        if kwargs or dim is None:
            return None
        dims = (dim,) if isinstance(dim, str) or not isinstance(dim, Sequence) else dim
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(dims=typing.cast("tuple[Hashable, ...]", dims))
        return None

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return data.qsel.mean(self.dims)

    def derivation_label(self) -> str:
        label_kwargs = {"dims": self.dims}
        return f"Average({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        arg = _format_qsel_dims_arg(self.dims)
        return f"{input_name}.qsel.mean({arg})"


class QSelAggregationOperation(ToolProvenanceOperation):
    op: typing.Literal["qsel_aggregate"] = "qsel_aggregate"
    batch_available: typing.ClassVar[bool] = True
    dims: ProvenanceHashableTuple
    func: typing.Literal["mean", "min", "max", "sum"] = "mean"

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        func = call.accessor_path[1] if len(call.accessor_path) == 2 else None
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method is not None
            or len(call.accessor_path) != 2
            or call.accessor_path[0] != "qsel"
            or func not in {"mean", "min", "max", "sum"}
            or len(call.args) > 1
        ):
            return None
        kwargs = dict(call.kwargs)
        if call.args:
            if "dim" in kwargs:
                return None
            dim = call.args[0]
        else:
            dim = kwargs.pop("dim", None)
        if kwargs or dim is None:
            return None
        dims = (dim,) if isinstance(dim, str) or not isinstance(dim, Sequence) else dim
        func = typing.cast("typing.Literal['mean', 'min', 'max', 'sum']", func)
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(
                dims=typing.cast("tuple[Hashable, ...]", dims),
                func=func,
            )
        return None

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return typing.cast("xr.DataArray", getattr(data.qsel, self.func)(self.dims))

    def derivation_label(self) -> str:
        label_kwargs = {"dims": self.dims, "func": self.func}
        return f"Aggregate({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        arg = _format_qsel_dims_arg(self.dims)
        return f"{input_name}.qsel.{self.func}({arg})"


class InterpolationOperation(ToolProvenanceOperation):
    op: typing.Literal["interpolate"] = "interpolate"
    batch_available: typing.ClassVar[bool] = True
    dim: ProvenanceHashable
    values: typing.Any
    method: typing.Literal["linear", "nearest"] = "linear"

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "interp"
            or call.accessor_path
        ):
            return None
        kwargs = dict(call.kwargs)
        method = kwargs.pop("method", "linear")
        if method not in {"linear", "nearest"}:
            return None
        for key, default in {"assume_sorted": False, "kwargs": None}.items():
            if key not in kwargs:
                continue
            if not _console_values_equal(kwargs[key], default):
                return None
            kwargs.pop(key)
        coords = _console_mapping_values(call.args, kwargs, mapping_kwargs=("coords",))
        if coords is None:
            return None
        if len(coords) != 1:
            return None
        ((dim, values),) = coords.items()
        if not isinstance(dim, Hashable):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(dim=dim, values=values, method=method)
        return None

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: typing.Any) -> typing.Any:
        encoded = cls._validate_encoded_field(
            value,
            error="interpolation values must be array-like",
            predicate=lambda encoded_value: (
                isinstance(encoded_value, list)
                or (
                    isinstance(encoded_value, Mapping)
                    and _TUPLE_MARKER in encoded_value
                )
            ),
        )
        values = np.asarray(cls._decode_stored_field(encoded))
        if values.ndim != 1:
            raise ValueError("interpolation values must be one-dimensional.")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise TypeError("interpolation values must be real numeric values.")
        if not np.all(np.isfinite(values.astype(np.float64, copy=False))):
            raise ValueError("interpolation values must be finite.")
        return encoded

    @property
    def decoded_values(self) -> np.ndarray:
        return np.asarray(self._decode_stored_field(self.values))

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return data.interp({self.dim: self.decoded_values}, method=self.method)

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        dim_code = erlab.interactive.utils._parse_single_arg(self.dim)
        values_code = erlab.interactive.utils.format_1d_numeric_array_code(
            self.decoded_values
        )
        method_code = erlab.interactive.utils._parse_single_arg(self.method)
        return (
            f"{input_name}.interp({{{dim_code}: {values_code}}}, method={method_code})"
        )

    def derivation_label(self) -> str:
        label_kwargs = {
            "dim": self.dim,
            "values": self.values,
            "method": self.method,
        }
        return f"Interpolate({_format_derivation_value(label_kwargs)})"


class UniformInterpolationOperation(ToolProvenanceOperation):
    """Interpolate dimensions over their current coordinate bounds."""

    op: typing.Literal["uniform_interpolation"] = "uniform_interpolation"
    batch_available: typing.ClassVar[bool] = True
    sizes: ProvenanceIntMapping = pydantic.Field(default_factory=dict)
    method: typing.Literal["linear", "nearest"] = "linear"

    @pydantic.field_validator("sizes", mode="before")
    @classmethod
    def _validate_integer_sizes(cls, value: typing.Any) -> typing.Any:
        return _require_integer_mapping(value, field_name="interpolation sizes")

    @pydantic.model_validator(mode="after")
    def _validate_sizes(self) -> typing.Self:
        if not self.sizes:
            raise ValueError("uniform interpolation requires at least one dimension")
        if any(size < 1 for size in self.sizes.values()):
            raise ValueError("uniform interpolation sizes must be positive integers")
        return self

    @staticmethod
    def _target_values(data: xr.DataArray, dim: Hashable, size: int) -> np.ndarray:
        coord = np.asarray(data[dim].values)
        if coord.ndim != 1 or coord.size == 0:
            raise ValueError(
                "uniform interpolation requires a nonempty 1D dimension coordinate"
            )
        if not np.issubdtype(coord.dtype, np.number) or np.issubdtype(
            coord.dtype, np.complexfloating
        ):
            raise ValueError(
                "uniform interpolation requires a real numeric dimension coordinate"
            )
        endpoints = coord.astype(np.float64, copy=False)[[0, -1]]
        if not np.all(np.isfinite(endpoints)):
            raise ValueError("uniform interpolation bounds must be finite")
        return np.linspace(float(endpoints[0]), float(endpoints[1]), size)

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return data.interp(
            {
                dim: self._target_values(data, dim, size)
                for dim, size in self.sizes.items()
            },
            method=self.method,
        )

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        target_lines = [
            (
                f"        {erlab.interactive.utils._parse_single_arg(dim)}: "
                f"np.linspace(*{input_name}["
                f"{erlab.interactive.utils._parse_single_arg(dim)}"
                f"].values[[0, -1]], {size}),"
            )
            for dim, size in self.sizes.items()
        ]
        method_code = erlab.interactive.utils._parse_single_arg(self.method)
        return "\n".join(
            (
                f"{input_name}.interp(",
                "    {",
                *target_lines,
                "    },",
                f"    method={method_code},",
                ")",
            )
        )

    def derivation_label(self) -> str:
        label_kwargs = {
            "sizes": self.sizes,
            "method": self.method,
        }
        return f"Interpolate Uniform Grid({_format_derivation_value(label_kwargs)})"

    def preferred_replay_output_name(self) -> str:
        return "processed_data"


class LeadingEdgeOperation(ToolProvenanceOperation):
    op: typing.Literal["leading_edge"] = "leading_edge"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.interpolate.leading_edge",
            fields=("fraction", "dim", "direction"),
            defaults={
                "fraction": 0.5,
                "dim": "eV",
                "direction": "positive",
                "on_failure": "nan",
            },
        ),
    )
    fraction: float = 0.5
    dim: ProvenanceHashable = "eV"
    direction: typing.Literal["positive", "negative"] = "positive"
    on_failure: typing.Literal["nan", "raise"] = "nan"

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "fraction": self.fraction,
            "dim": self.dim,
            "direction": self.direction,
            **({"on_failure": self.on_failure} if self.on_failure != "nan" else {}),
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.interpolate.leading_edge(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Leading Edge({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        kwargs = self.kwargs
        if not np.isfinite(self.fraction):
            kwargs["fraction"] = f"|{_provenance_value_code(self.fraction)}|"
        return erlab.interactive.utils.generate_code(
            erlab.analysis.interpolate.leading_edge,
            [f"|{input_name}|"],
            kwargs,
            module="era.interpolate",
        )


class DivideByCoordOperation(ToolProvenanceOperation):
    op: typing.Literal["divide_by_coord"] = "divide_by_coord"
    batch_available: typing.ClassVar[bool] = True
    coord_name: ProvenanceHashable

    @staticmethod
    def _raise_if_zero(coord: xr.DataArray) -> None:
        if np.any(np.asarray(coord.values) == 0):
            raise ValueError("Coordinate contains zero values and cannot be a divisor.")

    def divisor_code(self, data_name: str) -> str:
        if (
            isinstance(self.coord_name, str)
            and self.coord_name.isidentifier()
            and not keyword.iskeyword(self.coord_name)
            and not self.coord_name.startswith("_")
            and not hasattr(xr.DataArray, self.coord_name)
        ):
            return f"{data_name}.{self.coord_name}"
        coord_name = erlab.interactive.utils._parse_single_arg(self.coord_name)
        return f"{data_name}.coords[{coord_name}]"

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        coord = data.coords[self.coord_name]
        self._raise_if_zero(coord)
        return (data / coord).rename(data.name)

    def derivation_label(self) -> str:
        label_kwargs = {"coord_name": self.coord_name}
        return f"Divide by Coordinate({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        quotient = f"{input_name} / {self.divisor_code(input_name)}"
        return f"({quotient}).rename({input_name}.name)"


_AxisArgT = typing.TypeVar("_AxisArgT")
_FilterArg = typing.Annotated[
    _AxisArgT | tuple[_AxisArgT, ...] | dict[Hashable, _AxisArgT],
    pydantic.BeforeValidator(decode_provenance_value),
    pydantic.PlainSerializer(encode_provenance_value, when_used="json"),
]


class GaussianFilterOperation(ToolProvenanceOperation):
    xarray_option_names: typing.ClassVar[tuple[str, ...]] = ()
    op: typing.Literal["gaussian_filter"] = "gaussian_filter"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.image.gaussian_filter",
            fields=("sigma", "order", "mode", "cval", "truncate"),
            defaults={
                "order": 0,
                "mode": "nearest",
                "cval": 0.0,
                "truncate": 4.0,
                "radius": None,
            },
        ),
    )
    sigma: _FilterArg[float] = pydantic.Field(default_factory=dict)
    order: _FilterArg[pydantic.StrictInt] = 0
    mode: _FilterArg[str] = "nearest"
    cval: float = 0.0
    truncate: float = 4.0
    radius: _FilterArg[float] | None = None

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        kwargs: dict[str, typing.Any] = {"sigma": self.sigma}
        if self.order != 0:
            kwargs["order"] = self.order
        if self.mode != "nearest":
            kwargs["mode"] = self.mode
        if self.cval != 0.0:
            kwargs["cval"] = self.cval
        if self.truncate != 4.0:
            kwargs["truncate"] = self.truncate
        if self.radius is not None:
            kwargs["radius"] = self.radius
        return kwargs

    def sigma_for_dims(self, dims: tuple[Hashable, ...]) -> dict[Hashable, float]:
        if isinstance(self.sigma, dict):
            return dict(self.sigma)
        return erlab.analysis.image._parse_dict_arg(
            dims, self.sigma, "sigma", "DataArray", allow_subset=True
        )

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.image.gaussian_filter(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Gaussian Filter({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        kwargs = self.kwargs
        if "cval" in kwargs:
            kwargs["cval"] = f"|{_provenance_value_code(self.cval)}|"
        return erlab.interactive.utils.generate_code(
            erlab.analysis.image.gaussian_filter,
            [f"|{input_name}|"],
            kwargs,
            module="era.image",
        )


class BoxcarFilterOperation(ToolProvenanceOperation):
    """Apply a coordinate-preserving boxcar filter along selected dimensions."""

    xarray_option_names: typing.ClassVar[tuple[str, ...]] = ()
    op: typing.Literal["boxcar_filter"] = "boxcar_filter"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.image.boxcar_filter",
            fields=("size", "mode", "cval"),
            defaults={"mode": "nearest", "cval": 0.0},
        ),
    )
    size: _FilterArg[pydantic.StrictInt] = pydantic.Field(default_factory=dict)
    mode: ShiftMode = "nearest"
    cval: float = 0.0

    @pydantic.field_validator("size", mode="before")
    @classmethod
    def _validate_integer_sizes(cls, value: typing.Any) -> typing.Any:
        return _require_integer_mapping(value, field_name="boxcar filter sizes")

    @pydantic.model_validator(mode="after")
    def _validate_boxcar_filter(self) -> typing.Self:
        sizes = (
            self.size.values()
            if isinstance(self.size, dict)
            else ((self.size,) if isinstance(self.size, int) else self.size)
        )
        if not sizes:
            raise ValueError("boxcar filter requires at least one dimension")
        if any(value <= 0 for value in sizes):
            raise ValueError("boxcar filter sizes must be positive integers")
        return self

    def size_for_dims(self, dims: tuple[Hashable, ...]) -> dict[Hashable, int]:
        if isinstance(self.size, dict):
            return dict(self.size)
        return {
            dim: int(value)
            for dim, value in erlab.analysis.image._parse_dict_arg(
                dims, self.size, "size", "DataArray", allow_subset=True
            ).items()
        }

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {"size": self.size, "mode": self.mode, "cval": self.cval}

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.image.boxcar_filter(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Boxcar Filter({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        kwargs = self.kwargs
        if not np.isfinite(self.cval):
            kwargs["cval"] = f"|{_provenance_value_code(self.cval)}|"
        return erlab.interactive.utils.generate_code(
            erlab.analysis.image.boxcar_filter,
            [f"|{input_name}|"],
            kwargs,
            module="era.image",
        )


class FillNaOperation(ToolProvenanceOperation):
    """Replace missing values with a scalar through the public xarray API."""

    op: typing.Literal["fillna"] = "fillna"
    value: float = 0.0

    @pydantic.field_validator("value")
    @classmethod
    def _validate_fill_value(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("fill value must be finite")
        return value

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return data.fillna(self.value)

    def derivation_label(self) -> str:
        return f"Fill missing values with {_format_derivation_value(self.value)}"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return f"{input_name}.fillna({_provenance_value_code(self.value)})"


_IMAGE_DERIVATIVE_KWARGS: dict[str, frozenset[str]] = {
    "diffn": frozenset(("coord", "order")),
    "scaled_laplace": frozenset(("factor",)),
    "curvature1d": frozenset(("along", "a0")),
    "curvature": frozenset(("a0", "factor")),
    "minimum_gradient": frozenset(),
}


class ImageDerivativeOperation(ToolProvenanceOperation):
    """Apply one of the public image-derivative functions."""

    op: typing.Literal["image_derivative"] = "image_derivative"
    method: typing.Literal[
        "diffn",
        "scaled_laplace",
        "curvature1d",
        "curvature",
        "minimum_gradient",
    ]
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        func = call.func
        name = getattr(func, "__name__", "")
        if (
            func is None
            or name not in _IMAGE_DERIVATIVE_KWARGS
            or func is not getattr(erlab.analysis.image, name)
            or call.has_extra_tracked_inputs
        ):
            return None
        signature = inspect.signature(func)
        with contextlib.suppress(TypeError, ValueError):
            arguments = signature.bind(None, *call.args, **call.kwargs)
            arguments.apply_defaults()
            settings = dict(arguments.arguments)
            del settings[next(iter(signature.parameters))]
            settings.update(settings.pop("kwargs", {}))
            return cls.model_validate({"method": name, "kwargs": settings})
        return None

    @pydantic.model_validator(mode="after")
    def _validate_derivative(self) -> typing.Self:
        if any(not isinstance(key, str) for key in self.kwargs):
            raise ValueError("image derivative keyword names must be strings")
        expected = _IMAGE_DERIVATIVE_KWARGS[self.method]
        actual = frozenset(typing.cast("dict[str, typing.Any]", self.kwargs))
        if not expected.issubset(actual) or (
            self.method == "minimum_gradient" and actual - {"mode", "cval"}
        ):
            raise ValueError(
                f"{self.method} requires keyword arguments {sorted(expected)!r}"
            )

        kwargs = typing.cast("dict[str, typing.Any]", dict(self.kwargs))
        dimension_key = "coord" if self.method == "diffn" else "along"
        if dimension_key in kwargs and not isinstance(kwargs[dimension_key], Hashable):
            raise ValueError(f"{dimension_key} must be hashable")
        if self.method == "diffn":
            order = kwargs["order"]
            if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
                raise ValueError("diffn order must be a positive integer")
        for name in expected & {"a0", "factor"}:
            value = kwargs[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if "a0" in kwargs and kwargs["a0"] <= 0:
            raise ValueError("a0 must be positive")
        return self

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        function = getattr(erlab.analysis.image, self.method)
        kwargs = typing.cast("dict[str, typing.Any]", dict(self.kwargs))
        result = function(data, **kwargs)
        if not isinstance(result, xr.DataArray):  # pragma: no cover - validation guard.
            raise TypeError(f"{self.method} did not return a DataArray")
        return result

    def derivation_label(self) -> str:
        return (
            f"Image Derivative({self.method}, {_format_derivation_value(self.kwargs)})"
        )

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        kwargs = typing.cast("dict[str, typing.Any]", dict(self.kwargs))
        if "cval" in kwargs:
            kwargs["cval"] = f"|{_provenance_value_code(kwargs['cval'])}|"
        return erlab.interactive.utils.generate_code(
            getattr(erlab.analysis.image, self.method),
            [f"|{input_name}|"],
            kwargs,
            module="era.image",
            remove_defaults=False,
        )

    def preferred_replay_output_name(self) -> str:
        return "result"

    def preferred_replay_input_name(self) -> str:
        return "processed_data"


class RemoveMeshOperation(ToolProvenanceOperation):
    """Remove a mesh pattern and select one of the two DataArray outputs."""

    op: typing.Literal["remove_mesh"] = "remove_mesh"
    first_order_peaks: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    order: int = pydantic.Field(default=3, ge=0)
    n_pad: int = pydantic.Field(default=90, ge=0)
    roi_hw: int = pydantic.Field(default=25, ge=0)
    k: float = 0.5
    feather: float = pydantic.Field(default=3.0, ge=0.0)
    undo_edge_correction: bool = False
    method: typing.Literal["constant", "gaussian", "circular"] = "constant"
    output: typing.Literal["corrected", "mesh"] = "corrected"

    @pydantic.field_validator("order", "n_pad", "roi_hw", mode="before")
    @classmethod
    def _validate_integer_parameter(cls, value: typing.Any) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("mesh integer parameters must be integers")
        return int(value)

    @pydantic.field_validator("first_order_peaks", mode="before")
    @classmethod
    def _validate_peak_shape(
        cls, value: typing.Any
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        if isinstance(value, str | bytes):
            raise TypeError("first_order_peaks must contain three coordinate pairs")
        try:
            points = tuple(value)
        except TypeError as exc:
            raise TypeError(
                "first_order_peaks must contain three coordinate pairs"
            ) from exc
        if len(points) != 3:
            raise ValueError("first_order_peaks must contain exactly three points")
        normalized: list[tuple[int, int]] = []
        for point in points:
            if isinstance(point, str | bytes):
                raise TypeError("each first-order peak must be a coordinate pair")
            try:
                coordinates = tuple(point)
            except TypeError as exc:
                raise TypeError(
                    "each first-order peak must be a coordinate pair"
                ) from exc
            if len(coordinates) != 2:
                raise ValueError("each first-order peak must contain two coordinates")
            if any(
                isinstance(coordinate, bool) or not isinstance(coordinate, Integral)
                for coordinate in coordinates
            ):
                raise TypeError("first-order peak coordinates must be integers")
            normalized.append(
                typing.cast("tuple[int, int]", tuple(map(int, coordinates)))
            )
        return typing.cast(
            "tuple[tuple[int, int], tuple[int, int], tuple[int, int]]",
            tuple(normalized),
        )

    @pydantic.model_validator(mode="after")
    def _validate_remove_mesh(self) -> typing.Self:
        if any(
            coordinate < 0 for point in self.first_order_peaks for coordinate in point
        ):
            raise ValueError("first-order peak coordinates must be nonnegative")
        if not np.isfinite(self.k):
            raise ValueError("mesh threshold k must be finite")
        if not np.isfinite(self.feather):
            raise ValueError("mesh feather must be finite")
        return self

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "first_order_peaks": self.first_order_peaks,
            "order": self.order,
            "n_pad": self.n_pad,
            "roi_hw": self.roi_hw,
            "k": self.k,
            "feather": self.feather,
            "undo_edge_correction": self.undo_edge_correction,
            "method": self.method,
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        corrected, mesh = typing.cast(
            "tuple[xr.DataArray, xr.DataArray]",
            erlab.analysis.mesh.remove_mesh(data, **self.kwargs),
        )
        return corrected if self.output == "corrected" else mesh

    def derivation_label(self) -> str:
        action = "Remove Mesh" if self.output == "corrected" else "Extract Mesh"
        return f"{action}({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        raise NotImplementedError

    def statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
    ) -> str:
        return self._remove_mesh_statement_code(
            input_name,
            output_name=output_name,
            reserved_names=(),
        )

    def _statement_replay_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
        reserved_names: Collection[str] = (),
    ) -> str:
        return self._remove_mesh_statement_code(
            input_name,
            output_name=output_name,
            reserved_names=reserved_names,
        )

    def _remove_mesh_statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        reserved_names: Collection[str],
    ) -> str:
        other_base = "mesh" if self.output == "corrected" else "corrected"
        unavailable = {input_name, output_name, *reserved_names}
        other_name = other_base
        suffix = 2
        while other_name in unavailable:
            other_name = f"{other_base}_{suffix}"
            suffix += 1
        corrected_name, mesh_name = (
            (output_name, other_name)
            if self.output == "corrected"
            else (other_name, output_name)
        )
        return erlab.interactive.utils.generate_code(
            erlab.analysis.mesh.remove_mesh,
            [f"|{input_name}|"],
            self.kwargs,
            module="era.mesh",
            assign=(corrected_name, mesh_name),
            remove_defaults=False,
            copy=False,
        )

    def preferred_replay_output_name(self) -> str:
        return self.output

    def preferred_replay_input_name(self) -> str:
        return "mesh_data"


class NormalizeOperation(ToolProvenanceOperation):
    op: typing.Literal["normalize"] = "normalize"
    batch_available: typing.ClassVar[bool] = True
    dims: ProvenanceHashableTuple = ()
    mode: typing.Literal["area", "minmax", "min", "min_area"] = "area"
    denominator_rtol: float = 1e-12
    keep_attrs: bool | typing.Literal["default"] = "default"

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "mode": self.mode,
            "denominator_rtol": self.denominator_rtol,
            "keep_attrs": self.keep_attrs,
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.image.normalize(data, self.dims, **self.kwargs)

    def derivation_label(self) -> str:
        label_kwargs = {
            "dims": self.dims,
            "mode": self.mode,
            "denominator_rtol": self.denominator_rtol,
        }
        return f"Normalize({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.image.normalize,
            [f"|{input_name}|", self.dims],
            self.kwargs,
            module="era.image",
            name="normalize",
        )


class CoarsenOperation(ToolProvenanceOperation):
    op: typing.Literal["coarsen"] = "coarsen"
    batch_available: typing.ClassVar[bool] = True
    dim: ProvenanceIntMapping = pydantic.Field(default_factory=dict)
    boundary: str
    side: str
    coord_func: str
    reducer: str
    keep_attrs: bool | None = None

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "coarsen"
            or call.accessor_path
            or len(call.args) > 1
        ):
            return None
        kwargs = dict(call.kwargs)
        reducer = kwargs.pop("_reducer", None)
        if not isinstance(reducer, str):
            return None
        dim: dict[typing.Any, typing.Any] = {}
        if call.args:
            if not isinstance(call.args[0], Mapping):
                return None
            dim.update(call.args[0])
        dim.update(kwargs.pop("dim", {}) or {})
        dim.update(
            {
                key: kwargs.pop(key)
                for key in list(kwargs)
                if key not in {"boundary", "side", "coord_func"}
            }
        )
        values = {
            "dim": dim,
            "boundary": kwargs.pop("boundary", "exact"),
            "side": kwargs.pop("side", "left"),
            "coord_func": kwargs.pop("coord_func", "mean"),
            "reducer": reducer,
        }
        if kwargs:
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(**values)
        return None

    @property
    def coarsen_kwargs(self) -> dict[str, typing.Any]:
        return {
            "dim": self.dim,
            "boundary": self.boundary,
            "side": self.side,
            "coord_func": self.coord_func,
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        coarsened = data.coarsen(**self.coarsen_kwargs)
        return typing.cast(
            "xr.DataArray", getattr(coarsened, self.reducer)(keep_attrs=self.keep_attrs)
        )

    def _code_coarsen_kwargs(self) -> Mapping[typing.Any, typing.Any]:
        coarsen_kwargs: Mapping[typing.Any, typing.Any]
        if _is_identifier_string_mapping(self.dim):
            coarsen_kwargs = {
                **self.dim,
                "boundary": self.boundary,
                "side": self.side,
                "coord_func": self.coord_func,
            }
        else:
            coarsen_kwargs = self.coarsen_kwargs
        coarsen_kwargs = erlab.interactive.utils._remove_default_kwargs(
            xr.DataArray.coarsen, dict(coarsen_kwargs)
        )
        return coarsen_kwargs

    def derivation_label(self) -> str:
        label_kwargs = {"coarsen_kwargs": self.coarsen_kwargs, "reducer": self.reducer}
        return f"Coarsen({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        formatted_kwargs = erlab.interactive.utils.format_kwargs(
            self._code_coarsen_kwargs()
        )
        reducer_args = (
            "" if self.keep_attrs is None else f"keep_attrs={self.keep_attrs}"
        )
        return (
            f"{input_name}.coarsen({formatted_kwargs}).{self.reducer}({reducer_args})"
        )


class ThinOperation(ToolProvenanceOperation):
    op: typing.Literal["thin"] = "thin"
    batch_available: typing.ClassVar[bool] = True
    mode: typing.Literal["global", "per_dim"]
    factor: int | None = None
    factors: ProvenanceIntMapping = pydantic.Field(default_factory=dict)

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "thin"
            or call.accessor_path
        ):
            return None
        if (
            len(call.args) == 1
            and not call.kwargs
            and not isinstance(call.args[0], Mapping)
        ):
            with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
                return cls(mode="global", factor=call.args[0])
            return None
        if len(call.args) > 1:
            return None
        factors: typing.Any = dict(call.kwargs)
        if call.args:
            if call.args[0] is None:
                pass
            elif isinstance(call.args[0], Mapping):
                factors = {**call.args[0], **factors}
            else:
                return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(mode="per_dim", factors=factors)
        return None

    @pydantic.model_validator(mode="after")
    def _check_mode(self) -> typing.Self:
        if self.mode == "global" and self.factor is None:
            raise ValueError("thin global mode requires factor")
        if self.mode == "per_dim" and not self.factors:
            raise ValueError("thin per_dim mode requires factors")
        return self

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        if self.mode == "global":
            return {"mode": self.mode, "factor": int(typing.cast("int", self.factor))}
        return {"mode": self.mode, "factors": self.factors}

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        if self.mode == "global":
            return data.thin(int(typing.cast("int", self.factor)))
        return data.thin(self.factors)

    def derivation_label(self) -> str:
        return f"Thin({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        if self.mode == "global":
            return f"{input_name}.thin({int(typing.cast('int', self.factor))})"
        return (
            f"{input_name}.thin("
            f"{erlab.interactive.utils.format_call_kwargs(self.factors)})"
        )


class SymmetrizeOperation(ToolProvenanceOperation):
    op: typing.Literal["symmetrize"] = "symmetrize"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.transform.symmetrize",
            fields=("dim",),
            defaults={
                "center": 0.0,
                "subtract": False,
                "average": False,
                "mode": "full",
                "part": "both",
                "interp_kw": None,
            },
        ),
    )
    dim: ProvenanceHashable
    center: float
    subtract: bool = False
    average: bool = False
    mode: typing.Literal["full", "valid"] = "full"
    part: typing.Literal["both", "below", "above"] = "both"
    interp_kw: ProvenanceKeywords | None = None

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "dim": self.dim,
            "center": self.center,
            "subtract": self.subtract,
            "average": self.average,
            "mode": self.mode,
            "part": self.part,
            **(
                {"interp_kw": dict(self.interp_kw)}
                if self.interp_kw is not None
                else {}
            ),
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Symmetrize({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        kwargs = self.kwargs
        if self.interp_kw is not None:
            kwargs["interp_kw"] = f"|{_provenance_value_code(self.interp_kw)}|"
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.symmetrize,
            [f"|{input_name}|"],
            kwargs,
            module="era.transform",
        )


class SymmetrizeNfoldOperation(_RotationPlaneOperation):
    op: typing.Literal["symmetrize_nfold"] = "symmetrize_nfold"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.transform.symmetrize_nfold",
            fields=("fold", "axes", "center"),
            defaults=_ROTATION_DEFAULTS,
        ),
    )
    fold: int

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {"fold": self.fold, **self._plane_kwargs}

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize_nfold(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Rotational Symmetrize({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return self._rotation_code(
            erlab.analysis.transform.symmetrize_nfold, input_name, self.kwargs
        )


class CorrectWithEdgeOperation(ToolProvenanceOperation):
    op: typing.Literal["correct_with_edge"] = "correct_with_edge"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.gold.correct_with_edge",
            fields=("edge_fit",),
            field_aliases={"modelresult": "edge_fit"},
            defaults={"along": "alpha", "shift_coords": True},
            ignored_defaults={"plot": False, "plot_kw": None},
            extra_kwargs_field="shift_kwargs",
        ),
    )
    edge_fit: typing.Any
    along: str = "alpha"
    shift_kwargs: ProvenanceKeywords = pydantic.Field(default_factory=dict)
    shift_coords: bool = True

    @pydantic.field_validator("edge_fit", mode="before")
    @classmethod
    def _validate_edge_fit(cls, value: typing.Any) -> typing.Any:
        if isinstance(value, Mapping) and (
            _DATASET_MARKER in value or _FIT_DATASET_MARKER in value
        ):
            return value
        if isinstance(value, np.ndarray | ProvenanceNumericVector) or (
            isinstance(value, Mapping) and "dtype" in value and "values" in value
        ):
            return ProvenanceNumericVector.model_validate(value)
        if isinstance(value, tuple) or (
            isinstance(value, Mapping) and _TUPLE_MARKER in value
        ):
            return encode_provenance_value(
                tuple(float(x) for x in decode_provenance_value(value))
            )
        return cls._validate_encoded_field(
            value,
            error="correct_with_edge edge_fit must be an xarray.Dataset",
            predicate=lambda encoded: (
                isinstance(encoded, Mapping)
                and (_DATASET_MARKER in encoded or _FIT_DATASET_MARKER in encoded)
            ),
        )

    @property
    def decoded_edge_fit(self) -> xr.Dataset | np.ndarray | tuple[float, ...]:
        return self._decode_edge_fit()

    def _decode_edge_fit(
        self,
        authorization: object | None = None,
    ) -> xr.Dataset | np.ndarray | tuple[float, ...]:
        from erlab.interactive._code_trust import execution_capability_allows
        from erlab.interactive.imagetool._provenance._trust import (
            provenance_operation_code_trust_entries,
        )

        if isinstance(self.edge_fit, ProvenanceNumericVector):
            return self.edge_fit.array()
        if _TUPLE_MARKER in self.edge_fit:
            return typing.cast(
                "tuple[float, ...]", decode_provenance_value(self.edge_fit)
            )
        entries = provenance_operation_code_trust_entries(
            self,
            location_prefix="operation",
        )
        if not execution_capability_allows(authorization, entries):
            raise PermissionError("The serialized lmfit result is not authorized")
        return typing.cast(
            "xr.Dataset",
            decode_provenance_value(self.edge_fit, allow_code_payload=True),
        )

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "along": self.along,
            "shift_coords": self.shift_coords,
            **self.shift_kwargs,
        }

    def apply(
        self,
        data: xr.DataArray,
        *,
        authorization: object | None = None,
    ) -> xr.DataArray:
        return erlab.analysis.gold.correct_with_edge(
            data,
            self._decode_edge_fit(authorization),
            **self.kwargs,
        )

    def derivation_label(self) -> str:
        label_kwargs = {
            "edge_fit": self.edge_fit,
            **self.kwargs,
        }
        return f"Edge Correction({_format_derivation_value(label_kwargs)})"

    def derivation_entry(self) -> DerivationEntry:
        if isinstance(self.edge_fit, Mapping) and _FIT_DATASET_MARKER in self.edge_fit:
            return DerivationEntry(self.derivation_label(), None, False)
        return super().derivation_entry()

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        if isinstance(self.edge_fit, Mapping) and _FIT_DATASET_MARKER in self.edge_fit:
            raise NotImplementedError(
                "Copied code cannot represent a serialized lmfit result cleanly"
            )
        edge_fit = self.decoded_edge_fit
        if isinstance(edge_fit, xr.Dataset):
            payload = _provenance_value_code(edge_fit.to_dict(data="list"))
            edge_fit_expr = f"xr.Dataset.from_dict({payload})"
        elif isinstance(self.edge_fit, ProvenanceNumericVector):
            edge_fit_expr = self.edge_fit.expression_code()
        else:
            edge_fit_expr = _provenance_value_code(edge_fit)
        kwargs = self.kwargs
        kwargs.update(
            {
                str(key): f"|{_provenance_value_code(value)}|"
                for key, value in self.shift_kwargs.items()
            }
        )
        return erlab.interactive.utils.generate_code(
            erlab.analysis.gold.correct_with_edge,
            [f"|{input_name}|", f"|{edge_fit_expr}|"],
            kwargs,
            module="era.gold",
            name="correct_with_edge",
        )
