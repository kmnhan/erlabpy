"""Filtering, reduction, and image-analysis provenance operations."""

from __future__ import annotations

import contextlib
import keyword
import typing
from collections.abc import Collection, Hashable, Mapping, Sequence
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
    ProvenanceFloatMapping,
    ProvenanceHashable,
    ProvenanceHashablePair,
    ProvenanceHashableTuple,
    ProvenanceIntMapping,
    ProvenanceMapping,
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


class RotateOperation(ToolProvenanceOperation):
    op: typing.Literal["rotate"] = "rotate"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.transform.rotate",
            fields=("angle", "axes", "center"),
            defaults={
                "axes": (0, 1),
                "center": (0.0, 0.0),
                "reshape": True,
                "order": 1,
            },
            ignored_defaults={"mode": "constant", "cval": np.nan, "prefilter": True},
        ),
    )
    angle: float
    axes: ProvenanceHashablePair
    center: tuple[float, float]
    reshape: bool = True
    order: int = 1

    @pydantic.field_validator("center", mode="before")
    @classmethod
    def _validate_center(cls, value: typing.Any) -> tuple[float, float]:
        return typing.cast(
            "tuple[float, float]",
            _ensure_float_tuple(
                typing.cast("Sequence[typing.Any]", value), expected_len=2
            ),
        )

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "angle": self.angle,
            "axes": self.axes,
            "center": self.center,
            "reshape": self.reshape,
            "order": self.order,
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.rotate(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Rotate({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.rotate,
            [f"|{input_name}|"],
            self.kwargs,
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
            },
        ),
    )
    fraction: float = pydantic.Field(default=0.5, gt=0.0, le=1.0)
    dim: ProvenanceHashable = "eV"
    direction: typing.Literal["positive", "negative"] = "positive"

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "fraction": self.fraction,
            "dim": self.dim,
            "direction": self.direction,
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.interpolate.leading_edge(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Leading Edge({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.interpolate.leading_edge,
            [f"|{input_name}|"],
            self.kwargs,
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
        return f"{input_name} / {self.divisor_code(input_name)}"


class GaussianFilterOperation(ToolProvenanceOperation):
    op: typing.Literal["gaussian_filter"] = "gaussian_filter"
    batch_available: typing.ClassVar[bool] = True
    sigma: ProvenanceFloatMapping = pydantic.Field(default_factory=dict)

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.image.gaussian_filter(data, sigma=self.sigma)

    def derivation_label(self) -> str:
        label_kwargs = {"sigma": self.sigma}
        return f"Gaussian Filter({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.image.gaussian_filter,
            [f"|{input_name}|"],
            {"sigma": self.sigma},
            module="era.image",
        )


class BoxcarFilterOperation(ToolProvenanceOperation):
    """Apply a coordinate-preserving boxcar filter along selected dimensions."""

    op: typing.Literal["boxcar_filter"] = "boxcar_filter"
    batch_available: typing.ClassVar[bool] = True
    size: ProvenanceIntMapping = pydantic.Field(default_factory=dict)
    mode: typing.Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "nearest"
    cval: float = 0.0

    @pydantic.field_validator("size", mode="before")
    @classmethod
    def _validate_integer_sizes(cls, value: typing.Any) -> typing.Any:
        return _require_integer_mapping(value, field_name="boxcar filter sizes")

    @pydantic.model_validator(mode="after")
    def _validate_boxcar_filter(self) -> typing.Self:
        if not self.size:
            raise ValueError("boxcar filter requires at least one dimension")
        if any(value <= 0 for value in self.size.values()):
            raise ValueError("boxcar filter sizes must be positive integers")
        if not np.isfinite(self.cval):
            raise ValueError("boxcar filter cval must be finite")
        return self

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
        return erlab.interactive.utils.generate_code(
            erlab.analysis.image.boxcar_filter,
            [f"|{input_name}|"],
            self.kwargs,
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

    @pydantic.model_validator(mode="after")
    def _validate_derivative(self) -> typing.Self:
        if any(not isinstance(key, str) for key in self.kwargs):
            raise ValueError("image derivative keyword names must be strings")
        expected = _IMAGE_DERIVATIVE_KWARGS[self.method]
        actual = frozenset(typing.cast("dict[str, typing.Any]", self.kwargs))
        if actual != expected:
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
        return erlab.interactive.utils.generate_code(
            getattr(erlab.analysis.image, self.method),
            [f"|{input_name}|"],
            typing.cast("dict[str, typing.Any]", dict(self.kwargs)),
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

    @staticmethod
    def _safe_denominator(
        denominator: xr.DataArray, scale: xr.DataArray, rtol: float
    ) -> xr.DataArray:
        threshold = np.maximum(
            scale * rtol,
            np.finfo(np.float64).tiny,
        )
        return denominator.where(
            np.isfinite(denominator)
            & np.isfinite(scale)
            & (np.abs(denominator) > threshold)
        )

    @staticmethod
    def _dims_code(dims: tuple[Hashable, ...]) -> str:
        return _format_qsel_dims_arg(dims)

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        if not self.dims:
            return data

        if self.mode == "area":
            finite_abs_scale = (
                abs(data).where(np.isfinite(data)).max(self.dims, skipna=True)
            )
            area = self._safe_denominator(
                data.mean(self.dims),
                finite_abs_scale,
                self.denominator_rtol,
            )
            return data / area

        if self.mode == "minmax":
            minimum = data.min(self.dims)
            maximum = data.max(self.dims)
            finite_abs_scale = xr.where(
                abs(minimum) > abs(maximum),
                abs(minimum),
                abs(maximum),
            )
            denominator = self._safe_denominator(
                maximum - minimum,
                finite_abs_scale,
                self.denominator_rtol,
            )
            return (data - minimum) / denominator

        if self.mode == "min":
            minimum = data.min(self.dims)
            return data - minimum

        minimum = data.min(self.dims)
        finite_abs_scale = (
            abs(data).where(np.isfinite(data)).max(self.dims, skipna=True)
        )
        area = self._safe_denominator(
            data.mean(self.dims),
            finite_abs_scale,
            self.denominator_rtol,
        )
        return (data - minimum) / area

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
        dims_code = self._dims_code(self.dims)
        if self.mode == "area":
            return f"{input_name} / {input_name}.mean({dims_code})"
        if self.mode == "minmax":
            minimum: str = f"{input_name}.min({dims_code})"
            maximum: str = f"{input_name}.max({dims_code})"
            return f"({input_name} - {minimum}) / ({maximum} - {minimum})"
        if self.mode == "min":
            return f"{input_name} - {input_name}.min({dims_code})"
        minimum = f"{input_name}.min({dims_code})"
        return f"({input_name} - {minimum}) / {input_name}.mean({dims_code})"


class CoarsenOperation(ToolProvenanceOperation):
    op: typing.Literal["coarsen"] = "coarsen"
    batch_available: typing.ClassVar[bool] = True
    dim: ProvenanceIntMapping = pydantic.Field(default_factory=dict)
    boundary: str
    side: str
    coord_func: str
    reducer: str

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
        return typing.cast("xr.DataArray", getattr(coarsened, self.reducer)())

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
        return f"{input_name}.coarsen({formatted_kwargs}).{self.reducer}()"


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
                "mode": "full",
                "part": "both",
            },
            ignored_defaults={"interp_kw": None},
        ),
    )
    dim: ProvenanceHashable
    center: float
    subtract: bool = False
    mode: typing.Literal["full", "valid"] = "full"
    part: typing.Literal["both", "below", "above"] = "both"

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "dim": self.dim,
            "center": self.center,
            "subtract": self.subtract,
            "mode": self.mode,
            "part": self.part,
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Symmetrize({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.symmetrize,
            [f"|{input_name}|"],
            self.kwargs,
            module="era.transform",
        )


class SymmetrizeNfoldOperation(ToolProvenanceOperation):
    op: typing.Literal["symmetrize_nfold"] = "symmetrize_nfold"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.transform.symmetrize_nfold",
            fields=("fold", "axes", "center"),
            defaults={
                "axes": (0, 1),
                "center": (0.0, 0.0),
                "reshape": True,
                "order": 1,
            },
            ignored_defaults={"mode": "constant", "cval": np.nan, "prefilter": True},
        ),
    )
    fold: int
    axes: ProvenanceHashablePair
    center: typing.Any = (0.0, 0.0)
    reshape: bool = True
    order: int = 1

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
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "fold": self.fold,
            "axes": self.axes,
            "center": self.center,
            "reshape": self.reshape,
            "order": self.order,
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize_nfold(data, **self.kwargs)

    def derivation_label(self) -> str:
        return f"Rotational Symmetrize({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.symmetrize_nfold,
            [f"|{input_name}|"],
            self.kwargs,
            module="era.transform",
        )


class CorrectWithEdgeOperation(ToolProvenanceOperation):
    op: typing.Literal["correct_with_edge"] = "correct_with_edge"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.gold.correct_with_edge",
            fields=("edge_fit",),
            field_aliases={"modelresult": "edge_fit"},
            defaults={"shift_coords": True},
        ),
    )
    edge_fit: typing.Any
    shift_coords: bool = True

    @pydantic.field_validator("edge_fit", mode="before")
    @classmethod
    def _validate_edge_fit(cls, value: typing.Any) -> typing.Any:
        if isinstance(value, Mapping) and (
            _DATASET_MARKER in value or _FIT_DATASET_MARKER in value
        ):
            return value
        return cls._validate_encoded_field(
            value,
            error="correct_with_edge edge_fit must be an xarray.Dataset",
            predicate=lambda encoded: (
                isinstance(encoded, Mapping)
                and (_DATASET_MARKER in encoded or _FIT_DATASET_MARKER in encoded)
            ),
        )

    @property
    def decoded_edge_fit(self) -> xr.Dataset:
        return typing.cast("xr.Dataset", self._decode_stored_field(self.edge_fit))

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.gold.correct_with_edge(
            data,
            self.decoded_edge_fit,
            shift_coords=self.shift_coords,
        )

    def derivation_label(self) -> str:
        label_kwargs = {
            "edge_fit": self.edge_fit,
            "shift_coords": self.shift_coords,
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
        edge_fit_expr = "xr.Dataset.from_dict({})".format(
            _provenance_value_code(self.decoded_edge_fit.to_dict(data="list"))
        )
        return erlab.interactive.utils.generate_code(
            erlab.analysis.gold.correct_with_edge,
            [f"|{input_name}|", f"|{edge_fit_expr}|"],
            {"shift_coords": self.shift_coords},
            module="era.gold",
            name="correct_with_edge",
        )
