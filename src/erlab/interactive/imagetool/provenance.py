"""Helpers for recording and replaying actions in ImageTool windows.

The provenance model stores how a tool's data should be reconstructed from a parent
ImageTool window. A :class:`ToolProvenanceSpec` answers two questions:

1. What should be used as the starting point?
   Use :func:`full_data` when the derived tool should begin from the parent's current
   array exactly as shown to the caller.
   Use :func:`selection` when the derived tool should begin from the public ImageTool
   selection model, including restoration of non-uniform dimensions.
2. Which operations should be replayed on that starting point?
   Each operation is represented by an immutable :class:`_BaseOperation` subclass whose
   serialized fields are safe to persist in JSON.

Adding a new provenance-carrying operation follows the same pattern every time:

1. Define a new `_BaseOperation` subclass with a unique ``op`` discriminator literal.
2. Choose serialized fields that are stable in JSON and validate them with Pydantic.
3. If the operation needs non-JSON values such as slices, NumPy arrays, or xarray
   objects, encode them in field validators with :func:`encode_provenance_value` and
   expose decoded convenience properties for runtime use.
4. Implement :meth:`_BaseOperation.apply` so it transforms a derived array using the
   recorded parameters. ``parent_data`` is provided when the operation needs access to
   parent coordinates or ordering.
5. Implement :meth:`_BaseOperation.derivation_entry` so the manager can display a
   user-facing summary and optional copyable code. Return ``None`` when the operation
   should be omitted from the derivation list.
6. Register the class in :data:`ToolProvenanceOperation` so discriminated-union
   validation can deserialize it from saved JSON.
7. Export a small builder function near the end of this module so callers can create
   the operation without depending on the model class directly.
8. Add tests that cover round-trip validation, :meth:`apply`, and derivation text/code,
   plus any save/load path that persists the new operation.

Parsing of serialized payloads happens only through :func:`parse_tool_provenance_spec`
and :func:`parse_tool_provenance_operation`. Runtime authoring code should create specs
with :func:`full_data`, :func:`selection`, and the operation builder helpers.
"""

from __future__ import annotations

__all__ = [
    "DerivationEntry",
    "ToolProvenanceOperation",
    "ToolProvenanceSpec",
    "append_operations",
    "assign_coords",
    "average",
    "coarsen",
    "correct_with_edge",
    "decode_provenance_value",
    "encode_provenance_value",
    "full_data",
    "isel",
    "mask_with_polygon",
    "parse_tool_provenance_spec",
    "qsel",
    "rename",
    "rotate",
    "sel",
    "selection",
    "slice_along_path",
    "sort_coord_order",
    "squeeze",
    "swap_dims",
    "symmetrize",
    "symmetrize_nfold",
    "thin",
    "transpose",
]

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pydantic
import xarray as xr

import erlab

_SLICE_MARKER = "__erlab_slice__"
_DATASET_MARKER = "__erlab_xarray_dataset__"
_DATAARRAY_MARKER = "__erlab_xarray_dataarray__"


@dataclass(frozen=True)
class DerivationEntry:
    """One user-visible step in a provenance derivation listing."""

    label: str
    code: str | None
    copyable: bool = False


def encode_provenance_value(value: typing.Any) -> typing.Any:
    """Encode non-JSON provenance values into a JSON-safe representation."""
    if isinstance(value, xr.Dataset):
        return {_DATASET_MARKER: value.to_dict(data="list")}
    if isinstance(value, xr.DataArray):
        return {_DATAARRAY_MARKER: value.to_dict(data="list")}
    if isinstance(value, slice):
        return {
            _SLICE_MARKER: [
                encode_provenance_value(value.start),
                encode_provenance_value(value.stop),
                encode_provenance_value(value.step),
            ]
        }
    if isinstance(value, np.ndarray):
        return [encode_provenance_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [encode_provenance_value(item) for item in value]
    if isinstance(value, list):
        return [encode_provenance_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): encode_provenance_value(item) for key, item in value.items()}
    return erlab.utils.misc._convert_to_native(value)


def decode_provenance_value(value: typing.Any) -> typing.Any:
    """Decode values produced by :func:`encode_provenance_value`."""
    if isinstance(value, Mapping):
        if _DATASET_MARKER in value:
            return xr.Dataset.from_dict(
                typing.cast("dict[str, typing.Any]", value[_DATASET_MARKER])
            )
        if _DATAARRAY_MARKER in value:
            return xr.DataArray.from_dict(
                typing.cast("dict[str, typing.Any]", value[_DATAARRAY_MARKER])
            )
        if _SLICE_MARKER in value:
            start, stop, step = typing.cast("list[typing.Any]", value[_SLICE_MARKER])
            return slice(
                decode_provenance_value(start),
                decode_provenance_value(stop),
                decode_provenance_value(step),
            )
        return {str(key): decode_provenance_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_provenance_value(item) for item in value]
    return value


def _utils():
    import erlab.interactive.utils

    return erlab.interactive.utils


def _code_literal(value: typing.Any) -> str:
    return str(_utils()._parse_single_arg(value))


def _format_derivation_value(value: typing.Any) -> str:
    decoded = decode_provenance_value(value)
    if isinstance(decoded, Mapping):
        return _utils().format_kwargs(dict(decoded))
    if isinstance(decoded, (tuple, list)):
        return repr(tuple(decoded))
    return repr(decoded)


def _encode_mapping(
    value: Mapping[typing.Any, typing.Any] | None,
) -> dict[str, typing.Any]:
    if value is None:
        return {}
    return typing.cast("dict[str, typing.Any]", encode_provenance_value(dict(value)))


def _ensure_string_tuple(
    value: Sequence[typing.Any], *, expected_len: int | None = None
) -> tuple[str, ...]:
    items = tuple(str(item) for item in value)
    if expected_len is not None and len(items) != expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(items)}")
    return items


def _ensure_float_tuple(
    value: Sequence[typing.Any], *, expected_len: int | None = None
) -> tuple[float, ...]:
    items = tuple(float(item) for item in value)
    if expected_len is not None and len(items) != expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(items)}")
    return items


def _ensure_str_mapping(
    value: Mapping[typing.Any, typing.Any] | None,
) -> dict[str, str]:
    if value is None:
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _ensure_int_mapping(
    value: Mapping[typing.Any, typing.Any] | None,
) -> dict[str, int]:
    if value is None:
        return {}
    return {str(key): int(item) for key, item in value.items()}


def _ensure_float_mapping(
    value: Mapping[typing.Any, typing.Any] | None,
) -> dict[str, float]:
    if value is None:
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _format_selection_step(method: str, kwargs: dict[str, typing.Any]) -> str:
    if not kwargs:
        return f"derived = derived.{method}()"
    if all(key.isidentifier() for key in kwargs):
        args = _utils().format_kwargs(kwargs)
    else:
        args = "**" + _code_literal(kwargs)
    return f"derived = derived.{method}({args})"


class _BaseOperation(pydantic.BaseModel):
    """Base class for immutable provenance operations.

    New operations should keep persisted fields JSON-safe after validation, implement
    :meth:`apply` to replay the transformation, and implement
    :meth:`derivation_entry` to describe the step in manager UI.
    """

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError

    def derivation_entry(self) -> DerivationEntry | None:
        raise NotImplementedError


class QSelOperation(_BaseOperation):
    op: typing.Literal["qsel"] = "qsel"
    kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("kwargs", mode="before")
    @classmethod
    def _validate_kwargs(cls, value: typing.Any) -> dict[str, typing.Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("qsel kwargs must be a mapping")
        return _encode_mapping(value)

    @property
    def decoded_kwargs(self) -> dict[str, typing.Any]:
        return typing.cast(
            "dict[str, typing.Any]", decode_provenance_value(self.kwargs)
        )

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.qsel(**self.decoded_kwargs)

    def derivation_entry(self) -> DerivationEntry:
        kwargs = self.decoded_kwargs
        return DerivationEntry(
            f"qsel({_format_derivation_value(self.kwargs)})",
            _format_selection_step("qsel", kwargs),
            True,
        )


class IselOperation(_BaseOperation):
    op: typing.Literal["isel"] = "isel"
    kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("kwargs", mode="before")
    @classmethod
    def _validate_kwargs(cls, value: typing.Any) -> dict[str, typing.Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("isel kwargs must be a mapping")
        return _encode_mapping(value)

    @property
    def decoded_kwargs(self) -> dict[str, typing.Any]:
        return typing.cast(
            "dict[str, typing.Any]", decode_provenance_value(self.kwargs)
        )

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.isel(self.decoded_kwargs)

    def derivation_entry(self) -> DerivationEntry:
        kwargs = self.decoded_kwargs
        return DerivationEntry(
            f"isel({_format_derivation_value(self.kwargs)})",
            _format_selection_step("isel", kwargs),
            True,
        )


class SelOperation(_BaseOperation):
    op: typing.Literal["sel"] = "sel"
    kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("kwargs", mode="before")
    @classmethod
    def _validate_kwargs(cls, value: typing.Any) -> dict[str, typing.Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("sel kwargs must be a mapping")
        return _encode_mapping(value)

    @property
    def decoded_kwargs(self) -> dict[str, typing.Any]:
        return typing.cast(
            "dict[str, typing.Any]", decode_provenance_value(self.kwargs)
        )

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.sel(self.decoded_kwargs)

    def derivation_entry(self) -> DerivationEntry:
        kwargs = self.decoded_kwargs
        return DerivationEntry(
            f"sel({_format_derivation_value(self.kwargs)})",
            _format_selection_step("sel", kwargs),
            True,
        )


class SortCoordOrderOperation(_BaseOperation):
    op: typing.Literal["sort_coord_order"] = "sort_coord_order"

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array.sort_coord_order(data, parent_data.coords.keys())

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(
            "Sort coordinates to parent order",
            "derived = erlab.utils.array.sort_coord_order(derived, data.coords.keys())",
            True,
        )


class TransposeOperation(_BaseOperation):
    op: typing.Literal["transpose"] = "transpose"
    dims: tuple[str, ...] | None = None

    @pydantic.field_validator("dims", mode="before")
    @classmethod
    def _validate_dims(cls, value: typing.Any) -> tuple[str, ...] | None:
        if value is None:
            return None
        return _ensure_string_tuple(typing.cast("Sequence[typing.Any]", value))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        if self.dims:
            return data.transpose(*self.dims)
        return data.transpose(*reversed(data.dims))

    def derivation_entry(self) -> DerivationEntry:
        if self.dims:
            dims_tuple = tuple(self.dims)
            return DerivationEntry(
                f"transpose({_format_derivation_value(dims_tuple)})",
                f"derived = derived.transpose(*{_code_literal(dims_tuple)})",
                True,
            )
        return DerivationEntry(
            "transpose()",
            "derived = derived.transpose(*reversed(derived.dims))",
            True,
        )


class SqueezeOperation(_BaseOperation):
    op: typing.Literal["squeeze"] = "squeeze"

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.squeeze()

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry("squeeze()", "derived = derived.squeeze()", True)


class RenameOperation(_BaseOperation):
    op: typing.Literal["rename"] = "rename"
    name: str

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.rename(self.name)

    def derivation_entry(self) -> None:
        return None


class RotateOperation(_BaseOperation):
    op: typing.Literal["rotate"] = "rotate"
    angle: float
    axes: tuple[str, str]
    center: tuple[float, float]
    reshape: bool = True
    order: int = 1

    @pydantic.field_validator("axes", mode="before")
    @classmethod
    def _validate_axes(cls, value: typing.Any) -> tuple[str, str]:
        return typing.cast(
            "tuple[str, str]",
            _ensure_string_tuple(
                typing.cast("Sequence[typing.Any]", value), expected_len=2
            ),
        )

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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.rotate(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = _utils().generate_code(
            erlab.analysis.transform.rotate,
            ["|derived|"],
            self.kwargs,
            module="era.transform",
            assign="derived",
        )
        return DerivationEntry(
            f"Rotate({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class AverageOperation(_BaseOperation):
    op: typing.Literal["average"] = "average"
    dims: tuple[str, ...]

    @pydantic.field_validator("dims", mode="before")
    @classmethod
    def _validate_dims(cls, value: typing.Any) -> tuple[str, ...]:
        return _ensure_string_tuple(typing.cast("Sequence[typing.Any]", value))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.qsel.average(self.dims)

    def derivation_entry(self) -> DerivationEntry:
        arg = (
            _code_literal(self.dims)
            if len(self.dims) > 1
            else _code_literal(self.dims[0])
        )
        label_kwargs = {"dims": self.dims}
        return DerivationEntry(
            f"Average({_format_derivation_value(label_kwargs)})",
            f"derived = derived.qsel.average({arg})",
            True,
        )


class CoarsenOperation(_BaseOperation):
    op: typing.Literal["coarsen"] = "coarsen"
    dim: dict[str, int] = pydantic.Field(default_factory=dict)
    boundary: str
    side: str
    coord_func: str
    reducer: str

    @pydantic.field_validator("dim", mode="before")
    @classmethod
    def _validate_dim(cls, value: typing.Any) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("coarsen dim must be a mapping")
        return _ensure_int_mapping(value)

    @property
    def coarsen_kwargs(self) -> dict[str, typing.Any]:
        return {
            "dim": self.dim,
            "boundary": self.boundary,
            "side": self.side,
            "coord_func": self.coord_func,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        coarsened = data.coarsen(**self.coarsen_kwargs)
        return typing.cast("xr.DataArray", getattr(coarsened, self.reducer)())

    def derivation_entry(self) -> DerivationEntry:
        coarsen_kwargs = dict(self.coarsen_kwargs)
        if all(isinstance(key, str) for key in coarsen_kwargs["dim"]):
            window_kwargs: dict[str, typing.Any] = dict(coarsen_kwargs.pop("dim"))
            coarsen_kwargs = {**window_kwargs, **coarsen_kwargs}
        code = (
            "derived = "
            + _utils().generate_code(
                xr.DataArray.coarsen,
                [],
                coarsen_kwargs,
                module="derived",
            )
            + f".{self.reducer}()"
        )
        label_kwargs = {"coarsen_kwargs": self.coarsen_kwargs, "reducer": self.reducer}
        return DerivationEntry(
            f"Coarsen({_format_derivation_value(label_kwargs)})",
            code,
            True,
        )


class ThinOperation(_BaseOperation):
    op: typing.Literal["thin"] = "thin"
    mode: typing.Literal["global", "per_dim"]
    factor: int | None = None
    factors: dict[str, int] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("factors", mode="before")
    @classmethod
    def _validate_factors(cls, value: typing.Any) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("thin factors must be a mapping")
        return _ensure_int_mapping(value)

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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        if self.mode == "global":
            return data.thin(int(typing.cast("int", self.factor)))
        return data.thin(self.factors)

    def derivation_entry(self) -> DerivationEntry:
        if self.mode == "global":
            code = f"derived = derived.thin({int(typing.cast('int', self.factor))})"
        else:
            code = f"derived = derived.thin({_code_literal(self.factors)})"
        return DerivationEntry(
            f"Thin({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class SymmetrizeOperation(_BaseOperation):
    op: typing.Literal["symmetrize"] = "symmetrize"
    dim: str
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = _utils().generate_code(
            erlab.analysis.transform.symmetrize,
            ["|derived|"],
            self.kwargs,
            module="era.transform",
            assign="derived",
        )
        return DerivationEntry(
            f"Symmetrize({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class SymmetrizeNfoldOperation(_BaseOperation):
    op: typing.Literal["symmetrize_nfold"] = "symmetrize_nfold"
    fold: int
    axes: tuple[str, str]
    center: dict[str, float] = pydantic.Field(default_factory=dict)
    reshape: bool = True
    order: int = 1

    @pydantic.field_validator("axes", mode="before")
    @classmethod
    def _validate_axes(cls, value: typing.Any) -> tuple[str, str]:
        return typing.cast(
            "tuple[str, str]",
            _ensure_string_tuple(
                typing.cast("Sequence[typing.Any]", value), expected_len=2
            ),
        )

    @pydantic.field_validator("center", mode="before")
    @classmethod
    def _validate_center(cls, value: typing.Any) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("symmetrize_nfold center must be a mapping")
        return _ensure_float_mapping(value)

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "fold": self.fold,
            "axes": self.axes,
            "center": self.center,
            "reshape": self.reshape,
            "order": self.order,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize_nfold(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = _utils().generate_code(
            erlab.analysis.transform.symmetrize_nfold,
            ["|derived|"],
            self.kwargs,
            module="era.transform",
            assign="derived",
        )
        return DerivationEntry(
            f"Rotational Symmetrize({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class CorrectWithEdgeOperation(_BaseOperation):
    op: typing.Literal["correct_with_edge"] = "correct_with_edge"
    edge_fit: dict[str, typing.Any]
    shift_coords: bool = True

    @pydantic.field_validator("edge_fit", mode="before")
    @classmethod
    def _validate_edge_fit(cls, value: typing.Any) -> dict[str, typing.Any]:
        encoded = encode_provenance_value(value)
        if not (isinstance(encoded, Mapping) and _DATASET_MARKER in encoded):
            raise TypeError("correct_with_edge edge_fit must be an xarray.Dataset")
        return dict(encoded)

    @property
    def decoded_edge_fit(self) -> xr.Dataset:
        return typing.cast("xr.Dataset", decode_provenance_value(self.edge_fit))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.gold.correct_with_edge(
            data,
            self.decoded_edge_fit,
            shift_coords=self.shift_coords,
        )

    def derivation_entry(self) -> DerivationEntry:
        label_kwargs = {
            "edge_fit": self.edge_fit,
            "shift_coords": self.shift_coords,
        }
        return DerivationEntry(
            f"Edge Correction({_format_derivation_value(label_kwargs)})",
            None,
            False,
        )


class SwapDimsOperation(_BaseOperation):
    op: typing.Literal["swap_dims"] = "swap_dims"
    mapping: dict[str, str] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("mapping", mode="before")
    @classmethod
    def _validate_mapping(cls, value: typing.Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("swap_dims mapping must be a mapping")
        return _ensure_str_mapping(value)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.swap_dims(self.mapping)

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(
            f"Swap Dimensions({_format_derivation_value(self.mapping)})",
            f"derived = derived.swap_dims({_code_literal(self.mapping)})",
            True,
        )


class AssignCoordsOperation(_BaseOperation):
    op: typing.Literal["assign_coords"] = "assign_coords"
    coord_name: str
    values: list[typing.Any]

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: typing.Any) -> list[typing.Any]:
        encoded = encode_provenance_value(value)
        if not isinstance(encoded, list):
            raise TypeError("assign_coords values must be array-like")
        return encoded

    @property
    def decoded_values(self) -> np.ndarray:
        return np.asarray(decode_provenance_value(self.values))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array.sort_coord_order(
            data.assign_coords(
                {self.coord_name: data[self.coord_name].copy(data=self.decoded_values)}
            ),
            keys=data.coords.keys(),
            dims_first=False,
        )

    def derivation_entry(self) -> DerivationEntry:
        coord_name_code = repr(self.coord_name)
        values_code = _code_literal(self.decoded_values)
        code = (
            "derived = erlab.utils.array.sort_coord_order("
            f"derived.assign_coords({{{coord_name_code}: "
            f"derived[{coord_name_code}].copy(data={values_code})}}), "
            "keys=derived.coords.keys(), dims_first=False)"
        )
        label_kwargs = {
            "coord_name": self.coord_name,
            "values": self.values,
        }
        return DerivationEntry(
            f"Assign Coordinates({_format_derivation_value(label_kwargs)})",
            code,
            True,
        )


class SliceAlongPathOperation(_BaseOperation):
    op: typing.Literal["slice_along_path"] = "slice_along_path"
    vertices: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    step_size: float
    dim_name: str

    @pydantic.field_validator("vertices", mode="before")
    @classmethod
    def _validate_vertices(cls, value: typing.Any) -> dict[str, typing.Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("slice_along_path vertices must be a mapping")
        return _encode_mapping(value)

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "vertices": typing.cast(
                "dict[str, typing.Any]", decode_provenance_value(self.vertices)
            ),
            "step_size": self.step_size,
            "dim_name": self.dim_name,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.interpolate.slice_along_path(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = _utils().generate_code(
            erlab.analysis.interpolate.slice_along_path,
            ["|derived|"],
            self.kwargs,
            module="era.interpolate",
            assign="derived",
        )
        return DerivationEntry(
            f"Slice Along ROI Path({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class MaskWithPolygonOperation(_BaseOperation):
    op: typing.Literal["mask_with_polygon"] = "mask_with_polygon"
    vertices: list[typing.Any]
    dims: tuple[str, ...]
    invert: bool = False
    drop: bool = False

    @pydantic.field_validator("vertices", mode="before")
    @classmethod
    def _validate_vertices(cls, value: typing.Any) -> list[typing.Any]:
        encoded = encode_provenance_value(value)
        if not isinstance(encoded, list):
            raise TypeError("mask_with_polygon vertices must be array-like")
        return encoded

    @pydantic.field_validator("dims", mode="before")
    @classmethod
    def _validate_dims(cls, value: typing.Any) -> tuple[str, ...]:
        return _ensure_string_tuple(typing.cast("Sequence[typing.Any]", value))

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "vertices": np.asarray(decode_provenance_value(self.vertices)),
            "dims": self.dims,
            "invert": self.invert,
            "drop": self.drop,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.mask.mask_with_polygon(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = _utils().generate_code(
            erlab.analysis.mask.mask_with_polygon,
            ["|derived|"],
            self.kwargs,
            module="era.mask",
            assign="derived",
        )
        return DerivationEntry(
            f"Mask with ROI({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


ToolProvenanceOperation = typing.Annotated[
    QSelOperation
    | IselOperation
    | SelOperation
    | SortCoordOrderOperation
    | TransposeOperation
    | SqueezeOperation
    | RenameOperation
    | RotateOperation
    | AverageOperation
    | CoarsenOperation
    | ThinOperation
    | SymmetrizeOperation
    | SymmetrizeNfoldOperation
    | CorrectWithEdgeOperation
    | SwapDimsOperation
    | AssignCoordsOperation
    | SliceAlongPathOperation
    | MaskWithPolygonOperation,
    pydantic.Field(discriminator="op"),
]

_OPERATION_ADAPTER: pydantic.TypeAdapter[ToolProvenanceOperation] = (
    pydantic.TypeAdapter(ToolProvenanceOperation)
)


def _require_operation_instance(
    operation: ToolProvenanceOperation,
) -> ToolProvenanceOperation:
    """Require a fully constructed operation instance for runtime authoring helpers."""
    if not isinstance(operation, _BaseOperation):
        raise TypeError(
            "Runtime provenance builders accept ToolProvenanceOperation instances "
            "only. Use parse_tool_provenance_operation() when deserializing mappings."
        )
    return operation


def parse_tool_provenance_operation(
    value: ToolProvenanceOperation | Mapping[str, typing.Any],
) -> ToolProvenanceOperation:
    """Parse one serialized operation payload.

    This is the deserialize boundary for saved JSON. Runtime call sites should build
    operations with the dedicated helper functions in this module instead of passing
    raw mappings around.
    """
    if isinstance(value, _BaseOperation):
        return value
    return _OPERATION_ADAPTER.validate_python(value)


def append_operations(
    spec: ToolProvenanceSpec,
    *operations: ToolProvenanceOperation,
) -> ToolProvenanceSpec:
    """Return ``spec`` with additional operation instances appended."""
    return spec.append_operations(*operations)


class ToolProvenanceSpec(pydantic.BaseModel):
    """Immutable provenance recipe for rebuilding tool data from a parent ImageTool.

    Author new specs with :func:`full_data` or :func:`selection`. Deserialize saved
    payloads with :func:`parse_tool_provenance_spec`.
    """

    schema_version: typing.Literal[1] = 1
    kind: typing.Literal["full_data", "selection"]
    operations: tuple[ToolProvenanceOperation, ...] = ()

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.field_validator("operations", mode="before")
    @classmethod
    def _validate_operations(
        cls, value: typing.Any
    ) -> tuple[ToolProvenanceOperation, ...]:
        if value is None:
            return ()
        return tuple(
            parse_tool_provenance_operation(
                typing.cast("ToolProvenanceOperation | Mapping[str, typing.Any]", item)
            )
            for item in typing.cast("Sequence[typing.Any]", value)
        )

    def append_operations(
        self, *operations: ToolProvenanceOperation
    ) -> ToolProvenanceSpec:
        """Append operation instances to the spec.

        Runtime code should pass builder results from this module. Saved mappings should
        be normalized with :func:`parse_tool_provenance_spec` before calling this
        method.
        """
        appended = tuple(_require_operation_instance(op) for op in operations)
        return self.model_copy(update={"operations": (*self.operations, *appended)})

    def drop_trailing_rename(self) -> ToolProvenanceSpec:
        operations = self.operations
        if operations and isinstance(operations[-1], RenameOperation):
            operations = operations[:-1]
        return self.model_copy(update={"operations": operations})

    def append_replacement_operations(
        self, *operations: ToolProvenanceOperation
    ) -> ToolProvenanceSpec:
        """Replace a final rename, if present, then append new operation instances."""
        return self.drop_trailing_rename().append_operations(*operations)

    def append_final_rename(self, name: str) -> ToolProvenanceSpec:
        return self.drop_trailing_rename().append_operations(RenameOperation(name=name))

    def apply(self, parent_data: xr.DataArray) -> xr.DataArray:
        if self.kind == "full_data":
            data = parent_data.copy(deep=False)
        else:
            data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
                parent_data.copy(deep=False)
            )
        for operation in self.operations:
            data = operation.apply(data, parent_data=parent_data)
        return data

    def derivation_entries(self) -> list[DerivationEntry]:
        if self.kind == "full_data":
            entries: list[DerivationEntry] = [
                DerivationEntry("Start from current parent ImageTool data", None, False)
            ]
        else:
            entries = [
                DerivationEntry(
                    "Start from selected parent ImageTool data", None, False
                )
            ]
        for operation in self.operations:
            entry = operation.derivation_entry()
            if entry is not None:
                entries.append(entry)
        return entries

    def derivation_code(self) -> str | None:
        step_codes: list[str] = []
        for entry in self.derivation_entries()[1:]:
            if entry.code is None:
                return None
            step_codes.append(entry.code)
        if not step_codes:
            return None
        return "\n".join(["derived = data", *step_codes])


def parse_tool_provenance_spec(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> ToolProvenanceSpec | None:
    """Parse a serialized provenance payload into a validated spec instance.

    This is the deserialize boundary for saved tool and workspace metadata. Runtime
    authoring code should pass :class:`ToolProvenanceSpec` instances directly.
    """
    if value is None:
        return None
    if isinstance(value, ToolProvenanceSpec):
        return value
    return ToolProvenanceSpec.model_validate(value)


def full_data(
    *operations: ToolProvenanceOperation,
) -> ToolProvenanceSpec:
    """Build a spec that starts from the parent's full current data."""
    return ToolProvenanceSpec(kind="full_data").append_operations(*operations)


def selection(
    *operations: ToolProvenanceOperation,
) -> ToolProvenanceSpec:
    """Build a spec that starts from the parent's public selection model."""
    return ToolProvenanceSpec(kind="selection").append_operations(*operations)


def qsel(**kwargs: typing.Any) -> QSelOperation:
    return QSelOperation(kwargs=kwargs)


def isel(**kwargs: typing.Any) -> IselOperation:
    return IselOperation(kwargs=kwargs)


def sel(**kwargs: typing.Any) -> SelOperation:
    return SelOperation(kwargs=kwargs)


def sort_coord_order() -> SortCoordOrderOperation:
    return SortCoordOrderOperation()


def transpose(*dims: typing.Any) -> TransposeOperation:
    return TransposeOperation(
        dims=None if not dims else tuple(str(dim) for dim in dims)
    )


def squeeze() -> SqueezeOperation:
    return SqueezeOperation()


def rename(name: str) -> RenameOperation:
    return RenameOperation(name=name)


def rotate(
    *,
    angle: float,
    axes: Sequence[typing.Any],
    center: Sequence[typing.Any],
    reshape: bool = True,
    order: int = 1,
) -> RotateOperation:
    return RotateOperation(
        angle=angle,
        axes=typing.cast("tuple[str, str]", tuple(str(item) for item in axes)),
        center=typing.cast(
            "tuple[float, float]", tuple(float(item) for item in center)
        ),
        reshape=reshape,
        order=order,
    )


def average(*dims: typing.Any) -> AverageOperation:
    return AverageOperation(dims=tuple(str(dim) for dim in dims))


def coarsen(
    *,
    dim: Mapping[typing.Any, typing.Any],
    boundary: str,
    side: str,
    coord_func: str,
    reducer: str,
) -> CoarsenOperation:
    return CoarsenOperation(
        dim=typing.cast("dict[str, int]", dict(dim)),
        boundary=boundary,
        side=side,
        coord_func=coord_func,
        reducer=reducer,
    )


def thin(
    *, factor: int | None = None, factors: Mapping[typing.Any, typing.Any] | None = None
) -> ThinOperation:
    if factors is not None:
        return ThinOperation(mode="per_dim", factors=dict(factors))
    return ThinOperation(mode="global", factor=int(typing.cast("int", factor)))


def symmetrize(
    *,
    dim: str,
    center: float,
    subtract: bool = False,
    mode: typing.Literal["full", "valid"] = "full",
    part: typing.Literal["both", "below", "above"] = "both",
) -> SymmetrizeOperation:
    return SymmetrizeOperation(
        dim=dim,
        center=center,
        subtract=subtract,
        mode=mode,
        part=part,
    )


def symmetrize_nfold(
    *,
    fold: int,
    axes: Sequence[typing.Any],
    center: Mapping[typing.Any, typing.Any],
    reshape: bool = True,
    order: int = 1,
) -> SymmetrizeNfoldOperation:
    return SymmetrizeNfoldOperation(
        fold=fold,
        axes=typing.cast("tuple[str, str]", tuple(str(item) for item in axes)),
        center=dict(center),
        reshape=reshape,
        order=order,
    )


def correct_with_edge(
    *, edge_fit: xr.Dataset, shift_coords: bool = True
) -> CorrectWithEdgeOperation:
    return CorrectWithEdgeOperation(
        edge_fit=typing.cast(
            "dict[str, typing.Any]", encode_provenance_value(edge_fit)
        ),
        shift_coords=shift_coords,
    )


def swap_dims(mapping: Mapping[typing.Any, typing.Any]) -> SwapDimsOperation:
    return SwapDimsOperation(mapping=dict(mapping))


def assign_coords(*, coord_name: str, values: typing.Any) -> AssignCoordsOperation:
    return AssignCoordsOperation(coord_name=coord_name, values=values)


def slice_along_path(
    *, vertices: Mapping[typing.Any, typing.Any], step_size: float, dim_name: str
) -> SliceAlongPathOperation:
    return SliceAlongPathOperation(
        vertices=dict(vertices),
        step_size=step_size,
        dim_name=dim_name,
    )


def mask_with_polygon(
    *,
    vertices: typing.Any,
    dims: Sequence[typing.Any],
    invert: bool = False,
    drop: bool = False,
) -> MaskWithPolygonOperation:
    return MaskWithPolygonOperation(
        vertices=vertices,
        dims=tuple(str(dim) for dim in dims),
        invert=invert,
        drop=drop,
    )
