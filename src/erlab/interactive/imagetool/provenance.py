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
6. Give the class a unique ``op`` discriminator literal. Subclasses register
   themselves automatically so serialized payloads can dispatch to the right model.
7. Export the concrete operation class from this module so runtime call sites can
   instantiate it directly.
8. Add tests that cover round-trip validation, :meth:`apply`, and derivation text/code,
   plus any save/load path that persists the new operation.

Parsing of serialized payloads happens only through :func:`parse_tool_provenance_spec`
and :func:`parse_tool_provenance_operation`. Runtime authoring code should create specs
with :func:`full_data` and :func:`selection`, then instantiate concrete operation
models from this module directly.
"""

from __future__ import annotations

__all__ = [
    "AssignCoordsOperation",
    "AverageOperation",
    "CoarsenOperation",
    "CorrectWithEdgeOperation",
    "DerivationEntry",
    "IselOperation",
    "MaskWithPolygonOperation",
    "QSelOperation",
    "RenameOperation",
    "RotateOperation",
    "SelOperation",
    "SliceAlongPathOperation",
    "SortCoordOrderOperation",
    "SqueezeOperation",
    "SwapDimsOperation",
    "SymmetrizeNfoldOperation",
    "SymmetrizeOperation",
    "ThinOperation",
    "ToolProvenanceOperation",
    "ToolProvenanceSpec",
    "TransposeOperation",
    "append_operations",
    "decode_provenance_value",
    "encode_provenance_value",
    "full_data",
    "parse_tool_provenance_spec",
    "selection",
]

import typing
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pydantic
import xarray as xr

import erlab

_SLICE_MARKER = "__erlab_slice__"
_DATASET_MARKER = "__erlab_xarray_dataset__"
_DATAARRAY_MARKER = "__erlab_xarray_dataarray__"
_TUPLE_MARKER = "__erlab_tuple__"
_MAPPING_MARKER = "__erlab_mapping__"


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
    value = erlab.utils.misc._convert_to_native(value)
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
        return {_TUPLE_MARKER: [encode_provenance_value(item) for item in value]}
    if isinstance(value, list):
        return [encode_provenance_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            _MAPPING_MARKER: [
                [_encode_provenance_hashable(key), encode_provenance_value(item)]
                for key, item in value.items()
            ]
        }
    return value


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
        if _TUPLE_MARKER in value:
            return tuple(
                decode_provenance_value(item)
                for item in typing.cast("list[typing.Any]", value[_TUPLE_MARKER])
            )
        if _MAPPING_MARKER in value:
            return {
                decode_provenance_value(key): decode_provenance_value(item)
                for key, item in typing.cast(
                    "list[list[typing.Any]]", value[_MAPPING_MARKER]
                )
            }
        return {
            decode_provenance_value(key): decode_provenance_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [decode_provenance_value(item) for item in value]
    return erlab.utils.misc._convert_to_native(value)


def _normalize_provenance_hashable(value: typing.Any) -> Hashable:
    value = erlab.utils.misc._convert_to_native(value)
    if isinstance(value, tuple):
        return tuple(_normalize_provenance_hashable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return typing.cast("Hashable", value)
    raise TypeError(
        "provenance hashable fields only support str, int, float, bool, None, "
        "and tuples composed of those values"
    )


def _encode_provenance_hashable(value: typing.Any) -> typing.Any:
    normalized = _normalize_provenance_hashable(value)
    if isinstance(normalized, tuple):
        return {
            _TUPLE_MARKER: [_encode_provenance_hashable(item) for item in normalized]
        }
    return normalized


def _is_identifier_string_mapping(value: Mapping[typing.Any, typing.Any]) -> bool:
    return all(isinstance(key, str) and key.isidentifier() for key in value)


def _format_derivation_value(value: typing.Any) -> str:
    decoded = decode_provenance_value(value)
    if isinstance(decoded, Mapping):
        return erlab.interactive.utils.format_kwargs(dict(decoded))
    if isinstance(decoded, (tuple, list)):
        return repr(tuple(decoded))
    return repr(decoded)


def _ensure_float_tuple(
    value: Sequence[typing.Any], *, expected_len: int | None = None
) -> tuple[float, ...]:
    items = tuple(float(item) for item in value)
    if expected_len is not None and len(items) != expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(items)}")
    return items


def _coerce_float_sequence(value: typing.Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError("expected an array-like sequence")
    return [float(item) for item in value]


def _format_selection_step(method: str, kwargs: Mapping[Hashable, typing.Any]) -> str:
    if not kwargs:
        return f"derived = derived.{method}()"
    args = erlab.interactive.utils.format_call_kwargs(dict(kwargs))
    return f"derived = derived.{method}({args})"


_OPERATION_TYPES: dict[str, type[_BaseOperation]] = {}


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

    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        super().__pydantic_on_complete__()

        if "op" not in cls.__dict__.get("__annotations__", {}):
            return

        op_field = cls.model_fields.get("op")
        op = None if op_field is None else op_field.default
        if not isinstance(op, str):
            raise TypeError(f"{cls.__name__} must define `op` with a string default")

        existing = _OPERATION_TYPES.get(op)
        if existing is not None and existing is not cls:
            raise ValueError(f"Duplicate provenance operation discriminator {op!r}")

        _OPERATION_TYPES[op] = cls

    @staticmethod
    def _json_encode_field(value: typing.Any) -> typing.Any:
        return encode_provenance_value(value)

    @classmethod
    def _validate_encoded_field(
        cls,
        value: typing.Any,
        *,
        error: str,
        predicate: Callable[[typing.Any], bool],
    ) -> typing.Any:
        encoded = encode_provenance_value(value)
        if not predicate(encoded):
            raise TypeError(error)
        return encoded

    @staticmethod
    def _decode_stored_field(value: typing.Any) -> typing.Any:
        return decode_provenance_value(value)

    @staticmethod
    def _coerce_hashable_field(value: typing.Any) -> Hashable:
        return _normalize_provenance_hashable(decode_provenance_value(value))

    @staticmethod
    def _coerce_hashable_tuple_field(
        value: typing.Any, *, expected_len: int | None = None
    ) -> tuple[Hashable, ...]:
        decoded = decode_provenance_value(value)
        if not isinstance(decoded, Sequence) or isinstance(decoded, str):
            raise TypeError("expected a sequence of provenance-compatible hashables")
        items = tuple(_normalize_provenance_hashable(item) for item in decoded)
        if expected_len is not None and len(items) != expected_len:
            raise ValueError(f"Expected {expected_len} items, got {len(items)}")
        return items

    @staticmethod
    def _coerce_hashable_mapping_field(
        value: typing.Any,
        *,
        value_coerce: Callable[[typing.Any], typing.Any] | None = None,
    ) -> dict[Hashable, typing.Any]:
        if value is None:
            return {}
        decoded = decode_provenance_value(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("expected a mapping with provenance-compatible keys")
        transform = (lambda item: item) if value_coerce is None else value_coerce
        return {
            _normalize_provenance_hashable(key): transform(item)
            for key, item in decoded.items()
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError

    def derivation_entry(self) -> DerivationEntry | None:
        raise NotImplementedError


def _coerce_nullable_hashable_tuple_field(
    value: typing.Any,
) -> tuple[Hashable, ...] | None:
    if value is None:
        return None
    return _BaseOperation._coerce_hashable_tuple_field(value)


def _coerce_hashable_pair_field(value: typing.Any) -> tuple[Hashable, Hashable]:
    return typing.cast(
        "tuple[Hashable, Hashable]",
        _BaseOperation._coerce_hashable_tuple_field(value, expected_len=2),
    )


def _coerce_int_mapping_field(value: typing.Any) -> dict[Hashable, int]:
    return typing.cast(
        "dict[Hashable, int]",
        _BaseOperation._coerce_hashable_mapping_field(value, value_coerce=int),
    )


def _coerce_float_mapping_field(value: typing.Any) -> dict[Hashable, float]:
    return typing.cast(
        "dict[Hashable, float]",
        _BaseOperation._coerce_hashable_mapping_field(value, value_coerce=float),
    )


def _coerce_hashable_mapping_field(value: typing.Any) -> dict[Hashable, Hashable]:
    return typing.cast(
        "dict[Hashable, Hashable]",
        _BaseOperation._coerce_hashable_mapping_field(
            value, value_coerce=_normalize_provenance_hashable
        ),
    )


def _coerce_float_sequence_mapping_field(
    value: typing.Any,
) -> dict[Hashable, list[float]]:
    return typing.cast(
        "dict[Hashable, list[float]]",
        _BaseOperation._coerce_hashable_mapping_field(
            value, value_coerce=_coerce_float_sequence
        ),
    )


ProvenanceHashable: typing.TypeAlias = typing.Annotated[
    Hashable,
    pydantic.BeforeValidator(_BaseOperation._coerce_hashable_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]

NullableProvenanceHashableTuple: typing.TypeAlias = typing.Annotated[
    tuple[Hashable, ...] | None,
    pydantic.BeforeValidator(_coerce_nullable_hashable_tuple_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]

ProvenanceHashableTuple: typing.TypeAlias = typing.Annotated[
    tuple[Hashable, ...],
    pydantic.BeforeValidator(_BaseOperation._coerce_hashable_tuple_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]

ProvenanceHashablePair: typing.TypeAlias = typing.Annotated[
    tuple[Hashable, Hashable],
    pydantic.BeforeValidator(_coerce_hashable_pair_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]

ProvenanceMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, typing.Any],
    pydantic.BeforeValidator(_BaseOperation._coerce_hashable_mapping_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]

ProvenanceIntMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, int],
    pydantic.BeforeValidator(_coerce_int_mapping_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]

ProvenanceFloatMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, float],
    pydantic.BeforeValidator(_coerce_float_mapping_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]

ProvenanceHashableMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, Hashable],
    pydantic.BeforeValidator(_coerce_hashable_mapping_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]

ProvenanceFloatSequenceMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, list[float]],
    pydantic.BeforeValidator(_coerce_float_sequence_mapping_field),
    pydantic.PlainSerializer(_BaseOperation._json_encode_field, when_used="json"),
]


ToolProvenanceOperation = _BaseOperation


def _require_operation_instance(
    operation: ToolProvenanceOperation,
) -> ToolProvenanceOperation:
    """Require a fully constructed operation instance for runtime use."""
    if not isinstance(operation, _BaseOperation):
        raise TypeError(
            "Runtime provenance APIs accept ToolProvenanceOperation instances "
            "only. Use parse_tool_provenance_operation() when deserializing mappings."
        )
    return operation


def parse_tool_provenance_operation(
    value: ToolProvenanceOperation | Mapping[str, typing.Any],
) -> ToolProvenanceOperation:
    """Parse one serialized operation payload.

    This is the deserialize boundary for saved JSON. Runtime call sites should build
    concrete operation instances directly instead of passing raw mappings around.
    """
    if isinstance(value, _BaseOperation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Serialized provenance operations must be mappings")

    op = value.get("op")
    if not isinstance(op, str):
        raise TypeError("Serialized provenance operations must include a string `op`")

    operation_type = _OPERATION_TYPES.get(op)
    if operation_type is None:
        raise ValueError(
            f"Unknown provenance operation {op!r}; "
            f"expected one of {sorted(_OPERATION_TYPES)}"
        )
    return operation_type.model_validate(value)


def append_operations(
    spec: ToolProvenanceSpec,
    *operations: ToolProvenanceOperation,
) -> ToolProvenanceSpec:
    """Return ``spec`` with additional operation instances appended."""
    return spec.append_operations(*operations)


class ToolProvenanceSpec(pydantic.BaseModel):
    """Immutable provenance recipe for rebuilding tool data from a parent ImageTool.

    Author new specs with :func:`full_data` or :func:`selection` plus concrete
    operation instances from this module. Deserialize saved payloads with
    :func:`parse_tool_provenance_spec`.
    """

    schema_version: typing.Literal[1] = 1
    kind: typing.Literal["full_data", "selection"]
    operations: tuple[pydantic.SerializeAsAny[ToolProvenanceOperation], ...] = ()

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

        Runtime code should pass operation instances from this module. Saved mappings
        should be normalized with :func:`parse_tool_provenance_spec` before calling
        this method.
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


class QSelOperation(_BaseOperation):
    op: typing.Literal["qsel"] = "qsel"
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.qsel(self.decoded_kwargs)

    def derivation_entry(self) -> DerivationEntry:
        kwargs = self.decoded_kwargs
        return DerivationEntry(
            f"qsel({_format_derivation_value(self.kwargs)})",
            _format_selection_step("qsel", kwargs),
            True,
        )


class IselOperation(_BaseOperation):
    op: typing.Literal["isel"] = "isel"
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

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
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

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
    dims: NullableProvenanceHashableTuple = None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        if self.dims:
            return data.transpose(*self.dims)
        return data.transpose(*reversed(data.dims))

    def derivation_entry(self) -> DerivationEntry:
        if self.dims:
            dims_tuple = tuple(self.dims)
            return DerivationEntry(
                f"transpose({_format_derivation_value(dims_tuple)})",
                "derived = derived.transpose(*"
                f"{erlab.interactive.utils._parse_single_arg(dims_tuple)})",
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.rotate(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
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
    dims: ProvenanceHashableTuple

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.qsel.average(self.dims)

    def derivation_entry(self) -> DerivationEntry:
        arg = (
            erlab.interactive.utils._parse_single_arg(self.dims[0])
            if len(self.dims) == 1 and isinstance(self.dims[0], str)
            else erlab.interactive.utils._parse_single_arg(self.dims)
        )
        label_kwargs = {"dims": self.dims}
        return DerivationEntry(
            f"Average({_format_derivation_value(label_kwargs)})",
            f"derived = derived.qsel.average({arg})",
            True,
        )


class CoarsenOperation(_BaseOperation):
    op: typing.Literal["coarsen"] = "coarsen"
    dim: ProvenanceIntMapping = pydantic.Field(default_factory=dict)
    boundary: str
    side: str
    coord_func: str
    reducer: str

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
        formatted_kwargs = erlab.interactive.utils.format_kwargs(coarsen_kwargs)
        code = f"derived = derived.coarsen({formatted_kwargs}).{self.reducer}()"
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
    factors: ProvenanceIntMapping = pydantic.Field(default_factory=dict)

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
            code = (
                "derived = derived.thin("
                f"{erlab.interactive.utils.format_call_kwargs(self.factors)})"
            )
        return DerivationEntry(
            f"Thin({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class SymmetrizeOperation(_BaseOperation):
    op: typing.Literal["symmetrize"] = "symmetrize"
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
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
    axes: ProvenanceHashablePair
    center: ProvenanceFloatMapping = pydantic.Field(default_factory=dict)
    reshape: bool = True
    order: int = 1

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
        code = erlab.interactive.utils.generate_code(
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
    edge_fit: typing.Any
    shift_coords: bool = True

    @pydantic.field_validator("edge_fit", mode="before")
    @classmethod
    def _validate_edge_fit(cls, value: typing.Any) -> typing.Any:
        return cls._validate_encoded_field(
            value,
            error="correct_with_edge edge_fit must be an xarray.Dataset",
            predicate=lambda encoded: (
                isinstance(encoded, Mapping) and _DATASET_MARKER in encoded
            ),
        )

    @property
    def decoded_edge_fit(self) -> xr.Dataset:
        return typing.cast("xr.Dataset", self._decode_stored_field(self.edge_fit))

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
    mapping: ProvenanceHashableMapping = pydantic.Field(default_factory=dict)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.swap_dims(self.mapping)

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(
            f"Swap Dimensions({_format_derivation_value(self.mapping)})",
            "derived = derived.swap_dims("
            f"{erlab.interactive.utils.format_call_kwargs(self.mapping)})",
            True,
        )


class AssignCoordsOperation(_BaseOperation):
    op: typing.Literal["assign_coords"] = "assign_coords"
    coord_name: str
    values: typing.Any

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: typing.Any) -> typing.Any:
        return cls._validate_encoded_field(
            value,
            error="assign_coords values must be array-like",
            predicate=lambda encoded: (
                isinstance(encoded, list)
                or (isinstance(encoded, Mapping) and _TUPLE_MARKER in encoded)
            ),
        )

    @property
    def decoded_values(self) -> np.ndarray:
        return np.asarray(self._decode_stored_field(self.values))

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
        values_code = erlab.interactive.utils._parse_single_arg(self.decoded_values)
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
    vertices: ProvenanceFloatSequenceMapping = pydantic.Field(default_factory=dict)
    step_size: float
    dim_name: str

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "vertices": self.vertices,
            "step_size": self.step_size,
            "dim_name": self.dim_name,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.interpolate.slice_along_path(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
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
    vertices: typing.Any
    dims: ProvenanceHashableTuple
    invert: bool = False
    drop: bool = False

    @pydantic.field_validator("vertices", mode="before")
    @classmethod
    def _validate_vertices(cls, value: typing.Any) -> typing.Any:
        return cls._validate_encoded_field(
            value,
            error="mask_with_polygon vertices must be array-like",
            predicate=lambda encoded: (
                isinstance(encoded, list)
                or (isinstance(encoded, Mapping) and _TUPLE_MARKER in encoded)
            ),
        )

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "vertices": np.asarray(self._decode_stored_field(self.vertices)),
            "dims": self.dims,
            "invert": self.invert,
            "drop": self.drop,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.mask.mask_with_polygon(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
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
