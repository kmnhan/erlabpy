"""Public ImageTool provenance API."""

from __future__ import annotations

__all__ = [
    "AffineCoordOperation",
    "AssignAttrsOperation",
    "AssignCoord1DOperation",
    "AssignCoordsOperation",
    "AssignScalarCoordOperation",
    "AverageOperation",
    "CoarsenOperation",
    "ConsoleCall",
    "ConsoleOperationPattern",
    "CorrectWithEdgeOperation",
    "DerivationEntry",
    "DivideByCoordOperation",
    "FileDataSelection",
    "FileLoadSource",
    "FileReplayCall",
    "GaussianFilterOperation",
    "ImageToolSelectionSourceBinding",
    "InterpolationOperation",
    "IselOperation",
    "LeadingEdgeOperation",
    "MaskWithPolygonOperation",
    "NormalizeOperation",
    "QSelAggregationOperation",
    "QSelOperation",
    "RenameDimsCoordsOperation",
    "RenameOperation",
    "ReplayStage",
    "RestoreNonuniformDimsOperation",
    "RotateOperation",
    "ScriptCodeOperation",
    "ScriptInput",
    "ScriptInputDependencyRef",
    "SelOperation",
    "SelectCoordOperation",
    "SliceAlongPathOperation",
    "SortByOperation",
    "SortCoordOrderOperation",
    "SqueezeOperation",
    "SwapDimsOperation",
    "SymmetrizeNfoldOperation",
    "SymmetrizeOperation",
    "ThinOperation",
    "ToolProvenanceOperation",
    "ToolProvenanceSpec",
    "TransposeOperation",
    "compose_display_provenance",
    "compose_full_provenance",
    "decode_provenance_value",
    "direct_replay_input_name",
    "encode_provenance_value",
    "file_load",
    "full_data",
    "mark_promoted_1d_source",
    "operation_from_console_call",
    "operations_expression_code",
    "parse_tool_provenance_operation",
    "parse_tool_provenance_spec",
    "public_data",
    "rebase_default_replay_input",
    "rebase_script_input_node_uids",
    "replay_file_provenance",
    "replay_input_name",
    "replay_script_provenance",
    "require_live_source_spec",
    "script",
    "script_input_dependency_refs",
    "script_provenance_replayable",
    "selection",
    "to_replay_provenance_spec",
    "uses_default_replay_input",
]


import contextlib
import keyword
import typing
from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import pydantic
import xarray as xr

import erlab
from erlab.interactive.imagetool._provenance_framework import (
    _DATASET_MARKER,
    _FIT_DATASET_MARKER,
    _MAPPING_MARKER,
    _OPERATION_TYPES,
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    _SORT_COORD_ORDER_DERIVATION_LABEL,
    _TUPLE_MARKER,
    ConsoleCall,
    ConsoleOperationPattern,
    DerivationEntry,
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    NullableProvenanceHashableTuple,
    ProvenanceFloatMapping,
    ProvenanceFloatSequenceMapping,
    ProvenanceHashable,
    ProvenanceHashableMapping,
    ProvenanceHashablePair,
    ProvenanceHashableTuple,
    ProvenanceIntMapping,
    ProvenanceMapping,
    ReplayStage,
    ScriptInput,
    ScriptInputDependencyRef,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _as_script_replay_spec,
    _callable_paths,
    _coerce_float_mapping_field,
    _coerce_float_sequence,
    _console_mapping_values,
    _console_values_equal,
    _encode_provenance_hashable,
    _ensure_float_tuple,
    _expression_receiver_code,
    _format_derivation_value,
    _format_selection_expr,
    _format_selection_step,
    _is_identifier_string_mapping,
    _is_whole_array_rename_entry,
    _load_file_source_data,
    _normalize_provenance_hashable,
    _parse_replay_input,
    _provenance_numeric_array_code,
    _provenance_value_code,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    _replace_code_identifiers,
    _resolve_importable_callable,
    _scalar_coord_value,
    _select_replay_input,
    _simplify_display_code,
    _single_assign_coord,
    _statement_load_count,
    _statement_store_count,
    _validate_active_name,
    _validate_script_replay_code,
    compose_display_provenance,
    compose_full_provenance,
    decode_provenance_value,
    direct_replay_input_name,
    encode_provenance_value,
    file_load,
    full_data,
    mark_promoted_1d_source,
    operation_from_console_call,
    operations_expression_code,
    parse_tool_provenance_operation,
    parse_tool_provenance_spec,
    public_data,
    rebase_default_replay_input,
    rebase_script_input_node_uids,
    replay_file_provenance,
    replay_input_name,
    replay_script_provenance,
    require_live_source_spec,
    script,
    script_input_dependency_refs,
    script_provenance_replayable,
    selection,
    to_replay_provenance_spec,
    uses_default_replay_input,
)

# Keep these underscored framework helpers available on this module for existing
# package-internal code and targeted tests, without making them public API entries.
_PRIVATE_FRAMEWORK_REEXPORTS = (
    _MAPPING_MARKER,
    _OPERATION_TYPES,
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    _as_script_replay_spec,
    _callable_paths,
    _coerce_float_sequence,
    _encode_provenance_hashable,
    _expression_receiver_code,
    _format_selection_step,
    _is_whole_array_rename_entry,
    _load_file_source_data,
    _normalize_provenance_hashable,
    _parse_replay_input,
    _replace_code_identifiers,
    _resolve_importable_callable,
    _select_replay_input,
    _simplify_display_code,
    _statement_load_count,
    _statement_store_count,
    _validate_active_name,
    _validate_script_replay_code,
)


class ImageToolSelectionSourceBinding(pydantic.BaseModel):
    """Legacy ImageTool selection state used to build a source spec.

    Stores the parent dimension indices selected in an ImageTool plot. Current refresh
    paths materialize this once into explicit ``qsel`` or ``isel`` operations, then keep
    those operation arguments stable for later refreshes.
    """

    schema_version: typing.Literal[1] = 1
    kind: typing.Literal["imagetool_selection"] = "imagetool_selection"
    selection_mode: typing.Literal["qsel", "isel"] = "qsel"
    selection_indexers: ProvenanceMapping = pydantic.Field(default_factory=dict)
    selection_binned_dims: ProvenanceHashableTuple = ()
    crop_sel_indexers: ProvenanceMapping = pydantic.Field(default_factory=dict)
    crop_isel_indexers: ProvenanceMapping = pydantic.Field(default_factory=dict)
    transpose_dims: NullableProvenanceHashableTuple = None
    squeeze: bool = False

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def materialize(self, parent_data: xr.DataArray):
        """Build a source spec for the current parent data."""
        operations: list[ToolProvenanceOperation] = []
        selection_data = ToolProvenanceSpec._starting_data_for_kind(
            "selection", parent_data
        )
        selection_indexers = dict(self.selection_indexers)
        if selection_indexers:
            if self.selection_mode == "qsel":
                operation = QSelOperation(
                    kwargs=erlab.interactive.imagetool.slicer.qsel_args_from_indexers(
                        selection_data,
                        selection_indexers,
                        self.selection_binned_dims,
                    )
                )
            else:
                operation = IselOperation(kwargs=selection_indexers)
            operations.append(operation)

        crop_sel_indexers = dict(self.crop_sel_indexers)
        if crop_sel_indexers:
            crop_sel_kwargs: dict[Hashable, typing.Any] = {}
            for dim, selector in crop_sel_indexers.items():
                if dim not in selection_data.dims:
                    raise ValueError(f"Dimension `{dim}` not found in data")
                coord = selection_data[dim][selector].values
                if isinstance(selector, slice):
                    if coord.size == 0:
                        raise ValueError(f"Selection for dimension `{dim}` is empty")
                    crop_sel_kwargs[dim] = slice(
                        erlab.utils.misc._convert_to_native(coord[0]),
                        erlab.utils.misc._convert_to_native(coord[-1]),
                    )
                else:
                    crop_sel_kwargs[dim] = erlab.utils.misc._convert_to_native(coord)
            operations.append(SelOperation(kwargs=crop_sel_kwargs))

        crop_isel_indexers = dict(self.crop_isel_indexers)
        if crop_isel_indexers:
            operations.append(IselOperation(kwargs=crop_isel_indexers))

        operations.append(SortCoordOrderOperation())
        if self.transpose_dims is not None:
            operations.append(TransposeOperation(dims=self.transpose_dims))
        if self.squeeze:
            operations.append(SqueezeOperation())
        return ToolProvenanceSpec(kind="selection").append_operations(*operations)


class ScriptCodeOperation(ToolProvenanceOperation):
    op: typing.Literal["script_code"] = "script_code"
    label: str
    code: str | None
    copyable: bool = True

    live_applicable: typing.ClassVar[bool] = False

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        raise TypeError(
            "script_code operations do not support live updates from ImageTool data"
        )

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(self.label, self.code, self.copyable)


class QSelOperation(ToolProvenanceOperation):
    op: typing.Literal["qsel"] = "qsel"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            accessor_path=("qsel",),
            kwargs_field="kwargs",
            mapping_kwarg="indexers",
        ),
    )
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.qsel(self.decoded_kwargs)

    def derivation_label(self) -> str:
        return f"qsel({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return _format_selection_expr(input_name, "qsel", self.decoded_kwargs)


class IselOperation(ToolProvenanceOperation):
    op: typing.Literal["isel"] = "isel"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            dataarray_method="isel",
            kwargs_field="kwargs",
            mapping_kwarg="indexers",
            ignored_defaults={"drop": False, "missing_dims": "raise"},
        ),
    )
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.isel(self.decoded_kwargs)

    def derivation_label(self) -> str:
        return f"isel({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return _format_selection_expr(input_name, "isel", self.decoded_kwargs)


class SelOperation(ToolProvenanceOperation):
    op: typing.Literal["sel"] = "sel"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            dataarray_method="sel",
            kwargs_field="kwargs",
            mapping_kwarg="indexers",
            ignored_defaults={"method": None, "tolerance": None, "drop": False},
        ),
    )
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.sel(self.decoded_kwargs)

    def derivation_label(self) -> str:
        return f"sel({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return _format_selection_expr(input_name, "sel", self.decoded_kwargs)


class SortCoordOrderOperation(ToolProvenanceOperation):
    op: typing.Literal["sort_coord_order"] = "sort_coord_order"

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array.sort_coord_order(data, parent_data.coords.keys())

    def derivation_label(self) -> str:
        return _SORT_COORD_ORDER_DERIVATION_LABEL

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        source_name = input_name if source_name is None else source_name
        return (
            "erlab.utils.array.sort_coord_order("
            f"{input_name}, {source_name}.coords.keys())"
        )


class SortByOperation(ToolProvenanceOperation):
    op: typing.Literal["sortby"] = "sortby"
    batch_available: typing.ClassVar[bool] = True
    variables: ProvenanceHashableTuple
    ascending: bool = True

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "sortby"
            or call.accessor_path
            or len(call.args) > 1
        ):
            return None
        kwargs = dict(call.kwargs)
        ascending = kwargs.pop("ascending", True)
        if not isinstance(ascending, bool):
            return None
        if call.args:
            if "variables" in kwargs:
                return None
            variables = call.args[0]
        else:
            variables = kwargs.pop("variables", None)
        if kwargs or variables is None:
            return None
        variables = tuple(variables) if isinstance(variables, list) else (variables,)
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(variables=variables, ascending=ascending)
        return None

    @pydantic.field_validator("variables", mode="before")
    @classmethod
    def _validate_variables(cls, value: typing.Any) -> tuple[Hashable, ...]:
        value = decode_provenance_value(value)
        if callable(value) or isinstance(value, xr.DataArray):
            raise TypeError("sortby variables must be coordinate names")
        if isinstance(value, str) or not isinstance(value, Sequence):
            values = (value,)
        else:
            values = tuple(value)
        if not values:
            raise ValueError("sortby requires at least one variable")
        if any(callable(item) or isinstance(item, xr.DataArray) for item in values):
            raise TypeError("sortby variables must be coordinate names")
        return typing.cast("tuple[Hashable, ...]", values)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        variables: Hashable | list[Hashable] = (
            self.variables[0] if len(self.variables) == 1 else list(self.variables)
        )
        return data.sortby(variables, ascending=self.ascending)

    def derivation_label(self) -> str:
        label_kwargs = {"variables": self.variables, "ascending": self.ascending}
        return f"Sort By({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        variables: Hashable | list[Hashable] = (
            self.variables[0] if len(self.variables) == 1 else list(self.variables)
        )
        args = erlab.interactive.utils._parse_single_arg(variables)
        if self.ascending:
            return f"{input_name}.sortby({args})"
        return f"{input_name}.sortby({args}, ascending=False)"


class SelectCoordOperation(ToolProvenanceOperation):
    op: typing.Literal["select_coord"] = "select_coord"
    coord_name: ProvenanceHashable

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.coords[self.coord_name].copy(deep=False)

    def derivation_label(self) -> str:
        label_kwargs = {"coord_name": self.coord_name}
        return f"Select Coordinate({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        coord_name_code = erlab.interactive.utils._parse_single_arg(self.coord_name)
        return f"{input_name}.coords[{coord_name_code}]"


class TransposeOperation(ToolProvenanceOperation):
    op: typing.Literal["transpose"] = "transpose"
    dims: NullableProvenanceHashableTuple = None

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "transpose"
            or call.accessor_path
        ):
            return None
        kwargs = dict(call.kwargs)
        for key, default in {"transpose_coords": True, "missing_dims": "raise"}.items():
            if key not in kwargs:
                continue
            if not _console_values_equal(kwargs[key], default):
                return None
            kwargs.pop(key)
        if kwargs:
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(dims=tuple(call.args) or None)
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        if self.dims:
            return data.transpose(*self.dims)
        return data.transpose(*reversed(data.dims))

    def derivation_label(self) -> str:
        if self.dims:
            dims_tuple = tuple(self.dims)
            return f"transpose({_format_derivation_value(dims_tuple)})"
        return "transpose()"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        if self.dims:
            dims_tuple = tuple(self.dims)
            return (
                f"{input_name}.transpose(*"
                f"{erlab.interactive.utils._parse_single_arg(dims_tuple)})"
            )
        return f"{input_name}.transpose(*reversed({input_name}.dims))"


class SqueezeOperation(ToolProvenanceOperation):
    op: typing.Literal["squeeze"] = "squeeze"

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "squeeze"
            or call.accessor_path
            or call.args
        ):
            return None
        for key, default in {"dim": None, "axis": None, "drop": False}.items():
            if key not in call.kwargs:
                continue
            if not _console_values_equal(call.kwargs[key], default):
                return None
        if set(call.kwargs).difference({"dim", "axis", "drop"}):
            return None
        return cls()

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.squeeze()

    def derivation_label(self) -> str:
        return "squeeze()"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return f"{input_name}.squeeze()"


class RenameOperation(ToolProvenanceOperation):
    op: typing.Literal["rename"] = "rename"
    name: str

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "rename"
            or call.accessor_path
            or call.kwargs
            or len(call.args) != 1
            or isinstance(call.args[0], Mapping)
        ):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(name=call.args[0])
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.rename(self.name)

    def derivation_label(self) -> str:
        return f"rename({_format_derivation_value(self.name)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        name_code = erlab.interactive.utils._parse_single_arg(self.name)
        return f"{input_name}.rename({name_code})"


class RestoreNonuniformDimsOperation(ToolProvenanceOperation):
    op: typing.Literal["restore_nonuniform_dims"] = "restore_nonuniform_dims"
    batch_available: typing.ClassVar[bool] = True

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.interactive.imagetool.slicer.restore_nonuniform_dims(data)

    def derivation_label(self) -> str:
        return "Restore nonuniform dimensions"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return (
            f"erlab.interactive.imagetool.slicer.restore_nonuniform_dims({input_name})"
        )


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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        edge_fit_code = _provenance_value_code(self.edge_fit)
        edge_fit_expr = (
            "erlab.interactive.imagetool.provenance.decode_provenance_value("
            f"{edge_fit_code})"
        )
        return erlab.interactive.utils.generate_code(
            erlab.analysis.gold.correct_with_edge,
            [f"|{input_name}|", f"|{edge_fit_expr}|"],
            {"shift_coords": self.shift_coords},
            module="era.gold",
            name="correct_with_edge",
        )


class SwapDimsOperation(ToolProvenanceOperation):
    op: typing.Literal["swap_dims"] = "swap_dims"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            dataarray_method="swap_dims",
            kwargs_field="mapping",
            mapping_kwarg="dims_dict",
        ),
    )
    mapping: ProvenanceHashableMapping = pydantic.Field(default_factory=dict)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.swap_dims(self.mapping)

    def derivation_label(self) -> str:
        return f"Swap Dimensions({_format_derivation_value(self.mapping)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return (
            f"{input_name}.swap_dims("
            f"{erlab.interactive.utils.format_call_kwargs(self.mapping)})"
        )


class RenameDimsCoordsOperation(ToolProvenanceOperation):
    op: typing.Literal["rename_dims_coords"] = "rename_dims_coords"
    batch_available: typing.ClassVar[bool] = True
    mapping: ProvenanceHashableMapping = pydantic.Field(default_factory=dict)

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "rename"
            or call.accessor_path
            or len(call.args) > 1
        ):
            return None
        mapping = _console_mapping_values(
            call.args, call.kwargs, mapping_kwargs=("new_name_or_name_dict",)
        )
        if mapping is None:
            return None
        if not mapping:
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(mapping=mapping)
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.rename(self.mapping)

    def derivation_label(self) -> str:
        return f"Rename({_format_derivation_value(self.mapping)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return (
            f"{input_name}.rename("
            f"{erlab.interactive.utils.format_call_kwargs(self.mapping)})"
        )


class AffineCoordOperation(ToolProvenanceOperation):
    op: typing.Literal["affine_coord"] = "affine_coord"
    batch_available: typing.ClassVar[bool] = True
    coord_name: str
    scale: float
    offset: float

    @pydantic.field_validator("scale", "offset")
    @classmethod
    def _validate_finite_affine_value(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("affine coordinate scale and offset must be finite")
        return value

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        coord = data.coords[self.coord_name]
        return erlab.utils.array.sort_coord_order(
            data.assign_coords(
                {
                    self.coord_name: coord.copy(
                        data=self.scale * coord.values + self.offset
                    )
                }
            ),
            keys=data.coords.keys(),
            dims_first=False,
        )

    def derivation_label(self) -> str:
        label_kwargs = {
            "coord_name": self.coord_name,
            "scale": self.scale,
            "offset": self.offset,
        }
        return f"Scale/Offset Coordinate({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        coord_name_code = repr(self.coord_name)
        coord_values_code = f"{input_name}[{coord_name_code}].values"
        scale = float(self.scale)
        offset = float(self.offset)
        data_code = coord_values_code
        if scale != 1.0:
            scale_code = erlab.interactive.utils._parse_single_arg(scale)
            data_code = f"{scale_code} * {data_code}"
        if offset > 0.0:
            offset_code = erlab.interactive.utils._parse_single_arg(offset)
            data_code = f"{data_code} + {offset_code}"
        elif offset < 0.0:
            offset_code = erlab.interactive.utils._parse_single_arg(abs(offset))
            data_code = f"{data_code} - {offset_code}"
        return (
            f"{input_name}.assign_coords({{{coord_name_code}: "
            f"{input_name}[{coord_name_code}].copy(data={data_code})}})"
        )


class AssignCoordsOperation(ToolProvenanceOperation):
    op: typing.Literal["assign_coords"] = "assign_coords"
    batch_available: typing.ClassVar[bool] = True
    coord_name: str
    values: typing.Any

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        coord = _single_assign_coord(call)
        if coord is None or call.receiver_data is None:
            return None
        coord_name, values = coord
        if (
            not isinstance(coord_name, str)
            or coord_name not in call.receiver_data.coords
        ):
            return None
        if (
            isinstance(values, Sequence)
            and not isinstance(values, str)
            and len(values) == 2
            and isinstance(values[0], Hashable)
        ):
            return None
        if _scalar_coord_value(values):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(coord_name=coord_name, values=values)
        return None

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

    def derivation_label(self) -> str:
        label_kwargs = {
            "coord_name": self.coord_name,
            "values": self.values,
        }
        return f"Assign Coordinates({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        coord_name_code = repr(self.coord_name)
        values = self.decoded_values
        values_code = _provenance_numeric_array_code(values)
        return (
            f"{input_name}.assign_coords({{{coord_name_code}: "
            f"{input_name}[{coord_name_code}].copy(data={values_code})}})"
        )


class AssignScalarCoordOperation(ToolProvenanceOperation):
    op: typing.Literal["assign_scalar_coord"] = "assign_scalar_coord"
    batch_available: typing.ClassVar[bool] = True
    coord_name: ProvenanceHashable
    value: typing.Any

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        coord = _single_assign_coord(call)
        if coord is None:
            return None
        coord_name, value = coord
        if not _scalar_coord_value(value):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(coord_name=coord_name, value=value)
        return None

    @pydantic.field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: typing.Any) -> typing.Any:
        encoded = encode_provenance_value(value)
        decoded = decode_provenance_value(encoded)
        if isinstance(decoded, (Mapping, Sequence)) and not isinstance(decoded, str):
            raise TypeError("scalar coordinate value must be scalar")
        return encoded

    @property
    def decoded_value(self) -> typing.Any:
        return self._decode_stored_field(self.value)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array.sort_coord_order(
            data.assign_coords({self.coord_name: self.decoded_value}),
            keys=data.coords.keys(),
            dims_first=False,
        )

    def derivation_label(self) -> str:
        label_kwargs = {
            "coord_name": self.coord_name,
            "value": self.value,
        }
        return f"Assign Scalar Coordinate({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        coord_name_code = erlab.interactive.utils._parse_single_arg(self.coord_name)
        value_code = _provenance_value_code(self.decoded_value)
        return f"{input_name}.assign_coords({{{coord_name_code}: {value_code}}})"


class AssignCoord1DOperation(ToolProvenanceOperation):
    op: typing.Literal["assign_coord_1d"] = "assign_coord_1d"
    batch_available: typing.ClassVar[bool] = True
    coord_name: ProvenanceHashable
    dim: ProvenanceHashable
    values: typing.Any

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        coord = _single_assign_coord(call)
        if coord is None:
            return None
        coord_name, value = coord
        if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
            return None
        dim, values = value
        if not isinstance(dim, Hashable) or (
            isinstance(dim, Sequence) and not isinstance(dim, str)
        ):
            return None
        try:
            if np.asarray(values).ndim != 1:
                return None
        except (TypeError, ValueError):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(coord_name=coord_name, dim=dim, values=values)
        return None

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: typing.Any) -> typing.Any:
        return cls._validate_encoded_field(
            value,
            error="1D coordinate values must be array-like",
            predicate=lambda encoded: (
                isinstance(encoded, list)
                or (isinstance(encoded, Mapping) and _TUPLE_MARKER in encoded)
            ),
        )

    @property
    def decoded_values(self) -> np.ndarray:
        return np.asarray(self._decode_stored_field(self.values))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        values = self.decoded_values
        if values.ndim != 1:
            raise ValueError("1D coordinate values must be one-dimensional.")
        if self.dim not in data.sizes:
            raise ValueError(f"Dimension {self.dim!r} is not present in the data.")
        if values.size != data.sizes[self.dim]:
            raise ValueError(
                f"Coordinate {self.coord_name!r} has length {values.size}, expected "
                f"{data.sizes[self.dim]} for dimension {self.dim!r}."
            )
        return erlab.utils.array.sort_coord_order(
            data.assign_coords({self.coord_name: (self.dim, values)}),
            keys=data.coords.keys(),
            dims_first=False,
        )

    def derivation_label(self) -> str:
        label_kwargs = {
            "coord_name": self.coord_name,
            "dim": self.dim,
            "values": self.values,
        }
        return f"Assign 1D Coordinate({_format_derivation_value(label_kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        coord_name_code = erlab.interactive.utils._parse_single_arg(self.coord_name)
        dim_code = erlab.interactive.utils._parse_single_arg(self.dim)
        values_code = _provenance_numeric_array_code(self.decoded_values)
        return (
            f"{input_name}.assign_coords({{{coord_name_code}: "
            f"({dim_code}, {values_code})}})"
        )


class AssignAttrsOperation(ToolProvenanceOperation):
    op: typing.Literal["assign_attrs"] = "assign_attrs"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(dataarray_method="assign_attrs", kwargs_field="attrs"),
    )
    attrs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.assign_attrs(self.attrs)

    def derivation_label(self) -> str:
        return f"Assign Attributes({_format_derivation_value(self.attrs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        attrs_code = _provenance_value_code(self.attrs)
        return f"{input_name}.assign_attrs({attrs_code})"


class SliceAlongPathOperation(ToolProvenanceOperation):
    op: typing.Literal["slice_along_path"] = "slice_along_path"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.interpolate.slice_along_path",
            fields=("vertices", "step_size", "dim_name"),
            defaults={"dim_name": "path"},
            ignored_defaults={"interp_kwargs": None},
        ),
    )
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

    def derivation_label(self) -> str:
        return f"Slice Along ROI Path({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.interpolate.slice_along_path,
            [f"|{input_name}|"],
            self.kwargs,
            module="era.interpolate",
        )


class MaskWithPolygonOperation(ToolProvenanceOperation):
    op: typing.Literal["mask_with_polygon"] = "mask_with_polygon"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.mask.mask_with_polygon",
            fields=("vertices", "dims"),
            defaults={"invert": False, "drop": False},
        ),
    )
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

    def derivation_label(self) -> str:
        return f"Mask with ROI({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.mask.mask_with_polygon,
            [f"|{input_name}|"],
            self.kwargs,
            module="era.mask",
        )
