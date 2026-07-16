"""Selection and dimension-transform provenance operations."""

from __future__ import annotations

import contextlib
import typing
from collections.abc import Hashable, Mapping, Sequence

import pydantic
import xarray as xr

import erlab
from erlab.interactive.imagetool._provenance._code import (
    _dynamic_nonuniform_restore_replay_code,
    _format_selection_expr,
    _known_nonuniform_restore_statement_code,
)
from erlab.interactive.imagetool._provenance._model import (
    _SORT_COORD_ORDER_DERIVATION_LABEL,
    ConsoleCall,
    ConsoleOperationPattern,
    NullableProvenanceHashableTuple,
    ProvenanceHashable,
    ProvenanceHashableMapping,
    ProvenanceHashableTuple,
    ProvenanceMapping,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _console_values_equal,
    _format_derivation_value,
    decode_provenance_value,
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
        return data.transpose()

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
        return f"{input_name}.transpose()"


class SqueezeOperation(ToolProvenanceOperation):
    op: typing.Literal["squeeze"] = "squeeze"
    batch_available: typing.ClassVar[bool] = True
    dims: NullableProvenanceHashableTuple = None
    drop: bool = False

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "squeeze"
            or call.accessor_path
            or len(call.args) > 1
        ):
            return None
        kwargs = dict(call.kwargs)
        axis = kwargs.pop("axis", None)
        if not _console_values_equal(axis, None):
            return None
        drop = kwargs.pop("drop", False)
        if not isinstance(drop, bool):
            return None
        if call.args:
            if "dim" in kwargs:
                return None
            dim = call.args[0]
        else:
            dim = kwargs.pop("dim", None)
        if kwargs:
            return None
        if dim is None:
            return cls(drop=drop)
        dims = (dim,) if isinstance(dim, str) or not isinstance(dim, Sequence) else dim
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(
                dims=typing.cast("tuple[Hashable, ...]", tuple(dims)),
                drop=drop,
            )
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        if self.dims is None:
            return data.squeeze(drop=self.drop)
        return data.squeeze(dim=tuple(self.dims), drop=self.drop)

    def derivation_label(self) -> str:
        label_kwargs: dict[str, typing.Any] = {}
        if self.dims is not None:
            label_kwargs["dim"] = tuple(self.dims)
        if self.drop:
            label_kwargs["drop"] = self.drop
        if label_kwargs:
            return f"squeeze({_format_derivation_value(label_kwargs)})"
        return "squeeze()"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        kwargs: dict[str, typing.Any] = {}
        if self.dims is not None:
            kwargs["dim"] = tuple(self.dims)
        if self.drop:
            kwargs["drop"] = self.drop
        if not kwargs:
            return f"{input_name}.squeeze()"
        return f"{input_name}.squeeze({erlab.interactive.utils.format_kwargs(kwargs)})"


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
    dimension_mapping: ProvenanceHashableMapping | None = None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array._restore_nonuniform_dims(data, self.dimension_mapping)

    def derivation_label(self) -> str:
        return "Restore nonuniform dimensions"

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
        if self.dimension_mapping is not None:
            return _known_nonuniform_restore_statement_code(
                input_name,
                output_name=output_name,
                dimension_mapping=self.dimension_mapping,
            )
        return _dynamic_nonuniform_restore_replay_code(
            input_name,
            output_name=output_name,
        )
