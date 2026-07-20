"""Coordinate and metadata provenance operations."""

from __future__ import annotations

import contextlib
import typing
from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import pydantic

import erlab
from erlab.interactive.imagetool._provenance._code import (
    _TUPLE_MARKER,
    _provenance_numeric_array_code,
    _provenance_value_code,
)
from erlab.interactive.imagetool._provenance._model import (
    ConsoleCall,
    ConsoleOperationPattern,
    ProvenanceHashable,
    ProvenanceHashableMapping,
    ProvenanceMapping,
    ToolProvenanceOperation,
    _console_mapping_values,
    _format_derivation_value,
    _scalar_coord_value,
    _single_assign_coord,
    decode_provenance_value,
    encode_provenance_value,
)

if typing.TYPE_CHECKING:
    import xarray as xr


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

    def apply(self, data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray) -> xr.DataArray:
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

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return data.assign_attrs(self.attrs)

    def derivation_label(self) -> str:
        return f"Assign Attributes({_format_derivation_value(self.attrs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        attrs_code = _provenance_value_code(self.attrs)
        return f"{input_name}.assign_attrs({attrs_code})"
