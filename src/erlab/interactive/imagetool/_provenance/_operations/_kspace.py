"""Momentum-space and path-based provenance operations."""

from __future__ import annotations

import typing
from collections.abc import Mapping

import numpy as np
import pydantic

import erlab
from erlab.interactive.imagetool._provenance._code import _TUPLE_MARKER
from erlab.interactive.imagetool._provenance._model import (
    ConsoleOperationPattern,
    ProvenanceFloatSequenceMapping,
    ProvenanceHashableTuple,
    ToolProvenanceOperation,
    _format_derivation_value,
)

if typing.TYPE_CHECKING:
    import xarray as xr


class KspaceConfigurationOperation(ToolProvenanceOperation):
    op: typing.Literal["kspace_configuration"] = "kspace_configuration"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            accessor_path=("kspace", "as_configuration"),
            fields=("configuration",),
        ),
    )
    configuration: int

    @pydantic.field_validator("configuration")
    @classmethod
    def _validate_configuration(cls, value: int) -> int:
        return int(erlab.constants.AxesConfiguration(int(value)))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.kspace.as_configuration(self.configuration)

    def derivation_label(self) -> str:
        configuration = erlab.constants.AxesConfiguration(self.configuration)
        return f"Set kspace configuration({int(configuration)} {configuration.name})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return f"{input_name}.kspace.as_configuration({self.configuration})"


class _MutatingKspaceOperation(ToolProvenanceOperation):
    batch_available: typing.ClassVar[bool] = True
    console_applies_to_receiver: typing.ClassVar[bool] = True
    statement_mutates_input: typing.ClassVar[bool] = True

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        out = data.copy(deep=False)
        self._apply_kspace_statement(out)
        return out

    def statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
    ) -> str:
        if input_name == output_name:
            return self._kspace_statement_code(output_name)
        return "\n".join(
            (
                f"{output_name} = {input_name}.copy(deep=False)",
                self._kspace_statement_code(output_name),
            )
        )

    def _apply_kspace_statement(self, data: xr.DataArray) -> None:
        raise NotImplementedError

    def _kspace_statement_code(self, output_name: str) -> str:
        raise NotImplementedError


class KspaceWorkFunctionOperation(_MutatingKspaceOperation):
    op: typing.Literal["kspace_work_function"] = "kspace_work_function"
    work_function: float

    def derivation_label(self) -> str:
        return f"Set work function({self.work_function:g} eV)"

    def _apply_kspace_statement(self, data: xr.DataArray) -> None:
        data.kspace.work_function = self.work_function

    def _kspace_statement_code(self, output_name: str) -> str:
        return f"{output_name}.kspace.work_function = {self.work_function!r}"


class KspaceInnerPotentialOperation(_MutatingKspaceOperation):
    op: typing.Literal["kspace_inner_potential"] = "kspace_inner_potential"
    inner_potential: float

    def derivation_label(self) -> str:
        return f"Set inner potential({self.inner_potential:g} eV)"

    def _apply_kspace_statement(self, data: xr.DataArray) -> None:
        data.kspace.inner_potential = self.inner_potential

    def _kspace_statement_code(self, output_name: str) -> str:
        return f"{output_name}.kspace.inner_potential = {self.inner_potential!r}"


class KspaceSetNormalOperation(_MutatingKspaceOperation):
    op: typing.Literal["kspace_set_normal"] = "kspace_set_normal"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            accessor_path=("kspace", "set_normal"),
            fields=("alpha", "beta", "alpha_scale", "beta_scale"),
            defaults={"delta": None, "alpha_scale": None, "beta_scale": None},
        ),
    )
    alpha: float
    beta: float
    delta: float | None = None
    alpha_scale: float | None = None
    beta_scale: float | None = None

    @property
    def kwargs(self) -> dict[str, float]:
        kwargs = {"alpha": self.alpha, "beta": self.beta}
        if self.delta is not None:
            kwargs["delta"] = self.delta
        if self.alpha_scale is not None and not np.isclose(self.alpha_scale, 1.0):
            kwargs["alpha_scale"] = self.alpha_scale
        if self.beta_scale is not None and not np.isclose(self.beta_scale, 1.0):
            kwargs["beta_scale"] = self.beta_scale
        return kwargs

    def derivation_label(self) -> str:
        return f"Set normal emission({_format_derivation_value(self.kwargs)})"

    def _apply_kspace_statement(self, data: xr.DataArray) -> None:
        data.kspace.set_normal(**self.kwargs)

    def _kspace_statement_code(self, output_name: str) -> str:
        return (
            f"{output_name}.kspace.set_normal("
            f"{erlab.interactive.utils.format_call_kwargs(self.kwargs)})"
        )


class KspaceConvertOperation(ToolProvenanceOperation):
    op: typing.Literal["kspace_convert"] = "kspace_convert"
    batch_available: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            accessor_path=("kspace", "convert"),
            fields=("bounds", "resolution"),
            defaults={
                "bounds": None,
                "resolution": None,
                "method": "linear",
                "silent": True,
            },
        ),
    )
    bounds: dict[str, tuple[float, float]] | None = None
    resolution: dict[str, float] | None = None
    method: str = "linear"
    silent: bool = True

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        kwargs: dict[str, typing.Any] = {}
        if self.bounds is not None:
            kwargs["bounds"] = self.bounds
        if self.resolution is not None:
            kwargs["resolution"] = self.resolution
        if self.method != "linear":
            kwargs["method"] = self.method
        if not self.silent:
            kwargs["silent"] = self.silent
        return kwargs

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.kspace.convert(
            bounds=self.bounds,
            resolution=self.resolution,
            method=self.method,
            silent=self.silent,
        )

    def derivation_label(self) -> str:
        return f"Convert to momentum({_format_derivation_value(self.kwargs)})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        kwargs_code = erlab.interactive.utils.format_call_kwargs(self.kwargs)
        return f"{input_name}.kspace.convert({kwargs_code})"


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
