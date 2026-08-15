"""Core-owned provenance for user extension routines."""

from __future__ import annotations

import typing

import pydantic

import erlab
from erlab.extensions._models import (
    _require_finite_parameter_values,
    _validate_source_hash,
)
from erlab.interactive.imagetool._provenance._code import _provenance_value_code
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    ToolProvenanceOperation,
)

if typing.TYPE_CHECKING:
    from collections.abc import Collection

    import xarray as xr


class ExtensionRoutineOperation(ToolProvenanceOperation):
    """Apply a user extension routine from an identified source snapshot."""

    op: typing.Literal["extension_routine"] = "extension_routine"
    extension_id: str
    source_hash: str
    routine_id: str
    extension_name: str
    routine_name: str
    parameters: dict[str, bool | int | float | str | None]

    @pydantic.field_validator("source_hash")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return _validate_source_hash(value)

    @pydantic.field_validator("parameters")
    @classmethod
    def _validate_parameters(
        cls, value: dict[str, bool | int | float | str | None]
    ) -> dict[str, bool | int | float | str | None]:
        _require_finite_parameter_values(value)
        return value

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.extensions.run_routine(
            data,
            extension_id=self.extension_id,
            source_hash=self.source_hash,
            routine_id=self.routine_id,
            parameters=self.parameters,
        )

    def derivation_label(self) -> str:
        return f"Run {self.routine_name} ({self.extension_name})"

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(self.derivation_label(), None, False)

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
        raise NotImplementedError

    def _statement_replay_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
        reserved_names: Collection[str] = (),
    ) -> str:
        del input_name, output_name, source_name, reserved_names
        raise NotImplementedError

    def _bound_script_statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        module_name: str,
        function_name: str,
    ) -> str:
        """Emit only the call for a script binding owned by the graph compiler."""
        parameters = tuple(
            f"    {name}={_provenance_value_code(value)},"
            for name, value in self.parameters.items()
        )
        return "\n".join(
            (
                f"{output_name} = {module_name}.{function_name}(",
                f"    {input_name},",
                *parameters,
                ")",
            )
        )
