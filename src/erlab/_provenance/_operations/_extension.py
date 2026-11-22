"""Core-owned provenance for user extension routines."""

from __future__ import annotations

import typing

import pydantic

import erlab
from erlab._provenance._calls import Call, InputRef, WorkerCallOperation
from erlab._provenance._code import _provenance_value_code
from erlab._provenance._model import DerivationEntry
from erlab.extensions._models import (
    _require_finite_parameter_values,
    _script_name_key,
    _validate_source_hash,
)


class ExtensionRoutineOperation(WorkerCallOperation):
    """Apply a user extension routine from an identified source snapshot."""

    op: typing.Literal["extension_routine"] = "extension_routine"
    script_name: str
    source_hash: str
    routine_id: str
    routine_name: str
    parameters: dict[str, bool | int | float | str | None]

    @pydantic.field_validator("script_name")
    @classmethod
    def _valid_script_name(cls, value: str) -> str:
        _script_name_key(value)
        return value

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

    def invocation(self, *, script: str | None = None) -> Call:
        kwargs = {
            "routine_id": self.routine_id,
            "source_hash": self.source_hash,
            "parameters": self.parameters,
        }
        if script is None:
            kwargs["registered_script"] = self.script_name
        else:
            kwargs["script"] = script
        if self.execution_options is not None:
            kwargs["execution_options"] = self.execution_options
        return Call.function(
            erlab.extensions.run_routine, args=(InputRef("data"),), kwargs=kwargs
        )

    def derivation_label(self) -> str:
        return f"Run {self.routine_name} ({self.script_name})"

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(self.derivation_label(), None, False)

    def _bound_script_statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        module_name: str,
        function_name: str,
    ) -> str:
        """Emit only the call for a script binding owned by the graph compiler."""
        if not self.parameters:
            return f"{output_name} = {module_name}.{function_name}({input_name})"
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
