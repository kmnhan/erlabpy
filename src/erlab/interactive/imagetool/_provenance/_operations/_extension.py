"""Core-owned provenance for user extension routines."""

from __future__ import annotations

import keyword
import pathlib
import re
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
        try:
            code = self.replay_code(
                "derived", output_name="derived", source_name="data"
            )
        except NotImplementedError:
            code = None
        return DerivationEntry(self.derivation_label(), code, code is not None)

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
        return self._extension_statement_code(
            input_name, output_name=output_name, reserved_names=()
        )

    def _statement_replay_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
        reserved_names: Collection[str] = (),
    ) -> str:
        return self._extension_statement_code(
            input_name,
            output_name=output_name,
            reserved_names=reserved_names,
        )

    def _extension_statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        reserved_names: Collection[str],
    ) -> str:
        parameters = tuple(
            f"    {name}={_provenance_value_code(value)},"
            for name, value in self.parameters.items()
        )
        unavailable = {input_name, output_name, "load_script", *reserved_names}
        call_input_name = input_name
        prelude: list[str] = []
        if input_name == "load_script":
            call_input_name = "data"
            suffix = 2
            while call_input_name in unavailable:
                call_input_name = f"data_{suffix}"
                suffix += 1
            unavailable.add(call_input_name)
            prelude.extend((f"{call_input_name} = {input_name}", ""))
        from erlab.extensions._api import _resolved_script_capability_reference

        try:
            resolved_path, function_name = _resolved_script_capability_reference(
                self.extension_id,
                "routine",
                self.routine_id,
            )
        except erlab.extensions.ExtensionNotFoundError as error:
            raise NotImplementedError from error
        source_path = pathlib.Path(resolved_path)
        module_base = re.sub(r"\W", "_", source_path.stem)
        if not module_base.isidentifier() or keyword.iskeyword(module_base):
            module_base = "extension_script"
        module_name = module_base
        suffix = 2
        while module_name in unavailable:
            module_name = f"{module_base}_{suffix}"
            suffix += 1
        call_target = f"{module_name}.{function_name}"
        prelude.extend(
            (
                "from erlab.extensions import load_script",
                "",
                f"{module_name} = load_script({str(source_path)!r})",
            )
        )
        return "\n".join(
            (
                *prelude,
                "",
                f"{output_name} = {call_target}(",
                f"    {call_input_name},",
                *parameters,
                ")",
            )
        )
