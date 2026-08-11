"""Core-owned provenance for user extension routines."""

from __future__ import annotations

import typing

import erlab
from erlab.interactive.imagetool._provenance._code import _provenance_value_code
from erlab.interactive.imagetool._provenance._model import ToolProvenanceOperation

if typing.TYPE_CHECKING:
    import xarray as xr


class ExtensionRoutineOperation(ToolProvenanceOperation):
    """Apply one exact revision of a user extension routine."""

    op: typing.Literal["extension_routine"] = "extension_routine"
    extension_id: str
    revision_hash: str
    routine_id: str
    extension_name: str
    routine_name: str
    parameters: dict[str, bool | int | float | str | None]
    public_call: typing.Literal["erlab.extensions.run_routine"] = (
        "erlab.extensions.run_routine"
    )

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.extensions.run_routine(
            data,
            extension_id=self.extension_id,
            revision=self.revision_hash,
            routine_id=self.routine_id,
            parameters=self.parameters,
        )

    def derivation_label(self) -> str:
        return f"Run {self.routine_name} ({self.extension_name})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return "\n".join(
            (
                "erlab.extensions.run_routine(",
                f"    {input_name},",
                f"    extension_id={self.extension_id!r},",
                f"    revision={self.revision_hash!r},",
                f"    routine_id={self.routine_id!r},",
                f"    parameters={_provenance_value_code(self.parameters)},",
                ")",
            )
        )
