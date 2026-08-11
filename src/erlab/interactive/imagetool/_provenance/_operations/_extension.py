"""Core-owned provenance for user extension routines."""

from __future__ import annotations

import os
import typing

import pydantic

import erlab
from erlab.extensions._models import _require_finite_parameter_values
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
    source_type: typing.Literal["script", "environment-package"]
    function_name: str
    source_path: str | None
    entry_point_group: str | None
    entry_point_name: str | None
    parameters: dict[str, bool | int | float | str | None]

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
            revision=self.revision_hash,
            routine_id=self.routine_id,
            parameters=self.parameters,
        )

    def derivation_label(self) -> str:
        return f"Run {self.routine_name} ({self.extension_name})"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        parameters = tuple(
            f"    {name}={_provenance_value_code(value)},"
            for name, value in self.parameters.items()
        )
        if self.source_type == "script":
            from erlab.extensions._api import _resolved_revision

            source_path = self.source_path
            try:
                source_path = os.fspath(
                    _resolved_revision(self.extension_id, self.revision_hash)
                )
            except erlab.extensions.ExtensionNotFoundError:
                if source_path is None:
                    raise
            loader = (
                "erlab.extensions.load_script(\n"
                f"    {source_path!r},\n"
                f"    expected_revision={self.revision_hash!r},\n"
                ")"
            )
        else:
            if self.entry_point_group is None or self.entry_point_name is None:
                raise ValueError("Environment extension provenance is incomplete")
            loader = (
                "erlab.extensions.load_entry_point(\n"
                f"    {self.entry_point_group!r},\n"
                f"    {self.entry_point_name!r},\n"
                f"    expected_revision={self.revision_hash!r},\n"
                ")"
            )
        return "\n".join(
            (
                f"{loader}.{self.function_name}(",
                f"    {input_name},",
                *parameters,
                ")",
            )
        )
