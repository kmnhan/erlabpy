"""Persistent schema models for ImageTool Manager extensions."""

from __future__ import annotations

import pathlib
import typing

import pydantic

from erlab.extensions import LoaderDescriptor, RoutineDescriptor  # noqa: TC001
from erlab.extensions._models import _script_name_key, _validate_source_hash


class _ScriptRecord(pydantic.BaseModel):
    """Persistent state for one registered local Python script."""

    script_name: str
    source_path: str
    source_hash: str
    source_modified_at: str
    registered_at: str
    approved: bool = False
    enabled: bool = False
    embed_policy: typing.Literal["referenced", "always", "never"] = "referenced"
    routines: tuple[RoutineDescriptor, ...] = ()
    loaders: tuple[LoaderDescriptor, ...] = ()
    record_generation: int = 0

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("script_name")
    @classmethod
    def _valid_script_name(cls, value: str) -> str:
        _script_name_key(value)
        return value

    @pydantic.field_validator("source_path")
    @classmethod
    def _valid_source_path(cls, value: str) -> str:
        path = pathlib.Path(value)
        if not path.is_absolute():
            raise ValueError("registered script path must be absolute")
        return value

    @pydantic.field_validator("source_hash")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return _validate_source_hash(value)

    @pydantic.model_validator(mode="after")
    def _validate_identity_and_state(self) -> typing.Self:
        if pathlib.Path(self.source_path).name != self.script_name:
            raise ValueError("registered path basename must match the script name")
        if self.enabled and not self.approved:
            raise ValueError("an enabled extension script must be approved")
        return self


def _script_loader_name_filters(record: _ScriptRecord) -> tuple[str, ...]:
    """Return the file-dialog filters owned by one validated script."""
    name_filters: list[str] = []
    for descriptor in record.loaders:
        patterns = " ".join(f"*{suffix}" for suffix in descriptor.extensions) or "*"
        name_filters.append(f"{descriptor.name} ({patterns})")
    return tuple(name_filters)


class _ExtensionCatalogModel(pydantic.BaseModel):
    """Validated catalog of registered local scripts."""

    schema_version: typing.Literal[1] = 1
    generation: int = 0
    extensions: dict[str, _ScriptRecord] = pydantic.Field(default_factory=dict)
    routine_favorites: tuple[tuple[str, str], ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def _validate_script_identity(self) -> typing.Self:
        for script_key, record in self.extensions.items():
            if script_key != _script_name_key(record.script_name):
                raise ValueError("extension key does not match the script name")
        if len(set(self.routine_favorites)) != len(self.routine_favorites):
            raise ValueError("routine favorites must be unique")
        for script_key, _routine_id in self.routine_favorites:
            if script_key != _script_name_key(script_key):
                raise ValueError("routine favorite must use a normalized script key")
            if script_key not in self.extensions:
                raise ValueError("routine favorite refers to an unknown script")
        return self


class _WorkspaceScriptRequirement(pydantic.BaseModel):
    """Exact local script dependency persisted in a workspace."""

    script_name: str
    capability_id: str
    capability_name: str
    capability_kind: typing.Literal["routine", "loader"]
    source_hash: str
    extension_api_version: int
    referencing_nodes: tuple[str, ...] = ()
    file_sources: tuple[str, ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("script_name")
    @classmethod
    def _valid_script_name(cls, value: str) -> str:
        _script_name_key(value)
        return value

    @pydantic.field_validator("capability_id", "capability_name")
    @classmethod
    def _nonempty_capability_value(cls, value: str) -> str:
        if not value:
            raise ValueError("capability values must not be empty")
        return value

    @pydantic.field_validator("source_hash")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return _validate_source_hash(value)


_WorkspaceRequirementState = typing.Literal[
    "ready",
    "approval-required",
    "disabled",
    "missing",
    "hash-mismatch",
    "unsupported-api",
    "validation-failed",
]


class _ResolvedWorkspaceRequirement(pydantic.BaseModel):
    requirement: _WorkspaceScriptRequirement
    state: _WorkspaceRequirementState
    detail: str = ""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
