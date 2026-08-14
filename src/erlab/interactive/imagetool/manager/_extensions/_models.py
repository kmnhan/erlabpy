"""Persistent schema models for ImageTool Manager extensions."""

from __future__ import annotations

import typing

import pydantic

from erlab.extensions import LoaderDescriptor, RoutineDescriptor  # noqa: TC001
from erlab.extensions._models import _validate_source_hash


class _ExtensionSource(pydantic.BaseModel):
    """Identity and validation result for one registered script."""

    source_hash: str
    object_name: str
    source_path: str | None = None
    source_modified_at: str | None = None
    registered_at: str
    approved: bool = False
    routines: tuple[RoutineDescriptor, ...] = ()
    loaders: tuple[LoaderDescriptor, ...] = ()
    import_error: str | None = None

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("source_hash")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return _validate_source_hash(value)


def _source_loader_name_filters(source: _ExtensionSource) -> tuple[str, ...]:
    """Return the file-dialog filters owned by one validated source."""
    name_filters: list[str] = []
    for descriptor in source.loaders:
        patterns = " ".join(f"*{suffix}" for suffix in descriptor.extensions) or "*"
        name_filters.append(f"{descriptor.name} ({patterns})")
    return tuple(name_filters)


class _ExtensionRecord(pydantic.BaseModel):
    """Persistent record for one registered script."""

    id: str
    name: str
    enabled: bool = False
    embed_policy: typing.Literal["referenced", "always", "never"] = "referenced"
    source: _ExtensionSource
    record_generation: int = 0

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def _validate_source_identity(self) -> typing.Self:
        """Require the managed object name to match its content hash."""
        if self.source.object_name != f"{self.source.source_hash}.py":
            raise ValueError("a script source object name must match its source hash")
        if self.enabled and not self.source.approved:
            raise ValueError("an enabled extension source must be approved")
        return self


class _ExtensionCatalogModel(pydantic.BaseModel):
    """Validated catalog of registered scripts."""

    schema_version: typing.Literal[1] = 1
    generation: int = 0
    extensions: dict[str, _ExtensionRecord] = pydantic.Field(default_factory=dict)
    routine_favorites: tuple[tuple[str, str], ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def _validate_extension_identity(self) -> typing.Self:
        """Require persisted extension IDs to agree with their catalog keys."""
        for extension_id, record in self.extensions.items():
            if record.id != extension_id:
                raise ValueError("extension key does not match extension ID")
        if len(set(self.routine_favorites)) != len(self.routine_favorites):
            raise ValueError("routine favorites must be unique")
        return self


class _WorkspaceExtensionRequirement(pydantic.BaseModel):
    """Exact script dependency persisted in workspace schema 6."""

    extension_id: str
    capability_id: str
    capability_kind: typing.Literal["routine", "loader"]
    source_hash: str
    extension_api_version: int
    metadata_snapshot: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    embedded_object_id: str | None = None
    referencing_nodes: tuple[str, ...] = ()
    file_sources: tuple[str, ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("source_hash")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return _validate_source_hash(value)

    @pydantic.model_validator(mode="after")
    def _validate_embedded_object_id(self) -> typing.Self:
        if (
            self.embedded_object_id is not None
            and self.embedded_object_id != f"extension-{self.source_hash}"
        ):
            raise ValueError("embedded script object ID does not match its source")
        return self


_WorkspaceRequirementState = typing.Literal[
    "ready",
    "approval-required",
    "disabled",
    "missing",
    "hash-mismatch",
    "unsupported-api",
    "import-failed",
]


class _ResolvedWorkspaceRequirement(pydantic.BaseModel):
    requirement: _WorkspaceExtensionRequirement
    state: _WorkspaceRequirementState
    detail: str = ""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
