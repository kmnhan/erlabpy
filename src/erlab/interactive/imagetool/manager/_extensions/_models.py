"""Persistent schema models for ImageTool Manager extensions."""

from __future__ import annotations

import typing

import pydantic

from erlab.extensions import LoaderDescriptor, RoutineDescriptor  # noqa: TC001
from erlab.extensions._models import (
    _require_finite_parameter_values,
    _validate_revision_hash,
)


class _EnvironmentLoaderMethod(pydantic.BaseModel):
    """Serializable file-dialog entry from an installed ``LoaderBase`` plugin."""

    name_filter: str
    method: str | None = None
    defaults: dict[str, pydantic.JsonValue] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("defaults")
    @classmethod
    def _finite_defaults(
        cls, value: dict[str, pydantic.JsonValue]
    ) -> dict[str, pydantic.JsonValue]:
        _require_finite_parameter_values(value)
        return value


class _ExtensionRevision(pydantic.BaseModel):
    """One immutable source revision and its last validation result."""

    source_hash: str
    object_name: str
    change_summary: str = ""
    source_path: str | None = None
    source_modified_at: str | None = None
    created_at: str
    approved: bool = False
    routines: tuple[RoutineDescriptor, ...] = ()
    loaders: tuple[LoaderDescriptor, ...] = ()
    import_error: str | None = None
    entry_point_group: str | None = None
    entry_point_name: str | None = None
    entry_point_value: str | None = None
    distribution_name: str | None = None
    distribution_version: str | None = None
    editable: bool = False
    loader_always_single: bool | None = None
    loader_dialog_methods: tuple[_EnvironmentLoaderMethod, ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("source_hash")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return _validate_revision_hash(value)


def _revision_loader_name_filters(revision: _ExtensionRevision) -> tuple[str, ...]:
    """Return the file-dialog filters owned by one validated revision."""
    if revision.loader_dialog_methods:
        return tuple(item.name_filter for item in revision.loader_dialog_methods)
    if revision.entry_point_group == "erlab.io.loaders":
        return ()
    name_filters: list[str] = []
    for descriptor in revision.loaders:
        patterns = " ".join(f"*{suffix}" for suffix in descriptor.extensions) or "*"
        name_filters.append(f"{descriptor.name} ({patterns})")
    return tuple(name_filters)


class _ExtensionRecord(pydantic.BaseModel):
    """Application-wide extension state and immutable revision history."""

    id: str
    name: str
    source_type: typing.Literal["script", "environment-package"] = "script"
    enabled: bool = False
    embed_policy: typing.Literal["referenced", "always", "never"] = "referenced"
    current_revision: str
    revisions: dict[str, _ExtensionRevision]
    record_generation: int = 0

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def _validate_revision_identity(self) -> typing.Self:
        """Require every immutable revision to agree with its catalog key."""
        if self.current_revision not in self.revisions:
            raise ValueError("current extension revision is missing")
        for revision_hash, revision in self.revisions.items():
            if revision.source_hash != revision_hash:
                raise ValueError("extension revision key does not match source hash")
            entry_point_values = (
                revision.entry_point_group,
                revision.entry_point_name,
                revision.entry_point_value,
            )
            if self.source_type == "script" and any(
                value is not None for value in entry_point_values
            ):
                raise ValueError("a script revision cannot contain an entry point")
            if (
                self.source_type == "script"
                and revision.object_name != f"{revision_hash}.py"
            ):
                raise ValueError(
                    "a script revision object name must match its source hash"
                )
            if self.source_type == "environment-package" and not all(
                isinstance(value, str) and value for value in entry_point_values
            ):
                raise ValueError(
                    "an environment package revision requires an entry point"
                )
            if (
                self.source_type == "environment-package"
                and revision.object_name != revision.entry_point_value
            ):
                raise ValueError(
                    "an environment package object name must match its entry point"
                )
        if self.enabled and not self.revisions[self.current_revision].approved:
            raise ValueError("an enabled extension revision must be approved")
        return self


class _ExtensionCatalogModel(pydantic.BaseModel):
    """Validated application catalog stored as one atomic JSON document."""

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
        for extension_id, _routine_id in self.routine_favorites:
            if extension_id not in self.extensions:
                raise ValueError("routine favorite references an unknown extension")
        return self


class _WorkspaceExtensionRequirement(pydantic.BaseModel):
    """Exact extension dependency persisted in workspace schema 6."""

    extension_id: str
    capability_id: str
    capability_kind: typing.Literal["routine", "loader"]
    revision_hash: str
    extension_api_version: int
    source_type: typing.Literal["script", "environment-package"]
    metadata_snapshot: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    embedded_object_id: str | None = None
    referencing_nodes: tuple[str, ...] = ()
    file_sources: tuple[str, ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("revision_hash")
    @classmethod
    def _valid_revision_hash(cls, value: str) -> str:
        return _validate_revision_hash(value)

    @pydantic.model_validator(mode="after")
    def _validate_embedded_object_id(self) -> typing.Self:
        if self.source_type == "environment-package":
            if self.embedded_object_id is not None:
                raise ValueError(
                    "an environment package requirement cannot embed source"
                )
            return self
        if (
            self.embedded_object_id is not None
            and self.embedded_object_id != f"extension-{self.revision_hash}"
        ):
            raise ValueError("embedded script object ID does not match its revision")
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
