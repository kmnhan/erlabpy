"""Persistent schema models for ImageTool Manager extensions."""

from __future__ import annotations

import typing

import pydantic

from erlab.extensions import LoaderDescriptor, RoutineDescriptor  # noqa: TC001
from erlab.extensions._models import (
    _PackageExtensionReference,
    _require_finite_parameter_values,
    _validate_source_hash,
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


class _ExtensionSource(pydantic.BaseModel):
    """Source identity and the current process's validation result."""

    source_hash: str
    object_name: str
    source_path: str | None = None
    source_modified_at: str | None = None
    registered_at: str
    approved: bool = False
    routines: tuple[RoutineDescriptor, ...] = ()
    loaders: tuple[LoaderDescriptor, ...] = ()
    routine_call_references: dict[str, str] = pydantic.Field(default_factory=dict)
    loader_call_references: dict[str, str] = pydantic.Field(default_factory=dict)
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
        return _validate_source_hash(value)

    @property
    def package_reference(self) -> _PackageExtensionReference | None:
        """Return standard package identity for an entry-point source."""
        if not all(
            isinstance(value, str) and value
            for value in (
                self.entry_point_group,
                self.entry_point_name,
                self.entry_point_value,
            )
        ):
            return None
        entry_point_name = typing.cast("str", self.entry_point_name)
        return _PackageExtensionReference(
            distribution_name=self.distribution_name or entry_point_name,
            distribution_version=self.distribution_version or "",
            entry_point_group=typing.cast("str", self.entry_point_group),
            entry_point_name=entry_point_name,
            entry_point_value=typing.cast("str", self.entry_point_value),
            editable=self.editable,
        )


def _source_loader_name_filters(source: _ExtensionSource) -> tuple[str, ...]:
    """Return the file-dialog filters owned by one validated source."""
    if source.loader_dialog_methods:
        return tuple(item.name_filter for item in source.loader_dialog_methods)
    if source.entry_point_group == "erlab.io.loaders":
        return ()
    name_filters: list[str] = []
    for descriptor in source.loaders:
        patterns = " ".join(f"*{suffix}" for suffix in descriptor.extensions) or "*"
        name_filters.append(f"{descriptor.name} ({patterns})")
    return tuple(name_filters)


class _ExtensionRecord(pydantic.BaseModel):
    """Runtime record for one registered script or discovered package."""

    id: str
    name: str
    source_type: typing.Literal["script", "environment-package"] = "script"
    enabled: bool = False
    embed_policy: typing.Literal["referenced", "always", "never"] = "referenced"
    source: _ExtensionSource
    record_generation: int = 0

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def _validate_source_identity(self) -> typing.Self:
        """Require the current source to match its registration type."""
        entry_point_values = (
            self.source.entry_point_group,
            self.source.entry_point_name,
            self.source.entry_point_value,
        )
        if self.source_type == "script" and any(
            value is not None for value in entry_point_values
        ):
            raise ValueError("a script source cannot contain an entry point")
        if (
            self.source_type == "script"
            and self.source.object_name != f"{self.source.source_hash}.py"
        ):
            raise ValueError("a script source object name must match its source hash")
        if self.source_type == "environment-package" and not all(
            isinstance(value, str) and value for value in entry_point_values
        ):
            raise ValueError("an environment package source requires an entry point")
        if (
            self.source_type == "environment-package"
            and self.source.object_name != self.source.entry_point_value
        ):
            raise ValueError(
                "an environment package source object name must match its entry point"
            )
        if self.enabled and not self.source.approved:
            raise ValueError("an enabled extension source must be approved")
        return self


class _ExtensionCatalogModel(pydantic.BaseModel):
    """Validated script catalog or merged runtime extension view."""

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
    """Exact extension dependency persisted in workspace schema 6."""

    extension_id: str
    capability_id: str
    capability_kind: typing.Literal["routine", "loader"]
    source_hash: str
    extension_api_version: int
    source_type: typing.Literal["script", "environment-package"]
    metadata_snapshot: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    package: _PackageExtensionReference | None = pydantic.Field(
        default=None, exclude_if=lambda value: value is None
    )
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
        if self.source_type == "environment-package":
            if self.embedded_object_id is not None:
                raise ValueError(
                    "an environment package requirement cannot embed source"
                )
            if self.package is None:
                raise ValueError(
                    "an environment package requirement requires package identity"
                )
            return self
        if self.package is not None:
            raise ValueError("a script requirement cannot contain package identity")
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
