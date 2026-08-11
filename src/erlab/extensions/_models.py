"""Validated public models for ERLab extensions."""

from __future__ import annotations

import enum
import typing

import pydantic

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

EXTENSION_API_VERSION: typing.Literal[1] = 1


class ExtensionError(RuntimeError):
    """Base class for errors raised by the ERLab extension API.

    Catch this error when one handler must process import, validation, lookup, and
    execution failures in the same way.
    """


class ExtensionImportError(ExtensionError):
    """An extension source could not be read, verified, or imported."""


class ExtensionSignatureError(ExtensionError):
    """A decorated function has an unsupported signature or annotation."""


class ExtensionNotFoundError(ExtensionError):
    """A requested extension revision or capability is not available."""


class ExtensionExecutionError(ExtensionError):
    """An extension call failed or returned an unsupported value."""


class ParameterKind(enum.StrEnum):
    """Supported editor type for an extension parameter.

    Attributes
    ----------
    BOOLEAN
        A Boolean value.
    INTEGER
        An integer value.
    NUMBER
        A floating-point value.
    STRING
        A text value.
    PATH
        A file-system path.
    LITERAL
        One value from a ``Literal`` annotation.
    ENUM
        One value from an enumeration.
    """

    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    PATH = "path"
    LITERAL = "literal"
    ENUM = "enum"


class ParameterDescriptor(pydantic.BaseModel):
    """Description of one user-editable extension parameter.

    Parameters
    ----------
    id
        Python parameter name.
    kind
        Editor type used by graphical clients.
    required
        Whether the caller must supply a value.
    optional
        Whether ``None`` is an accepted value.
    default
        JSON-compatible default value, if one exists.
    choices
        Accepted values for a literal or enumeration parameter.
    """

    id: str
    kind: ParameterKind
    required: bool
    optional: bool = False
    default: bool | int | float | str | None = None
    choices: tuple[bool | int | float | str, ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("id")
    @classmethod
    def _nonempty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("parameter ID cannot be empty")
        return value


class RoutineDescriptor(pydantic.BaseModel):
    """Public description of a single-input analysis routine.

    Parameters
    ----------
    id
        Stable capability identifier.
    name
        Display name.
    category
        Display category.
    summary
        Short user-facing description.
    parameters
        Parameters shown after the input array.
    extension_api_version
        Extension protocol version used by this descriptor.
    """

    id: str
    name: str
    category: str
    summary: str
    parameters: tuple[ParameterDescriptor, ...] = ()
    extension_api_version: typing.Literal[1] = EXTENSION_API_VERSION

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("id", "name", "category")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("extension descriptor text cannot be empty")
        return value


class LoaderDescriptor(pydantic.BaseModel):
    """Public description of a path-based data loader.

    Parameters
    ----------
    id
        Stable capability identifier.
    name
        Display name.
    category
        Display category.
    summary
        Short description.
    parameters
        Parameters shown after the input path.
    extensions
        Optional filename extensions, including the leading dot.
    extension_api_version
        Extension protocol version used by this descriptor.
    """

    id: str
    name: str
    category: str
    summary: str
    parameters: tuple[ParameterDescriptor, ...] = ()
    extensions: tuple[str, ...] = ()
    extension_api_version: typing.Literal[1] = EXTENSION_API_VERSION

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("id", "name", "category")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("extension descriptor text cannot be empty")
        return value


CapabilityDescriptor = RoutineDescriptor | LoaderDescriptor


class LoadedScript:
    """Imported extension script and its validated capabilities.

    Instances are returned by :func:`erlab.extensions.load_script`. The normal
    decorated functions are also available from :attr:`module`.

    Parameters
    ----------
    path
        Resolved source path.
    revision
        SHA-256 digest of the imported source bytes.
    module
        Imported Python module.
    routines
        Routine descriptors and their corresponding functions, keyed by ID.
    loaders
        Loader descriptors and their corresponding functions, keyed by ID.
    """

    def __init__(
        self,
        *,
        path: Path,
        revision: str,
        module: typing.Any,
        routines: dict[str, tuple[RoutineDescriptor, Callable[..., typing.Any]]],
        loaders: dict[str, tuple[LoaderDescriptor, Callable[..., typing.Any]]],
    ) -> None:
        self.path = path
        self.revision = revision
        self.module = module
        self.routines = routines
        self.loaders = loaders

    @property
    def capabilities(self) -> tuple[CapabilityDescriptor, ...]:
        """All validated capabilities in source definition order.

        Returns
        -------
        tuple of RoutineDescriptor or LoaderDescriptor
            The routines followed by the loaders from the imported script.
        """
        routines: list[CapabilityDescriptor] = [
            entry[0] for entry in self.routines.values()
        ]
        loaders: list[CapabilityDescriptor] = [
            entry[0] for entry in self.loaders.values()
        ]
        return (*routines, *loaders)
