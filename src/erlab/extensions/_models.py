"""Validated public models for ERLab extensions."""

from __future__ import annotations

import enum
import functools
import inspect
import math
import typing
from collections.abc import Mapping

import pydantic

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

EXTENSION_API_VERSION: typing.Literal[1] = 1


def _validate_source_hash(value: str) -> str:
    """Validate one lowercase SHA-256 digest used as immutable identity."""
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("source hash must be a lowercase SHA-256 digest")
    return value


def _require_finite_parameter_values(values: Mapping[str, typing.Any]) -> None:
    """Reject non-finite floats before extension values cross persistence boundaries."""

    def check(value: typing.Any, path: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Extension parameter {path!r} must be finite")
        if isinstance(value, Mapping):
            for key, item in value.items():
                check(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check(item, f"{path}[{index}]")

    for name, value in values.items():
        check(value, name)


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
    """A requested extension source or capability is not available."""


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

    @pydantic.model_validator(mode="after")
    def _finite_numeric_values(self) -> typing.Self:
        _require_finite_parameter_values(
            {
                self.id: self.default,
                **{
                    f"{self.id} choice {index}": value
                    for index, value in enumerate(self.choices)
                },
            }
        )
        return self


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
    function_name
        Python function name in the source module.
    parameters
        Parameters shown after the input array.
    extension_api_version
        Extension protocol version used by this descriptor.
    """

    id: str
    name: str
    category: str
    summary: str
    function_name: str
    parameters: tuple[ParameterDescriptor, ...] = ()
    extension_api_version: typing.Literal[1] = EXTENSION_API_VERSION

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("id", "name", "category", "function_name")
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
    function_name
        Python function name in the source module.
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
    function_name: str
    parameters: tuple[ParameterDescriptor, ...] = ()
    extensions: tuple[str, ...] = ()
    extension_api_version: typing.Literal[1] = EXTENSION_API_VERSION

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("id", "name", "category", "function_name")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("extension descriptor text cannot be empty")
        return value

    @pydantic.field_validator("extensions")
    @classmethod
    def _valid_extensions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.startswith(".") or value == "." for value in values):
            raise ValueError(
                "loader filename extensions must start with a dot and contain a suffix"
            )
        if len(values) != len(set(values)):
            raise ValueError("loader filename extensions must be unique")
        return values


CapabilityDescriptor = RoutineDescriptor | LoaderDescriptor


def _extension_callable(func: Callable[..., typing.Any]) -> Callable[..., typing.Any]:
    """Restore descriptor-compatible values before a dynamically loaded call."""
    from erlab.extensions._api import _coerce_call_parameters

    signature = inspect.signature(func)
    input_name = next(iter(signature.parameters))

    @functools.wraps(func)
    def call(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        bound = signature.bind(*args, **kwargs)
        parameters = {
            name: value for name, value in bound.arguments.items() if name != input_name
        }
        bound.arguments.update(_coerce_call_parameters(func, parameters))
        return func(*bound.args, **bound.kwargs)

    return call


class LoadedScriptInfo:
    """ERLab information for one loaded extension script.

    Access this object through :attr:`LoadedScript.erlab`. Keeping ERLab-owned
    attributes in one namespace lets extension functions use ordinary names such as
    ``path``, ``module``, or ``loaders``.

    Parameters
    ----------
    path
        Resolved source path.
    source_hash
        SHA-256 digest of the imported source bytes.
    module
        Imported Python module.
    routines
        Routine descriptors and functions, keyed by capability ID.
    loaders
        Loader descriptors and functions, keyed by capability ID.
    """

    def __init__(
        self,
        *,
        path: Path,
        source_hash: str,
        module: typing.Any,
        routines: dict[str, tuple[RoutineDescriptor, Callable[..., typing.Any]]],
        loaders: dict[str, tuple[LoaderDescriptor, Callable[..., typing.Any]]],
    ) -> None:
        self.path = path
        self.source_hash = source_hash
        self.module = module
        self.routines = routines
        self.loaders = loaders

    @property
    def capabilities(self) -> tuple[CapabilityDescriptor, ...]:
        """Return all validated capabilities in source definition order.

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


class LoadedScript:
    """Imported extension script with natural access to its public functions.

    Instances are returned by :func:`erlab.extensions.load_script`. Call a decorated
    function as a normal attribute. Use :attr:`erlab` only when you need descriptors,
    the source hash, or other import information.

    Parameters
    ----------
    erlab
        ERLab information for the imported script.

    Examples
    --------
    >>> from erlab.extensions import load_script
    >>> extension = load_script("my_extension.py")  # doctest: +SKIP
    >>> result = extension.normalize(data)  # doctest: +SKIP
    >>> tuple(extension.erlab.routines)  # doctest: +SKIP
    ('normalize',)
    """

    __slots__ = ("__erlab_info",)

    def __init__(self, erlab: LoadedScriptInfo) -> None:
        self.__erlab_info = erlab

    @property
    def erlab(self) -> LoadedScriptInfo:
        """Return descriptors and import information owned by ERLab."""
        return self.__erlab_info

    def __getattr__(self, name: str) -> typing.Any:
        """Return a decorated function or another public script attribute."""
        for descriptor, func in self.erlab.routines.values():
            if descriptor.function_name == name:
                return _extension_callable(func)
        for descriptor, func in self.erlab.loaders.values():
            if descriptor.function_name == name:
                return _extension_callable(func)
        return getattr(self.erlab.module, name)
