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


class LoadedScript:
    """Imported extension script and its validated capabilities.

    Instances are returned by :func:`erlab.extensions.load_script`. Attributes from
    the imported module are available directly on the instance, so a loaded routine
    can be called as ``extension.normalize(data)``.

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

    def __getattr__(self, name: str) -> typing.Any:
        """Return a public attribute from the imported script module."""
        for descriptor, func in self.routines.values():
            if descriptor.function_name == name:
                return _extension_callable(func)
        for descriptor, func in self.loaders.values():
            if descriptor.function_name == name:
                return _extension_callable(func)
        return getattr(self.module, name)

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


class LoadedEntryPoint:
    """Imported package extension with direct access to its public callables.

    Parameters
    ----------
    group
        Python entry-point group.
    name
        Python entry-point name.
    revision
        Exact revision hash computed from package metadata and editable sources.
    value
        Object loaded from the entry point. A ``LoaderBase`` class entry point is
        instantiated so its loader methods are available directly.
    callables
        Validated extension callables keyed by their Python function names.
    loader_methods
        LoaderBase file-dialog callables keyed by their stable method references.
    """

    def __init__(
        self,
        *,
        group: str,
        name: str,
        revision: str,
        value: typing.Any,
        callables: dict[str, Callable[..., typing.Any]],
        loader_methods: dict[str | None, Callable[..., typing.Any]],
    ) -> None:
        self.group = group
        self.name = name
        self.revision = revision
        self.value = value
        self.callables = callables
        self.loader_methods = loader_methods

    def __getattr__(self, name: str) -> typing.Any:
        """Return a public attribute or the entry-point callable itself."""
        if name in self.callables:
            return _extension_callable(self.callables[name])
        return getattr(self.value, name)

    def resolve_loader(self, method: str | None = None) -> Callable[..., typing.Any]:
        """Return one declared LoaderBase file-dialog callable.

        Parameters
        ----------
        method
            Stable method reference reported by the loader entry point. Use ``None``
            for its normal ``load`` method.

        Returns
        -------
        collections.abc.Callable
            The exact callable declared by the verified entry point.

        Raises
        ------
        ExtensionNotFoundError
            If this entry point does not declare the requested loader method.

        Examples
        --------
        >>> extension = load_entry_point(  # doctest: +SKIP
        ...     "erlab.io.loaders",
        ...     "my_lab",
        ...     expected_revision="0a12...",
        ... )
        >>> load_preview = extension.resolve_loader(  # doctest: +SKIP
        ...     "my_lab.preview.load"
        ... )
        >>> data = load_preview("scan.dat")  # doctest: +SKIP
        """
        try:
            return self.loader_methods[method]
        except KeyError as error:
            raise ExtensionNotFoundError(
                f"Loader method {method!r} is not declared by entry point "
                f"{self.group}:{self.name}"
            ) from error
