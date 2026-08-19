"""Decorators and source loading for ERLab extensions."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import importlib.util
import inspect
import os
import pathlib
import sys
import threading
import types
import typing
import uuid
from collections.abc import Callable, Iterable, Mapping

import pydantic
import xarray as xr

from erlab.extensions._models import (
    EXTENSION_API_VERSION,
    ExtensionExecutionError,
    ExtensionImportError,
    ExtensionNotFoundError,
    ExtensionSignatureError,
    LoadedScript,
    LoadedScriptInfo,
    LoaderDescriptor,
    ParameterDescriptor,
    ParameterKind,
    RoutineDescriptor,
    _require_finite_parameter_values,
)

_CAPABILITY_ATTRIBUTE = "__erlab_extension_capability__"
_CapabilityStatus = typing.Literal[
    "ready",
    "disabled",
    "approval-required",
    "missing-source",
    "missing-capability",
    "hash-mismatch",
    "unsupported-api",
    "validation-failed",
]


@dataclasses.dataclass(frozen=True)
class _RegisteredScriptCapability:
    """One capability and its verified registered script source."""

    registered_path: pathlib.Path
    script_name: str
    source_hash: str
    descriptor: RoutineDescriptor | LoaderDescriptor
    source_bytes: bytes = dataclasses.field(repr=False)


class _RegisteredScriptUnavailable(LookupError):
    """Report why one registered script capability cannot be resolved."""

    def __init__(self, status: _CapabilityStatus) -> None:
        super().__init__(status)
        self.status = status


class _RegisteredScriptBackend(typing.Protocol):
    """Resolve registered scripts for public calls and copied code."""

    def resolve_registered_capability(
        self,
        script_name: str,
        kind: typing.Literal["routine", "loader"],
        capability_id: str,
        *,
        source_hash: str | None = None,
        require_enabled: bool = True,
    ) -> _RegisteredScriptCapability: ...


_registered_script_backends: dict[str, _RegisteredScriptBackend] = {}
_resolver_lock = threading.RLock()


def _metadata(
    *,
    kind: typing.Literal["routine", "loader"],
    name: str | None,
    id: str | None,  # noqa: A002 - public API uses the conventional metadata name.
    category: str,
    summary: str,
    extensions: tuple[str, ...] = (),
) -> dict[str, typing.Any]:
    return {
        "kind": kind,
        "name": name,
        "id": id,
        "category": category,
        "summary": summary,
        "extensions": extensions,
        "extension_api_version": EXTENSION_API_VERSION,
    }


def routine(
    *,
    name: str | None = None,
    id: str | None = None,  # noqa: A002
    category: str = "Other",
    summary: str = "",
) -> Callable[[Callable[..., xr.DataArray]], Callable[..., xr.DataArray]]:
    """Mark a normal Python function as an ImageTool analysis routine.

    The decorator does not wrap the function. Calling the decorated function in a
    notebook has the same behavior as calling the original function.

    Parameters
    ----------
    name
        Display name. The function name is used by default.
    id
        Stable capability identifier. Set this value before renaming a function.
    category
        Group shown in graphical clients.
    summary
        Short user-facing description.

    Returns
    -------
    callable
        A decorator that returns the supplied function unchanged.

    Examples
    --------
    >>> import xarray as xr
    >>> from erlab.extensions import routine
    >>> @routine(name="Double")
    ... def double(data: xr.DataArray) -> xr.DataArray:
    ...     return 2 * data
    >>> double(xr.DataArray([1, 2])).values.tolist()
    [2, 4]
    """

    def decorate(func: Callable[..., xr.DataArray]) -> Callable[..., xr.DataArray]:
        setattr(
            func,
            _CAPABILITY_ATTRIBUTE,
            _metadata(
                kind="routine",
                name=name,
                id=id,
                category=category,
                summary=summary,
            ),
        )
        return func

    return decorate


def loader(
    *,
    name: str | None = None,
    id: str | None = None,  # noqa: A002
    category: str = "Other",
    summary: str = "",
    extensions: str | Iterable[str] = (),
) -> Callable[[Callable[..., typing.Any]], Callable[..., typing.Any]]:
    """Mark a normal Python function as an external data loader.

    The first parameter must accept :class:`pathlib.Path`. The result must be an
    xarray ``DataArray``, ``Dataset``, or ``DataTree``.

    Parameters
    ----------
    name
        Display name. The function name is used by default.
    id
        Stable capability identifier.
    category
        Group shown in graphical clients.
    summary
        Short user-facing description.
    extensions
        One filename extension or an iterable of extensions accepted by the loader.
        The leading dot is optional.

    Returns
    -------
    callable
        A decorator that returns the supplied function unchanged.

    Examples
    --------
    >>> from pathlib import Path
    >>> import xarray as xr
    >>> from erlab.extensions import loader
    >>> @loader(name="Text values", extensions=(".txt",))
    ... def load_text(path: Path) -> xr.DataArray:
    ...     return xr.DataArray([float(path.read_text())])
    """
    extension_values = (extensions,) if isinstance(extensions, str) else extensions
    normalized_extensions: list[str] = []
    for value in extension_values:
        if not isinstance(value, str):
            raise TypeError("loader filename extensions must be strings")
        value = value.strip().casefold()
        if not value or value == ".":
            raise ValueError("a loader filename extension must contain a suffix")
        normalized_extensions.append(value if value.startswith(".") else f".{value}")

    def decorate(func: Callable[..., typing.Any]) -> Callable[..., typing.Any]:
        setattr(
            func,
            _CAPABILITY_ATTRIBUTE,
            _metadata(
                kind="loader",
                name=name,
                id=id,
                category=category,
                summary=summary,
                extensions=tuple(normalized_extensions),
            ),
        )
        return func

    return decorate


def _split_optional(annotation: typing.Any) -> tuple[typing.Any, bool]:
    arguments = typing.get_args(annotation)
    if arguments and type(None) in arguments:
        remaining = tuple(value for value in arguments if value is not type(None))
        if len(remaining) == 1:
            return remaining[0], True
    return annotation, False


def _parameter_descriptor(
    parameter: inspect.Parameter, annotation: typing.Any
) -> ParameterDescriptor:
    annotation, optional = _split_optional(annotation)
    choices: tuple[bool | int | float | str, ...] = ()
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        literal_values = typing.get_args(annotation)
        if not literal_values or any(
            not isinstance(value, (bool, int, float, str)) for value in literal_values
        ):
            raise ExtensionSignatureError(
                f"Parameter {parameter.name!r} has unsupported Literal values"
            )
        kind = ParameterKind.LITERAL
        choices = typing.cast("tuple[bool | int | float | str, ...]", literal_values)
    elif inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        enum_values = tuple(member.value for member in annotation)
        if not enum_values or any(
            not isinstance(value, (bool, int, float, str)) for value in enum_values
        ):
            raise ExtensionSignatureError(
                f"Parameter {parameter.name!r} has unsupported Enum values"
            )
        kind = ParameterKind.ENUM
        choices = typing.cast("tuple[bool | int | float | str, ...]", enum_values)
    else:
        kinds = {
            bool: ParameterKind.BOOLEAN,
            int: ParameterKind.INTEGER,
            float: ParameterKind.NUMBER,
            str: ParameterKind.STRING,
            pathlib.Path: ParameterKind.PATH,
        }
        kind = kinds.get(annotation)
        if kind is None:
            raise ExtensionSignatureError(
                f"Parameter {parameter.name!r} has unsupported annotation "
                f"{annotation!r}"
            )
    required = parameter.default is inspect.Parameter.empty
    default: bool | int | float | str | None = None
    if parameter.default is not inspect.Parameter.empty:
        raw_default = parameter.default
        if raw_default is None:
            if not optional:
                raise ExtensionSignatureError(
                    f"Parameter {parameter.name!r} has a None default but is not "
                    "optional"
                )
        elif kind is ParameterKind.ENUM:
            if not isinstance(raw_default, annotation):
                raise ExtensionSignatureError(
                    f"Parameter {parameter.name!r} has a default that is not a "
                    f"{annotation.__name__} member"
                )
            raw_default = raw_default.value
        elif kind is ParameterKind.LITERAL:
            if not any(
                type(raw_default) is type(choice) and raw_default == choice
                for choice in choices
            ):
                raise ExtensionSignatureError(
                    f"Parameter {parameter.name!r} has a default outside its Literal "
                    "choices"
                )
        elif kind is ParameterKind.PATH:
            if not isinstance(raw_default, pathlib.Path):
                raise ExtensionSignatureError(
                    f"Parameter {parameter.name!r} must use a pathlib.Path default"
                )
            raw_default = os.fspath(raw_default)
        elif kind is ParameterKind.BOOLEAN and type(raw_default) is not bool:
            raise ExtensionSignatureError(
                f"Parameter {parameter.name!r} must use a bool default"
            )
        elif kind is ParameterKind.INTEGER and type(raw_default) is not int:
            raise ExtensionSignatureError(
                f"Parameter {parameter.name!r} must use an int default"
            )
        elif kind is ParameterKind.NUMBER and (type(raw_default) not in (int, float)):
            raise ExtensionSignatureError(
                f"Parameter {parameter.name!r} must use a numeric default"
            )
        elif kind is ParameterKind.STRING and not isinstance(raw_default, str):
            raise ExtensionSignatureError(
                f"Parameter {parameter.name!r} must use a string default"
            )
        default = raw_default
    try:
        _require_finite_parameter_values(
            {
                parameter.name: default,
                **{
                    f"{parameter.name} choice {index}": value
                    for index, value in enumerate(choices)
                },
            }
        )
    except ValueError as error:
        raise ExtensionSignatureError(str(error)) from error
    return ParameterDescriptor(
        id=parameter.name,
        kind=kind,
        required=required,
        optional=optional,
        default=default,
        choices=choices,
    )


def _resolved_hints(func: Callable[..., typing.Any]) -> dict[str, typing.Any]:
    try:
        return typing.get_type_hints(func)
    except Exception as error:
        raise ExtensionSignatureError(
            f"Could not resolve annotations for {func.__name__!r}: {error}"
        ) from error


def _coerce_call_parameters(
    func: Callable[..., typing.Any], parameters: Mapping[str, typing.Any]
) -> dict[str, typing.Any]:
    """Validate and restore values represented by parameter descriptors."""
    hints = _resolved_hints(func)
    signature_parameters = tuple(inspect.signature(func).parameters.values())[1:]
    parameters_by_name = {
        parameter.name: parameter for parameter in signature_parameters
    }
    unknown = set(parameters) - parameters_by_name.keys()
    if unknown:
        raise TypeError(f"Unknown extension parameters: {', '.join(sorted(unknown))}")
    missing = tuple(
        parameter.name
        for parameter in signature_parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.name not in parameters
    )
    if missing:
        raise TypeError(f"Missing extension parameters: {', '.join(missing)}")
    values: dict[str, typing.Any] = {}
    for name, value in parameters.items():
        annotation, optional = _split_optional(hints.get(name))
        if value is None:
            if not optional:
                raise TypeError(f"Parameter {name!r} does not accept None")
            values[name] = None
            continue
        if annotation is pathlib.Path:
            if not isinstance(value, (str, os.PathLike)):
                raise TypeError(f"Parameter {name!r} must be a path")
            values[name] = pathlib.Path(value)
        elif inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            try:
                values[name] = annotation(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Parameter {name!r} is not a valid {annotation.__name__} value"
                ) from error
        elif typing.get_origin(annotation) is typing.Literal:
            choices = typing.get_args(annotation)
            if not any(
                type(value) is type(choice) and value == choice for choice in choices
            ):
                raise ValueError(f"Parameter {name!r} must be one of {choices!r}")
            values[name] = value
        elif annotation is bool:
            if type(value) is not bool:
                raise TypeError(f"Parameter {name!r} must be a bool")
            values[name] = value
        elif annotation is int:
            if type(value) is not int:
                raise TypeError(f"Parameter {name!r} must be an int")
            values[name] = value
        elif annotation is float:
            if type(value) not in (int, float):
                raise TypeError(f"Parameter {name!r} must be a number")
            _require_finite_parameter_values({name: value})
            values[name] = value
        elif annotation is str:
            if not isinstance(value, str):
                raise TypeError(f"Parameter {name!r} must be a string")
            values[name] = value
        else:
            raise ExtensionSignatureError(
                f"Parameter {name!r} has unsupported annotation {annotation!r}"
            )
    return values


def _descriptor_for(
    func: Callable[..., typing.Any], metadata: Mapping[str, typing.Any]
) -> RoutineDescriptor | LoaderDescriptor:
    if (
        inspect.iscoroutinefunction(func)
        or inspect.isgeneratorfunction(func)
        or inspect.isasyncgenfunction(func)
    ):
        raise ExtensionSignatureError(
            f"Extension function {func.__name__!r} must be a synchronous function"
        )
    signature = inspect.signature(func)
    parameters = tuple(signature.parameters.values())
    if not parameters:
        raise ExtensionSignatureError(
            f"Extension function {func.__name__!r} must have an input parameter"
        )
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        raise ExtensionSignatureError(
            f"Extension function {func.__name__!r} cannot use *args or **kwargs"
        )
    first_parameter = parameters[0]
    if (
        first_parameter.kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        or first_parameter.default is not inspect.Parameter.empty
    ):
        raise ExtensionSignatureError(
            f"Extension function {func.__name__!r} must require its input as the "
            "first positional parameter"
        )
    if any(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in parameters[1:]
    ):
        raise ExtensionSignatureError(
            f"Extension function {func.__name__!r} cannot use positional-only "
            "user parameters"
        )
    hints = _resolved_hints(func)
    first_annotation = hints.get(parameters[0].name)
    kind = metadata["kind"]
    if func.__name__ == "erlab":
        raise ExtensionSignatureError(
            "Extension function name 'erlab' is reserved for loaded script information"
        )
    if kind == "loader":
        reserved_parameters = {
            parameter.name for parameter in parameters[1:]
        }.intersection({"loader_extensions", "without_values"})
        if reserved_parameters:
            names = ", ".join(repr(name) for name in sorted(reserved_parameters))
            raise ExtensionSignatureError(
                f"Loader {func.__name__!r} uses parameter names reserved by ERLab: "
                f"{names}"
            )
    if kind == "routine" and first_annotation is not xr.DataArray:
        raise ExtensionSignatureError(
            f"Routine {func.__name__!r} must annotate its first parameter as "
            "xarray.DataArray"
        )
    if kind == "loader" and first_annotation is not pathlib.Path:
        raise ExtensionSignatureError(
            f"Loader {func.__name__!r} must annotate its first parameter as "
            "pathlib.Path"
        )
    descriptor_parameters = tuple(
        _parameter_descriptor(parameter, hints.get(parameter.name))
        for parameter in parameters[1:]
    )
    capability_id = metadata["id"] or func.__name__
    common = {
        "id": capability_id,
        "name": metadata["name"] or func.__name__.replace("_", " ").title(),
        "category": metadata["category"],
        "summary": metadata["summary"],
        "function_name": func.__name__,
        "parameters": descriptor_parameters,
    }
    if kind == "routine":
        return_annotation = hints.get("return")
        if return_annotation is not xr.DataArray:
            raise ExtensionSignatureError(
                f"Routine {func.__name__!r} must return xarray.DataArray"
            )
        return RoutineDescriptor(**common)
    return_annotation = hints.get("return")
    accepted_returns = {xr.DataArray, xr.Dataset, xr.DataTree}
    return_types = set(typing.get_args(return_annotation))
    if return_annotation not in accepted_returns and not (
        typing.get_origin(return_annotation) in (typing.Union, types.UnionType)
        and return_types
        and return_types <= accepted_returns
    ):
        raise ExtensionSignatureError(
            f"Loader {func.__name__!r} must return xarray.DataArray, "
            "xarray.Dataset, or xarray.DataTree"
        )
    return LoaderDescriptor(**common, extensions=metadata["extensions"])


def _module_capabilities(
    module: types.ModuleType,
) -> tuple[
    dict[str, tuple[RoutineDescriptor, Callable[..., typing.Any]]],
    dict[str, tuple[LoaderDescriptor, Callable[..., typing.Any]]],
]:
    routines: dict[str, tuple[RoutineDescriptor, Callable[..., typing.Any]]] = {}
    loaders: dict[str, tuple[LoaderDescriptor, Callable[..., typing.Any]]] = {}
    seen_callables: set[int] = set()
    for value in vars(module).values():
        metadata = getattr(value, _CAPABILITY_ATTRIBUTE, None)
        if (
            not callable(value)
            or not isinstance(metadata, Mapping)
            or getattr(value, "__module__", None) != module.__name__
        ):
            continue
        callable_identity = id(value)
        if callable_identity in seen_callables:
            continue
        seen_callables.add(callable_identity)
        try:
            descriptor = _descriptor_for(value, metadata)
        except pydantic.ValidationError as error:
            raise ExtensionSignatureError(
                f"Extension function {value.__name__!r} has invalid metadata: {error}"
            ) from error
        destination_ids = (
            routines if isinstance(descriptor, RoutineDescriptor) else loaders
        )
        if descriptor.id in destination_ids:
            raise ExtensionSignatureError(
                f"Capability ID {descriptor.id!r} is defined more than once"
            )
        if isinstance(descriptor, RoutineDescriptor):
            routines[descriptor.id] = (descriptor, value)
        else:
            loaders[descriptor.id] = (descriptor, value)
    return routines, loaders


def _load_script_bytes(
    source: bytes,
    source_path: pathlib.Path,
    *,
    module_name: str | None = None,
    expected_source_hash: str | None = None,
) -> LoadedScript:
    """Import an immutable source snapshot with ``source_path`` as its origin."""
    source_hash = hashlib.sha256(source).hexdigest()
    if expected_source_hash is not None and source_hash != expected_source_hash:
        raise ExtensionImportError(
            f"Extension source hash {source_hash} does not match expected hash "
            f"{expected_source_hash}"
        )
    import_name = (
        module_name
        if module_name is not None
        else f"_erlab_extension_{source_hash[:12]}_{uuid.uuid4().hex}"
    )
    if import_name in sys.modules:
        raise ExtensionImportError(
            f"Extension module name {import_name!r} is already in use"
        )
    spec = importlib.util.spec_from_file_location(import_name, source_path)
    if spec is None or spec.loader is None:
        raise ExtensionImportError(f"Could not create an import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[import_name] = module
    loaded = False
    try:
        try:
            code = compile(source, os.fspath(source_path), "exec")
            exec(code, module.__dict__)  # noqa: S102 - imports approved extension code
        except Exception as error:
            raise ExtensionImportError(
                f"Could not import extension source {source_path}: {error}"
            ) from error
        routines, loaders = _module_capabilities(module)
        if not routines and not loaders:
            raise ExtensionSignatureError(
                f"Extension source {source_path} contains no decorated capabilities"
            )
        loaded = True
    finally:
        if not loaded and sys.modules.get(import_name) is module:
            sys.modules.pop(import_name, None)
    return LoadedScript(
        LoadedScriptInfo(
            path=source_path,
            source_hash=source_hash,
            module=module,
            routines=routines,
            loaders=loaders,
        )
    )


def load_script(
    path: os.PathLike[str] | str,
    *,
    module_name: str | None = None,
    expected_source_hash: str | None = None,
) -> LoadedScript:
    """Import and validate all decorated capabilities in a Python script.

    The source directory is not added to ``sys.path``. Each call uses a new module
    name unless ``module_name`` is supplied.

    Parameters
    ----------
    path
        Python source file.
    module_name
        Import name. The name must not already exist in ``sys.modules``. Graphical
        clients use a name that identifies their session.
    expected_source_hash
        Required SHA-256 source hash. A mismatch stops the import.

    Returns
    -------
    LoadedScript
        Imported module, source hash, and validated capabilities.

    Raises
    ------
    ExtensionImportError
        If the source cannot be read or imported, or if ``module_name`` is already
        in use.
    ExtensionSignatureError
        If a decorated function has an unsupported signature.

    Examples
    --------
    A script can contain routines, loaders, or both.

    >>> from erlab.extensions import load_script
    >>> loaded = load_script("my_lab_extension.py")  # doctest: +SKIP
    >>> tuple(loaded.erlab.routines)  # doctest: +SKIP
    ('normalize',)
    """
    source_path = pathlib.Path(path).expanduser().resolve()
    try:
        source = source_path.read_bytes()
    except OSError as error:
        raise ExtensionImportError(
            f"Could not read extension source: {error}"
        ) from error
    return _load_script_bytes(
        source,
        source_path,
        module_name=module_name,
        expected_source_hash=expected_source_hash,
    )


def _set_registered_script_backend(
    owner: str, backend: _RegisteredScriptBackend
) -> None:
    """Register one live catalog backend without displacing other managers."""
    with _resolver_lock:
        _registered_script_backends[owner] = backend


def _remove_registered_script_backend(owner: str) -> None:
    """Remove only the catalog backend owned by one closing manager."""
    with _resolver_lock:
        _registered_script_backends.pop(owner, None)


def _resolve_from_registered_backends(
    script_name: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
    *,
    source_hash: str | None = None,
    require_enabled: bool = True,
) -> _RegisteredScriptCapability:
    """Resolve one capability through the single registered-script boundary."""
    with _resolver_lock:
        backends = tuple(_registered_script_backends.values())
    for backend in reversed(backends):
        try:
            return backend.resolve_registered_capability(
                script_name,
                kind,
                capability_id,
                source_hash=source_hash,
                require_enabled=require_enabled,
            )
        except KeyError:
            continue
    raise _RegisteredScriptUnavailable("missing-source")


def _resolve_registered_capability(
    script_name: str,
    source_hash: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> Callable[..., typing.Any]:
    try:
        reference = _resolve_from_registered_backends(
            script_name,
            kind,
            capability_id,
            source_hash=source_hash,
        )
    except _RegisteredScriptUnavailable as error:
        raise ExtensionNotFoundError(
            f"Registered script capability {script_name}:{capability_id} is "
            f"unavailable: {error.status}"
        ) from error
    loaded = _load_script_bytes(
        reference.source_bytes,
        reference.registered_path,
        expected_source_hash=reference.source_hash,
    )
    entries = loaded.erlab.routines if kind == "routine" else loaded.erlab.loaders
    try:
        return entries[capability_id][1]
    except KeyError as error:
        raise ExtensionNotFoundError(
            f"Registered script capability {script_name}:{capability_id} was not found"
        ) from error


def _registered_script_capability_status(
    script_name: str,
    source_hash: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> _CapabilityStatus:
    """Return why a catalog can or cannot run a capability without importing it."""
    try:
        _resolve_from_registered_backends(
            script_name,
            kind,
            capability_id,
            source_hash=source_hash,
        )
    except _RegisteredScriptUnavailable as error:
        return error.status
    return "ready"


def _resolve_registered_script_capability(
    script_name: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> _RegisteredScriptCapability:
    """Return one current local script capability for copied-code generation."""
    try:
        return _resolve_from_registered_backends(
            script_name,
            kind,
            capability_id,
            require_enabled=False,
        )
    except _RegisteredScriptUnavailable as error:
        raise ExtensionNotFoundError(
            f"Current script capability {script_name}:{capability_id} is unavailable: "
            f"{error.status}"
        ) from error


def run_routine(
    data: xr.DataArray,
    *,
    routine_id: str,
    script: os.PathLike[str] | str | None = None,
    registered_script: str | None = None,
    source_hash: str | None = None,
    parameters: Mapping[str, typing.Any] | None = None,
) -> xr.DataArray:
    """Run one decorated routine without manager or Qt knowledge.

    Supply ``script`` for a direct notebook call. Manager replay can instead supply
    ``registered_script`` and ``source_hash`` so the active catalog resolves the
    recorded source.

    Parameters
    ----------
    data
        Input array.
    routine_id
        Capability identifier.
    script
        Python source file. This is optional when a manager catalog resolver exists.
    registered_script
        Registered Python script filename.
    source_hash
        Required source SHA-256 hash for catalog-based replay. You can also use it to
        verify a direct script.
    parameters
        User parameter values.

    Returns
    -------
    xarray.DataArray
        Validated routine result.

    Raises
    ------
    ExtensionNotFoundError
        If the requested script, source, or routine is unavailable.
    ExtensionImportError
        If the script cannot be imported.
    ExtensionSignatureError
        If the script contains an unsupported decorated function.
    ExtensionExecutionError
        If the routine fails or does not return a ``DataArray``.

    Examples
    --------
    >>> import xarray as xr
    >>> from erlab.extensions import run_routine
    >>> data = xr.DataArray([1.0, 2.0])
    >>> result = run_routine(  # doctest: +SKIP
    ...     data,
    ...     script="my_lab_extension.py",
    ...     routine_id="normalize",
    ... )
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    if script is not None and registered_script is not None:
        raise ValueError("script and registered_script cannot both be supplied")
    if script is None:
        if registered_script is None or source_hash is None:
            raise ExtensionNotFoundError(
                "script or both registered_script and source_hash are required"
            )
        func = _resolve_registered_capability(
            registered_script, source_hash, "routine", routine_id
        )
    else:
        loaded = load_script(
            script,
            expected_source_hash=source_hash,
        )
        entry = loaded.erlab.routines.get(routine_id)
        if entry is None:
            raise ExtensionNotFoundError(f"Routine {routine_id!r} was not found")
        func = entry[1]
    try:
        values = _coerce_call_parameters(func, parameters or {})
        result = func(data, **values)
    except Exception as error:
        raise ExtensionExecutionError(
            f"Routine {routine_id!r} failed: {error}"
        ) from error
    if not isinstance(result, xr.DataArray):
        raise ExtensionExecutionError(
            f"Routine {routine_id!r} returned {type(result).__name__}; "
            "expected DataArray"
        )
    return result


def run_loader(
    path: os.PathLike[str] | str,
    *,
    loader_id: str,
    script: os.PathLike[str] | str | None = None,
    registered_script: str | None = None,
    source_hash: str | None = None,
    parameters: Mapping[str, typing.Any] | None = None,
) -> xr.DataArray | xr.Dataset | xr.DataTree:
    """Run one decorated loader without manager or Qt knowledge.

    Parameters
    ----------
    path
        Input file path.
    loader_id
        Capability identifier.
    script
        Direct Python source path.
    registered_script
        Registered Python script filename.
    source_hash
        Required source SHA-256 hash for catalog-based replay. You can also use it to
        verify a direct script.
    parameters
        Loader parameter values.

    Returns
    -------
    xarray.DataArray, xarray.Dataset, or xarray.DataTree
        Validated loader output.

    Raises
    ------
    ExtensionNotFoundError
        If the requested script, source, or loader is unavailable.
    ExtensionImportError
        If the script cannot be imported.
    ExtensionSignatureError
        If the script contains an unsupported decorated function.
    ExtensionExecutionError
        If the loader fails or does not return an xarray object.

    Examples
    --------
    >>> from erlab.extensions import run_loader
    >>> data = run_loader(  # doctest: +SKIP
    ...     "scan.txt",
    ...     script="my_lab_extension.py",
    ...     loader_id="load_scan",
    ... )
    """
    if script is not None and registered_script is not None:
        raise ValueError("script and registered_script cannot both be supplied")
    if script is None:
        if registered_script is None or source_hash is None:
            raise ExtensionNotFoundError(
                "script or both registered_script and source_hash are required"
            )
        func = _resolve_registered_capability(
            registered_script, source_hash, "loader", loader_id
        )
    else:
        loaded = load_script(
            script,
            expected_source_hash=source_hash,
        )
        entry = loaded.erlab.loaders.get(loader_id)
        if entry is None:
            raise ExtensionNotFoundError(f"Loader {loader_id!r} was not found")
        func = entry[1]
    try:
        values = _coerce_call_parameters(func, parameters or {})
        result = func(pathlib.Path(path), **values)
    except Exception as error:
        raise ExtensionExecutionError(
            f"Loader {loader_id!r} failed: {error}"
        ) from error
    if not isinstance(result, (xr.DataArray, xr.Dataset, xr.DataTree)):
        raise ExtensionExecutionError(
            f"Loader {loader_id!r} returned {type(result).__name__}; "
            "expected an xarray object"
        )
    return result
