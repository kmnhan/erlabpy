"""Decorators and source loading for ERLab extensions."""

from __future__ import annotations

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

import xarray as xr

from erlab.extensions._models import (
    EXTENSION_API_VERSION,
    ExtensionExecutionError,
    ExtensionImportError,
    ExtensionNotFoundError,
    ExtensionSignatureError,
    LoadedScript,
    LoaderDescriptor,
    ParameterDescriptor,
    ParameterKind,
    RoutineDescriptor,
    _require_finite_parameter_values,
)

_CAPABILITY_ATTRIBUTE = "__erlab_extension_capability__"
_SourceResolver = Callable[[str, str], os.PathLike[str] | str]
_ScriptCapabilityReferenceResolver = Callable[
    [str, str, str], tuple[os.PathLike[str] | str, str]
]
_CapabilityResolver = Callable[[str, str, str, str], Callable[..., typing.Any]]
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
_CapabilityStatusResolver = Callable[[str, str, str, str], _CapabilityStatus]
_source_resolvers: dict[str, _SourceResolver] = {}
_script_capability_reference_resolvers: dict[
    str, _ScriptCapabilityReferenceResolver
] = {}
_capability_resolvers: dict[str, _CapabilityResolver] = {}
_capability_status_resolvers: dict[str, _CapabilityStatusResolver] = {}
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
        value = value.strip()
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
    for value in vars(module).values():
        metadata = getattr(value, _CAPABILITY_ATTRIBUTE, None)
        if (
            not callable(value)
            or not isinstance(metadata, Mapping)
            or getattr(value, "__module__", None) != module.__name__
        ):
            continue
        descriptor = _descriptor_for(value, metadata)
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
    >>> tuple(loaded.routines)  # doctest: +SKIP
    ('normalize',)
    """
    source_path = pathlib.Path(path).expanduser().resolve()
    try:
        source = source_path.read_bytes()
    except OSError as error:
        raise ExtensionImportError(
            f"Could not read extension source: {error}"
        ) from error
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
        path=source_path,
        source_hash=source_hash,
        module=module,
        routines=routines,
        loaders=loaders,
    )


def _set_source_resolver(owner: str, resolver: _SourceResolver) -> None:
    """Register one live catalog resolver without displacing other managers."""
    with _resolver_lock:
        _source_resolvers[owner] = resolver


def _set_script_capability_reference_resolver(
    owner: str, resolver: _ScriptCapabilityReferenceResolver
) -> None:
    """Register metadata used to generate code for current local scripts."""
    with _resolver_lock:
        _script_capability_reference_resolvers[owner] = resolver


def _set_capability_resolver(owner: str, resolver: _CapabilityResolver) -> None:
    """Register one live capability resolver without displacing other managers."""
    with _resolver_lock:
        _capability_resolvers[owner] = resolver


def _set_capability_status_resolver(
    owner: str, resolver: _CapabilityStatusResolver
) -> None:
    """Register a metadata-only capability state check for one live catalog."""
    with _resolver_lock:
        _capability_status_resolvers[owner] = resolver


def _remove_resolvers(owner: str) -> None:
    """Remove only the resolvers owned by one closing manager."""
    with _resolver_lock:
        _source_resolvers.pop(owner, None)
        _script_capability_reference_resolvers.pop(owner, None)
        _capability_resolvers.pop(owner, None)
        _capability_status_resolvers.pop(owner, None)


def _resolved_capability(
    extension_id: str,
    source_hash: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> Callable[..., typing.Any] | None:
    with _resolver_lock:
        resolvers = tuple(_capability_resolvers.values())
    if not resolvers:
        return None
    for resolver in reversed(resolvers):
        try:
            return resolver(extension_id, source_hash, kind, capability_id)
        except KeyError:
            continue
    return None


def _capability_status(
    extension_id: str,
    source_hash: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> _CapabilityStatus:
    """Return why a catalog can or cannot run a capability without importing it."""
    with _resolver_lock:
        resolvers = tuple(_capability_status_resolvers.values())
    for resolver in reversed(resolvers):
        try:
            return resolver(extension_id, source_hash, kind, capability_id)
        except KeyError:
            continue
    return "missing-source"


def _resolved_source(extension_id: str, source_hash: str) -> os.PathLike[str] | str:
    with _resolver_lock:
        resolvers = tuple(_source_resolvers.values())
    if not resolvers:
        raise ExtensionNotFoundError("No extension catalog resolver is configured")
    for resolver in reversed(resolvers):
        try:
            return resolver(extension_id, source_hash)
        except (FileNotFoundError, KeyError):
            continue
    raise ExtensionNotFoundError(
        f"Extension source {extension_id}:{source_hash} was not found"
    )


def _resolved_script_capability_reference(
    extension_id: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> tuple[os.PathLike[str] | str, str]:
    """Return the current registered path and function for copied script code."""
    with _resolver_lock:
        resolvers = tuple(_script_capability_reference_resolvers.values())
    for resolver in reversed(resolvers):
        try:
            return resolver(extension_id, kind, capability_id)
        except (FileNotFoundError, KeyError):
            continue
    raise ExtensionNotFoundError(
        f"Current script capability {extension_id}:{capability_id} was not found"
    )


def run_routine(
    data: xr.DataArray,
    *,
    routine_id: str,
    script: os.PathLike[str] | str | None = None,
    extension_id: str | None = None,
    source_hash: str | None = None,
    parameters: Mapping[str, typing.Any] | None = None,
) -> xr.DataArray:
    """Run one decorated routine without manager or Qt knowledge.

    Supply ``script`` for a direct notebook call. Manager replay can instead supply
    ``extension_id`` and ``source_hash`` so the active catalog resolves the recorded
    source.

    Parameters
    ----------
    data
        Input array.
    routine_id
        Capability identifier.
    script
        Python source file. This is optional when a manager catalog resolver exists.
    extension_id
        Catalog extension identifier.
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
    source = script
    func: Callable[..., typing.Any] | None = None
    if source is None:
        if extension_id is None or source_hash is None:
            raise ExtensionNotFoundError(
                "script or both extension_id and source_hash are required"
            )
        func = _resolved_capability(extension_id, source_hash, "routine", routine_id)
        if func is None:
            source = _resolved_source(extension_id, source_hash)
    if func is None:
        loaded = load_script(
            typing.cast("os.PathLike[str] | str", source),
            expected_source_hash=source_hash,
        )
        entry = loaded.routines.get(routine_id)
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
    extension_id: str | None = None,
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
    extension_id
        Catalog extension identifier.
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
    source = script
    func: Callable[..., typing.Any] | None = None
    if source is None:
        if extension_id is None or source_hash is None:
            raise ExtensionNotFoundError(
                "script or both extension_id and source_hash are required"
            )
        func = _resolved_capability(extension_id, source_hash, "loader", loader_id)
        if func is None:
            source = _resolved_source(extension_id, source_hash)
    if func is None:
        loaded = load_script(
            typing.cast("os.PathLike[str] | str", source),
            expected_source_hash=source_hash,
        )
        entry = loaded.loaders.get(loader_id)
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
