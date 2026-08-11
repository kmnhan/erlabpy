"""Decorators and source loading for ERLab extensions."""

from __future__ import annotations

import enum
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import os
import pathlib
import sys
import threading
import types
import typing
import uuid
from collections.abc import Callable, Mapping

import xarray as xr

from erlab.extensions._entry_points import (
    _entry_point_revision,
    _EntryPointRevisionError,
    _load_entry_point_value,
)
from erlab.extensions._models import (
    EXTENSION_API_VERSION,
    ExtensionExecutionError,
    ExtensionImportError,
    ExtensionNotFoundError,
    ExtensionSignatureError,
    LoadedEntryPoint,
    LoadedScript,
    LoaderDescriptor,
    ParameterDescriptor,
    ParameterKind,
    RoutineDescriptor,
    _require_finite_parameter_values,
)

_CAPABILITY_ATTRIBUTE = "__erlab_extension_capability__"
_RevisionResolver = Callable[[str, str], os.PathLike[str] | str]
_CapabilityResolver = Callable[[str, str, str, str], Callable[..., typing.Any]]
_CapabilityStatus = typing.Literal[
    "ready",
    "disabled",
    "approval-required",
    "missing-revision",
    "missing-capability",
    "hash-mismatch",
    "unsupported-api",
    "import-failed",
]
_CapabilityStatusResolver = Callable[[str, str, str, str], _CapabilityStatus]
_revision_resolvers: dict[str, _RevisionResolver] = {}
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
    extensions: tuple[str, ...] = (),
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
        Filename extensions accepted by the loader. Include the leading dot.

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
    normalized_extensions = tuple(
        value if value.startswith(".") else f".{value}" for value in extensions
    )

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
                extensions=normalized_extensions,
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
    expected_revision: str | None = None,
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
    expected_revision
        Required SHA-256 source hash. A mismatch stops the import.

    Returns
    -------
    LoadedScript
        Imported module, source revision, and validated capabilities.

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
    revision = hashlib.sha256(source).hexdigest()
    if expected_revision is not None and revision != expected_revision:
        raise ExtensionImportError(
            f"Extension source hash {revision} does not match expected revision "
            f"{expected_revision}"
        )
    import_name = (
        module_name
        if module_name is not None
        else f"_erlab_extension_{revision[:12]}_{uuid.uuid4().hex}"
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
        revision=revision,
        module=module,
        routines=routines,
        loaders=loaders,
    )


def load_entry_point(
    group: str,
    name: str,
    *,
    expected_revision: str,
) -> LoadedEntryPoint:
    """Import one installed extension package at an exact revision.

    Entry-point and distribution metadata are inspected before package code is
    imported. Editable package revisions include a fingerprint of their Python
    sources.

    Parameters
    ----------
    group
        Entry-point group, such as ``"erlab.extensions"``.
    name
        Entry-point name declared by the package.
    expected_revision
        Required package revision hash.

    Returns
    -------
    LoadedEntryPoint
        Imported entry-point object with direct access to its callables.

    Raises
    ------
    ExtensionNotFoundError
        If no matching entry point or revision is installed.
    ExtensionImportError
        If package metadata is invalid or package code cannot be imported.

    Examples
    --------
    The returned object exposes functions from a module entry point directly.

    >>> from erlab.extensions import load_entry_point
    >>> extension = load_entry_point(  # doctest: +SKIP
    ...     "erlab.extensions",
    ...     "my_lab",
    ...     expected_revision="0a12...",
    ... )
    >>> result = extension.normalize(data)  # doctest: +SKIP
    """
    matching = importlib.metadata.entry_points().select(group=group, name=name)
    if not matching:
        raise ExtensionNotFoundError(
            f"Extension entry point {group}:{name} is not installed"
        )
    for entry_point in matching:
        try:
            revision = _entry_point_revision(entry_point)
        except _EntryPointRevisionError as error:
            raise ExtensionImportError(
                f"Could not inspect extension entry point {group}:{name}: {error}"
            ) from error
        if revision != expected_revision:
            continue
        try:
            value = _load_entry_point_value(entry_point, expected_revision)
        except Exception as error:
            raise ExtensionImportError(
                f"Could not import extension entry point {group}:{name}: {error}"
            ) from error
        loader_methods: dict[str | None, Callable[..., typing.Any]] = {}
        if group == "erlab.io.loaders":
            from erlab.io.dataloader import LoaderBase

            if isinstance(value, type) and issubclass(value, LoaderBase):
                value = value()
            if not isinstance(value, LoaderBase):
                raise ExtensionImportError(
                    f"Extension entry point {group}:{name} does not provide LoaderBase"
                )
            callables = {}
            loader_methods[None] = value.load
            for method, _defaults in value.file_dialog_methods.values():
                reference = _loader_method_reference(value, method)
                previous = loader_methods.get(reference)
                if previous is not None and previous != method:
                    raise ExtensionImportError(
                        f"Extension entry point {group}:{name} declares multiple "
                        f"loader methods as {reference!r}"
                    )
                loader_methods[reference] = method
        elif isinstance(value, types.ModuleType):
            routines, loaders = _module_capabilities(value)
            callables = {
                descriptor.function_name: func for descriptor, func in routines.values()
            } | {
                descriptor.function_name: func for descriptor, func in loaders.values()
            }
        elif (
            callable(value) and getattr(value, _CAPABILITY_ATTRIBUTE, None) is not None
        ):
            descriptor = _descriptor_for(value, getattr(value, _CAPABILITY_ATTRIBUTE))
            callables = {descriptor.function_name: value}
        else:
            callables = {}
        return LoadedEntryPoint(
            group=group,
            name=name,
            revision=revision,
            value=value,
            callables=callables,
            loader_methods=loader_methods,
        )
    raise ExtensionNotFoundError(
        f"Extension entry point {group}:{name} does not match revision "
        f"{expected_revision}"
    )


def _set_revision_resolver(owner: str, resolver: _RevisionResolver) -> None:
    """Register one live catalog resolver without displacing other managers."""
    with _resolver_lock:
        _revision_resolvers[owner] = resolver


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
        _revision_resolvers.pop(owner, None)
        _capability_resolvers.pop(owner, None)
        _capability_status_resolvers.pop(owner, None)


def _resolved_capability(
    extension_id: str,
    revision: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> Callable[..., typing.Any] | None:
    with _resolver_lock:
        resolvers = tuple(_capability_resolvers.values())
    if not resolvers:
        return None
    for resolver in reversed(resolvers):
        try:
            return resolver(extension_id, revision, kind, capability_id)
        except KeyError:
            continue
    return None


def _capability_status(
    extension_id: str,
    revision: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> _CapabilityStatus:
    """Return why a catalog can or cannot run a capability without importing it."""
    with _resolver_lock:
        resolvers = tuple(_capability_status_resolvers.values())
    for resolver in reversed(resolvers):
        try:
            return resolver(extension_id, revision, kind, capability_id)
        except KeyError:
            continue
    return "missing-revision"


def _resolved_revision(extension_id: str, revision: str) -> os.PathLike[str] | str:
    with _resolver_lock:
        resolvers = tuple(_revision_resolvers.values())
    if not resolvers:
        raise ExtensionNotFoundError("No extension catalog resolver is configured")
    for resolver in reversed(resolvers):
        try:
            return resolver(extension_id, revision)
        except (FileNotFoundError, KeyError):
            continue
    raise ExtensionNotFoundError(
        f"Extension revision {extension_id}:{revision} was not found"
    )


def _loader_method_reference(loader: object, method: Callable) -> str | None:
    """Return the persisted reference for one LoaderBase dialog callable."""
    if getattr(method, "__self__", None) is loader:
        name = getattr(method, "__name__", None)
        if not isinstance(name, str) or not callable(getattr(loader, name, None)):
            raise TypeError("Loader file-dialog methods must have a stable name")
        return None if name == "load" else name
    module = getattr(method, "__module__", None)
    qualname = getattr(method, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not isinstance(qualname, str)
        or "<locals>" in qualname
    ):
        raise TypeError("Loader file-dialog callables must be importable functions")
    return f"{module}.{qualname}"


def _resolve_loader_method(
    func: Callable[..., typing.Any], method: str | None
) -> Callable[..., typing.Any]:
    """Resolve a persisted ``LoaderBase`` method or package callable."""
    if method is None:
        return func
    owner = getattr(func, "__self__", None)
    candidate = getattr(owner, method, None)
    if callable(candidate):
        return candidate
    parts = method.split(".")
    module: types.ModuleType | None = None
    attr_start = 0
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name is None or not (
                error.name == module_name or module_name.startswith(f"{error.name}.")
            ):
                raise
            continue
        attr_start = index
        break
    if module is None:
        raise ExtensionNotFoundError(f"Loader method {method!r} was not found")
    resolved: typing.Any = module
    try:
        for attr in parts[attr_start:]:
            resolved = getattr(resolved, attr)
    except AttributeError as error:
        raise ExtensionNotFoundError(
            f"Loader method {method!r} was not found"
        ) from error
    if not callable(resolved):
        raise ExtensionNotFoundError(f"Loader method {method!r} is not callable")
    return resolved


def run_routine(
    data: xr.DataArray,
    *,
    routine_id: str,
    script: os.PathLike[str] | str | None = None,
    extension_id: str | None = None,
    revision: str | None = None,
    parameters: Mapping[str, typing.Any] | None = None,
) -> xr.DataArray:
    """Run one decorated routine without manager or Qt knowledge.

    Supply ``script`` for a direct notebook call. Manager-generated calls instead
    supply ``extension_id`` and ``revision`` so the configured catalog resolves the
    immutable source.

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
    revision
        Exact source SHA-256 hash.
    parameters
        User parameter values.

    Returns
    -------
    xarray.DataArray
        Validated routine result.

    Raises
    ------
    ExtensionNotFoundError
        If the requested script, revision, or routine is unavailable.
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
        if extension_id is None or revision is None:
            raise ExtensionNotFoundError(
                "script or both extension_id and revision are required"
            )
        func = _resolved_capability(extension_id, revision, "routine", routine_id)
        if func is None:
            source = _resolved_revision(extension_id, revision)
    if func is None:
        loaded = load_script(
            typing.cast("os.PathLike[str] | str", source), expected_revision=revision
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
    revision: str | None = None,
    method: str | None = None,
    parameters: Mapping[str, typing.Any] | None = None,
) -> (
    xr.DataArray
    | xr.Dataset
    | xr.DataTree
    | list[xr.DataArray | xr.Dataset | xr.DataTree]
):
    """Run one decorated loader with an exact source revision.

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
    revision
        Exact source or package revision.
    method
        Installed ``LoaderBase`` method name or importable package callable. Omit this
        value for the normal ``load`` method.
    parameters
        Loader parameter values.

    Returns
    -------
    xarray.DataArray, xarray.Dataset, xarray.DataTree, or list of xarray objects
        Validated loader output. Installed ``LoaderBase`` methods can return a list.

    Raises
    ------
    ExtensionNotFoundError
        If the requested script, revision, or loader is unavailable.
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
        if extension_id is None or revision is None:
            raise ExtensionNotFoundError(
                "script or both extension_id and revision are required"
            )
        func = _resolved_capability(extension_id, revision, "loader", loader_id)
        if func is None:
            source = _resolved_revision(extension_id, revision)
    if func is None:
        loaded = load_script(
            typing.cast("os.PathLike[str] | str", source), expected_revision=revision
        )
        entry = loaded.loaders.get(loader_id)
        if entry is None:
            raise ExtensionNotFoundError(f"Loader {loader_id!r} was not found")
        func = entry[1]
    func = _resolve_loader_method(func, method)
    decorated = getattr(func, _CAPABILITY_ATTRIBUTE, None) is not None
    try:
        values = (
            _coerce_call_parameters(func, parameters or {})
            if decorated
            else dict(parameters or {})
        )
        result = func(pathlib.Path(path), **values)
    except Exception as error:
        raise ExtensionExecutionError(
            f"Loader {loader_id!r} failed: {error}"
        ) from error
    if not isinstance(result, (xr.DataArray, xr.Dataset, xr.DataTree)) and not (
        not decorated
        and isinstance(result, list)
        and all(
            isinstance(item, (xr.DataArray, xr.Dataset, xr.DataTree)) for item in result
        )
    ):
        raise ExtensionExecutionError(
            f"Loader {loader_id!r} returned {type(result).__name__}; "
            "expected an xarray object or supported LoaderBase list"
        )
    return typing.cast(
        "xr.DataArray | xr.Dataset | xr.DataTree | "
        "list[xr.DataArray | xr.Dataset | xr.DataTree]",
        result,
    )
