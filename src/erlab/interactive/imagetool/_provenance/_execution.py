"""Execution and trust handling for ImageTool provenance replay."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import pathlib
import typing
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

import erlab
from erlab.extensions._api import _registered_script_capability_status
from erlab.interactive.imagetool._provenance._code import (
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    _code_uses_name_any_scope,
    _compile_untrusted_script_replay_code,
    _script_replay_import_names,
)
from erlab.interactive.imagetool._provenance._graph import (
    _REPLAY_ALIASES,
    InputProvenanceResolver,
    LiveInputResolver,
    ReplayGraph,
    ReplayGraphError,
    _memoized_by_identity,
    _memoized_input_provenance_resolver,
    _validate_script_provenance,
    compile_replay_graph,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileLoadSourceStatus,
    ScriptInput,
    ToolProvenanceSpec,
    _script_input_reference_text,
    has_file_load_source,
    iter_operation_refs,
    parse_tool_provenance_spec,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from erlab.extensions._api import _CapabilityStatus
    from erlab.interactive.imagetool._provenance._operations import (
        ExtensionRoutineOperation,
    )

    _CapabilityStatusResolver = Callable[
        [str, str, typing.Literal["routine", "loader"], str], _CapabilityStatus
    ]


_MAX_SCRIPT_REPLAY_DEPTH = 20


def _memoized_live_input_resolver(
    resolver: LiveInputResolver | None,
) -> LiveInputResolver | None:
    """Cache one live-input decision for each exact input object."""
    return None if resolver is None else _memoized_by_identity(resolver)


def _processed_replay_ndim(darr: xr.DataArray) -> int:
    if darr.ndim == 1:
        return 2
    if darr.ndim > 4:
        return len(tuple(size for size in darr.shape if size != 1))
    return darr.ndim


def _reducible_replay_shape(darr: xr.DataArray) -> bool:
    return _processed_replay_ndim(darr) >= 2


def _parse_replay_dataset(ds: xr.Dataset) -> tuple[xr.DataArray, ...]:
    return tuple(
        darr for darr in ds.data_vars.values() if _reducible_replay_shape(darr)
    )


def _parse_replay_input(data: typing.Any) -> list[xr.DataArray]:
    input_cls = data.__class__.__name__
    parsed: typing.Any = data
    if isinstance(data, np.ndarray | xr.DataArray):
        parsed = (data,)
    elif isinstance(data, xr.Dataset):
        parsed = _parse_replay_dataset(data)
    elif isinstance(data, xr.DataTree):
        parsed = tuple(
            darr for leaf in data.leaves for darr in _parse_replay_dataset(leaf.dataset)
        )

    if len(parsed) == 0:
        raise ValueError(f"No valid data for ImageTool found in {input_cls}")
    if not isinstance(next(iter(parsed)), xr.DataArray | np.ndarray):
        raise TypeError(
            f"Unsupported input type {input_cls}. Expected DataArray, Dataset, "
            "DataTree, numpy array, or a list of DataArray or numpy arrays."
        )
    return [
        xr.DataArray(item) if not isinstance(item, xr.DataArray) else item
        for item in parsed
    ]


def _require_replay_dataarray(data: typing.Any) -> xr.DataArray:
    if isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Selected file data must be a DataArray, got {type(data).__name__!r}"
        )
    if not _reducible_replay_shape(data):
        raise ValueError("Selected file data is not valid for ImageTool")
    return data


def _select_replay_input(
    data: typing.Any,
    selection: FileDataSelection,
) -> xr.DataArray:
    selection = _semantic_file_data_selection(data, selection)
    if selection.kind == "dataarray":
        return _require_replay_dataarray(data)
    if selection.kind == "dataset_variable":
        if not isinstance(data, xr.Dataset):
            raise TypeError(
                "Dataset variable file selections require the loader to return "
                "a Dataset"
            )
        try:
            selected = data[selection.value]
        except KeyError as err:
            raise KeyError(
                f"Selected file variable {selection.value!r} was not found"
            ) from err
        return _require_replay_dataarray(selected)
    if selection.kind == "datatree_variable":
        if not isinstance(data, xr.DataTree):
            raise TypeError(
                "DataTree variable selections require the loader to return a DataTree"
            )
        node_path, variable = typing.cast(
            "tuple[str, typing.Hashable]", selection.value
        )
        try:
            selected = data[node_path].dataset[variable]
        except KeyError as err:
            raise KeyError(
                f"Selected file DataTree variable {variable!r} at "
                f"{node_path!r} was not found"
            ) from err
        return _require_replay_dataarray(selected)
    if selection.kind == "sequence_index":
        if isinstance(data, str | bytes) or not isinstance(data, Sequence):
            raise TypeError(
                "Sequence-index file selections require the loader to return a sequence"
            )
        index = typing.cast("int", selection.value)
        try:
            selected = data[index]
        except IndexError as err:
            raise IndexError(
                f"Selected file sequence index {index} is out of range"
            ) from err
        return _require_replay_dataarray(selected)

    raise ValueError(f"Unsupported file data selection kind {selection.kind!r}")


def _semantic_file_data_selection(
    data: typing.Any,
    selection: FileDataSelection,
) -> FileDataSelection:
    """Resolve a legacy parsed index to stable loader-output semantics."""
    if selection.kind != "parsed_index":
        return selection

    index = typing.cast("int", selection.value)
    if isinstance(data, np.ndarray | xr.DataArray):
        if index != 0:
            raise IndexError("Selected file replay index is out of range for 1 array")
        return FileDataSelection(kind="dataarray")
    if isinstance(data, xr.Dataset):
        variables = tuple(
            name
            for name, darr in data.data_vars.items()
            if _reducible_replay_shape(darr)
        )
        if index >= len(variables):
            raise IndexError(
                f"Selected file replay index {index} is out of range for "
                f"{len(variables)} parsed arrays"
            )
        return FileDataSelection(kind="dataset_variable", value=variables[index])
    if isinstance(data, xr.DataTree):
        variables = tuple(
            (str(leaf.path), name)
            for leaf in data.leaves
            for name, darr in leaf.dataset.data_vars.items()
            if _reducible_replay_shape(darr)
        )
        if index >= len(variables):
            raise IndexError(
                f"Selected file replay index {index} is out of range for "
                f"{len(variables)} parsed arrays"
            )
        return FileDataSelection(kind="datatree_variable", value=variables[index])
    if isinstance(data, str | bytes) or not isinstance(data, Sequence):
        _parse_replay_input(data)
        raise TypeError("Unsupported file loader output")
    if index >= len(data):
        raise IndexError(
            f"Selected file replay index {index} is out of range for "
            f"{len(data)} parsed arrays"
        )
    _require_replay_dataarray(data[index])
    return FileDataSelection(kind="sequence_index", value=index)


def _resolve_importable_callable(target: str) -> Callable[..., typing.Any]:
    parts = target.split(".")
    if len(parts) < 2:
        raise ValueError(f"Importable callable target {target!r} must be dotted")

    module = None
    attr_start = 0
    for idx in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:idx])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue
        attr_start = idx
        break
    if module is None:
        raise ModuleNotFoundError(target)

    obj: typing.Any = module
    for attr in parts[attr_start:]:
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"Importable target {target!r} is not callable")
    return typing.cast("Callable[..., typing.Any]", obj)


def _load_file_source_object(
    load_source: FileLoadSource,
    *,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
) -> typing.Any:
    from erlab.interactive.imagetool._load_source import _deserialize_loader_kwargs

    call = load_source.replay_call
    if call is None:
        raise ValueError("File load source does not define replay metadata")
    file_path = pathlib.Path(load_source.path)
    if call.kind == "erlab_loader":
        func = erlab.io.loaders[call.target].load
    elif call.kind == "extension_loader":
        if extension_loader_executor is not None:
            return extension_loader_executor(load_source)
        return erlab.extensions.run_loader(
            file_path,
            registered_script=call.target,
            source_hash=typing.cast("str", call.source_hash),
            loader_id=typing.cast("str", call.capability_id),
            parameters=_deserialize_loader_kwargs(call.kwargs),
        )
    else:
        func = _resolve_importable_callable(call.target)

    return func(file_path, **_deserialize_loader_kwargs(call.kwargs))


def _load_file_source_data(
    load_source: FileLoadSource,
    *,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
) -> xr.DataArray:
    call = load_source.replay_call
    if call is None:
        raise ValueError("File load source does not define replay metadata")
    loaded = _load_file_source_object(
        load_source,
        extension_loader_executor=extension_loader_executor,
    )
    data = _select_replay_input(loaded, call.selection)
    if call.cast_float64:
        data = data.astype(np.float64)
    return data


def _shares_array_memory(first: xr.DataArray, second: xr.DataArray) -> bool:
    try:
        return bool(np.shares_memory(first.data, second.data))
    except (TypeError, ValueError):
        return False


def execute_replay_graph(
    graph: ReplayGraph,
    *,
    cache: dict[str, xr.DataArray] | None = None,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
) -> xr.DataArray:
    # Replay runs from manager actions; avoid optional native reduction accelerators
    # that can crash PySide6/Python 3.14 while Qt threads are alive.
    with xr.set_options(use_numbagg=False):
        return _execute_replay_graph(
            graph,
            cache=cache,
            extension_executor=extension_executor,
            extension_loader_executor=extension_loader_executor,
        )


def _execute_replay_graph(
    graph: ReplayGraph,
    *,
    cache: dict[str, xr.DataArray] | None = None,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
) -> xr.DataArray:
    replay_cache = {} if cache is None else cache
    values: dict[str, xr.DataArray] = {}

    for node in graph.nodes:
        if node.cacheable and node.key in replay_cache:
            values[node.key] = replay_cache[node.key].copy(deep=False)
            continue

        if node.kind == "file_load":
            data = _load_file_source_data(
                node.payload["load_source"],
                extension_loader_executor=extension_loader_executor,
            )
        elif node.kind == "setup":
            continue
        elif node.kind == "live_input":
            data = typing.cast("xr.DataArray", node.payload["data"]).copy(deep=False)
        elif node.kind == "relay":
            data = values[node.parents[0]].copy(deep=False)
        elif node.kind == "source_view":
            parent_data = values[node.parents[0]]
            data = ToolProvenanceSpec._starting_data_for_kind(
                node.payload["source_kind"],
                parent_data,
            )
        elif node.kind == "operation":
            operation = node.payload["operation"]
            if extension_executor is not None and operation.op == "extension_routine":
                data = extension_executor(operation, values[node.parents[0]])
            elif node.payload.get("legacy_parent_context", False):
                data = operation._apply_schema_v2(
                    values[node.parents[0]],
                    parent_data=values[node.parents[1]],
                )
            else:
                data = operation.apply(values[node.parents[0]])
        elif node.kind == "script":
            codes = typing.cast("tuple[str, ...]", node.payload["codes"])
            compiled_codes = tuple(
                compile(code, "<ImageTool script provenance>", "exec")
                if graph.trusted_user_code
                else _compile_untrusted_script_replay_code(code)
                for code in codes
            )
            replay_builtins = (
                vars(builtins)
                if graph.trusted_user_code
                else _SCRIPT_REPLAY_ALLOWED_BUILTINS
            )
            namespace: dict[str, typing.Any] = {
                "__builtins__": replay_builtins,
                "erlab": erlab,
                "np": np,
                "numpy": np,
                "xr": xr,
                "xarray": xr,
                "__erlab_replay_import_erlab": erlab,
                "__erlab_replay_import_numpy": np,
                "__erlab_replay_import_xarray": xr,
            }
            if not graph.trusted_user_code and any(
                "lmfit" in _script_replay_import_names(code) for code in codes
            ):
                namespace["__erlab_replay_import_lmfit"] = importlib.import_module(
                    "lmfit"
                )
            for alias, target in _REPLAY_ALIASES.items():
                if not any(_code_uses_name_any_scope(code, alias) for code in codes):
                    continue
                value: typing.Any = erlab
                for attr in target.split(".")[1:]:
                    value = getattr(value, attr)
                namespace[alias] = value
            for input_name, input_key in typing.cast(
                "tuple[tuple[str, str], ...]", node.payload["bindings"]
            ):
                namespace[input_name] = values[input_key].copy(deep=True)
            for compiled in compiled_codes:
                exec(compiled, namespace, namespace)  # noqa: S102
            active_name = typing.cast("str", node.payload["active_name"])
            if active_name not in namespace:
                raise ReplayGraphError(
                    f"Script provenance did not create active variable {active_name!r}"
                )
            result = namespace[active_name]
            if not isinstance(result, xr.DataArray):
                raise ReplayGraphError(
                    "Script provenance did not produce an xarray.DataArray for "
                    f"{active_name!r}"
                )
            data = result
        else:
            raise ReplayGraphError(f"Unknown replay graph node kind {node.kind!r}")

        if node.cacheable:
            replay_cache[node.key] = data.copy(deep=False)
        values[node.key] = data

    if graph.output_key is None:
        raise ReplayGraphError("Replay graph has no output")
    output = values[graph.output_key]
    if any(
        node.kind == "live_input" and _shares_array_memory(output, values[node.key])
        for node in graph.nodes
    ):
        return output.copy(deep=True)
    return output


def replay_file_provenance(
    spec: typing.Any,
    *,
    cache: dict[str, xr.DataArray] | None = None,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
) -> xr.DataArray:
    """Replay structured file provenance without executing generated Python."""
    try:
        graph = compile_replay_graph(spec, structured_file_replay=True)
        return execute_replay_graph(
            graph,
            cache=cache,
            extension_executor=extension_executor,
            extension_loader_executor=extension_loader_executor,
        )
    except ReplayGraphError as exc:
        raise TypeError("Expected structured file provenance") from exc


def _file_load_source_status(
    load_source: FileLoadSource | None,
    *,
    extension_status_resolver: _CapabilityStatusResolver | None = None,
) -> FileLoadSourceStatus:
    if load_source is None:
        return "no-file-load-source"
    if not pathlib.Path(load_source.path).exists():
        return "missing-file"
    replay_call = load_source.replay_call
    if replay_call is None:
        return "no-replay-call"
    if (
        replay_call.kind == "erlab_loader"
        and replay_call.target not in erlab.io.loaders
    ):
        return "missing-loader"
    if replay_call.kind == "callable":
        try:
            _resolve_importable_callable(replay_call.target)
        except (AttributeError, ModuleNotFoundError, TypeError, ValueError):
            return "missing-loader"
    if replay_call.kind == "extension_loader":
        capability_status = (
            _registered_script_capability_status
            if extension_status_resolver is None
            else extension_status_resolver
        )(
            replay_call.target,
            typing.cast("str", replay_call.source_hash),
            "loader",
            typing.cast("str", replay_call.capability_id),
        )
        if capability_status == "disabled":
            return "extension-disabled"
        if capability_status == "approval-required":
            return "extension-approval-required"
        if capability_status == "missing-source":
            return "extension-missing-source"
        if capability_status == "missing-capability":
            return "extension-missing-capability"
        if capability_status == "hash-mismatch":
            return "extension-hash-mismatch"
        if capability_status == "unsupported-api":
            return "extension-unsupported-api"
        if capability_status == "validation-failed":
            return "extension-validation-failed"
    return "loadable"


def file_load_source_status(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    *,
    extension_status_resolver: _CapabilityStatusResolver | None = None,
) -> FileLoadSourceStatus:
    """Return the current availability of the recorded file-load source.

    A managed ImageTool can supply its own resolver so session-specific extension
    validation failures do not affect other manager instances.
    """
    spec = parse_tool_provenance_spec(value)
    return _file_load_source_status(
        None if spec is None else spec.file_load_source,
        extension_status_resolver=extension_status_resolver,
    )


@dataclass(frozen=True)
class _ReplayCapability:
    replayable: bool = False
    requires_trust: bool = False
    reloadable: bool = False


def _require_file_load_source(load_source: FileLoadSource) -> None:
    status = _file_load_source_status(load_source)
    if status == "loadable":
        return
    if status == "missing-file":
        message = "The recorded source file is not available"
    elif status == "no-replay-call":
        message = "The recorded source does not define replay loader metadata"
    elif status == "missing-loader":
        message = "The recorded source loader is not available"
    else:
        message = "The recorded provenance does not define a file source"
    raise ReplayGraphError(message)


def _validate_replay_graph_file_sources(graph: ReplayGraph) -> None:
    """Require every reachable file node to have an available loader."""
    for node in graph.nodes:
        if node.kind == "file_load":
            _require_file_load_source(node.payload["load_source"])


def _script_validates(
    spec: ToolProvenanceSpec,
    external_input_names: set[str] | None,
    strict: bool,
) -> bool:
    try:
        _validate_script_provenance(
            spec,
            external_input_names=external_input_names,
            strict_replay_code=strict,
        )
    except (ReplayGraphError, TypeError, ValueError):
        return False
    return True


def _analyze_replay_capability(
    spec: ToolProvenanceSpec,
    external_input_names: set[str] | None,
    live_input_resolver: LiveInputResolver | None,
    input_provenance_resolver: InputProvenanceResolver,
    depth: int,
    *,
    check_reloadability: bool = True,
) -> _ReplayCapability:
    if spec.kind == "file":
        return _ReplayCapability(
            reloadable=check_reloadability
            and file_load_source_status(spec) == "loadable"
        )
    if spec.kind != "script":
        return _ReplayCapability()
    if depth > _MAX_SCRIPT_REPLAY_DEPTH:
        raise ReplayGraphError(
            "Nested script provenance exceeded the maximum reload depth"
        )
    child_capabilities: list[_ReplayCapability] = []
    inputs_reloadable = True
    for script_input in spec.script_inputs:
        if script_input.name in (external_input_names or ()) or (
            live_input_resolver is not None
            and live_input_resolver(script_input) is not None
        ):
            inputs_reloadable = False
            continue
        input_spec = input_provenance_resolver(script_input)
        if input_spec is None:
            inputs_reloadable = False
            child_capabilities.append(_ReplayCapability())
            continue
        capability = _analyze_replay_capability(
            input_spec,
            None,
            live_input_resolver,
            input_provenance_resolver,
            depth + int(input_spec.kind == "script"),
            check_reloadability=check_reloadability,
        )
        child_capabilities.append(capability)
        inputs_reloadable &= capability.reloadable

    local_strict = _script_validates(spec, external_input_names, True)
    local_trusted = local_strict or _script_validates(spec, external_input_names, False)
    replayable = local_strict
    requires_trust = (not local_strict and local_trusted) or (
        local_strict
        and any(capability.requires_trust for capability in child_capabilities)
    )
    file_source_reloadable = (
        not check_reloadability
        or spec.file_load_source is None
        or file_load_source_status(spec) == "loadable"
    )
    reloadable = (
        check_reloadability
        and replayable
        and inputs_reloadable
        and file_source_reloadable
    )
    return _ReplayCapability(replayable, requires_trust, reloadable)


def _compile_replay_preflight(
    spec: ToolProvenanceSpec,
    *,
    live_input_resolver: LiveInputResolver | None,
    input_provenance_resolver: InputProvenanceResolver,
) -> ReplayGraph:
    """Compile one reachable replay subtree without executing it."""
    return compile_replay_graph(
        spec,
        live_input_resolver=live_input_resolver,
        trusted_user_code=True,
        _input_provenance_resolver=input_provenance_resolver,
    )


def _require_replay_authorized(
    capability: _ReplayCapability,
    *,
    trusted_user_code: bool,
) -> None:
    if capability.requires_trust:
        authorized = trusted_user_code
    else:
        authorized = capability.replayable
    if authorized:
        return
    raise ReplayGraphError("The recorded operation cannot be replayed automatically")


def _replay_capability(
    value: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> _ReplayCapability:
    try:
        spec = parse_tool_provenance_spec(value)
        if spec is None:
            return _ReplayCapability()
        return _analyze_replay_capability(
            spec,
            external_input_names,
            _memoized_live_input_resolver(live_input_resolver),
            _memoized_input_provenance_resolver(),
            0,
        )
    except (ReplayGraphError, TypeError, ValueError):
        return _ReplayCapability()


def can_reload_without_trust(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    *,
    extension_status_resolver: _CapabilityStatusResolver | None = None,
) -> bool:
    """Return whether provenance can replay with the selected extension scope."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return False
    if spec.kind not in {"file", "script"}:
        return False
    if (spec.kind == "file" or has_file_load_source(spec)) and file_load_source_status(
        spec, extension_status_resolver=extension_status_resolver
    ) != "loadable":
        return False
    if spec.kind == "script" and not script_provenance_replayable(spec):
        return False
    capability_status = (
        _registered_script_capability_status
        if extension_status_resolver is None
        else extension_status_resolver
    )
    for _ref, operation in iter_operation_refs(spec):
        if getattr(operation, "op", None) != "extension_routine":
            continue
        extension_operation = typing.cast("ExtensionRoutineOperation", operation)
        if (
            capability_status(
                extension_operation.script_name,
                extension_operation.source_hash,
                "routine",
                extension_operation.routine_id,
            )
            != "ready"
        ):
            return False
    for script_input in spec.script_inputs:
        input_spec = script_input.parsed_provenance_spec()
        if not can_reload_without_trust(
            input_spec, extension_status_resolver=extension_status_resolver
        ):
            return False
    return True


def script_provenance_replayable(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> bool:
    return _replay_capability(
        spec,
        external_input_names=external_input_names,
        live_input_resolver=live_input_resolver,
    ).replayable


def script_provenance_requires_trust(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> bool:
    return _replay_capability(
        spec,
        external_input_names=external_input_names,
        live_input_resolver=live_input_resolver,
    ).requires_trust


def _reachable_script_trust_payload(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None,
    live_input_resolver: LiveInputResolver | None,
    input_provenance_resolver: InputProvenanceResolver,
) -> dict[str, typing.Any] | None:
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        return None
    operations = []
    for operation in parsed.operations:
        if getattr(operation, "op", None) != "script_code":
            continue
        operations.append(
            {
                "code": getattr(operation, "code", None),
                "copyable": bool(getattr(operation, "copyable", False)),
            }
        )
    inputs = []
    for script_input in parsed.script_inputs:
        if script_input.name in (external_input_names or ()) or (
            live_input_resolver is not None
            and live_input_resolver(script_input) is not None
        ):
            continue
        input_payload = _reachable_script_trust_payload(
            input_provenance_resolver(script_input),
            external_input_names=None,
            live_input_resolver=live_input_resolver,
            input_provenance_resolver=input_provenance_resolver,
        )
        if input_payload is None:
            continue
        inputs.append({"name": script_input.name, "payload": input_payload})
    return {
        "active_name": parsed.active_name,
        "inputs": inputs,
        "operations": operations,
        "seed_code": parsed.seed_code,
    }


def _script_trust_payload(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> dict[str, typing.Any] | None:
    """Return the trust payload for code reachable in one input snapshot."""
    return _reachable_script_trust_payload(
        spec,
        external_input_names=external_input_names,
        live_input_resolver=_memoized_live_input_resolver(live_input_resolver),
        input_provenance_resolver=_memoized_input_provenance_resolver(),
    )


def script_provenance_trust_key(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> str | None:
    payload = _script_trust_payload(
        spec,
        external_input_names=external_input_names,
        live_input_resolver=live_input_resolver,
    )
    if payload is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def replay_script_provenance(
    spec: typing.Any,
    inputs: Mapping[str, xr.DataArray],
    *,
    trusted_user_code: bool = False,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
) -> xr.DataArray:
    """Execute script provenance from already resolved input arrays."""
    try:
        graph = compile_replay_graph(
            spec,
            external_inputs=inputs,
            trusted_user_code=trusted_user_code,
            structured_file_replay=True,
        )
        return execute_replay_graph(
            graph,
            extension_executor=extension_executor,
            extension_loader_executor=extension_loader_executor,
        )
    except ReplayGraphError as exc:
        if "non-replayable" in str(exc):
            raise ValueError(str(exc)) from exc
        raise TypeError(str(exc)) from exc


def rebuild_script_provenance(
    spec: typing.Any,
    *,
    live_input_resolver: LiveInputResolver | None = None,
    cache: dict[str, xr.DataArray] | None = None,
    depth: int = 0,
    trusted_user_code: bool = False,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
    _input_provenance_resolver: InputProvenanceResolver | None = None,
    _preflighted: bool = False,
) -> tuple[xr.DataArray, typing.Any]:
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        raise ReplayGraphError("Selected provenance is not script-derived")
    if depth > _MAX_SCRIPT_REPLAY_DEPTH:
        raise ReplayGraphError(
            "Nested script provenance exceeded the maximum reload depth"
        )
    resolve_live = _memoized_live_input_resolver(live_input_resolver)
    resolve_input_provenance = (
        _memoized_input_provenance_resolver()
        if _input_provenance_resolver is None
        else _input_provenance_resolver
    )
    if not _preflighted:
        capability = _analyze_replay_capability(
            parsed,
            None,
            resolve_live,
            resolve_input_provenance,
            depth,
            check_reloadability=False,
        )
        _require_replay_authorized(
            capability,
            trusted_user_code=trusted_user_code,
        )
        preflight_graph = _compile_replay_preflight(
            parsed,
            live_input_resolver=resolve_live,
            input_provenance_resolver=resolve_input_provenance,
        )
        _validate_replay_graph_file_sources(preflight_graph)

    replay_cache = {} if cache is None else cache
    resolved_inputs, rebuilt_inputs = rebuild_script_inputs(
        parsed.script_inputs,
        live_input_resolver=resolve_live,
        cache=replay_cache,
        trusted_user_code=trusted_user_code,
        depth=depth,
        extension_executor=extension_executor,
        extension_loader_executor=extension_loader_executor,
        _input_provenance_resolver=resolve_input_provenance,
        _preflighted=True,
    )
    rebuilt_spec = parsed.model_copy(update={"script_inputs": rebuilt_inputs})
    graph = compile_replay_graph(
        rebuilt_spec,
        live_input_resolver=lambda script_input: (
            resolved_inputs[script_input.name],
            script_input,
        ),
        trusted_user_code=trusted_user_code,
        structured_file_replay=True,
    )
    return (
        execute_replay_graph(
            graph,
            cache=replay_cache,
            extension_executor=extension_executor,
            extension_loader_executor=extension_loader_executor,
        ),
        rebuilt_spec,
    )


def rebuild_script_inputs(
    script_inputs: Sequence[ScriptInput],
    *,
    live_input_resolver: LiveInputResolver | None = None,
    recorded_input_authorizer: (
        Callable[[ScriptInput, ToolProvenanceSpec], bool] | None
    ) = None,
    cache: dict[str, xr.DataArray] | None = None,
    trusted_user_code: bool = False,
    allow_recorded: bool = True,
    depth: int = 0,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
    _input_provenance_resolver: InputProvenanceResolver | None = None,
    _preflighted: bool = False,
) -> tuple[dict[str, xr.DataArray], tuple[ScriptInput, ...]]:
    """Resolve named inputs and refresh their durable source snapshots.

    Live manager nodes take priority. If ``allow_recorded`` is true and a live node is
    unavailable, the recorded script or file provenance is replayed. The optional
    authorizer runs only for a recorded script fallback that requires trusted execution.
    All inputs are resolved before the caller mutates a ToolWindow.
    """
    replay_cache = {} if cache is None else cache
    resolve_live = _memoized_live_input_resolver(live_input_resolver)
    resolve_input_provenance = (
        _memoized_input_provenance_resolver()
        if _input_provenance_resolver is None
        else _input_provenance_resolver
    )
    input_names = tuple(script_input.name for script_input in script_inputs)
    if len(set(input_names)) != len(input_names):
        raise ReplayGraphError("Duplicate named provenance input")

    planned_inputs: list[
        tuple[
            ScriptInput,
            tuple[xr.DataArray, typing.Any] | ToolProvenanceSpec,
        ]
    ] = []
    for script_input in script_inputs:
        resolved = None if resolve_live is None else resolve_live(script_input)
        if resolved is not None:
            planned_inputs.append((script_input, resolved))
            continue
        if not allow_recorded:
            raise ReplayGraphError(
                f"{script_input.label} is not available in this Manager"
            )

        input_spec = resolve_input_provenance(script_input)
        if input_spec is None:
            raise ReplayGraphError(
                f"{_script_input_reference_text(script_input)} is not open and "
                "does not contain recorded source provenance."
            )
        if input_spec.kind not in {"file", "script"}:
            raise ReplayGraphError(
                f"{_script_input_reference_text(script_input)} is not open and "
                "does not contain reloadable script or file provenance."
            )
        planned_inputs.append((script_input, input_spec))

    capabilities: dict[str, _ReplayCapability] = {}
    if not _preflighted:
        for script_input, resolved_or_spec in planned_inputs:
            if (
                not isinstance(resolved_or_spec, ToolProvenanceSpec)
                or resolved_or_spec.kind != "script"
            ):
                continue
            capabilities[script_input.name] = _analyze_replay_capability(
                resolved_or_spec,
                None,
                resolve_live,
                resolve_input_provenance,
                depth + 1,
                check_reloadability=False,
            )

    trusted_inputs: list[bool] = []
    for script_input, resolved_or_spec in planned_inputs:
        input_trusted_user_code = trusted_user_code
        if (
            isinstance(resolved_or_spec, ToolProvenanceSpec)
            and resolved_or_spec.kind == "script"
            and (capability := capabilities.get(script_input.name)) is not None
        ):
            if (
                capability.requires_trust
                and not input_trusted_user_code
                and recorded_input_authorizer is not None
            ):
                input_trusted_user_code = recorded_input_authorizer(
                    script_input,
                    resolved_or_spec,
                )
            _require_replay_authorized(
                capability,
                trusted_user_code=input_trusted_user_code,
            )
        trusted_inputs.append(input_trusted_user_code)

    preflight_graphs: dict[str, ReplayGraph] = {}
    if not _preflighted:
        for script_input, resolved_or_spec in planned_inputs:
            if not isinstance(resolved_or_spec, ToolProvenanceSpec):
                continue
            preflight_graphs[script_input.name] = _compile_replay_preflight(
                resolved_or_spec,
                live_input_resolver=resolve_live,
                input_provenance_resolver=resolve_input_provenance,
            )
        for graph in preflight_graphs.values():
            _validate_replay_graph_file_sources(graph)

    data_by_name: dict[str, xr.DataArray] = {}
    refreshed_inputs: list[ScriptInput] = []
    for (script_input, resolved_or_spec), input_trusted_user_code in zip(
        planned_inputs, trusted_inputs, strict=True
    ):
        if not isinstance(resolved_or_spec, ToolProvenanceSpec):
            data, refreshed_input = resolved_or_spec
            data_by_name[script_input.name] = data
            refreshed_inputs.append(refreshed_input)
            continue

        input_spec = resolved_or_spec
        if input_spec.kind == "file":
            graph = preflight_graphs.get(script_input.name)
            if graph is None:
                graph = compile_replay_graph(
                    input_spec,
                    _input_provenance_resolver=resolve_input_provenance,
                )
            data = execute_replay_graph(
                graph,
                cache=replay_cache,
                extension_executor=extension_executor,
                extension_loader_executor=extension_loader_executor,
            )
            rebuilt_spec = input_spec
        elif input_spec.kind == "script":
            data, rebuilt_spec = rebuild_script_provenance(
                input_spec,
                live_input_resolver=resolve_live,
                cache=replay_cache,
                depth=depth + 1,
                trusted_user_code=input_trusted_user_code,
                extension_executor=extension_executor,
                extension_loader_executor=extension_loader_executor,
                _input_provenance_resolver=resolve_input_provenance,
                _preflighted=True,
            )
        data_by_name[script_input.name] = data
        refreshed_inputs.append(
            script_input.model_copy(
                update={
                    "node_uid": None,
                    "node_snapshot_token": None,
                    "provenance_spec": rebuilt_spec.model_dump(mode="json"),
                }
            )
        )

    return data_by_name, tuple(refreshed_inputs)
