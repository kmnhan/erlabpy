"""Execution and trust handling for ImageTool provenance replay."""

from __future__ import annotations

import builtins
import importlib
import pathlib
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

import erlab
from erlab.extensions._api import _registered_script_capability_status
from erlab.interactive._code_trust import execution_capability_allows
from erlab.interactive.imagetool._load_source import (
    _deserialize_loader_kwargs,
    _registered_local_callable_loader,
)
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
    _reserved_names_from_spec,
    _validate_script_provenance,
    compile_replay_graph,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileLoadSourceStatus,
    ScriptInput,
    ToolProvenanceSpec,
    compose_full_provenance,
    has_file_load_source,
    iter_operation_refs,
    parse_tool_provenance_spec,
)
from erlab.interactive.imagetool._provenance._trust import (
    provenance_replay_graph_code_trust_entries,
    provenance_replay_node_code_trust_entries,
    provenance_requires_code_trust,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from erlab.extensions._api import _CapabilityStatus
    from erlab.interactive.imagetool._provenance._operations import (
        ExtensionRoutineOperation,
    )

    _CapabilityStatusResolver = Callable[
        [str, str, typing.Literal["routine", "loader"], str], _CapabilityStatus
    ]


ExecutionAuthorizer = Callable[[tuple[typing.Any, ...]], object | None]


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


def _load_file_source_object(
    load_source: FileLoadSource,
    *,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
) -> typing.Any:
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
        func = _registered_local_callable_loader(call.target)
        if func is None:
            raise ValueError(
                f"Callable file loader {call.target!r} is not registered locally"
            )

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
    authorization: object | None = None,
    authorize: ExecutionAuthorizer | None = None,
    _on_value: Callable[[str, xr.DataArray], None] | None = None,
) -> xr.DataArray:
    # Replay runs from manager actions; avoid optional native reduction accelerators
    # that can crash PySide6/Python 3.14 while Qt threads are alive.
    if authorization is None:
        authorization = _authorize_replay_graph(graph, authorize)
    with xr.set_options(use_numbagg=False):
        return _execute_replay_graph(
            graph,
            cache=cache,
            extension_executor=extension_executor,
            extension_loader_executor=extension_loader_executor,
            authorization=authorization,
            on_value=_on_value,
        )


def _execute_replay_graph(
    graph: ReplayGraph,
    *,
    cache: dict[str, xr.DataArray] | None = None,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
    authorization: object | None = None,
    on_value: Callable[[str, xr.DataArray], None] | None = None,
) -> xr.DataArray:
    replay_cache = {} if cache is None else cache
    values: dict[str, xr.DataArray] = {}

    for node_index, node in enumerate(graph.nodes):
        if node.cacheable and node.key in replay_cache:
            values[node.key] = replay_cache[node.key].copy(deep=False)
            if on_value is not None:
                on_value(node.key, values[node.key])
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
            else:
                entries = provenance_replay_node_code_trust_entries(
                    node,
                    location_prefix=f"runtime/nodes/{node_index}",
                )
                if not execution_capability_allows(authorization, entries):
                    raise ReplayGraphError(
                        "Recorded provenance contains Python content that is not "
                        "trusted"
                    )
                input_bindings = node.payload.get("input_bindings", ())
                if input_bindings:
                    data = operation.apply_with_inputs(
                        values[node.parents[0]],
                        {slot: values[input_key] for slot, input_key in input_bindings},
                        authorization=authorization,
                    )
                elif node.payload.get("legacy_parent_context", False):
                    data = operation._apply_schema_v2(
                        values[node.parents[0]],
                        parent_data=values[node.parents[1]],
                        authorization=authorization,
                    )
                elif entries:
                    data = operation.apply(
                        values[node.parents[0]],
                        authorization=authorization,
                    )
                else:
                    data = operation.apply(values[node.parents[0]])
        elif node.kind == "script":
            entries = provenance_replay_node_code_trust_entries(
                node,
                location_prefix=f"runtime/nodes/{node_index}",
            )
            if not execution_capability_allows(authorization, entries):
                raise ReplayGraphError("Recorded Python code is not trusted")
            codes = typing.cast("tuple[str, ...]", node.payload["codes"])
            compiled_codes = tuple(
                compile(
                    code, "<ImageTool script provenance>", "exec", dont_inherit=True
                )
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
        if on_value is not None:
            on_value(node.key, data)

    if graph.output_key is None:
        raise ReplayGraphError("Replay graph has no output")
    output = values[graph.output_key]
    if any(
        node.kind == "live_input" and _shares_array_memory(output, values[node.key])
        for node in graph.nodes
    ):
        return output.copy(deep=True)
    return output


def _authorize_replay_graph(
    graph: ReplayGraph,
    authorize: ExecutionAuthorizer | None,
) -> object | None:
    """Authorize the exact compiled graph once and return its opaque capability."""
    entries = provenance_replay_graph_code_trust_entries(
        graph,
        location_prefix="runtime",
    )
    if not entries or authorize is None:
        return None
    return authorize(entries)


def replay_file_provenance(
    spec: typing.Any,
    *,
    cache: dict[str, xr.DataArray] | None = None,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
    authorize: ExecutionAuthorizer | None = None,
) -> xr.DataArray:
    """Replay structured file provenance without executing generated Python."""
    try:
        graph = compile_replay_graph(
            spec,
            trusted_user_code=True,
            structured_file_replay=True,
        )
        return execute_replay_graph(
            graph,
            cache=cache,
            extension_executor=extension_executor,
            extension_loader_executor=extension_loader_executor,
            authorize=authorize,
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
    if (
        replay_call.kind == "callable"
        and _registered_local_callable_loader(replay_call.target) is None
    ):
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
class _ScriptReplayValidation:
    strict_replayable: bool = False
    permissive_replayable: bool = False


def _require_file_load_source(load_source: FileLoadSource) -> None:
    status = _file_load_source_status(load_source)
    if status == "loadable":
        return
    if status == "missing-file":
        message = "The recorded source file is not available"
    elif status == "missing-loader":
        message = "The recorded source loader is not available"
    else:
        message = "The recorded extension loader is not available"
    raise ReplayGraphError(message)


def _validate_replay_graph_file_sources(graph: ReplayGraph) -> None:
    """Require every reachable file node to have an available loader."""
    for node in graph.nodes:
        if node.kind == "file_load":
            _require_file_load_source(node.payload["load_source"])


def _script_provenance_validates(
    spec: ToolProvenanceSpec,
    *,
    external_input_names: set[str] | None = None,
    strict_replay_code: bool,
) -> bool:
    try:
        _validate_script_provenance(
            spec,
            external_input_names=external_input_names,
            strict_replay_code=strict_replay_code,
        )
    except (ReplayGraphError, TypeError, ValueError):
        return False
    return True


def _analyze_script_replay_validation(
    spec: ToolProvenanceSpec,
    external_input_names: set[str] | None,
    live_input_resolver: LiveInputResolver | None,
    input_provenance_resolver: InputProvenanceResolver,
) -> _ScriptReplayValidation:
    if spec.kind != "script":
        return _ScriptReplayValidation()
    _reserved_names_from_spec(
        spec,
        input_provenance_resolver,
        external_input_names=external_input_names or (),
        live_input_resolver=live_input_resolver,
    )
    spec = spec.as_bound_script()
    strict_replayable = _script_provenance_validates(
        spec,
        external_input_names=external_input_names,
        strict_replay_code=True,
    )
    permissive_replayable = strict_replayable or _script_provenance_validates(
        spec,
        external_input_names=external_input_names,
        strict_replay_code=False,
    )
    return _ScriptReplayValidation(
        strict_replayable=strict_replayable,
        permissive_replayable=permissive_replayable,
    )


def _script_replay_validation(
    value: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> _ScriptReplayValidation:
    try:
        spec = parse_tool_provenance_spec(value)
        if spec is None:
            return _ScriptReplayValidation()
        return _analyze_script_replay_validation(
            spec,
            external_input_names,
            _memoized_live_input_resolver(live_input_resolver),
            _memoized_input_provenance_resolver(),
        )
    except (ReplayGraphError, TypeError, ValueError):
        return _ScriptReplayValidation()


def _can_reload_provenance(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    *,
    allow_code: bool,
    extension_status_resolver: _CapabilityStatusResolver | None = None,
) -> bool:
    """Return whether provenance can replay with the selected extension scope."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return False
    if not allow_code and provenance_requires_code_trust(spec):
        return False
    if spec.kind not in {"file", "script"}:
        return False
    if (spec.kind == "file" or has_file_load_source(spec)) and file_load_source_status(
        spec, extension_status_resolver=extension_status_resolver
    ) != "loadable":
        return False
    if spec.kind == "script" and not _script_provenance_validates(
        spec.as_bound_script(),
        strict_replay_code=not allow_code,
    ):
        return False
    capability_status = (
        _registered_script_capability_status
        if extension_status_resolver is None
        else extension_status_resolver
    )
    checked_specs = [
        spec,
        *(
            source
            for item in spec.replay_inputs
            if item.fallback_role == "raw"
            and (source := item.parsed_source_spec()) is not None
        ),
    ]
    for _ref, operation in (
        entry for checked in checked_specs for entry in iter_operation_refs(checked)
    ):
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
    for script_input in spec.replay_inputs:
        input_spec = script_input.parsed_provenance_spec()
        if not _can_reload_provenance(
            input_spec,
            allow_code=allow_code,
            extension_status_resolver=extension_status_resolver,
        ):
            return False
    return True


def can_reload_without_trust(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    *,
    extension_status_resolver: _CapabilityStatusResolver | None = None,
) -> bool:
    """Return whether recorded provenance can replay without trusted user code."""
    return _can_reload_provenance(
        value,
        allow_code=False,
        extension_status_resolver=extension_status_resolver,
    )


def can_reload_with_trusted_code(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    *,
    extension_status_resolver: _CapabilityStatusResolver | None = None,
) -> bool:
    """Return whether trusted recorded provenance can rebuild its data."""
    return _can_reload_provenance(
        value,
        allow_code=True,
        extension_status_resolver=extension_status_resolver,
    )


def script_provenance_replayable(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    allow_code: bool = False,
    live_input_resolver: LiveInputResolver | None = None,
) -> bool:
    """Return whether a script can replay with the supplied input namespace."""
    validation = _script_replay_validation(
        spec,
        external_input_names=external_input_names,
        live_input_resolver=live_input_resolver,
    )
    return (
        validation.permissive_replayable if allow_code else validation.strict_replayable
    )


def replay_script_provenance(
    spec: typing.Any,
    inputs: Mapping[str, xr.DataArray],
    *,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
    authorize: ExecutionAuthorizer | None = None,
) -> xr.DataArray:
    """Execute script provenance from already resolved input arrays."""
    try:
        graph = compile_replay_graph(
            spec,
            external_inputs=inputs,
            trusted_user_code=True,
            structured_file_replay=True,
        )
        return execute_replay_graph(
            graph,
            extension_executor=extension_executor,
            extension_loader_executor=extension_loader_executor,
            authorize=authorize,
        )
    except ReplayGraphError as exc:
        if "non-replayable" in str(exc):
            raise ValueError(str(exc)) from exc
        raise TypeError(str(exc)) from exc


def _execute_rebuilt_graph(
    graph: ReplayGraph,
    spec: ToolProvenanceSpec,
    *,
    cache: dict[str, xr.DataArray] | None,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray] | None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None,
    authorize: ExecutionAuthorizer | None,
    capture: Callable[[xr.DataArray, ScriptInput], ScriptInput] | None,
) -> tuple[xr.DataArray, ToolProvenanceSpec, dict[int, xr.DataArray]]:
    boundaries = {
        id(boundary.original): boundary for boundary in graph.input_boundaries
    }
    refreshed: dict[int, ScriptInput] = {}
    resolved_data: dict[int, xr.DataArray] = {}
    events: dict[str, list[tuple[typing.Any, ToolProvenanceSpec | None]]] = {}
    for boundary in graph.input_boundaries:
        events.setdefault(boundary.raw_key, []).append((boundary, None))
        for key, source in boundary.sources:
            events.setdefault(key, []).append((boundary, source))

    def rebuilt(current: ToolProvenanceSpec | None) -> ToolProvenanceSpec | None:
        if current is None:
            return None
        completed: dict[int, ToolProvenanceSpec] = {}
        active: set[int] = set()
        pending = [(current, False)]
        while pending:
            owner, leaving = pending.pop()
            identity = id(owner)
            if leaving:
                inputs = []
                for item in owner.replay_inputs:
                    replacement = refreshed.get(id(item))
                    boundary = boundaries.get(id(item))
                    if replacement is None and boundary is not None:
                        producer = boundary.provenance_spec
                        nested = None if producer is None else completed[id(producer)]
                        replacement = boundary.refreshed.model_copy(
                            update={
                                "provenance_spec": None
                                if nested is None
                                else nested.model_dump(mode="json")
                            }
                        )
                    inputs.append(item if replacement is None else replacement)
                completed[identity] = owner.with_replay_inputs(inputs)
                active.remove(identity)
                continue
            if identity in completed:
                continue
            if identity in active:
                raise ReplayGraphError("Recorded provenance contains an input cycle")
            active.add(identity)
            pending.append((owner, True))
            for item in reversed(owner.replay_inputs):
                boundary = boundaries.get(id(item))
                if (
                    id(item) not in refreshed
                    and boundary is not None
                    and boundary.provenance_spec is not None
                ):
                    pending.append((boundary.provenance_spec, False))
        return completed[id(current)]

    def on_value(key: str, data: xr.DataArray) -> None:
        for boundary, source in events.get(key, ()):
            identity = id(boundary.original)
            if source is None:
                producer = rebuilt(boundary.provenance_spec)
                item = boundary.refreshed
                if boundary.capture_raw or boundary.provenance_spec is not None:
                    item = item.model_copy(
                        update={
                            "provenance_spec": None
                            if producer is None
                            else producer.model_dump(mode="json")
                        }
                    )
                if boundary.value_role == "resolved":
                    if boundary.capture_resolved and capture is not None:
                        selected = item.model_copy(
                            update={
                                "revision_id": item.resolved_revision_id,
                                "source_spec": None,
                                "fallback_role": "raw",
                            }
                        )
                        selected = capture(data, selected)
                        item = item.model_copy(
                            update={"resolved_revision_id": selected.revision_id}
                        )
                    refreshed[identity] = item
                    resolved_data[identity] = data.copy(deep=True)
                    continue
                item = item.model_copy(update={"resolved_revision_id": None})
                if boundary.capture_raw and capture is not None:
                    item = capture(data, item)
                if not boundary.sources:
                    item = item.model_copy(
                        update={"resolved_revision_id": item.revision_id}
                    )
                    resolved_data[identity] = data.copy(deep=True)
                refreshed[identity] = item
            else:
                item = refreshed[identity]
                previous = item.resolved_revision_id or item.revision_id
                selected = item
                if capture is not None:
                    producer = compose_full_provenance(
                        item.parsed_provenance_spec(), source
                    )
                    selected = item.model_copy(
                        update={
                            "revision_id": previous,
                            "resolved_revision_id": None,
                            "source_spec": None,
                            "fallback_role": "raw",
                            "provenance_spec": None
                            if producer is None
                            else producer.model_dump(mode="json"),
                        }
                    )
                    selected = capture(data, selected)
                refreshed[identity] = item.model_copy(
                    update={
                        "resolved_revision_id": selected.revision_id
                        if capture is not None
                        else None
                    }
                )
                resolved_data[identity] = data.copy(deep=True)

    output = execute_replay_graph(
        graph,
        cache=cache,
        extension_executor=extension_executor,
        extension_loader_executor=extension_loader_executor,
        authorize=authorize,
        _on_value=on_value,
    )
    result = rebuilt(spec)
    if result is None:
        raise ReplayGraphError("Replay did not preserve its provenance")
    return output, result, resolved_data


def rebuild_script_provenance(
    spec: typing.Any,
    *,
    live_input_resolver: LiveInputResolver | None = None,
    cache: dict[str, xr.DataArray] | None = None,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
    authorize: ExecutionAuthorizer | None = None,
    _input_provenance_resolver: InputProvenanceResolver | None = None,
    _capture_input_revision: Callable[[xr.DataArray, ScriptInput], ScriptInput]
    | None = None,
) -> tuple[xr.DataArray, typing.Any]:
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        raise ReplayGraphError("Selected provenance is not script-derived")
    graph = compile_replay_graph(
        parsed,
        live_input_resolver=live_input_resolver,
        trusted_user_code=True,
        structured_file_replay=True,
        _input_provenance_resolver=_input_provenance_resolver,
    )
    _validate_replay_graph_file_sources(graph)
    data, rebuilt, _resolved = _execute_rebuilt_graph(
        graph,
        parsed,
        cache=cache,
        extension_executor=extension_executor,
        extension_loader_executor=extension_loader_executor,
        authorize=authorize,
        capture=_capture_input_revision,
    )
    return data, rebuilt


def rebuild_script_inputs(
    script_inputs: Sequence[ScriptInput],
    *,
    live_input_resolver: LiveInputResolver | None = None,
    cache: dict[str, xr.DataArray] | None = None,
    allow_recorded: bool = True,
    extension_executor: Callable[[typing.Any, xr.DataArray], xr.DataArray]
    | None = None,
    extension_loader_executor: Callable[[FileLoadSource], typing.Any] | None = None,
    authorize: ExecutionAuthorizer | None = None,
    _input_provenance_resolver: InputProvenanceResolver | None = None,
    _capture_input_revision: Callable[[xr.DataArray, ScriptInput], ScriptInput]
    | None = None,
) -> tuple[dict[str, xr.DataArray], tuple[ScriptInput, ...]]:
    """Resolve named inputs in one graph and stage their exact checkpoints."""
    if not script_inputs:
        return {}, ()
    names = tuple(item.name for item in script_inputs)
    if len(set(names)) != len(names):
        raise ReplayGraphError("Duplicate named provenance input")
    resolve_live = _memoized_live_input_resolver(live_input_resolver)
    if not allow_recorded:
        for item in script_inputs:
            if resolve_live is None or resolve_live(item) is None:
                raise ReplayGraphError(f"{item.label} is not available in this Manager")
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Resolve source inputs",
        active_name=names[0],
        primary_input=names[0],
        script_inputs=tuple(script_inputs),
    )
    graph = compile_replay_graph(
        spec,
        live_input_resolver=resolve_live,
        trusted_user_code=True,
        structured_file_replay=True,
        _input_provenance_resolver=_input_provenance_resolver,
    )
    _validate_replay_graph_file_sources(graph)
    _data, rebuilt, resolved = _execute_rebuilt_graph(
        graph,
        spec,
        cache=cache,
        extension_executor=extension_executor,
        extension_loader_executor=extension_loader_executor,
        authorize=authorize,
        capture=_capture_input_revision,
    )
    return (
        {item.name: resolved[id(item)] for item in script_inputs},
        rebuilt.script_inputs,
    )
