"""Execution and trust handling for ImageTool provenance replay."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import pathlib
import typing
from collections.abc import Sequence

import numpy as np
import xarray as xr

import erlab
from erlab.interactive.imagetool._provenance._code import (
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    _code_uses_name_any_scope,
    _compile_untrusted_script_replay_code,
    _script_replay_import_names,
)
from erlab.interactive.imagetool._provenance._graph import (
    _REPLAY_ALIASES,
    LiveInputResolver,
    ReplayGraph,
    ReplayGraphError,
    _validate_script_provenance,
    compile_replay_graph,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileLoadSourceStatus,
    ToolProvenanceSpec,
    _script_input_reference_text,
    has_file_load_source,
    parse_tool_provenance_spec,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping


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


def _load_file_source_object(load_source: FileLoadSource) -> typing.Any:
    call = load_source.replay_call
    if call is None:
        raise ValueError("File load source does not define replay metadata")
    file_path = pathlib.Path(load_source.path)
    if call.kind == "erlab_loader":
        func = erlab.io.loaders[call.target].load
    else:
        func = _resolve_importable_callable(call.target)

    return func(file_path, **dict(call.kwargs))


def _load_file_source_data(load_source: FileLoadSource) -> xr.DataArray:
    call = load_source.replay_call
    if call is None:
        raise ValueError("File load source does not define replay metadata")
    loaded = _load_file_source_object(load_source)
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
) -> xr.DataArray:
    # Replay runs from manager actions; avoid optional native reduction accelerators
    # that can crash PySide6/Python 3.14 while Qt threads are alive.
    with xr.set_options(use_numbagg=False):
        return _execute_replay_graph(graph, cache=cache)


def _execute_replay_graph(
    graph: ReplayGraph,
    *,
    cache: dict[str, xr.DataArray] | None = None,
) -> xr.DataArray:
    replay_cache = {} if cache is None else cache
    values: dict[str, xr.DataArray] = {}

    for node in graph.nodes:
        if node.cacheable and node.key in replay_cache:
            values[node.key] = replay_cache[node.key].copy(deep=False)
            continue

        if node.kind == "file_load":
            data = _load_file_source_data(node.payload["load_source"])
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
            data = node.payload["operation"].apply(
                values[node.parents[0]],
                parent_data=values[node.parents[1]],
            )
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
) -> xr.DataArray:
    """Replay structured file provenance without executing generated Python."""
    try:
        graph = compile_replay_graph(spec)
        return execute_replay_graph(graph, cache=cache)
    except ReplayGraphError as exc:
        raise TypeError("Expected structured file provenance") from exc


def file_load_source_status(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> FileLoadSourceStatus:
    """Return the current availability of the recorded file-load source."""
    spec = parse_tool_provenance_spec(value)
    if spec is None or spec.file_load_source is None:
        return "no-file-load-source"
    load_source = spec.file_load_source
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
    return "loadable"


def can_reload_without_trust(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> bool:
    """Return whether recorded provenance can replay without trusted user code."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return False
    if spec.kind == "file":
        return file_load_source_status(spec) == "loadable"
    if spec.kind != "script":
        return False
    if has_file_load_source(spec) and file_load_source_status(spec) != "loadable":
        return False
    if not script_provenance_replayable(spec):
        return False
    for script_input in spec.script_inputs:
        input_spec = script_input.parsed_provenance_spec()
        if not can_reload_without_trust(input_spec):
            return False
    return True


def script_provenance_replayable(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
) -> bool:
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None:
        return False
    try:
        _validate_script_provenance(
            parsed,
            external_input_names=external_input_names,
        )
    except (ReplayGraphError, TypeError, ValueError):
        return False
    return True


def _script_provenance_validates(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    strict_replay_code: bool,
) -> bool:
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        return False
    try:
        _validate_script_provenance(
            parsed,
            external_input_names=external_input_names,
            strict_replay_code=strict_replay_code,
        )
    except (ReplayGraphError, TypeError, ValueError):
        return False
    return True


def script_provenance_requires_trust(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
) -> bool:
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        return False
    strict_replayable = _script_provenance_validates(
        parsed,
        external_input_names=external_input_names,
        strict_replay_code=True,
    )
    trusted_replayable = _script_provenance_validates(
        parsed,
        external_input_names=external_input_names,
        strict_replay_code=False,
    )
    current_requires_trust = not strict_replayable and trusted_replayable
    if current_requires_trust:
        return True
    if not strict_replayable:
        return False
    for script_input in parsed.script_inputs:
        input_spec = script_input.parsed_provenance_spec()
        if script_provenance_requires_trust(input_spec):
            return True
    return False


def _script_trust_payload(spec: typing.Any) -> dict[str, typing.Any] | None:
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
        input_payload = _script_trust_payload(script_input.parsed_provenance_spec())
        if input_payload is None:
            continue
        inputs.append({"name": script_input.name, "payload": input_payload})
    return {
        "active_name": parsed.active_name,
        "inputs": inputs,
        "operations": operations,
        "seed_code": parsed.seed_code,
    }


def script_provenance_trust_key(spec: typing.Any) -> str | None:
    payload = _script_trust_payload(spec)
    if payload is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def replay_script_provenance(
    spec: typing.Any,
    inputs: Mapping[str, xr.DataArray],
    *,
    trusted_user_code: bool = False,
) -> xr.DataArray:
    """Execute script provenance from already resolved input arrays."""
    try:
        graph = compile_replay_graph(
            spec,
            external_inputs=inputs,
            trusted_user_code=trusted_user_code,
        )
        return execute_replay_graph(graph)
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
) -> tuple[xr.DataArray, typing.Any]:
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        raise ReplayGraphError("Selected provenance is not script-derived")
    if depth > 20:
        raise ReplayGraphError(
            "Nested script provenance exceeded the maximum reload depth"
        )
    if trusted_user_code:
        replayable = _script_provenance_validates(
            parsed,
            strict_replay_code=False,
        )
    else:
        replayable = script_provenance_replayable(parsed)
    if not replayable:
        raise ReplayGraphError(
            "The recorded operation cannot be replayed automatically"
        )

    live_results: dict[
        tuple[str, str | None, str], tuple[xr.DataArray, typing.Any]
    ] = {}
    live_misses: set[tuple[str, str | None, str]] = set()

    def resolve_live(
        script_input: typing.Any,
    ) -> tuple[xr.DataArray, typing.Any] | None:
        if live_input_resolver is None:
            return None
        key = (script_input.name, script_input.node_uid, script_input.data_role)
        if key in live_results:
            return live_results[key]
        if key in live_misses:
            return None
        resolved = live_input_resolver(script_input)
        if resolved is None:
            live_misses.add(key)
            return None
        live_results[key] = resolved
        return resolved

    def resolve_inputs(current: typing.Any, current_depth: int) -> typing.Any:
        if current_depth > 20:
            raise ReplayGraphError(
                "Nested script provenance exceeded the maximum reload depth"
            )
        resolved_inputs = []
        for script_input in current.script_inputs:
            resolved = resolve_live(script_input)
            if resolved is not None:
                resolved_inputs.append(resolved[1])
                continue

            input_spec = script_input.parsed_provenance_spec()
            if input_spec is None:
                input_reference = _script_input_reference_text(script_input)
                raise ReplayGraphError(
                    f"{input_reference} "
                    "is not open and "
                    "does not contain recorded source provenance."
                )
            if input_spec.kind == "file":
                resolved_inputs.append(
                    script_input.model_copy(
                        update={
                            "node_uid": None,
                            "node_snapshot_token": None,
                            "provenance_spec": input_spec.model_dump(mode="json"),
                        }
                    )
                )
                continue
            if input_spec.kind == "script":
                if trusted_user_code:
                    input_replayable = _script_provenance_validates(
                        input_spec,
                        strict_replay_code=False,
                    )
                else:
                    input_replayable = script_provenance_replayable(input_spec)
                if not input_replayable:
                    raise ReplayGraphError(
                        "The recorded operation cannot be replayed automatically"
                    )
                rebuilt_input = resolve_inputs(input_spec, current_depth + 1)
                resolved_inputs.append(
                    script_input.model_copy(
                        update={
                            "node_uid": None,
                            "node_snapshot_token": None,
                            "provenance_spec": rebuilt_input.model_dump(mode="json"),
                        }
                    )
                )
                continue
            raise ReplayGraphError(
                f"{_script_input_reference_text(script_input)} "
                "is not open and "
                "does not contain reloadable script or file provenance."
            )
        return current.model_copy(update={"script_inputs": tuple(resolved_inputs)})

    rebuilt_spec = resolve_inputs(parsed, depth)
    graph = compile_replay_graph(
        rebuilt_spec,
        live_input_resolver=resolve_live,
        trusted_user_code=trusted_user_code,
    )
    return execute_replay_graph(graph, cache=cache), rebuilt_spec
