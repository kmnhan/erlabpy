"""Emit notebook code from the recorded edges of a data revision."""

from __future__ import annotations

import copy
import pathlib
import typing
import unicodedata
from dataclasses import replace

from erlab._provenance import Application, FileLoad
from erlab.interactive.imagetool._load_source import (
    _deserialize_loader_kwargs,
    _registered_local_callable_loader,
    _resolve_load_func,
)
from erlab.interactive.imagetool._provenance._graph import (
    ReplayGraph,
    ReplayGraphError,
    _add_file_load_node,
    _compile_replay_steps,
    _compile_spec,
    _file_seed_code_parts,
    _memoized_input_provenance_resolver,
    _reserved_names_from_spec,
    _validate_external_input_name,
    emit_replay_code,
)
from erlab.interactive.imagetool._provenance._model import (
    FileLoadSource,
    FileReplayCall,
    ToolProvenanceSpec,
)
from erlab.interactive.imagetool._provenance._program import (
    PreparedFileCall,
    RecordedProgram,
)
from erlab.utils.misc import _is_valid_identifier

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from erlab._provenance import DataRevision, ProvenanceGraph


def recorded_file_source(
    load: FileLoad, *, assign: str = "data", generate_code: bool = False
) -> FileLoadSource:
    """Describe the file edge without reading the file."""
    call = load.call
    code = None
    label = "Loader"
    text = call.target
    kwargs_text = ""
    if load.presentation is not None:
        description = load.presentation
        label, text, kwargs_text = (
            description.loader_label,
            description.loader_text,
            description.kwargs_text,
        )
        code = description.load_code
    if generate_code and call.kind != "extension_loader":
        loader = (
            call.target
            if call.kind == "erlab_loader"
            else _registered_local_callable_loader(call.target)
        )
        resolved = (
            _resolve_load_func(
                (loader, _deserialize_loader_kwargs(call.kwargs), call.selection)
            )
            if loader is not None
            else None
        )
        if resolved is not None:
            label = resolved.loader_label
            text = resolved.loader_text
            kwargs_text = resolved.kwargs_text
            code = replace(resolved, cast_float64=call.cast_float64).load_code(
                pathlib.Path(load.source.path), assign=assign
            )
    return FileLoadSource(
        path=load.source.path,
        replay_call=FileReplayCall.model_validate(
            call.model_dump(include=set(FileReplayCall.model_fields))
        ),
        loader_label=label,
        loader_text=text,
        kwargs_text=kwargs_text,
        load_code=code,
    )


def _add_recorded_file(graph: ReplayGraph, revision_id: str, load: FileLoad) -> None:
    call = load.call
    source = recorded_file_source(load, generate_code=True)
    setup = code = None
    if call.kind != "extension_loader":
        if source.load_code is None:
            raise ReplayGraphError(
                f"File load cannot emit Python for revision {revision_id!r}. "
                "Supply a checkpoint name."
            )
        setup, code = _file_seed_code_parts(source.load_code, "data")
    file_key = _add_file_load_node(
        graph,
        source,
        setup_code=setup,
        load_code=code,
        active_name="data",
        key=None if isinstance(call, PreparedFileCall) else revision_id,
    )
    if isinstance(call, PreparedFileCall):
        prepared = _compile_replay_steps(graph, file_key, call.steps, display=True)
        graph.add_node(revision_id, "relay", parents=(prepared,))


def _notebook_name(name: str) -> str:
    if not _is_valid_identifier(name):
        raise ValueError(f"Expected a Python identifier, got {name!r}")
    # Python normalizes identifiers before looking them up in a notebook namespace.
    normalized = unicodedata.normalize("NFKC", name)
    if not _is_valid_identifier(normalized):
        raise ValueError(f"Expected a Python identifier, got {normalized!r}")
    _validate_external_input_name(normalized)
    return normalized


def full_code(
    graph: ProvenanceGraph,
    revision_id: str,
    *,
    output_name: str,
    source_names: Mapping[str, str] | None = None,
    watched_sources: Mapping[str, tuple[str, ToolProvenanceSpec]] | None = None,
    verify_file: Callable[[DataRevision], bool] | None = None,
) -> str:
    """Emit the selected revision using its applied parameters and exact inputs.

    Supply notebook names for checkpointed inputs by revision ID. Emit a file
    read only when the caller verifies that the recorded source can reproduce that
    revision. The file must still be that version when the notebook code runs.
    """
    output_name = _notebook_name(output_name)
    names = {key: _notebook_name(name) for key, name in (source_names or {}).items()}
    watched = watched_sources or {}
    notebook_names = [
        *names.values(),
        *(_notebook_name(name) for name, _ in watched.values()),
    ]
    if len(set(notebook_names)) != len(notebook_names) or names.keys() & watched.keys():
        raise ValueError("Distinct revisions require distinct checkpoint names")
    with graph.retain(revision_id):
        lineage = graph.recorded_lineage(revision_id, stop_at=names.keys())
        roots = {
            node.revision_id
            for node in lineage
            if not isinstance(node.producer, Application) or node.checkpointed
        }
        watched_origins = {
            node.revision_id
            for node in lineage
            if not isinstance(node.producer, Application)
        }
        if unknown := (names.keys() - roots) | (watched.keys() - watched_origins):
            raise ValueError(
                f"Checkpoint names must identify recorded origins: {sorted(unknown)!r}"
            )

        resolve = _memoized_input_provenance_resolver()
        reserved = set(notebook_names)
        for _, seed in watched.values():
            reserved.update(_reserved_names_from_spec(seed, resolve))
        for node in lineage:
            if isinstance(node.producer, Application) and isinstance(
                node.producer.operation, RecordedProgram
            ):
                reserved.update(
                    _reserved_names_from_spec(node.producer.operation.spec, resolve)
                )
        replay = ReplayGraph(display=True, reserved_names=reserved)
        for node in lineage:
            key = node.revision_id
            producer = node.producer
            if key in watched:
                seed_key = _compile_spec(
                    replay,
                    watched[key][1],
                    display=True,
                    trusted_user_code=False,
                    structured_file_replay=False,
                    external_inputs=None,
                    live_input_resolver=None,
                    allow_unresolved_inputs=False,
                    input_provenance_resolver=resolve,
                )
                replay.add_node(key, "relay", parents=(seed_key,))
            elif key in names:
                if key == revision_id:
                    return (
                        names[key]
                        if names[key] == output_name
                        else f"{output_name} = {names[key]}"
                    )
                replay.add_node(key, "external_input", payload={"name": names[key]})
            elif isinstance(producer, Application):
                if isinstance(producer.operation, RecordedProgram):
                    program = producer.operation
                    output = _compile_spec(
                        replay,
                        program.spec,
                        display=True,
                        trusted_user_code=False,
                        structured_file_replay=False,
                        external_inputs=None,
                        live_input_resolver=None,
                        allow_unresolved_inputs=False,
                        input_provenance_resolver=resolve,
                        input_keys=program.input_names(producer.inputs),
                    )
                    replay.add_node(key, "relay", parents=(output,))
                    continue
                (_, primary), *others = producer.inputs
                # The code emitter takes both the receiver and its source context.
                replay.add_node(
                    key,
                    "operation",
                    parents=(primary, primary, *(source for _, source in others)),
                    payload={
                        "operation": producer.operation,
                        "input_bindings": tuple(others),
                    },
                )
            elif isinstance(producer, FileLoad):
                unavailable = (
                    f"Supply a notebook name for checkpointed revision {key!r} "
                    f"(file {producer.source.path!r})."
                )
                try:
                    verified = verify_file is not None and verify_file(
                        copy.deepcopy(node)
                    )
                except OSError as exc:
                    raise ReplayGraphError(unavailable) from exc
                if not verified:
                    raise ReplayGraphError(unavailable)
                _add_recorded_file(replay, key, producer)
            else:
                raise ReplayGraphError(
                    f"Supply a notebook name for checkpointed revision {key!r} "
                    "(memory source)."
                )
    replay.output_key = revision_id
    return emit_replay_code(
        replay,
        output_name=output_name,
        input_name_overrides=names,
        include_framework_imports=True,
        publish_output=True,
        preserve_extension_versions=True,
    )
