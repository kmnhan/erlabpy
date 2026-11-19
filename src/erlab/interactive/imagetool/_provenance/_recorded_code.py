"""Emit notebook code from the recorded edges of a data revision."""

from __future__ import annotations

import copy
import pathlib
import typing
import unicodedata
from dataclasses import replace

from erlab._provenance import Application, FileLoad, FileVersion
from erlab._provenance._files import (
    _capture_loader_file,
    _deserialize_loader_kwargs,
    _registered_local_callable_loader,
    _resolve_load_func,
)
from erlab._provenance._model import FileLoadSource
from erlab.interactive.imagetool._provenance._graph import (
    ReplayGraph,
    ReplayGraphError,
    _add_file_load_node,
    _file_seed_code_parts,
    _validate_external_input_name,
    emit_replay_code,
)
from erlab.utils.misc import _is_valid_identifier

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from erlab._provenance import DataRevision, ProvenanceGraph


def _add_recorded_file(
    graph: ReplayGraph, revision_id: str, load: FileLoad, backing: FileVersion
) -> None:
    call = load.call
    source = FileLoadSource(
        path=load.source.path,
        replay_call=call,
        loader_text=call.target,
        loader_label="Loader",
        kwargs_text="",
    )
    setup = code = None
    if call.kind != "extension_loader":
        loader = (
            call.target
            if call.kind == "erlab_loader"
            else _registered_local_callable_loader(call.target)
        )
        if loader is None:
            raise ReplayGraphError(f"File loader {call.target!r} is unavailable")
        resolved = _resolve_load_func(
            (loader, _deserialize_loader_kwargs(call.kwargs), call.selection)
        )
        if resolved is not None:
            code = replace(resolved, cast_float64=call.cast_float64).load_code(
                pathlib.Path(load.source.path), assign="data", backing=backing
            )
        if code is None:
            raise ReplayGraphError(
                f"File load cannot emit Python for revision {revision_id!r}. "
                "Supply a checkpoint name."
            )
        setup, code = _file_seed_code_parts(code, "data")
    _add_file_load_node(
        graph,
        source,
        setup_code=setup,
        load_code=code,
        active_name="data",
        key=revision_id,
        backing=backing,
    )


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
    verify_file: Callable[[DataRevision], bool] | None = None,
) -> str:
    """Emit the selected revision using its applied parameters and exact inputs.

    Supply notebook names for checkpointed origins by revision ID. Emit a file
    read only when the caller verifies that the recorded source can reproduce that
    revision. A versioned source is also checked when the notebook reads it.
    """
    output_name = _notebook_name(output_name)
    names = {key: _notebook_name(name) for key, name in (source_names or {}).items()}
    if len(set(names.values())) != len(names):
        raise ValueError("Distinct revisions require distinct checkpoint names")
    with graph.retain(revision_id):
        lineage = graph.recorded_lineage(revision_id)
        roots = {
            node.revision_id
            for node in lineage
            if not isinstance(node.producer, Application)
        }
        if unknown := names.keys() - roots:
            raise ValueError(
                f"Checkpoint names must identify recorded origins: {sorted(unknown)!r}"
            )

        replay = ReplayGraph(display=True)
        for node in lineage:
            key = node.revision_id
            producer = node.producer
            if key in names:
                if key == revision_id:
                    return (
                        names[key]
                        if names[key] == output_name
                        else f"{output_name} = {names[key]}"
                    )
                replay.add_node(key, "external_input", payload={"name": names[key]})
            elif isinstance(producer, Application):
                (_, primary), *others = producer.inputs
                # The code emitter takes both the receiver and its source context.
                replay.add_node(
                    key,
                    "operation",
                    parents=(primary, primary, *(source for _, source in others)),
                    payload={
                        "operation": producer.operation,
                        "execution_settings": producer.local_execution,
                        "input_bindings": tuple(others),
                    },
                )
            elif isinstance(producer, FileLoad):
                unavailable = (
                    f"Supply a notebook name for checkpointed revision {key!r} "
                    f"(file {producer.source.path!r})."
                )
                if verify_file is None:
                    raise ReplayGraphError(unavailable)
                backing = node.backing
                if backing is None:
                    # A caller can verify an older checkpoint by content.
                    # Bind that check to the file the notebook will reopen.
                    call = producer.call
                    if call.kind == "callable":
                        backing = _capture_loader_file(
                            producer.source.path,
                            _registered_local_callable_loader(call.target),
                            _deserialize_loader_kwargs(call.kwargs),
                        )
                if backing is None:
                    raise ReplayGraphError(unavailable)
                try:
                    backing.verify()
                    if not verify_file(copy.deepcopy(node)):
                        raise ReplayGraphError(unavailable)
                    backing.verify()
                except OSError as exc:
                    raise ReplayGraphError(unavailable) from exc
                _add_recorded_file(replay, key, producer, backing)
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
