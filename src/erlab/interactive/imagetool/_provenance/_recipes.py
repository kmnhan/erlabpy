"""Build complete recipes from explicit sources and operations.

These builders use the existing saved model. They do not inspect Manager windows,
capture array data, or publish results. Source policies are supplied by the caller.
"""

from __future__ import annotations

import typing

from erlab.interactive.imagetool._provenance._graph import (
    ReplayGraphError,
    _validate_script_provenance,
)
from erlab.interactive.imagetool._provenance._model import (
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    recipe,
)

if typing.TYPE_CHECKING:
    from collections.abc import Sequence


def _check_producer(producer: ToolProvenanceSpec) -> None:
    if producer.kind not in {"file", "script"}:
        raise ValueError("A producer must contain a durable source recipe")
    for source in producer.replay_inputs:
        _check_input(source)
    if producer.kind == "script":
        try:
            _validate_script_provenance(
                producer.as_bound_script(), strict_replay_code=False
            )
        except ReplayGraphError as exc:
            raise ValueError(
                f"A producer must contain a complete recipe: {exc}"
            ) from exc


def _check_input(source: ScriptInput) -> None:
    has_revision = (
        source.revision_id is not None or source.resolved_revision_id is not None
    )
    if not source.follow_source:
        if not has_revision:
            raise ValueError("A fixed input must identify its recorded revision")
        return
    producer = source.parsed_provenance_spec()
    if producer is not None:
        _check_producer(producer)
    elif not has_revision:
        raise ValueError("A recipe input must contain a producer or recorded data")


def producer_input(
    producer: ToolProvenanceSpec,
    *,
    name: str = "data",
    label: str | None = None,
) -> ScriptInput:
    """Follow a producer recipe without binding it to a displaying window."""
    _check_producer(producer)
    return ScriptInput(
        name=name,
        label=name if label is None else label,
        data_role="source",
        provenance_spec=producer.model_dump(mode="json"),
    )


def snapshot_input(
    revision_id: str,
    *,
    name: str = "data",
    label: str | None = None,
    producer: ToolProvenanceSpec | None = None,
) -> ScriptInput:
    """Use an exact recorded value, with optional producer history.

    Replay needs a revision resolver to supply the recorded value. The producer
    explains its history; it does not select the value of the fixed input.
    """
    if producer is not None:
        _check_producer(producer)
    return ScriptInput(
        name=name,
        label=name if label is None else label,
        revision_id=revision_id,
        follow_source=False,
        data_role="source",
        provenance_spec=(
            None if producer is None else producer.model_dump(mode="json")
        ),
    )


def append_recipe(
    source: ScriptInput,
    steps: Sequence[ReplayStep],
    *,
    inputs: Sequence[ScriptInput] = (),
    start_label: str,
    active_name: str,
) -> ToolProvenanceSpec:
    """Apply typed steps to an explicit primary input and its named dependencies.

    Input policies, transforms, and recorded revisions are supplied by the caller.
    Construction does not infer them from window ownership or the number of inputs.
    The returned recipe owns its nested values independently of the arguments.
    """
    sources = (source, *inputs)
    for item in sources:
        _check_input(item)
    return recipe(
        *steps,
        start_label=start_label,
        active_name=active_name,
        script_inputs=sources,
        primary_input=source.name,
    ).model_copy(deep=True)


def graft_recipe(
    fragment: ToolProvenanceSpec,
    source: ScriptInput,
) -> ToolProvenanceSpec:
    """Attach a typed fragment's unbound primary input to a complete source.

    A fragment contains the selected steps and a bare primary-input placeholder.
    Auxiliary inputs retain their producer recipes and source policies. A fragment
    that still owns its original primary producer must first be extracted from that
    producer; grafting never silently removes an existing dependency or transform.
    """
    if (
        fragment.kind != "script"
        or fragment.replay_primary_input is None
        or fragment.active_name is None
        or fragment.seed_code is not None
        or fragment.file_load_source is not None
    ):
        raise ValueError("A recipe fragment must declare an unbound primary input")
    primary_name = fragment.replay_primary_input
    primary = next(item for item in fragment.replay_inputs if item.name == primary_name)
    if (
        primary.node_uid is not None
        or primary.node_snapshot_token is not None
        or primary.revision_id is not None
        or primary.resolved_revision_id is not None
        or primary.provenance_spec is not None
        or primary.source_spec is not None
        or not primary.follow_source
        or primary.fallback_role != "raw"
    ):
        raise ValueError("A recipe fragment's primary input must be a bare placeholder")
    return append_recipe(
        source.model_copy(update={"name": primary_name}),
        fragment.steps,
        inputs=tuple(
            item for item in fragment.replay_inputs if item.name != primary_name
        ),
        start_label=typing.cast("str", fragment.start_label),
        active_name=fragment.active_name,
    )
