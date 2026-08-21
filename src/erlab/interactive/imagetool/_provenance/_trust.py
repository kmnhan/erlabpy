"""Extract stored Python at the provenance replay boundaries."""

from __future__ import annotations

import base64
import itertools
import json
import typing
from collections.abc import Mapping

from erlab.interactive._code_trust import create_entry
from erlab.interactive._fit_code_trust import (
    lmfit_parameter_expression_entries,
    lmfit_result_code_entry,
)
from erlab.interactive.imagetool._provenance._code import _FIT_DATASET_MARKER
from erlab.interactive.imagetool._provenance._model import (
    _direct_replay_source_name,
    parse_tool_provenance_spec,
)


def _model_fit_parameter_expressions(
    operation: typing.Any,
) -> tuple[tuple[str, str], ...]:
    parameters = getattr(operation, "parameters", None)
    if getattr(operation, "op", None) != "model_fit" or not isinstance(
        parameters, Mapping
    ):
        return ()
    return tuple(
        (name, expression)
        for name, parameter in parameters.items()
        if isinstance(name, str)
        and isinstance(expression := getattr(parameter, "expr", None), str)
        and expression.strip()
    )


def _opaque_fit_dataset_payload(operation: typing.Any) -> bytes | None:
    edge_fit = getattr(operation, "edge_fit", None)
    if (
        getattr(operation, "op", None) != "correct_with_edge"
        or not isinstance(edge_fit, Mapping)
        or _FIT_DATASET_MARKER not in edge_fit
    ):
        return None
    encoded = edge_fit[_FIT_DATASET_MARKER]
    if isinstance(encoded, str):
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError):
            pass
    return json.dumps(
        encoded,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def provenance_operation_code_trust_entries(
    operation: typing.Any,
    *,
    location_prefix: str,
):
    """Return executable lmfit content stored by one operation."""
    entries = lmfit_parameter_expression_entries(
        _model_fit_parameter_expressions(operation),
        feature="erlab.provenance.model-fit-parameter-expression",
        location_prefix=f"{location_prefix}/parameters",
    )
    payload = _opaque_fit_dataset_payload(operation)
    if payload is None:
        return entries
    entry = lmfit_result_code_entry(
        payload,
        feature="erlab.provenance.lmfit-result",
        location=f"{location_prefix}/edge-fit",
    )
    return entries if entry is None else (*entries, entry)


def provenance_operation_requires_code_trust(operation: typing.Any) -> bool:
    """Return whether one operation can reach an lmfit code boundary."""
    return bool(
        provenance_operation_code_trust_entries(
            operation,
            location_prefix="operation",
        )
    )


def provenance_replay_node_code_trust_entries(
    node: typing.Any,
    *,
    location_prefix: str,
):
    """Return entries attached to one executable replay node."""
    if node.kind == "operation":
        return provenance_operation_code_trust_entries(
            node.payload["operation"],
            location_prefix=location_prefix,
        )
    if node.kind != "script":
        return ()
    context = {
        "active_name": node.payload.get("active_name"),
        "binding_names": sorted(
            {name for name, _key in node.payload.get("bindings", ())}
        ),
    }
    return tuple(
        create_entry(
            feature="erlab.provenance.script-code",
            location=f"{location_prefix}/{index}",
            code=code,
            context=context,
        )
        for index, code in enumerate(node.payload.get("stored_code", ()))
        if isinstance(code, str) and code.strip()
    )


def provenance_replay_graph_code_trust_entries(
    graph: typing.Any,
    *,
    location_prefix: str,
):
    """Return entries attached to all executable nodes in a replay graph."""
    return tuple(
        entry
        for index, node in enumerate(graph.nodes)
        for entry in provenance_replay_node_code_trust_entries(
            node,
            location_prefix=f"{location_prefix}/nodes/{index}",
        )
    )


def _stored_spec_code_trust_entries(
    spec: typing.Any,
    *,
    location_prefix: str,
):
    """Conservatively inventory stored code when a graph cannot compile."""
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None:
        return ()
    context = {
        "active_name": parsed.active_name if parsed.kind == "script" else "derived",
        "binding_names": (
            sorted({item.name for item in parsed.script_inputs})
            if parsed.kind == "script"
            else ["data", "derived", "parent_data"]
        ),
    }
    entries = []
    if parsed.kind == "script" and parsed.seed_code and parsed.seed_code.strip():
        entries.append(
            create_entry(
                feature="erlab.provenance.script-code",
                location=f"{location_prefix}/seed",
                code=parsed.seed_code,
                context=context,
            )
        )
    for index, operation in enumerate(parsed.operations):
        operation_location = f"{location_prefix}/operations/{index}"
        entries.extend(
            provenance_operation_code_trust_entries(
                operation,
                location_prefix=operation_location,
            )
        )
        code = getattr(operation, "code", None)
        if (
            getattr(operation, "op", None) == "script_code"
            and bool(getattr(operation, "copyable", False))
            and not bool(getattr(operation, "framework_owned", False))
            and isinstance(code, str)
            and code.strip()
        ):
            entries.append(
                create_entry(
                    feature="erlab.provenance.script-code",
                    location=operation_location,
                    code=code,
                    context=context,
                )
            )
    for index, script_input in enumerate(parsed.script_inputs):
        entries.extend(
            _stored_spec_code_trust_entries(
                script_input.parsed_provenance_spec(),
                location_prefix=f"{location_prefix}/inputs/{index}:{script_input.name}",
            )
        )
    return tuple(entries)


def script_replay_source_input_names(spec: typing.Any) -> tuple[str, ...]:
    """Return the smallest live-source namespace that makes a script replayable."""
    from erlab.interactive.imagetool._provenance._graph import (
        ReplayGraphError,
        compile_replay_graph,
    )

    parsed = parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script" or parsed.script_inputs:
        return ()

    candidates = ["data", "derived", "parent_data"]
    source_name = _direct_replay_source_name(parsed)
    if source_name is not None and source_name not in candidates:
        candidates.insert(0, source_name)
    for count in range(len(candidates) + 1):
        for names in itertools.combinations(candidates, count):
            external_inputs = {name: typing.cast("typing.Any", None) for name in names}
            try:
                compile_replay_graph(
                    parsed,
                    trusted_user_code=True,
                    external_inputs=external_inputs or None,
                )
            except (ReplayGraphError, TypeError, ValueError):
                continue
            return names
    return ()


def provenance_code_trust_entries(
    spec: typing.Any,
    *,
    location_prefix: str,
    external_input_names: set[str] | None = None,
):
    """Return stored code that can reach a provenance execution boundary."""
    from erlab.interactive.imagetool._provenance._graph import (
        ReplayGraphError,
        compile_replay_graph,
    )

    parsed = parse_tool_provenance_spec(spec)
    if parsed is None:
        return ()
    if external_input_names is None:
        external_input_names = set(script_replay_source_input_names(parsed))
    external_inputs = {
        name: typing.cast("typing.Any", None) for name in external_input_names or ()
    }
    try:
        graph = compile_replay_graph(
            parsed,
            trusted_user_code=True,
            external_inputs=external_inputs or None,
            allow_unresolved_inputs=True,
        )
    except (ReplayGraphError, TypeError, ValueError):
        return _stored_spec_code_trust_entries(
            parsed,
            location_prefix=location_prefix,
        )
    return provenance_replay_graph_code_trust_entries(
        graph,
        location_prefix=location_prefix,
    )


def provenance_requires_code_trust(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
) -> bool:
    """Return whether provenance contains code at a replay boundary."""
    return bool(
        provenance_code_trust_entries(
            spec,
            location_prefix="provenance",
            external_input_names=external_input_names,
        )
    )
