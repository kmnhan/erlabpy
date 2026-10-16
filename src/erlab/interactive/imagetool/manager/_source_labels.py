from __future__ import annotations

import pathlib
import typing
from collections import deque
from collections.abc import Callable, Mapping

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool._provenance._model import ScriptInput
    from erlab.interactive.imagetool.manager import ImageToolManager


def _recorded_file_basename(script_input: ScriptInput) -> str | None:
    pending = deque((script_input.provenance_spec, script_input.source_spec))
    seen: set[int] = set()
    while pending:
        payload = pending.popleft()
        if not isinstance(payload, Mapping) or id(payload) in seen:
            continue
        seen.add(id(payload))
        file_source = payload.get("file_load_source")
        if isinstance(file_source, Mapping):
            path = file_source.get("path")
            if isinstance(path, str) and path:
                basename = pathlib.PurePath(path).name
                if basename:
                    return basename
        inputs = payload.get("script_inputs", ())
        primary = payload.get("primary_input")
        nested = payload.get("root_input")
        if not isinstance(nested, Mapping):
            nested = (
                next((item for item in inputs if item.get("name") == primary), None)
                if primary is not None
                else inputs[0]
                if len(inputs) == 1
                else None
            )
        if isinstance(nested, Mapping):
            pending.extend((nested.get("provenance_spec"), nested.get("source_spec")))
    return None


def script_input_source_label(
    manager: ImageToolManager,
    script_input: ScriptInput,
    *,
    retained_input_resolver: Callable[[ScriptInput], bool] | None = None,
) -> str:
    # Import here because managed wrappers use this resolver when rendering rows.
    from erlab.interactive.imagetool.manager._wrapper import _ImageToolWrapper

    node = (
        None
        if script_input.node_uid is None
        else manager._tool_graph.nodes.get(script_input.node_uid)
    )
    if node is not None:
        if isinstance(node, _ImageToolWrapper):
            return f"ImageTool {node.index}"
        if node.is_imagetool:
            return "ImageTool child"
        return node.type_badge_text or node.name or "Tool"
    basename = _recorded_file_basename(script_input)
    if basename is not None:
        return basename
    if retained_input_resolver is not None and retained_input_resolver(script_input):
        return "Saved"
    return "Missing"
