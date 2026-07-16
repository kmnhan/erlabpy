"""Describe, compose, and replay ImageTool data provenance.

.. versionchanged:: 3.25.0
   This module now focuses on provenance models, source constructors, composition,
   grouping, parsing, and replay. Concrete operation models live in the evolving
   ``erlab.interactive.imagetool._provenance._operations`` catalog.
"""

from __future__ import annotations

__all__ = [
    "DerivationEntry",
    "FileDataSelection",
    "FileLoadSource",
    "FileReplayCall",
    "ReplayStage",
    "ScriptInput",
    "ToolProvenanceOperation",
    "ToolProvenanceSpec",
    "compose_display_provenance",
    "compose_full_provenance",
    "file_load",
    "full_data",
    "operation_group_range",
    "parse_tool_provenance_operation",
    "parse_tool_provenance_spec",
    "public_data",
    "replay_file_provenance",
    "replay_script_provenance",
    "script",
    "selection",
    "stamp_operation_group",
    "strip_operation_groups",
]

from erlab.interactive.imagetool._provenance._execution import (
    replay_file_provenance,
    replay_script_provenance,
)
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ReplayStage,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    compose_display_provenance,
    compose_full_provenance,
    file_load,
    full_data,
    operation_group_range,
    parse_tool_provenance_operation,
    parse_tool_provenance_spec,
    public_data,
    script,
    selection,
    stamp_operation_group,
    strip_operation_groups,
)
