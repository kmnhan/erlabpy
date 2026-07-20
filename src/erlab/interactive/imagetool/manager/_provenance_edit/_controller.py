"""Coordinate provenance edits in ImageTool Manager."""

from __future__ import annotations

import dataclasses
import pathlib
import traceback
import typing
import warnings

import numpy as np
from qtpy import QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
import erlab.interactive.utils
from erlab.interactive.imagetool import dialogs
from erlab.interactive.imagetool._provenance._execution import (
    file_load_source_status,
    replay_file_provenance,
    replay_script_provenance,
    script_provenance_replayable,
    script_provenance_requires_trust,
)
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    _ProvenanceReorderSection,
    _ProvenanceStepRef,
    compose_display_provenance,
    compose_full_provenance,
    full_data,
    iter_operation_refs,
    require_live_source_spec,
    restamp_operation_groups,
    script,
    script_input_dependency_refs,
)
from erlab.interactive.imagetool._provenance._operations import (
    AffineCoordOperation,
    DivideByCoordOperation,
    GaussianFilterOperation,
    NormalizeOperation,
    ScriptCodeOperation,
    SortByOperation,
)
from erlab.interactive.imagetool.manager._provenance_edit._editors import (
    _NATIVE_TERMINAL_CURRENT_DATA_EDITORS,
    _dialog_match_for_operation_ref,
    _editable_group_range_for_ref,
    _OperationDialogMatch,
    _ScriptCodeEditDialog,
    _uneditable_operation_reason,
)
from erlab.interactive.imagetool.manager._provenance_edit._files import (
    _file_load_edit_active_name,
    _file_not_found_path_from_exception,
    _FileLoadBatchPeer,
    _FileLoadEditDialog,
    _loader_summary,
    _missing_source_file_error_from_exception,
    _MissingProvenanceSourceFileError,
    _normalized_path,
    _replace_file_load_fields,
    _same_replay_loader,
)
from erlab.interactive.imagetool.manager._provenance_edit._reorder import (
    _ProvenanceReorderDialog,
)
from erlab.interactive.imagetool.manager._widgets import _TrustedScriptReplayCancelled

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )


@dataclasses.dataclass(frozen=True)
class _ValidatedProvenanceEdit:
    node: _ImageToolWrapper | _ManagedWindowNode
    scope: typing.Literal["display", "source"]
    data: xr.DataArray
    spec: ToolProvenanceSpec
    filter_operation: ToolProvenanceOperation | None


@dataclasses.dataclass(frozen=True)
class _ProvenanceReorderSession:
    node_uid: str
    scope: typing.Literal["display", "source"]
    spec: ToolProvenanceSpec
    sections: tuple[_ProvenanceReorderSection, ...]
    node_snapshot_token: str
    parent_snapshot_token: str | None
    dependency_snapshot_tokens: tuple[
        tuple[
            str,
            typing.Literal["source", "displayed"],
            str | None,
        ],
        ...,
    ]


class _ProvenanceReplayFailure(RuntimeError):
    def __init__(self, where: str, cause: Exception) -> None:
        super().__init__(f"{where}: {cause}")
        self.where = where
        self.cause = cause
        self.__cause__ = cause

    @property
    def missing_source_file(self) -> _MissingProvenanceSourceFileError | None:
        return _missing_source_file_error_from_exception(self.cause)


class _ProvenanceEditController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager

    def _reorder_target(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
    ) -> tuple[
        typing.Literal["display", "source"],
        ToolProvenanceSpec | None,
    ]:
        if node.parent_uid is not None and node.source_spec is not None:
            return "source", node.displayed_source_spec
        return "display", node.displayed_provenance_spec

    def _dependency_snapshot_tokens(
        self,
        spec: ToolProvenanceSpec,
    ) -> tuple[
        tuple[
            str,
            typing.Literal["source", "displayed"],
            str | None,
        ],
        ...,
    ]:
        tokens: list[
            tuple[
                str,
                typing.Literal["source", "displayed"],
                str | None,
            ]
        ] = []
        seen: set[tuple[str, typing.Literal["source", "displayed"]]] = set()
        for ref in script_input_dependency_refs(spec):
            key = (ref.node_uid, ref.data_role)
            if key in seen:
                continue
            seen.add(key)
            dependency = self._manager._tool_graph.nodes.get(ref.node_uid)
            token = (
                None
                if dependency is None
                else dependency.snapshot_token_for_role(ref.data_role)
            )
            tokens.append((ref.node_uid, ref.data_role, token))
        return tuple(tokens)

    def _reorder_sections(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: ToolProvenanceSpec,
    ) -> tuple[_ProvenanceReorderSection, ...]:
        active_filter_ref = self._active_filter_ref(node, spec)
        return spec._reorder_sections(
            fixed_refs=(() if active_filter_ref is None else (active_filter_ref,))
        )

    def _reorder_replay_unavailable_reason(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
    ) -> str | None:
        source_status = file_load_source_status(spec)
        if source_status not in {"no-file-load-source", "loadable"}:
            return {
                "missing-file": "The recorded source file is not available.",
                "no-replay-call": (
                    "The recorded source does not include replay loader information."
                ),
                "missing-loader": "The recorded source loader is not available.",
            }[source_status]

        if spec.kind == "file":
            if source_status != "loadable":
                return "This file provenance does not have a replayable source."
            return None

        if spec.kind == "script":
            if not (
                script_provenance_replayable(spec)
                or script_provenance_requires_trust(spec)
            ):
                return "This provenance contains recorded code that cannot be replayed."
            for script_input in spec.script_inputs:
                reason = self._manager._script_input_unavailable_reason(
                    script_input,
                    target_node_uid=node.uid,
                )
                if reason is not None:
                    return reason
            return None

        external_input_names = {"data", "derived", "parent_data"}
        for operation in spec.operations:
            if not isinstance(operation, ScriptCodeOperation):
                continue
            step_spec = self._live_script_step_spec(operation)
            if not (
                script_provenance_replayable(
                    step_spec,
                    external_input_names=external_input_names,
                )
                or script_provenance_requires_trust(
                    step_spec,
                    external_input_names=external_input_names,
                )
            ):
                return (
                    "This live provenance contains recorded code that cannot be "
                    "replayed."
                )

        if scope == "source":
            if (
                node.parent_uid is None
                or node.parent_uid not in self._manager._tool_graph.nodes
            ):
                return "The parent ImageTool source is no longer available."
            return None
        if node.detached_live_parent_data is None:
            return "This live provenance no longer has its parent source data."
        return None

    def can_reorder_steps(self) -> tuple[bool, str]:
        node = self._metadata_node()
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to reorder provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        scope, spec = self._reorder_target(node)
        if spec is None:
            return False, "This ImageTool does not have replayable provenance."
        if not self._reorder_sections(node, spec):
            return (
                False,
                "There are no two movable steps inside the same provenance section.",
            )
        if reason := self._reorder_replay_unavailable_reason(node, scope, spec):
            return False, reason
        return True, ""

    def open_reorder_dialog(self) -> None:
        available, reason = self.can_reorder_steps()
        if not available:
            self._show_unavailable(reason)
            return
        node = typing.cast(
            "_ImageToolWrapper | _ManagedWindowNode",
            self._metadata_node(),
        )
        scope, spec = self._reorder_target(node)
        if spec is None:  # pragma: no cover - guarded by can_reorder_steps.
            self._show_unavailable("No provenance spec is available.")
            return
        sections = self._reorder_sections(node, spec)
        parent_snapshot_token = None
        if scope == "source":
            parent_snapshot_token = self._manager._parent_node(node).snapshot_token
        session = _ProvenanceReorderSession(
            node_uid=node.uid,
            scope=scope,
            spec=spec,
            sections=sections,
            node_snapshot_token=node.snapshot_token,
            parent_snapshot_token=parent_snapshot_token,
            dependency_snapshot_tokens=self._dependency_snapshot_tokens(spec),
        )
        dialog = _ProvenanceReorderDialog(
            sections=sections,
            parent=self._manager,
        )
        dialog.apply_requested.connect(
            lambda: self._apply_reorder_dialog(dialog, session)
        )
        dialog.exec()

    def _reorder_session_current(
        self,
        session: _ProvenanceReorderSession,
    ) -> tuple[_ImageToolWrapper | _ManagedWindowNode | None, str]:
        node = self._manager._tool_graph.nodes.get(session.node_uid)
        if not self._node_editable(node):
            return None, "The selected ImageTool is no longer available."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        scope, spec = self._reorder_target(node)
        if scope != session.scope or spec != session.spec:
            return None, "The provenance changed while the reorder dialog was open."
        if node.snapshot_token != session.node_snapshot_token:
            return None, "The ImageTool data changed while the dialog was open."
        if session.scope == "source":
            parent = self._manager._parent_node(node)
            if parent.snapshot_token != session.parent_snapshot_token:
                return (
                    None,
                    "The parent ImageTool data changed while the dialog was open.",
                )
        if self._dependency_snapshot_tokens(spec) != session.dependency_snapshot_tokens:
            return None, "A provenance input changed while the dialog was open."
        if self._reorder_sections(node, spec) != session.sections:
            return (
                None,
                "The available provenance steps changed while the dialog was open.",
            )
        return node, ""

    def _apply_reorder_dialog(
        self,
        dialog: _ProvenanceReorderDialog,
        session: _ProvenanceReorderSession,
    ) -> None:
        node, stale_reason = self._reorder_session_current(session)
        if node is None:
            QtWidgets.QMessageBox.information(
                dialog,
                "Provenance Changed",
                f"{stale_reason}\n\nClose this dialog and reopen it to reorder the "
                "current provenance.",
            )
            return

        dialog.set_busy(True)
        try:
            candidate = session.spec._reorder_operation_blocks(
                session.sections,
                dialog.reorder_plan(),
            )
            self._validate_and_replace(
                node,
                session.scope,
                candidate,
                where="validating the reordered provenance",
            )
        except _TrustedScriptReplayCancelled:
            dialog.set_busy(False)
            return
        except Exception as exc:
            dialog.set_busy(False)
            self._show_failed(
                "Could Not Apply Reordered Provenance",
                exc,
                text="The reordered provenance could not be applied.",
                unchanged_reason=(
                    "The complete reordered recipe could not be replayed, so the "
                    "ImageTool data and recorded provenance were left unchanged. "
                    "Adjust the order or cancel the dialog."
                ),
            )
            return
        dialog.finish_success()

    def can_paste_steps(
        self,
        operations: Sequence[ToolProvenanceOperation] | None,
    ) -> tuple[bool, str]:
        if not operations:
            return False, "The clipboard does not contain copied ImageTool steps."
        if not self._paste_target_nodes():
            return False, "Select an available ImageTool row to paste provenance."
        return True, ""

    def paste_steps(
        self,
        operations: Sequence[ToolProvenanceOperation],
        *,
        active_name: str,
        contains_script: bool,
    ) -> None:
        paste_enabled, paste_reason = self.can_paste_steps(operations)
        if not paste_enabled:
            self._show_unavailable(paste_reason)
            return
        targets = self._paste_target_nodes()
        failures: list[tuple[_ImageToolWrapper | _ManagedWindowNode, Exception]] = []
        pasted_count = 0
        for node in targets:
            steps = restamp_operation_groups(operations)
            try:
                self._paste_steps_into_node(
                    node,
                    steps,
                    active_name=active_name,
                    contains_script=contains_script,
                )
            except _TrustedScriptReplayCancelled:
                return
            except Exception as exc:
                failures.append((node, exc))
            else:
                pasted_count += 1

        if pasted_count > 0:
            if failures:
                self._show_partial_paste_failures(pasted_count, failures)
            return

        if failures:
            self._show_failed(
                "Could Not Paste Provenance Steps",
                failures[0][1],
                text="The copied provenance steps could not be applied.",
                unchanged_reason=(
                    "The copied steps could not be replayed on the selected "
                    "ImageTool's current data, so nothing was changed. Check that "
                    "the destination data has the dimensions, coordinates, and "
                    "inputs expected by the copied steps."
                ),
            )

    def _paste_steps_into_node(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        steps: tuple[ToolProvenanceOperation, ...],
        *,
        active_name: str,
        contains_script: bool,
    ) -> None:
        if contains_script or any(not operation.live_applicable for operation in steps):
            self._paste_detached_steps(
                node,
                script(
                    *steps,
                    start_label="Start from current ImageTool data",
                    active_name=active_name or "derived",
                ),
                where="validating the pasted script provenance steps",
            )
            return
        self._paste_structured_steps(node, steps)

    def _paste_structured_steps(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        operations: tuple[ToolProvenanceOperation, ...],
    ) -> None:
        for operation in operations:
            if not operation.live_applicable:
                raise TypeError("Only live provenance operations can be pasted here")

        if node.parent_uid is not None and node.displayed_source_spec is not None:
            candidate = node.displayed_source_spec.append_replacement_operations(
                *operations
            )
            try:
                data, candidate = self._replay_candidate_result(
                    node,
                    "source",
                    candidate,
                )
                erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(data)
            except Exception as exc:
                raise _ProvenanceReplayFailure(
                    "validating the pasted provenance steps: "
                    "replaying the requested provenance",
                    exc,
                ) from exc
            self._replace_node_data(node, "source", data, candidate, None)
            self._manager._update_info(uid=node.uid)
            return

        local = full_data(*operations)
        self._paste_detached_steps(
            node,
            local,
            where="validating the pasted provenance steps",
        )

    def _paste_detached_steps(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        local: ToolProvenanceSpec,
        *,
        where: str,
    ) -> None:
        current_data = node.current_public_data()
        try:
            if local.kind == "script":
                trusted_user_code = script_provenance_requires_trust(local)
                if trusted_user_code:
                    self._manager._ensure_script_provenance_trusted(
                        local,
                        reason="paste these provenance steps",
                    )
                data = replay_script_provenance(
                    local,
                    {
                        "data": current_data,
                        "derived": current_data,
                    },
                    trusted_user_code=trusted_user_code,
                )
            else:
                data = local.apply(current_data)
            erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(data)
        except _TrustedScriptReplayCancelled:
            raise
        except Exception as exc:
            raise _ProvenanceReplayFailure(
                f"{where}: replaying the requested provenance",
                exc,
            ) from exc

        if local.kind == "script":
            spec = compose_full_provenance(
                node.displayed_provenance_spec,
                local,
                script_context_names=("data", "derived"),
            )
        else:
            spec = compose_full_provenance(
                node.displayed_provenance_spec,
                local,
            )
        if spec is None:
            spec = local.to_replay_spec()
        node.replace_with_detached_data(data, spec, preserve_filter=False)
        self._manager._update_info(uid=node.uid)

    def can_delete_row(
        self,
        row: _ProvenanceDisplayRow | None,
    ) -> tuple[bool, str]:
        node = self._metadata_node()
        if row is None or row.replay_ref is None:
            return False, "This row cannot be deleted."
        if row.replay_ref.kind != "operation":
            return False, "Only operation rows can be deleted."
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to delete provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        if self._source_child_parent_row(node, row):
            return False, "Delete the parent ImageTool row directly."
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            return False, "This row does not have replayable provenance."
        if spec._operation_for_ref(row.replay_ref) is None:
            return False, "This operation is not available."
        if spec.kind == "script":
            try:
                group = _editable_group_range_for_ref(spec, row.replay_ref)
                if group is None:
                    candidate = spec._replace_operation_ref(row.replay_ref, ())
                else:
                    candidate = spec._replace_operation_range_ref(
                        row.replay_ref,
                        group[0],
                        group[1],
                        (),
                    )
            except (IndexError, ValueError):
                return False, "This script row is not a replayable step."
            if not (
                script_provenance_replayable(candidate)
                or script_provenance_requires_trust(candidate)
            ):
                return False, "Deleting this script step would make replay invalid."
        if spec.kind in {"full_data", "public_data", "selection"}:
            if row.scope == "source" and node.parent_uid is not None:
                return True, ""
            return False, "This live row needs a parent source to replay."
        return True, ""

    def can_edit_row(
        self,
        row: _ProvenanceDisplayRow | None,
    ) -> tuple[bool, str]:
        node = self._metadata_node()
        if row is None or row.edit_ref is None:
            return False, "This row does not support editing."
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to edit provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        if self._source_child_parent_row(node, row):
            return False, "Edit the parent ImageTool row directly."

        spec = self._display_spec_for_row(node, row)
        if spec is None:
            return False, "This row does not have replayable provenance."
        if row.edit_ref.kind == "file_load":
            if spec.kind not in {"file", "script"} or spec.file_load_source is None:
                return False, "This row is not a file load step."
            if spec.file_load_source.replay_call is None:
                return False, "This file load step cannot be replayed."
            return True, ""
        operation = spec._operation_for_ref(row.edit_ref)
        if operation is None:
            return False, "This operation is not available."
        script_operation = (
            operation
            if isinstance(
                operation,
                ScriptCodeOperation,
            )
            else None
        )
        if script_operation is not None:
            if script_operation.code is None or not script_operation.copyable:
                return False, "This script step does not contain editable code."
            if spec.kind == "script":
                return True, ""
        if spec.kind == "script":
            input_spec = spec._prefix_before_ref(row.edit_ref)
            if not (
                script_provenance_replayable(input_spec)
                or script_provenance_requires_trust(input_spec)
            ):
                return False, "This script step cannot be replayed."
        active_filter_ref = self._active_filter_ref(node, spec)
        if (
            spec.kind in {"full_data", "public_data", "selection"}
            and row.scope != "source"
            and active_filter_ref != row.edit_ref
            and node.detached_live_parent_data is None
        ):
            return False, "This live row needs a parent source to replay."
        if script_operation is not None:
            return True, ""
        dialog_match = _dialog_match_for_operation_ref(spec, row.edit_ref)
        if dialog_match is None:
            reason = _uneditable_operation_reason(operation)
            return False, reason or "No editing dialog is available for this step."
        if not row.script_input_path and active_filter_ref == row.edit_ref:
            return True, ""
        return True, ""

    def can_revert_row(
        self,
        row: _ProvenanceDisplayRow | None,
    ) -> tuple[bool, str]:
        node = self._metadata_node()
        if row is None or row.replay_ref is None:
            return False, "This row is not replayable."
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to revert provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        if self._source_child_parent_row(node, row):
            return False, "Revert the parent ImageTool row directly."
        if row.replay_ref.kind == "script_input":
            return False, "Dependency input rows are not revert targets."
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            return False, "This row does not have replayable provenance."
        if row.replay_ref.kind == "operation":
            group = _editable_group_range_for_ref(spec, row.replay_ref)
            if group is not None and row.replay_ref.operation_index != group[1] - 1:
                return (
                    False,
                    "Revert is available from the final momentum-conversion row.",
                )
        try:
            candidate = spec._prefix_through_ref(row.replay_ref)
        except ValueError:
            if spec.kind == "script":
                return False, "This script row is not a replayable step."
            return False, "This row is not a replayable step."
        if candidate == spec:
            return False, "Already at this provenance step."
        if spec.kind == "script":
            if not (
                script_provenance_replayable(candidate)
                or script_provenance_requires_trust(candidate)
            ):
                return False, "This script step cannot be replayed."
            if not all(
                self._manager._script_input_can_reload(
                    script_input,
                    target_node_uid=node.uid,
                )
                for script_input in candidate.script_inputs
            ):
                return False, "This script step has unavailable inputs."
            return True, ""
        if spec.kind in {"full_data", "public_data", "selection"}:
            if row.scope == "source" and node.parent_uid is not None:
                return True, ""
            return False, "This live row needs a parent source to replay."
        return True, ""

    def edit_row(
        self,
        row: _ProvenanceDisplayRow | None,
    ) -> None:
        editable, reason = self.can_edit_row(row)
        if not editable:
            self._show_unavailable(reason)
            return
        node = typing.cast(
            "_ImageToolWrapper | _ManagedWindowNode",
            self._metadata_node(),
        )
        row = typing.cast("_ProvenanceDisplayRow", row)
        ref = typing.cast(
            "_ProvenanceStepRef",
            row.edit_ref,
        )
        try:
            if ref.kind == "file_load":
                self._edit_file_load_row(node, row)
            else:
                self._edit_operation_row(node, row)
        except _TrustedScriptReplayCancelled:
            return
        except Exception as exc:
            if self._handle_missing_source_file(
                node,
                row,
                title="Could Not Apply Provenance Edit",
                exc=exc,
            ):
                return
            self._show_failed("Could Not Apply Provenance Edit", exc)

    def revert_row(
        self,
        row: _ProvenanceDisplayRow | None,
    ) -> None:
        revertible, reason = self.can_revert_row(row)
        if not revertible:
            self._show_unavailable(reason)
            return
        if not self._confirm_revert():
            return

        node = typing.cast(
            "_ImageToolWrapper | _ManagedWindowNode",
            self._metadata_node(),
        )
        row = typing.cast("_ProvenanceDisplayRow", row)
        ref = typing.cast(
            "_ProvenanceStepRef",
            row.replay_ref,
        )
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            self._show_failed(
                "Could Not Revert Provenance Step",
                RuntimeError("No provenance spec is available"),
            )
            return
        repair_root_candidate: ToolProvenanceSpec | None = None
        try:
            candidate = spec._prefix_through_ref(ref)
            repair_root_candidate = self._root_candidate_for_row(node, row, candidate)
            self._validate_and_replace(
                node,
                row.scope,
                repair_root_candidate,
                where="validating the provenance revert target",
            )
        except _TrustedScriptReplayCancelled:
            return
        except Exception as exc:
            if self._handle_missing_source_file(
                node,
                row,
                title="Could Not Revert Provenance Step",
                exc=exc,
                repair_spec=repair_root_candidate,
            ):
                return
            self._show_failed("Could Not Revert Provenance Step", exc)

    def delete_row(
        self,
        row: _ProvenanceDisplayRow | None,
    ) -> None:
        deletable, reason = self.can_delete_row(row)
        if not deletable:
            self._show_unavailable(reason)
            return
        node = typing.cast(
            "_ImageToolWrapper | _ManagedWindowNode",
            self._metadata_node(),
        )
        row = typing.cast("_ProvenanceDisplayRow", row)
        ref = typing.cast(
            "_ProvenanceStepRef",
            row.replay_ref,
        )
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            self._show_failed(
                "Could Not Delete Provenance Step",
                RuntimeError("No provenance spec is available"),
            )
            return
        repair_root_candidate: ToolProvenanceSpec | None = None
        try:
            group = _editable_group_range_for_ref(spec, ref)
            if group is None:
                candidate = spec._replace_operation_ref(ref, ())
            else:
                candidate = spec._replace_operation_range_ref(
                    ref,
                    group[0],
                    group[1],
                    (),
                )
            repair_root_candidate = self._root_candidate_for_row(node, row, candidate)
            self._validate_and_replace(
                node,
                row.scope,
                repair_root_candidate,
                where="validating the provenance delete target",
            )
        except _TrustedScriptReplayCancelled:
            return
        except Exception as exc:
            if self._handle_missing_source_file(
                node,
                row,
                title="Could Not Delete Provenance Step",
                exc=exc,
                repair_spec=repair_root_candidate,
            ):
                return
            self._show_failed("Could Not Delete Provenance Step", exc)

    def _metadata_node(self) -> _ImageToolWrapper | _ManagedWindowNode | None:
        uid = self._manager._metadata_node_uid
        if uid is None:
            return None
        return self._manager._tool_graph.nodes.get(uid)

    def _paste_target_nodes(self) -> list[_ImageToolWrapper | _ManagedWindowNode]:
        selected_targets = self._manager._selected_imagetool_targets()
        if selected_targets:
            nodes: list[_ImageToolWrapper | _ManagedWindowNode] = []
            seen_uids: set[str] = set()
            for target in selected_targets:
                try:
                    node = self._manager._node_for_target(target)
                except (KeyError, IndexError):
                    continue
                if node.uid in seen_uids or not self._node_editable(node):
                    continue
                seen_uids.add(node.uid)
                nodes.append(node)
            return nodes

        node = self._metadata_node()
        if not self._node_editable(node):
            return []
        return [typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)]

    @staticmethod
    def _node_editable(
        node: _ImageToolWrapper | _ManagedWindowNode | None,
    ) -> bool:
        return (
            node is not None
            and node.is_imagetool
            and (
                node.imagetool is not None
                or node.pending_workspace_memory_payload is not None
            )
        )

    @staticmethod
    def _source_child_parent_row(
        node: _ImageToolWrapper | _ManagedWindowNode | None,
        row: _ProvenanceDisplayRow,
    ) -> bool:
        return (
            node is not None
            and node.parent_uid is not None
            and node.source_spec is not None
            and row.scope == "display"
        )

    def _root_display_spec_for_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: _ProvenanceDisplayRow,
    ) -> ToolProvenanceSpec | None:
        if row.scope == "source":
            return node.displayed_source_spec
        return node.displayed_provenance_spec

    def _display_spec_for_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: _ProvenanceDisplayRow,
    ) -> ToolProvenanceSpec | None:
        spec = self._root_display_spec_for_row(node, row)
        if spec is None or not row.script_input_path:
            return spec
        return self._script_input_path_spec(spec, row.script_input_path)

    @staticmethod
    def _script_input_path_spec(
        spec: ToolProvenanceSpec,
        path: tuple[int, ...],
    ) -> ToolProvenanceSpec | None:
        current = spec
        for index in path:
            if index < 0 or index >= len(current.script_inputs):
                return None
            nested = current.script_inputs[index].parsed_provenance_spec()
            if nested is None:
                return None
            current = nested
        return current

    def _root_candidate_for_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: _ProvenanceDisplayRow,
        candidate: ToolProvenanceSpec,
    ) -> ToolProvenanceSpec:
        if not row.script_input_path:
            return candidate
        root = self._root_display_spec_for_row(node, row)
        if root is None:
            raise RuntimeError("No root provenance spec is available")
        return self._replace_script_input_path_spec(
            root,
            row.script_input_path,
            candidate,
        )

    def _replace_script_input_path_spec(
        self,
        spec: ToolProvenanceSpec,
        path: tuple[int, ...],
        replacement: ToolProvenanceSpec,
    ) -> ToolProvenanceSpec:
        if not path:
            return replacement
        index = path[0]
        if index < 0 or index >= len(spec.script_inputs):
            raise IndexError("Script input provenance path is not available")
        script_input = spec.script_inputs[index]
        nested = script_input.parsed_provenance_spec()
        if nested is None:
            raise RuntimeError("Script input does not have replayable provenance")
        replaced = self._replace_script_input_path_spec(
            nested,
            path[1:],
            replacement,
        )
        script_inputs = list(spec.script_inputs)
        script_inputs[index] = script_input.model_copy(
            update={
                "node_uid": None,
                "node_snapshot_token": None,
                "provenance_spec": replaced.model_dump(mode="json"),
            }
        )
        return spec.model_copy(update={"script_inputs": tuple(script_inputs)})

    def _replace_file_load_target_spec(
        self,
        root: ToolProvenanceSpec,
        target: _FileLoadBatchPeer,
        replacement: ToolProvenanceSpec,
    ) -> ToolProvenanceSpec:
        if target.spec.kind == "script" and replacement.kind != "script":
            replacement = _replace_file_load_fields(target.spec, replacement)
        if not target.script_input_path:
            return replacement
        return self._replace_script_input_path_spec(
            root,
            target.script_input_path,
            replacement,
        )

    def _file_load_targets(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec | None,
    ) -> tuple[_FileLoadBatchPeer, ...]:
        targets: list[_FileLoadBatchPeer] = []
        self._append_file_load_targets(
            targets,
            node,
            scope,
            spec,
            script_input_path=(),
            display_label=node.display_text,
        )
        return tuple(targets)

    def _append_file_load_targets(
        self,
        targets: list[_FileLoadBatchPeer],
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec | None,
        *,
        script_input_path: tuple[int, ...],
        display_label: str,
    ) -> None:
        if spec is None:
            return
        if spec.kind in {"file", "script"} and spec.file_load_source is not None:
            load_source = spec.file_load_source
            targets.append(
                _FileLoadBatchPeer(
                    node=node,
                    scope=scope,
                    spec=spec,
                    original_path=pathlib.Path(load_source.path).expanduser(),
                    loader_summary=_loader_summary(load_source),
                    script_input_path=script_input_path,
                    display_label=display_label,
                )
            )
            if spec.kind != "script":
                return
        if spec.kind != "script":
            return
        for index, script_input in enumerate(spec.script_inputs):
            nested = script_input.parsed_provenance_spec()
            self._append_file_load_targets(
                targets,
                node,
                scope,
                nested,
                script_input_path=(*script_input_path, index),
                display_label=f"{display_label}: {script_input.label}",
            )

    def _file_load_target_for_path(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec | None,
        path: pathlib.Path,
    ) -> _FileLoadBatchPeer | None:
        target_path = _normalized_path(path)
        for target in self._file_load_targets(node, scope, spec):
            if _normalized_path(target.original_path) == target_path:
                return target
        return None

    def _missing_file_load_repair_peers(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        root_spec: ToolProvenanceSpec | None,
        focused: _FileLoadBatchPeer,
    ) -> tuple[_FileLoadBatchPeer, ...]:
        peers: list[_FileLoadBatchPeer] = []
        for target in self._file_load_targets(node, scope, root_spec):
            if target.target_id == focused.target_id or not target.script_input_path:
                continue
            load_source = target.spec.file_load_source
            replay_call = None if load_source is None else load_source.replay_call
            if load_source is None or replay_call is None:
                continue
            if pathlib.Path(load_source.path).expanduser().exists():
                continue
            peers.append(dataclasses.replace(target, preserve_loader=True))
        return tuple(peers)

    @staticmethod
    def _root_spec_for_batch_peer(
        peer: _FileLoadBatchPeer,
    ) -> ToolProvenanceSpec:
        root = (
            peer.node.displayed_source_spec
            if peer.scope == "source"
            else peer.node.displayed_provenance_spec
        )
        if root is None:
            raise RuntimeError("Matching file load has no root provenance")
        return root

    def _file_load_source_edit_target(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: pathlib.Path,
    ) -> tuple[
        _ImageToolWrapper | _ManagedWindowNode | None,
        typing.Literal["display", "source"] | None,
        _FileLoadBatchPeer | None,
        str,
    ]:
        source_bound = node.parent_uid is not None and node.source_spec is not None
        candidates: tuple[
            tuple[
                typing.Literal["display", "source"],
                ToolProvenanceSpec | None,
            ],
            ...,
        ] = (
            (("source", node.displayed_source_spec),)
            if source_bound
            else (("display", node.displayed_provenance_spec),)
        )

        for scope, spec in candidates:
            target = self._file_load_target_for_path(node, scope, spec, path)
            if target is None:
                continue
            load_source = target.spec.file_load_source
            if load_source is None:
                continue
            if load_source.replay_call is None:
                return None, None, None, "This file load step cannot be replayed."
            return (
                node,
                scope,
                target,
                "Select the current source file and update the recorded "
                "file-load step.",
            )
        if source_bound:
            parent = self._manager._parent_node(node)
            parent_node, parent_scope, parent_spec, parent_reason = (
                self._file_load_source_edit_target(parent, path)
            )
            if parent_node is not None and parent_scope is not None:
                return parent_node, parent_scope, parent_spec, parent_reason
            return None, None, None, parent_reason
        return (
            None,
            None,
            None,
            "This source was not recorded as an editable file-load step.",
        )

    def can_edit_file_load_source(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: pathlib.Path,
    ) -> tuple[bool, str]:
        if not self._node_editable(node):
            return False, "This ImageTool is not available for editing."
        _node, _scope, target, reason = self._file_load_source_edit_target(node, path)
        return target is not None, reason

    def edit_file_load_source(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: pathlib.Path,
    ) -> None:
        if not self._node_editable(node):
            self._show_unavailable("This ImageTool is not available for editing.")
            return
        edit_node, scope, target, reason = self._file_load_source_edit_target(
            node, path
        )
        if edit_node is None or scope is None or target is None:
            self._show_unavailable(reason)
            return
        row = (
            None
            if not target.script_input_path
            else _ProvenanceDisplayRow(
                DerivationEntry("File load", None),
                scope=scope,
                script_input_path=target.script_input_path,
            )
        )
        try:
            self._edit_file_load_spec(
                edit_node,
                scope,
                target.spec,
                where="validating the edited file-load provenance",
                row=row,
                batch_peers=self._file_load_batch_peers(
                    edit_node, target.spec, row=row
                ),
            )
        except Exception as exc:
            if isinstance(exc, _ProvenanceReplayFailure):
                missing = exc.missing_source_file
            else:
                missing = (
                    exc if isinstance(exc, _MissingProvenanceSourceFileError) else None
                )
            if missing is not None:
                self._show_missing_source_file(
                    "Could Not Update Source File",
                    exc,
                    missing,
                    can_edit=False,
                )
                return
            self._show_failed("Could Not Update Source File", exc)

    def _edit_file_load_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: _ProvenanceDisplayRow,
    ) -> None:
        spec = self._display_spec_for_row(node, row)
        if (
            spec is None
            or spec.kind not in {"file", "script"}
            or spec.file_load_source is None
        ):
            raise RuntimeError("Selected row is not a file load step")
        self._edit_file_load_spec(
            node,
            row.scope,
            spec,
            where="validating the edited file-load provenance",
            row=row,
            batch_peers=(
                self._file_load_batch_peers(node, spec, row=row)
                if row.script_input_path
                else self._file_load_batch_peers(node, spec)
            ),
        )

    def _edit_file_load_spec(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
        *,
        where: str,
        row: _ProvenanceDisplayRow | None = None,
        batch_peers: Sequence[_FileLoadBatchPeer] = (),
        batch_apply_default: bool = False,
        checked_batch_peer_ids: frozenset[str] | None = None,
        root_spec: ToolProvenanceSpec | None = None,
    ) -> None:
        if spec.kind not in {"file", "script"} or spec.file_load_source is None:
            raise RuntimeError("Selected provenance does not have a file load step")
        dialog = _FileLoadEditDialog(
            spec.file_load_source,
            self._manager,
            batch_peers=batch_peers,
            batch_apply_default=batch_apply_default,
            checked_batch_peer_ids=checked_batch_peer_ids,
        )
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        candidate = dialog.provenance_spec(
            active_name=_file_load_edit_active_name(spec),
            replay_steps=spec.steps,
        )
        if spec.kind == "script":
            candidate = _replace_file_load_fields(spec, candidate)
        if row is None:
            edit_candidate = candidate
        elif root_spec is None:
            edit_candidate = self._root_candidate_for_row(node, row, candidate)
        else:
            edit_candidate = self._replace_script_input_path_spec(
                root_spec,
                row.script_input_path,
                candidate,
            )
        peer_edits = [
            (peer, dialog.peer_provenance_spec(peer))
            for peer in dialog.selected_batch_peers()
        ]
        deferred_peer_edits: list[
            tuple[
                _FileLoadBatchPeer,
                ToolProvenanceSpec,
            ]
        ] = []
        for peer, peer_candidate in peer_edits:
            if peer.node.uid == node.uid and peer.scope == scope:
                edit_candidate = self._replace_file_load_target_spec(
                    edit_candidate,
                    peer,
                    peer_candidate,
                )
            else:
                deferred_peer_edits.append((peer, peer_candidate))
        validated_edits = [
            self._validated_edit(node, scope, edit_candidate, where=where),
        ]
        failures: list[tuple[_FileLoadBatchPeer, Exception]] = []
        for peer, peer_candidate in deferred_peer_edits:
            try:
                peer_where = (
                    "validating the edited file-load provenance for "
                    f"{peer.node.display_text}"
                )
                if peer.script_input_path:
                    peer_candidate = self._replace_file_load_target_spec(
                        self._root_spec_for_batch_peer(peer),
                        peer,
                        peer_candidate,
                    )
                validated_edits.append(
                    self._validated_edit(
                        peer.node,
                        peer.scope,
                        peer_candidate,
                        where=peer_where,
                    )
                )
            except Exception as exc:
                failures.append((peer, exc))
        if failures and not self._confirm_apply_valid_batch(
            valid_peer_count=len(validated_edits) - 1,
            failures=failures,
        ):
            return
        for edit in validated_edits:
            self._apply_validated_edit(edit)

    def _edit_operation_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: _ProvenanceDisplayRow,
    ) -> None:
        ref = typing.cast(
            "_ProvenanceStepRef",
            row.edit_ref,
        )
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            raise RuntimeError("No provenance spec is available")
        operation = spec._operation_for_ref(ref)
        if operation is None:
            raise RuntimeError("Selected operation is not available")
        if isinstance(
            operation,
            ScriptCodeOperation,
        ):
            self._edit_script_code_operation_row(node, row, spec, ref, operation)
            return
        dialog_match = _dialog_match_for_operation_ref(spec, ref)
        if dialog_match is None:
            raise RuntimeError("No editing dialog is available for this step")
        if not row.script_input_path and self._active_filter_ref(node, spec) == ref:
            self._edit_active_filter(node, operation, dialog_match.dialog_cls)
            return

        replacements = self._edited_native_operations(
            node,
            row,
            spec,
            ref,
            dialog_match,
        )
        if replacements is None:
            return
        candidate = spec._replace_operation_range_ref(
            ref,
            dialog_match.start,
            dialog_match.stop,
            replacements,
        )
        root_candidate = self._root_candidate_for_row(node, row, candidate)
        self._validate_and_replace(
            node,
            row.scope,
            root_candidate,
            where="validating the edited provenance step",
        )

    def _edit_script_code_operation_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: _ProvenanceDisplayRow,
        spec: ToolProvenanceSpec,
        ref: _ProvenanceStepRef,
        operation: ScriptCodeOperation,
    ) -> None:
        dialog = _ScriptCodeEditDialog(operation, self._manager)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        replacement = operation.model_copy(update={"code": dialog.code()})
        candidate = spec._replace_operation_ref(ref, (replacement,))
        root_candidate = self._root_candidate_for_row(node, row, candidate)
        self._validate_and_replace(
            node,
            row.scope,
            root_candidate,
            where="validating the edited Python code",
        )

    def _edited_native_operations(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: _ProvenanceDisplayRow,
        spec: ToolProvenanceSpec,
        ref: _ProvenanceStepRef,
        dialog_match: _OperationDialogMatch,
    ) -> list[ToolProvenanceOperation] | None:
        operations = tuple(spec.operations[dialog_match.start : dialog_match.stop])
        if not operations:
            raise ValueError("No provenance operations were provided for editing")
        start_ref = _ProvenanceStepRef(
            "operation",
            operation_index=dialog_match.start,
        )
        prefix_data = self._native_edit_seed_data_without_replay(
            node,
            spec,
            ref,
            dialog_match,
            operations,
        )
        if prefix_data is None:
            try:
                prefix_data, _prefix = self._replay_candidate_result(
                    node,
                    row.scope,
                    spec._prefix_before_ref(start_ref),
                )
            except _TrustedScriptReplayCancelled:
                raise
            except Exception as exc:
                raise _ProvenanceReplayFailure(
                    "opening the provenance editor: replaying the input to this step",
                    exc,
                ) from exc

        manager: object = self._manager
        dialog_parent = manager if isinstance(manager, QtWidgets.QWidget) else None
        temp_tool = None
        try:
            if issubclass(dialog_match.dialog_cls, dialogs.SelectionDialog):
                selection_dialog_cls = dialog_match.dialog_cls
                dialog = selection_dialog_cls(
                    provenance_edit_mode=True,
                    dialog_parent=dialog_parent,
                    source_data=prefix_data,
                )
            else:
                temp_tool = erlab.interactive.imagetool.ImageTool(prefix_data)
                temp_tool.hide()
                dialog = dialog_match.dialog_cls(
                    temp_tool.slicer_area,
                    provenance_edit_mode=True,
                    dialog_parent=dialog_parent,
                )
            self._restore_native_edit_dialog(dialog, operations, dialog_match.focus)
            if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
                return None
            return dialog.provenance_edit_operations()
        finally:
            if temp_tool is not None:
                temp_tool.close()
                temp_tool.deleteLater()

    def _native_edit_seed_data_without_replay(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: ToolProvenanceSpec,
        ref: _ProvenanceStepRef,
        dialog_match: _OperationDialogMatch,
        operations: Sequence[ToolProvenanceOperation],
    ) -> xr.DataArray | None:
        for seed_builder in (
            self._terminal_current_data_edit_seed,
            self._terminal_affine_coord_edit_seed,
        ):
            seed_data = seed_builder(node, spec, ref, dialog_match, operations)
            if seed_data is not None:
                return seed_data
        return None

    def _terminal_current_data_edit_seed(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: ToolProvenanceSpec,
        ref: _ProvenanceStepRef,
        dialog_match: _OperationDialogMatch,
        operations: Sequence[ToolProvenanceOperation],
    ) -> xr.DataArray | None:
        if len(operations) != 1:
            return None
        if dialog_match.stop != len(spec.operations):
            return None
        operation = operations[0]
        if not any(
            issubclass(dialog_match.dialog_cls, dialog_cls)
            and isinstance(operation, operation_cls)
            for dialog_cls, operation_cls in _NATIVE_TERMINAL_CURRENT_DATA_EDITORS
        ):
            return None
        try:
            data = node.current_source_data().copy(deep=False)
        except Exception:
            return None
        if isinstance(
            operation,
            NormalizeOperation,
        ) and not set(operation.dims).issubset(data.dims):
            return None
        if isinstance(
            operation,
            GaussianFilterOperation,
        ):
            if not set(operation.sigma).issubset(data.dims):
                return None
            try:
                for dim in operation.sigma:
                    coord = np.asarray(data[dim].values, dtype=np.float64)
                    if (
                        coord.size < 2
                        or np.allclose(np.diff(coord), 0.0)
                        or not erlab.utils.array.is_uniform_spaced(coord)
                    ):
                        return None
            except Exception:
                return None
        if isinstance(
            operation,
            DivideByCoordOperation,
        ):
            if operation.coord_name not in data.coords:
                return None
            coord = data.coords[operation.coord_name]
            if (
                not set(coord.dims).issubset(data.dims)
                or not np.issubdtype(coord.dtype, np.number)
                or np.issubdtype(coord.dtype, np.complexfloating)
            ):
                return None
        if isinstance(
            operation,
            SortByOperation,
        ):
            sort_keys = set(data.dims)
            for name, coord in data.coords.items():
                if (
                    coord.ndim == 1
                    and len(coord.dims) == 1
                    and coord.dims[0] in data.dims
                ):
                    sort_keys.add(name)
            if not all(key in sort_keys for key in operation.variables):
                return None
        return data

    def _terminal_affine_coord_edit_seed(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: ToolProvenanceSpec,
        ref: _ProvenanceStepRef,
        dialog_match: _OperationDialogMatch,
        operations: Sequence[ToolProvenanceOperation],
    ) -> xr.DataArray | None:
        if not issubclass(dialog_match.dialog_cls, dialogs.AssignCoordsDialog):
            return None
        if len(operations) != 1:
            return None
        operation = operations[0]
        if not isinstance(
            operation,
            AffineCoordOperation,
        ):
            return None
        if operation.scale == 0.0:
            return None

        if dialog_match.stop != len(spec.operations):
            return None

        try:
            data = node.current_source_data().copy(deep=False)
            coord = data.coords[operation.coord_name]
            coord_values = np.asarray(coord.values)
            if not np.issubdtype(coord_values.dtype, np.number) or np.issubdtype(
                coord_values.dtype,
                np.complexfloating,
            ):
                return None
            old_coord_values = (coord_values - operation.offset) / operation.scale
            if not np.all(np.isfinite(old_coord_values)):
                return None
            return data.assign_coords(
                {operation.coord_name: coord.copy(data=old_coord_values)}
            )
        except Exception:
            return None

    @staticmethod
    def _restore_native_edit_dialog(
        dialog: dialogs._DataManipulationDialog,
        operations: Sequence[ToolProvenanceOperation],
        focus: str | None,
    ) -> None:
        if isinstance(dialog, dialogs.DataTransformDialog):
            dialog.restore_transform_operations(operations)
            dialog.focus_operation_group_control(focus)
            return
        if isinstance(dialog, dialogs.DataFilterDialog):
            if len(operations) != 1:
                raise ValueError("Filter edit dialogs can only restore one operation")
            dialog.restore_filter_operation(operations[0])
            return
        raise TypeError("Provenance edits require a transform or filter dialog")

    def _edit_active_filter(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        operation: ToolProvenanceOperation,
        dialog_cls: type[dialogs._DataManipulationDialog],
    ) -> None:
        if not node.materialize_pending_workspace_payload():
            raise RuntimeError(
                "Could not read this ImageTool's saved data from the workspace file."
            )
        if node.imagetool is None or not issubclass(
            dialog_cls,
            dialogs.DataFilterDialog,
        ):
            raise RuntimeError("Active display filter is not editable")
        dialog = dialog_cls(node.slicer_area)
        dialog.restore_filter_operation(operation)
        if dialog.exec() == int(QtWidgets.QDialog.DialogCode.Accepted):
            self._manager._update_info(uid=node.uid)

    def _validate_and_replace(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
        *,
        where: str = "validating the provenance change",
    ) -> None:
        self._apply_validated_edit(
            self._validated_edit(node, scope, candidate, where=where)
        )

    def _validated_edit(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
        *,
        where: str,
    ) -> _ValidatedProvenanceEdit:
        base_candidate, filter_operation = self._split_active_filter(node, candidate)
        try:
            source_data, base_candidate = self._replay_candidate_result(
                node,
                scope,
                base_candidate,
            )
        except _TrustedScriptReplayCancelled:
            raise
        except Exception as exc:
            replay_target = (
                "provenance before the active display filter"
                if filter_operation is not None
                else "the requested provenance"
            )
            raise _ProvenanceReplayFailure(
                f"{where}: replaying {replay_target}",
                exc,
            ) from exc
        if filter_operation is not None:
            self._validate_filter_operation(
                node,
                source_data,
                filter_operation,
                where=where,
            )
        return _ValidatedProvenanceEdit(
            node=node,
            scope=scope,
            data=source_data,
            spec=base_candidate,
            filter_operation=filter_operation,
        )

    def _validate_filter_operation(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        data: xr.DataArray,
        operation: ToolProvenanceOperation,
        *,
        where: str,
    ) -> None:
        try:
            slicer_area = getattr(node, "slicer_area", None)
            filter_result = getattr(
                slicer_area,
                "_filter_operation_result_for_replacement",
                None,
            )
            if node.imagetool is not None and callable(filter_result):
                filter_result(data, operation)
                return
            operation.apply(data)
        except Exception as exc:
            raise _ProvenanceReplayFailure(
                f"{where}: validating the active display filter",
                exc,
            ) from exc

    def _apply_validated_edit(self, edit: _ValidatedProvenanceEdit) -> None:
        self._replace_node_data(
            edit.node,
            edit.scope,
            edit.data,
            edit.spec,
            edit.filter_operation,
        )
        self._manager._update_info(uid=edit.node.uid)

    def _file_load_batch_peers(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: ToolProvenanceSpec,
        *,
        row: _ProvenanceDisplayRow | None = None,
    ) -> tuple[_FileLoadBatchPeer, ...]:
        load_source = spec.file_load_source
        replay_call = None if load_source is None else load_source.replay_call
        if load_source is None or replay_call is None:
            return ()
        source_folder = _normalized_path(pathlib.Path(load_source.path).parent)
        peers: list[_FileLoadBatchPeer] = []
        if row is not None and row.script_input_path:
            root = self._root_display_spec_for_row(node, row)
            for target in self._file_load_targets(node, row.scope, root):
                if target.script_input_path == row.script_input_path:
                    continue
                peer_load_source = target.spec.file_load_source
                peer_replay_call = (
                    None if peer_load_source is None else peer_load_source.replay_call
                )
                if peer_load_source is None or peer_replay_call is None:
                    continue
                if (
                    _normalized_path(pathlib.Path(peer_load_source.path).parent)
                    != source_folder
                ):
                    continue
                if not _same_replay_loader(replay_call, peer_replay_call):
                    continue
                peers.append(target)

        for peer_node in self._manager._tool_graph.nodes.values():
            if (
                peer_node.uid == node.uid
                or not self._node_editable(peer_node)
                or (
                    peer_node.parent_uid is not None
                    and peer_node.source_spec is not None
                )
            ):
                continue
            peer_spec = peer_node.displayed_provenance_spec
            if peer_spec is None or peer_spec.kind not in {"file", "script"}:
                continue
            peer_load_source = peer_spec.file_load_source
            peer_replay_call = (
                None if peer_load_source is None else peer_load_source.replay_call
            )
            if peer_load_source is None or peer_replay_call is None:
                continue
            if (
                _normalized_path(pathlib.Path(peer_load_source.path).parent)
                != source_folder
            ):
                continue
            if not _same_replay_loader(replay_call, peer_replay_call):
                continue
            peers.append(
                _FileLoadBatchPeer(
                    node=peer_node,
                    scope="display",
                    spec=peer_spec,
                    original_path=pathlib.Path(peer_load_source.path).expanduser(),
                    loader_summary=_loader_summary(peer_load_source),
                )
            )
        return tuple(peers)

    def _replay_candidate(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
    ) -> xr.DataArray:
        data, _spec = self._replay_candidate_result(node, scope, spec)
        return data

    def _replay_candidate_result(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        if spec.kind == "file":
            return self._replay_file_candidate(spec), spec
        if spec.kind in {"full_data", "public_data", "selection"}:
            if any(
                isinstance(
                    operation,
                    ScriptCodeOperation,
                )
                for operation in spec.operations
            ):
                return self._replay_live_script_candidate(node, scope, spec), spec
            if scope == "source" and node.parent_uid is not None:
                parent = self._manager._parent_node(node)
                return spec.apply(parent.current_source_data()), spec
            parent_data = node.detached_live_parent_data
            if parent_data is None:
                raise RuntimeError("Live provenance needs a parent source to replay")
            return spec.apply(parent_data), spec
        if spec.kind == "script":
            result = self._manager._rebuild_script_provenance(
                spec,
                target_node_uid=node.uid,
            )
            return result.data, result.provenance_spec
        raise RuntimeError("Unsupported provenance kind")

    def _replay_live_script_candidate(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
    ) -> xr.DataArray:
        if scope == "source" and node.parent_uid is not None:
            parent = self._manager._parent_node(node)
            parent_data = parent.current_source_data()
        else:
            parent_data = node.detached_live_parent_data
            if parent_data is None:
                raise RuntimeError("Live provenance needs a parent source to replay")
        data = ToolProvenanceSpec._starting_data_for_kind(
            typing.cast(
                "typing.Literal['full_data', 'public_data', 'selection']",
                spec.kind,
            ),
            parent_data,
        )
        for operation in spec.operations:
            if not isinstance(
                operation,
                ScriptCodeOperation,
            ):
                data = operation._apply_schema_v2(data, parent_data=parent_data)
                continue
            step_spec = self._live_script_step_spec(operation)
            replay_inputs = {
                "data": data,
                "derived": data,
                "parent_data": parent_data,
            }
            trusted_user_code = script_provenance_requires_trust(
                step_spec,
                external_input_names=set(replay_inputs),
            )
            if trusted_user_code:
                self._manager._ensure_script_provenance_trusted(
                    step_spec,
                    reason="apply this provenance step",
                    external_input_names=set(replay_inputs),
                )
            data = replay_script_provenance(
                step_spec,
                replay_inputs,
                trusted_user_code=trusted_user_code,
            )
        return data

    @staticmethod
    def _live_script_step_spec(
        operation: ScriptCodeOperation,
    ) -> ToolProvenanceSpec:
        return script(
            operation,
            start_label=operation.label,
            seed_code="derived = data",
            active_name="derived",
        )

    def _replay_file_candidate(
        self,
        spec: ToolProvenanceSpec,
    ) -> xr.DataArray:
        load_source = spec.file_load_source
        if load_source is not None:
            source_path = pathlib.Path(load_source.path)
            if not source_path.exists():
                raise _MissingProvenanceSourceFileError(source_path)
        with warnings.catch_warnings(record=True) as replay_warnings:
            warnings.simplefilter("always")
            try:
                return replay_file_provenance(spec)
            except Exception as exc:
                if warning_details := _replay_warning_details(replay_warnings):
                    exc.add_note(
                        "Warnings emitted while replaying provenance:\n"
                        f"{warning_details}"
                    )
                raise

    def _replace_node_data(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        data: xr.DataArray,
        spec: ToolProvenanceSpec,
        filter_operation: ToolProvenanceOperation | None,
    ) -> None:
        if not node.materialize_pending_workspace_payload():
            raise RuntimeError(
                "Could not read this ImageTool's saved data from the workspace file."
            )
        if node.imagetool is None:
            raise RuntimeError("ImageTool is not available for editing")
        preserve_filter = filter_operation is not None
        if scope == "source":
            live_spec = require_live_source_spec(spec)
            if live_spec is None or node.parent_uid is None:
                raise RuntimeError("Source-bound edits need live source provenance")
            parent = self._manager._parent_node(node)
            parent_data = parent.current_source_data()
            displayed = compose_display_provenance(
                parent.displayed_provenance_spec,
                live_spec,
                parent_data=parent_data,
            )
            node.set_source_binding(
                live_spec,
                auto_update=node.source_auto_update,
                state="fresh",
                provenance_spec=displayed,
            )
            node._replace_imagetool_data(
                data,
                displayed,
                propagate_descendants=True,
                preserve_filter=preserve_filter,
            )
            return
        live_parent_data = None
        if spec.kind in {"full_data", "public_data", "selection"}:
            live_parent_data = node.detached_live_parent_data
        node.replace_with_detached_data(
            data,
            spec,
            preserve_filter=preserve_filter,
            live_parent_data=live_parent_data,
        )

    def _active_filter_ref(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: ToolProvenanceSpec,
    ) -> _ProvenanceStepRef | None:
        if node.imagetool is None:
            return None
        active = node.slicer_area._accepted_filter_provenance_operation
        if active is None:
            return None
        for ref, operation in reversed(tuple(iter_operation_refs(spec))):
            if operation == active:
                return ref
        return None

    def _split_active_filter(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: ToolProvenanceSpec,
    ) -> tuple[
        ToolProvenanceSpec,
        ToolProvenanceOperation | None,
    ]:
        active_ref = self._active_filter_ref(node, spec)
        if active_ref is None:
            return spec, None
        active_operation = spec._operation_for_ref(active_ref)
        if active_operation is None:
            return spec, None
        base_spec = spec._replace_operation_ref(active_ref, ())
        return base_spec, active_operation

    def _confirm_revert(self) -> bool:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Revert Provenance")
        msg_box.setText("Revert to the selected provenance step?")
        msg_box.setInformativeText(
            "Later provenance steps for this ImageTool will be dropped."
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        return msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes

    def _show_unavailable(self, reason: str) -> None:
        QtWidgets.QMessageBox.information(
            self._manager,
            "Provenance Step Unavailable",
            reason,
        )

    def _show_partial_paste_failures(
        self,
        pasted_count: int,
        failures: Sequence[tuple[_ImageToolWrapper | _ManagedWindowNode, Exception]],
    ) -> None:
        failed_labels = "\n".join(
            f"- {node.display_text}: {exc}" for node, exc in failures
        )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title="Some Provenance Steps Could Not Be Pasted",
            text=(
                f"Pasted provenance steps into {pasted_count} selected ImageTool "
                f"{'row' if pasted_count == 1 else 'rows'}."
            ),
            informative_text=(
                f"{len(failures)} selected ImageTool "
                f"{'row was' if len(failures) == 1 else 'rows were'} left unchanged."
            ),
            detailed_text=failed_labels,
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        dialog.exec()

    def _show_failed(
        self,
        title: str,
        exc: Exception,
        *,
        text: str = "The provenance change could not be applied.",
        unchanged_reason: str | None = None,
    ) -> None:
        exc_text = "".join(traceback.TracebackException.from_exception(exc).format())
        where = (
            exc.where
            if isinstance(exc, _ProvenanceReplayFailure)
            else "replaying the requested provenance"
        )
        if unchanged_reason is None:
            unchanged_reason = (
                "The current ImageTool data was left unchanged because the requested "
                "provenance could not be replayed. Use Revert to This Step to drop "
                "later provenance, or adjust the earlier steps so the full chain is "
                "valid again."
            )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title=title,
            text=text,
            informative_text=f"Failed while: {where}\n\n{unchanged_reason}",
            detailed_text=erlab.interactive.utils._format_traceback(exc_text),
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        dialog.exec()

    def _handle_missing_source_file(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: _ProvenanceDisplayRow,
        *,
        title: str,
        exc: Exception,
        repair_spec: ToolProvenanceSpec | None = None,
    ) -> bool:
        root_spec = repair_spec
        if (
            root_spec is not None
            and row.script_input_path
            and self._script_input_path_spec(root_spec, row.script_input_path) is None
        ):
            root_spec = self._root_candidate_for_row(node, row, root_spec)
        if root_spec is None:
            root_spec = self._root_display_spec_for_row(node, row)

        missing = _missing_source_file_error_from_exception(exc)
        if missing is None:
            missing_path = _file_not_found_path_from_exception(exc)
            if missing_path is None:
                return False
            missing_target = self._file_load_target_for_path(
                node,
                row.scope,
                root_spec,
                missing_path,
            )
            if missing_target is None:
                return False
            missing = _MissingProvenanceSourceFileError(missing_target.original_path)

        target = self._file_load_target_for_path(
            node,
            row.scope,
            root_spec,
            missing.source_path,
        )
        if target is None:
            self._show_missing_source_file(title, exc, missing, can_edit=False)
            return True
        repair_target = target
        prompt_title = title
        while self._show_missing_source_file(
            prompt_title,
            exc,
            missing,
            can_edit=True,
        ):
            try:
                target = (
                    self._file_load_target_for_path(
                        node,
                        row.scope,
                        root_spec,
                        missing.source_path,
                    )
                    or repair_target
                )
                repair_target = target
                repair_row = (
                    None
                    if not target.script_input_path
                    else _ProvenanceDisplayRow(
                        DerivationEntry("File load", None),
                        scope=row.scope,
                        script_input_path=target.script_input_path,
                    )
                )
                matching_peers = self._file_load_batch_peers(
                    node,
                    target.spec,
                    row=repair_row,
                )
                required_repair_peers = self._missing_file_load_repair_peers(
                    node,
                    row.scope,
                    root_spec,
                    target,
                )
                batch_peer_by_id = {
                    peer.target_id: peer for peer in required_repair_peers
                }
                for peer in matching_peers:
                    batch_peer_by_id.setdefault(peer.target_id, peer)
                checked_batch_peer_ids = frozenset(
                    peer.target_id for peer in required_repair_peers
                )
                self._edit_file_load_spec(
                    node,
                    row.scope,
                    target.spec,
                    where=(
                        "validating provenance after selecting a replacement "
                        "source file"
                    ),
                    row=repair_row,
                    batch_peers=tuple(batch_peer_by_id.values()),
                    batch_apply_default=bool(checked_batch_peer_ids),
                    checked_batch_peer_ids=checked_batch_peer_ids,
                    root_spec=root_spec,
                )
                break
            except Exception as repair_exc:
                if isinstance(repair_exc, _ProvenanceReplayFailure):
                    repair_missing = repair_exc.missing_source_file
                else:
                    repair_missing = (
                        repair_exc
                        if isinstance(
                            repair_exc,
                            _MissingProvenanceSourceFileError,
                        )
                        else None
                    )
                if repair_missing is None:
                    self._show_failed("Could Not Update Source File", repair_exc)
                    break
                prompt_title = "Could Not Update Source File"
                exc = repair_exc
                missing = repair_missing
        return True

    def _show_missing_source_file(
        self,
        title: str,
        exc: Exception,
        missing: _MissingProvenanceSourceFileError,
        *,
        can_edit: bool,
    ) -> bool:
        where = (
            exc.where
            if isinstance(exc, _ProvenanceReplayFailure)
            else "replaying the requested provenance"
        )
        exc_text = "".join(traceback.TracebackException.from_exception(exc).format())
        buttons = QtWidgets.QDialogButtonBox.StandardButton.Ok
        if can_edit:
            buttons = (
                QtWidgets.QDialogButtonBox.StandardButton.Yes
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title=title,
            text="The recorded source file is no longer accessible.",
            informative_text=(
                f"Failed while: {where}\n\n"
                f"Missing file: {missing.source_path}\n\n"
                "Update the file load step and try again."
            ),
            detailed_text=erlab.interactive.utils._format_traceback(exc_text),
            buttons=buttons,
            default_button=(
                QtWidgets.QDialogButtonBox.StandardButton.Yes
                if can_edit
                else QtWidgets.QDialogButtonBox.StandardButton.Ok
            ),
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        if can_edit:
            edit_button = dialog._button_box.button(
                QtWidgets.QDialogButtonBox.StandardButton.Yes
            )
            if edit_button is not None:
                edit_button.setText("Edit File Load…")
        result = dialog.exec()
        return can_edit and result == int(QtWidgets.QDialog.DialogCode.Accepted)

    def _confirm_apply_valid_batch(
        self,
        *,
        valid_peer_count: int,
        failures: Sequence[tuple[_FileLoadBatchPeer, Exception]],
    ) -> bool:
        details: list[str] = []
        for peer, exc in failures:
            details.append(f"{peer.node.display_text}\n{peer.original_path}")
            details.append(
                "".join(traceback.TracebackException.from_exception(exc).format())
            )
        valid_peer_text = (
            f" and {valid_peer_count} matching ImageTool(s)" if valid_peer_count else ""
        )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title="Some File Load Edits Failed",
            text="Some matching ImageTools could not be updated.",
            informative_text=(
                f"The current ImageTool{valid_peer_text} "
                "can be updated. Failed ImageTools will be left unchanged."
            ),
            detailed_text=erlab.interactive.utils._format_traceback(
                "\n\n".join(details)
            ),
            buttons=(
                QtWidgets.QDialogButtonBox.StandardButton.Yes
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            ),
            default_button=QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        apply_button = dialog._button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Yes
        )
        if apply_button is not None:
            apply_button.setText("Apply Valid Tools")
        return dialog.exec() == int(QtWidgets.QDialog.DialogCode.Accepted)


def _replay_warning_details(
    replay_warnings: list[warnings.WarningMessage],
) -> str:
    lines: list[str] = []
    for warning in replay_warnings:
        category_name = warning.category.__name__
        message = str(warning.message).strip()
        if not message:
            continue
        indented_message = "\n  ".join(message.splitlines())
        lines.append(f"- {category_name}: {indented_message}")
    return "\n".join(lines)
