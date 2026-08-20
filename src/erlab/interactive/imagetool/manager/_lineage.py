from __future__ import annotations

import itertools
import json
import logging
import pathlib
import traceback
import typing

import numpy as np
from qtpy import QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool._provenance._execution import (
    _memoized_live_input_resolver,
    _replay_capability,
    can_reload_without_trust,
    file_load_source_status,
    rebuild_script_inputs,
    rebuild_script_provenance,
    script_provenance_requires_trust,
    script_provenance_trust_key,
)
from erlab.interactive.imagetool._provenance._graph import (
    LiveInputResolver,
    ReplayGraphError,
)
from erlab.interactive.imagetool._provenance._model import (
    ScriptInput,
    ScriptInputDataRole,
    ScriptInputDependencyRef,
    ToolProvenanceSpec,
    compose_full_provenance,
    has_file_load_source,
    rebase_script_input_node_uids,
    rebase_script_inputs_node_uids,
    script,
    to_replay_provenance_spec,
)
from erlab.interactive.imagetool._provenance._operations import ScriptCodeOperation
from erlab.interactive.imagetool.manager._dependency import _combine_source_states
from erlab.interactive.imagetool.manager._node_change import _ManagedNodeChange
from erlab.interactive.imagetool.manager._widgets import (
    _DEPENDENCY_STATUS_BADGES,
    _DEPENDENCY_STATUS_LABELS,
    _DEPENDENCY_STATUS_TOOLTIPS,
    _ScriptRebuildError,
    _ScriptRebuildResult,
    _TrustedScriptReplayCancelled,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import xarray as xr

    from erlab.interactive.imagetool.manager._dependency import _DependencyStatus
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager


logger = logging.getLogger(__name__)

_InputResolutionAction = tuple[typing.Literal["reload", "refresh", "apply"], str]


class _InputResolutionPlan(typing.NamedTuple):
    target_uid: str | None
    script_inputs: tuple[ScriptInput, ...]
    actions: tuple[_InputResolutionAction, ...]
    live_uids: frozenset[str]
    unavailable_reason: str | None


class _LineageController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager
        self._tool_input_refresh_uids: set[str] = set()

    def _ensure_script_provenance_trusted(
        self,
        spec: ToolProvenanceSpec,
        *,
        reason: str,
        external_input_names: set[str] | None = None,
        live_input_resolver: LiveInputResolver | None = None,
    ) -> bool:
        resolve_live = _memoized_live_input_resolver(live_input_resolver)
        if not script_provenance_requires_trust(
            spec,
            external_input_names=external_input_names,
            live_input_resolver=resolve_live,
        ):
            return False
        trust_key = script_provenance_trust_key(
            spec,
            external_input_names=external_input_names,
            live_input_resolver=resolve_live,
        )
        if (
            trust_key is not None
            and trust_key in self._manager._trusted_script_replay_keys
        ):
            return True
        if not self._prompt_trusted_script_replay(spec, reason=reason):
            raise _TrustedScriptReplayCancelled
        if trust_key is not None:
            self._manager._trusted_script_replay_keys.add(trust_key)
        return True

    def _prompt_trusted_script_replay(
        self,
        spec: ToolProvenanceSpec,
        *,
        reason: str,
    ) -> bool:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setObjectName("managerTrustedScriptReplayDialog")
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Run Recorded Python Code")
        msg_box.setText("Run recorded Python code?")
        msg_box.setInformativeText(
            f"ImageTool cannot verify this recorded code as safe to replay "
            f"automatically. It needs to run Python code to {reason}."
        )
        if code := spec.derivation_code():
            msg_box.setDetailedText(code)
        run_button = msg_box.addButton(
            "Run Code", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(typing.cast("QtWidgets.QPushButton", cancel_button))
        msg_box.exec()
        return msg_box.clickedButton() is run_button

    def _dependency_refs_for_uid(
        self, uid: str
    ) -> tuple[ScriptInputDependencyRef, ...]:
        return self._manager._dependency_tracker.refs_for_uid(uid)

    def dependency_status_for_uid(self, uid: str) -> _DependencyStatus | None:
        return self._manager._dependency_tracker.status_for_uid(uid)

    def dependency_status_label_for_uid(self, uid: str) -> str | None:
        status = self._manager.dependency_status_for_uid(uid)
        if status is None:
            return None
        return _DEPENDENCY_STATUS_LABELS[status]

    def dependency_status_badge_for_uid(self, uid: str) -> str | None:
        status = self._manager.dependency_status_for_uid(uid)
        if status is None:
            return None
        return _DEPENDENCY_STATUS_BADGES.get(status)

    def dependency_status_tooltip_for_uid(self, uid: str) -> str | None:
        status = self._manager.dependency_status_for_uid(uid)
        if status is None:
            return None
        tooltip = _DEPENDENCY_STATUS_TOOLTIPS[status]
        node = self._manager._tool_graph.nodes.get(uid)
        if node is not None and self._node_can_reload_script_inputs(node):
            tooltip += " Click for Reload Data options."
        if status == "missing" and self._missing_dependencies_have_recorded_file(uid):
            tooltip += " Recorded source files found for at least one missing input."
        return tooltip

    def dependency_input_summary_for_uid(self, uid: str) -> str | None:
        refs = self._dependency_refs_for_uid(uid)
        if not refs:
            return None

        script_inputs = self._dependency_script_inputs(
            self._manager._tool_graph.nodes.get(uid)
        )
        parts: list[str] = []
        seen: set[tuple[str, str, str | None, str]] = set()
        for ref in refs:
            key = (
                ref.name,
                ref.node_uid,
                ref.node_snapshot_token,
                ref.data_role,
            )
            if key in seen:
                continue
            seen.add(key)
            parent = self._manager._tool_graph.nodes.get(ref.node_uid)
            if isinstance(parent, _ImageToolWrapper):
                current = f"currently ImageTool {parent.index}"
            elif parent is not None:
                current = f"currently {parent.display_text}"
            else:
                current = "parent no longer open"
                has_recorded_file = self._dependency_ref_has_recorded_file_in_inputs(
                    script_inputs,
                    ref,
                )
                if has_recorded_file:
                    current += "; recorded source file found"
            name = " ".join(ref.name.split())
            label = " ".join(ref.label.split())
            current = " ".join(current.split())
            if name and label and current:
                if label == name:
                    parts.append(f"{name} ({current})")
                else:
                    parts.append(f"{name}: {label} ({current})")
        if not parts:
            return None
        return "\n".join(parts)

    def _show_dependency_reload_dialog(self, target: int | str) -> None:
        node = self._manager._node_for_target(target)
        status = self._manager.dependency_status_for_uid(node.uid)
        if status is None:
            return

        details = self._manager.dependency_input_summary_for_uid(node.uid)
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setWindowTitle("Reload Data")
        if details:
            msg_box.setDetailedText(details)

        if not self._node_can_reload_script_inputs(node):
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setText("This result cannot be reloaded from its recorded inputs.")
            msg_box.setInformativeText(
                "The recorded provenance is not complete enough to replay."
            )
            msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Close)
            msg_box.exec()
            return

        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        if status == "changed":
            msg_box.setText("Reload this result from the current inputs?")
        else:
            msg_box.setText("Reload this result from its recorded inputs?")
        msg_box.setInformativeText(
            "The current ImageTool data will be replaced only if reload succeeds."
        )
        reload_button = msg_box.addButton(
            "Reload Data", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(typing.cast("QtWidgets.QPushButton", reload_button))
        msg_box.exec()
        if msg_box.clickedButton() is reload_button:
            self._reload_script_derived_target(target)
        elif msg_box.clickedButton() is cancel_button:
            return

    def _script_input_has_recorded_file(
        self,
        script_input: ScriptInput,
    ) -> bool:
        spec = script_input.parsed_provenance_spec()
        if spec is None:
            return False
        source_status = file_load_source_status(
            spec,
            extension_status_resolver=self._manager._extensions.capability_status,
        )
        if source_status != "no-file-load-source":
            return source_status != "missing-file"
        for nested_input in spec.script_inputs:
            if self._script_input_has_recorded_file(nested_input):
                return True
        return False

    def _file_load_source_unavailable_reason(
        self,
        spec: ToolProvenanceSpec,
        label: str,
    ) -> str | None:
        source_status = file_load_source_status(
            spec,
            extension_status_resolver=self._manager._extensions.capability_status,
        )
        load_source = spec.file_load_source
        if source_status == "no-file-load-source" or load_source is None:
            return f"{label} has no recorded source file."
        file_path = pathlib.Path(load_source.path)
        if source_status == "missing-file":
            return f"The source file for {label} is not available:\n{file_path}"
        replay_call = load_source.replay_call
        if source_status == "no-replay-call" or replay_call is None:
            return f"{label} does not have recorded loader information."
        if source_status == "missing-loader":
            return (
                f"The saved loader {replay_call.target!r} for {label} is unavailable."
            )
        if source_status == "extension-disabled":
            return (
                f"The registered script {replay_call.target!r} for {label} is "
                "disabled. Enable it in Manage Extensions, then try again."
            )
        if source_status == "extension-approval-required":
            return (
                f"The required script {replay_call.target!r} for {label} is not "
                "approved. Review it in Workspace Requirements, then try again."
            )
        if source_status == "extension-missing-source":
            return (
                f"The required script {replay_call.target!r} for {label} is not "
                "registered. Restore it from Workspace Requirements, then try again."
            )
        if source_status == "extension-missing-capability":
            return (
                f"The registered script {replay_call.target!r} for {label} does not "
                f"provide loader {replay_call.capability_id!r}."
            )
        if source_status == "extension-hash-mismatch":
            return (
                f"The registered script {replay_call.target!r} for {label} does not "
                "match the recorded source hash. Restore matching contents, then "
                "try again."
            )
        if source_status == "extension-unsupported-api":
            return (
                f"The loader from registered script {replay_call.target!r} for "
                f"{label} uses an unsupported extension API version."
            )
        if source_status == "extension-validation-failed":
            return (
                f"The loader from registered script {replay_call.target!r} for "
                f"{label} could not be validated. Open Manage Extensions for details."
            )
        return None

    def _dependency_ref_has_recorded_file_in_inputs(
        self,
        script_inputs: Sequence[ScriptInput],
        ref: ScriptInputDependencyRef,
    ) -> bool:
        for script_input in script_inputs:
            if (
                script_input.name == ref.name
                and script_input.node_uid == ref.node_uid
                and script_input.node_snapshot_token == ref.node_snapshot_token
                and script_input.data_role == ref.data_role
                and self._script_input_has_recorded_file(script_input)
            ):
                return True
            fallback = script_input.parsed_provenance_spec()
            if self._dependency_ref_has_recorded_file_in_inputs(
                () if fallback is None else fallback.script_inputs,
                ref,
            ):
                return True
        return False

    @staticmethod
    def _dependency_script_inputs(
        node: _ImageToolWrapper | _ManagedWindowNode | None,
    ) -> tuple[ScriptInput, ...]:
        if node is None or node.tool_script_inputs:
            return () if node is None else node.tool_script_inputs
        spec = node.provenance_spec
        return () if spec is None else spec.script_inputs

    def _missing_dependencies_have_recorded_file(self, uid: str) -> bool:
        script_inputs = self._dependency_script_inputs(
            self._manager._tool_graph.nodes.get(uid)
        )
        return any(
            self._manager._tool_graph.nodes.get(ref.node_uid) is None
            and self._dependency_ref_has_recorded_file_in_inputs(script_inputs, ref)
            for ref in self._dependency_refs_for_uid(uid)
        )

    def _refresh_dependency_dependents(self, uid: str) -> None:
        refresh_figures = False
        tree_uids: list[str] = []
        for dependent_uid in self._manager._dependency_tracker.dependent_uids(uid):
            dependent = self._manager._tool_graph.nodes.get(dependent_uid)
            if dependent is not None and dependent.tool_script_inputs:
                state = self._tool_input_source_state(dependent)
                status = self._manager._dependency_tracker.status_for_uid(dependent_uid)
                tool = dependent.tool_window
                if state == "unavailable":
                    self._propagate_source_state_from_uid(
                        dependent.uid,
                        "unavailable",
                    )
                elif state == "stale" or (
                    status == "changed"
                    and (tool is None or not tool.source_auto_update)
                ):
                    self._propagate_source_state_from_uid(
                        dependent.uid,
                        "stale",
                    )
                elif status == "changed":
                    if tool is not None and tool._source_refresh_deferred:
                        self._manager._dependency_tracker.queue_source_refresh(
                            dependent_uid,
                            dependent_uid,
                            automatic=True,
                        )
                    else:
                        self._refresh_tool_inputs(
                            dependent_uid,
                            allow_recorded=False,
                        )
            self._manager._schedule_details_refresh(dependent_uid)
            if self._manager._is_figure_uid(dependent_uid):
                refresh_figures = True
            else:
                tree_uids.append(dependent_uid)
        self._manager.tree_view.refresh_many(tree_uids)
        if refresh_figures:
            self._manager._figure_collection.sync()

    def _tool_input_source_state(
        self,
        node: _ManagedWindowNode,
        *,
        state_overrides: Mapping[str, _ManagedWindowNode._source_state_type]
        | None = None,
    ) -> _ManagedWindowNode._source_state_type:
        """Return the aggregate availability state of all live ToolWindow inputs.

        Snapshot changes are separate. A fresh input with a changed snapshot can be
        refreshed automatically, while a stale input cannot.
        """
        overrides = {} if state_overrides is None else state_overrides
        source_states: list[_ManagedWindowNode._source_state_type] = []
        for ref in self._dependency_refs_for_uid(node.uid):
            source = self._manager._tool_graph.nodes.get(ref.node_uid)
            if source is None:
                return "unavailable"
            source_states.append(overrides.get(ref.node_uid, source.source_state))
        return _combine_source_states(source_states)

    def _script_input_name_for_node(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str:
        if isinstance(node, _ImageToolWrapper):
            return f"data_{node.index}"
        suffix = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in node.uid
        )
        if not suffix or suffix[0].isdigit():
            suffix = f"_{suffix}"
        return f"data_{suffix}"

    def _script_input_for_node(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        name: str | None = None,
        source_spec: ToolProvenanceSpec | None = None,
        fallback_provenance_spec: dict[str, typing.Any] | None = None,
        detached_input_uid: str | None = None,
        data_role: ScriptInputDataRole = "displayed",
    ) -> ScriptInput:
        input_provenance = to_replay_provenance_spec(
            node.provenance_for_role(data_role)
        )
        resolved_provenance = (
            None
            if input_provenance is None
            else compose_full_provenance(input_provenance, source_spec)
        )
        provenance_spec = (
            resolved_provenance.model_dump(mode="json")
            if resolved_provenance is not None
            else fallback_provenance_spec
        )
        source_payload = (
            None if source_spec is None else source_spec.model_dump(mode="json")
        )
        input_name = self._script_input_name_for_node(node) if name is None else name
        if isinstance(node, _ImageToolWrapper):
            label = f"ImageTool {node.index}"
        else:
            fallback_label = (
                "ImageTool child"
                if node.is_imagetool
                else node.type_badge_text or "Tool"
            )
            label = node.display_text or fallback_label
        if isinstance(node, _ImageToolWrapper) and node.name:
            label += f": {node.name}"
        if node.uid == detached_input_uid:
            return ScriptInput(
                name=input_name,
                label=label,
                data_role=data_role,
                source_spec=source_payload,
                provenance_spec=provenance_spec,
            )
        return ScriptInput(
            name=input_name,
            label=label,
            node_uid=node.uid,
            node_snapshot_token=node.snapshot_token_for_role(data_role),
            data_role=data_role,
            source_spec=source_payload,
            provenance_spec=provenance_spec,
        )

    def _multi_input_script_provenance(
        self,
        input_targets: Iterable[int | str],
        *,
        operation_label: str,
        operation_code: str,
        active_name: str = "derived",
        start_label: str = "Run ImageTool manager action",
        detached_input_uid: str | None = None,
        data_role: ScriptInputDataRole = "displayed",
    ) -> ToolProvenanceSpec:
        return script(
            ScriptCodeOperation(
                label=operation_label,
                code=operation_code,
            ),
            start_label=start_label,
            active_name=active_name,
            script_inputs=tuple(
                self._script_input_for_node(
                    self._manager._node_for_target(target),
                    detached_input_uid=detached_input_uid,
                    data_role=data_role,
                )
                for target in input_targets
            ),
        )

    def _show_multi_input_script_result(
        self,
        data: xr.DataArray,
        input_targets: Iterable[int | str],
        *,
        operation_label: str,
        operation_code: str,
        data_role: ScriptInputDataRole = "displayed",
    ) -> int | None:
        input_targets = tuple(input_targets)
        tool = erlab.interactive.itool(data, manager=False, execute=False)
        if not isinstance(tool, ImageTool):
            return None
        return self._manager.add_imagetool(
            tool,
            show=True,
            activate=True,
            provenance_spec=self._multi_input_script_provenance(
                input_targets,
                operation_label=operation_label,
                operation_code=operation_code,
                data_role=data_role,
            ),
        )

    def _live_script_input_node(
        self,
        script_input: ScriptInput,
        target_node_uid: str | None = None,
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        uid = script_input.node_uid
        node = self._manager._tool_graph.nodes.get(uid or "")
        if (
            uid == target_node_uid
            or node is None
            or node.source_state != "fresh"
            or self.dependency_status_for_uid(node.uid) in {"changed", "missing"}
        ):
            return None
        return node

    def _resolve_live_script_input_for_reload(
        self,
        script_input: ScriptInput,
        *,
        target_node_uid: str | None = None,
    ) -> tuple[xr.DataArray, ScriptInput] | None:
        node = self._live_script_input_node(
            script_input, target_node_uid=target_node_uid
        )
        if node is None:
            return None
        owner_node = self._manager._tool_graph.nodes.get(target_node_uid or "")
        data = node.data_for_role(script_input.data_role, owner_node=owner_node)
        source_spec = script_input.parsed_source_spec()
        if source_spec is not None:
            data = source_spec.apply(
                data,
                extension_executor=self._manager._extensions.execution.run_operation,
            )
        return data, self._script_input_for_node(
            node,
            name=script_input.name,
            source_spec=source_spec,
            data_role=script_input.data_role,
        )

    def _live_input_resolver(
        self,
        *,
        target_node_uid: str | None = None,
        live_uids: frozenset[str] | None = None,
    ) -> LiveInputResolver | None:
        return _memoized_live_input_resolver(
            lambda item: (
                self._resolve_live_script_input_for_reload(
                    item, target_node_uid=target_node_uid
                )
                if live_uids is None or item.node_uid in live_uids
                else None
            )
        )

    def _input_resolution_plan(
        self,
        script_inputs: Sequence[ScriptInput],
        *,
        target_node_uid: str | None = None,
        allow_recorded: bool = True,
        force_live_reload: bool = False,
        reload_node: _ImageToolWrapper | _ManagedWindowNode | None = None,
    ) -> _InputResolutionPlan:
        root_inputs = tuple(script_inputs)
        actions: list[_InputResolutionAction] = []
        live_uids: set[str] = set()

        def add_action(
            kind: typing.Literal["reload", "refresh", "apply"], uid: str
        ) -> None:
            if (action := (kind, uid)) not in actions:
                actions.append(action)

        def provenance_runnable(
            spec: ToolProvenanceSpec, target_uid: str | None
        ) -> bool:
            def resolve(item: ScriptInput):
                if (
                    self._live_script_input_node(item, target_node_uid=target_uid)
                    is not None
                    or item.node_uid in live_uids
                ):
                    return typing.cast("xr.DataArray", None), item
                return None

            capability = _replay_capability(spec, live_input_resolver=resolve)
            return capability.replayable or capability.requires_trust

        def plan_inputs(
            inputs: Sequence[ScriptInput],
            target_uid: str | None,
            force: bool,
            visiting: frozenset[str],
            depth: int,
        ) -> str | None:
            if depth > 20:
                return "Nested inputs exceeded the maximum reload depth."
            for script_input in inputs:
                reason = plan_input(script_input, target_uid, force, visiting, depth)
                if reason is not None:
                    return reason
            return None

        def plan_node(
            node: _ImageToolWrapper | _ManagedWindowNode,
            force: bool,
            visiting: frozenset[str],
            depth: int,
        ) -> str | None:
            if depth > 20:
                return "Nested inputs exceeded the maximum reload depth."
            if not node.is_imagetool and node.tool_window is None:
                return "This tool cannot be reloaded directly."
            if node.uid in visiting:
                return "The named inputs contain a dependency cycle."
            visiting = visiting | {node.uid}

            extension_reason = self._manager._extensions.unavailable_reason_for_node(
                node.uid
            )
            if extension_reason is not None:
                return extension_reason

            tool = node.tool_window
            if tool is not None:
                inputs = tool.script_inputs
                if not inputs:
                    return "This tool does not have recorded source inputs."
                if reason := plan_inputs(inputs, node.uid, force, visiting, depth + 1):
                    return reason
                add_action("apply", node.uid)
                return None

            if node.imagetool is None:
                if node.pending_workspace_memory_payload is None:
                    return "This ImageTool window is not open."
                spec = node.provenance_spec
                script_inputs = (
                    spec.script_inputs
                    if spec is not None and spec.kind == "script"
                    else ()
                )
                if script_inputs and (
                    reason := plan_inputs(
                        script_inputs,
                        node.uid,
                        False,
                        visiting,
                        depth + 1,
                    )
                ):
                    return reason
                reason = self._pending_imagetool_reload_unavailable_reason(
                    node,
                    resolved_live_uids=(
                        frozenset(live_uids) if script_inputs else None
                    ),
                )
                if reason is None:
                    add_action("reload", node.uid)
                return reason
            spec = node.provenance_spec
            if node.slicer_area._direct_reloadable():
                add_action("reload", node.uid)
                return None
            if spec is not None and spec.kind == "script" and spec.script_inputs:
                if reason := plan_inputs(
                    spec.script_inputs, node.uid, False, visiting, depth + 1
                ):
                    return reason
                if not provenance_runnable(spec, node.uid):
                    return "This result contains code that cannot be replayed."
                add_action("reload", node.uid)
                return None
            if node.slicer_area._provenance_reloadable():
                add_action("reload", node.uid)
                return None
            if spec is not None and (spec.kind == "file" or has_file_load_source(spec)):
                if reason := self._file_load_source_unavailable_reason(
                    spec, "This result"
                ):
                    return reason
                if spec.kind == "file":
                    add_action("reload", node.uid)
                    return None
            if spec is not None and spec.kind == "script" and not spec.script_inputs:
                return "This result has no recorded inputs."
            return node.slicer_area._local_reload_unavailable_reason()

        def plan_input(
            script_input: ScriptInput,
            target_uid: str | None,
            force: bool,
            visiting: frozenset[str],
            depth: int,
        ) -> str | None:
            source_uid = script_input.node_uid
            if source_uid is not None and source_uid == target_uid:
                return "The named inputs contain a dependency cycle."
            if (
                self._live_script_input_node(script_input, target_node_uid=target_uid)
                is not None
                and not force
            ):
                live_uids.add(typing.cast("str", script_input.node_uid))
                return None
            if not allow_recorded:
                return f"{script_input.label} is not available in this Manager."

            source = (
                None
                if source_uid is None
                else self._manager._tool_graph.nodes.get(source_uid)
            )
            if source is not None:
                boundary = source
                if not source.tool_script_inputs and not isinstance(
                    source, _ImageToolWrapper
                ):
                    boundary = None
                    if source.is_imagetool:
                        boundary = self._reload_boundary_for_child(source.uid) or source
                if boundary is not None:
                    action_count = len(actions)
                    previous_live_uids = set(live_uids)
                    reason = plan_node(boundary, force, visiting, depth + 1)
                    if reason is None:
                        if boundary.uid != source.uid:
                            add_action("refresh", source.uid)
                        live_uids.add(source.uid)
                        return None
                    if force:
                        return reason
                    del actions[action_count:]
                    live_uids.intersection_update(previous_live_uids)
                elif force:
                    return f"{script_input.label} is not connected to a reload source."

            spec = script_input.parsed_provenance_spec()
            if spec is None:
                return f"{script_input.label} has no recorded reload source."
            if spec.kind == "file" or has_file_load_source(spec):
                if reason := self._file_load_source_unavailable_reason(
                    spec, script_input.label
                ):
                    return reason
                if spec.kind == "file":
                    return None
            if spec.kind != "script":
                return f"{script_input.label} has no replayable recorded provenance."
            reason = plan_inputs(
                spec.script_inputs, target_uid, False, visiting, depth + 1
            )
            if reason is not None:
                return reason
            if not provenance_runnable(spec, target_uid):
                return f"{script_input.label} contains code that cannot be replayed."
            return None

        unavailable_reason = (
            plan_inputs(root_inputs, target_node_uid, force_live_reload, frozenset(), 0)
            if reload_node is None
            else plan_node(reload_node, force_live_reload, frozenset(), 0)
        )
        return _InputResolutionPlan(
            target_node_uid,
            root_inputs,
            tuple(actions),
            frozenset(live_uids),
            unavailable_reason,
        )

    def _execute_input_resolution_plan(
        self,
        plan: _InputResolutionPlan,
        *,
        allow_recorded: bool = True,
        reloaded_uids: set[str],
    ) -> _InputResolutionPlan | typing.Literal["deferred", "failed"]:
        if plan.unavailable_reason is not None:
            return "failed"
        deferred = False
        target_uid = plan.target_uid
        if target_uid is None:
            return "failed"
        tracker = self._manager._dependency_tracker
        for kind, uid in plan.actions:
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None:
                return "failed"
            if kind != "refresh" and uid in reloaded_uids:
                continue
            if (
                kind == "apply"
                and node.source_state == "fresh"
                and self.dependency_status_for_uid(uid) == "current"
                and any(
                    item.node_uid in plan.live_uids for item in node.tool_script_inputs
                )
            ):
                reloaded_uids.add(uid)
                continue
            if kind == "reload" and self._node_can_reload_script_inputs(node):
                updated = self._reload_script_derived_target(
                    uid,
                    continuation_uids=() if target_uid == uid else (target_uid,),
                    reloaded_uids=reloaded_uids,
                )
            elif kind == "reload":
                updated = node.reload_source_data()
            elif kind == "refresh":
                updated = self._refresh_source_chain_to_uid(uid)
            else:
                updated = self._refresh_tool_inputs(
                    uid,
                    allow_recorded=True,
                    continuation_uids=() if target_uid == uid else (target_uid,),
                    reloaded_uids=reloaded_uids,
                )
            if updated:
                if kind != "refresh":
                    reloaded_uids.add(uid)
                continue

            tool = node.tool_window
            blocker = (
                node if tool is not None and tool._source_refresh_deferred else None
            )
            if blocker is None and not isinstance(node, _ImageToolWrapper):
                boundary = self._reload_boundary_for_child(uid)
                boundary_tool = None if boundary is None else boundary.tool_window
                if boundary_tool is not None and boundary_tool._source_refresh_deferred:
                    blocker = boundary
            if blocker is None:
                if kind == "apply" and tracker.source_refresh_queued(uid, target_uid):
                    deferred = True
                    continue
                return "failed"
            if blocker.uid != uid:
                tracker.queue_source_refresh(blocker.uid, uid)
            tracker.queue_source_refresh(uid, target_uid)
            deferred = True
        if deferred:
            return "deferred"
        plan = self._input_resolution_plan(
            plan.script_inputs,
            target_node_uid=target_uid,
            allow_recorded=allow_recorded,
        )
        return (
            plan if plan.unavailable_reason is None and not plan.actions else "failed"
        )

    def _refresh_tool_inputs(
        self,
        target_uid: str,
        *,
        allow_recorded: bool,
        continuation_uids: Sequence[str] = (),
        force_live_reload: bool = False,
        reloaded_uids: set[str] | None = None,
    ) -> bool:
        reloaded_uids = set() if reloaded_uids is None else reloaded_uids
        if target_uid in reloaded_uids:
            return True
        if target_uid in self._tool_input_refresh_uids:
            return False
        self._tool_input_refresh_uids.add(target_uid)
        try:
            node = self._manager._child_node(target_uid)
            tool = node.tool_window
            inputs = () if tool is None else tool.script_inputs
            if tool is None or not inputs:
                self._manager._dependency_tracker.pop_source_refreshes(target_uid)
                return False

            def fail_refresh() -> bool:
                self._manager._dependency_tracker.pop_source_refreshes(target_uid)
                return False

            def defer_refresh() -> bool:
                for continuation_uid in continuation_uids:
                    self._manager._dependency_tracker.queue_source_refresh(
                        target_uid,
                        continuation_uid,
                    )
                return False

            if tool._source_refresh_deferred:
                return defer_refresh()

            plan = self._input_resolution_plan(
                inputs,
                target_node_uid=target_uid,
                allow_recorded=allow_recorded,
                force_live_reload=force_live_reload,
            )
            plan = self._execute_input_resolution_plan(
                plan,
                allow_recorded=allow_recorded,
                reloaded_uids=reloaded_uids,
            )
            if plan == "deferred":
                return defer_refresh()
            if plan == "failed":
                return fail_refresh()
            resolve_live = self._live_input_resolver(
                target_node_uid=plan.target_uid,
                live_uids=plan.live_uids,
            )

            try:
                resolved, refreshed_inputs = rebuild_script_inputs(
                    plan.script_inputs,
                    live_input_resolver=resolve_live,
                    recorded_input_authorizer=lambda _input, spec: (
                        self._ensure_script_provenance_trusted(
                            spec,
                            reason="reload this tool input",
                            live_input_resolver=resolve_live,
                        )
                    ),
                    allow_recorded=allow_recorded,
                )
                with node._suspend_descendant_propagation():
                    updated = tool._apply_inputs(resolved, refreshed_inputs)
            except _TrustedScriptReplayCancelled:
                return fail_refresh()
            except Exception as exc:
                if not isinstance(exc, ReplayGraphError):
                    logger.exception(
                        "Failed to update %s from named Manager inputs", tool.tool_name
                    )
                self._propagate_source_state_from_uid(node.uid, "unavailable")
                return fail_refresh()

            if updated:
                self._propagate_source_change_from_uid(target_uid)
                self._resume_pending_source_refreshes(target_uid)
            elif tool.source_state != "fresh":
                self._propagate_source_state_from_uid(target_uid, tool.source_state)
                if tool._source_refresh_deferred:
                    defer_refresh()
                else:
                    fail_refresh()
            node._notify_change(_ManagedNodeChange.ROW)
            if updated:
                reloaded_uids.add(target_uid)
            return updated
        finally:
            self._tool_input_refresh_uids.discard(target_uid)

    def _script_input_unavailable_reason(
        self,
        script_input: ScriptInput,
        *,
        target_node_uid: str | None = None,
    ) -> str | None:
        return self._input_resolution_plan(
            (script_input,), target_node_uid=target_node_uid
        ).unavailable_reason

    def _rebuild_script_provenance(
        self,
        spec: ToolProvenanceSpec,
        *,
        target_node_uid: str | None = None,
    ) -> _ScriptRebuildResult:
        resolve_live = self._live_input_resolver(
            target_node_uid=target_node_uid,
        )

        try:
            trusted_user_code = self._ensure_script_provenance_trusted(
                spec,
                reason="reload this result",
                live_input_resolver=resolve_live,
            )
            data, rebuilt_spec = rebuild_script_provenance(
                spec,
                live_input_resolver=resolve_live,
                trusted_user_code=trusted_user_code,
                extension_executor=self._manager._extensions.execution.run_operation,
                extension_loader_executor=self._manager._extensions.replay_loader,
            )
        except _TrustedScriptReplayCancelled:
            raise
        except ReplayGraphError as exc:
            raise _ScriptRebuildError(
                "Could not reload data.",
                details=str(exc),
            ) from exc
        except Exception as exc:
            raise _ScriptRebuildError(
                "Could not reload data.",
                details=traceback.format_exc(),
            ) from exc
        return _ScriptRebuildResult(
            data=data,
            provenance_spec=rebuilt_spec,
        )

    def _node_can_reload_script_inputs(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> bool:
        spec = node.provenance_spec
        return (
            node.is_imagetool
            and self._manager._extensions.unavailable_reason_for_node(node.uid) is None
            and node.imagetool is not None
            and spec is not None
            and spec.kind == "script"
            and bool(spec.script_inputs)
            and self._node_reload_unavailable_reason(node) is None
        )

    def _node_reload_unavailable_reason(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        if not node.is_imagetool:
            return "This tool cannot be reloaded directly."
        return self._input_resolution_plan(
            (),
            target_node_uid=node.uid,
            reload_node=node,
        ).unavailable_reason

    def _pending_imagetool_reload_unavailable_reason(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        resolved_live_uids: frozenset[str] | None = None,
    ) -> str | None:
        spec = node.provenance_spec
        extension_reason = self._manager._extensions.unavailable_reason_for_node(
            node.uid
        )
        if extension_reason is not None:
            return extension_reason
        if spec is not None:
            if can_reload_without_trust(
                spec,
                extension_status_resolver=self._manager._extensions.capability_status,
            ):
                return None
            if spec.kind == "file" or has_file_load_source(spec):
                reason = self._file_load_source_unavailable_reason(spec, "This result")
                if reason is not None:
                    return reason
        if spec is not None and spec.kind == "script":
            if spec.script_inputs and resolved_live_uids is None:
                plan = self._input_resolution_plan(
                    spec.script_inputs,
                    target_node_uid=node.uid,
                )
                if plan.unavailable_reason is not None:
                    return plan.unavailable_reason
                resolved_live_uids = plan.live_uids

            def resolve_live(item: ScriptInput):
                if item.node_uid in (resolved_live_uids or ()) or (
                    resolved_live_uids is None
                    and self._live_script_input_node(
                        item,
                        target_node_uid=node.uid,
                    )
                    is not None
                ):
                    return typing.cast("xr.DataArray", None), item
                return None

            capability = _replay_capability(
                spec,
                live_input_resolver=resolve_live,
            )
            if capability.requires_trust:
                return "Open the ImageTool to approve recorded script code."
            if capability.replayable:
                return None
            return "The recorded script steps cannot be reloaded."
        details = node._load_source_details()
        if details is None:
            return "This data does not have a reloadable source."
        if not details.path.exists():
            return f"The source file is not available:\n{details.path}"
        if details.load_code is None:
            return "The source file does not have loader information."
        return None

    def _script_reload_from_slicer_area(
        self,
        slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea,
        *,
        execute: bool,
    ) -> bool:
        target = self._manager.target_from_slicer_area(slicer_area)
        if target is None:
            return False
        if not self._node_can_reload_script_inputs(
            self._manager._node_for_target(target)
        ):
            return False
        return not execute or self._reload_script_derived_target(target)

    def _rebase_loaded_workspace_dependency_refs(
        self, loaded_targets_by_uid: Mapping[str, int | str]
    ) -> None:
        uid_map: dict[str, str] = {}
        for saved_uid, target in loaded_targets_by_uid.items():
            try:
                actual_uid = self._manager._node_for_target(target).uid
            except KeyError:
                continue
            if actual_uid != saved_uid:
                uid_map[saved_uid] = actual_uid
        if not uid_map:
            return

        self._rebase_node_dependency_refs(loaded_targets_by_uid.values(), uid_map)

    @staticmethod
    def _rebase_tool_data_reference_node_uids(
        references: Mapping[str, Mapping[str, typing.Any]],
        uid_map: Mapping[str, str],
    ) -> dict[str, dict[str, typing.Any]]:
        rebased = {name: dict(reference) for name, reference in references.items()}
        for reference in rebased.values():
            if reference.get("kind") != "manager_node":
                continue
            node_uid = reference.get("node_uid")
            if isinstance(node_uid, str) and node_uid in uid_map:
                reference["node_uid"] = uid_map[node_uid]
        return rebased

    def _rebase_node_dependency_refs(
        self,
        targets: Iterable[int | str],
        uid_map: Mapping[str, str],
    ) -> None:
        """Rebase all framework and tool-owned references for selected nodes."""
        if not uid_map:
            return

        for target in targets:
            try:
                node = self._manager._node_for_target(target)
            except KeyError:
                continue
            if node.tool_window is not None:
                tool = node.tool_window
                rebased_inputs = rebase_script_inputs_node_uids(
                    tool.script_inputs,
                    uid_map,
                )
                if rebased_inputs != tool.script_inputs:
                    tool.set_script_inputs(
                        rebased_inputs,
                        primary_input=tool.primary_input,
                        auto_update=tool.source_auto_update,
                        state=tool.source_state,
                    )
                tool.rebase_source_node_uids(uid_map)
                references = node._workspace_tool_data_references
                rebased_references = self._rebase_tool_data_reference_node_uids(
                    references, uid_map
                )
                if rebased_references != references:
                    node._set_workspace_tool_data_references(rebased_references)
            elif node.pending_workspace_payload_kind == "tool":
                attrs = node.pending_workspace_payload_attrs
                tool_inputs = node.tool_script_inputs
                rebased_inputs = rebase_script_inputs_node_uids(tool_inputs, uid_map)
                updated_attrs: dict[str, typing.Any] | None = None
                if attrs is not None and rebased_inputs != tool_inputs:
                    updated_attrs = dict(attrs)
                    updated_attrs.update(
                        erlab.interactive.utils.ToolWindow._saved_script_input_attrs(
                            rebased_inputs, node.tool_primary_input
                        )
                    )
                if attrs is not None:
                    raw_references = attrs.get(
                        erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR
                    )
                    if isinstance(raw_references, (str, bytes, bytearray)):
                        try:
                            references = json.loads(raw_references)
                        except (TypeError, ValueError, UnicodeDecodeError):
                            pass
                        else:
                            if (
                                isinstance(references, dict)
                                and (
                                    rebased_references := (
                                        self._rebase_tool_data_reference_node_uids(
                                            references, uid_map
                                        )
                                    )
                                )
                                != references
                            ):
                                if updated_attrs is None:
                                    updated_attrs = dict(attrs)
                                updated_attrs[
                                    erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR
                                ] = json.dumps(rebased_references)
                if updated_attrs is not None:
                    node.update_pending_workspace_payload_attrs(updated_attrs)
            if node.provenance_spec is None:
                continue
            rebased = rebase_script_input_node_uids(
                node.provenance_spec,
                uid_map,
            )
            if rebased != node.provenance_spec:
                node.set_displayed_provenance(rebased, advance_snapshot=False)

    def _selected_reload_candidates(
        self,
    ) -> tuple[list[int | str], dict[int | str, list[str]], str | None] | None:
        selected_roots = self._manager.tree_view.selected_imagetool_indices
        selected_children = self._manager.tree_view.selected_childtool_uids
        if not selected_roots and not selected_children:
            return None

        reload_targets: list[int | str] = []
        seen_targets: set[int | str] = set()
        child_targets: dict[int | str, list[str]] = {}

        def _add_reload_target(target: int | str) -> None:
            if target in seen_targets:
                return
            seen_targets.add(target)
            reload_targets.append(target)

        for index in selected_roots:
            unavailable_reason = self._reload_unavailable_reason_for_target(index)
            if unavailable_reason is not None:
                return [], {}, unavailable_reason
            _add_reload_target(index)

        for uid in selected_children:
            reload_target = self._reload_target_for_child(uid)
            if reload_target is None:
                return [], {}, self._reload_unavailable_reason_for_child(uid)
            _add_reload_target(reload_target)
            if reload_target != uid:
                child_targets.setdefault(reload_target, []).append(uid)

        return reload_targets, child_targets, None

    def _reload_boundary_for_child(
        self, uid: str
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        try:
            current: _ImageToolWrapper | _ManagedWindowNode = self._manager._child_node(
                uid
            )
        except KeyError:
            return None
        if not current.has_source_binding:
            return None

        reload_boundary: _ImageToolWrapper | _ManagedWindowNode | None = None
        while True:
            if current.tool_script_inputs:
                return current
            if (
                current.is_imagetool
                and self._node_reload_unavailable_reason(current) is None
            ):
                reload_boundary = current
            if isinstance(current, _ImageToolWrapper):
                break
            try:
                current = self._manager._parent_node(current)
            except KeyError:
                break
        return reload_boundary

    def _reload_target_for_child(self, uid: str) -> int | str | None:
        boundary = self._reload_boundary_for_child(uid)
        if boundary is None:
            return None
        if boundary.tool_script_inputs and not boundary.can_reload_source_data():
            return None
        if isinstance(boundary, _ImageToolWrapper):
            return boundary.index
        return boundary.uid

    def _reload_unavailable_reason_for_child(self, uid: str) -> str:
        try:
            current = self._manager._child_node(uid)
        except KeyError:
            return "The selected tool is no longer available. Select an open item."
        if not current.has_source_binding:
            return (
                "This tool does not have a recorded source input. Reopen or "
                "recreate it from reloadable ImageTool data to enable reload."
            )
        boundary = self._reload_boundary_for_child(uid)
        if boundary is not None and boundary.tool_script_inputs:
            return boundary.reload_unavailable_reason() or (
                "One or more named inputs cannot be reloaded. Restore or reopen "
                "the missing inputs, then try again."
            )
        return (
            "This tool cannot reload because its source chain has no reloadable "
            "ImageTool. Restore or reopen the source data, then try again."
        )

    def _reload_unavailable_reason_for_target(self, target: int | str) -> str | None:
        try:
            node = self._manager._node_for_target(target)
        except KeyError:
            return "The selected item is no longer available. Select an open item."
        if isinstance(target, str):
            if self._reload_target_for_child(target) is not None:
                return None
            return self._reload_unavailable_reason_for_child(target)
        return self._node_reload_unavailable_reason(node)

    def _reload_source_chain_for_child(self, uid: str) -> bool:
        reload_target = self._reload_target_for_child(uid)
        if reload_target is None:
            return False
        return self._reload_target_with_continuations(
            reload_target,
            () if reload_target == uid else (uid,),
        )

    def _reload_target_with_continuations(
        self,
        target: int | str,
        target_uids: Sequence[str],
        *,
        reloaded_uids: set[str] | None = None,
    ) -> bool:
        node = self._manager._node_for_target(target)
        reloaded_uids = set() if reloaded_uids is None else reloaded_uids
        if node.uid in reloaded_uids:
            reloaded = True
        elif node.tool_script_inputs:
            reloaded = self._refresh_tool_inputs(
                node.uid,
                allow_recorded=True,
                continuation_uids=target_uids,
                force_live_reload=True,
                reloaded_uids=reloaded_uids,
            )
        elif self._node_can_reload_script_inputs(node):
            reloaded = self._reload_script_derived_target(
                target,
                continuation_uids=target_uids,
                reloaded_uids=reloaded_uids,
            )
        else:
            reloaded = node.reload_source_data()
        if not reloaded:
            tool = node.tool_window
            if (
                tool is not None
                and tool.source_state == "stale"
                and tool._source_refresh_deferred
            ):
                for target_uid in target_uids:
                    self._manager._dependency_tracker.queue_source_refresh(
                        node.uid,
                        target_uid,
                    )
            return False
        reloaded_uids.add(node.uid)
        refreshed = True
        for target_uid in target_uids:
            refreshed &= self._refresh_source_chain_to_uid(target_uid)
        return refreshed

    def show_selected_source_updates(self) -> None:
        """Show automatic update controls for the selected child window."""
        uid = self._manager._selected_source_update_child_uid()
        if uid is None:
            return
        self._manager._child_node(uid).show_source_update_dialog(parent=self._manager)

    def _refresh_source_chain_to_uid(self, uid: str) -> bool:
        node = self._manager._tool_graph.nodes.get(uid)
        if (
            node is not None
            and not node.has_source_binding
            and self._node_can_reload_script_inputs(node)
        ):
            if (
                node.source_state == "fresh"
                and self.dependency_status_for_uid(uid) == "current"
            ):
                return True
            return self._reload_script_derived_target(uid)
        try:
            node = self._manager._child_node(uid)
        except KeyError:
            return False

        refresh_chain = [node]
        while True:
            try:
                parent = self._manager._parent_node(node)
            except KeyError:
                return False
            if isinstance(parent, _ImageToolWrapper):
                break
            refresh_chain.append(parent)
            node = parent

        refresh_chain.reverse()
        for index, node in enumerate(refresh_chain):
            current_uid = node.uid

            if not node.has_source_binding or node.source_state == "fresh":
                continue
            updated = node._update_from_parent_source()
            if updated and node.source_state == "fresh":
                self._resume_pending_source_refreshes(current_uid)
                continue
            tool = node.tool_window
            if (
                tool is not None
                and tool.source_state == "stale"
                and tool._source_refresh_deferred
            ):
                for blocker, target in itertools.pairwise(refresh_chain[index:]):
                    self._manager._dependency_tracker.queue_source_refresh(
                        blocker.uid,
                        target.uid,
                    )
                return False
            if node.source_state != "fresh":
                self._propagate_source_state_from_uid(current_uid, node.source_state)
            self._manager._dependency_tracker.pop_source_refreshes(current_uid)
            return False

        try:
            return self._manager._child_node(uid).source_state == "fresh"
        except KeyError:
            return False

    def _resume_pending_source_refreshes(
        self,
        uid: str,
        *,
        require_self_refresh: bool = False,
    ) -> bool:
        target_intents = self._manager._dependency_tracker.pop_source_refresh_intents(
            uid
        )
        target_uids = set(target_intents)
        if require_self_refresh and uid not in target_uids:
            self._manager._dependency_tracker.discard_source_refreshes(target_uids)
            return False
        self_refresh = uid in target_uids
        if self_refresh:
            target_uids.remove(uid)
            source = self._manager._tool_graph.nodes.get(uid)
            if (
                target_intents[uid]
                and all(target_intents.values())
                and source is not None
                and source.tool_window is not None
                and not source.tool_window.source_auto_update
            ):
                source._set_source_state("stale")
                return False
            refreshed = bool(
                source is not None
                and source.tool_script_inputs
                and self._refresh_tool_inputs(uid, allow_recorded=True)
            )
            if not refreshed:
                if (
                    source is not None
                    and source.tool_window is not None
                    and source.tool_window._source_refresh_deferred
                ):
                    for target_uid in target_uids:
                        self._manager._dependency_tracker.queue_source_refresh(
                            uid,
                            target_uid,
                        )
                    return True
                self._manager._dependency_tracker.discard_source_refreshes(target_uids)
                return False
        for target_uid in list(target_uids):
            target = self._manager._tool_graph.nodes.get(target_uid)
            if target is None:
                continue
            if target.tool_script_inputs:
                if target.source_state == "fresh" and self.dependency_status_for_uid(
                    target_uid
                ) not in {"changed", "missing"}:
                    continue
                self._refresh_tool_inputs(
                    target_uid,
                    allow_recorded=True,
                )
            else:
                self._refresh_source_chain_to_uid(target_uid)
        return self_refresh

    def _parent_source_data_for_uid(self, uid: str) -> xr.DataArray:
        node = self._manager._child_node(uid)
        parent = self._manager._parent_node(node)
        return parent.current_source_data()

    def _propagate_source_state_from_uid(
        self,
        uid: str,
        state: _ManagedWindowNode._source_state_type,
    ) -> None:
        if state == "fresh":
            raise ValueError("fresh source state requires a completed data update")
        source = self._manager._tool_graph.nodes.get(uid)
        if source is not None:
            source._set_source_state(state)
        pending = [uid]
        seen = {uid}
        propagated_states = {uid: state}
        while pending:
            source_uid = pending.pop()
            source = self._manager._tool_graph.nodes.get(source_uid)
            if source is None:
                continue

            target_uids: list[str] = []
            for child_uid in source._childtool_indices:
                child = self._manager._tool_graph.nodes.get(child_uid)
                if (
                    child is not None
                    and child.is_imagetool
                    and child.has_source_binding
                ):
                    target_uids.append(child_uid)
            for dependent_uid in self._manager._dependency_tracker.dependent_uids(
                source_uid
            ):
                dependent = self._manager._tool_graph.nodes.get(dependent_uid)
                if dependent is not None and dependent.tool_script_inputs:
                    target_uids.append(dependent_uid)

            for target_uid in target_uids:
                if target_uid in seen:
                    continue
                seen.add(target_uid)
                target = self._manager._tool_graph.nodes.get(target_uid)
                if target is None:
                    continue
                target_state = propagated_states[source_uid]
                if target.tool_script_inputs:
                    target_state = self._tool_input_source_state(
                        target,
                        state_overrides=propagated_states,
                    )
                    if target_state == "fresh":
                        continue
                target._set_source_state(target_state)
                propagated_states[target_uid] = target_state
                pending.append(target_uid)

    def _propagate_source_change_from_uid(
        self, uid: str, parent_data: xr.DataArray | None = None
    ) -> None:
        if parent_data is None:
            try:
                parent_data = self._manager._node_for_target(uid).current_source_data()
            except Exception:
                self._propagate_source_state_from_uid(uid, "unavailable")
                return
        for child_uid in list(self._manager._node_for_target(uid)._childtool_indices):
            try:
                child = self._manager._child_node(child_uid)
            except KeyError:
                continue
            if not child.is_imagetool:
                # The dependency graph owns named-input refreshes. The tree edge
                # only selects where the tool appears.
                continue
            previous_state = child.source_state
            updated = child.handle_parent_source_replaced(parent_data)
            if updated or child.source_state != previous_state:
                self._manager.tree_view.refresh(child_uid)
            if updated:
                self._propagate_source_change_from_uid(child_uid)
            elif child.source_state != "fresh":
                self._propagate_source_state_from_uid(child_uid, child.source_state)

    def show_selected(self) -> None:
        """Show selected windows."""
        index_list = self._manager._selected_imagetool_targets()
        for index in index_list:
            self._manager._node_for_target(index).show()

        uid_list = self._manager._selected_tool_uids()

        for uid in uid_list:
            self._manager.show_childtool(uid)

    def hide_selected(self) -> None:
        """Hide selected windows."""
        for index in self._manager._selected_imagetool_targets():
            self._manager._node_for_target(index).hide()
        for uid in self._manager._selected_tool_uids():
            self._manager.get_childtool(uid).hide()

    def hide_all(self) -> None:
        """Hide all windows."""
        for node in self._manager._tool_graph.nodes.values():
            node.hide()

    def reload_selected(self) -> None:
        selected_reload_candidates = self._selected_reload_candidates()
        if selected_reload_candidates is None:
            return

        reload_targets, child_targets, unavailable_reason = selected_reload_candidates
        if unavailable_reason is not None:
            erlab.interactive.utils._show_reload_unavailable_dialog(
                self._manager,
                unavailable_reason,
            )
            return

        reloaded_uids: set[str] = set()
        for target in reload_targets:
            self._reload_target_with_continuations(
                target,
                child_targets.get(target, ()),
                reloaded_uids=reloaded_uids,
            )

    @staticmethod
    def _reload_incompatibility_details(
        current: xr.DataArray, rebuilt: xr.DataArray
    ) -> str:
        current, rebuilt = (
            erlab.interactive.imagetool.slicer._cursor_compatibility_pair(
                current, rebuilt
            )
        )
        lines = [
            f"Current dims: {tuple(current.dims)} shape {tuple(current.shape)}",
            f"Reloaded dims: {tuple(rebuilt.dims)} shape {tuple(rebuilt.shape)}",
        ]
        current_dims = set(current.dims)
        rebuilt_dims = set(rebuilt.dims)
        missing_dims = tuple(dim for dim in current.dims if dim not in rebuilt_dims)
        new_dims = tuple(dim for dim in rebuilt.dims if dim not in current_dims)
        if missing_dims:
            lines.append(f"Missing reloaded dimensions: {missing_dims}")
        if new_dims:
            lines.append(f"New reloaded dimensions: {new_dims}")
        for dim in current.dims:
            if dim not in rebuilt_dims:
                continue
            if current.sizes[dim] != rebuilt.sizes[dim]:
                lines.append(
                    f"{dim}: size changed from {current.sizes[dim]} to "
                    f"{rebuilt.sizes[dim]}"
                )
            old_coord = current.coords.get(dim)
            new_coord = rebuilt.coords.get(dim)
            if old_coord is None or new_coord is None:
                continue
            old_values = old_coord.values
            new_values = new_coord.values
            missing_count = int(
                np.count_nonzero(~np.isin(old_values, new_values, assume_unique=True))
            )
            if missing_count:
                lines.append(
                    f"{dim}: {missing_count} current coordinate value"
                    f"{'' if missing_count == 1 else 's'} not found in reloaded data"
                )
        return "\n".join(lines)

    def _prompt_incompatible_reload_commit(self, details: str) -> str:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Reload Data")
        msg_box.setText("The reloaded data has different coordinates.")
        msg_box.setInformativeText(
            "The current ImageTool view can only be preserved when the reloaded data "
            "keeps the current cursor coordinates."
        )
        msg_box.setDetailedText(details)
        replace_button = msg_box.addButton(
            "Replace and Reset View", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        new_button = msg_box.addButton(
            "Open as New", QtWidgets.QMessageBox.ButtonRole.ActionRole
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(typing.cast("QtWidgets.QPushButton", new_button))
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked is replace_button:
            return "replace"
        if clicked is new_button:
            return "new"
        if clicked is cancel_button:
            return "cancel"
        return "cancel"

    def _replace_script_reload_target(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        result: _ScriptRebuildResult,
    ) -> None:
        node.replace_with_detached_data(
            result.data,
            result.provenance_spec,
            propagate_descendants=True,
            preserve_filter=True,
            replay_source_data=node.replay_source_data,
        )
        self._manager.tree_view.refresh(node.uid)
        self._resume_pending_source_refreshes(node.uid)

    def _reload_script_derived_target(
        self,
        target: int | str,
        *,
        continuation_uids: Sequence[str] = (),
        reloaded_uids: set[str] | None = None,
    ) -> bool:
        node = self._manager._node_for_target(target)
        spec = node.provenance_spec
        if spec is None:
            return False
        reloaded_uids = set() if reloaded_uids is None else reloaded_uids
        plan = self._input_resolution_plan(
            spec.script_inputs,
            target_node_uid=node.uid,
        )
        plan = self._execute_input_resolution_plan(
            plan,
            reloaded_uids=reloaded_uids,
        )
        if isinstance(plan, str):
            if plan == "deferred":
                for continuation_uid in continuation_uids:
                    self._manager._dependency_tracker.queue_source_refresh(
                        node.uid,
                        continuation_uid,
                    )
            return False
        with (
            self._manager._extensions.execution.capture_replay_sources()
        ) as publication:
            try:
                result = self._rebuild_script_provenance(
                    spec,
                    target_node_uid=node.uid,
                )
            except _TrustedScriptReplayCancelled:
                return False
            except _ScriptRebuildError as exc:
                erlab.interactive.utils.MessageDialog.critical(
                    self._manager,
                    "Error",
                    str(exc),
                    detailed_text=exc.details,
                )
                return False

            current = node.current_source_data()
            if erlab.interactive.imagetool.slicer.check_cursors_compatible(
                current, result.data
            ):
                publication.require_current_for_publication()
                self._replace_script_reload_target(node, result)
                self._manager._status_bar.showMessage("Reloaded data from inputs", 5000)
                publication.publish()
                return True

            details = self._reload_incompatibility_details(current, result.data)
            match self._prompt_incompatible_reload_commit(details):
                case "replace":
                    publication.require_current_for_publication()
                    self._replace_script_reload_target(node, result)
                    self._manager._status_bar.showMessage(
                        "Reloaded data from inputs", 5000
                    )
                    publication.publish()
                    return True
                case "new":
                    tool = erlab.interactive.itool(
                        result.data, manager=False, execute=False
                    )
                    if not isinstance(tool, ImageTool):
                        erlab.interactive.utils.MessageDialog.critical(
                            self._manager,
                            "Error",
                            "An error occurred while opening reloaded data.",
                            detailed_text="",
                        )
                        return False
                    publication.require_current_for_publication()
                    self._manager.add_imagetool(
                        tool,
                        show=True,
                        activate=True,
                        provenance_spec=result.provenance_spec,
                    )
                    self._manager._status_bar.showMessage(
                        "Opened reloaded data as a new tool", 5000
                    )
                    publication.publish()
                    return True
                case _:
                    return False

    def remove_selected(self) -> None:
        """Discard selected ImageTool windows."""
        indices = list(self._manager._selected_imagetool_targets())
        child_uids = list(self._manager._selected_tool_uids())

        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Remove selected windows?")

        count: int = len(indices)
        num_selected_children: int = len(child_uids)
        num_implicit_children: int = 0
        for i in indices:
            for uid in self._manager._node_for_target(i)._childtool_indices:
                if uid not in child_uids:  # pragma: no branch
                    num_implicit_children += 1

        text = f"{count} selected ImageTool window{'' if count == 1 else 's'}"
        if num_implicit_children > 0:
            text += (
                f", along with {num_implicit_children} associated child tool"
                f"{'' if num_implicit_children == 1 else 's'}"
            )
        if num_selected_children > 0:
            text += (
                f" and {num_selected_children} selected child tool"
                f"{'' if num_selected_children == 1 else 's'}"
            )
        text += " will be removed."

        msg_box.setInformativeText(text)
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

        if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
            self._manager._remove_imagetools(indices, child_uids=child_uids)
