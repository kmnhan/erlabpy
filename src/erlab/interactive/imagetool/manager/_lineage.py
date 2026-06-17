from __future__ import annotations

import logging
import pathlib
import typing

import numpy as np
from qtpy import QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive.imagetool import _replay_graph, provenance
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool.manager._widgets import (
    _DEPENDENCY_STATUS_BADGES,
    _DEPENDENCY_STATUS_LABELS,
    _DEPENDENCY_STATUS_TOOLTIPS,
    _ScriptRebuildError,
    _ScriptRebuildResult,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import xarray as xr

    from erlab.interactive.imagetool.manager._dependency import _DependencyStatus
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager


logger = logging.getLogger(__name__)


class _LineageController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager

    def _dependency_refs_for_uid(
        self, uid: str
    ) -> tuple[provenance.ScriptInputDependencyRef, ...]:
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
        if node is not None and self._manager._node_can_reload_script_inputs(node):
            tooltip += " Click for Reload Data options."
        if (
            status == "missing"
            and self._manager._missing_dependencies_have_recorded_file(uid)
        ):
            tooltip += " Recorded source files found for at least one missing input."
        return tooltip

    def dependency_input_summary_for_uid(self, uid: str) -> str | None:
        refs = self._manager._dependency_refs_for_uid(uid)
        if not refs:
            return None

        node = self._manager._tool_graph.nodes.get(uid)
        spec = None if node is None else node.provenance_spec
        parts: list[str] = []
        seen: set[tuple[str, str, str | None]] = set()
        for ref in refs:
            key = (ref.name, ref.node_uid, ref.node_snapshot_token)
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
                if self._manager._dependency_ref_has_recorded_file(spec, ref):
                    current += "; recorded source file found"
            name = " ".join(ref.name.split())
            label = " ".join(ref.label.split())
            current = " ".join(current.split())
            if name and label and current:
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

        if not self._manager._node_can_reload_script_inputs(node):
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
            self._manager._reload_script_derived_target(target)
        elif msg_box.clickedButton() is cancel_button:
            return

    @classmethod
    def _script_input_has_recorded_file(
        cls,
        script_input: provenance.ScriptInput,
    ) -> bool:
        spec = script_input.parsed_provenance_spec()
        if spec is None:
            return False
        if spec.kind == "file" and spec.file_load_source is not None:
            return pathlib.Path(spec.file_load_source.path).exists()
        for nested_input in spec.script_inputs:
            if cls._script_input_has_recorded_file(nested_input):
                return True
        return False

    @classmethod
    def _dependency_ref_has_recorded_file(
        cls,
        spec: provenance.ToolProvenanceSpec | None,
        ref: provenance.ScriptInputDependencyRef,
    ) -> bool:
        if spec is None:
            return False
        for script_input in spec.script_inputs:
            if (
                script_input.name == ref.name
                and script_input.node_uid == ref.node_uid
                and script_input.node_snapshot_token == ref.node_snapshot_token
                and cls._script_input_has_recorded_file(script_input)
            ):
                return True
            if cls._dependency_ref_has_recorded_file(
                script_input.parsed_provenance_spec(), ref
            ):
                return True
        return False

    def _missing_dependencies_have_recorded_file(self, uid: str) -> bool:
        node = self._manager._tool_graph.nodes.get(uid)
        spec = None if node is None else node.provenance_spec
        return any(
            self._manager._tool_graph.nodes.get(ref.node_uid) is None
            and self._manager._dependency_ref_has_recorded_file(spec, ref)
            for ref in self._manager._dependency_refs_for_uid(uid)
        )

    def _dependency_dependent_uids(self, uid: str) -> list[str]:
        return self._manager._dependency_tracker.dependent_uids(uid)

    def _refresh_dependency_dependents(self, uid: str) -> None:
        for dependent_uid in self._manager._dependency_dependent_uids(uid):
            if self._manager._is_figure_uid(dependent_uid):
                self._manager._sync_figures_ui()
                self._manager._update_info(uid=dependent_uid)
            else:
                self._manager.tree_view.refresh(dependent_uid)
            if self._manager._metadata_node_uid == dependent_uid:
                self._manager._set_metadata_node(
                    self._manager._tool_graph.nodes[dependent_uid]
                )

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
        detached_input_uid: str | None = None,
        use_displayed_provenance: bool = True,
    ) -> provenance.ScriptInput:
        input_provenance = (
            node.displayed_provenance_spec
            if use_displayed_provenance
            else node.provenance_spec
        )
        provenance_spec = (
            input_provenance.model_dump(mode="json")
            if input_provenance is not None
            else None
        )
        if isinstance(node, _ImageToolWrapper):
            label = f"ImageTool {node.index}"
        else:
            label = node.display_text
        if isinstance(node, _ImageToolWrapper) and node.name:
            label += f": {node.name}"
        if node.uid == detached_input_uid:
            return provenance.ScriptInput(
                name=self._manager._script_input_name_for_node(node),
                label=label,
                provenance_spec=provenance_spec,
            )
        return provenance.ScriptInput(
            name=self._manager._script_input_name_for_node(node),
            label=label,
            node_uid=node.uid,
            node_snapshot_token=node.snapshot_token,
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
        use_displayed_provenance: bool = True,
    ) -> provenance.ToolProvenanceSpec:
        return provenance.script(
            provenance.ScriptCodeOperation(
                label=operation_label,
                code=operation_code,
            ),
            start_label=start_label,
            active_name=active_name,
            script_inputs=tuple(
                self._manager._script_input_for_node(
                    self._manager._node_for_target(target),
                    detached_input_uid=detached_input_uid,
                    use_displayed_provenance=use_displayed_provenance,
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
        use_displayed_provenance: bool = True,
    ) -> int | None:
        input_targets = tuple(input_targets)
        tool = erlab.interactive.itool(data, manager=False, execute=False)
        if not isinstance(tool, ImageTool):
            return None
        return self._manager.add_imagetool(
            tool,
            show=True,
            activate=True,
            provenance_spec=self._manager._multi_input_script_provenance(
                input_targets,
                operation_label=operation_label,
                operation_code=operation_code,
                use_displayed_provenance=use_displayed_provenance,
            ),
        )

    def _script_provenance_inputs_current(
        self, spec: provenance.ToolProvenanceSpec
    ) -> bool:
        for ref in provenance.script_input_dependency_refs(spec):
            parent = self._manager._tool_graph.nodes.get(ref.node_uid)
            if parent is None:
                return False
            if (
                ref.node_snapshot_token is not None
                and parent.snapshot_token != ref.node_snapshot_token
            ):
                return False
        return True

    def _resolve_live_script_input_for_reload(
        self,
        script_input: provenance.ScriptInput,
        *,
        target_node_uid: str | None = None,
    ) -> tuple[xr.DataArray, provenance.ScriptInput] | None:
        spec = script_input.parsed_provenance_spec()
        if script_input.node_uid is None:
            return None
        if target_node_uid is not None and script_input.node_uid == target_node_uid:
            return None
        node = self._manager._tool_graph.nodes.get(script_input.node_uid)
        if node is None:
            return None
        if (
            spec is not None
            and spec.kind == "script"
            and not self._manager._script_provenance_inputs_current(spec)
        ):
            return None
        return (
            node.current_source_data(),
            self._manager._script_input_for_node(node).model_copy(
                update={"name": script_input.name}
            ),
        )

    def _script_input_can_reload(
        self,
        script_input: provenance.ScriptInput,
        *,
        target_node_uid: str | None = None,
    ) -> bool:
        spec = script_input.parsed_provenance_spec()
        is_target_input = (
            target_node_uid is not None and script_input.node_uid == target_node_uid
        )
        if script_input.node_uid is not None and not is_target_input:
            node = self._manager._tool_graph.nodes.get(script_input.node_uid)
            if node is not None and (
                spec is None
                or spec.kind != "script"
                or self._manager._script_provenance_inputs_current(spec)
            ):
                return True
        if spec is None:
            return False
        if spec.kind == "file":
            return (
                spec.file_load_source is not None
                and pathlib.Path(spec.file_load_source.path).exists()
            )
        if spec.kind != "script":
            return False
        return provenance.script_provenance_replayable(spec) and all(
            self._manager._script_input_can_reload(
                nested_input,
                target_node_uid=target_node_uid,
            )
            for nested_input in spec.script_inputs
        )

    def _rebuild_script_provenance(
        self,
        spec: provenance.ToolProvenanceSpec,
        *,
        target_node_uid: str | None = None,
    ) -> _ScriptRebuildResult:
        def _resolve_live_input(
            script_input: provenance.ScriptInput,
        ) -> (
            tuple[
                xr.DataArray,
                provenance.ScriptInput,
            ]
            | None
        ):
            return self._manager._resolve_live_script_input_for_reload(
                script_input,
                target_node_uid=target_node_uid,
            )

        try:
            data, rebuilt_spec = _replay_graph.rebuild_script_provenance(
                spec,
                live_input_resolver=_resolve_live_input,
            )
        except _replay_graph.ReplayGraphError as exc:
            raise _ScriptRebuildError(
                "Could not reload data.",
                details=str(exc),
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
            and node.imagetool is not None
            and spec is not None
            and spec.kind == "script"
            and bool(spec.script_inputs)
            and provenance.script_provenance_replayable(spec)
            and all(
                self._manager._script_input_can_reload(
                    script_input,
                    target_node_uid=node.uid,
                )
                for script_input in spec.script_inputs
            )
        )

    def _script_reload_from_slicer_area(
        self,
        slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea,
        *,
        execute: bool,
    ) -> bool:
        """Check or reload script provenance for a managed slicer area."""
        target = self._manager.target_from_slicer_area(slicer_area)
        if target is None:
            return False
        if not self._manager._node_can_reload_script_inputs(
            self._manager._node_for_target(target)
        ):
            return False
        return not execute or self._manager._reload_script_derived_target(target)

    def _workspace_loaded_uid_map(
        self, loaded_targets_by_uid: Mapping[str, int | str]
    ) -> dict[str, str]:
        uid_map: dict[str, str] = {}
        for saved_uid, target in loaded_targets_by_uid.items():
            try:
                actual_uid = self._manager._node_for_target(target).uid
            except KeyError:
                continue
            if actual_uid != saved_uid:
                uid_map[saved_uid] = actual_uid
        return uid_map

    def _rebase_loaded_workspace_dependency_refs(
        self, loaded_targets_by_uid: Mapping[str, int | str]
    ) -> None:
        uid_map = self._manager._workspace_loaded_uid_map(loaded_targets_by_uid)
        if not uid_map:
            return

        for target in loaded_targets_by_uid.values():
            try:
                node = self._manager._node_for_target(target)
            except KeyError:
                continue
            if node.provenance_spec is None:
                continue
            if node.tool_window is not None:
                node.tool_window.rebase_source_node_uids(uid_map)
            rebased = provenance.rebase_script_input_node_uids(
                node.provenance_spec,
                uid_map,
            )
            if rebased != node.provenance_spec:
                node.set_displayed_provenance(rebased, advance_snapshot=False)

    def _selected_reload_targets(
        self,
    ) -> tuple[list[int | str], dict[int | str, list[str]]] | None:
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
            if not self._manager._node_for_target(index).reloadable:
                return None
            _add_reload_target(index)

        for uid in selected_children:
            reload_target = self._manager._reload_target_for_child(uid)
            if reload_target is None:
                return None
            _add_reload_target(reload_target)
            child_targets.setdefault(reload_target, []).append(uid)

        return reload_targets, child_targets

    def _reload_target_for_child(self, uid: str) -> int | str | None:
        try:
            current: _ImageToolWrapper | _ManagedWindowNode = self._manager._child_node(
                uid
            )
        except KeyError:
            return None
        if not current.has_source_binding:
            return None

        reload_target: int | str | None = None
        while True:
            try:
                parent = self._manager._parent_node(current)
            except KeyError:
                break
            if parent.is_imagetool and parent.reloadable:
                reload_target = (
                    parent.index
                    if isinstance(parent, _ImageToolWrapper)
                    else parent.uid
                )
            if isinstance(parent, _ImageToolWrapper):
                break
            current = parent
        return reload_target

    def _reload_source_chain_for_child(self, uid: str) -> bool:
        """Reload the nearest reloadable ancestor, then refresh a child node."""
        reload_target = self._manager._reload_target_for_child(uid)
        if reload_target is None:
            return False
        node = self._manager._node_for_target(reload_target)
        if node.imagetool is None or not node.slicer_area._reload():
            return False
        return self._manager._refresh_source_chain_to_uid(uid)

    def show_selected_source_updates(self) -> None:
        """Show automatic update controls for the selected child window."""
        uid = self._manager._selected_source_update_child_uid()
        if uid is None:
            return
        self._manager._child_node(uid).show_source_update_dialog(parent=self._manager)

    def _child_targets_of(self, target: int | str) -> list[str]:
        return list(self._manager._node_for_target(target)._childtool_indices)

    def _refresh_source_chain_to_uid(self, uid: str) -> bool:
        """Refresh stale ancestors before refreshing a managed child node."""
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

        for node in reversed(refresh_chain):
            current_uid = node.uid

            if not node.has_source_binding or node.source_state == "fresh":
                continue
            updated = node._update_from_parent_source()
            if updated and node.source_state == "fresh":
                continue
            tool = node.tool_window
            if (
                tool is not None
                and tool.source_state == "stale"
                and getattr(tool, "_source_refresh_deferred", False)
            ):
                self._manager._dependency_tracker.queue_source_refresh(current_uid, uid)
                return False
            if node.source_state != "fresh":
                self._manager._mark_descendants_source_state(
                    current_uid, node.source_state
                )
            return False

        try:
            return self._manager._child_node(uid).source_state == "fresh"
        except KeyError:
            return False

    def _resume_pending_source_refreshes(self, uid: str) -> None:
        target_uids = self._manager._dependency_tracker.pop_source_refreshes(uid)
        for target_uid in list(target_uids):
            if target_uid not in self._manager._tool_graph.nodes:
                continue
            self._manager._refresh_source_chain_to_uid(target_uid)

    def _parent_source_data_for_uid(self, uid: str) -> xr.DataArray:
        node = self._manager._child_node(uid)
        parent = self._manager._parent_node(node)
        return parent.current_source_data()

    def _mark_descendants_source_state(
        self,
        uid: str,
        state: _ManagedWindowNode._source_state_type,
    ) -> None:
        for child_uid in self._manager._iter_descendant_uids(uid):
            node = self._manager._child_node(child_uid)
            if node.tool_window is not None and node.tool_window.has_source_binding:
                node.tool_window._set_source_state(state)
            elif node.has_source_binding:
                node._set_source_state(state)

    def _mark_descendants_source_unavailable(self, uid: str) -> None:
        self._manager._mark_descendants_source_state(uid, "unavailable")

    def _propagate_source_change_from_uid(
        self, uid: str, parent_data: xr.DataArray | None = None
    ) -> None:
        if parent_data is None:
            try:
                parent_data = self._manager._node_for_target(uid).current_source_data()
            except Exception:
                self._manager._mark_descendants_source_unavailable(uid)
                return
        for child_uid in list(self._manager._node_for_target(uid)._childtool_indices):
            try:
                child = self._manager._child_node(child_uid)
            except KeyError:
                continue
            updated = child.handle_parent_source_replaced(parent_data)
            self._manager.tree_view.refresh(child_uid)
            if updated:
                self._manager._propagate_source_change_from_uid(child_uid)
            elif child.source_state != "fresh":
                self._manager._mark_descendants_source_state(
                    child_uid, child.source_state
                )

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
        """Reload data in selected ImageTool windows."""
        selected_reload_targets = self._manager._selected_reload_targets()
        if selected_reload_targets is None:
            return

        reload_targets, child_targets = selected_reload_targets
        reloaded_targets: set[int | str] = set()
        for target in reload_targets:
            node = self._manager._node_for_target(target)
            if node.imagetool is not None and node.slicer_area._reload():
                reloaded_targets.add(target)

        for target, child_uids in child_targets.items():
            if target not in reloaded_targets:
                continue
            for uid in child_uids:
                self._manager._refresh_source_chain_to_uid(uid)

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
        )
        self._manager.tree_view.refresh(node.uid)
        if self._manager._metadata_node_uid == node.uid:
            self._manager._set_metadata_node(node)

    def _reload_script_derived_target(self, target: int | str) -> bool:
        """Reload a script-derived ImageTool from its recorded inputs."""
        node = self._manager._node_for_target(target)
        spec = node.provenance_spec
        if spec is None:
            return False
        try:
            result = self._manager._rebuild_script_provenance(
                spec,
                target_node_uid=node.uid,
            )
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
            self._manager._replace_script_reload_target(node, result)
            self._manager._status_bar.showMessage("Reloaded data from inputs", 5000)
            return True

        details = self._manager._reload_incompatibility_details(current, result.data)
        match self._manager._prompt_incompatible_reload_commit(details):
            case "replace":
                self._manager._replace_script_reload_target(node, result)
                self._manager._status_bar.showMessage("Reloaded data from inputs", 5000)
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
                self._manager.add_imagetool(
                    tool,
                    show=True,
                    activate=True,
                    provenance_spec=result.provenance_spec,
                )
                self._manager._status_bar.showMessage(
                    "Opened reloaded data as a new tool", 5000
                )
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
            for uid in self._manager._child_targets_of(i):
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
