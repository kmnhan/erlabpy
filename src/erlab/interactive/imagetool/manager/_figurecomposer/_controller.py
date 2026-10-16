"""Coordinate manager-owned Figure Composer windows and source workflows."""

from __future__ import annotations

import copy
import functools
import typing
import uuid

from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._figurecomposer import _seeding
from erlab.interactive._figurecomposer._model._axes import _all_axes_for_shape
from erlab.interactive._figurecomposer._providers import (
    FigureSourceTarget,
    ImageToolSourceProvider,
)
from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer import (
        FigureAxesSelectionState,
        FigureComposerTool,
        FigureOperationState,
        FigureSourceState,
        FigureSubplotsState,
    )
    from erlab.interactive._figurecomposer._model._document import FigureSourceAddResult
    from erlab.interactive._figurecomposer._providers import FigureSourceProvider
    from erlab.interactive._fit1d import Fit1DTool
    from erlab.interactive.imagetool.manager._figurecomposer._collection import (
        _FigureCollectionController,
    )
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import _ImageToolWrapper
    from erlab.interactive.imagetool.viewer import ImageSlicerArea


class _FigureComposerWorkflowController(QtCore.QObject):
    """Coordinate manager-owned Figure Composer source and recipe workflows."""

    def __init__(
        self,
        host: ImageToolManager,
        collection: _FigureCollectionController,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._collection = collection
        self._parent_widget = parent

    def _configure_materialized_figure_tool(
        self, node: _ManagedWindowNode, tool: erlab.interactive.utils.ToolWindow
    ) -> None:
        from erlab.interactive._figurecomposer import FigureComposerTool

        tool._refresh_reload_data_action()
        if isinstance(tool, FigureComposerTool):
            tool.set_options_getter(lambda: self._host.effective_interactive_options)
            tool.set_settings_opener(self._host.open_settings)
            self._install_figure_source_refresh_callbacks(node.uid, tool)

    def _figure_operations_from_image_targets(
        self, targets: tuple[int | str, ...], source_names: tuple[str, ...]
    ) -> tuple[FigureOperationState, ...] | None:
        from erlab.interactive._figurecomposer import (
            FigureOperationKind,
            FigureOperationState,
        )
        from erlab.interactive.imagetool._figurecomposer_adapter import (
            build_figure_composer_operation,
        )

        if not source_names:
            return None
        source_operations: list[FigureOperationState] = []
        for index, target in enumerate(targets):
            if index >= len(source_names):
                break
            tool = self._host.get_imagetool(target)
            if not tool.slicer_area.axes:
                return None
            plot = tool.slicer_area.axes[0]
            if not plot.is_image:
                return None
            source_operation = build_figure_composer_operation(
                plot, source_name=source_names[index]
            )
            if source_operation.kind not in {
                FigureOperationKind.PLOT_ARRAY,
                FigureOperationKind.PLOT_SLICES,
            }:
                return None
            source_operations.append(source_operation)
        if not source_operations:
            return None
        if all(
            operation.kind == FigureOperationKind.PLOT_ARRAY
            for operation in source_operations
        ):
            if len(source_operations) > 1 and all(
                len(operation.map_selections) == 1 for operation in source_operations
            ):
                first_operation = source_operations[0]
                plot_array_updates = {
                    "transpose": first_operation.transpose,
                    "invert_x": first_operation.invert_x,
                    "invert_y": first_operation.invert_y,
                    "xlim": first_operation.xlim,
                    "ylim": first_operation.ylim,
                    "crop": first_operation.crop,
                    "axis": "image"
                    if first_operation.aspect == "equal"
                    else first_operation.axis,
                    "colorbar": first_operation.colorbar,
                    "hide_colorbar_ticks": first_operation.hide_colorbar_ticks,
                    "annotate": first_operation.annotate,
                    "cmap": first_operation.cmap,
                    "gamma": first_operation.gamma,
                    "norm_name": first_operation.norm_name,
                    "norm_gamma": first_operation.norm_gamma,
                    "norm_clip": first_operation.norm_clip,
                    "norm_kwargs": dict(first_operation.norm_kwargs),
                    "vmin": first_operation.vmin,
                    "vmax": first_operation.vmax,
                    "vcenter": first_operation.vcenter,
                    "halfrange": first_operation.halfrange,
                    "colorbar_kw": dict(first_operation.colorbar_kw),
                    "extra_kwargs": dict(first_operation.extra_kwargs),
                }
                operation = FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=source_names,
                    map_selections=tuple(
                        selection
                        for source_operation in source_operations
                        for selection in source_operation.map_selections
                    ),
                ).model_copy(update=plot_array_updates)
                if len(source_names) > 1:
                    operation = operation.model_copy(update={"order": "F"})
                return (
                    _seeding.plot_slices_operation_with_source_styles(
                        operation,
                        tuple(source_operations),
                        selections_per_source=1,
                    ),
                )
            return tuple(source_operations)
        if any(
            operation.kind != FigureOperationKind.PLOT_SLICES
            for operation in source_operations
        ):
            return None

        if any(operation.map_selections for operation in source_operations):
            from erlab.interactive._figurecomposer._exceptions import (
                FigureComposerPlotSlicesSelectionError,
            )

            raise FigureComposerPlotSlicesSelectionError
        operation = source_operations[0]
        source_updates: dict[str, typing.Any] = {"sources": source_names}
        if len(source_names) > 1:
            source_updates["order"] = "F"
        expanded_operation = operation.model_copy(update=source_updates)
        return (
            _seeding.plot_slices_operation_with_source_styles(
                expanded_operation,
                tuple(source_operations),
                selections_per_source=1,
            ),
        )

    def _figure_bz_overlay_operation_from_target(
        self,
        target: int | str,
        data: xr.DataArray,
        *,
        axes: FigureAxesSelectionState,
    ) -> FigureOperationState | None:
        from erlab.interactive.kspace import KspaceTool

        node = self._host._node_for_target(target)
        if node.output_id == KspaceTool.Output.CONVERTED.value:
            parent = self._host._parent_node(node)
            if isinstance(parent.tool_window, KspaceTool):
                return _seeding.bz_overlay_operation_from_ktool(
                    parent.tool_window,
                    data,
                    axes=axes,
                )
            return None
        return _seeding.bz_overlay_operation_from_momentum_data(data, axes=axes)

    def _figure_bz_overlay_operation_from_targets(
        self,
        targets: tuple[int | str, ...],
        source_data: Mapping[str, xr.DataArray],
        *,
        axes: FigureAxesSelectionState,
    ) -> FigureOperationState | None:
        if len(targets) != 1 or len(source_data) != 1:
            return None
        target = targets[0]
        data = next(iter(source_data.values()))
        return self._figure_bz_overlay_operation_from_target(target, data, axes=axes)

    def _show_figure_plot_slices_selection_error(self, error: Exception) -> None:
        from erlab.interactive._figurecomposer._exceptions import (
            PLOT_SLICES_SELECTION_ERROR_TITLE,
        )

        QtWidgets.QMessageBox.warning(
            self._parent_widget, PLOT_SLICES_SELECTION_ERROR_TITLE, str(error)
        )

    def _figure_source_from_node(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        data: xr.DataArray,
        reserved: set[str],
    ) -> FigureSourceState:
        from erlab.interactive._figurecomposer import FigureSourceState
        from erlab.interactive._figurecomposer._model._sources import (
            _source_alias_candidate,
            _source_unique_name,
        )

        source = FigureSourceState.from_script_input(
            self._host._lineage_controller._script_input_for_node(node)
        )
        alias = _source_unique_name(
            _source_alias_candidate(data) or source.name, reserved
        )
        return source.model_copy(update={"name": alias})

    def _figure_sources_from_targets(
        self,
        targets: Iterable[int | str | FigureSourceTarget],
        *,
        reserved_sources: Iterable[str] = (),
    ) -> tuple[
        tuple[int | str, ...],
        tuple[FigureSourceState, ...],
        dict[str, xr.DataArray],
    ]:
        resolved_targets = self._figure_imagetool_targets(targets)
        source_data: dict[str, xr.DataArray] = {}
        sources = []
        reserved = set(reserved_sources)
        for target in resolved_targets:
            node = self._source_node_for_target(target)
            data = node.data_for_role("displayed")
            source = self._figure_source_from_node(node, data, reserved)
            source_data[source.name] = data
            sources.append(source)
        return resolved_targets, tuple(sources), source_data

    def _figure_source_name_map_for_targets(
        self,
        targets: Iterable[int | str | FigureSourceTarget],
        sources: Iterable[FigureSourceState],
    ) -> dict[str, str]:
        source_name_map: dict[str, str] = {}
        for target, source in zip(targets, sources, strict=True):
            old_name = self._host._script_input_name_for_node(
                self._source_node_for_target(target)
            )
            if old_name != source.name:
                source_name_map[old_name] = source.name
        return source_name_map

    def _figure_source_uid_for_target(
        self, target: int | str | FigureSourceTarget
    ) -> str | None:
        try:
            node = self._source_node_for_target(target)
        except KeyError:
            return None
        return node.uid if self._providers_for_target(target) else None

    def _figure_providers_for_node(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[FigureSourceProvider, ...]:
        if node.is_imagetool:
            return (
                ImageToolSourceProvider(
                    node, self._host._lineage_controller._script_input_for_node
                ),
            )
        tool = node.tool_window
        if tool is None or not erlab.interactive.utils.qt_is_valid(tool):
            return ()
        return tool.figure_source_providers()

    def _figure_node_is_source(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        source_kinds: frozenset[str] | None = None,
    ) -> bool:
        return any(
            (source_kinds is None or provider.payload_kind in source_kinds)
            and provider.is_ready()
            for provider in self._figure_providers_for_node(node)
        )

    def _source_node_for_target(
        self, target: int | str | FigureSourceTarget
    ) -> _ImageToolWrapper | _ManagedWindowNode:
        return self._host._node_for_target(
            target.node_uid if isinstance(target, FigureSourceTarget) else target
        )

    def _providers_for_target(
        self, target: int | str | FigureSourceTarget
    ) -> tuple[FigureSourceProvider, ...]:
        return tuple(
            provider
            for provider in self._figure_providers_for_node(
                self._source_node_for_target(target)
            )
            if (
                not isinstance(target, FigureSourceTarget)
                or provider.provider_id == target.provider_id
            )
            and provider.is_ready()
        )

    def _capture_figure_sources(
        self,
        targets: Iterable[int | str | FigureSourceTarget],
        *,
        reserved_sources: Iterable[str] = (),
    ) -> tuple[tuple[FigureSourceState, ...], dict[str, object], dict[str, str]]:
        from erlab.interactive._figurecomposer import FigureSourceState
        from erlab.interactive._figurecomposer._model._sources import (
            _source_unique_name,
        )
        from erlab.interactive._figurecomposer._sources import source_payload_kind

        sources = []
        payloads = {}
        name_map: dict[str, str] = {}
        reserved = set(reserved_sources)
        seen: set[tuple[str, str]] = set()
        for target in self._figure_source_targets(targets):
            node = self._source_node_for_target(target)
            for provider in self._providers_for_target(target):
                key = (node.uid, provider.provider_id)
                if key in seen:
                    continue
                seen.add(key)
                capture = provider.capture()
                kind = source_payload_kind(capture.payload)
                if kind != provider.payload_kind:
                    raise ValueError(
                        "Figure source provider returned an incompatible payload"
                    )
                alias = _source_unique_name(capture.name, reserved)
                source = FigureSourceState(
                    name=alias,
                    kind=kind,
                    label=capture.label or provider.label,
                    node_uid=node.uid,
                    node_snapshot_token=capture.revision_token,
                    provider_id=provider.provider_id,
                    provider_selection=copy.deepcopy(capture.selection),
                    provenance_spec=copy.deepcopy(capture.provenance_spec),
                    data_role=capture.data_role,
                    use_node_data=capture.use_node_data,
                )
                sources.append(source)
                payloads[alias] = capture.payload
                name_map.setdefault(capture.name, alias)
                name_map.setdefault(self._host._script_input_name_for_node(node), alias)
        name_map.update({source.name: source.name for source in sources})
        return tuple(sources), payloads, name_map

    def _figure_source_targets(
        self, targets: Iterable[int | str | FigureSourceTarget]
    ) -> tuple[int | str | FigureSourceTarget, ...]:
        resolved: list[int | str | FigureSourceTarget] = []
        seen_uids: set[tuple[str, str | None]] = set()
        for target in targets:
            uid = self._figure_source_uid_for_target(target)
            if uid is None:
                continue
            key = (
                uid,
                target.provider_id if isinstance(target, FigureSourceTarget) else None,
            )
            if key in seen_uids:
                continue
            resolved.append(target)
            seen_uids.add(key)
        return tuple(resolved)

    def _figure_imagetool_targets(
        self, targets: Iterable[int | str | FigureSourceTarget]
    ) -> tuple[int | str, ...]:
        return tuple(
            target.node_uid if isinstance(target, FigureSourceTarget) else target
            for target in self._figure_source_targets(targets)
            if self._source_node_for_target(target).is_imagetool
        )

    def _selected_figure_source_uids(self) -> tuple[str, ...]:
        uids: list[str] = []
        for target in self._host._selected_figure_source_targets():
            uid = self._figure_source_uid_for_target(target)
            if uid is not None and uid not in uids:
                uids.append(uid)
        return tuple(uids)

    def _figure_source_uids_from_mime(
        self, mime: QtCore.QMimeData | None
    ) -> tuple[str, ...]:
        uids = self._host.tree_view.figure_source_uids_from_mime(mime)
        return tuple(
            uid
            for uid in uids
            if (node := self._host._tool_graph.nodes.get(uid)) is not None
            and self._figure_node_is_source(node)
        )

    def _selected_figure_uid_for_figure_dialog(self) -> str | None:
        selected_uids = self._host._selected_figure_uids()
        if len(selected_uids) == 1:
            return selected_uids[0]
        return None

    def _add_sources_to_figure(
        self,
        figure_uid: str,
        sources: tuple[FigureSourceState, ...],
        source_data: Mapping[str, xr.DataArray],
        *,
        show: bool = True,
    ) -> FigureSourceAddResult | None:
        """Add or update figure sources without appending recipe operations."""
        from erlab.interactive._figurecomposer import FigureComposerTool

        if not self._host._is_figure_uid(figure_uid):
            return None
        node = self._host._child_node(figure_uid)
        tool = node.tool_window
        if not isinstance(tool, FigureComposerTool):
            return None
        result = tool.add_sources(sources, source_data)
        if not result:
            return result
        self._host._mark_workspace_dirty(uid=figure_uid, state=True)
        self._collection.select_uid(figure_uid)
        if show:
            node.show()
        return result

    def _add_imagetool_sources_to_figure(
        self,
        figure_uid: str,
        targets: Iterable[int | str | FigureSourceTarget],
        *,
        show: bool = False,
    ) -> bool:
        requested_targets = tuple(targets)
        sources, payloads, _name_map = self._capture_figure_sources(requested_targets)
        if not sources:
            return False
        from erlab.interactive._figurecomposer import FigureComposerTool

        if not self._host._is_figure_uid(figure_uid):
            return False
        node = self._host._child_node(figure_uid)
        tool = node.tool_window
        if not isinstance(tool, FigureComposerTool):
            return False
        result = tool.add_source_payloads(sources, payloads)
        if not result:
            return False
        self._host._mark_workspace_dirty(uid=figure_uid, data=True, state=True)
        self._collection.select_uid(figure_uid)
        if show:
            node.show()
        parts = []
        if result.added:
            parts.append(f"Added {len(result.added)} sources")
        if result.updated:
            parts.append(f"Updated {len(result.updated)} sources")
        if result.skipped:
            parts.append(f"Skipped {len(result.skipped)} source updates")
        skipped_targets = sum(
            self._figure_source_uid_for_target(target) is None
            for target in requested_targets
        )
        if skipped_targets:
            parts.append(f"Skipped {skipped_targets} unsupported selections")
        status = "; ".join(parts) + "."
        if result.skipped:
            status += " Could not update source data for: " + ", ".join(
                detail for _name, detail in result.skipped
            )
        tool._set_source_panel_status(status)
        return True

    def _replace_figure_source(
        self,
        figure_uid: str,
        alias: str,
        sources: tuple[FigureSourceState, ...],
        source_data: Mapping[str, xr.DataArray],
        *,
        show: bool = True,
    ) -> bool:
        """Replace one stored figure source with one selected ImageTool source."""
        from erlab.interactive._figurecomposer import FigureComposerTool

        if len(sources) != 1 or not self._host._is_figure_uid(figure_uid):
            return False
        replacement = sources[0]
        data = source_data.get(replacement.name)
        if data is None:
            return False
        node = self._host._child_node(figure_uid)
        tool = node.tool_window
        if not isinstance(tool, FigureComposerTool):
            return False
        if not tool.replace_source(alias, replacement, data):
            return False
        self._host._mark_workspace_dirty(uid=figure_uid, state=True)
        self._collection.select_uid(figure_uid)
        if show:
            node.show()
        return True

    def _replace_figure_source_payload(
        self, figure_uid: str, alias: str, source: FigureSourceState, payload: object
    ) -> bool:
        from erlab.interactive._figurecomposer import FigureComposerTool

        if not self._host._is_figure_uid(figure_uid):
            return False
        node = self._host._child_node(figure_uid)
        tool = node.tool_window
        if not isinstance(tool, FigureComposerTool):
            return False
        if not tool.replace_source_payload(alias, source, payload):
            return False
        self._host._mark_workspace_dirty(uid=figure_uid, data=True, state=True)
        self._collection.select_uid(figure_uid)
        node.show()
        return True

    def _install_figure_source_refresh_callbacks(
        self, figure_uid: str, tool: FigureComposerTool
    ) -> None:
        """Connect Figure Composer source refresh controls to live manager nodes."""
        tool._set_source_refresh_callbacks(
            can_refresh_source=lambda source_name: self._can_refresh_figure_source(
                figure_uid, source_name
            ),
            refresh_source=lambda source_name: self._refresh_figure_source(
                figure_uid, source_name
            ),
            source_label=lambda source_name: self._figure_source_refresh_label(
                figure_uid, source_name
            ),
        )
        tool._set_source_reveal_callbacks(
            can_reveal_source=lambda source_name: self._can_reveal_figure_source(
                figure_uid, source_name
            ),
            reveal_sources=lambda source_names: self._reveal_figure_sources(
                figure_uid, source_names
            ),
        )
        tool._set_source_add_callbacks(
            can_add_sources=functools.partial(
                self._can_request_add_sources_to_figure, figure_uid
            ),
            add_sources=functools.partial(
                self._request_add_sources_to_figure, figure_uid
            ),
            can_drop_sources=functools.partial(
                self._can_add_figure_sources_from_mime, figure_uid
            ),
            drop_sources=functools.partial(
                self._add_figure_sources_from_mime, figure_uid
            ),
        )

    def _can_request_add_sources_to_figure(
        self, figure_uid: str, source_kinds: tuple[str, ...] | None = None
    ) -> bool:
        return self._host._is_figure_uid(figure_uid) and any(
            self._figure_node_is_source(
                node, None if source_kinds is None else frozenset(source_kinds)
            )
            for node in self._host._tool_graph.nodes.values()
        )

    def _request_add_sources_to_figure(
        self, figure_uid: str, source_kinds: tuple[str, ...] | None = None
    ) -> bool:
        from erlab.interactive.imagetool.manager._figurecomposer import _dialogs

        if not self._host._is_figure_uid(figure_uid):
            return False
        dialog = _dialogs._FigureSourcePickerDialog(
            self._host,
            prechecked_uids=self._selected_figure_source_uids(),
            source_kinds=source_kinds,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        return self._add_imagetool_sources_to_figure(
            figure_uid, dialog.selected_targets(), show=False
        )

    def _can_add_figure_sources_from_mime(
        self, figure_uid: str, mime: QtCore.QMimeData | None
    ) -> bool:
        return self._host._is_figure_uid(figure_uid) and bool(
            self._figure_source_uids_from_mime(mime)
        )

    def _add_figure_sources_from_mime(
        self, figure_uid: str, mime: QtCore.QMimeData | None
    ) -> bool:
        targets = self._figure_source_uids_from_mime(mime)
        if not targets:
            return False
        return self._add_imagetool_sources_to_figure(figure_uid, targets, show=False)

    @staticmethod
    def _figure_source_state(
        tool: FigureComposerTool, source_name: str
    ) -> FigureSourceState | None:
        for source in tool.source_states():
            if source.name == source_name:
                return source
        return None

    def _figure_source_node(
        self, figure_uid: str, source_name: str
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        from erlab.interactive._figurecomposer import FigureComposerTool

        if not self._host._is_figure_uid(figure_uid):
            return None
        figure_node = self._host._child_node(figure_uid)
        tool = figure_node.tool_window
        if not isinstance(tool, FigureComposerTool):
            return None
        source = self._figure_source_state(tool, source_name)
        if source is None or source.node_uid is None:
            return None
        return self._host._tool_graph.nodes.get(source.node_uid or "")

    def _figure_source_live_node(
        self, figure_uid: str, source_name: str
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        node = self._figure_source_node(figure_uid, source_name)
        if node is None:
            return None
        window = node.window
        if window is None or not erlab.interactive.utils.qt_is_valid(window):
            # Invalid Qt wrappers are binding- and lifetime-dependent.
            return None  # pragma: no cover
        return node

    def _can_reveal_figure_source(self, figure_uid: str, source_name: str) -> bool:
        node = self._figure_source_node(figure_uid, source_name)
        return node is not None

    def _reveal_figure_sources(
        self, figure_uid: str, source_names: Sequence[str]
    ) -> bool:
        if not self._host._is_figure_uid(figure_uid):
            return False
        uids = tuple(
            dict.fromkeys(
                node.uid
                for source_name in source_names
                if (node := self._figure_source_node(figure_uid, source_name))
                is not None
                and self._can_reveal_figure_source(figure_uid, source_name)
            )
        )
        return self._host.reveal_nodes(uids)

    def _can_refresh_figure_source(self, figure_uid: str, source_name: str) -> bool:
        from erlab.interactive._figurecomposer import FigureComposerTool

        node = self._figure_source_live_node(figure_uid, source_name)
        if node is None:
            return False
        figure_tool = self._host._child_node(figure_uid).tool_window
        if not isinstance(figure_tool, FigureComposerTool):
            return False
        source = self._figure_source_state(figure_tool, source_name)
        if source is None:
            return False
        return any(
            provider.provider_id == source.provider_id
            and provider.payload_kind == source.kind
            and provider.is_ready(source.provider_selection)
            for provider in self._figure_providers_for_node(node)
        )

    def _figure_source_refresh_label(
        self, figure_uid: str, source_name: str
    ) -> str | None:
        node = self._figure_source_node(figure_uid, source_name)
        return None if node is None else node.display_text

    def _refresh_figure_source(self, figure_uid: str, source_name: str) -> bool:
        """Refresh a captured source from its original provider and selection."""
        from erlab.interactive._figurecomposer import FigureComposerTool

        source_node = self._figure_source_live_node(figure_uid, source_name)
        if source_node is None or not self._host._is_figure_uid(figure_uid):
            return False
        tool = self._host._child_node(figure_uid).tool_window
        if not isinstance(tool, FigureComposerTool):
            return False
        current = self._figure_source_state(tool, source_name)
        if current is None:
            return False
        provider = next(
            (
                provider
                for provider in self._figure_providers_for_node(source_node)
                if provider.provider_id == current.provider_id
                and provider.payload_kind == current.kind
                and provider.is_ready(current.provider_selection)
            ),
            None,
        )
        if provider is None:
            return False
        capture = provider.capture(current.provider_selection)
        replacement = current.model_copy(
            update={
                "provenance_spec": copy.deepcopy(capture.provenance_spec),
                "node_snapshot_token": capture.revision_token,
                "provider_selection": copy.deepcopy(capture.selection),
                "data_role": capture.data_role,
                "use_node_data": capture.use_node_data,
            }
        )
        if not tool.replace_source_payload(source_name, replacement, capture.payload):
            return False
        self._host._mark_workspace_dirty(uid=figure_uid, data=True, state=True)
        self._collection.update_gallery_icon(figure_uid)
        return True

    def _refresh_figure_source_controls(self) -> None:
        if self._host._workspace_ui_refresh_defer_depth > 0:
            self._host._deferred_workspace_source_controls_refresh = True
            return

        from erlab.interactive._figurecomposer import FigureComposerTool

        for figure_uid in self._host._figure_uids():
            node = self._host._tool_graph.nodes.get(figure_uid)
            if not isinstance(node, _ManagedWindowNode):
                continue
            tool = node.tool_window
            if isinstance(tool, FigureComposerTool):
                tool.refresh_source_controls()

    def create_figure_from_targets(
        self,
        targets: Iterable[int | str | FigureSourceTarget],
        *,
        operation: FigureOperationState | None = None,
        custom_code: str | None = None,
        title: str | None = None,
        show: bool = True,
    ) -> str | None:
        from erlab.interactive._figurecomposer import (
            FigureAxesSelectionState,
            FigureComposerTool,
            FigureOperationKind,
            FigureOperationState,
        )
        from erlab.interactive._figurecomposer._defaults import figure_options_context
        from erlab.interactive._figurecomposer._model._sources import (
            _public_source_data,
        )

        targets = self._figure_source_targets(targets)
        if any(
            not self._source_node_for_target(target).is_imagetool for target in targets
        ):
            return self._create_figure_from_provider_targets(
                targets,
                operation=operation,
                custom_code=custom_code,
                title=title,
                show=show,
            )
        resolved_targets, sources, source_data = self._figure_sources_from_targets(
            targets
        )
        if not resolved_targets:
            return None
        source_name_map = self._figure_source_name_map_for_targets(
            resolved_targets, sources
        )
        if operation is not None:
            operation = _seeding._operation_with_source_names(
                operation,
                source_name_map,
            )

        primary_source = sources[0].name
        source_names = tuple(source.name for source in sources)
        auto_operations: tuple[FigureOperationState, ...] = ()
        if (
            operation is None
            and custom_code is None
            and all(
                _public_source_data(data).squeeze(drop=True).ndim > 1
                for data in source_data.values()
            )
        ):
            from erlab.interactive._figurecomposer._exceptions import (
                FigureComposerPlotSlicesSelectionError,
            )

            try:
                auto_operations = (
                    self._figure_operations_from_image_targets(
                        resolved_targets, source_names
                    )
                    or ()
                )
            except FigureComposerPlotSlicesSelectionError as exc:
                self._show_figure_plot_slices_selection_error(exc)
                return None
        with figure_options_context(self._host.effective_interactive_options):
            setup_operation = None if custom_code is not None else operation
            if setup_operation is None and len(auto_operations) == 1:
                setup_operation = auto_operations[0]
            setup = _seeding._setup_for_operation(
                setup_operation,
                source_data,
            )
            all_axes = FigureAxesSelectionState(
                axes=_all_axes_for_shape(setup.nrows, setup.ncols)
            )
            bz_operation = (
                None
                if operation is not None or custom_code is not None
                else self._figure_bz_overlay_operation_from_targets(
                    resolved_targets,
                    source_data,
                    axes=all_axes,
                )
            )
            operations: tuple[FigureOperationState, ...]
            if operation is not None:
                if (
                    operation.kind
                    in {FigureOperationKind.PLOT_ARRAY, FigureOperationKind.PLOT_SLICES}
                    and not operation.axes.expression
                ):
                    operation = operation.model_copy(update={"axes": all_axes})
                operations = (operation,)
            elif custom_code is not None:
                custom_operation = _seeding._operation_with_source_names(
                    FigureOperationState.custom(
                        label=title or "custom code",
                        code=custom_code,
                    ),
                    source_name_map,
                )
                operations = (custom_operation,)
            elif auto_operations:
                operations = _seeding._operations_with_append_axes(
                    auto_operations, all_axes
                )
            else:
                operations = _seeding._make_operations_for_sources(
                    source_data,
                    setup=setup,
                )
            if bz_operation is not None:
                operations = (*operations, bz_operation)

            tool = FigureComposerTool.from_sources(
                source_data,
                sources=tuple(sources),
                operations=operations,
                setup=setup,
                primary_source=primary_source,
            )
        tool._tool_display_name = (
            title if title is not None else self._collection.next_display_name()
        )
        uid = self._host.add_figuretool(tool, show=show)
        self._collection.select_uid(uid)
        return uid

    def create_empty_figure(
        self, *, title: str | None = None, show: bool = True
    ) -> str:
        """Create a manager-owned figure without data sources or recipe steps."""
        from erlab.interactive._figurecomposer import FigureComposerTool
        from erlab.interactive._figurecomposer._defaults import figure_options_context

        with figure_options_context(self._host.effective_interactive_options):
            tool = FigureComposerTool.empty()
        if title is not None:
            tool._tool_display_name = title
        uid = self._host.add_figuretool(tool, show=show)
        self._collection.select_uid(uid)
        return uid

    def _source_operations_for_payloads(
        self,
        sources: tuple[FigureSourceState, ...],
        payloads: Mapping[str, object],
        *,
        setup: FigureSubplotsState | None,
    ) -> tuple[FigureOperationState, ...]:
        from erlab.interactive._figurecomposer._sources import (
            _is_array_source_payload,
            source_spec_for_kind,
        )

        if setup is None:
            setup = _seeding._setup_for_operation(
                None,
                {
                    name: payload.data
                    for name, payload in payloads.items()
                    if _is_array_source_payload(payload) and payload.data is not None
                },
            )
        array_sources = tuple(
            source
            for source in sources
            if _is_array_source_payload(payloads[source.name])
            and source_spec_for_kind(source.kind).default_operation is None
        )
        array_data = {
            source.name: payload.data
            for source in array_sources
            if _is_array_source_payload(payload := payloads[source.name])
            and payload.data is not None
        }
        image_targets = tuple(
            node.uid
            for source in array_sources
            if (node := self._host._tool_graph.nodes.get(source.node_uid or ""))
            is not None
            and node.is_imagetool
        )
        array_operations = None
        if (
            array_sources
            and len(image_targets) == len(array_sources)
            and all(data.squeeze(drop=True).ndim > 1 for data in array_data.values())
        ):
            array_operations = self._figure_operations_from_image_targets(
                image_targets, tuple(source.name for source in array_sources)
            )
        array_operations = array_operations or _seeding._make_operations_for_sources(
            array_data, setup=setup
        )
        operations = []
        arrays_added = False
        for source in sources:
            spec = source_spec_for_kind(source.kind)
            if spec.default_operation is not None:
                operations.append(spec.default_operation(source.name))
            elif source.name in array_data and not arrays_added:
                operations.extend(array_operations)
                arrays_added = True
        return tuple(operations)

    @staticmethod
    def _validate_source_operation(
        operation: FigureOperationState,
        sources: tuple[FigureSourceState, ...],
    ) -> None:
        from erlab.interactive._figurecomposer._model import _operation_metadata
        from erlab.interactive._figurecomposer._operations import _registry

        accepted = _registry.spec_for(operation.kind).accepted_source_kinds
        if accepted is None:
            return
        source_kinds = {source.name: source.kind for source in sources}
        if any(
            source_kinds.get(name) not in accepted
            for name in _operation_metadata.declared_operation_source_names(operation)
        ):
            raise ValueError("Operation is incompatible with the figure source type.")

    def _create_figure_from_provider_targets(
        self,
        targets: Iterable[int | str | FigureSourceTarget],
        *,
        operation: FigureOperationState | None = None,
        custom_code: str | None = None,
        title: str | None = None,
        show: bool = True,
    ) -> str | None:
        from erlab.interactive._figurecomposer import (
            FigureAxesSelectionState,
            FigureComposerTool,
            FigureOperationState,
        )
        from erlab.interactive._figurecomposer._defaults import figure_options_context
        from erlab.interactive._figurecomposer._exceptions import (
            FigureComposerPlotSlicesSelectionError,
        )
        from erlab.interactive._figurecomposer._sources import _is_array_source_payload

        sources, payloads, name_map = self._capture_figure_sources(targets)
        if not sources:
            return None
        source_data = {
            name: payload.data
            for name, payload in payloads.items()
            if _is_array_source_payload(payload) and payload.data is not None
        }
        requested = operation
        if custom_code is not None:
            requested = FigureOperationState.custom(
                label=title or "custom code", code=custom_code
            )
        if requested is not None:
            requested = _seeding._operation_with_source_names(requested, name_map)
            self._validate_source_operation(requested, sources)
        with figure_options_context(self._host.effective_interactive_options):
            setup = _seeding._setup_for_operation(requested, source_data)
            try:
                operations = (
                    (requested,)
                    if requested is not None
                    else self._source_operations_for_payloads(
                        sources, payloads, setup=setup
                    )
                )
            except FigureComposerPlotSlicesSelectionError as exc:
                self._show_figure_plot_slices_selection_error(exc)
                return None
            if requested is None:
                bz_operation = self._figure_bz_overlay_operation_from_targets(
                    tuple(
                        source.node_uid
                        for source in sources
                        if source.name in source_data and source.node_uid is not None
                    ),
                    source_data,
                    axes=FigureAxesSelectionState(
                        axes=_all_axes_for_shape(setup.nrows, setup.ncols)
                    ),
                )
                if bz_operation is not None:
                    operations = (*operations, bz_operation)
            tool = FigureComposerTool.from_source_payloads(
                payloads,
                sources=sources,
                operations=operations,
                setup=setup,
                primary_source=sources[0].name,
            )
        tool._tool_display_name = (
            title if title is not None else self._collection.next_display_name()
        )
        uid = self._host.add_figuretool(tool, show=show)
        self._collection.select_uid(uid)
        return uid

    def _add_fit_targets_to_figure(
        self,
        figure_uid: str,
        targets: Iterable[int | str | FigureSourceTarget],
        *,
        show: bool = False,
    ) -> bool:
        """Compatibility entry point for existing fit actions."""
        return self._add_imagetool_sources_to_figure(figure_uid, targets, show=show)

    def _append_provider_targets_to_figure(
        self,
        figure_uid: str,
        targets: Iterable[int | str | FigureSourceTarget],
        *,
        axes: FigureAxesSelectionState,
        operation: FigureOperationState | None = None,
        show: bool = True,
    ) -> bool:
        from erlab.interactive._figurecomposer import FigureComposerTool
        from erlab.interactive._figurecomposer._exceptions import (
            FigureComposerPlotSlicesSelectionError,
        )
        from erlab.interactive._figurecomposer._sources import _is_array_source_payload

        if not self._host._is_figure_uid(figure_uid):
            return False
        node = self._host._child_node(figure_uid)
        composer = node.tool_window
        if not isinstance(composer, FigureComposerTool):
            return False
        sources, payloads, name_map = self._capture_figure_sources(
            targets,
            reserved_sources=(source.name for source in composer.source_states()),
        )
        if not sources:
            return False
        if operation is not None:
            operation = _seeding._operation_with_source_names(operation, name_map)
            self._validate_source_operation(operation, sources)
        try:
            operations = (
                (operation,)
                if operation is not None
                else self._source_operations_for_payloads(
                    sources, payloads, setup=composer.tool_status.setup
                )
            )
        except FigureComposerPlotSlicesSelectionError as exc:
            self._show_figure_plot_slices_selection_error(exc)
            return False
        if operation is None:
            array_data = {
                name: payload.data
                for name, payload in payloads.items()
                if _is_array_source_payload(payload) and payload.data is not None
            }
            bz_operation = self._figure_bz_overlay_operation_from_targets(
                tuple(
                    source.node_uid
                    for source in sources
                    if source.name in array_data and source.node_uid is not None
                ),
                array_data,
                axes=axes,
            )
            if bz_operation is not None:
                operations = (*operations, bz_operation)
        operations = _seeding._operations_with_append_axes(operations, axes)
        if any(composer._operation_has_invalid_axes(item) for item in operations):
            return False
        added = composer.add_source_payloads(sources, payloads)
        if not added:
            return False
        if not added.skipped:
            for item in operations:
                item = _seeding._operation_with_source_names(item, added.name_map)
                composer.add_operation(
                    item.model_copy(update={"operation_id": uuid.uuid4().hex})
                )
        self._host._mark_workspace_dirty(uid=figure_uid, data=True, state=True)
        self._collection.select_uid(figure_uid)
        if show:
            node.show()
        return True

    def add_fit_to_figure(self, fit_tool: Fit1DTool) -> str | None:
        """Add the current completed fit through the existing destination dialog."""
        from erlab.interactive._figurecomposer import (
            FigureComposerTool,
            FigureOperationState,
        )
        from erlab.interactive._figurecomposer._defaults import figure_options_context
        from erlab.interactive.imagetool.manager._figurecomposer import _dialogs

        if not fit_tool.has_figure_fit_source():
            return None
        tool_uid = self._host._node_uid_from_window(fit_tool)
        if tool_uid is None:
            return None
        sources, payloads, _name_map = self._capture_figure_sources(
            (FigureSourceTarget(tool_uid, "fit"),)
        )
        if len(sources) != 1:
            return None
        source = sources[0]
        payload = payloads[source.name]
        operation = FigureOperationState.fit_plot(label="Fit", source=source.name)
        figure_uids = tuple(self._host._figure_uids())
        if not figure_uids:
            action = _dialogs._FIGURE_DIALOG_NEW
            dialog = None
        else:
            dialog = _dialogs._AppendFigureTargetDialog(
                self._host,
                figure_uids,
                operation,
                allow_new_figure=True,
                source_count=1,
                source_kind="fit",
                selected_figure_uid=self._selected_figure_uid_for_figure_dialog(),
            )
            dialog.action_combo.setCurrentIndex(
                dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_ADD_STEP)
            )
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return None
            action = dialog.selected_action()
        if action == _dialogs._FIGURE_DIALOG_NEW:
            with figure_options_context(self._host.effective_interactive_options):
                composer = FigureComposerTool.from_source_payloads(
                    payloads, sources=(source,), operations=(operation,)
                )
            uid = self._host.add_figuretool(composer)
            self._collection.select_uid(uid)
            return uid
        if dialog is None:
            return None
        figure_uid = dialog.figure_uid()
        if not self._host._is_figure_uid(figure_uid):
            return None
        node = self._host._child_node(figure_uid)
        composer = node.tool_window
        if not isinstance(composer, FigureComposerTool):
            return None
        if action == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE:
            alias = dialog.selected_source_alias()
            if alias is None or not composer.replace_source_payload(
                alias, source, payload
            ):
                return None
        else:
            axes = None
            if action == _dialogs._FIGURE_DIALOG_ADD_STEP:
                selected = dialog.selected_target()
                if selected is None:
                    return None
                _, axes = selected
                if composer._operation_has_invalid_axes(
                    operation.model_copy(update={"axes": axes})
                ):
                    return None
            added = composer.add_source_payloads((source,), payloads)
            if not added:
                return None
            if action == _dialogs._FIGURE_DIALOG_ADD_STEP:
                composer.add_operation(
                    FigureOperationState.fit_plot(
                        label="Fit", source=added.name_map[source.name], axes=axes
                    )
                )
        self._host._mark_workspace_dirty(uid=figure_uid, data=True, state=True)
        self._collection.select_uid(figure_uid)
        node.show()
        return figure_uid

    @QtCore.Slot()
    def create_figure_from_selection(self) -> None:
        from erlab.interactive._figurecomposer._sources import source_spec_for_kind
        from erlab.interactive.imagetool.manager._figurecomposer import _dialogs

        targets = self._host._selected_figure_source_targets()
        sources, payloads, _name_map = self._capture_figure_sources(targets)
        if not sources:
            return
        figure_uids = tuple(self._host._figure_uids())
        if not figure_uids:
            self.create_figure_from_targets(targets)
            return
        default_operations = tuple(
            factory(source.name)
            for source in sources
            if (factory := source_spec_for_kind(source.kind).default_operation)
            is not None
        )
        source_kinds = {source.kind for source in sources}
        dialog = _dialogs._AppendFigureTargetDialog(
            self._host,
            figure_uids,
            None,
            allow_new_figure=True,
            source_count=len(sources),
            source_kind=next(iter(source_kinds)) if len(source_kinds) == 1 else None,
            default_operations=default_operations,
            selected_figure_uid=self._selected_figure_uid_for_figure_dialog(),
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        action = dialog.selected_action()
        if action == _dialogs._FIGURE_DIALOG_NEW:
            self.create_figure_from_targets(targets)
        elif action == _dialogs._FIGURE_DIALOG_ADD_SOURCE:
            self._add_imagetool_sources_to_figure(
                dialog.figure_uid(), targets, show=True
            )
        elif action == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE and len(sources) == 1:
            alias = dialog.selected_source_alias()
            if alias is not None:
                self._replace_figure_source_payload(
                    dialog.figure_uid(), alias, sources[0], payloads[sources[0].name]
                )
        elif action == _dialogs._FIGURE_DIALOG_ADD_STEP:
            selected = dialog.selected_target()
            if selected is not None:
                figure_uid, axes_selection = selected
                self.append_figure_from_targets(
                    targets, figure_uid=figure_uid, axes_selection=axes_selection
                )

    def create_figure_from_slicer_area(
        self,
        slicer_area: ImageSlicerArea,
        *,
        operation: FigureOperationState | None = None,
        custom_code: str | None = None,
        title: str | None = None,
        show: bool = True,
    ) -> str | None:
        target = self._host.target_from_slicer_area(slicer_area)
        if target is None:
            return None
        return self.create_figure_from_targets(
            (target,),
            operation=operation,
            custom_code=custom_code,
            title=title,
            show=show,
        )

    def _append_single_axis_selection(
        self, figure_uid: str
    ) -> FigureAxesSelectionState | None:
        from erlab.interactive._figurecomposer import FigureAxesSelectionState
        from erlab.interactive._figurecomposer._model._gridspec import (
            _gridspec_all_axes_ids,
            _gridspec_valid_axes_ids,
        )

        tool = self._host._child_node(figure_uid).tool_window
        if tool is None:
            return None
        setup = tool.tool_status.setup
        if setup.layout_mode == "gridspec":
            axes_ids = _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup))
            if len(axes_ids) == 1:
                return FigureAxesSelectionState(axes_ids=axes_ids)
            return None
        all_axes = _all_axes_for_shape(setup.nrows, setup.ncols)
        if len(all_axes) == 1:
            return FigureAxesSelectionState(axes=all_axes)
        return None

    def _prompt_append_figure_target(
        self,
        operation: FigureOperationState | None,
        *,
        figure_uid: str | None = None,
        default_operations: tuple[FigureOperationState, ...] = (),
    ) -> tuple[str, FigureAxesSelectionState] | None:
        from erlab.interactive.imagetool.manager._figurecomposer import _dialogs

        figure_uids = self._host._figure_uids()
        if not figure_uids:
            return None
        if figure_uid is not None:
            if not self._host._is_figure_uid(figure_uid):
                return None
            automatic = self._append_single_axis_selection(figure_uid)
            if automatic is not None:
                return figure_uid, automatic
            figure_uids = (figure_uid,)
        elif len(figure_uids) == 1:
            automatic = self._append_single_axis_selection(figure_uids[0])
            if automatic is not None:
                return figure_uids[0], automatic

        dialog = _dialogs._AppendFigureTargetDialog(
            self._host,
            tuple(figure_uids),
            operation,
            default_operations=default_operations,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.selected_target()

    def append_figure_from_targets(
        self,
        targets: Iterable[int | str | FigureSourceTarget],
        *,
        figure_uid: str | None = None,
        axes_selection: FigureAxesSelectionState | None = None,
        operation: FigureOperationState | None = None,
        show: bool = True,
    ) -> bool:
        from erlab.interactive._figurecomposer import FigureComposerTool
        from erlab.interactive._figurecomposer._defaults import figure_options_context
        from erlab.interactive._figurecomposer._model._sources import (
            _public_source_data,
        )

        targets = self._figure_source_targets(targets)
        if any(
            not self._source_node_for_target(target).is_imagetool for target in targets
        ):
            if axes_selection is None:
                from erlab.interactive._figurecomposer._exceptions import (
                    FigureComposerPlotSlicesSelectionError,
                )

                sources, payloads, _name_map = self._capture_figure_sources(targets)
                try:
                    defaults = self._source_operations_for_payloads(
                        sources, payloads, setup=None
                    )
                except FigureComposerPlotSlicesSelectionError as exc:
                    self._show_figure_plot_slices_selection_error(exc)
                    return False
                prompt_operation = operation or (
                    defaults[0] if len(defaults) == 1 else None
                )
                prompt = self._prompt_append_figure_target(
                    prompt_operation,
                    figure_uid=figure_uid,
                    default_operations=defaults,
                )
                if prompt is None:
                    return False
                figure_uid, axes_selection = prompt
            if figure_uid is None:
                return False
            return self._append_provider_targets_to_figure(
                figure_uid, targets, axes=axes_selection, operation=operation, show=show
            )
        resolved_targets = self._figure_imagetool_targets(targets)
        if not resolved_targets:
            return False

        _, sources, source_data = self._figure_sources_from_targets(resolved_targets)
        prompt_operation = (
            None
            if operation is None
            else _seeding._operation_with_source_names(
                operation,
                self._figure_source_name_map_for_targets(resolved_targets, sources),
            )
        )

        source_names = tuple(source.name for source in sources)
        auto_operations: tuple[FigureOperationState, ...] = ()
        if operation is None and all(
            _public_source_data(data).squeeze(drop=True).ndim > 1
            for data in source_data.values()
        ):
            from erlab.interactive._figurecomposer._exceptions import (
                FigureComposerPlotSlicesSelectionError,
            )

            try:
                auto_operations = (
                    self._figure_operations_from_image_targets(
                        resolved_targets, source_names
                    )
                    or ()
                )
            except FigureComposerPlotSlicesSelectionError as exc:
                self._show_figure_plot_slices_selection_error(exc)
                return False
        if prompt_operation is None and len(auto_operations) == 1:
            prompt_operation = auto_operations[0]

        if axes_selection is None:
            prompt = self._prompt_append_figure_target(
                prompt_operation, figure_uid=figure_uid
            )
            if prompt is None:
                return False
            resolved_figure_uid, axes_selection = prompt
        else:
            if figure_uid is None or not self._host._is_figure_uid(figure_uid):
                return False
            resolved_figure_uid = figure_uid

        node = self._host._child_node(resolved_figure_uid)
        tool = node.tool_window
        if not isinstance(tool, FigureComposerTool):
            return False

        existing_source_names = tuple(tool.source_data()) + tuple(
            source.name for source in tool.source_states()
        )
        _, sources, source_data = self._figure_sources_from_targets(
            resolved_targets,
            reserved_sources=existing_source_names,
        )
        append_operation = (
            None
            if operation is None
            else _seeding._operation_with_source_names(
                operation,
                self._figure_source_name_map_for_targets(resolved_targets, sources),
            )
        )
        source_names = tuple(source.name for source in sources)
        auto_operations = ()
        if operation is None and all(
            _public_source_data(data).squeeze(drop=True).ndim > 1
            for data in source_data.values()
        ):
            from erlab.interactive._figurecomposer._exceptions import (
                FigureComposerPlotSlicesSelectionError,
            )

            try:
                auto_operations = (
                    self._figure_operations_from_image_targets(
                        resolved_targets, source_names
                    )
                    or ()
                )
            except FigureComposerPlotSlicesSelectionError as exc:
                self._show_figure_plot_slices_selection_error(exc)
                return False

        add_result = tool.add_sources(tuple(sources), source_data)
        if not add_result:
            return False
        if add_result.skipped:
            self._host._mark_workspace_dirty(uid=resolved_figure_uid, state=True)
            self._collection.select_uid(resolved_figure_uid)
            if show:
                node.show()
            return True
        with figure_options_context(self._host.effective_interactive_options):
            operations = (
                (append_operation,)
                if append_operation is not None
                else auto_operations
                or _seeding._make_operations_for_sources(
                    source_data, setup=tool.tool_status.setup
                )
            )
            if operation is None:
                bz_operation = self._figure_bz_overlay_operation_from_targets(
                    resolved_targets,
                    source_data,
                    axes=axes_selection,
                )
                if bz_operation is not None:
                    operations = (*operations, bz_operation)
        source_name_map = {
            requested: stored
            for requested, stored in add_result.name_map.items()
            if requested != stored
        }
        if source_name_map:
            operations = tuple(
                _seeding._operation_with_source_names(appended, source_name_map)
                for appended in operations
            )
        for appended in _seeding._operations_with_append_axes(
            operations,
            axes_selection,
        ):
            tool.add_operation(appended)

        self._host._mark_workspace_dirty(uid=resolved_figure_uid, state=True)
        self._collection.select_uid(resolved_figure_uid)
        if show:
            node.show()
        return True

    def append_figure_from_slicer_area(
        self,
        slicer_area: ImageSlicerArea,
        *,
        operation: FigureOperationState,
        show: bool = True,
    ) -> bool:
        target = self._host.target_from_slicer_area(slicer_area)
        if target is None:
            return False
        return self.append_figure_from_targets(
            (target,), operation=operation, show=show
        )
