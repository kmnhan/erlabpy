"""Coordinate manager-owned Figure Composer windows and source workflows."""

from __future__ import annotations

import functools
import typing

from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._figurecomposer import _seeding
from erlab.interactive._figurecomposer._model._axes import _all_axes_for_shape
from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer import (
        FigureAxesSelectionState,
        FigureComposerTool,
        FigureOperationState,
        FigureSourceState,
    )
    from erlab.interactive._figurecomposer._model._document import FigureSourceAddResult
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
                    "xlim": first_operation.xlim,
                    "ylim": first_operation.ylim,
                    "crop": first_operation.crop,
                    "axis": first_operation.axis,
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
            self._host._script_input_for_node(node)
        )
        alias = _source_unique_name(
            _source_alias_candidate(data) or source.name, reserved
        )
        return source.model_copy(update={"name": alias})

    def _figure_sources_from_targets(
        self,
        targets: Iterable[int | str],
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
            node = self._host._node_for_target(target)
            data = node.current_source_data()
            source = self._figure_source_from_node(node, data, reserved)
            source_data[source.name] = data
            sources.append(source)
        return resolved_targets, tuple(sources), source_data

    def _figure_source_name_map_for_targets(
        self, targets: Iterable[int | str], sources: Iterable[FigureSourceState]
    ) -> dict[str, str]:
        source_name_map: dict[str, str] = {}
        for target, source in zip(targets, sources, strict=True):
            old_name = self._host._script_input_name_for_node(
                self._host._node_for_target(target)
            )
            if old_name != source.name:
                source_name_map[old_name] = source.name
        return source_name_map

    def _figure_source_uid_for_target(self, target: int | str) -> str | None:
        try:
            node = self._host._node_for_target(target)
        except KeyError:
            return None
        return node.uid if node.is_imagetool else None

    def _figure_imagetool_targets(
        self, targets: Iterable[int | str]
    ) -> tuple[int | str, ...]:
        resolved: list[int | str] = []
        seen_uids: set[str] = set()
        for target in targets:
            uid = self._figure_source_uid_for_target(target)
            if uid is None or uid in seen_uids:
                continue
            resolved.append(target)
            seen_uids.add(uid)
        return tuple(resolved)

    def _selected_figure_source_uids(self) -> tuple[str, ...]:
        uids: list[str] = []
        for target in self._host._selected_imagetool_targets():
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
            and node.is_imagetool
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
        targets: Iterable[int | str],
        *,
        show: bool = False,
    ) -> bool:
        requested_targets = tuple(targets)
        skipped_targets = sum(
            1
            for target in requested_targets
            if self._figure_source_uid_for_target(target) is None
        )
        resolved_targets, sources, source_data = self._figure_sources_from_targets(
            requested_targets
        )
        if not resolved_targets:
            return False
        node = (
            self._host._child_node(figure_uid)
            if self._host._is_figure_uid(figure_uid)
            else None
        )
        from erlab.interactive._figurecomposer import FigureComposerTool

        result = self._add_sources_to_figure(
            figure_uid, sources, source_data, show=show
        )
        if not result:
            return False
        tool = node.tool_window if node is not None else None
        if isinstance(tool, FigureComposerTool):
            source_error = (
                "Could not update source data for: "
                + ", ".join(detail for _name, detail in result.skipped)
                if result.skipped
                else ""
            )
            added = len(result.added)
            updated = len(result.updated)
            parts: list[str] = []
            if added:
                suffix = "source" if added == 1 else "sources"
                parts.append(f"Added {added} ImageTool {suffix}")
            if updated:
                suffix = "source" if updated == 1 else "sources"
                parts.append(f"Updated {updated} ImageTool {suffix}")
            if result.skipped:
                skipped = len(result.skipped)
                suffix = "update" if skipped == 1 else "updates"
                parts.append(f"Skipped {skipped} ImageTool source {suffix}")
            if skipped_targets:
                suffix = "selection" if skipped_targets == 1 else "selections"
                parts.append(f"Skipped {skipped_targets} unsupported {suffix}")
            status = "; ".join(parts) + "."
            if source_error:
                status = f"{status} {source_error}"
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

    def _can_request_add_sources_to_figure(self, figure_uid: str) -> bool:
        return self._host._is_figure_uid(figure_uid) and any(
            node.is_imagetool for node in self._host._tool_graph.nodes.values()
        )

    def _request_add_sources_to_figure(self, figure_uid: str) -> bool:
        from erlab.interactive.imagetool.manager._figurecomposer import _dialogs

        if not self._host._is_figure_uid(figure_uid):
            return False
        dialog = _dialogs._FigureSourcePickerDialog(
            self._host, prechecked_uids=self._selected_figure_source_uids()
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
        return self._host._tool_graph.nodes.get(source.node_uid)

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
        return node is not None and node.is_imagetool

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
                and node.is_imagetool
            )
        )
        return self._host.reveal_nodes(uids)

    def _can_refresh_figure_source(self, figure_uid: str, source_name: str) -> bool:
        return self._figure_source_live_node(figure_uid, source_name) is not None

    def _figure_source_refresh_label(
        self, figure_uid: str, source_name: str
    ) -> str | None:
        node = self._figure_source_node(figure_uid, source_name)
        return None if node is None else node.display_text

    def _refresh_figure_source(self, figure_uid: str, source_name: str) -> bool:
        """Refresh one figure source from its linked open ImageTool window."""
        from erlab.interactive._figurecomposer import (
            FigureComposerTool,
            FigureSourceState,
        )

        source_node = self._figure_source_live_node(figure_uid, source_name)
        if source_node is None or not self._host._is_figure_uid(figure_uid):
            return False
        figure_node = self._host._child_node(figure_uid)
        tool = figure_node.tool_window
        if not isinstance(tool, FigureComposerTool):
            return False

        data = source_node.current_source_data()
        source = FigureSourceState.from_script_input(
            self._host._script_input_for_node(source_node)
        )
        if not tool.replace_source(source_name, source, data):
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
        targets: Iterable[int | str],
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
                        trusted=True,
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

    @QtCore.Slot()
    def create_figure_from_selection(self) -> None:
        from erlab.interactive.imagetool.manager._figurecomposer import _dialogs

        targets = self._host._selected_figure_source_targets()
        if not targets:
            return
        resolved_targets, sources, source_data = self._figure_sources_from_targets(
            targets
        )
        if not resolved_targets:
            return
        figure_uids = tuple(self._host._figure_uids())
        if not figure_uids:
            self.create_figure_from_targets(resolved_targets)
            return

        dialog = _dialogs._AppendFigureTargetDialog(
            self._host,
            figure_uids,
            None,
            allow_new_figure=True,
            source_count=len(sources),
            selected_figure_uid=self._selected_figure_uid_for_figure_dialog(),
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        action = dialog.selected_action()
        if action == _dialogs._FIGURE_DIALOG_NEW:
            self.create_figure_from_targets(resolved_targets)
            return
        if action == _dialogs._FIGURE_DIALOG_ADD_SOURCE:
            self._add_sources_to_figure(dialog.figure_uid(), sources, source_data)
            return
        if action == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE:
            alias = dialog.selected_source_alias()
            if alias is None:
                return
            self._replace_figure_source(
                dialog.figure_uid(),
                alias,
                sources,
                source_data,
            )
            return
        target = dialog.selected_target()
        if target is None:
            return
        figure_uid, axes_selection = target
        self.append_figure_from_targets(
            resolved_targets,
            figure_uid=figure_uid,
            axes_selection=axes_selection,
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
        self, operation: FigureOperationState | None, *, figure_uid: str | None = None
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
            self._host, tuple(figure_uids), operation
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.selected_target()

    def append_figure_from_targets(
        self,
        targets: Iterable[int | str],
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
