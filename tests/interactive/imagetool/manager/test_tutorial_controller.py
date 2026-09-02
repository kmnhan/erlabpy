from __future__ import annotations

import contextlib
import json
import pathlib
import threading

import pytest
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._tutorial.controller as tutorial
import erlab.interactive.imagetool.manager._tutorial.framework as tutorial_framework


class _WorkspaceController:
    def __init__(self) -> None:
        self.clean_calls = 0

    def _mark_workspace_clean(self) -> None:
        self.clean_calls += 1


class _Manager(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._workspace_controller = _WorkspaceController()
        self._file_handlers: set[object] = set()
        self._metadata_copy_selected_action = QtGui.QAction(self)
        self.close_calls = 0
        self.reject_close = False
        self._option_overrides: dict[str, object] = {}

    def workspace_option_overrides(self) -> dict[str, object]:
        return dict(self._option_overrides)

    def _set_workspace_option_overrides(
        self, overrides, *, mark_dirty: bool = True
    ) -> None:
        del mark_dirty
        self._option_overrides = dict(overrides)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self.close_calls += 1
        if self.reject_close and event is not None:
            event.ignore()
            return
        super().closeEvent(event)


class _LoaderContext:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> object:
        self.entered += 1
        return object()

    def __exit__(self, *_args: object) -> None:
        self.exited += 1


def _data_files(directory: pathlib.Path) -> tutorial.TutorialDataFiles:
    return tutorial.TutorialDataFiles(
        map=directory / "tutorial_map.h5",
        cut=directory / "tutorial_cut.h5",
    )


def _immediate_generation(
    directory: pathlib.Path,
    *,
    is_cancelled,
    on_file_published,
) -> tutorial.TutorialDataFiles:
    del is_cancelled
    files = _data_files(pathlib.Path(directory))
    on_file_published(files.map)
    on_file_published(files.cut)
    return files


def _controller(
    monkeypatch, manager: _Manager
) -> tuple[tutorial._TutorialController, _LoaderContext]:
    loader_context = _LoaderContext()
    monkeypatch.setattr(
        tutorial, "tutorial_loader_registration", lambda: loader_context
    )
    controller = tutorial._TutorialController(manager)
    return controller, loader_context


def test_tutorial_sequence_has_stable_forward_only_ids(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)

    ids = [step.id for step in controller.steps]
    assert ids[0] == "welcome"
    assert ids[-1] == "tutorial-complete"
    assert len(ids) == len(set(ids))
    assert ids.index("manager-introduction") < ids.index("open-data-explorer")
    assert ids.index("open-data-explorer") < ids.index("data-explorer-introduction")
    assert ids.index("data-explorer-introduction") < ids.index("select-map")
    assert ids.index("select-map") < ids.index("enable-map-preview")
    assert ids.index("enable-map-preview") < ids.index("explorer-preview")
    assert ids.index("explorer-preview") < ids.index("explorer-folder")
    assert ids.index("explorer-folder") < ids.index("explorer-loader")
    assert ids.index("explorer-loader") < ids.index("open-map-in-manager")
    preview_step = next(
        step for step in controller.steps if step.id == "explorer-preview"
    )
    assert {"mouse", "key", "wheel"} <= preview_step.allowed_inputs
    assert preview_step.allowed_objects
    energy_step = next(
        step for step in controller.steps if step.id == "inspect-kinetic-energy"
    )
    assert energy_step.allowed_inputs == frozenset({"mouse"})
    assert energy_step.allowed_objects
    assert energy_step.event_predicate is tutorial._cursor_modifier_key_event_predicate
    geometry_step = next(
        step
        for step in controller.steps
        if step.id == "set-normal-emission-and-azimuth"
    )
    assert geometry_step.allowed_objects == (controller._cursor_visibility_action,)
    assert ids.index("open-map-in-manager") < ids.index("imagetool-plots")
    assert ids.index("imagetool-plots") < ids.index("ctrl-drag-cursor")
    assert ids.index("ctrl-drag-cursor") < ids.index("imagetool-cursor-controls")
    assert ids.index("ctrl-drag-cursor") < ids.index("transpose-alpha-beta")
    assert ids.index("add-second-cursor") < ids.index("move-second-cursor")
    assert ids.index("move-second-cursor") < ids.index("move-all-cursors")
    assert ids.index("move-all-cursors") < ids.index("set-second-cursor-bin")
    assert ids.index("set-second-cursor-bin") < ids.index("select-first-cursor")
    assert "reveal-map-in-manager" not in ids
    assert "open-map-imagetool" not in ids
    assert ids.index("imagetool-menus") < ids.index("inspect-kinetic-energy")
    assert ids.index("inspect-kinetic-energy") < ids.index("open-coordinate-editor")
    assert ids.index("open-coordinate-editor") < ids.index("select-energy-coordinate")
    assert ids.index("select-energy-coordinate") < ids.index("select-scale-offset")
    assert ids.index("select-scale-offset") < ids.index("set-energy-offset")
    assert ids.index("set-energy-offset") < ids.index("apply-energy-correction")
    assert ids.index("apply-energy-correction") < ids.index("inspect-binding-energy")
    assert ids.index("apply-energy-correction") < ids.index("open-ktool")
    assert ids.index("ktool-grid") < ids.index("select-ktool-visualization")
    assert ids.index("select-ktool-visualization") < ids.index("ktool-brillouin-zone")
    assert ids.index("ktool-brillouin-zone") < ids.index("ktool-energy-preview")
    assert ids.index("ktool-energy-preview") < ids.index("select-ktool-parameters")
    assert ids.index("select-ktool-parameters") < ids.index("open-converted-map")
    assert ids.index("switch-to-manager-provenance") < ids.index("manager-overview")
    assert ids.index("manager-overview") < ids.index("select-manager-provenance")
    assert ids.index("select-manager-provenance") < ids.index("provenance-overview")
    assert ids.index("switch-to-manager-operations") < ids.index("top-level-cut")
    assert ids.index("top-level-cut") < ids.index("select-converted-map")
    assert ids.index("select-converted-map") < ids.index("expand-input-history")
    assert ids.index("expand-input-history") < ids.index("select-reusable-operations")
    assert ids.index("select-raw-cut") < ids.index("select-raw-cut-provenance")
    assert ids.index("select-raw-cut-provenance") < ids.index(
        "paste-reusable-operations"
    )
    assert ids.index("paste-reusable-operations") < ids.index("new-figure")
    assert "open-figure-context-menu" not in ids
    assert "switch-to-figure-composer" not in ids
    assert ids.index("new-figure") < ids.index("figure-composer-output")
    assert ids.index("figure-composer-output") < ids.index(
        "select-figure-composer-sources"
    )
    assert ids.index("select-figure-composer-sources") < ids.index(
        "figure-composer-sources"
    )
    assert ids.index("figure-composer-sources") < ids.index(
        "select-figure-composer-layout"
    )
    assert ids.index("select-figure-composer-layout") < ids.index(
        "figure-composer-layout"
    )
    assert ids.index("figure-composer-layout") < ids.index(
        "select-figure-composer-recipe"
    )
    assert ids.index("select-figure-composer-recipe") < ids.index(
        "figure-composer-recipe"
    )
    assert ids.index("figure-composer-recipe") < ids.index("reveal-figure-in-manager")
    assert ids.index("reveal-figure-in-manager") < ids.index("manager-figures")
    assert ids.index("manager-figures") < ids.index("tutorial-complete")
    assert all(
        step.card_position == "center" and step.auto_advance
        for step in controller.steps
        if step.id.startswith("switch-to-")
    )
    assert all(
        step.card_position == "bottom"
        for step in controller.steps
        if step.id
        in {
            "select-figure-composer-sources",
            "figure-composer-sources",
            "select-figure-composer-layout",
            "figure-composer-layout",
            "select-figure-composer-recipe",
            "figure-composer-recipe",
        }
    )
    assert {
        step.id
        for step in controller.steps
        if step.mode == "action" and not step.auto_advance
    } == {
        "select-map",
        "enable-map-preview",
        "ctrl-drag-cursor",
        "transpose-alpha-beta",
        "set-energy-bin",
        "add-second-cursor",
        "move-second-cursor",
        "move-all-cursors",
        "set-second-cursor-bin",
        "select-first-cursor",
        "select-energy-coordinate",
        "select-scale-offset",
        "set-energy-offset",
        "select-c6-guideline",
        "set-normal-emission-and-azimuth",
        "select-ktool-visualization",
        "ktool-brillouin-zone",
        "ktool-energy-preview",
        "select-ktool-parameters",
        "select-manager-provenance",
        "select-cut",
        "select-converted-map",
        "expand-input-history",
        "select-reusable-operations",
        "copy-reusable-operations",
        "select-raw-cut",
        "select-raw-cut-provenance",
        "paste-reusable-operations",
        "select-figure-composer-sources",
        "select-figure-composer-layout",
        "select-figure-composer-recipe",
    }
    assert controller.steps[-1].continue_label == "Finish"
    assert all(
        step.debug_action is not None
        for step in controller.steps
        if step.mode == "action"
    )
    assert manager._option_overrides == tutorial._TUTORIAL_OPTION_OVERRIDES

    controller._finish_cleanup()


def test_tutorial_clipboard_requires_the_copied_operations(monkeypatch, qtbot) -> None:
    from erlab.interactive.imagetool import _kspace_conversion
    from erlab.interactive.imagetool._provenance._model import (
        ReplayStep,
        stamp_operation_group,
    )
    from erlab.interactive.imagetool._provenance._operations import (
        AffineCoordOperation,
        AverageOperation,
        KspaceConvertOperation,
        KspaceSetNormalOperation,
    )
    from erlab.interactive.imagetool.manager import _details_panel

    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)
    grouped = stamp_operation_group(
        (
            KspaceSetNormalOperation(alpha=2.0, beta=-1.5, delta=-4.0),
            KspaceConvertOperation(),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )
    expected = (
        AffineCoordOperation(coord_name="eV", scale=1.0, offset=-45.5),
        *grouped,
    )

    class _DetailsPanel:
        @staticmethod
        def _selected_derivation_step_payload():
            return _details_panel._ProvenanceStepsClipboardPayload(
                steps=tuple(ReplayStep(operation=operation) for operation in expected),
                active_name="example_map",
            )

    manager._details_panel = _DetailsPanel()
    monkeypatch.setattr(controller, "_reusable_operations_selected", lambda: True)

    def set_operations(operations) -> None:
        payload = {
            "type": _details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
            "version": _details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_VERSION,
            "active_name": "example_map",
            "steps": [
                ReplayStep(operation=operation).model_dump(mode="json")
                for operation in operations
            ],
        }
        mime_data = QtCore.QMimeData()
        mime_data.setData(
            _details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
            json.dumps(payload).encode("utf-8"),
        )
        QtWidgets.QApplication.clipboard().setMimeData(mime_data)

    set_operations(expected)
    controller._operations_copy_triggered()
    assert controller._provenance_steps_on_clipboard()

    QtWidgets.QApplication.clipboard().setText("unrelated clipboard text")
    assert not controller._provenance_steps_on_clipboard()

    controller._operations_copy_triggered()
    assert not controller._provenance_steps_on_clipboard()
    set_operations(expected)
    assert controller._provenance_steps_on_clipboard()

    set_operations((AverageOperation(dims=("eV",)),))
    assert not controller._provenance_steps_on_clipboard()
    menu = QtWidgets.QMenu(manager)
    event = QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress)
    assert not controller._paste_operations_event_predicate(menu, event)

    set_operations(
        (
            AffineCoordOperation(coord_name="eV", scale=1.0, offset=-1.0),
            *grouped,
        )
    )
    controller._operations_copy_triggered()
    assert not controller._provenance_steps_on_clipboard()

    set_operations(
        (
            expected[0],
            expected[1],
            expected[2].model_copy(update={"method": "nearest"}),
        )
    )
    controller._operations_copy_triggered()
    assert not controller._provenance_steps_on_clipboard()

    set_operations(expected)
    controller._operations_copy_triggered()
    assert controller._provenance_steps_on_clipboard()
    assert controller._paste_operations_event_predicate(menu, event)
    QtWidgets.QApplication.clipboard().clear()
    controller._finish_cleanup()


def test_provenance_completion_requires_the_visible_tab(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    layout = QtWidgets.QVBoxLayout(manager)
    manager.inspector_tabs = QtWidgets.QTabWidget(manager)
    manager.metadata_details_page = QtWidgets.QWidget(manager.inspector_tabs)
    manager.metadata_provenance_page = QtWidgets.QWidget(manager.inspector_tabs)
    provenance_layout = QtWidgets.QVBoxLayout(manager.metadata_provenance_page)
    manager.metadata_derivation_list = QtWidgets.QTreeWidget(
        manager.metadata_provenance_page
    )
    provenance_layout.addWidget(manager.metadata_derivation_list)
    manager.inspector_tabs.addTab(manager.metadata_details_page, "Details")
    manager.inspector_tabs.addTab(manager.metadata_provenance_page, "Provenance")
    layout.addWidget(manager.inspector_tabs)
    manager.show()
    manager._metadata_node_uid = "previous"

    controller, _loader_context = _controller(monkeypatch, manager)
    manager.inspector_tabs.setCurrentWidget(manager.metadata_details_page)
    assert not controller._manager_provenance_is_visible_for("selected")

    manager.inspector_tabs.setCurrentWidget(manager.metadata_provenance_page)
    qtbot.waitUntil(manager.metadata_derivation_list.isVisible)
    assert not controller._manager_provenance_is_visible_for("selected")

    manager._metadata_node_uid = "selected"
    assert controller._manager_provenance_is_visible_for("selected")

    manager.metadata_derivation_list.hide()
    assert not controller._manager_provenance_is_visible_for("selected")
    controller._finish_cleanup()


def test_visible_converted_cut_still_requires_open_action(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)
    monkeypatch.setattr(controller, "_converted_cut_uid", lambda: "converted-cut")
    monkeypatch.setattr(controller, "_uid_tool_visible", lambda _uid: True)

    assert not controller._converted_cut_was_opened()
    controller._converted_cut_open_requested = True
    assert controller._converted_cut_was_opened()
    controller._finish_cleanup()


def test_figure_composer_target_uses_the_tutorial_manager(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)

    unrelated = type("FigureComposerTool", (QtWidgets.QWidget,), {})()
    qtbot.addWidget(unrelated)
    unrelated.show()

    monkeypatch.setattr(manager, "_figure_uids", list, raising=False)
    monkeypatch.setattr(manager, "_selected_figure_uids", list, raising=False)
    assert controller._figure_composer() is None

    owned = QtWidgets.QWidget()
    qtbot.addWidget(owned)

    class _FigureNode:
        tool_window = owned

    monkeypatch.setattr(manager, "_figure_uids", lambda: ["figure"], raising=False)
    monkeypatch.setattr(manager, "_selected_figure_uids", list, raising=False)
    monkeypatch.setattr(
        manager, "_child_node", lambda _uid: _FigureNode(), raising=False
    )
    assert controller._figure_composer() is owned

    monkeypatch.setattr(
        manager,
        "_figure_uids",
        lambda: ["figure", "another-figure"],
        raising=False,
    )
    monkeypatch.setattr(manager, "_selected_figure_uids", list, raising=False)
    assert controller._figure_composer() is owned

    controller._figure_composer_uid = None
    monkeypatch.setattr(
        manager, "_selected_figure_uids", lambda: ["figure"], raising=False
    )
    assert controller._figure_composer() is owned

    controller._finish_cleanup()


def test_show_figure_composer_raises_and_activates(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)
    composer = QtWidgets.QWidget()
    qtbot.addWidget(composer)
    calls: list[str] = []
    monkeypatch.setattr(controller, "_figure_composer", lambda: composer)
    monkeypatch.setattr(composer, "show", lambda: calls.append("show"))
    monkeypatch.setattr(composer, "raise_", lambda: calls.append("raise"))
    monkeypatch.setattr(composer, "activateWindow", lambda: calls.append("activate"))

    controller._show_figure_composer()

    assert calls == ["show", "raise", "activate"]
    controller._finish_cleanup()


def test_ui_text_resolver_uses_visible_control_text(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    action = QtGui.QAction("&Open && Inspect", manager)
    action.setObjectName("testAction")
    label = QtWidgets.QLabel("Loader", manager)
    label.setObjectName("testLabel")
    combo = QtWidgets.QComboBox(manager)
    combo.setObjectName("testCombo")
    combo.addItem("Selected loader")
    icon_button = QtWidgets.QToolButton(manager)
    icon_button.setObjectName("testIconButton")
    icon_button.setText("Hidden text")
    icon_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
    tabs = QtWidgets.QTabWidget(manager)
    page = QtWidgets.QWidget(tabs)
    page.setObjectName("testTabPage")
    tabs.addTab(page, "&Preview")
    manager.show()
    manager.activateWindow()
    QtWidgets.QApplication.setActiveWindow(manager)

    controller, _loader_context = _controller(monkeypatch, manager)
    assert controller._resolve_ui_text("testAction") == "Open & Inspect"
    assert controller._resolve_ui_text("testLabel") == "Loader"
    assert controller._resolve_ui_text("testTabPage") == "Preview"
    assert controller._resolve_ui_text("testCombo") is None
    assert controller._resolve_ui_text("testIconButton") is None
    assert controller._resolve_ui_text("missingObject") is None

    duplicate = QtWidgets.QLabel("Duplicate", manager)
    duplicate.setObjectName("testLabel")
    assert controller._resolve_ui_text("testLabel") is None
    controller._finish_cleanup()


def test_ui_text_resolver_uses_current_target_window(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    manager.show()
    controller, _loader_context = _controller(monkeypatch, manager)

    target_window = QtWidgets.QMainWindow()
    other_window = QtWidgets.QMainWindow()
    qtbot.addWidget(target_window)
    qtbot.addWidget(other_window)
    target_action = QtGui.QAction("Reveal target", target_window)
    other_action = QtGui.QAction("Reveal other", other_window)
    for action in (target_action, other_action):
        action.setObjectName("duplicateRevealAction")
    target_window.addAction(target_action)
    other_window.addAction(other_action)
    target_window.show()
    other_window.show()
    other_window.activateWindow()
    QtWidgets.QApplication.setActiveWindow(other_window)

    controller._steps = (
        tutorial_framework.TourStep(
            "scoped-label",
            "Scoped label",
            "Select [[ui:duplicateRevealAction]].",
            target=target_window,
        ),
    )
    controller._index = 0

    assert controller._resolve_ui_text("duplicateRevealAction") == "Reveal target"
    controller._finish_cleanup()


def test_cursor_drag_predicate_requires_primary_modifier(qapp) -> None:
    position = QtCore.QPointF(4.0, 5.0)
    plain = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        position,
        position,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    controlled = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        position,
        position,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    control_key = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Control,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    other_key = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_A,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    controlled_and_alt = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        position,
        position,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier
        | QtCore.Qt.KeyboardModifier.AltModifier,
    )
    alt_key = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Alt,
        QtCore.Qt.KeyboardModifier.AltModifier,
    )

    assert not tutorial._cursor_drag_event_predicate(None, plain)
    assert tutorial._cursor_drag_event_predicate(None, controlled)
    assert tutorial._cursor_drag_event_predicate(None, control_key)
    assert not tutorial._cursor_drag_event_predicate(None, other_key)
    assert not tutorial._multi_cursor_drag_event_predicate(None, plain)
    assert not tutorial._multi_cursor_drag_event_predicate(None, controlled)
    assert tutorial._multi_cursor_drag_event_predicate(None, controlled_and_alt)
    assert tutorial._multi_cursor_drag_event_predicate(None, control_key)
    assert tutorial._multi_cursor_drag_event_predicate(None, alt_key)
    assert not tutorial._multi_cursor_drag_event_predicate(None, other_key)


def test_cursor_drag_suspends_tutorial_refresh_until_release(
    monkeypatch, qtbot
) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)
    refreshes: list[str | None] = []
    monkeypatch.setattr(
        tutorial.TourController,
        "notify_state_changed",
        lambda self: refreshes.append(
            None if self.current_step is None else self.current_step.id
        ),
    )
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "mouseButtons",
        staticmethod(lambda: QtCore.Qt.MouseButton.LeftButton),
    )

    ids = [step.id for step in controller.steps]
    for step_id in tutorial._CURSOR_DRAG_STEP_IDS:
        controller._index = ids.index(step_id)
        controller.notify_state_changed()
    assert refreshes == []

    controller._index = ids.index("transpose-alpha-beta")
    controller.notify_state_changed()
    assert refreshes == ["transpose-alpha-beta"]

    monkeypatch.setattr(
        QtWidgets.QApplication,
        "mouseButtons",
        staticmethod(lambda: QtCore.Qt.MouseButton.NoButton),
    )
    controller._index = ids.index("ctrl-drag-cursor")
    controller.notify_state_changed()
    assert refreshes == ["transpose-alpha-beta", "ctrl-drag-cursor"]

    controller._finish_cleanup()


def test_second_cursor_drag_stops_polling_until_release(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)
    refreshes: list[str | None] = []
    monkeypatch.setattr(
        controller,
        "notify_state_changed",
        lambda: refreshes.append(
            None if controller.current_step is None else controller.current_step.id
        ),
    )
    controller._index = [step.id for step in controller.steps].index(
        "move-second-cursor"
    )
    controller._running = True
    controller._state_timer.start()
    position = QtCore.QPointF(4.0, 5.0)

    press = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        position,
        position,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    assert not controller.eventFilter(manager, press)
    assert not controller._state_timer.isActive()

    release = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonRelease,
        position,
        position,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    assert not controller.eventFilter(manager, release)
    assert controller._state_timer.isActive()
    qtbot.waitUntil(lambda: refreshes == ["move-second-cursor"])

    assert not controller.eventFilter(manager, release)
    controller._index = [step.id for step in controller.steps].index(
        "transpose-alpha-beta"
    )
    controller._step_activation += 1
    qtbot.wait(50)
    assert refreshes == ["move-second-cursor"]

    controller._state_timer.stop()
    controller._running = False
    controller._finish_cleanup()


def test_coordinate_dialog_steps_use_widget_signals_instead_of_polling(
    monkeypatch, qtbot
) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)
    dialog = QtWidgets.QDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    monkeypatch.setattr(
        controller,
        "_coordinate_dialog",
        lambda: dialog if dialog.isVisible() else None,
    )
    polls: list[str | None] = []
    monkeypatch.setattr(
        controller,
        "notify_state_changed",
        lambda: polls.append(
            None if controller.current_step is None else controller.current_step.id
        ),
    )

    ids = [step.id for step in controller.steps]
    for step_id in tutorial._COORDINATE_DIALOG_STEP_IDS:
        controller._index = ids.index(step_id)
        controller._poll_state_changed()
    assert polls == []

    dialog.hide()
    controller._poll_state_changed()
    assert polls == [controller.current_step.id]

    controller._finish_cleanup()


def test_unavailable_step_stops_manager_polling(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    manager.show()
    controller, _loader_context = _controller(monkeypatch, manager)
    controller._steps = (
        tutorial_framework.TourStep(
            "missing-target",
            "Missing target",
            "Body",
            target=lambda: None,
            timeout_ms=0,
        ),
    )
    controller._index = 0
    controller._running = True
    controller._state_timer.start()

    with pytest.raises(tutorial_framework.TutorialStepUnavailableError):
        controller._refresh()

    assert not controller._state_timer.isActive()
    controller._poll_state_changed()
    controller.close()
    controller._finish_cleanup()


def test_imagetool_menu_target_uses_window_for_native_menu(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)
    tool = QtWidgets.QMainWindow()
    qtbot.addWidget(tool)
    tool.show()
    menu_bar = tool.menuBar()
    menu_bar.hide()
    monkeypatch.setattr(controller, "_map_tool", lambda: tool)

    monkeypatch.setattr(tutorial, "_NATIVE_MENU_BAR", True)
    assert controller._map_menus_target() is tool
    assert (
        tutorial_framework.target_geometry(controller._map_menus_target()) is not None
    )

    monkeypatch.setattr(tutorial, "_NATIVE_MENU_BAR", False)
    assert controller._map_menus_target() is menu_bar
    assert tutorial_framework.target_geometry(controller._map_menus_target()) is None

    controller._finish_cleanup()


def test_cleanup_during_generation_is_asynchronous(monkeypatch, qtbot) -> None:
    entered = threading.Event()

    def generate(directory, *, is_cancelled, on_file_published):
        del directory, on_file_published
        entered.set()
        while not is_cancelled():
            threading.Event().wait(0.005)
        raise tutorial.TutorialDataGenerationCancelled

    monkeypatch.setattr(tutorial, "generate_tutorial_data_files", generate)
    manager = _Manager()
    qtbot.addWidget(manager)
    manager.show()
    controller, loader_context = _controller(monkeypatch, manager)

    controller.start()
    assert entered.wait(1.0)
    controller._begin_cleanup()
    assert controller._cleaning
    assert not manager.isVisible()
    qtbot.waitUntil(lambda: controller.is_cleaned)

    assert loader_context.entered == 1
    assert loader_context.exited == 1
    assert manager._workspace_controller.clean_calls == 1
    assert manager.close_calls == 1
    assert not controller.directory.exists()


def test_cleanup_waits_for_manager_file_handlers(monkeypatch, qtbot) -> None:
    monkeypatch.setattr(tutorial, "generate_tutorial_data_files", _immediate_generation)
    manager = _Manager()
    qtbot.addWidget(manager)
    manager.show()
    controller, loader_context = _controller(monkeypatch, manager)

    controller.start()
    qtbot.waitUntil(lambda: controller._data_ready)
    handler = object()
    manager._file_handlers.add(handler)
    controller._begin_cleanup()
    qtbot.wait(75)
    assert not controller.is_cleaned

    manager._file_handlers.remove(handler)
    qtbot.waitUntil(lambda: controller.is_cleaned)
    assert loader_context.exited == 1
    assert manager._workspace_controller.clean_calls == 1
    assert manager.close_calls == 1


def test_cleanup_quits_application_after_manager_closes(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    manager.show()
    controller, _loader_context = _controller(monkeypatch, manager)
    quit_calls: list[None] = []
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "quit",
        lambda _application: quit_calls.append(None),
    )

    controller._finish_cleanup()

    assert manager.close_calls == 1
    assert quit_calls == [None]


def test_cleanup_does_not_quit_if_manager_refuses_close(monkeypatch, qtbot) -> None:
    manager = _Manager()
    manager.reject_close = True
    qtbot.addWidget(manager)
    manager.show()
    controller, _loader_context = _controller(monkeypatch, manager)
    quit_calls: list[None] = []
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "quit",
        lambda _application: quit_calls.append(None),
    )

    controller._finish_cleanup()

    assert manager.close_calls == 1
    assert quit_calls == []


def test_welcome_card_has_no_stale_wait_hint(monkeypatch, qtbot) -> None:
    monkeypatch.setattr(tutorial, "generate_tutorial_data_files", _immediate_generation)
    manager = _Manager()
    qtbot.addWidget(manager)
    manager.show()
    controller, _loader_context = _controller(monkeypatch, manager)

    controller.start()
    qtbot.waitUntil(lambda: controller._data_ready)
    card = controller._card
    assert card is not None
    assert card.continue_button.isEnabled()
    assert not card.hint.isVisible()
    assert any(
        span.kind == "ui" and span.text == card.continue_button.text()
        for span in card.body._text.spans
    )

    controller._begin_cleanup()
    qtbot.waitUntil(lambda: controller.is_cleaned)


def test_generation_failure_raises_diagnostic_error(monkeypatch, qtbot) -> None:
    manager = _Manager()
    qtbot.addWidget(manager)
    controller, _loader_context = _controller(monkeypatch, manager)
    error = ValueError("invalid tutorial data")

    with pytest.raises(RuntimeError, match="generate the tutorial data") as exc_info:
        controller._generation_failed(error)

    assert exc_info.value.__cause__ is error
    assert controller._generation_error == str(error)
    controller._finish_cleanup()


@pytest.mark.parametrize(
    ("raise_error", "error_type"),
    [
        (
            lambda controller, step: controller._raise_unavailable_step(
                step, target_missing=True, text_missing=False
            ),
            tutorial_framework.TutorialStepUnavailableError,
        ),
        (
            lambda controller, step: controller._raise_debug_error(
                step, "test failure"
            ),
            tutorial_framework.TutorialDebugActionError,
        ),
    ],
)
def test_fatal_errors_use_tutorial_cleanup(
    monkeypatch, qtbot, raise_error, error_type
) -> None:
    monkeypatch.setattr(tutorial, "generate_tutorial_data_files", _immediate_generation)
    manager = _Manager()
    qtbot.addWidget(manager)
    manager.show()
    controller, loader_context = _controller(monkeypatch, manager)

    controller.start()
    step = controller.current_step
    assert step is not None
    with pytest.raises(error_type):
        raise_error(controller, step)

    assert controller._fatal_error is not None
    assert controller._cleaning
    assert not controller._state_timer.isActive()
    qtbot.waitUntil(lambda: controller.is_cleaned)

    assert not controller.is_running
    assert loader_context.exited == 1
    assert manager._workspace_controller.clean_calls == 1
    assert manager.close_calls == 1
    assert not controller.directory.exists()


def test_cancelled_exit_keeps_tutorial_running(monkeypatch, qtbot) -> None:
    monkeypatch.setattr(tutorial, "generate_tutorial_data_files", _immediate_generation)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    manager = _Manager()
    qtbot.addWidget(manager)
    manager.show()
    controller, _loader_context = _controller(monkeypatch, manager)

    controller.start()
    qtbot.waitUntil(lambda: controller._data_ready)
    controller.request_exit()
    assert controller.is_running
    assert not controller._cleaning

    controller._begin_cleanup()
    qtbot.waitUntil(lambda: controller.is_cleaned)


def test_start_retains_only_tutorial_manager(monkeypatch, qtbot) -> None:
    monkeypatch.setattr(tutorial, "generate_tutorial_data_files", _immediate_generation)
    loader_contexts: list[_LoaderContext] = []

    @contextlib.contextmanager
    def loader_context():
        context = _LoaderContext()
        loader_contexts.append(context)
        context.__enter__()
        try:
            yield object()
        finally:
            context.__exit__(None, None, None)

    monkeypatch.setattr(tutorial, "tutorial_loader_registration", loader_context)
    original = _Manager()
    tutorial_manager = _Manager()
    qtbot.addWidget(original)
    qtbot.addWidget(tutorial_manager)
    tutorial_manager.show()

    controller = tutorial.start_tutorial(tutorial_manager)
    assert tutorial_manager._tutorial_controller is controller
    assert not hasattr(original, "_tutorial_controller")
    assert original._workspace_controller.clean_calls == 0
    assert original.close_calls == 0

    controller._begin_cleanup()
    qtbot.waitUntil(lambda: controller.is_cleaned)
    assert loader_contexts[0].exited == 1


def test_tutorial_real_workflow(
    monkeypatch, qtbot, accept_dialog, manager_context
) -> None:
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda _application: None)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    def step_id() -> str | None:
        step = controller.current_step
        return None if step is None else step.id

    def wait_step(expected: str, timeout: int = 20_000) -> None:
        qtbot.waitUntil(lambda: step_id() == expected, timeout=timeout)
        step = controller.current_step
        assert step is not None
        qtbot.waitUntil(
            lambda: not controller._step_text(step)[-1],
            timeout=timeout,
        )

    def continue_to(expected: str) -> None:
        while step_id() != expected:
            step = controller.current_step
            if step is None or step.mode != "information":
                raise AssertionError(f"Expected an information step, got {step_id()!r}")
            qtbot.waitUntil(
                lambda current_step=step: not controller._step_text(current_step)[-1],
                timeout=20_000,
            )
            qtbot.waitUntil(
                lambda current_step=step: (
                    controller._resolve_target(current_step)[0] is not None
                ),
                timeout=20_000,
            )
            geometry, missing = controller._resolve_target(step)
            assert not missing, step.id
            assert geometry is not None, step.id
            previous = step.id
            controller.continue_step()
            qtbot.waitUntil(lambda previous_step=previous: step_id() != previous_step)

    def complete_action(expected: str | None = None, timeout: int = 20_000) -> None:
        step = controller.current_step
        if expected is not None and step_id() == expected:
            wait_step(expected, timeout=timeout)
            return
        if expected is None and step is not None and step.id.startswith("switch-to-"):
            return
        if step is None or step.mode != "action":
            raise AssertionError(f"Expected an action step, got {step_id()!r}")
        qtbot.waitUntil(
            lambda: not controller._step_text(step)[-1],
            timeout=timeout,
        )
        qtbot.waitUntil(lambda: controller._is_complete(step), timeout=timeout)
        controller.notify_state_changed()
        previous = step.id
        if not step.auto_advance:
            card = controller._card
            assert card is not None
            qtbot.waitUntil(card.continue_button.isEnabled, timeout=timeout)
            controller.continue_step()
        qtbot.waitUntil(lambda: step_id() != previous, timeout=timeout)
        if expected is not None:
            wait_step(expected, timeout=timeout)

    def complete_switch(
        switch_step: str,
        destination: QtWidgets.QWidget,
        expected: str,
        timeout: int = 20_000,
    ) -> None:
        qtbot.waitUntil(
            lambda: step_id() in {switch_step, expected},
            timeout=timeout,
        )
        if step_id() == switch_step:
            destination.activateWindow()
            QtWidgets.QApplication.setActiveWindow(destination)
            qtbot.waitUntil(destination.isActiveWindow, timeout=timeout)
            controller.notify_state_changed()
        wait_step(expected, timeout=timeout)

    def select_tab(tabs: QtWidgets.QTabWidget, index: int, expected: str) -> None:
        tab_bar = tabs.tabBar()
        qtbot.mouseClick(
            tab_bar,
            QtCore.Qt.MouseButton.LeftButton,
            pos=tab_bar.tabRect(index).center(),
        )
        complete_action(expected)

    def assert_tab_step(
        window: QtWidgets.QWidget,
        tabs: QtWidgets.QTabWidget,
        index: int,
        *,
        unobscured: bool = False,
    ) -> None:
        controller.notify_state_changed()
        card = controller._card
        assert card is not None
        qtbot.waitUntil(lambda: card.isVisible() and card.parentWidget() is window)
        assert any(
            span.kind == "ui" and span.text == tabs.tabText(index)
            for span in card.body._text.spans
        )
        tab_bar = tabs.tabBar()
        tab_center = tab_bar.mapTo(window, tab_bar.tabRect(index).center())
        overlay = next(
            overlay
            for window_ref, overlay in controller._overlays
            if window_ref() is window
        )
        assert overlay._spotlight is not None
        assert overlay._spotlight.contains(tab_center)
        if unobscured:
            assert not card.geometry().intersects(overlay._spotlight)

    with manager_context() as manager:
        manager.show()
        controller = tutorial._TutorialController(manager)
        manager._tutorial_controller = controller
        controller.start()
        qtbot.waitUntil(lambda: controller._data_ready, timeout=20_000)

        continue_to("open-data-explorer")
        assert controller._explorer_window() is None
        manager.explorer_action.trigger()
        complete_action("data-explorer-introduction")
        explorer_window = controller._explorer_window()
        assert explorer_window is not None
        continue_to("select-map")
        explorer = manager.explorer.current_explorer
        assert explorer is not None
        map_index = explorer._model_index_for_path(controller.data_files.map)
        selection_model = explorer._tree_view.selectionModel()
        assert selection_model is not None
        selection_model.select(
            map_index,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        qtbot.waitUntil(
            lambda: controller._explorer_file_selected(controller.data_files.map),
            timeout=20_000,
        )
        controller.notify_state_changed()
        complete_action("enable-map-preview")
        explorer._preview_check.setChecked(True)
        qtbot.waitUntil(
            lambda: controller._explorer_file_ready(controller.data_files.map),
            timeout=20_000,
        )
        controller.notify_state_changed()
        complete_action("explorer-preview")
        continue_to("open-map-in-manager")

        open_in_manager_button = controller._explorer_open_button()
        assert isinstance(open_in_manager_button, QtWidgets.QPushButton)
        open_in_manager_button.click()
        qtbot.waitUntil(
            lambda: isinstance(
                controller._map_tool(), erlab.interactive.imagetool.ImageTool
            ),
            timeout=60_000,
        )
        tool = controller._map_tool()
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        wait_step("imagetool-plots")
        continue_to("ctrl-drag-cursor")

        axis = tool.slicer_area.data.get_axis_num("alpha")
        tool.slicer_area.set_index(axis, 10)
        controller.notify_state_changed()
        complete_action("imagetool-cursor-controls")
        continue_to("transpose-alpha-beta")
        tool.cursor_controls.btn_transpose[0].click()
        assert controller._main_dims() == ("beta", "alpha")
        complete_action("imagetool-color-controls")
        continue_to("set-energy-bin")

        energy_axis = tool.slicer_area.data.get_axis_num("eV")
        tool.binning_controls.spins[energy_axis].setValue(5)
        complete_action("add-second-cursor")
        tool.cursor_controls.btn_add.click()
        complete_action("move-second-cursor")
        tool.slicer_area.set_index(axis, 15, cursor=1)
        complete_action("move-all-cursors")
        tool.slicer_area.set_index(axis, 20, cursor=0)
        tool.slicer_area.set_index(axis, 20, cursor=1)
        complete_action("set-second-cursor-bin")
        tool.binning_controls.spins[energy_axis].setValue(3)
        complete_action("select-first-cursor")
        tool.slicer_area.set_current_cursor(0)
        complete_action("imagetool-menus")
        assert tool.slicer_area.n_cursors == 2
        continue_to("inspect-kinetic-energy")
        energy_profile_view = controller._energy_profile_view()
        assert isinstance(energy_profile_view, QtWidgets.QGraphicsView)
        assert not controller.eventFilter(
            energy_profile_view.viewport(),
            QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress),
        )
        main_image_view = tool.slicer_area.main_image.getViewWidget()
        assert isinstance(main_image_view, QtWidgets.QGraphicsView)
        assert controller.eventFilter(
            main_image_view.viewport(),
            QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress),
        )
        assert controller.eventFilter(
            energy_profile_view.viewport(),
            QtCore.QEvent(QtCore.QEvent.Type.Wheel),
        )
        assert not controller.eventFilter(
            energy_profile_view,
            QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_Control,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            ),
        )
        continue_to("open-coordinate-editor")

        def set_energy_coordinate(dialog) -> None:
            controller.notify_state_changed()
            wait_step("select-energy-coordinate")
            dialog._coord_combo.setCurrentText("eV")
            complete_action("select-scale-offset")
            card = controller._card
            assert card is not None
            assert card.isVisible()
            qtbot.waitUntil(lambda: card.parentWidget() is dialog)
            tab_bar = controller._coordinate_edit_mode_tab_bar()
            assert tab_bar is not None
            assert dialog.coord_widget.edit_mode_tabs.currentIndex() == 0
            qtbot.mouseClick(
                tab_bar,
                QtCore.Qt.MouseButton.LeftButton,
                pos=tab_bar.tabRect(1).center(),
            )
            complete_action("set-energy-offset")
            dialog.coord_widget.offset_spin.setValue(-45.5)
            complete_action("apply-energy-correction")
            assert dialog.launch_mode_combo.currentText() == "Replace Current"
            button_box = controller._coordinate_button_box()
            assert button_box is dialog.buttonBox
            assert not controller.eventFilter(
                button_box,
                QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress),
            )
            cancel_button = button_box.button(
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            assert cancel_button is not None
            assert controller.eventFilter(
                cancel_button,
                QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress),
            )

        def click_coordinate_ok(_dialog) -> None:
            button = controller._coordinate_apply_button()
            assert button is not None
            qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)

        accept_dialog(
            tool.mnb._assign_coords,
            pre_call=set_energy_coordinate,
            accept_call=click_coordinate_ok,
            timeout=20.0,
        )
        wait_step("inspect-binding-energy")
        continue_to("select-c6-guideline")

        image = tool.slicer_area.main_image
        image.set_guidelines(3)
        complete_action("set-normal-emission-and-azimuth")
        tool.slicer_area.set_value(tool.slicer_area.data.get_axis_num("alpha"), 2.0)
        tool.slicer_area.set_value(tool.slicer_area.data.get_axis_num("beta"), -1.5)
        controller.notify_state_changed()
        assert step_id() == "set-normal-emission-and-azimuth"
        assert not controller._is_complete(controller.current_step)
        image._guidelines_items[0].setAngle(86.0)
        controller.notify_state_changed()
        complete_action("open-ktool")

        tool.slicer_area.open_in_ktool()
        ktool = controller._ktool()
        assert ktool is not None
        complete_switch("switch-to-ktool", ktool, "ktool-previews")
        continue_to("select-ktool-visualization")
        assert ktool.tabWidget.currentIndex() == 0
        assert_tab_step(ktool, ktool.tabWidget, 1)
        select_tab(ktool.tabWidget, 1, "ktool-brillouin-zone")
        assert ktool.a_spin.value() == pytest.approx(6.97)
        assert ktool.b_spin.value() == pytest.approx(6.97)
        assert ktool.alpha_spin.value() == pytest.approx(90.0)
        assert ktool.beta_spin.value() == pytest.approx(90.0)
        assert ktool.gamma_spin.value() == pytest.approx(120.0)
        assert ktool.centering_combo.currentText() == "P"
        assert ktool.rot_spin.value() == pytest.approx(30.0)
        assert not ktool.bz_group.isChecked()
        ktool.bz_group.setChecked(True)
        complete_action("ktool-energy-preview")
        assert ktool.center_spin.isEnabled()
        assert ktool.width_spin.isEnabled()
        ktool.center_spin.setValue(-0.2)
        ktool.width_spin.setValue(5)
        complete_action("select-ktool-parameters")
        assert_tab_step(ktool, ktool.tabWidget, 0)
        select_tab(ktool.tabWidget, 0, "open-converted-map")

        assert ktool._normal_emission_spins["alpha"].value() == 2.0
        assert ktool._normal_emission_spins["beta"].value() == -1.5
        assert ktool._offset_spins["delta"].value() == -4.0

        open_button = ktool.findChild(
            QtWidgets.QPushButton, "ktoolOpenInImageToolButton"
        )
        assert open_button is not None
        qtbot.waitUntil(open_button.isEnabled, timeout=20_000)
        open_button.click()
        qtbot.waitUntil(
            lambda: isinstance(
                controller._converted_map_tool(),
                erlab.interactive.imagetool.ImageTool,
            ),
            timeout=60_000,
        )
        converted_map = controller._converted_map_tool()
        assert isinstance(converted_map, erlab.interactive.imagetool.ImageTool)
        complete_switch(
            "switch-to-converted-map",
            converted_map,
            "reveal-converted-map",
            timeout=60_000,
        )

        manager.inspector_tabs.setCurrentWidget(manager.metadata_details_page)
        converted_map.reveal_in_manager_act.trigger()
        complete_switch(
            "switch-to-manager-provenance",
            manager,
            "manager-overview",
        )
        converted_uid = controller._converted_map_uid()
        assert converted_uid is not None
        ktool_uid = manager._tool_graph.nodes[converted_uid].parent_uid
        assert ktool_uid is not None
        assert manager._tool_graph.nodes[ktool_uid].parent_uid == (
            controller._find_node_uid("example_map")
        )
        continue_to("select-manager-provenance")
        assert manager.inspector_tabs.currentWidget() is manager.metadata_details_page
        provenance_index = manager.inspector_tabs.indexOf(
            manager.metadata_provenance_page
        )
        assert_tab_step(
            manager,
            manager.inspector_tabs,
            provenance_index,
            unobscured=True,
        )
        select_tab(manager.inspector_tabs, provenance_index, "provenance-overview")
        continue_to("switch-to-explorer-cut")
        manager.explorer_action.trigger()
        complete_switch("switch-to-explorer-cut", explorer_window, "select-cut")

        cut_index = explorer._model_index_for_path(controller.data_files.cut)
        selection_model.select(
            cut_index,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        qtbot.waitUntil(
            lambda: controller._explorer_file_ready(controller.data_files.cut),
            timeout=20_000,
        )
        controller.notify_state_changed()
        complete_action("open-cut-in-manager")
        open_in_manager_button = controller._explorer_open_button()
        assert isinstance(open_in_manager_button, QtWidgets.QPushButton)
        open_in_manager_button.click()
        complete_action("switch-to-manager-operations")
        raw_cut = controller._tool_for_uid(controller._raw_cut_uid())
        assert isinstance(raw_cut, erlab.interactive.imagetool.ImageTool)
        raw_cut.reveal_in_manager_act.trigger()
        continue_to("select-converted-map")

        converted_index = manager.tree_view._model._row_index(
            controller._converted_map_uid()
        )
        assert converted_index.isValid()
        qtbot.waitUntil(
            lambda: not manager.tree_view.visualRect(converted_index).isEmpty()
        )
        qtbot.mouseClick(
            manager.tree_view.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=manager.tree_view.visualRect(converted_index).center(),
        )
        complete_action("expand-input-history")

        input_item = controller._reusable_input_item()
        assert input_item is not None
        assert input_item.childCount() > 0
        assert not input_item.isExpanded()
        input_item.setExpanded(True)
        complete_action("select-reusable-operations")

        from erlab.interactive.imagetool import _kspace_conversion
        from erlab.interactive.imagetool._provenance._operations import (
            AffineCoordOperation,
        )
        from erlab.interactive.imagetool.manager._widgets import (
            _METADATA_DERIVATION_ROW_ROLE,
        )

        converted_uid = controller._converted_map_uid()
        assert converted_uid is not None
        node = manager._tool_graph.nodes[converted_uid]
        assert node.displayed_provenance_spec is not None

        def select_reusable_operation_items() -> None:
            operation_items = []
            group_items = []
            operation_list = manager.metadata_derivation_list
            for row_index in range(operation_list.conceptual_count()):
                item = operation_list.conceptual_item(row_index)
                if item is None:
                    continue
                row = item.data(0, _METADATA_DERIVATION_ROW_ROLE)
                ref = getattr(row, "replay_ref", None)
                spec = manager._provenance_edit_controller._display_spec_for_row(
                    node,
                    row,
                )
                operation = (
                    None
                    if ref is None or spec is None
                    else spec._operation_for_ref(ref)
                )
                if operation is None:
                    continue
                if (
                    isinstance(operation, AffineCoordOperation)
                    and operation.coord_name == "eV"
                ):
                    operation_items.append(item)
                if (
                    operation.group is not None
                    and operation.group.kind
                    == _kspace_conversion.KSPACE_CONVERSION_GROUP_KIND
                ):
                    group_items.append(item)

            assert len(operation_items) == 1
            assert len(group_items) == 2
            assert operation_items[0].parent() is input_item
            assert input_item.isExpanded()
            operation_list.clearSelection()
            for item in (*operation_items, *group_items):
                item.setSelected(True)
            operation_list.setCurrentItem(
                group_items[0],
                0,
                QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
            )

        select_reusable_operation_items()
        controller.notify_state_changed()
        complete_action("copy-reusable-operations")
        manager._build_metadata_derivation_menu()
        assert manager._metadata_copy_selected_action.isEnabled()
        manager._metadata_copy_selected_action.trigger()
        complete_action("select-raw-cut")
        QtWidgets.QApplication.processEvents()

        controller._select_uid(controller._raw_cut_uid())
        controller.notify_state_changed()
        complete_action("select-raw-cut-provenance")
        manager.inspector_tabs.setCurrentWidget(manager.metadata_details_page)
        controller.notify_state_changed()
        card = controller._card
        assert card is not None
        assert not card.continue_button.isEnabled()
        select_tab(
            manager.inspector_tabs,
            provenance_index,
            "paste-reusable-operations",
        )

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText("unrelated clipboard text")
        qtbot.waitUntil(card.recovery_button.isVisible)
        assert card.recovery_button.text() == "Copy Again"
        assert card.recovery_button.focusPolicy() == QtCore.Qt.FocusPolicy.StrongFocus
        assert not card.continue_button.isEnabled()
        assert "clipboard no longer contains" in card.hint.text()
        card.recovery_button.click()
        wait_step("select-converted-map")

        controller._select_uid(controller._converted_map_uid())
        controller.notify_state_changed()
        complete_action("expand-input-history")
        input_item = controller._reusable_input_item()
        assert input_item is not None
        input_item.setExpanded(True)
        complete_action("select-reusable-operations")
        select_reusable_operation_items()
        controller.notify_state_changed()
        complete_action("copy-reusable-operations")
        manager._build_metadata_derivation_menu()
        assert manager._metadata_copy_selected_action.isEnabled()
        manager._metadata_copy_selected_action.trigger()
        complete_action("select-raw-cut")
        controller._select_uid(controller._raw_cut_uid())
        controller.notify_state_changed()
        complete_action("select-raw-cut-provenance")
        select_tab(
            manager.inspector_tabs,
            provenance_index,
            "paste-reusable-operations",
        )

        manager._build_metadata_derivation_menu(include_row_actions=False)
        assert manager._metadata_paste_steps_action.isEnabled()
        manager._metadata_paste_steps_action.trigger()
        complete_action("validate-converted-cut", timeout=60_000)
        continue_to("open-converted-cut")

        assert controller.current_step is not None
        assert controller.current_step.id == "open-converted-cut"
        converted_cut_index = manager.tree_view._model._row_index(
            controller._converted_cut_uid()
        )
        assert converted_cut_index.isValid()
        qtbot.waitUntil(
            lambda: not manager.tree_view.visualRect(converted_cut_index).isEmpty()
        )
        qtbot.mouseDClick(
            manager.tree_view.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=manager.tree_view.visualRect(converted_cut_index).center(),
        )
        converted_cut = controller._tool_for_uid(controller._converted_cut_uid())
        assert isinstance(converted_cut, erlab.interactive.imagetool.ImageTool)
        complete_switch(
            "switch-to-converted-cut",
            converted_cut,
            "new-figure",
        )
        figure_menu = converted_cut.slicer_area.main_image.getMenu()
        assert figure_menu is not None
        image_view = converted_cut.slicer_area.main_image.getViewWidget()
        assert isinstance(image_view, QtWidgets.QGraphicsView)
        assert not controller.eventFilter(
            image_view.viewport(), QtCore.QEvent(QtCore.QEvent.Type.ContextMenu)
        )
        figure_menu.popup(
            image_view.viewport().mapToGlobal(image_view.viewport().rect().center())
        )
        qtbot.waitUntil(figure_menu.isVisible)
        controller.notify_state_changed()
        assert controller.current_step is not None
        assert controller.current_step.id == "new-figure"
        figure_menu.close()
        qtbot.waitUntil(lambda: not figure_menu.isVisible())
        controller.notify_state_changed()
        step = controller.current_step
        assert step is not None
        assert step.id == "new-figure"
        assert not controller._resolve_target(step)[1]
        assert not controller._step_text(step)[-1]
        figure_menu.popup(
            image_view.viewport().mapToGlobal(image_view.viewport().rect().center())
        )
        qtbot.waitUntil(figure_menu.isVisible)
        new_figure_action = controller._new_figure_action()
        assert new_figure_action is not None
        new_figure_action.trigger()
        figure_menu.close()
        qtbot.waitUntil(
            lambda: controller._figure_composer() is not None,
            timeout=60_000,
        )
        composer = controller._figure_composer()
        assert composer is not None
        qtbot.waitUntil(
            lambda: step_id() == "figure-composer-output",
            timeout=60_000,
        )
        assert composer.editor_tabs.currentWidget() is composer.operation_panel
        continue_to("select-figure-composer-sources")
        assert composer.editor_tabs.currentWidget() is composer.operation_panel
        assert_tab_step(composer, composer.editor_tabs, 0)
        select_tab(composer.editor_tabs, 0, "figure-composer-sources")
        continue_to("select-figure-composer-layout")
        assert_tab_step(composer, composer.editor_tabs, 1)
        select_tab(composer.editor_tabs, 1, "figure-composer-layout")
        continue_to("select-figure-composer-recipe")
        assert_tab_step(composer, composer.editor_tabs, 2)
        select_tab(composer.editor_tabs, 2, "figure-composer-recipe")
        continue_to("reveal-figure-in-manager")
        composer.reveal_in_manager_action.trigger()
        complete_action("manager-figures")

        figure_pane = manager._figure_collection.pane
        assert figure_pane is not None
        assert manager.left_tabs.currentWidget() is figure_pane
        figure_uid = controller._figure_composer_uid
        assert figure_uid is not None
        selected_figure_uids = {
            manager._figure_collection.uid_from_item(item)
            for item in figure_pane.list_widget.selectedItems()
        }
        assert selected_figure_uids == {figure_uid}
        continue_to("tutorial-complete")

        controller.continue_step()
        qtbot.waitUntil(lambda: controller.is_cleaned, timeout=20_000)


def test_tutorial_debug_skip_completes_real_workflow(
    monkeypatch, qtbot, manager_context
) -> None:
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda _application: None)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    with manager_context() as manager:
        manager.show()
        controller = tutorial._TutorialController(manager, debug=True)
        manager._tutorial_controller = controller
        controller.start()
        card = controller._card
        assert card is not None
        assert card.skip_button.isVisible()

        visited: list[str] = []
        clipboard_wiped = False
        clipboard_recovered = False
        driver = QtCore.QTimer()
        driver.setInterval(25)

        def skip_current_step() -> None:
            nonlocal clipboard_recovered, clipboard_wiped
            step = controller.current_step
            if step is not None and step.id == "paste-reusable-operations":
                if not clipboard_wiped:
                    QtWidgets.QApplication.clipboard().setText(
                        "unrelated clipboard text"
                    )
                    clipboard_wiped = True
                    return
                if (
                    card.recovery_button.isVisible()
                    and card.recovery_button.isEnabled()
                ):
                    clipboard_recovered = True
                    card.recovery_button.click()
                    return
            if (
                step is None
                or (visited and visited[-1] == step.id)
                or not card.skip_button.isEnabled()
            ):
                return
            visited.append(step.id)
            card.skip_button.click()

        driver.timeout.connect(skip_current_step)
        driver.start()
        try:
            while not (controller.is_cleaned or controller._fatal_error is not None):
                activation = controller._step_activation
                qtbot.waitUntil(
                    lambda activation=activation: (
                        controller.is_cleaned
                        or controller._fatal_error is not None
                        or controller._step_activation != activation
                    ),
                    timeout=120_000,
                )
        finally:
            driver.stop()

        figure_state = {
            uid: {
                "raw_tool_type": type(node._tool_window).__name__,
                "raw_tool_valid": erlab.interactive.utils.qt_is_valid(
                    node._tool_window
                ),
                "tool_available": node.tool_window is not None,
            }
            for uid in manager._figure_uids()
            for node in (manager._child_node(uid),)
        }
        assert controller._fatal_error is None, {
            "cached_uid": controller._figure_composer_uid,
            "figure_uids": manager._figure_uids(),
            "selected_figure_uids": manager._selected_figure_uids(),
            "figure_state": figure_state,
            "manager_is_global": (
                erlab.interactive.imagetool.manager._manager_instance is manager
            ),
            "closing_document": manager._workspace_state.closing_document,
        }
        assert clipboard_wiped
        assert clipboard_recovered
        assert "open-coordinate-editor" in visited
        assert "new-figure" in visited
