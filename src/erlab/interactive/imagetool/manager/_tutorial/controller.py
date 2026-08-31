"""Controller for the ImageTool Manager tutorial."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import sys
import tempfile
import threading
import typing

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.explorer._base_explorer import DataExplorerTabState
from erlab.interactive.imagetool.manager._tutorial.data import (
    TutorialDataFiles,
    TutorialDataGenerationCancelled,
    generate_tutorial_data_files,
    tutorial_loader_registration,
)
from erlab.interactive.imagetool.manager._tutorial.framework import (
    ActionTarget,
    CompositeTarget,
    GraphicsItemTarget,
    ModelIndexTarget,
    RectTarget,
    TourController,
    TourStep,
    target_geometry,
)

if typing.TYPE_CHECKING:
    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager


_PROVENANCE_STEPS_CLIPBOARD_MIME = "application/x-erlab-imagetool-provenance-steps+json"
_NATIVE_MENU_BAR = sys.platform == "darwin"
_CURSOR_MODIFIER_LABEL = "Command" if sys.platform == "darwin" else "Ctrl"
_CURSOR_DRAG_STEP_IDS = frozenset(
    {"ctrl-drag-cursor", "move-second-cursor", "set-normal-emission-and-azimuth"}
)
_COORDINATE_DIALOG_STEP_IDS = frozenset(
    {
        "select-energy-coordinate",
        "select-scale-offset",
        "set-energy-offset",
        "apply-energy-correction",
    }
)


def _menu_instruction(path: str) -> str:
    if _NATIVE_MENU_BAR:
        return f"In the macOS menu bar, select {path}."
    return f"Select {path}."


class _GenerationSignals(QtCore.QObject):
    published = QtCore.Signal(object)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(object)


def _mouse_has_cursor_modifier(event: QtCore.QEvent) -> bool:
    # Qt maps ControlModifier to the Command key on macOS.
    return isinstance(event, QtGui.QMouseEvent) and bool(
        event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
    )


def _cursor_drag_event_predicate(
    watched: QtCore.QObject | None, event: QtCore.QEvent
) -> bool:
    """Permit only platform cursor-modifier key and mouse input."""
    if isinstance(event, QtGui.QKeyEvent):
        return _cursor_modifier_key_event_predicate(watched, event)
    return _mouse_has_cursor_modifier(event)


def _cursor_modifier_key_event_predicate(
    watched: QtCore.QObject | None, event: QtCore.QEvent
) -> bool:
    """Permit only the platform cursor-modifier key."""
    del watched
    return isinstance(event, QtGui.QKeyEvent) and (
        event.key() == QtCore.Qt.Key.Key_Control
    )


class _TutorialController(TourController):
    """Run the disposable, forward-only Manager tutorial."""

    def __init__(self, manager: ImageToolManager, *, debug: bool = False) -> None:
        self._manager = manager
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="erlab-manager-tutorial-"
        )
        self.directory = pathlib.Path(self._temporary_directory.name)
        self.data_files = TutorialDataFiles(
            map=self.directory / "tutorial_map.h5",
            cut=self.directory / "tutorial_cut.h5",
        )
        self._cancel_generation = threading.Event()
        self._generation_signals = _GenerationSignals()
        self._generation_thread: threading.Thread | None = None
        self._loader_context = tutorial_loader_registration()
        self._loader_context_entered = False
        self._data_ready = False
        self._generation_error: str | None = None
        self._published_paths: set[pathlib.Path] = set()
        self._cleaning = False
        self._cleaned = False
        self._allow_manager_close = False
        self._explorer_configured = False
        self._cursor_start: tuple[int, ...] | None = None
        self._node_uids_before_conversion: set[str] = set()
        self._expected_reveal_uid: str | None = None
        self._revealed_uid: str | None = None
        self._reveal_action: QtGui.QAction | None = None
        self._operations_copied = False
        self._debug_active_window: QtWidgets.QWidget | None = None
        self._figure_composer_uid: str | None = None

        super().__init__(
            self._build_steps(),
            manager,
            parent=manager,
            text_resolver=self._resolve_ui_text,
            debug=debug,
        )

        self._state_timer = QtCore.QTimer(self)
        self._state_timer.setInterval(100)
        self._state_timer.timeout.connect(self._poll_state_changed)
        self.finished.connect(self._finish_requested)
        self._generation_signals.published.connect(self._file_published)
        self._generation_signals.completed.connect(self._generation_completed)
        self._generation_signals.failed.connect(self._generation_failed)
        self._manager._metadata_copy_selected_action.triggered.connect(
            self._operations_copy_triggered
        )

    @property
    def steps(self) -> tuple[TourStep, ...]:
        """Get the declared tutorial steps for tests and diagnostics."""
        return self._steps

    @property
    def is_cleaned(self) -> bool:
        return self._cleaned

    def _resolve_ui_text(self, object_name: str) -> str | None:
        application = typing.cast(
            "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
        )
        if application is None:
            return None

        target_text = self._current_target_ui_text(object_name, application)
        if target_text is not None:
            return target_text

        active_window = application.activeWindow()
        if active_window is not None:
            active_matches = self._named_objects(active_window, object_name)
            if len(active_matches) == 1:
                return self._object_ui_text(active_matches[0], application)
            if active_matches:
                return None

        matches: list[QtCore.QObject] = []
        seen: set[int] = set()
        for window in application.topLevelWidgets():
            for obj in self._named_objects(window, object_name):
                if id(obj) not in seen:
                    seen.add(id(obj))
                    matches.append(obj)
        if len(matches) != 1:
            return None
        return self._object_ui_text(matches[0], application)

    def _current_target_ui_text(
        self,
        object_name: str,
        application: QtWidgets.QApplication,
    ) -> str | None:
        step = self.current_step
        if step is None or step.target is None:
            return None
        try:
            target = step.target() if callable(step.target) else step.target
        except (RuntimeError, TypeError):
            return None
        candidates: tuple[QtCore.QObject | None, ...]
        if isinstance(target, ActionTarget):
            candidates = (target.action, target.menu)
        elif isinstance(target, QtCore.QObject):
            candidates = (target,)
        else:
            candidates = ()
        matches = [
            candidate
            for candidate in candidates
            if candidate is not None and candidate.objectName() == object_name
        ]
        if len(matches) == 1:
            return self._object_ui_text(matches[0], application)
        allowed_matches: list[QtCore.QObject] = []
        for resolver in step.allowed_objects:
            try:
                candidate = resolver() if callable(resolver) else resolver
            except (RuntimeError, TypeError):
                continue
            if (
                isinstance(candidate, QtCore.QObject)
                and candidate.objectName() == object_name
            ):
                allowed_matches.append(candidate)
        if len(allowed_matches) == 1:
            return self._object_ui_text(allowed_matches[0], application)
        try:
            geometry = target_geometry(target)
        except (RuntimeError, TypeError):
            geometry = None
        if geometry is None or geometry.window is None:
            return None
        matches = self._named_objects(geometry.window, object_name)
        if len(matches) != 1:
            return None
        return self._object_ui_text(matches[0], application)

    @staticmethod
    def _named_objects(
        root: QtWidgets.QWidget, object_name: str
    ) -> list[QtCore.QObject]:
        candidates: list[QtCore.QObject] = [root]
        candidates.extend(root.findChildren(QtCore.QObject, object_name))
        widgets: list[QtWidgets.QWidget] = [root]
        widgets.extend(root.findChildren(QtWidgets.QWidget))
        for widget in widgets:
            candidates.extend(widget.actions())
            if isinstance(widget, QtWidgets.QMenu):
                menu_action = widget.menuAction()
                if menu_action is not None:  # pragma: no branch
                    candidates.append(menu_action)

        matches: list[QtCore.QObject] = []
        seen: set[int] = set()
        for candidate in candidates:
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            with contextlib.suppress(RuntimeError):
                if candidate.objectName() == object_name:
                    matches.append(candidate)
        return matches

    @classmethod
    def _object_ui_text(
        cls,
        obj: QtCore.QObject,
        application: QtWidgets.QApplication,
    ) -> str | None:
        text: str | None = None
        if isinstance(obj, QtGui.QAction):
            text = obj.text()
        elif isinstance(obj, QtWidgets.QMenu):
            text = obj.title()
        elif isinstance(obj, QtWidgets.QLabel):
            text = obj.text()
        elif isinstance(obj, QtWidgets.QToolButton):
            if obj.toolButtonStyle() != QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly:
                text = obj.text()
        elif isinstance(obj, QtWidgets.QAbstractButton):
            text = obj.text()
        elif isinstance(obj, QtWidgets.QGroupBox):
            text = obj.title()
        elif isinstance(obj, QtWidgets.QWidget):
            for widget in application.allWidgets():
                if not isinstance(widget, QtWidgets.QTabWidget):
                    continue
                index = widget.indexOf(obj)
                if index >= 0:
                    text = widget.tabText(index)
                    break
        if not text:
            return None
        return cls._strip_mnemonics(text).strip() or None

    @staticmethod
    def _strip_mnemonics(text: str) -> str:
        result: list[str] = []
        index = 0
        while index < len(text):
            if text[index] != "&":
                result.append(text[index])
                index += 1
                continue
            if index + 1 < len(text) and text[index + 1] == "&":
                result.append("&")
                index += 2
            else:
                index += 1
        return "".join(result)

    def start(self) -> None:
        if self.is_running:
            return
        if not self._loader_context_entered:
            self._loader_context.__enter__()
            self._loader_context_entered = True
        self._start_generation()
        super().start()
        self._state_timer.start()

    def request_exit(self) -> None:
        """Request confirmation and keep the tutorial open after cancellation."""
        if not self.is_running or self._cleaning:
            return
        answer = QtWidgets.QMessageBox.question(
            self._manager,
            "Exit Tutorial?",
            "Do you want to exit the tutorial?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._begin_cleanup()

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if (
            not self._allow_manager_close
            and not self._cleaning
            and watched is self._manager
            and event is not None
            and event.type() == QtCore.QEvent.Type.Close
        ):
            if isinstance(event, QtGui.QCloseEvent):
                event.ignore()
            self.request_exit()
            return True
        filtered = super().eventFilter(watched, event)
        step = self.current_step
        if (
            not filtered
            and step is not None
            and step.id in _CURSOR_DRAG_STEP_IDS
            and isinstance(event, QtGui.QMouseEvent)
            and not isinstance(watched, QtGui.QWindow)
        ):
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                self._state_timer.stop()
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                self._state_timer.start()
                activation = self._step_activation
                erlab.interactive.utils.single_shot(
                    self,
                    25,
                    lambda: self._notify_state_changed_if_active(activation),
                )
        return filtered

    def _notify_state_changed_if_active(self, activation: int) -> None:
        if self.is_running and activation == self._step_activation:
            self.notify_state_changed()

    def notify_state_changed(self) -> None:
        step = self.current_step
        if (
            step is not None
            and step.id in _CURSOR_DRAG_STEP_IDS
            and QtWidgets.QApplication.mouseButtons() & QtCore.Qt.MouseButton.LeftButton
        ):
            return
        super().notify_state_changed()

    def _poll_state_changed(self) -> None:
        step = self.current_step
        if (
            step is not None
            and step.id in _COORDINATE_DIALOG_STEP_IDS
            and self._coordinate_dialog() is not None
        ):
            return
        self.notify_state_changed()

    def _handle_fatal_error(self, error: RuntimeError) -> None:
        self._state_timer.stop()
        super()._handle_fatal_error(error)

    def _start_generation(self) -> None:
        if self._generation_thread is not None and self._generation_thread.is_alive():
            return
        self._generation_error = None
        self._cancel_generation.clear()

        def generate() -> None:
            try:
                files = generate_tutorial_data_files(
                    self.directory,
                    is_cancelled=self._cancel_generation.is_set,
                    on_file_published=self._generation_signals.published.emit,
                )
            except TutorialDataGenerationCancelled:
                return
            except Exception as exc:
                self._generation_signals.failed.emit(exc)
            else:
                self._generation_signals.completed.emit(files)

        self._generation_thread = threading.Thread(
            target=generate,
            name="ERLabTutorialData",
            daemon=True,
        )
        self._generation_thread.start()

    @QtCore.Slot(object)
    def _file_published(self, path: pathlib.Path) -> None:
        if not self._cleaning:
            self._published_paths.add(pathlib.Path(path))

    @QtCore.Slot(object)
    def _generation_completed(self, files: TutorialDataFiles) -> None:
        if self._cleaning:
            return
        self.data_files = files
        self._data_ready = True
        self.update_current(
            body=(
                "The tutorial is ready. Select [[ui:tutorialContinueButton]] to "
                "begin the tutorial."
            )
        )
        self.notify_state_changed()

    @QtCore.Slot(object)
    def _generation_failed(self, error: Exception) -> None:
        if self._cleaning:
            return
        self._generation_error = str(error)
        raise RuntimeError("Could not generate the tutorial data.") from error

    def _finish_requested(self) -> None:
        self._begin_cleanup()

    def _begin_cleanup(self) -> None:
        if self._cleaning or self._cleaned:
            return
        self._cleaning = True
        self._state_timer.stop()
        self._cancel_generation.set()
        application = typing.cast(
            "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
        )
        if application is not None:
            for window in application.topLevelWidgets():
                with contextlib.suppress(RuntimeError):
                    window.hide()
        self.close()
        QtCore.QTimer.singleShot(0, self._poll_cleanup)

    def _poll_cleanup(self) -> None:
        if self._cleaned:
            return
        thread = self._generation_thread
        if thread is not None and thread.is_alive():
            QtCore.QTimer.singleShot(25, self._poll_cleanup)
            return
        if getattr(self._manager, "_file_handlers", None):
            QtCore.QTimer.singleShot(25, self._poll_cleanup)
            return
        self._finish_cleanup()

    def _finish_cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        if self._loader_context_entered:
            self._loader_context.__exit__(None, None, None)
            self._loader_context_entered = False
        with contextlib.suppress(RuntimeError, TypeError):
            self._manager._metadata_copy_selected_action.triggered.disconnect(
                self._operations_copy_triggered
            )
        if self._reveal_action is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                self._reveal_action.triggered.disconnect(self._manager_reveal_triggered)
            self._reveal_action = None
        self._temporary_directory.cleanup()
        with contextlib.suppress(RuntimeError, AttributeError):
            self._manager._workspace_controller._mark_workspace_clean()
        self._allow_manager_close = True
        try:
            manager_closed = self._manager.close()
        except RuntimeError:
            manager_closed = False
        if manager_closed:
            application = typing.cast(
                "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
            )
            if application is not None:
                application.quit()

    def _build_steps(self) -> list[TourStep]:
        information: frozenset[str] = frozenset()
        actions: frozenset[str] = frozenset(
            {"mouse", "key", "shortcut", "context_menu"}
        )

        def switch_window(
            step_id: str,
            window_name: str,
            source: typing.Callable[[], QtWidgets.QWidget | None],
            destination: typing.Callable[[], QtWidgets.QWidget | None],
            reveal: typing.Callable[[], None] | None = None,
        ) -> TourStep:
            return TourStep(
                step_id,
                f"Switch to {window_name}",
                f"The next step uses the {window_name} window. Switch to it now.",
                mode="action",
                target=lambda: self._active_window_or(source()),
                allowed_inputs=actions,
                allowed_objects=(destination,),
                completion=lambda: self._window_is_active(destination()),
                reveal=reveal,
                debug_action=lambda: self._debug_activate_window(destination()),
                card_position="center",
            )

        steps = [
            TourStep(
                "welcome",
                "ImageTool Manager tutorial",
                "The tutorial is loading. This may take a moment.",
                target=lambda: self._manager,
                ready=lambda: self._data_ready,
                continue_label="Start",
                allowed_inputs=information,
                reveal=self._show_manager,
                card_position="center",
            ),
            TourStep(
                "manager-introduction",
                "ImageTool Manager",
                "This is the ImageTool Manager window. It organizes loaded data, "
                "ImageTool windows, processed data, and workspaces.",
                target=lambda: self._manager,
                allowed_inputs=information,
                reveal=self._show_manager,
                card_position="center",
            ),
            TourStep(
                "open-data-explorer",
                "Open Data Explorer",
                _menu_instruction("[[menu:manager_file_menu|manager_explorer_action]]"),
                mode="action",
                target=self._explorer_action_target,
                allowed_inputs=actions,
                allowed_objects=(lambda: self._manager.explorer_action,),
                completion=self._configure_opened_explorer,
                reveal=self._reveal_explorer_action,
                card_position="center" if _NATIVE_MENU_BAR else "target",
            ),
            TourStep(
                "data-explorer-introduction",
                "Data Explorer",
                "This is the Data Explorer window. It browses data files and uses "
                "the selected loader to read them.",
                target=self._explorer_window,
                allowed_inputs=information,
                reveal=self._reveal_explorer,
                card_position="center",
            ),
            TourStep(
                "select-map",
                "Select the map",
                "Select tutorial_map.h5 in the file list.",
                mode="action",
                target=lambda: self._widget("dataExplorerFileTree"),
                allowed_inputs=actions,
                allowed_objects=(lambda: self._widget("dataExplorerFileTree"),),
                completion=lambda: self._explorer_file_selected(self.data_files.map),
                reveal=self._reveal_explorer,
                auto_advance=False,
            ),
            TourStep(
                "enable-map-preview",
                "Enable data preview",
                "Select the [[ui:dataExplorerPreviewCheck]] checkbox.",
                mode="action",
                target=lambda: self._widget("dataExplorerPreviewCheck"),
                allowed_inputs=actions,
                allowed_objects=(lambda: self._widget("dataExplorerPreviewCheck"),),
                completion=lambda: self._explorer_file_ready(self.data_files.map),
                reveal=self._reveal_explorer,
                auto_advance=False,
            ),
            TourStep(
                "explorer-preview",
                "Preview and metadata",
                "The preview shows the selected map, and the metadata panel lists its "
                "coordinates and attributes.",
                target=lambda: self._widget("dataExplorerPreviewSplitter"),
                allowed_inputs=actions | frozenset({"wheel"}),
                allowed_objects=(lambda: self._widget("dataExplorerPreviewSplitter"),),
                reveal=self._reveal_explorer,
            ),
            TourStep(
                "explorer-folder",
                "Folder",
                "The highlighted field shows the current folder.",
                target=lambda: self._widget("dataExplorerDirectoryField"),
                allowed_inputs=information,
                reveal=self._reveal_explorer,
            ),
            TourStep(
                "explorer-loader",
                "Loader",
                "[[ui:dataExplorerLoaderLabel]] determines how Data Explorer reads "
                "the files. This folder uses the ERLab tutorial data loader.",
                target=lambda: self._widget("dataExplorerLoaderSelector"),
                allowed_inputs=information,
                reveal=self._reveal_explorer,
            ),
            TourStep(
                "open-map-in-manager",
                "Open the data",
                "Select the highlighted button to open the selected data.",
                mode="action",
                target=self._explorer_open_button,
                allowed_inputs=actions,
                completion=lambda: self._named_tool_visible("example_map"),
                reveal=self._reveal_explorer,
            ),
            TourStep(
                "imagetool-plots",
                "ImageTool",
                "This is an ImageTool window. The main image shows two dimensions. "
                "The profiles show slices along the other dimensions at the active "
                "cursor.",
                target=lambda: self._map_tool_widget("slicer_area"),
                allowed_inputs=information,
                reveal=self._show_map_tool,
            ),
            TourStep(
                "ctrl-drag-cursor",
                "Move the cursor",
                f"Hold {_CURSOR_MODIFIER_LABEL} and drag in the main image. The "
                "profiles and coordinate values follow the cursor.",
                mode="action",
                target=self._main_image_target,
                allowed_inputs=frozenset(),
                event_predicate=_cursor_drag_event_predicate,
                completion=self._cursor_moved,
                reveal=self._capture_cursor_start,
                hint=f"Keep {_CURSOR_MODIFIER_LABEL} pressed while you drag.",
                auto_advance=False,
            ),
            TourStep(
                "imagetool-cursor-controls",
                "Cursor controls",
                "These controls show the coordinates of the cursor that you moved. "
                "They also select the active cursor.",
                target=lambda: self._map_tool_widget("cursor_controls"),
                allowed_inputs=information,
                reveal=self._show_map_tool,
            ),
            TourStep(
                "transpose-alpha-beta",
                "Alpha and beta axes",
                "Select the highlighted transpose button. The arrows on the "
                "transpose buttons show the direction in which the image panels "
                "will swap. The main image changes from alpha-beta to beta-alpha.",
                mode="action",
                target=lambda: self._widget("itoolTransposeAxis0Button"),
                allowed_inputs=actions,
                completion=lambda: self._main_dims() == ("beta", "alpha"),
                reveal=self._show_map_tool,
                auto_advance=False,
            ),
            TourStep(
                "imagetool-display-controls",
                "Display and binning controls",
                "Color controls change the display. Binning controls average adjacent "
                "points in the displayed slices.",
                target=self._map_display_controls_target,
                allowed_inputs=information,
                reveal=self._show_map_tool,
            ),
            TourStep(
                "set-energy-bin",
                "Bin the energy axis",
                "Set the eV bin width to 5 points. The first binning operation "
                "after installation can take a moment while ImageTool prepares and "
                "caches its optimized binning code.",
                mode="action",
                target=self._energy_bin_spin,
                allowed_inputs=actions,
                completion=lambda: self._energy_bin() == 5,
                reveal=self._show_map_tool,
                auto_advance=False,
            ),
            TourStep(
                "add-second-cursor",
                "Add a cursor",
                "Select the highlighted button. A second cursor gives an independent "
                "set of profiles and bin widths.",
                mode="action",
                target=lambda: self._widget("itoolAddCursorButton"),
                allowed_inputs=actions,
                completion=lambda: self._cursor_count() == 2,
                reveal=self._show_map_tool,
                auto_advance=False,
            ),
            TourStep(
                "move-second-cursor",
                "Second cursor position",
                f"Cursor 1 is active. Hold {_CURSOR_MODIFIER_LABEL} and drag it to "
                "a different point. The profiles for cursor 0 stay at their original "
                "coordinates.",
                mode="action",
                target=self._main_image_target,
                allowed_inputs=actions,
                completion=self._second_cursor_moved,
                event_predicate=_cursor_drag_event_predicate,
                reveal=self._show_map_tool,
                hint=f"Keep {_CURSOR_MODIFIER_LABEL} pressed while you drag.",
                auto_advance=False,
            ),
            TourStep(
                "set-second-cursor-bin",
                "Second cursor binning",
                "Set the eV bin width to 3 points. The new width applies only to "
                "cursor 1.",
                mode="action",
                target=self._energy_bin_spin,
                allowed_inputs=actions,
                completion=lambda: (
                    self._current_cursor() == 1 and self._energy_bin(cursor=1) == 3
                ),
                reveal=self._show_map_tool,
                auto_advance=False,
            ),
            TourStep(
                "select-first-cursor",
                "Cursor comparison",
                "Select cursor 0. Its coordinates, profiles, and eV bin width become "
                "active again. Cursor 1 stays in the ImageTool.",
                mode="action",
                target=lambda: self._widget("itoolCursorSelector"),
                allowed_inputs=actions,
                completion=lambda: (
                    self._cursor_count() == 2
                    and self._current_cursor() == 0
                    and self._energy_bin(cursor=0) == 5
                    and self._energy_bin(cursor=1) == 3
                ),
                reveal=self._show_map_tool,
                auto_advance=False,
            ),
            TourStep(
                "imagetool-menus",
                "ImageTool menus",
                (
                    "The menu bar at the top of the screen contains the ImageTool "
                    "menus. These menus provide data operations, view controls, "
                    "interactive tools, and Manager actions."
                    if _NATIVE_MENU_BAR
                    else "The ImageTool menus contain data operations, view controls, "
                    "interactive tools, and Manager actions."
                ),
                target=self._map_menus_target,
                allowed_inputs=information,
                reveal=self._show_map_tool,
                card_position="center" if _NATIVE_MENU_BAR else "target",
            ),
            TourStep(
                "inspect-kinetic-energy",
                "Kinetic energy",
                "Inspect the eV axis. Its values are kinetic energies. The Fermi "
                "level for this simulated measurement is at 45.5 eV.",
                target=self._energy_profile_target,
                allowed_inputs=frozenset({"mouse"}),
                allowed_objects=(self._energy_profile_view,),
                event_predicate=_cursor_modifier_key_event_predicate,
                reveal=self._show_map_tool,
            ),
            TourStep(
                "open-coordinate-editor",
                "Energy coordinate correction",
                _menu_instruction("[[menu:itoolEditMenu|itoolEditCoordinatesAction]]"),
                mode="action",
                target=self._coordinate_action_target,
                allowed_inputs=actions,
                allowed_objects=(self._coordinate_action,),
                completion=lambda: self._coordinate_dialog() is not None,
                reveal=self._reveal_coordinate_action,
            ),
            TourStep(
                "select-energy-coordinate",
                "Energy coordinate",
                "Select eV in the Coordinate list.",
                mode="action",
                target=self._coordinate_selector,
                allowed_inputs=actions,
                allowed_objects=(
                    self._coordinate_selector,
                    self._coordinate_selector_popup,
                ),
                subscriptions=(self._coordinate_selector_signal,),
                completion=self._energy_coordinate_is_selected,
                auto_advance=False,
            ),
            TourStep(
                "select-scale-offset",
                "Scale and offset",
                "Select the [[ui:coordinateEditorScaleOffsetPage]] tab.",
                mode="action",
                target=self._coordinate_edit_mode_tab_bar,
                allowed_inputs=actions,
                allowed_objects=(self._coordinate_edit_mode_tab_bar,),
                subscriptions=(self._coordinate_edit_mode_signal,),
                completion=self._scale_offset_is_selected,
                auto_advance=False,
            ),
            TourStep(
                "set-energy-offset",
                "Energy offset",
                "Set [[ui:coordinateEditorOffsetLabel]] to -45.5. This places the "
                "Fermi level at 0 eV.",
                mode="action",
                target=self._coordinate_offset_spin,
                allowed_inputs=actions,
                subscriptions=(self._coordinate_offset_signal,),
                completion=self._energy_offset_is_set,
                auto_advance=False,
            ),
            TourStep(
                "apply-energy-correction",
                "Coordinate correction",
                "Select [[ui:coordinateEditorApplyButton]] to replace the current "
                "data with the corrected energy coordinate.",
                mode="action",
                target=self._coordinate_apply_button,
                allowed_inputs=actions,
                allowed_objects=(self._coordinate_apply_button,),
                event_predicate=self._coordinate_apply_event_predicate,
                subscriptions=(self._coordinate_dialog_finished_signal,),
                completion=self._energy_is_corrected,
            ),
            TourStep(
                "inspect-binding-energy",
                "Binding energy",
                "Inspect the eV axis again. Its values are now binding energies, and "
                "the Fermi level is at 0 eV.",
                target=self._energy_profile_target,
                allowed_inputs=information,
                reveal=self._show_map_tool,
            ),
            TourStep(
                "select-c6-guideline",
                "Add a rotation guideline",
                _menu_instruction(
                    "[[menu:itoolViewMenu|itoolRotationGuidelinesMenu|"
                    "itoolGuidelineC6Action]]"
                ),
                mode="action",
                target=self._c6_action_target,
                allowed_inputs=actions,
                allowed_objects=(self._c6_action,),
                completion=lambda: self._guideline_count() == 6,
                reveal=self._reveal_c6_action,
                auto_advance=False,
            ),
            TourStep(
                "set-normal-emission-and-azimuth",
                "Normal emission and sample azimuth",
                f"Hold {_CURSOR_MODIFIER_LABEL} and drag the cursor to the expected "
                "normal emission, which is the Brillouin-zone center in momentum "
                "space. For this data, set alpha = 2.0° and beta = −1.5°. Drag the "
                "rotation guideline to −4.0°. You can adjust them in either order.",
                mode="action",
                target=self._main_image_target,
                allowed_inputs=actions,
                event_predicate=_cursor_drag_event_predicate,
                completion=lambda: (
                    self._normal_emission_is_set()
                    and bool(np.isclose(self._guideline_angle(), -4.0, atol=0.5))
                ),
                reveal=self._show_map_tool,
                hint=(
                    f"Keep {_CURSOR_MODIFIER_LABEL} pressed while you drag the cursor."
                ),
                auto_advance=False,
            ),
            TourStep(
                "open-ktool",
                "Open ktool",
                _menu_instruction("[[menu:itoolViewMenu|itoolOpenKtoolAction]]"),
                mode="action",
                target=self._ktool_action_target,
                allowed_inputs=actions,
                allowed_objects=(self._ktool_action,),
                completion=lambda: self._ktool() is not None,
                reveal=self._reveal_ktool_action,
            ),
            switch_window(
                "switch-to-ktool",
                "ktool",
                self._map_tool,
                self._ktool,
            ),
            TourStep(
                "ktool-previews",
                "ktool",
                "This is ktool. It converts ARPES data from angular coordinates to "
                "momentum coordinates.",
                target=lambda: self._ktool_child("graphics_layout"),
                allowed_inputs=information,
                reveal=self._show_ktool,
            ),
            TourStep(
                "ktool-geometry",
                "Normal emission and offsets",
                "Normal emission identifies the angular coordinates at which the "
                "photoelectron emission direction is normal to the sample surface. "
                "Here, you can see that they are automatically filled in from the "
                "cursor position. The offsets show the azimuthal offset and the "
                "parameters used by the momentum conversion functions, calculated "
                "from the normal emission angles.",
                target=lambda: self._ktool_composite(
                    "normal_emission_group", "offsets_group"
                ),
                allowed_inputs=information,
                reveal=self._show_ktool,
            ),
            TourStep(
                "ktool-grid",
                "Bounds and resolution",
                "Bounds set the output coordinate range. Resolution sets the output "
                "grid spacing or point count.",
                target=lambda: self._ktool_composite(
                    "bounds_supergroup", "resolution_supergroup"
                ),
                allowed_inputs=information,
                reveal=self._show_ktool,
            ),
            TourStep(
                "select-ktool-visualization",
                "Visualization",
                "Select the [[ui:ktoolVisualizationPage]] tab.",
                mode="action",
                target=self._ktool_visualization_tab_target,
                allowed_inputs=actions,
                allowed_objects=(self._ktool_tab_bar,),
                subscriptions=(self._ktool_tab_signal,),
                completion=self._ktool_visualization_is_selected,
                reveal=self._show_ktool,
                card_position="center",
                auto_advance=False,
            ),
            TourStep(
                "ktool-brillouin-zone",
                "Brillouin-zone controls",
                "These controls add reciprocal-space guides to the preview. They do "
                "not change the converted data.",
                target=lambda: self._ktool_child("bz_group"),
                allowed_inputs=information,
                reveal=self._show_ktool,
            ),
            TourStep(
                "select-ktool-parameters",
                "Parameters",
                "Select the [[ui:ktoolParametersPage]] tab.",
                mode="action",
                target=self._ktool_parameters_tab_target,
                allowed_inputs=actions,
                allowed_objects=(self._ktool_tab_bar,),
                subscriptions=(self._ktool_tab_signal,),
                completion=self._ktool_parameters_is_selected,
                reveal=self._show_ktool,
                card_position="center",
                auto_advance=False,
            ),
            TourStep(
                "open-converted-map",
                "Converted data",
                "Select [[ui:ktoolOpenInImageToolButton]].",
                mode="action",
                target=lambda: self._ktool_button("ktoolOpenInImageToolButton"),
                allowed_inputs=actions,
                completion=lambda: self._converted_map_uid() is not None,
                reveal=self._prepare_map_conversion,
            ),
            switch_window(
                "switch-to-converted-map",
                "converted ImageTool",
                self._ktool,
                self._converted_map_tool,
            ),
            TourStep(
                "reveal-converted-map",
                "Reveal the converted data in ImageTool Manager",
                _menu_instruction(
                    "[[menu:itoolWindowMenu|itool_reveal_in_manager_action]]"
                ),
                mode="action",
                target=lambda: self._reveal_in_manager_action_target(
                    self._converted_map_uid()
                ),
                allowed_inputs=actions,
                allowed_objects=(
                    lambda: self._reveal_in_manager_action(self._converted_map_uid()),
                ),
                completion=lambda: self._uid_was_revealed(self._converted_map_uid()),
                reveal=lambda: self._prepare_reveal_in_manager(
                    self._converted_map_uid()
                ),
            ),
            switch_window(
                "switch-to-manager-provenance",
                "ImageTool Manager",
                self._converted_map_tool,
                lambda: self._manager,
                self._show_manager,
            ),
            TourStep(
                "manager-overview",
                "Workspace tree",
                "The ktool row is a child of example_map because you opened ktool "
                "from that map. The converted data is a child of ktool. The "
                "[[ui:managerDetailsPage]] and [[ui:managerProvenancePage]] tabs "
                "describe the selected row.",
                target=lambda: CompositeTarget(
                    self._manager.tree_view, self._manager.inspector_tabs
                ),
                allowed_inputs=information,
                reveal=self._show_converted_map_provenance,
            ),
            TourStep(
                "select-manager-provenance",
                "Provenance",
                "Select the [[ui:managerProvenancePage]] tab.",
                mode="action",
                target=self._manager_provenance_tab_target,
                allowed_inputs=actions,
                allowed_objects=(self._manager_inspector_tab_bar,),
                subscriptions=(self._manager_inspector_tab_signal,),
                completion=self._manager_provenance_is_selected,
                reveal=self._show_converted_map_provenance,
                auto_advance=False,
            ),
            TourStep(
                "provenance-overview",
                "Provenance",
                "The selected converted data appears below ktool because ktool "
                "created it from example_map. The [[ui:managerProvenancePage]] tab "
                "shows Scale/Offset Coordinate, Set normal emission, and Convert "
                "to momentum.",
                target=lambda: self._manager.metadata_derivation_list,
                allowed_inputs=information,
                reveal=self._show_converted_map_provenance,
            ),
            TourStep(
                "switch-to-explorer-cut",
                "Return to Data Explorer",
                _menu_instruction("[[menu:manager_file_menu|manager_explorer_action]]"),
                mode="action",
                target=self._explorer_action_target,
                allowed_inputs=actions,
                allowed_objects=(lambda: self._manager.explorer_action,),
                completion=lambda: self._window_is_active(self._explorer_window()),
                reveal=self._reveal_explorer_action,
                card_position="center",
            ),
            TourStep(
                "select-cut",
                "Select the dispersion cut",
                "Select tutorial_cut.h5. The Preview pane updates to show the cut.",
                mode="action",
                target=lambda: self._widget("dataExplorerFileTree"),
                allowed_inputs=actions,
                allowed_objects=(lambda: self._widget("dataExplorerFileTree"),),
                completion=lambda: self._explorer_file_ready(self.data_files.cut),
                reveal=self._reveal_explorer,
                auto_advance=False,
            ),
            TourStep(
                "open-cut-in-manager",
                "Open the dispersion cut",
                "Select the highlighted button.",
                mode="action",
                target=self._explorer_open_button,
                allowed_inputs=actions,
                completion=lambda: self._raw_cut_uid() is not None,
                reveal=self._reveal_explorer,
            ),
            TourStep(
                "switch-to-manager-operations",
                "Reveal the dispersion cut in ImageTool Manager",
                _menu_instruction(
                    "[[menu:itoolWindowMenu|itool_reveal_in_manager_action]]"
                ),
                mode="action",
                target=lambda: self._reveal_in_manager_action_target(
                    self._raw_cut_uid()
                ),
                allowed_inputs=actions,
                allowed_objects=(
                    lambda: self._reveal_in_manager_action(self._raw_cut_uid()),
                ),
                completion=lambda: self._uid_was_revealed(self._raw_cut_uid()),
                reveal=lambda: self._prepare_reveal_in_manager(self._raw_cut_uid()),
                card_position="center",
            ),
            TourStep(
                "select-converted-map",
                "Momentum-converted data",
                "Select the momentum-converted data below ktool in the workspace tree.",
                mode="action",
                target=lambda: self._manager_row_target(self._converted_map_uid()),
                allowed_inputs=actions,
                completion=lambda: self._node_is_selected(self._converted_map_uid()),
                reveal=self._show_converted_map_in_tree,
                auto_advance=False,
            ),
            TourStep(
                "expand-input-history",
                "Input history",
                "Select the arrow beside the highlighted input row to show the "
                "steps that were applied before the momentum conversion.",
                mode="action",
                target=self._reusable_input_target,
                allowed_inputs=actions,
                allowed_objects=(lambda: self._manager.metadata_derivation_list,),
                completion=self._reusable_input_expanded,
                reveal=self._show_converted_map_provenance,
                auto_advance=False,
            ),
            TourStep(
                "select-reusable-operations",
                "Steps to reuse",
                "In the [[ui:managerProvenancePage]] tab, select these three rows: "
                "Scale/Offset Coordinate, Set normal emission, and Convert "
                "to momentum.",
                mode="action",
                hint=f"Hold {_CURSOR_MODIFIER_LABEL} while you select all three rows.",
                target=lambda: self._manager.metadata_derivation_list,
                allowed_inputs=actions,
                completion=self._reusable_operations_selected,
                reveal=self._show_converted_map_provenance,
                auto_advance=False,
            ),
            TourStep(
                "copy-reusable-operations",
                "Copy the selected steps",
                "Right-click on the selected rows, then select "
                "[[ui:manager_copy_selected_code_action]].",
                mode="action",
                target=lambda: self._manager.metadata_derivation_list,
                allowed_inputs=actions,
                event_predicate=self._context_menu_event_predicate,
                completion=self._provenance_steps_on_clipboard,
                reveal=self._prepare_operations_copy,
                auto_advance=False,
            ),
            TourStep(
                "select-raw-cut",
                "Dispersion cut",
                "Select dispersion_cut in ImageTool Manager.",
                mode="action",
                target=lambda: self._manager_row_target(self._raw_cut_uid()),
                allowed_inputs=actions,
                completion=lambda: self._node_is_selected(self._raw_cut_uid()),
                reveal=self._show_manager,
                auto_advance=False,
            ),
            TourStep(
                "paste-reusable-operations",
                "Paste the selected steps",
                "Right-click inside the [[ui:managerProvenancePage]] tab, then "
                "select [[ui:manager_paste_provenance_steps_action]].",
                mode="action",
                target=lambda: self._manager.metadata_derivation_list,
                allowed_inputs=actions,
                event_predicate=self._context_menu_event_predicate,
                completion=lambda: self._converted_cut_uid() is not None,
                reveal=self._show_raw_cut_provenance,
                auto_advance=False,
            ),
            TourStep(
                "validate-converted-cut",
                "Reused corrections",
                "The same procedure is now applied to dispersion_cut.",
                target=lambda: self._manager_row_target(self._converted_cut_uid()),
                allowed_inputs=information,
                ready=self._converted_cut_is_valid,
                reveal=self._select_converted_cut,
            ),
            TourStep(
                "open-converted-cut",
                "Open the converted cut",
                "Double-click the highlighted item.",
                mode="action",
                target=lambda: self._manager_row_target(self._converted_cut_uid()),
                allowed_inputs=actions,
                completion=lambda: self._uid_tool_visible(self._converted_cut_uid()),
                reveal=self._select_converted_cut,
            ),
            switch_window(
                "switch-to-converted-cut",
                "converted ImageTool",
                lambda: self._manager,
                self._converted_cut_tool,
            ),
            TourStep(
                "new-figure",
                "Open Figure Composer",
                "Right-click the image to open its menu. This menu provides data "
                "export, selection code, new windows, and interactive analysis "
                "tools for the displayed data. Select "
                "[[ui:itool_plot_with_matplotlib_action]] to open Figure Composer.",
                mode="action",
                target=self._converted_cut_image_target,
                allowed_inputs=actions,
                allowed_objects=(self._new_figure_action,),
                event_predicate=self._context_menu_event_predicate,
                completion=lambda: self._figure_composer() is not None,
            ),
            switch_window(
                "switch-to-figure-composer",
                "Figure Composer",
                self._converted_cut_tool,
                self._figure_composer,
                self._show_figure_composer,
            ),
            TourStep(
                "figure-composer-output",
                "Figure Composer",
                "This is Figure Composer. It builds a Matplotlib figure from data "
                "selected in ImageTool Manager. The separate plot window shows the "
                "rendered result. [[ui:figureComposerShowFigureButton]] opens or "
                "raises that window.",
                target=self._figure_output_target,
                allowed_inputs=information,
                reveal=self._show_figure_composer,
            ),
            TourStep(
                "select-figure-composer-sources",
                "Sources",
                "Select the [[ui:figureComposerSourcesPage]] tab.",
                mode="action",
                target=lambda: self._figure_composer_tab_target(0),
                allowed_inputs=actions,
                allowed_objects=(self._figure_composer_tab_bar,),
                subscriptions=(self._figure_composer_tab_signal,),
                completion=lambda: self._figure_composer_tab_is_selected(0),
                reveal=self._show_figure_composer,
                auto_advance=False,
            ),
            TourStep(
                "figure-composer-sources",
                "Sources",
                "The Sources panel lists the data variables available to the figure.",
                target=lambda: self._figure_composer_panel("source_panel"),
                allowed_inputs=information,
                reveal=self._show_figure_composer,
            ),
            TourStep(
                "select-figure-composer-layout",
                "Layout",
                "Select the [[ui:figureComposerLayoutPage]] tab.",
                mode="action",
                target=lambda: self._figure_composer_tab_target(1),
                allowed_inputs=actions,
                allowed_objects=(self._figure_composer_tab_bar,),
                subscriptions=(self._figure_composer_tab_signal,),
                completion=lambda: self._figure_composer_tab_is_selected(1),
                reveal=self._show_figure_composer,
                auto_advance=False,
            ),
            TourStep(
                "figure-composer-layout",
                "Layout",
                "Layout sets the axes grid and shared-axis relationships.",
                target=lambda: self._figure_composer_panel("layout_panel"),
                allowed_inputs=information,
                reveal=self._show_figure_composer,
            ),
            TourStep(
                "select-figure-composer-recipe",
                "Recipe",
                "Select the [[ui:figureComposerRecipePage]] tab.",
                mode="action",
                target=lambda: self._figure_composer_tab_target(2),
                allowed_inputs=actions,
                allowed_objects=(self._figure_composer_tab_bar,),
                subscriptions=(self._figure_composer_tab_signal,),
                completion=lambda: self._figure_composer_tab_is_selected(2),
                reveal=self._show_figure_composer,
                auto_advance=False,
            ),
            TourStep(
                "figure-composer-recipe",
                "Recipe",
                "The Recipe panel lists the plotting steps and their settings.",
                target=lambda: self._figure_composer_panel("operation_panel"),
                allowed_inputs=information,
                reveal=self._show_figure_composer,
            ),
            TourStep(
                "select-figure-composer-export",
                "Export",
                "Select the [[ui:figureComposerExportPage]] tab.",
                mode="action",
                target=lambda: self._figure_composer_tab_target(3),
                allowed_inputs=actions,
                allowed_objects=(self._figure_composer_tab_bar,),
                subscriptions=(self._figure_composer_tab_signal,),
                completion=lambda: self._figure_composer_tab_is_selected(3),
                reveal=self._show_figure_composer,
                auto_advance=False,
            ),
            TourStep(
                "figure-composer-export",
                "Export and Python code",
                "Export controls file output. "
                "[[ui:figureComposerCopyPythonButton]] copies a standalone script "
                "for the current recipe.",
                target=self._figure_export_target,
                allowed_inputs=information,
                reveal=self._show_figure_composer,
            ),
            switch_window(
                "switch-to-manager-finish",
                "ImageTool Manager",
                self._figure_composer,
                lambda: self._manager,
                self._show_manager,
            ),
            TourStep(
                "workspace-save-as",
                "Workspace files",
                (
                    "In the macOS menu bar, "
                    "[[menu:manager_file_menu|manager_save_workspace_as_action]] "
                    "stores this "
                    if _NATIVE_MENU_BAR
                    else "The [[menu:manager_file_menu|"
                    "manager_save_workspace_as_action]] command stores this "
                )
                + "Manager tree, Provenance history, ImageTool state, and the Figure "
                "Composer recipe. Select Finish to end the tutorial.",
                target=self._save_as_target,
                allowed_inputs=information,
                continue_label="Finish",
                reveal=self._reveal_save_as,
            ),
        ]
        debug_actions = self._debug_actions()
        action_ids = {step.id for step in steps if step.mode == "action"}
        missing = (
            action_ids
            - debug_actions.keys()
            - {step.id for step in steps if step.debug_action is not None}
        )
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise RuntimeError(
                f"Tutorial action steps do not have debug actions: {missing_text}."
            )
        return [
            dataclasses.replace(step, debug_action=debug_actions.get(step.id))
            if step.debug_action is None
            else step
            for step in steps
        ]

    def _debug_actions(self) -> dict[str, typing.Callable[[], None]]:
        return {
            "open-data-explorer": self._debug_open_explorer,
            "select-map": lambda: self._debug_select_explorer_path(self.data_files.map),
            "enable-map-preview": self._debug_enable_preview,
            "open-map-in-manager": lambda: self._debug_click(
                self._explorer_open_button(), "Open in Manager"
            ),
            "ctrl-drag-cursor": lambda: self._debug_move_cursor(cursor=0),
            "transpose-alpha-beta": self._debug_transpose_alpha_beta,
            "set-energy-bin": lambda: self._debug_set_energy_bin(5),
            "add-second-cursor": self._debug_add_cursor,
            "move-second-cursor": lambda: self._debug_move_cursor(cursor=1),
            "set-second-cursor-bin": lambda: self._debug_set_energy_bin(3),
            "select-first-cursor": lambda: self._debug_select_cursor(0),
            "open-coordinate-editor": lambda: self._debug_trigger(
                self._coordinate_action(), "Edit Coordinates"
            ),
            "select-energy-coordinate": self._debug_select_energy_coordinate,
            "select-scale-offset": lambda: self._debug_select_coordinate_tab(1),
            "set-energy-offset": self._debug_set_energy_offset,
            "apply-energy-correction": lambda: self._debug_click(
                self._coordinate_apply_button(), "coordinate correction"
            ),
            "select-c6-guideline": lambda: self._debug_trigger(
                self._c6_action(), "C6 rotation guideline"
            ),
            "set-normal-emission-and-azimuth": self._debug_set_geometry,
            "open-ktool": lambda: self._debug_trigger(
                self._ktool_action(), "Open ktool"
            ),
            "select-ktool-visualization": lambda: self._debug_select_ktool_tab(1),
            "select-ktool-parameters": lambda: self._debug_select_ktool_tab(0),
            "open-converted-map": lambda: self._debug_click(
                self._ktool_button("ktoolOpenInImageToolButton"),
                "Open in ImageTool",
            ),
            "reveal-converted-map": lambda: self._debug_trigger(
                self._reveal_in_manager_action(self._converted_map_uid()),
                "Reveal in Manager",
            ),
            "select-manager-provenance": self._debug_select_provenance_tab,
            "switch-to-explorer-cut": self._debug_open_explorer,
            "select-cut": lambda: self._debug_select_explorer_path(self.data_files.cut),
            "open-cut-in-manager": lambda: self._debug_click(
                self._explorer_open_button(), "Open in Manager"
            ),
            "switch-to-manager-operations": lambda: self._debug_trigger(
                self._reveal_in_manager_action(self._raw_cut_uid()),
                "Reveal in Manager",
            ),
            "select-converted-map": lambda: self._select_uid(self._converted_map_uid()),
            "expand-input-history": self._debug_expand_reusable_input,
            "select-reusable-operations": self._debug_select_reusable_operations,
            "copy-reusable-operations": lambda: self._debug_trigger(
                self._manager._metadata_copy_selected_action,
                "Copy selected steps",
            ),
            "select-raw-cut": lambda: self._select_uid(self._raw_cut_uid()),
            "paste-reusable-operations": self._debug_paste_reusable_operations,
            "open-converted-cut": lambda: self._manager.show_selected(),
            "new-figure": self._debug_open_figure_composer,
            "select-figure-composer-sources": lambda: self._debug_select_figure_tab(0),
            "select-figure-composer-layout": lambda: self._debug_select_figure_tab(1),
            "select-figure-composer-recipe": lambda: self._debug_select_figure_tab(2),
            "select-figure-composer-export": lambda: self._debug_select_figure_tab(3),
        }

    def _debug_activate_window(self, window: QtWidgets.QWidget | None) -> None:
        if window is None:
            raise RuntimeError("The destination window is not available.")
        window.show()
        window.raise_()
        window.activateWindow()
        handle = window.windowHandle()
        if handle is not None:
            handle.requestActivate()
        self._debug_active_window = window

    def _debug_open_explorer(self) -> None:
        self._debug_trigger(
            getattr(self._manager, "explorer_action", None), "Data Explorer"
        )
        self._debug_activate_window(self._explorer_window(include_hidden=True))

    @staticmethod
    def _debug_click(
        button: QtWidgets.QWidget | None,
        description: str,
    ) -> None:
        if not isinstance(button, QtWidgets.QAbstractButton):
            raise TypeError(f"The {description} button is not available.")
        if not button.isEnabled():
            raise RuntimeError(f"The {description} button is disabled.")
        button.click()

    @staticmethod
    def _debug_trigger(action: QtGui.QAction | None, description: str) -> None:
        if action is None:
            raise RuntimeError(f"The {description} action is not available.")
        if not action.isEnabled():
            raise RuntimeError(f"The {description} action is disabled.")
        action.trigger()

    def _debug_select_explorer_path(self, path: pathlib.Path) -> None:
        explorer = self._explorer()
        if explorer is None:
            raise RuntimeError("Data Explorer is not available.")
        index = explorer._model_index_for_path(path)
        if not index.isValid():
            raise RuntimeError(f"The tutorial file is not available: {path.name}.")
        selection_model = explorer._tree_view.selectionModel()
        if selection_model is None:
            raise RuntimeError("The Data Explorer selection model is not available.")
        explorer._tree_view.setCurrentIndex(index)
        selection_model.select(
            index,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )

    def _debug_enable_preview(self) -> None:
        preview = self._widget("dataExplorerPreviewCheck")
        if not isinstance(preview, QtWidgets.QCheckBox):
            raise TypeError("The data preview checkbox is not available.")
        preview.setChecked(True)

    def _debug_move_cursor(self, *, cursor: int) -> None:
        tool = self._map_tool()
        if tool is None:
            raise RuntimeError("The tutorial ImageTool is not available.")
        axis = tool.slicer_area.data.get_axis_num("alpha")
        current = tool.slicer_area.array_slicer.get_indices(cursor)[axis]
        size = tool.slicer_area.data.sizes["alpha"]
        target = (current + max(1, size // 8)) % size
        tool.slicer_area.set_index(axis, target, cursor=cursor)

    def _debug_transpose_alpha_beta(self) -> None:
        tool = self._map_tool()
        if tool is None:
            raise RuntimeError("The tutorial ImageTool is not available.")
        tool.cursor_controls.btn_transpose[0].click()

    def _debug_set_energy_bin(self, value: int) -> None:
        tool = self._map_tool()
        axis = self._energy_axis()
        if tool is None or axis is None:
            raise RuntimeError("The energy bin-width control is not available.")
        tool.binning_controls.spins[axis].setValue(value)

    def _debug_add_cursor(self) -> None:
        tool = self._map_tool()
        if tool is None:
            raise RuntimeError("The tutorial ImageTool is not available.")
        tool.cursor_controls.btn_add.click()

    def _debug_select_cursor(self, cursor: int) -> None:
        tool = self._map_tool()
        if tool is None:
            raise RuntimeError("The tutorial ImageTool is not available.")
        tool.slicer_area.set_current_cursor(cursor)

    def _debug_select_energy_coordinate(self) -> None:
        selector = self._coordinate_selector()
        if selector is None:
            raise RuntimeError("The coordinate selector is not available.")
        selector.setCurrentText("eV")

    def _debug_select_coordinate_tab(self, index: int) -> None:
        tabs = self._coordinate_edit_mode_tabs()
        if tabs is None:
            raise RuntimeError("The coordinate editor tabs are not available.")
        tabs.setCurrentIndex(index)

    def _debug_set_energy_offset(self) -> None:
        scale = self._coordinate_scale_spin()
        offset = self._coordinate_offset_spin()
        if scale is None or offset is None:
            raise RuntimeError("The coordinate scale and offset controls are missing.")
        scale.setValue(1.0)
        offset.setValue(-45.5)

    def _debug_set_geometry(self) -> None:
        tool = self._map_tool()
        if tool is None:
            raise RuntimeError("The tutorial ImageTool is not available.")
        for dim, value in (("alpha", 2.0), ("beta", -1.5)):
            tool.slicer_area.set_value(tool.slicer_area.data.get_axis_num(dim), value)
        guidelines = tool.slicer_area.main_image._guidelines_items
        if not guidelines:
            raise RuntimeError("The rotation guideline is not available.")
        guidelines[0].setAngle(86.0)

    def _debug_select_ktool_tab(self, index: int) -> None:
        tabs = self._ktool_tabs()
        if tabs is None:
            raise RuntimeError("The ktool tabs are not available.")
        tabs.setCurrentIndex(index)

    def _debug_select_provenance_tab(self) -> None:
        self._manager.inspector_tabs.setCurrentWidget(
            self._manager.metadata_provenance_page
        )

    def _debug_select_reusable_operations(self) -> None:
        items = self._reusable_operation_items()
        if items is None:
            raise RuntimeError("The three reusable provenance rows are not available.")
        operation_list = self._manager.metadata_derivation_list
        operation_list.clearSelection()
        for item in items:
            item.setSelected(True)
        operation_list.setCurrentItem(
            items[1],
            0,
            QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
        )

    def _debug_expand_reusable_input(self) -> None:
        item = self._reusable_input_item()
        if item is None:
            raise RuntimeError("The input provenance row is not available.")
        item.setExpanded(True)

    def _reusable_operation_items(
        self,
    ) -> (
        tuple[
            QtWidgets.QTreeWidgetItem,
            QtWidgets.QTreeWidgetItem,
            QtWidgets.QTreeWidgetItem,
        ]
        | None
    ):
        from erlab.interactive.imagetool import _kspace_conversion
        from erlab.interactive.imagetool._provenance._operations import (
            AffineCoordOperation,
        )
        from erlab.interactive.imagetool.manager._widgets import (
            _METADATA_DERIVATION_ROW_ROLE,
        )

        uid = self._converted_map_uid()
        if uid is None:
            return None
        node = self._manager._tool_graph.nodes.get(uid)
        if node is None or node.displayed_provenance_spec is None:
            return None
        operation_items: list[QtWidgets.QTreeWidgetItem] = []
        group_items: list[QtWidgets.QTreeWidgetItem] = []
        operation_list = self._manager.metadata_derivation_list
        for row_index in range(operation_list.conceptual_count()):
            item = operation_list.conceptual_item(row_index)
            if item is None:
                continue
            row = item.data(0, _METADATA_DERIVATION_ROW_ROLE)
            ref = getattr(row, "replay_ref", None)
            spec = self._manager._provenance_edit_controller._display_spec_for_row(
                node,
                row,
            )
            operation = (
                None if ref is None or spec is None else spec._operation_for_ref(ref)
            )
            if operation is None:
                continue
            if (
                isinstance(operation, AffineCoordOperation)
                and operation.coord_name == "eV"
                and np.isclose(operation.scale, 1.0)
                and np.isclose(operation.offset, -45.5)
            ):
                operation_items.append(item)
            if (
                operation.group is not None
                and operation.group.kind
                == _kspace_conversion.KSPACE_CONVERSION_GROUP_KIND
            ):
                group_items.append(item)
        if len(operation_items) != 1 or len(group_items) != 2:
            return None
        return operation_items[0], group_items[0], group_items[1]

    def _reusable_input_item(self) -> QtWidgets.QTreeWidgetItem | None:
        items = self._reusable_operation_items()
        if items is None:
            return None
        return items[0].parent()

    def _reusable_input_target(self) -> ModelIndexTarget | None:
        item = self._reusable_input_item()
        if item is None:
            return None
        operation_list = self._manager.metadata_derivation_list
        index = operation_list.indexFromItem(item)
        if not index.isValid():
            return None
        return ModelIndexTarget(operation_list, index)

    def _reusable_input_expanded(self) -> bool:
        item = self._reusable_input_item()
        return item is not None and item.isExpanded()

    def _debug_paste_reusable_operations(self) -> None:
        self._manager._build_metadata_derivation_menu(include_row_actions=False)
        action = self._manager._metadata_paste_steps_action
        self._debug_trigger(action, "Paste selected steps")

    def _debug_open_figure_composer(self) -> None:
        self._converted_cut_image_target()
        self._debug_trigger(self._new_figure_action(), "New Figure")

    def _debug_select_figure_tab(self, index: int) -> None:
        tabs = self._figure_composer_tabs()
        if tabs is None:
            raise RuntimeError("The Figure Composer tabs are not available.")
        tabs.setCurrentIndex(index)

    def _show_manager(self) -> None:
        self._show_without_activating(self._manager)

    @staticmethod
    def _show_without_activating(window: QtWidgets.QWidget) -> None:
        if window.isVisible():
            return
        attribute = QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating
        previous = window.testAttribute(attribute)
        window.setAttribute(attribute, True)
        window.show()
        window.setAttribute(attribute, previous)

    def _window_is_active(self, window: QtWidgets.QWidget | None) -> bool:
        return window is not None and (
            window.isActiveWindow()
            or (self._debug and window is self._debug_active_window)
        )

    @staticmethod
    def _active_window_or(
        fallback: QtWidgets.QWidget | None,
    ) -> QtWidgets.QWidget | None:
        active = QtWidgets.QApplication.activeWindow()
        if active is not None and active.isVisible():
            return active
        return fallback

    def _reveal_explorer(self) -> None:
        explorer_window = self._explorer_window(include_hidden=True)
        if explorer_window is not None:
            self._show_without_activating(explorer_window)

    def _explorer_window(
        self, *, include_hidden: bool = False
    ) -> _TabbedExplorer | None:
        with contextlib.suppress(AttributeError, RuntimeError):
            explorer = self._manager.explorer
            if include_hidden or explorer.isVisible():
                return explorer
        return None

    def _explorer_action_target(self) -> ActionTarget | QtWidgets.QWidget:
        if _NATIVE_MENU_BAR:
            return self._manager
        return ActionTarget(self._manager.explorer_action, self._manager.file_menu)

    def _reveal_explorer_action(self) -> None:
        self._show_manager()
        if _NATIVE_MENU_BAR:
            return
        menu_bar = self._manager.menuBar()
        menu_height = 0 if menu_bar is None else menu_bar.height()
        self._manager.file_menu.popup(
            self._manager.mapToGlobal(QtCore.QPoint(8, menu_height))
        )

    def _configure_opened_explorer(self) -> bool:
        explorer_window = self._explorer_window()
        if explorer_window is None:
            return False
        if not self._explorer_configured:
            explorer = explorer_window.current_explorer
            if explorer is None:
                return False
            explorer.restore_workspace_state(
                DataExplorerTabState(
                    root_path=str(self.directory), loader_name="tutorial"
                )
            )
            self._explorer_configured = True
        return True

    def _explorer(self) -> typing.Any | None:
        with contextlib.suppress(AttributeError, RuntimeError):
            return self._manager.explorer.current_explorer
        return None

    def _explorer_file_ready(self, path: pathlib.Path) -> bool:
        explorer = self._explorer()
        if explorer is None:
            return False
        return (
            self._explorer_file_selected(path)
            and explorer._preview_check.isChecked()
            and explorer._up_to_date
        )

    def _explorer_file_selected(self, path: pathlib.Path) -> bool:
        explorer = self._explorer()
        return bool(
            explorer is not None
            and explorer.loader_name == "tutorial"
            and explorer._current_selection == [path]
        )

    def _explorer_open_button(self) -> QtWidgets.QWidget | None:
        return self._widget("dataExplorerOpenInManagerButton")

    def _widget(self, object_name: str) -> QtWidgets.QWidget | None:
        application = typing.cast(
            "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
        )
        if application is None:
            return None
        return next(
            (
                widget
                for widget in application.allWidgets()
                if widget.objectName() == object_name and widget.isVisible()
            ),
            None,
        )

    def _action(self, object_name: str) -> QtGui.QAction | None:
        application = typing.cast(
            "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
        )
        if application is None:
            return None
        for window in application.topLevelWidgets():
            action = window.findChild(QtGui.QAction, object_name)
            if action is not None:
                return action
        return None

    def _find_node_uid(self, name: str, *, parent_uid: str | None = None) -> str | None:
        for uid, node in self._manager._tool_graph.nodes.items():
            if node.name == name and (
                parent_uid is None or node.parent_uid == parent_uid
            ):
                return uid
        return None

    def _raw_cut_uid(self) -> str | None:
        return next(
            (
                uid
                for uid, node in self._manager._tool_graph.nodes.items()
                if node.name == "dispersion_cut" and node.parent_uid is None
            ),
            None,
        )

    def _converted_map_uid(self) -> str | None:
        map_uid = self._find_node_uid("example_map")
        if map_uid is None:
            return None
        descendants = self._manager._tool_graph.descendant_uids(map_uid)
        return next(
            (
                uid
                for uid in descendants
                if uid not in self._node_uids_before_conversion
                and self._manager._tool_graph.nodes[uid].is_imagetool
            ),
            None,
        )

    def _converted_cut_uid(self) -> str | None:
        cut_uid = self._raw_cut_uid()
        tool = self._tool_for_uid(cut_uid)
        if tool is not None:
            dims = set(tool.slicer_area.data.dims)
            if dims & {"kx", "ky", "kz"}:
                return cut_uid
        return None

    def _manager_row_target(self, uid: str | None) -> ModelIndexTarget | None:
        if uid is None:
            return None
        index = self._manager.tree_view._model._row_index(uid)
        return ModelIndexTarget(self._manager.tree_view, index)

    def _select_uid(self, uid: str | None) -> None:
        index = self._show_uid(uid)
        if index is None:
            return
        self._manager.tree_view.setCurrentIndex(index)
        selection_model = self._manager.tree_view.selectionModel()
        if selection_model is None:
            return
        selection_model.select(
            index,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )

    def _show_uid(self, uid: str | None) -> QtCore.QModelIndex | None:
        if uid is None:
            return None
        self._show_manager()
        index = self._manager.tree_view._model._row_index(uid)
        if not index.isValid():
            return None
        parent = index.parent()
        while parent.isValid():
            self._manager.tree_view.expand(parent)
            parent = parent.parent()
        self._manager.tree_view.scrollTo(index)
        return index

    def _node_is_selected(self, uid: str | None) -> bool:
        if uid is None:
            return False
        return uid in self._manager.tree_view.selected_childtool_uids or any(
            self._manager._tool_graph.root_wrappers[index].uid == uid
            for index in self._manager.tree_view.selected_imagetool_indices
        )

    def _tool_for_uid(self, uid: str | None) -> typing.Any | None:
        if uid is None:
            return None
        node = self._manager._tool_graph.nodes.get(uid)
        return None if node is None else node.imagetool

    def _map_tool(self) -> typing.Any | None:
        return self._tool_for_uid(self._find_node_uid("example_map"))

    def _show_map_tool(self) -> None:
        tool = self._map_tool()
        if tool is not None:
            tool.show()

    def _map_tool_widget(self, name: str) -> QtWidgets.QWidget | None:
        tool = self._map_tool()
        widget = None if tool is None else getattr(tool, name, None)
        return widget if isinstance(widget, QtWidgets.QWidget) else None

    def _map_display_controls_target(self) -> CompositeTarget | None:
        color = self._map_tool_widget("colormap_controls")
        binning = self._map_tool_widget("binning_controls")
        if color is None or binning is None:
            return None
        return CompositeTarget(color, binning)

    def _map_menus_target(self) -> QtWidgets.QWidget | None:
        tool = self._map_tool()
        if tool is None or _NATIVE_MENU_BAR:
            return tool
        return tool.menuBar()

    def _named_tool_visible(self, name: str) -> bool:
        tool = self._tool_for_uid(self._find_node_uid(name))
        return tool is not None and tool.isVisible()

    def _uid_tool_visible(self, uid: str | None) -> bool:
        tool = self._tool_for_uid(uid)
        return tool is not None and tool.isVisible()

    def _main_image_target(self) -> GraphicsItemTarget | None:
        tool = self._map_tool()
        if tool is None:
            return None
        return GraphicsItemTarget(tool.slicer_area.main_image)

    def _capture_cursor_start(self) -> None:
        self._show_map_tool()
        tool = self._map_tool()
        if tool is not None:
            self._cursor_start = tuple(
                tool.slicer_area.array_slicer.get_indices(
                    tool.slicer_area.current_cursor
                )
            )

    def _cursor_moved(self) -> bool:
        tool = self._map_tool()
        if tool is None or self._cursor_start is None:
            return False
        return (
            tuple(
                tool.slicer_area.array_slicer.get_indices(
                    tool.slicer_area.current_cursor
                )
            )
            != self._cursor_start
        )

    def _main_dims(self) -> tuple[str | None, str | None]:
        tool = self._map_tool()
        if tool is None:
            return (None, None)
        return tuple(tool.slicer_area.main_image.axis_dims_uniform)

    def _energy_axis(self) -> int | None:
        tool = self._map_tool()
        if tool is None:
            return None
        with contextlib.suppress(ValueError):
            return list(tool.slicer_area.data.dims).index("eV")
        return None

    def _energy_profile_target(self) -> GraphicsItemTarget | None:
        profile = self._energy_profile()
        return None if profile is None else GraphicsItemTarget(profile)

    def _energy_profile(self) -> typing.Any | None:
        tool = self._map_tool()
        axis = self._energy_axis()
        if tool is None or axis is None:
            return None
        return next(
            (
                item
                for item in tool.slicer_area.profiles
                if tuple(item.display_axis) == (axis,)
            ),
            None,
        )

    def _energy_profile_view(self) -> QtWidgets.QGraphicsView | None:
        profile = self._energy_profile()
        if profile is None:
            return None
        view = profile.getViewWidget()
        return view if isinstance(view, QtWidgets.QGraphicsView) else None

    def _energy_bin_spin(self) -> QtWidgets.QWidget | None:
        axis = self._energy_axis()
        return None if axis is None else self._widget(f"itoolBinAxis{axis}Spin")

    def _energy_bin(self, *, cursor: int | None = None) -> int | None:
        tool = self._map_tool()
        axis = self._energy_axis()
        if tool is None or axis is None:
            return None
        if cursor is None:
            cursor = tool.slicer_area.current_cursor
        if not 0 <= cursor < tool.slicer_area.n_cursors:
            return None
        return int(tool.slicer_area.array_slicer.get_bins(cursor)[axis])

    def _current_cursor(self) -> int | None:
        tool = self._map_tool()
        return None if tool is None else int(tool.slicer_area.current_cursor)

    def _second_cursor_moved(self) -> bool:
        tool = self._map_tool()
        if tool is None or tool.slicer_area.n_cursors != 2:
            return False
        indices = tool.slicer_area.array_slicer.get_indices
        return tool.slicer_area.current_cursor == 1 and tuple(indices(1)) != tuple(
            indices(0)
        )

    def _cursor_count(self) -> int:
        tool = self._map_tool()
        return 0 if tool is None else int(tool.slicer_area.n_cursors)

    def _menu_action_target(
        self, action_name: str, menu: QtWidgets.QMenu | None
    ) -> ActionTarget | None:
        if _NATIVE_MENU_BAR:
            return None
        action = self._action(action_name)
        return None if action is None else ActionTarget(action, menu)

    def _coordinate_action(self) -> QtGui.QAction | None:
        return self._action("itoolEditCoordinatesAction")

    def _coordinate_action_target(
        self,
    ) -> ActionTarget | QtWidgets.QWidget | None:
        tool = self._map_tool()
        if tool is None:
            return None
        if _NATIVE_MENU_BAR:
            return tool
        return self._menu_action_target(
            "itoolEditCoordinatesAction", tool.mnb.menu_dict["editMenu"]
        )

    def _reveal_coordinate_action(self) -> None:
        self._show_map_tool()
        tool = self._map_tool()
        if tool is not None and not _NATIVE_MENU_BAR:
            tool.mnb.menu_dict["editMenu"].popup(
                tool.mapToGlobal(QtCore.QPoint(50, tool.menuBar().height()))
            )

    def _coordinate_dialog(self) -> QtWidgets.QDialog | None:
        return next(
            (
                widget
                for widget in QtWidgets.QApplication.topLevelWidgets()
                if isinstance(widget, QtWidgets.QDialog)
                and type(widget).__name__ == "AssignCoordsDialog"
                and widget.isVisible()
            ),
            None,
        )

    def _coordinate_selector(self) -> QtWidgets.QComboBox | None:
        dialog = self._coordinate_dialog()
        if dialog is None:
            return None
        return dialog.findChild(
            QtWidgets.QComboBox, "coordinateEditorCoordinateSelector"
        )

    def _coordinate_selector_popup(self) -> QtWidgets.QWidget | None:
        selector = self._coordinate_selector()
        if selector is None:
            return None
        view = selector.view()
        return None if view is None else view.window()

    def _coordinate_selector_signal(self) -> typing.Any:
        selector = self._coordinate_selector()
        return None if selector is None else selector.currentTextChanged

    def _energy_coordinate_is_selected(self) -> bool:
        selector = self._coordinate_selector()
        return selector is not None and selector.currentText() == "eV"

    def _coordinate_edit_mode_tabs(self) -> QtWidgets.QTabWidget | None:
        dialog = self._coordinate_dialog()
        if dialog is None:
            return None
        coordinate_editor = getattr(dialog, "coord_widget", None)
        tabs = getattr(coordinate_editor, "edit_mode_tabs", None)
        return tabs if isinstance(tabs, QtWidgets.QTabWidget) else None

    def _coordinate_edit_mode_tab_bar(self) -> QtWidgets.QTabBar | None:
        tabs = self._coordinate_edit_mode_tabs()
        return None if tabs is None else tabs.tabBar()

    def _coordinate_edit_mode_signal(self) -> typing.Any:
        tabs = self._coordinate_edit_mode_tabs()
        return None if tabs is None else tabs.currentChanged

    def _scale_offset_is_selected(self) -> bool:
        tabs = self._coordinate_edit_mode_tabs()
        return tabs is not None and tabs.currentIndex() == 1

    def _coordinate_scale_spin(self) -> typing.Any:
        dialog = self._coordinate_dialog()
        if dialog is None:
            return None
        coordinate_editor = getattr(dialog, "coord_widget", None)
        spin = getattr(coordinate_editor, "scale_spin", None)
        return spin if isinstance(spin, QtWidgets.QAbstractSpinBox) else None

    def _coordinate_offset_spin(self) -> typing.Any:
        dialog = self._coordinate_dialog()
        if dialog is None:
            return None
        coordinate_editor = getattr(dialog, "coord_widget", None)
        spin = getattr(coordinate_editor, "offset_spin", None)
        return spin if isinstance(spin, QtWidgets.QAbstractSpinBox) else None

    def _coordinate_offset_signal(self) -> typing.Any:
        spin = self._coordinate_offset_spin()
        return None if spin is None else spin.valueChanged

    def _energy_offset_is_set(self) -> bool:
        scale = self._coordinate_scale_spin()
        offset = self._coordinate_offset_spin()
        return bool(
            scale is not None
            and offset is not None
            and np.isclose(scale.value(), 1.0)
            and np.isclose(offset.value(), -45.5)
        )

    def _coordinate_button_box(self) -> QtWidgets.QDialogButtonBox | None:
        dialog = self._coordinate_dialog()
        if dialog is None:
            return None
        button_box = getattr(dialog, "buttonBox", None)
        return (
            button_box if isinstance(button_box, QtWidgets.QDialogButtonBox) else None
        )

    def _coordinate_apply_button(self) -> QtWidgets.QPushButton | None:
        button_box = self._coordinate_button_box()
        if button_box is None:
            return None
        return button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)

    def _coordinate_apply_event_predicate(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent
    ) -> bool:
        return watched is self._coordinate_button_box() and event.type() in {
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonRelease,
        }

    def _coordinate_dialog_finished_signal(self) -> typing.Any:
        dialog = self._coordinate_dialog()
        return None if dialog is None else dialog.finished

    def _energy_is_corrected(self) -> bool:
        tool = self._map_tool()
        if tool is None or "eV" not in tool.slicer_area.data.coords:
            return False
        values = np.asarray(tool.slicer_area.data.eV.values)
        return bool(values.size and np.nanmax(values) < 1.0)

    def _c6_action(self) -> QtGui.QAction | None:
        tool = self._map_tool()
        if tool is None:
            return None
        return tool.slicer_area.main_image._guideline_actions[3]

    def _c6_action_target(self) -> ActionTarget | QtWidgets.QWidget | None:
        tool = self._map_tool()
        action = self._c6_action()
        if tool is None or action is None:
            return None
        if _NATIVE_MENU_BAR:
            return tool
        menu = tool.mnb.menu_dict.get("Rotation Guidelines")
        return ActionTarget(action, menu)

    def _reveal_c6_action(self) -> None:
        self._show_map_tool()
        tool = self._map_tool()
        if tool is None or _NATIVE_MENU_BAR:
            return
        view_menu = tool.mnb.menu_dict["viewMenu"]
        guideline_menu = tool.mnb.menu_dict.get("Rotation Guidelines")
        view_menu.popup(tool.mapToGlobal(QtCore.QPoint(110, tool.menuBar().height())))
        if guideline_menu is not None:
            guideline_menu.popup(
                view_menu.mapToGlobal(QtCore.QPoint(view_menu.width(), 80))
            )

    def _guideline_count(self) -> int:
        tool = self._map_tool()
        if tool is None:
            return 0
        return max(0, len(tool.slicer_area.main_image._guidelines_items) - 1) * 2

    def _normal_emission_is_set(self) -> bool:
        tool = self._map_tool()
        if tool is None:
            return False
        values = {
            dim: tool.slicer_area.array_slicer.get_value(
                tool.slicer_area.current_cursor,
                tool.slicer_area.data.get_axis_num(dim),
                uniform=True,
            )
            for dim in ("alpha", "beta")
        }
        return bool(
            np.isclose(values["alpha"], 2.0, atol=0.5)
            and np.isclose(values["beta"], -1.5, atol=1.0)
        )

    def _guideline_angle(self) -> float:
        tool = self._map_tool()
        if tool is None:
            return 999.0
        return float(tool.slicer_area.main_image._guideline_angle)

    def _ktool_action(self) -> QtGui.QAction | None:
        return self._action("itoolOpenKtoolAction")

    def _ktool_action_target(self) -> ActionTarget | QtWidgets.QWidget | None:
        tool = self._map_tool()
        if tool is None:
            return None
        if _NATIVE_MENU_BAR:
            return tool
        return self._menu_action_target(
            "itoolOpenKtoolAction", tool.mnb.menu_dict["viewMenu"]
        )

    def _reveal_ktool_action(self) -> None:
        self._show_map_tool()
        tool = self._map_tool()
        if tool is not None and not _NATIVE_MENU_BAR:
            tool.mnb.menu_dict["viewMenu"].popup(
                tool.mapToGlobal(QtCore.QPoint(110, tool.menuBar().height()))
            )

    def _ktool(self) -> QtWidgets.QWidget | None:
        return self._widget("ktoolWindow")

    def _show_ktool(self) -> None:
        tool = self._ktool()
        if tool is not None:
            tool.show()

    def _ktool_child(self, name: str) -> QtWidgets.QWidget | None:
        tool = self._ktool()
        return None if tool is None else tool.findChild(QtWidgets.QWidget, name)

    def _ktool_tabs(self) -> QtWidgets.QTabWidget | None:
        tool = self._ktool()
        return (
            None if tool is None else tool.findChild(QtWidgets.QTabWidget, "tabWidget")
        )

    def _ktool_tab_bar(self) -> QtWidgets.QTabBar | None:
        tabs = self._ktool_tabs()
        return None if tabs is None else tabs.tabBar()

    def _ktool_tab_target(self, index: int) -> RectTarget | None:
        tab_bar = self._ktool_tab_bar()
        if tab_bar is None:
            return None
        return RectTarget(tab_bar.tabRect(index), tab_bar)

    def _ktool_visualization_tab_target(self) -> RectTarget | None:
        return self._ktool_tab_target(1)

    def _ktool_parameters_tab_target(self) -> RectTarget | None:
        return self._ktool_tab_target(0)

    def _ktool_tab_signal(self) -> typing.Any:
        tabs = self._ktool_tabs()
        return None if tabs is None else tabs.currentChanged

    def _ktool_visualization_is_selected(self) -> bool:
        tabs = self._ktool_tabs()
        return tabs is not None and tabs.currentIndex() == 1

    def _ktool_parameters_is_selected(self) -> bool:
        tabs = self._ktool_tabs()
        return tabs is not None and tabs.currentIndex() == 0

    def _ktool_composite(self, *names: str) -> CompositeTarget | None:
        widgets = tuple(self._ktool_child(name) for name in names)
        if any(widget is None for widget in widgets):
            return None
        return CompositeTarget(*typing.cast("tuple[QtWidgets.QWidget, ...]", widgets))

    def _ktool_button(self, name: str) -> QtWidgets.QWidget | None:
        tool = self._ktool()
        return None if tool is None else tool.findChild(QtWidgets.QWidget, name)

    def _prepare_map_conversion(self) -> None:
        self._node_uids_before_conversion = set(self._manager._tool_graph.nodes)
        self._show_ktool()

    def _converted_map_tool(self) -> typing.Any | None:
        return self._tool_for_uid(self._converted_map_uid())

    def _reveal_in_manager_action(self, uid: str | None) -> QtGui.QAction | None:
        tool = self._tool_for_uid(uid)
        if tool is None:
            return None
        return tool.reveal_in_manager_act

    def _reveal_in_manager_action_target(
        self, uid: str | None
    ) -> ActionTarget | QtWidgets.QWidget | None:
        tool = self._tool_for_uid(uid)
        action = self._reveal_in_manager_action(uid)
        if tool is None or action is None:
            return None
        if _NATIVE_MENU_BAR:
            return tool
        return ActionTarget(action, tool.mnb.menu_dict["windowMenu"])

    def _prepare_reveal_in_manager(self, uid: str | None) -> None:
        tool = self._tool_for_uid(uid)
        if tool is not None:
            action = tool.reveal_in_manager_act
            if action is not self._reveal_action:
                if self._reveal_action is not None:
                    with contextlib.suppress(RuntimeError, TypeError):
                        self._reveal_action.triggered.disconnect(
                            self._manager_reveal_triggered
                        )
                self._reveal_action = action
                if action is not None:
                    action.triggered.connect(self._manager_reveal_triggered)
            self._expected_reveal_uid = uid
            self._revealed_uid = None
            tool.show()
            if not _NATIVE_MENU_BAR:
                tool.mnb.menu_dict["windowMenu"].popup(
                    tool.mapToGlobal(QtCore.QPoint(170, tool.menuBar().height()))
                )

    def _uid_was_revealed(self, uid: str | None) -> bool:
        return (
            uid is not None
            and self._revealed_uid == uid
            and self._node_is_selected(uid)
        )

    def _manager_reveal_triggered(self) -> None:
        self._revealed_uid = self._expected_reveal_uid
        self.notify_state_changed()

    def _manager_inspector_tab_bar(self) -> QtWidgets.QTabBar | None:
        return self._manager.inspector_tabs.tabBar()

    def _manager_provenance_tab_target(self) -> RectTarget | None:
        tabs = self._manager.inspector_tabs
        index = tabs.indexOf(self._manager.metadata_provenance_page)
        tab_bar = self._manager_inspector_tab_bar()
        if index < 0 or tab_bar is None:
            return None
        return RectTarget(tab_bar.tabRect(index), tab_bar)

    def _manager_inspector_tab_signal(self) -> typing.Any:
        return self._manager.inspector_tabs.currentChanged

    def _manager_provenance_is_selected(self) -> bool:
        return (
            self._manager.inspector_tabs.currentWidget()
            is self._manager.metadata_provenance_page
        )

    def _show_converted_map_provenance(self) -> None:
        self._show_manager()
        self._select_uid(self._converted_map_uid())

    def _show_converted_map_in_tree(self) -> None:
        self._show_uid(self._converted_map_uid())

    def _prepare_operations_copy(self) -> None:
        self._operations_copied = False
        self._show_converted_map_provenance()

    def _operations_copy_triggered(self) -> None:
        self._operations_copied = True
        self.notify_state_changed()

    def _reusable_operations_selected(self) -> bool:
        from erlab.interactive.imagetool import _kspace_conversion
        from erlab.interactive.imagetool._provenance._operations import (
            AffineCoordOperation,
        )

        payload = self._manager._details_panel._selected_derivation_step_payload()
        if payload is None:
            return False
        operations = payload[0]
        has_energy_edit = any(
            isinstance(operation, AffineCoordOperation)
            and operation.coord_name == "eV"
            and np.isclose(operation.scale, 1.0)
            and np.isclose(operation.offset, -45.5)
            for operation in operations
        )
        has_momentum_group = any(
            operation.group is not None
            and operation.group.kind == _kspace_conversion.KSPACE_CONVERSION_GROUP_KIND
            for operation in operations
        )
        return has_energy_edit and has_momentum_group

    def _provenance_steps_on_clipboard(self) -> bool:
        clipboard = QtWidgets.QApplication.clipboard()
        mime_data = None if clipboard is None else clipboard.mimeData()
        return bool(
            self._operations_copied
            and mime_data is not None
            and mime_data.hasFormat(_PROVENANCE_STEPS_CLIPBOARD_MIME)
        )

    @staticmethod
    def _context_menu_event_predicate(
        watched: QtCore.QObject | None, _event: QtCore.QEvent
    ) -> bool:
        current = watched
        while current is not None:
            if isinstance(current, QtWidgets.QMenu):
                return True
            current = current.parent()
        return False

    def _show_raw_cut_provenance(self) -> None:
        self._show_manager()
        self._select_uid(self._raw_cut_uid())

    def _select_converted_cut(self) -> None:
        self._select_uid(self._converted_cut_uid())

    def _converted_cut_is_valid(self) -> bool:
        tool = self._tool_for_uid(self._converted_cut_uid())
        if tool is None:
            return False
        converted_dims = set(tool.slicer_area.data.dims)
        return (
            "eV" in converted_dims
            and bool(converted_dims & {"kx", "ky", "kz"})
            and float(np.nanmax(tool.slicer_area.data.eV.values)) < 1.0
        )

    def _converted_cut_tool(self) -> QtWidgets.QWidget | None:
        tool = self._tool_for_uid(self._converted_cut_uid())
        return tool if isinstance(tool, QtWidgets.QWidget) else None

    def _show_converted_cut(self) -> None:
        tool = self._converted_cut_tool()
        if tool is not None:
            tool.show()

    def _converted_cut_image_target(self) -> GraphicsItemTarget | None:
        tool = self._tool_for_uid(self._converted_cut_uid())
        if tool is None:
            return None
        image = tool.slicer_area.main_image
        image.getMenu()
        image.ensure_manager_figure_actions()
        return GraphicsItemTarget(image)

    def _new_figure_action(self) -> QtGui.QAction | None:
        tool = self._tool_for_uid(self._converted_cut_uid())
        if tool is None:
            return None
        action = tool.slicer_area.main_image._plot_with_matplotlib_action
        return action if isinstance(action, QtGui.QAction) else None

    def _figure_composer(self) -> QtWidgets.QWidget | None:
        uid = self._figure_composer_uid
        if uid is None:
            figure_uids = self._manager._figure_uids()
            selected_uids = [
                candidate
                for candidate in self._manager._selected_figure_uids()
                if candidate in figure_uids
            ]
            if len(selected_uids) == 1:
                uid = selected_uids[0]
            elif len(figure_uids) == 1:
                uid = figure_uids[0]
            else:
                return None
            self._figure_composer_uid = uid
        try:
            tool = self._manager._child_node(uid).tool_window
        except KeyError:
            return None
        return tool if isinstance(tool, QtWidgets.QWidget) else None

    def _show_figure_composer(self) -> None:
        composer = self._figure_composer()
        if composer is not None:
            composer.show()

    def _figure_output_target(self) -> QtWidgets.QWidget | None:
        composer = self._figure_composer()
        if composer is None:
            return None
        return composer.findChild(QtWidgets.QWidget, "figureComposerShowFigureButton")

    def _figure_composer_tabs(self) -> QtWidgets.QTabWidget | None:
        composer = self._figure_composer()
        tabs = None if composer is None else getattr(composer, "editor_tabs", None)
        return tabs if isinstance(tabs, QtWidgets.QTabWidget) else None

    def _figure_composer_tab_bar(self) -> QtWidgets.QTabBar | None:
        tabs = self._figure_composer_tabs()
        return None if tabs is None else tabs.tabBar()

    def _figure_composer_tab_target(self, index: int) -> RectTarget | None:
        tab_bar = self._figure_composer_tab_bar()
        if tab_bar is None:
            return None
        return RectTarget(tab_bar.tabRect(index), tab_bar)

    def _figure_composer_tab_signal(self) -> typing.Any:
        tabs = self._figure_composer_tabs()
        return None if tabs is None else tabs.currentChanged

    def _figure_composer_tab_is_selected(self, index: int) -> bool:
        tabs = self._figure_composer_tabs()
        return tabs is not None and tabs.currentIndex() == index

    def _figure_composer_panel(self, name: str) -> QtWidgets.QWidget | None:
        composer = self._figure_composer()
        panel = None if composer is None else getattr(composer, name, None)
        return panel if isinstance(panel, QtWidgets.QWidget) else None

    def _figure_export_target(self) -> CompositeTarget | None:
        composer = self._figure_composer()
        if composer is None:
            return None
        panel = self._figure_composer_panel("export_panel")
        copy_button = composer.findChild(
            QtWidgets.QWidget, "figureComposerCopyPythonButton"
        )
        if panel is None or copy_button is None:
            return None
        return CompositeTarget(panel, copy_button)

    def _save_as_target(self) -> ActionTarget | QtWidgets.QWidget:
        if _NATIVE_MENU_BAR:
            return self._manager
        return ActionTarget(self._manager.save_as_action, self._manager.file_menu)

    def _reveal_save_as(self) -> None:
        self._show_manager()
        if _NATIVE_MENU_BAR:
            return
        menu_bar = self._manager.menuBar()
        menu_height = 0 if menu_bar is None else menu_bar.height()
        self._manager.file_menu.popup(
            self._manager.mapToGlobal(QtCore.QPoint(8, menu_height))
        )


def start_tutorial(
    manager: ImageToolManager,
    *,
    debug: bool = False,
) -> _TutorialController:
    """Start the tutorial and retain its controller on the Manager."""
    existing = getattr(manager, "_tutorial_controller", None)
    if isinstance(existing, _TutorialController) and not existing.is_cleaned:
        existing.start()
        return existing
    controller = _TutorialController(manager, debug=debug)
    typing.cast("typing.Any", manager)._tutorial_controller = controller
    controller.start()
    return controller
