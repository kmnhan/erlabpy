"""Keyboard shortcut reference for ImageTool Manager."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive import _shortcut_sequences

_ShortcutSource = str | QtGui.QKeySequence.StandardKey


@dataclass(frozen=True)
class _ShortcutEntrySpec:
    id: str
    command: str
    keys: tuple[_ShortcutSource, ...] = ()
    display_keys: tuple[tuple[str, ...], ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class _ShortcutGroupSpec:
    id: str
    title: str
    entries: tuple[_ShortcutEntrySpec, ...]


@dataclass(frozen=True)
class _ShortcutTabSpec:
    id: str
    title: str
    groups: tuple[_ShortcutGroupSpec, ...]


@dataclass(frozen=True)
class _RenderedShortcut:
    chords: tuple[tuple[str, ...], ...]
    native_text: str
    portable_text: str


def _modifier_names(
    modifiers: QtCore.Qt.KeyboardModifier,
) -> tuple[str, ...]:
    if sys.platform == "darwin":
        ordered_modifiers = (
            (QtCore.Qt.KeyboardModifier.MetaModifier, "Meta"),
            (QtCore.Qt.KeyboardModifier.AltModifier, "Alt"),
            (QtCore.Qt.KeyboardModifier.ShiftModifier, "Shift"),
            (QtCore.Qt.KeyboardModifier.ControlModifier, "Ctrl"),
        )
    else:
        ordered_modifiers = (
            (QtCore.Qt.KeyboardModifier.ControlModifier, "Ctrl"),
            (QtCore.Qt.KeyboardModifier.AltModifier, "Alt"),
            (QtCore.Qt.KeyboardModifier.ShiftModifier, "Shift"),
            (QtCore.Qt.KeyboardModifier.MetaModifier, "Meta"),
        )
    return tuple(
        name for modifier, name in ordered_modifiers if modifiers & modifier == modifier
    )


def _display_token(token: str) -> str:
    if sys.platform != "darwin":
        return token
    # Qt maps ControlModifier to Command and MetaModifier to Control on macOS.
    return {
        "Meta": "⌃",
        "Alt": "⌥",
        "Shift": "⇧",
        "Ctrl": "⌘",
    }.get(token, token)


def _key_name(key: QtCore.Qt.Key) -> str:
    special_keys = {
        QtCore.Qt.Key.Key_Escape: "Esc",
        QtCore.Qt.Key.Key_Tab: "Tab",
        QtCore.Qt.Key.Key_Backtab: "Backtab",
        QtCore.Qt.Key.Key_Backspace: "Backspace",
        QtCore.Qt.Key.Key_Return: "Return",
        QtCore.Qt.Key.Key_Enter: "Enter",
        QtCore.Qt.Key.Key_Insert: "Insert",
        QtCore.Qt.Key.Key_Delete: "Delete",
        QtCore.Qt.Key.Key_Home: "Home",
        QtCore.Qt.Key.Key_End: "End",
        QtCore.Qt.Key.Key_Left: "Left",
        QtCore.Qt.Key.Key_Up: "Up",
        QtCore.Qt.Key.Key_Right: "Right",
        QtCore.Qt.Key.Key_Down: "Down",
        QtCore.Qt.Key.Key_PageUp: "Page Up",
        QtCore.Qt.Key.Key_PageDown: "Page Down",
        QtCore.Qt.Key.Key_Space: "Space",
    }
    if key in special_keys:
        return special_keys[key]
    return QtGui.QKeySequence(key).toString(
        QtGui.QKeySequence.SequenceFormat.PortableText
    )


def _combination_tokens(combination: QtCore.QKeyCombination) -> tuple[str, ...]:
    modifiers = tuple(
        _display_token(name)
        for name in _modifier_names(combination.keyboardModifiers())
    )
    return (*modifiers, _key_name(combination.key()))


def _shortcut_tokens(source: _ShortcutSource) -> tuple[tuple[str, ...], ...]:
    sequence = QtGui.QKeySequence(source)
    return tuple(
        _combination_tokens(sequence[index]) for index in range(sequence.count())
    )


def _arrow_key_tokens(sequence: str) -> tuple[str, ...]:
    combination = QtGui.QKeySequence(sequence)[0]
    modifiers = tuple(
        _display_token(name)
        for name in _modifier_names(combination.keyboardModifiers())
    )
    return (*modifiers, "Arrow key")


def _display_tokens(*tokens: str) -> tuple[str, ...]:
    return tuple(_display_token(token) for token in tokens)


def _shortcut_tabs() -> tuple[_ShortcutTabSpec, ...]:
    return (
        _ShortcutTabSpec(
            id="manager",
            title="Manager",
            groups=(
                _ShortcutGroupSpec(
                    id="manager-workspaces",
                    title="Workspaces",
                    entries=(
                        _ShortcutEntrySpec(
                            "manager-open-workspace",
                            "Open a workspace",
                            (_shortcut_sequences.MANAGER_OPEN_WORKSPACE,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-save-workspace",
                            "Save the workspace",
                            (_shortcut_sequences.MANAGER_SAVE_WORKSPACE,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-save-workspace-as",
                            "Save the workspace as a new file",
                            (_shortcut_sequences.MANAGER_SAVE_WORKSPACE_AS,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-workspace-properties",
                            "Show workspace properties",
                            (_shortcut_sequences.MANAGER_WORKSPACE_PROPERTIES,),
                        ),
                    ),
                ),
                _ShortcutGroupSpec(
                    id="manager-windows",
                    title="Windows and data",
                    entries=(
                        _ShortcutEntrySpec(
                            "manager-data-explorer",
                            "Open Data Explorer",
                            (_shortcut_sequences.MANAGER_DATA_EXPLORER,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-periodic-table",
                            "Open Periodic Table",
                            (_shortcut_sequences.MANAGER_PERIODIC_TABLE,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-hide-windows",
                            "Hide selected windows",
                            (_shortcut_sequences.MANAGER_HIDE_WINDOWS,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-remove-windows",
                            "Remove selected windows",
                            (_shortcut_sequences.MANAGER_REMOVE_WINDOWS,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-reload-data",
                            "Reload selected data",
                            (_shortcut_sequences.MANAGER_RELOAD_DATA,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-link-windows",
                            "Link selected windows",
                            (_shortcut_sequences.MANAGER_LINK_WINDOWS,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-unlink-windows",
                            "Unlink selected windows",
                            (_shortcut_sequences.MANAGER_UNLINK_WINDOWS,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-console",
                            "Show or hide the console",
                            (_shortcut_sequences.MANAGER_CONSOLE,),
                        ),
                        _ShortcutEntrySpec(
                            "manager-settings",
                            "Open settings",
                            (_shortcut_sequences.MANAGER_SETTINGS,),
                        ),
                    ),
                ),
                _ShortcutGroupSpec(
                    id="manager-tree",
                    title="Manager tree",
                    entries=(
                        _ShortcutEntrySpec(
                            "manager-show-selection",
                            "Show the selected window",
                            _shortcut_sequences.manager_show_selection(),
                            detail="The Manager tree must have keyboard focus.",
                        ),
                        _ShortcutEntrySpec(
                            "manager-rename-selection",
                            "Rename the selected window",
                            _shortcut_sequences.manager_rename_selection(),
                            detail="The Manager tree must have keyboard focus.",
                        ),
                    ),
                ),
            ),
        ),
        _ShortcutTabSpec(
            id="imagetool",
            title="ImageTool",
            groups=(
                _ShortcutGroupSpec(
                    id="imagetool-file",
                    title="Files and windows",
                    entries=(
                        _ShortcutEntrySpec(
                            "imagetool-open",
                            "Open data",
                            (_shortcut_sequences.IMAGETOOL_OPEN,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-save-as",
                            "Save data as",
                            (_shortcut_sequences.IMAGETOOL_SAVE_AS,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-close",
                            "Close the window",
                            (_shortcut_sequences.IMAGETOOL_CLOSE,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-remove",
                            "Remove the window from Manager",
                            (_shortcut_sequences.IMAGETOOL_REMOVE,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-reveal-manager",
                            "Reveal the data in Manager",
                            (_shortcut_sequences.IMAGETOOL_REVEAL_MANAGER,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-settings",
                            "Open settings",
                            (_shortcut_sequences.IMAGETOOL_SETTINGS,),
                        ),
                    ),
                ),
                _ShortcutGroupSpec(
                    id="imagetool-view",
                    title="View",
                    entries=(
                        _ShortcutEntrySpec(
                            "imagetool-view-all",
                            "View all data",
                            (_shortcut_sequences.IMAGETOOL_VIEW_ALL,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-transpose",
                            "Transpose the main image",
                            (_shortcut_sequences.IMAGETOOL_TRANSPOSE,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-reverse-colormap",
                            "Reverse the colormap",
                            (_shortcut_sequences.IMAGETOOL_REVERSE_COLORMAP,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-snap",
                            "Turn pixel snapping on or off",
                            (_shortcut_sequences.IMAGETOOL_SNAP_TO_PIXELS,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-reload",
                            "Reload the data",
                            (_shortcut_sequences.IMAGETOOL_RELOAD,),
                        ),
                    ),
                ),
                _ShortcutGroupSpec(
                    id="imagetool-cursors",
                    title="Cursors",
                    entries=(
                        _ShortcutEntrySpec(
                            "imagetool-add-cursor",
                            "Add a cursor",
                            (_shortcut_sequences.IMAGETOOL_ADD_CURSOR,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-remove-cursor",
                            "Remove a cursor",
                            (_shortcut_sequences.IMAGETOOL_REMOVE_CURSOR,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-toggle-cursor",
                            "Show or hide cursors",
                            (_shortcut_sequences.IMAGETOOL_TOGGLE_CURSORS,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-center-cursor",
                            "Center the active cursor",
                            (_shortcut_sequences.IMAGETOOL_CENTER_CURSOR,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-center-all-cursors",
                            "Center all cursors",
                            (_shortcut_sequences.IMAGETOOL_CENTER_ALL_CURSORS,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-move-cursor-one",
                            "Move the active cursor by one point",
                            display_keys=(
                                _arrow_key_tokens(
                                    _shortcut_sequences.IMAGETOOL_MOVE_CURSOR[0]
                                ),
                            ),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-move-cursor-ten",
                            "Move the active cursor by ten points",
                            display_keys=(
                                _arrow_key_tokens(
                                    _shortcut_sequences.IMAGETOOL_MOVE_CURSOR[4]
                                ),
                            ),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-move-all-cursors-one",
                            "Move all cursors by one point",
                            display_keys=(
                                _arrow_key_tokens(
                                    _shortcut_sequences.IMAGETOOL_MOVE_ALL_CURSORS[0]
                                ),
                            ),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-move-all-cursors-ten",
                            "Move all cursors by ten points",
                            display_keys=(
                                _arrow_key_tokens(
                                    _shortcut_sequences.IMAGETOOL_MOVE_ALL_CURSORS[4]
                                ),
                            ),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-copy-cursor-values",
                            "Copy cursor values",
                            (_shortcut_sequences.IMAGETOOL_COPY_CURSOR_VALUES,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-copy-cursor-indices",
                            "Copy cursor indices",
                            (_shortcut_sequences.IMAGETOOL_COPY_CURSOR_INDICES,),
                        ),
                    ),
                ),
                _ShortcutGroupSpec(
                    id="imagetool-history",
                    title="History",
                    entries=(
                        _ShortcutEntrySpec(
                            "imagetool-undo",
                            "Undo the last action",
                            (_shortcut_sequences.IMAGETOOL_UNDO,),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-redo",
                            "Redo the last action",
                            (_shortcut_sequences.IMAGETOOL_REDO,),
                        ),
                    ),
                ),
                _ShortcutGroupSpec(
                    id="imagetool-pointer",
                    title="Pointer gestures",
                    entries=(
                        _ShortcutEntrySpec(
                            "imagetool-pan",
                            "Pan an image or profile",
                            display_keys=(("Left drag",),),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-zoom",
                            "Zoom an image or profile",
                            display_keys=(("Right drag",), ("Wheel",)),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-drag-active-cursor",
                            "Move the active cursor",
                            display_keys=(_display_tokens("Ctrl", "Left drag"),),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-drag-all-cursors",
                            "Move all cursors",
                            display_keys=(_display_tokens("Ctrl", "Alt", "Left drag"),),
                        ),
                        _ShortcutEntrySpec(
                            "imagetool-drag-cursor-lines",
                            "Move corresponding cursor lines",
                            display_keys=(_display_tokens("Alt", "Cursor-line drag"),),
                        ),
                    ),
                ),
            ),
        ),
        _ShortcutTabSpec(
            id="explorer",
            title="Data Explorer",
            groups=(
                _ShortcutGroupSpec(
                    id="explorer-files",
                    title="Files and folders",
                    entries=(
                        _ShortcutEntrySpec(
                            "explorer-open-manager",
                            "Open the selection in Manager",
                            (_shortcut_sequences.EXPLORER_OPEN_IN_MANAGER,),
                        ),
                        _ShortcutEntrySpec(
                            "explorer-open-folder",
                            "Open a folder",
                            (_shortcut_sequences.EXPLORER_OPEN_FOLDER,),
                        ),
                        _ShortcutEntrySpec(
                            "explorer-reload-folder",
                            "Reload the current folder",
                            (_shortcut_sequences.EXPLORER_RELOAD_FOLDER,),
                        ),
                        _ShortcutEntrySpec(
                            "explorer-enclosing-folder",
                            "Go to the enclosing folder",
                            (_shortcut_sequences.explorer_enclosing_folder(),),
                        ),
                        _ShortcutEntrySpec(
                            "explorer-close",
                            "Close the tab or window",
                            (_shortcut_sequences.EXPLORER_CLOSE,),
                        ),
                    ),
                ),
                _ShortcutGroupSpec(
                    id="explorer-tabs",
                    title="Tabs",
                    entries=(
                        _ShortcutEntrySpec(
                            "explorer-new-tab",
                            "Open a new tab",
                            (_shortcut_sequences.EXPLORER_NEW_TAB,),
                        ),
                        _ShortcutEntrySpec(
                            "explorer-next-tab",
                            "Go to the next tab",
                            (_shortcut_sequences.explorer_next_tab(),),
                        ),
                        _ShortcutEntrySpec(
                            "explorer-previous-tab",
                            "Go to the previous tab",
                            (_shortcut_sequences.explorer_previous_tab(),),
                        ),
                    ),
                ),
            ),
        ),
        _ShortcutTabSpec(
            id="figure-composer",
            title="Figure Composer",
            groups=(
                _ShortcutGroupSpec(
                    id="figure-composer-figure",
                    title="Figure",
                    entries=(
                        _ShortcutEntrySpec(
                            "figure-composer-undo",
                            "Undo an edit",
                            (_shortcut_sequences.FIGURE_COMPOSER_UNDO,),
                        ),
                        _ShortcutEntrySpec(
                            "figure-composer-redo",
                            "Redo an edit",
                            (_shortcut_sequences.FIGURE_COMPOSER_REDO,),
                        ),
                        _ShortcutEntrySpec(
                            "figure-composer-save-workspace",
                            "Save the workspace",
                            (_shortcut_sequences.FIGURE_COMPOSER_SAVE_WORKSPACE,),
                            detail=(
                                "This shortcut is available when Figure Composer is "
                                "managed."
                            ),
                        ),
                        _ShortcutEntrySpec(
                            "figure-composer-close",
                            "Close the figure window",
                            (_shortcut_sequences.FIGURE_COMPOSER_CLOSE,),
                        ),
                    ),
                ),
                _ShortcutGroupSpec(
                    id="figure-composer-sources",
                    title="Sources and editors",
                    entries=(
                        _ShortcutEntrySpec(
                            "figure-composer-rename-source",
                            "Rename the selected source",
                            (_shortcut_sequences.FIGURE_COMPOSER_RENAME_SOURCE,),
                            detail="The Sources list must have keyboard focus.",
                        ),
                        _ShortcutEntrySpec(
                            "figure-composer-completion",
                            "Show available completions",
                            (_shortcut_sequences.FIGURE_COMPOSER_COMPLETION,),
                            detail="A supported text editor must have keyboard focus.",
                        ),
                    ),
                ),
            ),
        ),
    )


def _native_shortcut(source: _ShortcutSource) -> str:
    return QtGui.QKeySequence(source).toString(
        QtGui.QKeySequence.SequenceFormat.NativeText
    )


def _portable_shortcut(source: _ShortcutSource) -> str:
    return QtGui.QKeySequence(source).toString(
        QtGui.QKeySequence.SequenceFormat.PortableText
    )


def _modifier_search_aliases(text: str) -> str:
    return (
        text.replace("⌘", " command ")
        .replace("⌥", " option ")
        .replace("⌃", " control ")
        .replace("⇧", " shift ")
    )


def _render_shortcut(source: _ShortcutSource) -> _RenderedShortcut:
    return _RenderedShortcut(
        chords=_shortcut_tokens(source),
        native_text=_native_shortcut(source),
        portable_text=_portable_shortcut(source),
    )


def _render_display_shortcut(tokens: tuple[str, ...]) -> _RenderedShortcut:
    text = " ".join(tokens)
    return _RenderedShortcut(
        chords=(tokens,),
        native_text=text,
        portable_text=text,
    )


class _ShortcutKeyLabel(QtWidgets.QLabel):
    def __init__(self, text: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(text, parent)
        self.setObjectName("keyboardShortcutKeycap")
        self.setProperty("shortcutToken", text)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._update_palette_style()
        self.setFixedHeight(self.fontMetrics().height() + 10)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        if event is not None and event.type() in {
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._update_palette_style()
        super().changeEvent(event)

    def _update_palette_style(self) -> None:
        if getattr(self, "_palette_update_active", False):
            return
        self._palette_update_active = True
        palette = self.palette()
        try:
            background = palette.button().color()
            foreground = palette.buttonText().color()
            border = palette.mid().color()
            color_format = QtGui.QColor.NameFormat.HexArgb
            style_sheet = (
                "QLabel#keyboardShortcutKeycap {"
                f"background-color: {background.name(color_format)};"
                f"color: {foreground.name(color_format)};"
                f"border: 1px solid {border.name(color_format)};"
                "border-radius: 5px;"
                "padding: 3px 7px;"
                "}"
            )
            if self.styleSheet() != style_sheet:
                self.setStyleSheet(style_sheet)
        finally:
            self._palette_update_active = False


class _ShortcutSequenceWidget(QtWidgets.QWidget):
    def __init__(
        self,
        shortcut: _RenderedShortcut,
        *,
        shortcut_id: str,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("keyboardShortcutSequence")
        self.setProperty("shortcutId", shortcut_id)
        self.setProperty("shortcutText", shortcut.native_text)
        self.setAccessibleName(shortcut.native_text)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        for chord_index, tokens in enumerate(shortcut.chords):
            if chord_index:
                separator = QtWidgets.QLabel("then", self)
                separator.setObjectName("keyboardShortcutChordSeparator")
                separator.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
                layout.addWidget(separator)
            for token in tokens:
                key_label = _ShortcutKeyLabel(token, self)
                key_label.setProperty("shortcutId", shortcut_id)
                layout.addWidget(key_label)


class _ShortcutRow(QtWidgets.QWidget):
    def __init__(
        self,
        spec: _ShortcutEntrySpec,
        *,
        search_prefix: str,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.setObjectName(f"keyboardShortcutRow_{spec.id}")
        self.setProperty("shortcutId", spec.id)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(2, 3, 2, 3)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(2)
        layout.setColumnStretch(0, 1)

        command_label = QtWidgets.QLabel(spec.command, self)
        command_label.setObjectName(f"keyboardShortcutCommand_{spec.id}")
        command_label.setWordWrap(True)
        layout.addWidget(command_label, 0, 0)

        if spec.detail:
            detail_label = QtWidgets.QLabel(spec.detail, self)
            detail_label.setObjectName(f"keyboardShortcutDetail_{spec.id}")
            detail_label.setWordWrap(True)
            detail_label.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
            layout.addWidget(detail_label, 1, 0)

        shortcuts = tuple(_render_shortcut(key) for key in spec.keys)
        shortcuts += tuple(
            _render_display_shortcut(tokens) for tokens in spec.display_keys
        )
        keys_widget = QtWidgets.QWidget(self)
        keys_widget.setObjectName(f"keyboardShortcutKeys_{spec.id}")
        keys_layout = QtWidgets.QHBoxLayout(keys_widget)
        keys_layout.setContentsMargins(0, 0, 0, 0)
        keys_layout.setSpacing(5)
        keys_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        for index, shortcut in enumerate(shortcuts):
            if index:
                separator = QtWidgets.QLabel("or", keys_widget)
                separator.setObjectName("keyboardShortcutAlternative")
                separator.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
                keys_layout.addWidget(separator)
            keys_layout.addWidget(
                _ShortcutSequenceWidget(
                    shortcut,
                    shortcut_id=spec.id,
                    parent=keys_widget,
                )
            )
        layout.addWidget(keys_widget, 0, 1, 2, 1)

        shortcut_text_parts: list[str] = []
        for shortcut in shortcuts:
            shortcut_text_parts.extend((shortcut.native_text, shortcut.portable_text))
            shortcut_text_parts.extend(
                token for chord in shortcut.chords for token in chord
            )
        shortcut_text = tuple(shortcut_text_parts)
        search_aliases = tuple(_modifier_search_aliases(text) for text in shortcut_text)
        self._search_text = " ".join(
            (
                spec.id,
                search_prefix,
                spec.command,
                spec.detail,
                *shortcut_text,
                *search_aliases,
            )
        ).casefold()

    def matches(self, query: str) -> bool:
        return not query or query in self._search_text


class _ShortcutTab(QtWidgets.QScrollArea):
    def __init__(self, spec: _ShortcutTabSpec, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.spec = spec
        self.setObjectName(f"keyboardShortcutTab_{spec.id}")
        self.setProperty("shortcutContext", spec.id)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget(self)
        content.setObjectName(f"keyboardShortcutContent_{spec.id}")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(10)

        self._groups: list[tuple[QtWidgets.QGroupBox, list[_ShortcutRow]]] = []
        for group_spec in spec.groups:
            group = QtWidgets.QGroupBox(group_spec.title, content)
            group.setObjectName(f"keyboardShortcutGroup_{group_spec.id}")
            group_layout = QtWidgets.QVBoxLayout(group)
            group_layout.setContentsMargins(10, 8, 10, 8)
            group_layout.setSpacing(2)
            rows: list[_ShortcutRow] = []
            search_prefix = f"{spec.title} {group_spec.title}"
            for entry_spec in group_spec.entries:
                row = _ShortcutRow(
                    entry_spec,
                    search_prefix=search_prefix,
                    parent=group,
                )
                rows.append(row)
                group_layout.addWidget(row)
            self._groups.append((group, rows))
            layout.addWidget(group)
        layout.addStretch()
        self.setWidget(content)

    def apply_filter(self, query: str) -> bool:
        has_match = False
        for group, rows in self._groups:
            group_has_match = False
            for row in rows:
                row_matches = row.matches(query)
                row.setVisible(row_matches)
                group_has_match |= row_matches
            group.setVisible(group_has_match)
            has_match |= group_has_match
        return has_match


class KeyboardShortcutsDialog(QtWidgets.QDialog):
    """Modeless keyboard shortcut reference for Manager applications."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("managerKeyboardShortcutsDialog")
        self.setWindowTitle("Keyboard Shortcuts")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setSizeGripEnabled(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        summary = QtWidgets.QLabel(self)
        summary.setObjectName("keyboardShortcutsSummary")
        summary.setWordWrap(True)
        summary.setText(
            "Shortcuts apply to the active window. The keys below use the "
            "conventions of this operating system."
        )
        layout.addWidget(summary)

        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setObjectName("keyboardShortcutsSearchEdit")
        self.search_edit.setAccessibleName("Search keyboard shortcuts")
        self.search_edit.setPlaceholderText("Search shortcuts")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_edit)

        self.tab_widget = QtWidgets.QTabWidget(self)
        self.tab_widget.setObjectName("keyboardShortcutsTabWidget")
        self._tabs: list[_ShortcutTab] = []
        for tab_spec in _shortcut_tabs():
            tab = _ShortcutTab(tab_spec, self.tab_widget)
            self._tabs.append(tab)
            self.tab_widget.addTab(tab, tab_spec.title)
        layout.addWidget(self.tab_widget, 1)

        self.empty_label = QtWidgets.QLabel("No matching shortcuts", self)
        self.empty_label.setObjectName("keyboardShortcutsEmptyLabel")
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setVisible(False)
        layout.addWidget(self.empty_label, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            parent=self,
        )
        buttons.setObjectName("keyboardShortcutsButtonBox")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._close_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Close), self
        )
        self._close_shortcut.activated.connect(self.close)

        self.setMinimumSize(560, 420)
        self.resize(760, 580)

    def _apply_filter(self, text: str) -> None:
        query = " ".join(text.casefold().split())
        matches = [tab.apply_filter(query) for tab in self._tabs]
        for index, has_match in enumerate(matches):
            self.tab_widget.setTabVisible(index, has_match)
        any_matches = any(matches)
        self.tab_widget.setVisible(any_matches)
        self.empty_label.setVisible(not any_matches)
        if any_matches and not self.tab_widget.isTabVisible(
            self.tab_widget.currentIndex()
        ):
            self.tab_widget.setCurrentIndex(matches.index(True))
