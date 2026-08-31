"""Qt text inputs with reusable completion support."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive import _shortcut_sequences

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _create_completer(
    widget: QtWidgets.QWidget,
    completions: Sequence[str],
) -> tuple[QtCore.QStringListModel, QtWidgets.QCompleter, QtWidgets.QAbstractItemView]:
    model = QtCore.QStringListModel(list(completions), widget)
    completer = QtWidgets.QCompleter(model, widget)
    completer.setWidget(widget)
    completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
    completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
    completer.setFilterMode(QtCore.Qt.MatchFlag.MatchStartsWith)
    completer.setWrapAround(False)
    popup = typing.cast("QtWidgets.QAbstractItemView", completer.popup())
    return model, completer, popup


def _create_completion_shortcut(
    widget: QtWidgets.QWidget,
    callback: Callable[[], None],
) -> QtGui.QShortcut:
    shortcut = QtGui.QShortcut(
        QtGui.QKeySequence(_shortcut_sequences.FIGURE_COMPOSER_COMPLETION), widget
    )
    shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
    shortcut.activated.connect(callback)
    return shortcut


def _prepare_completions(
    completer: QtWidgets.QCompleter,
    popup: QtWidgets.QAbstractItemView,
    prefix: str,
) -> bool:
    completer.setCompletionPrefix(prefix)
    if completer.completionCount() == 0:
        popup.hide()
        return False
    completion_model = typing.cast(
        "QtCore.QAbstractItemModel", completer.completionModel()
    )
    popup.setCurrentIndex(completion_model.index(0, 0))
    return True


class CompletingLineEdit(QtWidgets.QLineEdit):
    """Single-line text editor with curated, optional completions."""

    COMPLETIONS: tuple[str, ...] = ()

    def __init__(
        self,
        text: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        completions: Sequence[str] | None = None,
    ) -> None:
        super().__init__(text, parent)
        completion_values = self.COMPLETIONS if completions is None else completions
        (
            self.completion_model,
            self.completion_completer,
            self.completer_popup,
        ) = _create_completer(self, completion_values)
        self.setCompleter(self.completion_completer)
        self.completion_completer.activated[str].connect(self._insert_completion)
        self.completion_shortcut = _create_completion_shortcut(
            self, self.show_all_completions
        )

    @QtCore.Slot()
    def show_all_completions(self) -> None:
        """Show all configured completions."""
        if _prepare_completions(self.completion_completer, self.completer_popup, ""):
            self.completion_completer.complete()

    @QtCore.Slot(str)
    def _insert_completion(self, completion: str) -> None:
        if self.text() != completion:
            self.selectAll()
            self.insert(completion)
        self.setModified(True)
        self.completer_popup.hide()


class CompletingPlainTextEdit(QtWidgets.QPlainTextEdit):
    """Multiline text editor that completes the current line."""

    COMPLETIONS: tuple[str, ...] = ()

    def __init__(
        self,
        text: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        completions: Sequence[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setPlainText(text)
        completion_values = self.COMPLETIONS if completions is None else completions
        (
            self.completion_model,
            self.completion_completer,
            self.completer_popup,
        ) = _create_completer(self, completion_values)
        self.completion_completer.activated[str].connect(self._insert_completion)
        self.completion_shortcut = _create_completion_shortcut(
            self, self.show_all_completions
        )
        self._inserting_completion = False
        self.textChanged.connect(self._update_completions)

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:  # pragma: no cover - Qt always supplies a key event.
            return
        if self.completer_popup.isVisible() and event.key() in {
            QtCore.Qt.Key.Key_Enter,
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Escape,
            QtCore.Qt.Key.Key_Tab,
            QtCore.Qt.Key.Key_Backtab,
        }:
            event.ignore()
            return
        super().keyPressEvent(event)

    @QtCore.Slot()
    def show_all_completions(self) -> None:
        """Show all configured completions for the current line."""
        self._show_completions("")

    @QtCore.Slot()
    def _update_completions(self) -> None:
        if self._inserting_completion:
            return
        prefix = self._current_line_prefix()
        if not prefix or prefix in self.completion_model.stringList():
            self.completer_popup.hide()
            return
        self._show_completions(prefix)

    def _current_line_prefix(self) -> str:
        cursor = self.textCursor()
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.StartOfLine,
            QtGui.QTextCursor.MoveMode.KeepAnchor,
        )
        return cursor.selectedText()

    def _show_completions(self, prefix: str) -> None:
        if not _prepare_completions(
            self.completion_completer, self.completer_popup, prefix
        ):
            return
        scrollbar = typing.cast(
            "QtWidgets.QScrollBar", self.completer_popup.verticalScrollBar()
        )
        popup_rect = self.cursorRect()
        popup_rect.setWidth(
            self.completer_popup.sizeHintForColumn(0) + scrollbar.sizeHint().width()
        )
        self.completion_completer.complete(popup_rect)

    @QtCore.Slot(str)
    def _insert_completion(self, completion: str) -> None:
        self._inserting_completion = True
        try:
            cursor = self.textCursor()
            cursor.beginEditBlock()
            cursor.select(QtGui.QTextCursor.SelectionType.LineUnderCursor)
            cursor.insertText(completion)
            cursor.endEditBlock()
            self.setTextCursor(cursor)
        finally:
            self._inserting_completion = False
        self.completer_popup.hide()
