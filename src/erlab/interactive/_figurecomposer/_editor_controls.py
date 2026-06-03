"""Editor-control adapters for batch editing and stale-signal safety.

Every recipe-mutating control in a Figure Composer operation editor should be
represented by an :class:`EditorControlAdapter` or created through a
``FigureComposerTool`` factory that uses one.  This keeps three requirements in
one place:

- stale widgets from an older editor rebuild must not mutate the current recipe;
- compatible multi-selection must display a clear ``(multiple values)`` state;
- unchanged mixed-value controls must not overwrite selected operations.

When adding a new widget kind, add an adapter here instead of connecting Qt
signals directly in an operation module.
"""

from __future__ import annotations

import abc
import typing
import weakref

from qtpy import QtCore, QtWidgets

if typing.TYPE_CHECKING:
    from collections.abc import Callable

MIXED_VALUES_TEXT = "(multiple values)"
MIXED_VALUE = object()


class EditorControlAdapter(abc.ABC):
    """Adapter contract for operation-editor controls."""

    def __init__(self, widget: QtWidgets.QWidget) -> None:
        self._widget_ref = weakref.ref(widget)

    @property
    def widget(self) -> QtWidgets.QWidget:
        widget = self._widget_ref()
        if widget is None:
            raise RuntimeError("Editor control was destroyed")
        return widget

    @abc.abstractmethod
    def connect_commit(
        self,
        connector: Callable[[QtWidgets.QWidget, typing.Any, Callable[..., None]], None],
        callback: Callable[[typing.Any], None],
    ) -> None:
        """Connect user commits through a guarded connector."""

    @abc.abstractmethod
    def set_mixed(self, mixed: bool) -> None:
        """Put the control into or out of its mixed-value presentation."""

    def mixed_row_widget(
        self, *, mixed: bool, parent: QtWidgets.QWidget | None = None
    ) -> QtWidgets.QWidget:
        """Return the widget that should be inserted into a form row."""
        if not mixed:
            return self.widget
        container = QtWidgets.QWidget(parent or self.widget.parentWidget())
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        marker = QtWidgets.QLabel(MIXED_VALUES_TEXT, container)
        marker.setObjectName("figureComposerMixedValueMarker")
        marker.setEnabled(False)
        layout.addWidget(self.widget, 1)
        layout.addWidget(marker)
        return container


class LineEditControlAdapter(EditorControlAdapter):
    """Adapter for single-line text edits."""

    @property
    def widget(self) -> QtWidgets.QLineEdit:
        return typing.cast("QtWidgets.QLineEdit", super().widget)

    def connect_commit(
        self,
        connector: Callable[[QtWidgets.QWidget, typing.Any, Callable[..., None]], None],
        callback: Callable[[typing.Any], None],
    ) -> None:
        edit = self.widget
        connector(
            edit,
            edit.editingFinished,
            lambda edit=edit: None if self.unchanged_mixed() else callback(edit.text()),
        )

    def set_mixed(self, mixed: bool) -> None:
        edit = self.widget
        if mixed:
            edit.setPlaceholderText(MIXED_VALUES_TEXT)
            edit.setProperty("batch_mixed", True)
            edit.setModified(False)
        else:
            edit.setProperty("batch_mixed", False)

    def unchanged_mixed(self) -> bool:
        edit = self.widget
        return bool(edit.property("batch_mixed")) and not edit.isModified()


class PlainTextControlAdapter(EditorControlAdapter):
    """Adapter for plain-text edits."""

    @property
    def widget(self) -> QtWidgets.QPlainTextEdit:
        return typing.cast("QtWidgets.QPlainTextEdit", super().widget)

    def connect_commit(
        self,
        connector: Callable[[QtWidgets.QWidget, typing.Any, Callable[..., None]], None],
        callback: Callable[[typing.Any], None],
    ) -> None:
        edit = self.widget
        connector(
            edit,
            edit.textChanged,
            lambda edit=edit: (
                None if self.unchanged_mixed() else callback(edit.toPlainText())
            ),
        )

    def set_mixed(self, mixed: bool) -> None:
        edit = self.widget
        if mixed:
            edit.setPlaceholderText(MIXED_VALUES_TEXT)
            edit.setProperty("batch_mixed", True)
            document = edit.document()
            if document is not None:
                document.setModified(False)
        else:
            edit.setProperty("batch_mixed", False)

    def unchanged_mixed(self) -> bool:
        edit = self.widget
        document = edit.document()
        return (
            bool(edit.property("batch_mixed"))
            and document is not None
            and not document.isModified()
        )


class ComboBoxControlAdapter(EditorControlAdapter):
    """Adapter for combo boxes with a disabled mixed-value sentinel."""

    @property
    def widget(self) -> QtWidgets.QComboBox:
        return typing.cast("QtWidgets.QComboBox", super().widget)

    def connect_commit(
        self,
        connector: Callable[[QtWidgets.QWidget, typing.Any, Callable[..., None]], None],
        callback: Callable[[typing.Any], None],
    ) -> None:
        combo = self.widget
        connector(
            combo,
            combo.currentIndexChanged,
            lambda _index, combo=combo: (
                None
                if combo.currentData() is MIXED_VALUE
                or combo.currentText() == MIXED_VALUES_TEXT
                else callback(combo.currentText())
            ),
        )

    def set_mixed(self, mixed: bool) -> None:
        if mixed:
            self.insert_mixed_placeholder()

    def insert_mixed_placeholder(self) -> None:
        combo = self.widget
        combo.insertItem(0, MIXED_VALUES_TEXT, MIXED_VALUE)
        item = typing.cast("typing.Any", combo.model()).item(0)
        if item is not None:
            item.setEnabled(False)
        combo.setCurrentIndex(0)


class ComboBoxDataControlAdapter(ComboBoxControlAdapter):
    """Adapter for combo boxes whose committed value lives in itemData."""

    def connect_commit(
        self,
        connector: Callable[[QtWidgets.QWidget, typing.Any, Callable[..., None]], None],
        callback: Callable[[typing.Any], None],
    ) -> None:
        combo = self.widget
        connector(
            combo,
            combo.activated,
            lambda _index, combo=combo: (
                None
                if combo.currentData() is MIXED_VALUE
                else callback(combo.currentData())
            ),
        )


class CheckBoxControlAdapter(EditorControlAdapter):
    """Adapter for boolean controls with partial mixed state."""

    @property
    def widget(self) -> QtWidgets.QCheckBox:
        return typing.cast("QtWidgets.QCheckBox", super().widget)

    def connect_commit(
        self,
        connector: Callable[[QtWidgets.QWidget, typing.Any, Callable[..., None]], None],
        callback: Callable[[typing.Any], None],
    ) -> None:
        check = self.widget
        if check.isTristate():
            connector(
                check,
                check.stateChanged,
                lambda state: (
                    None
                    if QtCore.Qt.CheckState(state)
                    == QtCore.Qt.CheckState.PartiallyChecked
                    else callback(
                        QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
                    )
                ),
            )
        else:
            connector(check, check.toggled, callback)

    def set_mixed(self, mixed: bool) -> None:
        check = self.widget
        if mixed:
            check.setTristate(True)
            check.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)


class SignalValueControlAdapter(EditorControlAdapter):
    """Adapter for custom controls that expose a signal and value getter."""

    def __init__(
        self,
        widget: QtWidgets.QWidget,
        signal: typing.Any,
        value_getter: Callable[..., typing.Any],
        *,
        unchanged_mixed: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(widget)
        self._signal = signal
        self._value_getter = value_getter
        self._unchanged_mixed = unchanged_mixed

    def connect_commit(
        self,
        connector: Callable[[QtWidgets.QWidget, typing.Any, Callable[..., None]], None],
        callback: Callable[[typing.Any], None],
    ) -> None:
        connector(
            self.widget,
            self._signal,
            lambda *args: (
                None
                if self._unchanged_mixed is not None and self._unchanged_mixed()
                else callback(self._value_getter(*args))
            ),
        )

    def set_mixed(self, _mixed: bool) -> None:
        return
