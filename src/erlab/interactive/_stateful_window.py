"""Shared window history without data, persistence, or Manager requirements."""

from __future__ import annotations

import collections
import contextlib
import typing

from qtpy import QtCore, QtGui, QtWidgets, compat

from erlab.interactive import _shortcut_sequences

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import pydantic

M = typing.TypeVar("M", bound="pydantic.BaseModel")


class StatefulWindow(QtWidgets.QMainWindow, typing.Generic[M]):
    """Record model snapshots and restore them through Undo and Redo.

    Subclasses provide ``_history_state``. Its getter must return a detached
    snapshot, and its setter must restore the controls represented by that snapshot.
    Call ``_reset_history_stack`` after initialization and document replacement.
    Call ``_write_state`` after setting changes. Bursts are coalesced for 150 ms.

    History actions are attached to the window. Subclasses can also add them to
    their own menus. Register numeric controls with ``_register_history_spinboxes``
    to commit typed values before routing their Undo and Redo shortcuts here.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prev_states: collections.deque[M] = collections.deque(maxlen=5000)
        self._next_states: collections.deque[M] = collections.deque(maxlen=5000)
        self._write_history = True
        self._history_write_pending = False
        self._history_write_timer = QtCore.QTimer(self)
        self._history_write_timer.setSingleShot(True)
        self._history_write_timer.setInterval(150)
        self._history_write_timer.timeout.connect(self._flush_pending_history_write)
        self._history_spins: dict[QtCore.QObject, QtWidgets.QAbstractSpinBox] = {}

        self.undo_action = QtGui.QAction("&Undo", self)
        self.redo_action = QtGui.QAction("&Redo", self)
        for action, name, shortcut, callback in (
            (self.undo_action, "undo", QtGui.QKeySequence.StandardKey.Undo, self.undo),
            (self.redo_action, "redo", QtGui.QKeySequence.StandardKey.Redo, self.redo),
        ):
            action.setObjectName(f"tool_{name}_action")
            action.setShortcut(shortcut)
            action.setShortcutContext(
                QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
            )
            action.setIcon(QtGui.QIcon.fromTheme(f"edit-{name}"))
            action.setEnabled(False)
            action.triggered.connect(callback)
        self.addActions((self.undo_action, self.redo_action))
        self._close_shortcut = QtWidgets.QShortcut(
            _shortcut_sequences.WINDOW_CLOSE, self, self._hide_or_close
        )
        self._close_shortcut.setContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )

    @QtCore.Slot()
    def _hide_or_close(self) -> None:
        """Close this window. Hosts can override this to hide a managed window."""
        self.close()

    @property
    def _history_state(self) -> M:
        raise NotImplementedError

    @_history_state.setter
    def _history_state(self, state: M) -> None:
        raise NotImplementedError

    @staticmethod
    def _history_state_equal(left: M | None, right: M | None) -> bool:
        if left is None or right is None:
            return left is right
        return left.model_dump() == right.model_dump()

    @property
    def undoable(self) -> bool:
        """Return whether a previous state is available."""
        return len(self._prev_states) > 1

    @property
    def redoable(self) -> bool:
        """Return whether an undone state is available."""
        return len(self._next_states) > 0

    @contextlib.contextmanager
    def _history_suppressed(self) -> Iterator[None]:
        self._flush_pending_history_changes()
        original = self._write_history
        self._write_history = False
        with self._history_suppression_context():
            try:
                yield
            finally:
                # Resume recording before the host emits mutation notifications.
                self._write_history = original

    def _history_suppression_context(self) -> contextlib.AbstractContextManager[None]:
        """Coalesce host notifications while history recording is suppressed."""
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _history_transition(self, target_state: M) -> Iterator[None]:
        """Apply one history transition inside the host's mutation boundary."""
        with self._history_suppressed():
            yield

    def _flush_pending_history_changes(self) -> None:
        """Flush all pending history owned by this window."""
        self._flush_pending_history_write()

    def _reset_history_stack(self) -> None:
        self._history_write_timer.stop()
        self._history_write_pending = False
        self._prev_states.clear()
        self._next_states.clear()
        self._prev_states.append(self._history_state)
        self._update_history_actions()

    @QtCore.Slot()
    def _write_state(self, *_args: typing.Any) -> None:
        if self._write_history:
            self._record_state_change()

    def _record_state_change(self) -> None:
        if not self._prev_states:
            self._prev_states.append(self._history_state)
            self._update_history_actions()
            return
        self._history_write_pending = True
        self._history_write_timer.start()

    @QtCore.Slot()
    def _flush_pending_history_write(self) -> bool:
        if not self._history_write_pending:
            return False
        self._history_write_timer.stop()
        self._history_write_pending = False
        if not self._write_history:
            return False
        curr_state = self._history_state
        last_state = self._prev_states[-1] if self._prev_states else None
        if self._history_state_equal(last_state, curr_state):
            return False
        self._prev_states.append(curr_state)
        self._next_states.clear()
        self._update_history_actions()
        return True

    @QtCore.Slot()
    def _replace_last_state(self, *_args: typing.Any) -> None:
        if self._write_history:
            self._replace_recorded_state()

    def _replace_recorded_state(self) -> None:
        self._flush_pending_history_changes()
        curr_state = self._history_state
        if self._prev_states:
            self._prev_states[-1] = curr_state
        else:
            self._prev_states.append(curr_state)
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        for name, enabled in (
            ("undo_action", self.undoable),
            ("redo_action", self.redoable),
        ):
            action = getattr(self, name, None)
            if action is not None and compat.isalive(action):
                action.setEnabled(enabled)

    @typing.final
    @QtCore.Slot()
    def undo(self) -> None:
        """Undo the most recent recorded state change."""
        self._flush_pending_history_changes()
        if not self.undoable:
            return
        with self._history_transition(self._prev_states[-2]):
            self._undo_recorded_state()
        self._update_history_actions()

    def _undo_recorded_state(self) -> None:
        self._next_states.append(self._prev_states.pop())
        self._history_state = self._prev_states[-1]

    @typing.final
    @QtCore.Slot()
    def redo(self) -> None:
        """Redo the most recently undone state change."""
        self._flush_pending_history_changes()
        if not self.redoable:
            return
        with self._history_transition(self._next_states[-1]):
            self._redo_recorded_state()
        self._update_history_actions()

    def _redo_recorded_state(self) -> None:
        next_state = self._next_states.pop()
        self._prev_states.append(next_state)
        self._history_state = next_state

    def _register_history_spinboxes(
        self, spins: Iterable[QtWidgets.QAbstractSpinBox]
    ) -> None:
        """Route numeric Undo and Redo through this window's history.

        Keep both wrappers alive while the controls are registered. Controls must
        belong to this window; deleted controls are never accessed by queued work.
        """
        for spin in spins:
            for widget in (spin, spin.lineEdit()):
                if widget is not None:
                    self._history_spins[widget] = spin
                    widget.installEventFilter(self)

    def eventFilter(
        self, obj: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if (
            isinstance(event, QtGui.QKeyEvent)
            and obj is not None
            and obj in self._history_spins
            and event.type()
            in (QtCore.QEvent.Type.ShortcutOverride, QtCore.QEvent.Type.KeyPress)
        ):
            undo = event.matches(QtGui.QKeySequence.StandardKey.Undo)
            if undo or event.matches(QtGui.QKeySequence.StandardKey.Redo):
                if event.type() == QtCore.QEvent.Type.KeyPress:
                    self._history_spins[obj].interpretText()
                    if undo:
                        self.undo()
                    else:
                        self.redo()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        # Qt normally commits the focused editor only when the window is hidden,
        # after closeEvent returns. Commit now while change recording is active.
        focused = self.focusWidget()
        if focused is not None and compat.isalive(focused):
            focused.clearFocus()
        self._flush_pending_history_changes()
        super().closeEvent(event)
