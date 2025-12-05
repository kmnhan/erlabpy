from __future__ import annotations

import numbers
import pathlib
import typing
import weakref
from collections.abc import Mapping, Sequence

import numpy as np
from qtpy import QtCore, QtWidgets

if typing.TYPE_CHECKING:
    import erlab


def _normalize_value(value: typing.Any) -> typing.Any:
    if isinstance(value, numbers.Number):
        return value
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def _flatten_state(
    state: Mapping[str, typing.Any], prefix: str | None = None
) -> dict[str, typing.Any]:
    flat: dict[str, typing.Any] = {}
    for key, value in state.items():
        key_path = key if prefix is None else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_state(value, key_path))
        else:
            flat[key_path] = _normalize_value(value)
    return flat


def parse_change(key: str, old, new) -> str:
    match key:
        case "cursor_colors":
            return "Cursor colors changed"
        case "slice.indices" | "slice.values":
            return "Cursor moved"
        case "slice.bins":
            return "Number of bins changed"
        case "slice.dims":
            return "Transposed"
        case "color.reverse":
            return "Colormap reversed"
        case "color.cmap":
            return f"Changed colormap to {new}"
        case "color.gamma" | "color.high_contrast" | "color.zero_centered":
            return "Colormap options changed"
        case "color.levels_locked":
            return "Colormap levels locked" if new else "Colormap levels unlocked"
        case "color.levels":
            return "Colormap levels changed"
        case _:
            return f"{key} changed"


def describe_state_diff(
    prev: Mapping[str, typing.Any], curr: Mapping[str, typing.Any]
) -> str | None:
    prev_flat, curr_flat = _flatten_state(prev), _flatten_state(curr)
    keys: set[str] = set.intersection(set(prev_flat.keys()), set(curr_flat.keys()))

    old_ncursors = len(prev_flat["slice.indices"])
    new_ncursors = len(curr_flat["slice.indices"])
    if old_ncursors > new_ncursors:
        return "Cursor removed"

    changes: set[str] = {key for key in keys if prev_flat[key] != curr_flat[key]}

    if "current_cursor" in changes:
        if old_ncursors == new_ncursors:
            return "Cursor changed"
        return "Cursor added"

    # Ignore value changes if indices didn't change
    changes.discard("slice.values")

    if "slice.dims" in changes and set(prev_flat["slice.dims"]) == set(
        curr_flat["slice.dims"]
    ):
        # Transposed
        changes.discard("slice.indices")

    return (
        ", ".join(
            sorted(
                {parse_change(key, prev_flat[key], curr_flat[key]) for key in changes}
            )
        )
        if changes
        else None
    )


class HistoryMenu(QtWidgets.QMenu):
    def __init__(
        self,
        slicer_area: erlab.interactive.imagetool.core.ImageSlicerArea,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._slicer_area = weakref.ref(slicer_area)
        self.aboutToShow.connect(self.update_actions)
        self._actions: list[QtWidgets.QAction] = []
        self.slicer_area.sigHistoryChanged.connect(self._history_changed)

    @property
    def slicer_area(self) -> erlab.interactive.imagetool.core.ImageSlicerArea:
        """Get the parent main window."""
        slicer_area = self._slicer_area()
        if slicer_area:
            return slicer_area
        raise LookupError("Parent was destroyed")

    @QtCore.Slot()
    def _history_changed(self) -> None:
        if self.isVisible():
            self.update_actions()

    def update_actions(self) -> None:
        self.clear()
        self._actions.clear()
        states, current_index = self.slicer_area.history_states()
        if len(states) < 2:
            action = QtWidgets.QAction("No history recorded yet")
            action.setEnabled(False)
            self.addAction(action)
            self._actions.append(action)
            return

        for i in range(len(states) - 1, 0, -1):
            prev_state, this_state = states[i - 1], states[i]
            summary = describe_state_diff(prev_state, this_state) or "No changes"
            marker = "•" if i == current_index else " "
            action = QtWidgets.QAction(f"{marker} {summary}")
            action.triggered.connect(
                lambda *, idx=i - current_index: self.slicer_area.go_to_history_index(
                    idx
                )
            )
            self.addAction(action)
            self._actions.append(action)
