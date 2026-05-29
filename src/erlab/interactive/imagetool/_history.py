from __future__ import annotations

import copy
import dataclasses
import datetime
import numbers
import pathlib
import typing
import weakref
from collections.abc import Hashable, Mapping, MutableMapping, Sequence

import numpy as np
from qtpy import QtCore, QtWidgets

if typing.TYPE_CHECKING:
    import erlab


_MISSING = object()

_DetailRow = tuple[str, str, str]
_StatePath = tuple[Hashable, ...]
_HistoryEntryRow = tuple[
    int, int, str, tuple[_DetailRow, ...], bool, datetime.datetime | None
]


@dataclasses.dataclass(slots=True, eq=False)
class HistoryEntry:
    before_state: dict[str, typing.Any]
    after_state: dict[str, typing.Any]
    changed_paths: frozenset[_StatePath]
    transaction_id: str | None
    created_at: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now().astimezone()
    )


_DETAIL_LABELS = {
    "color.cmap": "Colormap",
    "color.gamma": "Gamma",
    "color.high_contrast": "High contrast",
    "color.levels": "Levels",
    "color.levels_locked": "Lock levels",
    "color.reverse": "Reverse colormap",
    "color.zero_centered": "Zero centered",
    "slice.snap_to_data": "Snap to pixels",
    "slice.twin_coord_names": "Associated coordinates",
    "slice.cursor_color_params": "Cursor color mapping",
    "cursor_colors": "Cursor colors",
    "controls_visible": "Controls visible",
}


def _normalize_value(value: typing.Any) -> typing.Any:
    if isinstance(value, np.ndarray):
        return [_normalize_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, numbers.Number):
        return value
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
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


def _format_value(value: typing.Any) -> str:
    value = _normalize_value(value)
    if value is _MISSING:
        text = "Not set"
    elif value is None:
        text = "None"
    elif isinstance(value, bool):
        text = "On" if value else "Off"
    elif isinstance(value, str):
        text = value or "Empty"
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        text = "[" + ", ".join(_format_value(v) for v in value) + "]"
    elif isinstance(value, Mapping):
        text = (
            "{" + ", ".join(f"{k}: {_format_value(v)}" for k, v in value.items()) + "}"
        )
    else:
        text = repr(value)
    if len(text) > 96:
        text = f"{text[:93]}..."
    return text


def _change_detail(label: str, old: typing.Any, new: typing.Any) -> _DetailRow:
    return (label, _format_value(old), _format_value(new))


def _detail_note(text: str) -> _DetailRow:
    return (text, "", "")


def _detail_rows_to_lines(details: Sequence[_DetailRow]) -> tuple[str, ...]:
    lines: list[str] = []
    for label, before, after in details:
        if before or after:
            lines.append(f"{label} changed from {before} to {after}")
        else:
            lines.append(label)
    return tuple(lines)


def _format_history_datetime(timestamp: datetime.datetime | None) -> str:
    if timestamp is None:
        return ""
    return timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _changed_paths(
    prev_flat: Mapping[str, typing.Any], curr_flat: Mapping[str, typing.Any]
) -> set[str]:
    paths = set(prev_flat) | set(curr_flat)
    return {
        path
        for path in paths
        if not (path == "splitter_sizes" or path.startswith("splitter_sizes."))
        and prev_flat.get(path, _MISSING) != curr_flat.get(path, _MISSING)
    }


def changed_paths(
    prev: Mapping[str, typing.Any], curr: Mapping[str, typing.Any]
) -> frozenset[_StatePath]:
    paths: set[_StatePath] = set()

    def visit(old: typing.Any, new: typing.Any, prefix: _StatePath = ()) -> None:
        if prefix and prefix[0] == "splitter_sizes":
            return

        if isinstance(old, Mapping) and isinstance(new, Mapping):
            for key in set(old) | set(new):
                visit(
                    old.get(key, _MISSING),
                    new.get(key, _MISSING),
                    (*prefix, key),
                )
            return
        if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            for i, (old_item, new_item) in enumerate(zip(old, new, strict=True)):
                visit(old_item, new_item, (*prefix, i))
            return
        old = _normalize_value(old)
        new = _normalize_value(new)
        if prefix and old != new:
            paths.add(prefix)

    visit(prev, curr)
    return frozenset(paths)


def _path_value(
    state: Mapping[str, typing.Any], path: _StatePath, *, normalize: bool = True
) -> typing.Any:
    value: typing.Any = state
    for part in path:
        if isinstance(value, Mapping):
            if part not in value:
                return _MISSING
            value = value[part]
            continue
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if not isinstance(part, int) or isinstance(part, bool):
                return _MISSING
            index = part
            if index < 0 or index >= len(value):
                return _MISSING
            value = value[index]
            continue
        return _MISSING
    if normalize:
        return _normalize_value(value)
    return value


def _set_path(
    state: dict[str, typing.Any], path: _StatePath, value: typing.Any
) -> None:
    if not path:
        return

    target: typing.Any = state
    for i, part in enumerate(path[:-1]):
        next_part = path[i + 1]
        if isinstance(target, MutableMapping):
            child = target.get(part, _MISSING)
            if not isinstance(child, (dict, list)):
                if value is _MISSING:
                    return
                child = (
                    []
                    if isinstance(next_part, int) and not isinstance(next_part, bool)
                    else {}
                )
                target[part] = child
            target = child
            continue
        if isinstance(target, list):
            if not isinstance(part, int) or isinstance(part, bool):
                return
            index = part
            if index < 0 or index >= len(target):
                return
            child = target[index]
            if not isinstance(child, (dict, list)):
                if value is _MISSING:
                    return
                child = (
                    []
                    if isinstance(next_part, int) and not isinstance(next_part, bool)
                    else {}
                )
                target[index] = child
            target = child
            continue
        return

    last = path[-1]
    if isinstance(target, MutableMapping):
        if value is _MISSING:
            target.pop(last, None)
        else:
            target[last] = copy.deepcopy(value)
    elif isinstance(target, list):
        if not isinstance(last, int) or isinstance(last, bool):
            return
        index = last
        if index < 0 or index >= len(target):
            return
        if value is _MISSING:
            target.pop(index)
        else:
            target[index] = copy.deepcopy(value)


def entry_matches_current(
    entry: HistoryEntry, current: Mapping[str, typing.Any], *, undo: bool
) -> bool:
    expected = entry.after_state if undo else entry.before_state
    return all(
        _normalize_value(_path_value(current, path)) == _path_value(expected, path)
        for path in entry.changed_paths
    )


def patch_state(
    base: Mapping[str, typing.Any],
    source: Mapping[str, typing.Any],
    paths: frozenset[_StatePath],
) -> dict[str, typing.Any]:
    patched = copy.deepcopy(dict(base))
    for path in paths:
        _set_path(patched, path, _path_value(source, path, normalize=False))
    return patched


def _consume(changes: set[str], *paths: str, prefixes: Sequence[str] = ()) -> None:
    for path in paths:
        changes.discard(path)
    for path in tuple(changes):
        if any(path == prefix or path.startswith(f"{prefix}.") for prefix in prefixes):
            changes.discard(path)


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _state_value(state: Mapping[str, typing.Any], key: str) -> typing.Any:
    return state.get(key, _MISSING)


def _safe_len(value: typing.Any) -> int:
    return len(value) if isinstance(value, Sequence) else 0


def _cursor_detail_rows(
    prev_indices: typing.Any,
    curr_indices: typing.Any,
    prev_values: typing.Any,
    curr_values: typing.Any,
) -> tuple[_DetailRow, ...]:
    details: list[_DetailRow] = []
    if not isinstance(prev_indices, Sequence) or not isinstance(curr_indices, Sequence):
        return ()

    for i, (old_index, new_index) in enumerate(
        zip(prev_indices, curr_indices, strict=False), start=1
    ):
        old_value = prev_values[i - 1] if i <= _safe_len(prev_values) else _MISSING
        new_value = curr_values[i - 1] if i <= _safe_len(curr_values) else _MISSING
        if old_index == new_index and old_value == new_value:
            continue

        if old_index != new_index:
            details.append(_change_detail(f"Cursor {i} index", old_index, new_index))
        if old_value != new_value:
            details.append(_change_detail(f"Cursor {i} value", old_value, new_value))
    return tuple(details)


def _plot_state_changes(
    prev_plot_states: typing.Any, curr_plot_states: typing.Any
) -> tuple[list[str], list[_DetailRow]]:
    summaries: list[str] = []
    details: list[_DetailRow] = []

    if not isinstance(prev_plot_states, Sequence) or not isinstance(
        curr_plot_states, Sequence
    ):
        return ["Plot view changed"], [_detail_note("Plot state changed")]

    old_roi_count = 0
    new_roi_count = 0
    roi_changed = False
    guideline_changed = False
    view_changed = len(prev_plot_states) != len(curr_plot_states)

    for old, new in zip(prev_plot_states, curr_plot_states, strict=False):
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            view_changed = True
            continue

        old_rois = old.get("roi_states", [])
        new_rois = new.get("roi_states", [])
        old_roi_count += _safe_len(old_rois)
        new_roi_count += _safe_len(new_rois)
        roi_changed = roi_changed or old_rois != new_rois

        old_guideline = old.get("guideline_state", None)
        new_guideline = new.get("guideline_state", None)
        guideline_changed = guideline_changed or old_guideline != new_guideline

        view_keys = {
            "vb_aspect_locked",
            "vb_autorange",
        }
        view_changed = view_changed or any(
            old.get(key, _MISSING) != new.get(key, _MISSING) for key in view_keys
        )

    if roi_changed:
        summaries.append("ROI changed")
        details.append(_change_detail("ROI count", old_roi_count, new_roi_count))
    if guideline_changed:
        summaries.append("Guidelines changed")
        details.append(_detail_note("Guideline state changed"))
    if view_changed:
        summaries.append("Plot view changed")
        details.append(_detail_note("Plot view state changed"))

    return summaries, details


def parse_change(key: str, old, new) -> str:
    match key:
        case "cursor_colors":
            return "Cursor colors changed"
        case "slice.indices" | "slice.values":
            return "Cursor moved"
        case "slice.bins":
            return "Bins changed"
        case "slice.dims":
            return "Transposed"
        case "slice.snap_to_data":
            if new is _MISSING or old is _MISSING:
                return "Snap to pixels changed"
            return "Snap to pixels enabled" if new else "Snap to pixels disabled"
        case "slice.twin_coord_names":
            return "Associated coordinates changed"
        case "slice.cursor_color_params":
            return "Cursor color mapping changed"
        case "controls_visible":
            if new is _MISSING or old is _MISSING:
                return "Controls visibility changed"
            return "Controls shown" if new else "Controls hidden"
        case "color.reverse":
            if new is _MISSING or old is _MISSING:
                return "Colormap reversal changed"
            return "Colormap reversed" if new else "Colormap unreversed"
        case "color.cmap":
            if new is _MISSING:
                return "Colormap changed"
            return f"Changed colormap to {new}"
        case "color.gamma" | "color.high_contrast" | "color.zero_centered":
            return "Colormap options changed"
        case "color.levels_locked":
            if new is _MISSING or old is _MISSING:
                return "Colormap levels lock changed"
            return "Colormap levels locked" if new else "Colormap levels unlocked"
        case "color.levels":
            return "Colormap levels changed"
        case _:
            return f"{key} changed"


def _describe_state_change_rows(
    prev: Mapping[str, typing.Any], curr: Mapping[str, typing.Any]
) -> tuple[str, tuple[_DetailRow, ...]] | None:
    prev_flat, curr_flat = _flatten_state(prev), _flatten_state(curr)
    changes = _changed_paths(prev_flat, curr_flat)
    if not changes:
        return None

    summaries: list[str] = []
    details: list[_DetailRow] = []

    prev_indices = prev_flat.get("slice.indices", [])
    curr_indices = curr_flat.get("slice.indices", [])
    prev_values = prev_flat.get("slice.values", [])
    curr_values = curr_flat.get("slice.values", [])
    old_ncursors = _safe_len(prev_indices)
    new_ncursors = _safe_len(curr_indices)

    transposed = (
        "slice.dims" in changes
        and isinstance(prev_flat.get("slice.dims"), Sequence)
        and isinstance(curr_flat.get("slice.dims"), Sequence)
        and set(prev_flat["slice.dims"]) == set(curr_flat["slice.dims"])
    )

    if transposed:
        summaries.append("Transposed")
        details.append(
            _change_detail(
                "Dimensions",
                _state_value(prev_flat, "slice.dims"),
                _state_value(curr_flat, "slice.dims"),
            )
        )
        _consume(changes, "slice.dims", "slice.indices", "slice.values")

    if old_ncursors > new_ncursors:
        summaries.append("Cursor removed")
        details.append(_change_detail("Cursors", old_ncursors, new_ncursors))
        _consume(
            changes,
            "current_cursor",
            "cursor_colors",
            "slice.indices",
            "slice.values",
            "slice.bins",
        )
    elif old_ncursors < new_ncursors:
        summaries.append("Cursor added")
        details.append(_change_detail("Cursors", old_ncursors, new_ncursors))
        _consume(
            changes,
            "current_cursor",
            "cursor_colors",
            "slice.indices",
            "slice.values",
            "slice.bins",
        )
    elif "current_cursor" in changes:
        summaries.append("Cursor changed")
        details.append(
            _change_detail(
                "Active cursor",
                _state_value(prev_flat, "current_cursor"),
                _state_value(curr_flat, "current_cursor"),
            )
        )
        _consume(changes, "current_cursor")

    if not transposed and {"slice.indices", "slice.values"} & changes:
        indices_changed = prev_indices != curr_indices
        values_changed = prev_values != curr_values
        if indices_changed:
            _append_unique(summaries, "Cursor moved")
        elif values_changed:
            _append_unique(summaries, "Cursor moved within pixel")
        details.extend(
            _cursor_detail_rows(prev_indices, curr_indices, prev_values, curr_values)
        )
        _consume(changes, "slice.indices", "slice.values")

    if "slice.bins" in changes:
        summaries.append("Bins changed")
        details.append(
            _change_detail(
                "Bins",
                _state_value(prev_flat, "slice.bins"),
                _state_value(curr_flat, "slice.bins"),
            )
        )
        _consume(changes, "slice.bins")

    for key in (
        "slice.snap_to_data",
        "slice.twin_coord_names",
        "slice.cursor_color_params",
        "cursor_colors",
        "controls_visible",
    ):
        if key in changes:
            old_value = _state_value(prev_flat, key)
            new_value = _state_value(curr_flat, key)
            summaries.append(parse_change(key, old_value, new_value))
            details.append(_change_detail(_DETAIL_LABELS[key], old_value, new_value))
            _consume(changes, key)

    for key in (
        "color.cmap",
        "color.reverse",
        "color.gamma",
        "color.high_contrast",
        "color.zero_centered",
        "color.levels_locked",
        "color.levels",
    ):
        if key in changes:
            old_value = _state_value(prev_flat, key)
            new_value = _state_value(curr_flat, key)
            summaries.append(parse_change(key, old_value, new_value))
            details.append(_change_detail(_DETAIL_LABELS[key], old_value, new_value))
            _consume(changes, key)

    if any(
        path == "manual_limits" or path.startswith("manual_limits.") for path in changes
    ):
        summaries.append("View limits changed")
        details.append(_detail_note("Manual view limits changed"))
        _consume(changes, "manual_limits", prefixes=("manual_limits",))

    if any(
        path == "axis_inversions" or path.startswith("axis_inversions.")
        for path in changes
    ):
        summaries.append("Axis inversion changed")
        details.append(_detail_note("Axis inversion changed"))
        _consume(changes, "axis_inversions", prefixes=("axis_inversions",))

    if any(
        path == "filter_operation" or path.startswith("filter_operation.")
        for path in changes
    ):
        old_filter = prev.get("filter_operation", _MISSING)
        new_filter = curr.get("filter_operation", _MISSING)
        if new_filter is _MISSING or new_filter is None:
            summaries.append("Filter cleared")
        elif old_filter is _MISSING or old_filter is None:
            summaries.append("Filter applied")
        else:
            summaries.append("Filter changed")
        details.append(_detail_note("Filter operation changed"))
        _consume(changes, "filter_operation", prefixes=("filter_operation",))

    if "plotitem_states" in changes:
        plot_summaries, plot_details = _plot_state_changes(
            prev_flat.get("plotitem_states", []), curr_flat.get("plotitem_states", [])
        )
        summaries.extend(plot_summaries)
        details.extend(plot_details)
        _consume(changes, "plotitem_states")

    if {"file_path", "load_func"} & changes:
        summaries.append("Source changed")
        details.extend(
            _change_detail(
                key, _state_value(prev_flat, key), _state_value(curr_flat, key)
            )
            for key in ("file_path", "load_func")
            if key in changes
        )
        _consume(changes, "file_path", "load_func")

    if changes:
        if not summaries:
            summaries.append("State changed")
        details.append(_detail_note(f"Changed paths: {', '.join(sorted(changes))}"))

    if not summaries:
        return None
    return ", ".join(dict.fromkeys(summaries)), tuple(dict.fromkeys(details))


def describe_state_change(
    prev: Mapping[str, typing.Any], curr: Mapping[str, typing.Any]
) -> tuple[str, tuple[str, ...]] | None:
    change = _describe_state_change_rows(prev, curr)
    if change is None:
        return None
    summary, details = change
    return summary, _detail_rows_to_lines(details)


def describe_state_diff(
    prev: Mapping[str, typing.Any], curr: Mapping[str, typing.Any]
) -> str | None:
    change = describe_state_change(prev, curr)
    if change is None:
        return None
    return change[0]


def _history_entries(
    slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea,
) -> list[_HistoryEntryRow]:
    history_entry_changes = getattr(slicer_area, "history_entry_changes", None)
    if callable(history_entry_changes):
        entry_rows: list[_HistoryEntryRow] = []
        for (
            i,
            relative_index,
            prev_state,
            this_state,
            is_current,
            created_at,
        ) in reversed(history_entry_changes(finalize_pending=False)):
            change = _describe_state_change_rows(prev_state, this_state)
            if change is None:
                if not is_current:
                    continue
                summary, details = "Current state", ()
            else:
                summary, details = change
            entry_rows.append(
                (i, relative_index, summary, details, is_current, created_at)
            )
        return entry_rows

    states, current_index = slicer_area.history_states()
    entries: list[_HistoryEntryRow] = []
    for i in range(len(states) - 1, 0, -1):
        prev_state, this_state = states[i - 1], states[i]
        change = _describe_state_change_rows(prev_state, this_state)
        if change is None:
            if i != current_index:
                continue
            summary, details = "Current state", ()
        else:
            summary, details = change
        is_current = i == current_index
        entries.append((i, i - current_index, summary, details, is_current, None))
    if current_index == 0 and len(states) > 1:
        entries.append((0, 0, "Current state", (), True, None))
    return entries


class HistoryDialog(QtWidgets.QDialog):
    def __init__(
        self,
        slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._slicer_area = weakref.ref(slicer_area)
        self.setObjectName("itool_history_dialog")
        self.setWindowTitle("History")
        self.setModal(False)
        self.resize(460, 210)
        self.setSizeGripEnabled(False)
        self._fitting_height = False

        self._history_list = QtWidgets.QListWidget()
        self._history_list.setObjectName("itool_history_list")
        self._history_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._history_list.installEventFilter(self)
        self._history_list.setMinimumHeight(140)

        self._details = QtWidgets.QWidget()
        self._details.setObjectName("itool_history_details")
        self._details.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._details_layout = QtWidgets.QGridLayout(self._details)
        self._details_layout.setContentsMargins(0, 0, 0, 0)
        self._details_layout.setHorizontalSpacing(16)
        self._details_layout.setVerticalSpacing(6)
        self._details_layout.setColumnMinimumWidth(0, 96)
        self._details_layout.setColumnMinimumWidth(1, 80)
        self._details_layout.setColumnMinimumWidth(2, 80)
        self._details_layout.setColumnStretch(1, 1)
        self._details_layout.setColumnStretch(2, 1)

        self._details_panel = QtWidgets.QGroupBox("Details")
        self._details_panel.setObjectName("itool_history_details_group")
        self._details_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        details_layout = QtWidgets.QVBoxLayout(self._details_panel)
        details_layout.addWidget(self._details)

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self._button_box.setObjectName("itool_history_button_box")
        self._go_to_btn = QtWidgets.QPushButton("Restore State")
        self._go_to_btn.setObjectName("itool_history_go_to_button")
        self._button_box.addButton(
            self._go_to_btn, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self._close_btn = self._button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        if self._close_btn is not None:
            self._close_btn.setObjectName("itool_history_close_button")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._history_list)
        layout.addWidget(self._details_panel)
        layout.addWidget(self._button_box)

        self._history_list.currentItemChanged.connect(self._selection_changed)
        self._history_list.itemDoubleClicked.connect(self._go_to_item)
        self._go_to_btn.clicked.connect(self._go_to_selected)
        self._button_box.rejected.connect(self.close)
        self.slicer_area.sigHistoryChanged.connect(self.refresh)

        self.refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._fitting_height:
            QtCore.QTimer.singleShot(0, self._fit_height_to_content)

    @property
    def slicer_area(self) -> erlab.interactive.imagetool.viewer.ImageSlicerArea:
        slicer_area = self._slicer_area()
        if slicer_area:
            return slicer_area
        raise LookupError("Parent was destroyed")

    def eventFilter(
        self, obj: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if (
            obj is self._history_list
            and event is not None
            and event.type() == QtCore.QEvent.Type.KeyPress
        ):
            key_event = typing.cast("typing.Any", event)
            if key_event.key() in (
                QtCore.Qt.Key.Key_Return,
                QtCore.Qt.Key.Key_Enter,
            ):
                self._go_to_selected()
                return True
        return super().eventFilter(obj, event)

    @QtCore.Slot()
    def refresh(self) -> None:
        self._history_list.clear()
        entries = _history_entries(self.slicer_area)
        if not entries:
            self._set_detail_rows((_detail_note("No history recorded yet."),))
            self._go_to_btn.setEnabled(False)
            item = QtWidgets.QListWidgetItem("No history recorded yet")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, None)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, ())
            self._history_list.addItem(item)
            self._history_list.setCurrentItem(item)
            self._update_history_list_height()
            return

        current_item: QtWidgets.QListWidgetItem | None = None
        for _, relative_index, summary, details, is_current, created_at in entries:
            timestamp = _format_history_datetime(created_at)
            text = f"Current: {summary}" if is_current else summary
            if timestamp:
                text = f"{text}\n{timestamp}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, relative_index)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, details)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 2, timestamp)
            if timestamp:
                item.setToolTip(timestamp)
            if is_current:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._history_list.addItem(item)
            if is_current:
                current_item = item

        self._history_list.setCurrentItem(current_item or self._history_list.item(0))
        self._update_history_list_height()
        self._selection_changed()

    @QtCore.Slot()
    def _selection_changed(self, *_args) -> None:
        item = self._history_list.currentItem()
        if item is None:
            self._clear_detail_rows()
            self._go_to_btn.setEnabled(False)
            return

        details = item.data(QtCore.Qt.ItemDataRole.UserRole + 1)
        if details:
            self._set_detail_rows(details)
        elif item.data(QtCore.Qt.ItemDataRole.UserRole) is None:
            self._set_detail_rows((_detail_note("No history recorded yet."),))
        else:
            self._set_detail_rows((_detail_note("No additional details."),))
        self._go_to_btn.setEnabled(self._selected_relative_index() not in (None, 0))

    def _set_detail_rows(self, details: Sequence[_DetailRow]) -> None:
        self._clear_detail_rows()
        row = 0
        if any(before or after for _, before, after in details):
            self._details_layout.addWidget(self._detail_heading("Before"), row, 1)
            self._details_layout.addWidget(self._detail_heading("After"), row, 2)
            row += 1
        for label, before, after in details:
            self._add_detail_row(row, label, before, after)
            row += 1
        self._fit_height_to_content()
        QtCore.QTimer.singleShot(0, self._fit_height_to_content)

    def _update_history_list_height(self) -> None:
        row_count = max(1, min(self._history_list.count(), 8))
        row_height = self._history_list.sizeHintForRow(0)
        if row_height <= 0:
            row_height = self.fontMetrics().height() + 6
        height = row_count * row_height + 2 * self._history_list.frameWidth() + 4
        self._history_list.setFixedHeight(height)

    def _fit_height_to_content(self) -> None:
        main_layout = self.layout()
        if main_layout is None:
            return
        self._fitting_height = True
        try:
            detail_width = max(1, self._details.width())
            detail_height = self._details_layout.sizeHint().height()
            if self._details_layout.hasHeightForWidth():
                detail_height = max(
                    detail_height,
                    self._details_layout.heightForWidth(detail_width),
                )
            self._details.setFixedHeight(detail_height)
            details_panel_layout = self._details_panel.layout()
            if details_panel_layout is not None:
                details_panel_layout.activate()
            details_margins = (
                details_panel_layout.contentsMargins()
                if details_panel_layout is not None
                else QtCore.QMargins()
            )
            detail_top = max(
                self._details.y(),
                self.fontMetrics().height() + details_margins.top() + 4,
            )
            panel_height = detail_height + detail_top + details_margins.bottom()
            self._details_panel.setFixedHeight(panel_height)
            self._details_layout.activate()
            main_layout.activate()
            main_margins = main_layout.contentsMargins()
            main_spacing = max(0, main_layout.spacing())
            height = (
                main_margins.top()
                + self._history_list.height()
                + main_spacing
                + panel_height
                + main_spacing
                + self._button_box.sizeHint().height()
                + main_margins.bottom()
            )
            self.setFixedHeight(height)
            self.resize(self.width(), height)
        finally:
            self._fitting_height = False

    def _clear_detail_rows(self) -> None:
        while self._details_layout.count():
            item = self._details_layout.takeAt(0)
            if item is None:
                continue
            if widget := item.widget():
                widget.setParent(None)
                widget.deleteLater()

    def _add_detail_row(self, row: int, label: str, before: str, after: str) -> None:
        label_widget = self._detail_label(label)
        label_widget.setObjectName("itool_history_detail_row")
        label_widget.setProperty("detail", label)
        label_widget.setProperty("before", before)
        label_widget.setProperty("after", after)
        if before or after:
            self._details_layout.addWidget(label_widget, row, 0)
            self._details_layout.addWidget(self._value_label(before), row, 1)
            self._details_layout.addWidget(self._value_label(after), row, 2)
        else:
            self._details_layout.addWidget(label_widget, row, 0, 1, 3)

    def _detail_heading(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("itool_history_detail_heading")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        return label

    def _detail_label(self, text: str) -> QtWidgets.QLabel:
        label_widget = QtWidgets.QLabel(text)
        label_widget.setObjectName("itool_history_detail_label")
        label_widget.setWordWrap(True)
        label_widget.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        label_widget.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        return label_widget

    def _value_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("itool_history_detail_value")
        label.setWordWrap(True)
        label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        return label

    def _selected_relative_index(self) -> int | None:
        item = self._history_list.currentItem()
        if item is None:
            return None
        value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return value if isinstance(value, int) else None

    def _go_to_item(self, item: QtWidgets.QListWidgetItem) -> None:
        self._history_list.setCurrentItem(item)
        self._go_to_selected()

    @QtCore.Slot()
    def _go_to_selected(self) -> None:
        relative_index = self._selected_relative_index()
        if relative_index is None or relative_index == 0:
            return
        self.slicer_area.go_to_history_index(relative_index)


class HistoryMenu(QtWidgets.QMenu):
    def __init__(
        self,
        slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._slicer_area = weakref.ref(slicer_area)
        self.aboutToShow.connect(self.update_actions)
        self._actions: list[QtWidgets.QAction] = []
        self._dialog: HistoryDialog | None = None
        self.slicer_area.sigHistoryChanged.connect(self._history_changed)

    @property
    def slicer_area(self) -> erlab.interactive.imagetool.viewer.ImageSlicerArea:
        """Get the parent main window."""
        slicer_area = self._slicer_area()
        if slicer_area:
            return slicer_area
        raise LookupError("Parent was destroyed")

    @QtCore.Slot()
    def _history_changed(self) -> None:
        if self.isVisible():
            self.update_actions()

    @QtCore.Slot()
    def show_history(self) -> None:
        if self._dialog is None:
            slicer_area = self.slicer_area
            self._dialog = HistoryDialog(slicer_area, slicer_area.window())
        self._dialog.refresh()
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def update_actions(self) -> None:
        self.clear()
        self._actions.clear()

        show_action = QtWidgets.QAction("Show History", self)
        show_action.setObjectName("itool_history_show_action")
        show_action.setData({"kind": "show_history"})
        show_action.triggered.connect(self.show_history)
        self.addAction(show_action)
        self.addSeparator()
        self._actions.append(show_action)

        entries = _history_entries(self.slicer_area)
        if not entries:
            action = QtWidgets.QAction("No history recorded yet")
            action.setObjectName("itool_history_empty_action")
            action.setData({"kind": "empty"})
            action.setEnabled(False)
            self.addAction(action)
            self._actions.append(action)
            return

        for state_index, relative_index, summary, _, is_current, _ in entries:
            marker = "•" if is_current else " "
            action = QtWidgets.QAction(f"{marker} {summary}")
            action.setObjectName(f"itool_history_action_{state_index}")
            action.setData(relative_index)
            action.triggered.connect(
                lambda *, idx=relative_index: self.slicer_area.go_to_history_index(idx)
            )
            self.addAction(action)
            self._actions.append(action)
