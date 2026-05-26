from __future__ import annotations

import copy
import unittest.mock

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab.interactive.utils
from erlab.interactive.imagetool import ImageTool, _history
from erlab.interactive.imagetool.controls import (
    ItoolBinningControls,
    ItoolColormapControls,
    ItoolCrosshairControls,
)


def _base_state() -> dict:
    return {
        "current_cursor": 0,
        "cursor_colors": ["#000000"],
        "color": {
            "cmap": "viridis",
            "gamma": 1.0,
            "reverse": False,
            "high_contrast": False,
            "zero_centered": False,
            "levels_locked": False,
        },
        "slice": {
            "indices": [[np.int64(0), np.int64(0)]],
            "values": [[np.float64(0.0), np.float64(1.0)]],
            "bins": [[1, 1]],
            "dims": ["x", "y"],
            "snap_to_data": False,
            "twin_coord_names": (),
            "cursor_color_params": None,
        },
        "manual_limits": {},
        "axis_inversions": {},
        "file_path": None,
        "load_func": None,
        "controls_visible": True,
        "plotitem_states": [
            {
                "vb_aspect_locked": False,
                "vb_autorange": (True, True),
                "roi_states": [],
            }
        ],
    }


class _DummySlicerArea(QtCore.QObject):
    sigHistoryChanged = QtCore.Signal()

    def __init__(self, states, current_index: int):
        super().__init__()
        self._states = states
        self._current_index = current_index
        self.go_to_history_calls: list[int] = []

    def history_states(self):
        return self._states, self._current_index

    def go_to_history_index(self, idx: int) -> None:
        self.go_to_history_calls.append(idx)

    def window(self) -> QtWidgets.QWidget | None:
        return None


@pytest.mark.parametrize(
    ("key", "new", "expected"),
    [
        ("cursor_colors", None, "Cursor colors changed"),
        ("slice.indices", None, "Cursor moved"),
        ("slice.bins", None, "Bins changed"),
        ("slice.dims", None, "Transposed"),
        ("slice.snap_to_data", True, "Snap to pixels enabled"),
        ("slice.twin_coord_names", None, "Associated coordinates changed"),
        ("slice.cursor_color_params", None, "Cursor color mapping changed"),
        ("controls_visible", True, "Controls shown"),
        ("controls_visible", False, "Controls hidden"),
        ("color.reverse", True, "Colormap reversed"),
        ("color.cmap", "magma", "Changed colormap to magma"),
        ("color.gamma", 2.0, "Colormap options changed"),
        ("color.high_contrast", True, "Colormap options changed"),
        ("color.zero_centered", True, "Colormap options changed"),
        ("color.levels_locked", True, "Colormap levels locked"),
        ("color.levels_locked", False, "Colormap levels unlocked"),
        ("color.levels", [0, 1], "Colormap levels changed"),
        ("other.key", None, "other.key changed"),
    ],
)
def test_parse_change(key, new, expected):
    assert _history.parse_change(key, None, new) == expected


@pytest.mark.parametrize(
    ("key", "old", "new", "expected"),
    [
        ("slice.snap_to_data", _history._MISSING, True, "Snap to pixels changed"),
        (
            "controls_visible",
            True,
            _history._MISSING,
            "Controls visibility changed",
        ),
        ("color.reverse", _history._MISSING, False, "Colormap reversal changed"),
        ("color.cmap", "viridis", _history._MISSING, "Colormap changed"),
        (
            "color.levels_locked",
            _history._MISSING,
            True,
            "Colormap levels lock changed",
        ),
    ],
)
def test_parse_change_reports_missing_values_generically(key, old, new, expected):
    assert _history.parse_change(key, old, new) == expected


def test_history_value_formatting_handles_edge_values() -> None:
    long_text = "x" * 120

    assert _history._format_value(_history._MISSING) == "Not set"
    assert _history._format_value(None) == "None"
    assert (
        _history._format_value({"flag": True, "empty": ""})
        == "{flag: On, empty: Empty}"
    )
    assert _history._format_value(long_text) == f"{long_text[:93]}..."


def test_changed_paths_ignores_splitter_sizes() -> None:
    before = {"splitter_sizes": [1, 2], "value": 1}
    after = {"splitter_sizes": [3, 4], "value": 2}

    assert _history.changed_paths(before, after) == frozenset({("value",)})


@pytest.mark.parametrize(
    ("state", "path"),
    [
        ({}, ("missing",)),
        ({"items": [1]}, ("items", "bad")),
        ({"items": [1]}, ("items", -1)),
        ({"items": [1]}, ("items", 2)),
        ({"value": 1}, ("value", "child")),
    ],
)
def test_path_value_returns_missing_for_invalid_paths(state, path) -> None:
    assert _history._path_value(state, path) is _history._MISSING


def test_path_value_normalizes_returned_value() -> None:
    value = _history._path_value({"values": np.array([1, 2])}, ("values",))

    assert value == [1, 2]


def test_patch_state_ignores_empty_path() -> None:
    state = {"value": 1}

    assert _history.patch_state(state, {"value": 2}, frozenset({()})) == state


def test_patch_state_creates_and_removes_nested_mapping_values() -> None:
    current: dict[str, object] = {}
    patched = _history.patch_state(
        current,
        {"outer": {"inner": 1}},
        frozenset({("outer", "inner")}),
    )
    assert patched == {"outer": {"inner": 1}}

    patched = _history.patch_state(
        {"outer": {"inner": 1, "kept": 2}},
        {"outer": {}},
        frozenset({("outer", "inner")}),
    )
    assert patched == {"outer": {"kept": 2}}


def test_patch_state_handles_invalid_and_missing_sequence_paths() -> None:
    state = {"items": [{"old": 1}]}
    before = {"items": [{"new": 2}]}

    assert _history.patch_state(state, before, frozenset({("items", "bad")})) == state
    assert (
        _history.patch_state(state, {}, frozenset({("items", 0, "missing")})) == state
    )
    assert _history.patch_state(state, before, frozenset({("items", -1)})) == state
    assert (
        _history.patch_state(state, before, frozenset({("items", 2, "new")})) == state
    )
    assert (
        _history.patch_state(state, before, frozenset({("items", 0, "old", 0)}))
        == state
    )
    assert _history.patch_state(state, before, frozenset({("items", 0, "new")})) == {
        "items": [{"old": 1, "new": 2}]
    }
    assert _history.patch_state(
        {"items": [0]},
        {"items": [[1]]},
        frozenset({("items", 0, 0)}),
    ) == {"items": [[]]}
    assert _history.patch_state(
        {"items": [{"old": 1, "new": 2}]},
        {"items": [{"old": 1}]},
        frozenset({("items", 0, "new")}),
    ) == {"items": [{"old": 1}]}
    assert _history.patch_state(
        {"items": [0]},
        {"items": []},
        frozenset({("items", 0)}),
    ) == {"items": []}


def test_set_path_handles_invalid_intermediate_targets() -> None:
    state: dict[str, object] = {"items": [1], "value": 1}

    _history._set_path(state, ("items", "bad", "value"), 2)
    _history._set_path(state, ("items", 0, "missing"), _history._MISSING)
    _history._set_path(state, ("value", "child"), 2)

    assert state == {"items": [1], "value": {"child": 2}}


def test_entry_matches_current_checks_before_or_after_state() -> None:
    entry = _history.HistoryEntry(
        {"value": np.array([1, 2])},
        {"value": np.array([1, 3])},
        frozenset({("value",)}),
        None,
    )

    assert _history.entry_matches_current(entry, {"value": [1, 3]}, undo=True)
    assert _history.entry_matches_current(entry, {"value": [1, 2]}, undo=False)
    assert not _history.entry_matches_current(entry, {"value": [9]}, undo=True)


def test_cursor_detail_rows_handle_invalid_and_changed_values() -> None:
    assert _history._cursor_detail_rows(1, [[0]], [], []) == ()
    assert _history._cursor_detail_rows([[0]], 1, [], []) == ()
    assert _history._cursor_detail_rows([[0]], [[0]], [[1.0]], [[1.0]]) == ()
    assert _history._cursor_detail_rows([[0]], [[1]], [], [[2.0]]) == (
        ("Cursor 1 index", "[0]", "[1]"),
        ("Cursor 1 value", "Not set", "[2.0]"),
    )


def test_plot_state_changes_handles_non_sequence_and_non_mapping_states() -> None:
    assert _history._plot_state_changes(1, []) == (
        ["Plot view changed"],
        [("Plot state changed", "", "")],
    )
    summaries, details = _history._plot_state_changes([{}], [1])

    assert summaries == ["Plot view changed"]
    assert details == [("Plot view state changed", "", "")]


def test_describe_state_change_reports_same_pixel_cursor_move():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["slice"]["values"] = [[0.5, 1.0]]

    label, details = _history.describe_state_change(prev, curr)

    assert label == "Cursor moved within pixel"
    assert details == ("Cursor 1 value changed from [0.0, 1.0] to [0.5, 1.0]",)
    assert all("->" not in detail for detail in details)
    assert _history.describe_state_diff(prev, curr) == label


@pytest.mark.parametrize(
    ("prev_indices", "curr_indices", "prev_cursor", "curr_cursor", "expected"),
    [
        (
            np.array([[0, 0], [1, 1]]),
            np.array([[0, 0]]),
            0,
            0,
            "Cursor removed",
        ),
        (
            [[0, 0]],
            [[0, 0], [1, 1]],
            0,
            1,
            "Cursor added",
        ),
        (
            [[0, 0]],
            [[0, 2]],
            0,
            0,
            "Cursor moved",
        ),
    ],
)
def test_describe_state_diff_cursor_changes(
    prev_indices, curr_indices, prev_cursor, curr_cursor, expected
):
    prev = _base_state()
    curr = copy.deepcopy(prev)
    prev["slice"]["indices"] = prev_indices
    prev["slice"]["values"] = [[float(i) for i in row] for row in prev_indices]
    prev["current_cursor"] = prev_cursor
    curr["slice"]["indices"] = curr_indices
    curr["slice"]["values"] = [[float(i) for i in row] for row in curr_indices]
    curr["current_cursor"] = curr_cursor

    assert _history.describe_state_diff(prev, curr) == expected


def test_describe_state_change_reports_colormap_and_dims_without_cursor_noise():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["color"]["reverse"] = True
    curr["slice"]["dims"] = ["y", "x"]
    curr["slice"]["indices"] = [[np.int64(1), np.int64(0)]]
    curr["slice"]["values"] = [[2.0, 3.0]]

    label, details = _history.describe_state_change(prev, curr)

    assert "Transposed" in label
    assert "Colormap reversed" in label
    assert "Cursor moved" not in label
    assert details[0] == "Dimensions changed from [x, y] to [y, x]"


def test_describe_state_change_reports_snap_toggle():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["slice"]["snap_to_data"] = True

    label, details = _history.describe_state_change(prev, curr)

    assert label == "Snap to pixels enabled"
    assert details == ("Snap to pixels changed from Off to On",)


def test_describe_state_change_reports_active_cursor_and_bins() -> None:
    prev = _base_state()
    curr = copy.deepcopy(prev)
    prev["slice"]["indices"] = [[0, 0], [1, 1]]
    prev["slice"]["values"] = [[0.0, 1.0], [1.0, 2.0]]
    prev["slice"]["bins"] = [[1, 1], [1, 1]]
    curr["slice"]["indices"] = [[0, 0], [1, 1]]
    curr["slice"]["values"] = [[0.0, 1.0], [1.0, 2.0]]
    curr["slice"]["bins"] = [[1, 1], [2, 1]]
    curr["current_cursor"] = 1

    label, details = _history.describe_state_change(prev, curr)

    assert "Cursor changed" in label
    assert "Bins changed" in label
    assert "Active cursor changed from 0 to 1" in details
    assert "Bins changed from [[1, 1], [1, 1]] to [[1, 1], [2, 1]]" in details


def test_describe_state_change_reports_roi_guideline_and_plot_changes():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["plotitem_states"][0]["roi_states"] = [{"points": [(0.0, 0.0), (1.0, 1.0)]}]
    curr["plotitem_states"][0]["guideline_state"] = {
        "count": 1,
        "angle": 45.0,
        "offset": (0.0, 0.0),
        "follow_cursor": False,
    }
    curr["plotitem_states"][0]["vb_autorange"] = (False, False)

    label, details = _history.describe_state_change(prev, curr)

    assert "ROI changed" in label
    assert "Guidelines changed" in label
    assert "Plot view changed" in label
    assert "ROI count changed from 0 to 1" in details
    assert "Guideline state changed" in details


def test_describe_state_change_reports_manual_view_limits():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["manual_limits"] = {"x": [0.0, 1.0]}

    label, details = _history.describe_state_change(prev, curr)

    assert label == "View limits changed"
    assert details == ("Manual view limits changed",)


def test_describe_state_change_reports_axis_inversions():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["axis_inversions"] = {"x": True}

    label, details = _history.describe_state_change(prev, curr)

    assert label == "Axis inversion changed"
    assert details == ("Axis inversion changed",)


def test_describe_state_change_reports_added_keys_and_generic_fallback():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["new_state"] = {"flag": True}

    label, details = _history.describe_state_change(prev, curr)

    assert label == "State changed"
    assert details == ("Changed paths: new_state.flag",)


def test_describe_state_change_reports_source_change_and_noop() -> None:
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["file_path"] = "/data/source.h5"
    curr["load_func"] = ("loader:func", "arg")

    label, details = _history.describe_state_change(prev, curr)

    assert label == "Source changed"
    assert "file_path changed from None to /data/source.h5" in details
    assert "load_func changed from None to [loader:func, arg]" in details
    assert _history.describe_state_change(prev, copy.deepcopy(prev)) is None
    assert _history.describe_state_diff(prev, copy.deepcopy(prev)) is None


def test_history_entries_skip_noop_noncurrent_entries() -> None:
    states = [
        {"color": {"cmap": "viridis"}},
        {"color": {"cmap": "viridis"}},
        {"color": {"cmap": "magma"}},
    ]
    area = _DummySlicerArea(states, 2)

    assert _history._history_entries(area) == [
        (
            2,
            0,
            "Changed colormap to magma",
            (("Colormap", "viridis", "magma"),),
            True,
            None,
        )
    ]


def test_history_entries_keep_noop_current_entries() -> None:
    states = [{"value": 1}, {"value": 1}]
    area = _DummySlicerArea(states, 1)

    assert _history._history_entries(area) == [(1, 0, "Current state", (), True, None)]


def test_history_entries_keep_noop_current_callable_entries() -> None:
    class AreaWithEntryChanges(_DummySlicerArea):
        def history_entry_changes(self, *, finalize_pending: bool = True):
            return [(1, 0, {"value": 1}, {"value": 1}, True, None)]

    area = AreaWithEntryChanges([{"value": 1}], 0)

    assert _history._history_entries(area) == [(1, 0, "Current state", (), True, None)]


def test_history_states_include_current_and_future(qtbot):
    data = xr.DataArray(np.arange(4).reshape((2, 2)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    base_state = area._history_state_snapshot()
    area.set_index(0, 1)
    current_state = area._history_state_snapshot()
    area.set_index(1, 1)
    future_state = area._history_state_snapshot()
    area.undo()

    states, current_index = area.history_states()

    assert states[0] == base_state
    assert states[1] == current_state
    assert states[2] == future_state
    assert current_index == 1
    win.close()


def test_go_to_history_index_triggers_expected_actions(qtbot):
    data = xr.DataArray(np.arange(4).reshape((2, 2)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    undo_mock = unittest.mock.Mock()
    redo_mock = unittest.mock.Mock()
    area.undo_act = type("Act", (), {"trigger": undo_mock})()
    area.redo_act = type("Act", (), {"trigger": redo_mock})()

    area.go_to_history_index(-2)
    assert undo_mock.call_count == 2
    assert redo_mock.call_count == 0

    undo_mock.reset_mock()
    redo_mock.reset_mock()

    area.go_to_history_index(3)
    assert undo_mock.call_count == 0
    assert redo_mock.call_count == 3
    win.close()


def test_history_group_coalesces_cursor_spinbox_steps(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    control = win.docks[0].widget().findChild(ItoolCrosshairControls)
    assert control is not None
    control.update_content()
    area = win.slicer_area
    start_index = area.current_indices[0]

    control.spin_idx[0].stepBy(1)
    control.spin_idx[0].stepBy(1)

    assert area.current_indices[0] == start_index + 2
    assert len(area._prev_states) == 1
    area.undo()
    assert area.current_indices[0] == start_index
    win.close()


def test_controls_history_grouping_skips_nested_controls_and_disconnects(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    control = win.docks[1].widget().findChild(ItoolColormapControls)
    assert control is not None
    nested_spin = erlab.interactive.utils.BetterSpinBox(control.misc_controls)
    qtbot.addWidget(nested_spin)
    orphan_spin = erlab.interactive.utils.BetterSpinBox()
    qtbot.addWidget(orphan_spin)
    control._history_group_spins.append(orphan_spin)

    control._connect_history_grouping()
    control.disconnect_signals()

    assert nested_spin not in control._history_group_spins
    assert control._history_group_spins == []
    win.close()


def test_history_entries_record_timestamps(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.set_index(0, 1)

    entry = area._prev_states[-1]
    assert entry.created_at.tzinfo is not None
    assert entry.created_at.utcoffset() is not None
    win.close()


def test_noop_history_write_does_not_evict_full_undo_stack(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area
    base_state = area._history_state_snapshot()

    area._prev_states.clear()
    for i in range(area._prev_states.maxlen or 0):
        before_state = copy.deepcopy(base_state)
        after_state = copy.deepcopy(base_state)
        before_state["current_cursor"] = i
        after_state["current_cursor"] = i + 1
        area._prev_states.append(
            _history.HistoryEntry(
                before_state,
                after_state,
                frozenset({("current_cursor",)}),
                None,
            )
        )
    oldest_entry = area._prev_states[0]

    area.sigWriteHistory.emit()
    area.finalize_history_entry()

    assert len(area._prev_states) == area._prev_states.maxlen
    assert area._prev_states[0] is oldest_entry
    win.close()


def test_noop_history_write_does_not_clear_redo_stack(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area
    base_state = area._history_state_snapshot()
    future_state = copy.deepcopy(base_state)
    future_state["current_cursor"] = base_state["current_cursor"] + 1
    future_entry = _history.HistoryEntry(
        base_state,
        future_state,
        frozenset({("current_cursor",)}),
        None,
    )
    area._next_states.append(future_entry)

    area.sigWriteHistory.emit()
    area.finalize_history_entry()

    assert list(area._next_states) == [future_entry]
    win.close()


def test_history_patch_preserves_unrelated_sequence_items():
    before = {
        "slice": {
            "indices": [[2, 2], [2, 2]],
            "values": [[2.0, 2.0], [2.0, 2.0]],
        }
    }
    after = {
        "slice": {
            "indices": [[4, 2], [2, 2]],
            "values": [[4.0, 2.0], [2.0, 2.0]],
        }
    }
    current = {
        "slice": {
            "indices": [[4, 2], [2, 4]],
            "values": [[4.0, 2.0], [2.0, 4.0]],
        }
    }

    paths = _history.changed_paths(before, after)

    assert paths == frozenset({("slice", "indices", 0, 0), ("slice", "values", 0, 0)})
    assert _history.patch_state(current, before, paths) == {
        "slice": {
            "indices": [[2, 2], [2, 4]],
            "values": [[2.0, 2.0], [2.0, 4.0]],
        }
    }


def test_history_patch_uses_whole_path_for_sequence_shape_changes():
    before = {"slice": {"indices": [[2, 2]]}}
    after = {"slice": {"indices": [[2, 2], [4, 4]]}}
    paths = _history.changed_paths(before, after)

    assert paths == frozenset({("slice", "indices")})
    assert _history.patch_state(after, before, paths) == before


def test_history_patch_preserves_mapping_keys_containing_dots():
    before = {"manual_limits": {"a.b": [0.0, 1.0]}}
    after = {"manual_limits": {"a.b": [0.0, 2.0]}}
    paths = _history.changed_paths(before, after)

    assert paths == frozenset({("manual_limits", "a.b", 1)})
    assert _history.patch_state(after, before, paths) == before


def test_history_patch_uses_whole_path_for_arrays():
    before = {"values": np.array([1, 2])}
    after = {"values": np.array([1, 3])}
    paths = _history.changed_paths(before, after)

    assert paths == frozenset({("values",)})
    patched = _history.patch_state(after, before, paths)
    assert np.array_equal(patched["values"], before["values"])


def test_history_group_context_manager_finalizes_entry(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area
    start_index = area.current_indices[0]

    with area.history_group():
        area.set_index(0, start_index + 1)

    assert not area._history_group_active
    assert len(area._prev_states) == 1
    win.close()


def test_history_finalize_removes_committed_noop_entry_and_suppressed_write(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.begin_history_entry(None)
    entry = area._pending_history_entry
    assert entry is not None
    area._prev_states.append(entry)
    area._pending_history_committed = True
    area.finalize_history_entry()

    with area.history_suppressed():
        area.write_state()

    assert entry not in area._prev_states
    assert area._pending_history_entry is None
    win.close()


def test_history_entry_changes_finalizes_pending_entry(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.begin_history_entry(None)
    entries = area.history_entry_changes()

    assert entries == []
    assert area._pending_history_entry is None
    win.close()


def test_manual_limit_undo_preserves_dotted_dimension_name(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["a.b", "c"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.set_manual_limits({"a.b": [0.0, 1.0]})
    area.sigWriteHistory.emit()
    area.set_manual_limits({"a.b": [0.0, 2.0]})
    area.finalize_history_entry()
    area.undo()

    assert area.manual_limits == {"a.b": [0.0, 1.0]}
    win.close()


def test_history_dialog_open_does_not_consume_new_entry(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area
    start_index = area.current_indices[0]
    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)

    area.set_index(0, start_index + 1)
    area.undo()

    assert area.current_indices[0] == start_index
    dialog.close()
    win.close()


def test_history_group_coalesces_binning_spinbox_steps(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    control = win.docks[2].widget().findChild(ItoolBinningControls)
    assert control is not None
    control.update_content()
    area = win.slicer_area

    control.spins[0].stepBy(1)
    control.spins[0].stepBy(1)

    assert area.array_slicer.get_bins(area.current_cursor)[0] == 5
    assert len(area._prev_states) == 1
    area.undo()
    assert area.array_slicer.get_bins(area.current_cursor)[0] == 1
    win.close()


def test_history_group_records_gamma_slider_as_one_entry(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    control = win.docks[1].widget().findChild(ItoolColormapControls)
    assert control is not None
    area = win.slicer_area
    initial_gamma = area.colormap_properties["gamma"]

    control.gamma_widget.slider.sliderPressed.emit()
    control.gamma_widget.setValue(1.2)
    control.gamma_widget.setValue(1.4)

    assert area.colormap_properties["gamma"] == pytest.approx(1.4)
    assert len(area._prev_states) == 1
    area.undo()
    assert area.colormap_properties["gamma"] == pytest.approx(initial_gamma)
    win.close()


def test_history_group_coalesces_gamma_spinbox_steps(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    control = win.docks[1].widget().findChild(ItoolColormapControls)
    assert control is not None
    area = win.slicer_area
    initial_gamma = area.colormap_properties["gamma"]

    control.gamma_widget.spin.stepBy(1)
    control.gamma_widget.spin.stepBy(1)

    assert area.colormap_properties["gamma"] != pytest.approx(initial_gamma)
    assert len(area._prev_states) == 1
    area.undo()
    assert area.colormap_properties["gamma"] == pytest.approx(initial_gamma)
    win.close()


def test_compound_colormap_change_records_one_entry(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area
    initial_gamma = area.colormap_properties["gamma"]

    area.set_colormap(gamma=1.2, reverse=True)

    assert area.colormap_properties["gamma"] == pytest.approx(1.2)
    assert area.colormap_properties["reverse"] is True
    assert area.reverse_act.isChecked()
    assert len(area._prev_states) == 1
    assert area._prev_states[-1].changed_paths == frozenset(
        {("color", "gamma"), ("color", "reverse")}
    )

    area.undo()
    assert area.colormap_properties["gamma"] == pytest.approx(initial_gamma)
    assert area.colormap_properties["reverse"] is False
    assert not area.reverse_act.isChecked()
    win.close()


def test_history_group_idle_timeout_starts_new_entry(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.begin_history_group()
    area.set_index(0, 3)
    qtbot.wait(area._HISTORY_GROUP_IDLE_TIMEOUT_MS + 50)
    area.begin_history_group()
    area.set_index(0, 4)

    assert len(area._prev_states) == 2
    win.close()


def test_history_group_undo_redo_cancels_active_group(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area
    start_index = area.current_indices[0]

    area.begin_history_group()
    area.set_index(0, start_index + 1)
    assert area._history_group_active

    area.undo()
    assert not area._history_group_active
    assert area.current_indices[0] == start_index

    area.redo()
    assert not area._history_group_active
    assert area.current_indices[0] == start_index + 1
    win.close()


def test_history_group_reuses_pending_linked_transaction_id(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.begin_history_group()
    transaction_id = area.next_linked_history_transaction_id()
    area.record_history_mutation(
        transaction_id,
        lambda: area.set_manual_limits({"x": [0.0, 1.0]}),
        keep_pending=True,
    )
    assert area.next_linked_history_transaction_id() == transaction_id

    area.record_history_mutation(
        transaction_id,
        lambda: area.set_manual_limits({"x": [0.0, 2.0]}),
        keep_pending=True,
    )
    area.end_history_group()

    assert [entry.transaction_id for entry in area._prev_states] == [transaction_id]
    assert area._prev_states[-1].after_state["manual_limits"] == {"x": [0.0, 2.0]}
    win.close()


def test_history_group_linked_transaction_id_resets_after_local_split(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.begin_history_group()
    first_transaction_id = area.next_linked_history_transaction_id()
    area.record_history_mutation(
        first_transaction_id,
        lambda: area.set_manual_limits({"x": [0.0, 1.0]}),
        keep_pending=True,
    )
    area.record_history_mutation(
        None,
        lambda: area.set_manual_limits({"x": [0.0, 1.0], "y": [0.0, 1.0]}),
        keep_pending=True,
    )
    second_transaction_id = area.next_linked_history_transaction_id()
    area.record_history_mutation(
        second_transaction_id,
        lambda: area.set_manual_limits({"x": [0.0, 2.0], "y": [0.0, 1.0]}),
        keep_pending=True,
    )
    area.end_history_group()

    assert second_transaction_id != first_transaction_id
    assert [entry.transaction_id for entry in area._prev_states] == [
        first_transaction_id,
        None,
        second_transaction_id,
    ]
    win.close()


def test_quick_discrete_actions_remain_separate_history_entries(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.swap_axes(0, 1)
    area.swap_axes(0, 1)

    assert len(area._prev_states) == 2
    win.close()


def test_history_entries_include_current_oldest_state():
    states = [
        {"color": {"cmap": "viridis"}},
        {"color": {"cmap": "magma"}},
    ]
    area = _DummySlicerArea(states, 0)

    entries = _history._history_entries(area)

    assert entries[-1] == (0, 0, "Current state", (), True, None)


def test_history_entries_include_current_after_undoing_to_oldest_state(qtbot):
    data = xr.DataArray(np.arange(4).reshape((2, 2)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.set_index(0, 1)
    area.undo()

    entries = _history._history_entries(area)

    assert entries[-1] == (0, 0, "Current state", (), True, None)
    win.close()


def test_history_menu_handles_empty_history(qtbot):
    area = _DummySlicerArea([{"state": 1}], 0)
    menu = _history.HistoryMenu(area)
    qtbot.addWidget(menu)

    menu.update_actions()

    actions = menu.actions()
    assert len(actions) == 3
    assert actions[0].objectName() == "itool_history_show_action"
    assert actions[0].data() == {"kind": "show_history"}
    assert actions[1].isSeparator()
    assert actions[2].objectName() == "itool_history_empty_action"
    assert actions[2].data() == {"kind": "empty"}
    assert not actions[2].isEnabled()


def test_history_menu_populates_actions_and_triggers(qtbot):
    states = [
        {"color": {"cmap": "viridis"}},
        {"color": {"cmap": "magma"}},
        {"color": {"cmap": "plasma"}},
    ]
    area = _DummySlicerArea(states, 1)

    menu = _history.HistoryMenu(area)
    qtbot.addWidget(menu)

    menu.update_actions()

    actions = menu.actions()
    assert actions[0].objectName() == "itool_history_show_action"
    assert actions[1].isSeparator()
    history_actions = actions[2:]
    assert [action.objectName() for action in history_actions] == [
        "itool_history_action_2",
        "itool_history_action_1",
    ]
    assert [action.data() for action in history_actions] == [1, 0]
    assert all("Colormap changed" not in action.toolTip() for action in history_actions)
    assert [action.statusTip() for action in history_actions] == ["", ""]

    actions[0].trigger()
    assert isinstance(menu._dialog, _history.HistoryDialog)
    qtbot.addWidget(menu._dialog)

    history_actions[0].trigger()
    history_actions[1].trigger()
    assert area.go_to_history_calls == [1, 0]


def _dialog_widgets(dialog):
    history_list = dialog.findChild(QtWidgets.QListWidget, "itool_history_list")
    details = dialog.findChild(QtWidgets.QWidget, "itool_history_details")
    go_to_btn = dialog.findChild(QtWidgets.QPushButton, "itool_history_go_to_button")
    assert history_list is not None
    assert details is not None
    assert dialog.findChild(QtWidgets.QScrollArea, "itool_history_details") is None
    assert go_to_btn is not None
    return history_list, details, go_to_btn


def _states_for_dialog_tests():
    past = _base_state()
    past["color"]["cmap"] = "magma"
    current = _base_state()
    future = copy.deepcopy(current)
    future["slice"]["values"] = [[0.5, 1.0]]
    return [past, current, future]


def _detail_rows(details: QtWidgets.QWidget) -> list[tuple[str, str, str]]:
    return [
        (
            row.property("detail"),
            row.property("before"),
            row.property("after"),
        )
        for row in details.findChildren(QtWidgets.QWidget, "itool_history_detail_row")
    ]


def test_history_dialog_populates_current_entry_and_details(qtbot):
    area = _DummySlicerArea(_states_for_dialog_tests(), 1)
    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)

    history_list, details, go_to_btn = _dialog_widgets(dialog)

    assert history_list.count() == 2
    assert history_list.currentItem().data(QtCore.Qt.ItemDataRole.UserRole) == 0
    assert history_list.currentItem().text().startswith("Current: ")
    assert ("Colormap", "magma", "viridis") in _detail_rows(details)
    assert all("->" not in "".join(row) for row in _detail_rows(details))
    assert not go_to_btn.isEnabled()


def test_history_dialog_shows_entry_timestamp(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area
    area.set_index(0, 1)

    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)
    history_list, _, _ = _dialog_widgets(dialog)

    timestamp = _history._format_history_datetime(area._prev_states[-1].created_at)
    item = history_list.item(0)
    assert item.data(QtCore.Qt.ItemDataRole.UserRole + 2) == timestamp
    assert item.toolTip() == timestamp
    assert timestamp in item.text()
    dialog.close()
    win.close()


def test_history_dialog_selection_updates_visible_details(qtbot):
    area = _DummySlicerArea(_states_for_dialog_tests(), 1)
    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)
    history_list, details, go_to_btn = _dialog_widgets(dialog)

    history_list.setCurrentItem(history_list.item(0))

    assert history_list.currentItem().data(QtCore.Qt.ItemDataRole.UserRole) == 1
    assert ("Cursor 1 value", "[0.0, 1.0]", "[0.5, 1.0]") in _detail_rows(details)
    assert go_to_btn.isEnabled()


def test_history_dialog_resizes_long_cursor_details(qtbot):
    past = _base_state()
    past["slice"]["indices"] = [[123456789, 987654321]]
    past["slice"]["values"] = [[12345.678901234567, -98765.43210987654]]
    current = copy.deepcopy(past)
    current["slice"]["indices"] = [[123456790, 987654111]]
    current["slice"]["values"] = [[12345.678901239999, -98765.43210900001]]
    future = copy.deepcopy(current)
    future["slice"]["values"] = [[12345.678901240123, -98765.43210900077]]
    area = _DummySlicerArea([past, current, future], 1)
    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)
    dialog.resize(460, dialog.height())

    history_list, details, go_to_btn = _dialog_widgets(dialog)
    history_list.setCurrentItem(history_list.item(0))

    assert go_to_btn.isEnabled()
    assert any(row[0] == "Cursor 1 value" for row in _detail_rows(details))


def test_history_dialog_go_to_selected_entry(qtbot):
    area = _DummySlicerArea(_states_for_dialog_tests(), 1)
    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)
    history_list, _, go_to_btn = _dialog_widgets(dialog)

    history_list.setCurrentItem(history_list.item(0))
    qtbot.mouseClick(go_to_btn, QtCore.Qt.MouseButton.LeftButton)

    assert area.go_to_history_calls == [1]


def test_history_dialog_handles_empty_selection_and_row_height_fallback(qtbot):
    area = _DummySlicerArea(_states_for_dialog_tests(), 1)
    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)
    history_list, _, _ = _dialog_widgets(dialog)

    history_list.sizeHintForRow = lambda _: 0
    dialog._update_history_list_height()
    history_list.clearSelection()
    history_list.setCurrentItem(None)

    dialog._go_to_selected()

    assert dialog._selected_relative_index() is None
    assert area.go_to_history_calls == []


def test_history_dialog_activates_entry_from_double_click_and_return(qtbot):
    area = _DummySlicerArea(_states_for_dialog_tests(), 1)
    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)
    dialog.show()
    history_list, _, _ = _dialog_widgets(dialog)
    history_list.setCurrentItem(history_list.item(0))

    history_list.itemDoubleClicked.emit(history_list.currentItem())
    assert area.go_to_history_calls == [1]

    area.go_to_history_calls.clear()
    history_list.setFocus()
    qtbot.keyClick(history_list, QtCore.Qt.Key.Key_Return)
    assert area.go_to_history_calls == [1]


def test_history_dialog_refreshes_to_current_entry(qtbot):
    area = _DummySlicerArea(_states_for_dialog_tests(), 1)
    dialog = _history.HistoryDialog(area)
    qtbot.addWidget(dialog)
    history_list, details, go_to_btn = _dialog_widgets(dialog)

    history_list.setCurrentItem(history_list.item(0))
    assert go_to_btn.isEnabled()

    area._current_index = 2
    area.sigHistoryChanged.emit()

    assert history_list.currentItem().data(QtCore.Qt.ItemDataRole.UserRole) == 0
    assert not go_to_btn.isEnabled()
    assert ("Cursor 1 value", "[0.0, 1.0]", "[0.5, 1.0]") in _detail_rows(details)
