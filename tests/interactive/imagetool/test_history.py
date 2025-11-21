from __future__ import annotations

import copy
import unittest.mock

import numpy as np
import pytest
import xarray as xr

from erlab.interactive.imagetool import ImageTool, _history


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
            "bins": [1, 1],
            "dims": ["x", "y"],
        },
    }


def test_describe_state_diff_ignores_value_only_changes():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["slice"]["values"] = [[5.0, 6.0]]

    assert _history.describe_state_diff(prev, curr) is None


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


def test_describe_state_diff_reports_colormap_and_dims():
    prev = _base_state()
    curr = copy.deepcopy(prev)
    curr["color"]["reverse"] = True
    curr["slice"]["dims"] = ["y", "x"]
    curr["slice"]["indices"] = [[np.int64(1), np.int64(0)]]
    curr["slice"]["values"] = [[2.0, 3.0]]

    assert _history.describe_state_diff(prev, curr) == "Colormap reversed, Transposed"


def test_history_states_include_current_and_future(qtbot):
    data = xr.DataArray(np.arange(4).reshape((2, 2)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    base_state = area.state
    past_state = copy.deepcopy(base_state)
    past_state["color"]["reverse"] = True
    future_state = copy.deepcopy(base_state)
    future_state["current_cursor"] = base_state["current_cursor"] + 1

    area._prev_states.clear()
    area._next_states.clear()
    area._prev_states.append(past_state)
    area._next_states.append(future_state)

    states, current_index = area.history_states()

    assert states[0] == past_state
    assert states[1] == base_state
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
