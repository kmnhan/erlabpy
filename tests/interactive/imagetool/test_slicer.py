import numpy as np
import xarray as xr
from qtpy import QtCore

from erlab.interactive.imagetool.slicer import ArraySlicer


def test_nonuniform_axes_ignores_user_idx_dim(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x_idx", "y"),
        coords={"x_idx": np.arange(3), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer._nonuniform_axes == []


def test_nonuniform_axes_detects_generated_idx_dim(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.array([0.0, 1.0, 1.5]), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert str(slicer._obj.dims[0]).endswith("_idx")
    assert slicer._nonuniform_axes == [0]


def test_set_array_shallow_copy_does_not_require_deep_copy(qtbot) -> None:
    data1 = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    data2 = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data1, parent=QtCore.QObject())
    slicer.set_array(data2, validate=False, reset=True)

    assert slicer._obj.equals(data2)


def test_set_array_preserves_requested_dimension_order(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_array(data, preserve_dims=("y", "x"))

    assert slicer._obj.dims == ("y", "x")


def test_validate_array_does_not_deepcopy_attrs(qtbot) -> None:
    class _NoDeepCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("deepcopy should not be called")

    sentinel = _NoDeepCopy()
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
        attrs={"sentinel": sentinel},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer._obj.attrs["sentinel"] is sentinel


def test_validate_array_copy_values_detaches_values_buffer(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=np.float32).reshape((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
    )

    validated = ArraySlicer.validate_array(data, copy_values=True)

    validated.values[0, 0] = -1.0

    assert float(data.values[0, 0]) == 0.0


def test_refresh_array_layout_cache_ignores_invalid_generated_idx_coord(qtbot) -> None:
    slicer = ArraySlicer(
        xr.DataArray(
            np.zeros((3, 4), dtype=np.float32),
            dims=("x", "y"),
            coords={"x": np.arange(3), "y": np.arange(4)},
        ),
        parent=QtCore.QObject(),
    )
    slicer._obj = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("x_idx", "y"),
        coords={
            "x_idx": np.arange(3),
            "x": (("x_idx", "y"), np.zeros((3, 4), dtype=np.float32)),
            "y": np.arange(4),
        },
    )
    slicer._refresh_array_layout_cache()

    assert slicer._nonuniform_axes == []


def test_index_of_value_nonuniform_descending_axis(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((4, 3)),
        dims=("x", "y"),
        coords={"x": np.array([5.0, 3.0, 2.0, -1.0]), "y": np.arange(3)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    for value in (6.0, 4.2, 2.4, 0.0, -2.0):
        idx = slicer.index_of_value(0, value, uniform=False)
        expected = int(np.argmin(np.abs(data.x.values - value)))
        assert idx == expected


def test_array_rect_uses_uniform_limits(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("x", "y"),
        coords={
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([10.0, 20.0, 30.0, 40.0]),
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    np.testing.assert_allclose(slicer.array_rect(0, 1), (0.5, 5.0, 3.0, 40.0))


def test_index_of_value_uniform_axis_clamps(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((4, 3), dtype=np.float32),
        dims=("x", "y"),
        coords={
            "x": np.arange(4, dtype=np.float32),
            "y": np.arange(3, dtype=np.float32),
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer.index_of_value(0, -1.0) == 0
    assert slicer.index_of_value(0, 1.6) == 2
    assert slicer.index_of_value(0, 10.0) == 3


def test_index_of_value_uniform_axis_preserves_round_behavior(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((5, 3), dtype=np.float32),
        dims=("x", "y"),
        coords={
            "x": np.arange(5, dtype=np.float32),
            "y": np.arange(3, dtype=np.float32),
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer.index_of_value(0, 0.5) == 0
    assert slicer.index_of_value(0, 1.5) == 2


def test_bin_along_multiaxis_unbinned_axes_returns_selected_data(qtbot) -> None:
    values = np.arange(4 * 5 * 6 * 7, dtype=np.float32).reshape(4, 5, 6, 7)
    data = xr.DataArray(
        values,
        dims=("a", "b", "c", "d"),
        coords={
            dim: np.arange(size, dtype=np.float32)
            for dim, size in zip(("a", "b", "c", "d"), values.shape, strict=True)
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [1, 2, 3, 4], update=False)

    result = slicer._bin_along_multiaxis(0, (1, 3))

    np.testing.assert_allclose(result, values[:, 2, :, 4])


def test_bin_along_multiaxis_mixed_binned_axes_remaps_selected_axes(qtbot) -> None:
    values = np.arange(4 * 5 * 6 * 7, dtype=np.float32).reshape(4, 5, 6, 7)
    data = xr.DataArray(
        values,
        dims=("a", "b", "c", "d"),
        coords={
            dim: np.arange(size, dtype=np.float32)
            for dim, size in zip(("a", "b", "c", "d"), values.shape, strict=True)
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [1, 2, 3, 4], update=False)
    slicer.set_bin(0, 1, 3, update=False)

    result = slicer._bin_along_multiaxis(0, (1, 3))

    np.testing.assert_allclose(result, np.nanmean(values[:, 1:4, :, 4], axis=1))


def test_bin_along_multiaxis_point_value_single_binned_axis_reduces_1d(qtbot) -> None:
    values = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    data = xr.DataArray(
        values,
        dims=("a", "b", "c"),
        coords={
            dim: np.arange(size, dtype=np.float32)
            for dim, size in zip(("a", "b", "c"), values.shape, strict=True)
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [1, 2, 3], update=False)
    slicer.set_bin(0, 1, 3, update=False)

    result = slicer._bin_along_multiaxis(0, (0, 1, 2))

    assert result == np.nanmean(values[1, 1:4, 3])


def test_get_binned_cache_tracks_bin_updates_and_axis_swaps(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype=np.float32),
        dims=("a", "b", "c"),
        coords={
            dim: np.arange(size)
            for dim, size in zip(("a", "b", "c"), (3, 4, 5), strict=True)
        },
    )

    parent = QtCore.QObject()
    slicer = ArraySlicer(data, parent=parent)

    assert slicer.get_binned(0) == (False, False, False)

    slicer.set_bin(0, 1, 3, update=False)
    assert slicer.get_binned(0) == (False, True, False)
    assert slicer.is_binned(0)

    slicer.swap_axes(0, 1)
    assert slicer.get_binned(0) == (True, False, False)


def test_clear_dim_cache_resets_dimension_memos(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype=np.float32),
        dims=("a", "b", "c"),
        coords={
            dim: np.arange(size, dtype=np.float32)
            for dim, size in zip(("a", "b", "c"), (3, 4, 5), strict=True)
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer._hidden_axes_for_disp((0, 1)) == (2,)
    _ = slicer.uniform_index_params

    assert slicer._hidden_axes_cache == {(0, 1): (2,)}
    assert slicer._hidden_axes_has_nonuniform_cache == {(0, 1): False}
    assert "uniform_index_params" in slicer.__dict__

    slicer.clear_dim_cache()

    assert slicer._hidden_axes_cache == {}
    assert slicer._hidden_axes_has_nonuniform_cache == {}
    assert "uniform_index_params" not in slicer.__dict__
    assert slicer._hidden_axes_for_disp((0, 1)) == (2,)


def test_state_restore_rebuilds_layout_caches_before_cursor_restore(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype=np.float32),
        dims=("x", "y", "z"),
        coords={
            "x": np.array([0.0, 1.0, 1.5], dtype=np.float32),
            "y": np.arange(4, dtype=np.float32),
            "z": np.arange(5, dtype=np.float32),
        },
    )

    parent = QtCore.QObject()
    slicer = ArraySlicer(data, parent=parent)
    slicer.set_indices(0, [2, 1, 3], update=False)
    saved_state = slicer.state

    slicer.swap_axes(0, 1)
    slicer.state = saved_state

    assert slicer._obj.dims == saved_state["dims"]
    assert slicer._nonuniform_axes == [0]
    assert slicer._nonuniform_axes_set == {0}
    assert slicer._dim_indices[slicer._obj.dims[0]] == 0
    assert slicer.get_indices(0) == saved_state["indices"][0]
    np.testing.assert_allclose(slicer.get_values(0), saved_state["values"][0])


def test_bin_along_axis_unbinned_matches_integer_index_selection(qtbot) -> None:
    values = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    data = xr.DataArray(
        values,
        dims=("a", "b", "c"),
        coords={
            dim: np.arange(size, dtype=np.float32)
            for dim, size in zip(("a", "b", "c"), values.shape, strict=True)
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [2, 3, 4], update=False)

    np.testing.assert_allclose(slicer._bin_along_axis(0, 1), values[:, 3, :])


def test_point_value_unbinned_returns_current_scalar(qtbot) -> None:
    values = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    data = xr.DataArray(
        values,
        dims=("x", "y"),
        coords={"x": np.arange(4), "y": np.arange(5)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [2, 3], update=False)

    assert slicer.point_value(0, binned=False) == values[2, 3]


def test_value_of_index_uniform_path_on_nonuniform_axis(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((4, 3), dtype=np.float32),
        dims=("x", "y"),
        coords={"x": np.array([0.0, 1.0, 1.5, 2.5]), "y": np.arange(3)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer.value_of_index(0, 2, uniform=True) == 2


def test_index_of_value_constant_uniform_axis_returns_zero(qtbot) -> None:
    parent = QtCore.QObject()
    slicer = ArraySlicer(
        xr.DataArray(
            np.zeros((4, 3), dtype=np.float32),
            dims=("x", "y"),
            coords={"x": np.arange(4), "y": np.arange(3)},
        ),
        parent=parent,
    )
    slicer.set_array(
        xr.DataArray(
            np.zeros((4, 3), dtype=np.float32),
            dims=("x", "y"),
            coords={"x": np.ones(4, dtype=np.float32), "y": np.arange(3)},
        ),
        validate=False,
        reset=True,
    )

    assert slicer.index_of_value(0, 10.0) == 0


def test_qsel_args_falls_back_to_dims_index_lookup(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((4, 5, 6), dtype=np.float32),
        dims=("x", "y", "z"),
        coords={"x": np.arange(4), "y": np.arange(5), "z": np.arange(6)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer._dim_indices = {}

    assert slicer.qsel_args(0, (0, 1)) == {"z": 2.0}
