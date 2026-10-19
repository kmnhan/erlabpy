import zipfile

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore

import erlab.utils.array
from erlab.interactive.imagetool._provenance._model import FileDataSelection
from erlab.interactive.imagetool.slicer import (
    ArraySlicer,
    _display_safe_float,
    _display_safe_minmax,
    _display_safe_values,
    check_cursors_compatible,
    qsel_args_from_indexers,
)
from erlab.io.plugins.da30 import load_zip


def test_nonuniform_axes_ignores_user_idx_dim(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x_idx", "y"),
        coords={"x_idx": np.arange(3), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer._nonuniform_axes == []
    assert slicer.physical_dims == data.dims


def test_nonuniform_axes_detects_generated_idx_dim(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.array([0.0, 1.0, 1.5]), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert str(slicer._obj.dims[0]).endswith("_idx")
    assert slicer._nonuniform_axes == [0]
    assert slicer.physical_dims == data.dims
    np.testing.assert_array_equal(slicer.coords[0], data.x.values)


def test_physical_dimensions_follow_display_transpose_and_coordinate_changes(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(4, 3),
        dims=("energy", "scan_idx"),
        coords={"energy": [0.0, 0.2, 0.8, 1.5], "scan_idx": [1.0, 2.0, 3.0]},
    )
    owner = QtCore.QObject()
    slicer = ArraySlicer(data, parent=owner)
    slicer.swap_axes(0, 1)

    assert slicer._obj.dims == ("scan_idx", "energy_idx")
    assert slicer.physical_dims == ("scan_idx", "energy")
    assert slicer.isel_args(0, (0,), int_if_one=True) == {
        "energy": slicer.get_indices(0)[1]
    }
    code = "result = data" + slicer.qsel_code(0, (0,))
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102
    xr.testing.assert_identical(
        namespace["result"], data.isel(energy=slicer.get_indices(0)[1])
    )

    updated = data.assign_coords(energy=[0.0, 0.5, 1.0, 1.5])
    slicer.set_array(updated, preserve_dims=("scan_idx", "energy"))

    assert slicer.physical_dims == ("scan_idx", "energy")
    assert slicer._nonuniform_axes == []


def test_restore_nonuniform_dims_ignores_user_idx_dim() -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x_idx", "y"),
        coords={
            "x_idx": np.arange(3),
            "x": ("x_idx", np.arange(3)),
            "y": np.arange(4),
        },
    )

    restored = erlab.utils.array._restore_nonuniform_dims(data)

    xr.testing.assert_identical(restored, data)


@pytest.mark.parametrize("disp", [(0,), (1,)])
@pytest.mark.parametrize("width", [1, 3])
def test_xslice_uses_physical_coordinates_and_shared_values(qtbot, disp, width) -> None:
    data = xr.DataArray(
        np.arange(20.0).reshape(4, 5),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 0.8, 1.5], "y": np.arange(5.0)},
    )
    owner = QtCore.QObject()
    slicer = ArraySlicer(data, parent=owner)
    slicer.set_bin(0, 0, width, update=False)

    result = slicer.xslice(0, disp)

    assert not any(str(name).endswith("_idx") for name in result.coords)
    assert np.shares_memory(slicer._physical_obj.data, slicer._obj.data)
    if disp == (1,):
        center = slicer.get_index(0, 0)
        if width > 1:
            expected = data.isel(
                x=slice(center - width // 2, center + width // 2 + 1)
            ).qsel.mean("x")
        else:
            expected = data.isel(x=center)
    else:
        expected = data.isel(y=slicer.get_index(0, 1))
    xr.testing.assert_identical(result, expected)

    slicer.swap_axes(0, 1)
    assert slicer._physical_obj.dims == ("y", "x")
    replacement = data.assign_coords(x=[0.0, 0.3, 0.9, 1.8])
    slicer.set_array(replacement, copy_values=False)
    xr.testing.assert_identical(slicer._physical_obj, replacement)


def test_check_cursors_compatible_normalizes_promoted_1d_stack_dim() -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
    )
    promoted = ArraySlicer.validate_array(data, copy_values=False)

    assert promoted.dims == ("x", "stack_dim")
    assert check_cursors_compatible(promoted, data)
    assert check_cursors_compatible(data, promoted)
    assert not check_cursors_compatible(
        promoted, data.assign_coords(x=[0.0, 1.0, 2.0, 4.0])
    )


def test_user_idx_dimension_with_associated_coordinate_stays_physical(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(4, 3),
        dims=("scan_idx", "y"),
        coords={
            "scan_idx": np.arange(4.0),
            "scan": ("scan_idx", [0.0, 0.2, 0.8, 1.5]),
            "y": np.arange(3.0),
        },
    )
    owner = QtCore.QObject()
    slicer = ArraySlicer(data, parent=owner)

    assert slicer.physical_dims == data.dims
    assert slicer._nonuniform_axes == []
    xr.testing.assert_identical(slicer._to_physical_data(slicer._obj), data)
    assert not check_cursors_compatible(
        data, data.assign_coords(scan_idx=[0.0, 1.0, 2.0, 4.0])
    )


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


def test_array_slicer_limits_ignore_values_unsafe_for_display(qtbot) -> None:
    data = xr.DataArray(
        np.array([[0.0, np.inf], [1e15, 1e300]]),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer.limits == (0.0, 1e15)
    assert np.isinf(slicer._obj.values[0, 1])
    assert slicer._obj.values[1, 1] == 1e300


def test_display_safe_values_returns_original_safe_array() -> None:
    values = np.array([[0.0, 1e15], [2.0, np.nan]])

    assert _display_safe_values(values) is values


def test_display_safe_helpers_handle_degenerate_values() -> None:
    empty = np.array([], dtype=float)
    strings = np.array(["a", "b"])
    complex_values = np.array([1.0 + 1.0j])

    assert _display_safe_values(empty) is empty
    assert _display_safe_values(strings) is strings
    assert _display_safe_values(complex_values) is complex_values
    assert np.isnan(_display_safe_float(empty))
    assert _display_safe_float(np.array([1.0, np.nan, 3.0])) == 2.0
    assert _display_safe_minmax(
        xr.DataArray(np.array([[np.nan]])), raw_limits=(np.nan, np.nan)
    ) == (0.0, 1.0)
    assert _display_safe_minmax(xr.DataArray(np.array([[np.inf, 1e300]]))) == (0.0, 1.0)


def test_display_safe_minmax_masks_dask_values() -> None:
    data = xr.DataArray(
        np.array([[0.0, np.inf], [1e15, 1e300]]),
        dims=("x", "y"),
    ).chunk({"x": 1})

    assert _display_safe_minmax(data, raw_limits=(0.0, np.inf)) == (0.0, 1e15)


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


def test_da30_zip_readonly_array_binning(qtbot, tmp_path) -> None:
    values = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "Spectrum_R1.ini",
            """[spectrum]
depth = 3
depthoffset = 0
depthdelta = 1
depthlabel = a
height = 5
heightoffset = 0
heightdelta = 1
heightlabel = b
width = 7
widthoffset = 0
widthdelta = 1
widthlabel = c
name = data
""",
        )
        zf.writestr("R1.ini", "[attrs]\n")
        zf.writestr("Spectrum_R1.bin", values.tobytes())

    data = load_zip(zip_path)
    assert isinstance(data, xr.DataArray)
    assert not data.values.flags.writeable
    slicer = ArraySlicer(data, parent=QtCore.QObject())
    assert not slicer._obj.values.flags.writeable
    assert np.shares_memory(data.values, slicer._obj.values)
    selected = slicer._obj.values[:, 1:4, :]
    assert not selected.flags.c_contiguous
    assert not selected.flags.f_contiguous
    slicer.set_indices(0, [1, 2, 3], update=False)
    slicer.set_bin(0, 1, 3, update=False)

    result = slicer.extract_avg_slice(0, (1,))

    np.testing.assert_allclose(result, np.nanmean(values[:, 1:4, :], axis=1))


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


def test_swap_axes_rejects_axes_outside_data_dimensions(qtbot) -> None:
    data = xr.DataArray(np.zeros((3, 4), dtype=np.float32), dims=("a", "b"))
    slicer = ArraySlicer(data, parent=QtCore.QObject())

    with pytest.raises(IndexError, match="incompatible with 2-dimensional data"):
        slicer.swap_axes(0, 2)


def test_swap_axes_normalizes_legacy_short_cursor_state(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("a", "b"),
        coords={"a": np.arange(3), "b": np.arange(4)},
    )
    parent = QtCore.QObject()
    slicer = ArraySlicer(data, parent=parent)
    slicer.set_bin(0, 0, 3, update=False)
    slicer.set_index(0, 0, 2, update=False)
    slicer._bins[0] = slicer._bins[0][:1]
    slicer._indices[0][1] = 99
    slicer._values[0][1] = np.float64(99.0)
    slicer._binned[0] = slicer._binned[0][:1]

    slicer.swap_axes(0, 1)

    assert slicer._obj.dims == ("b", "a")
    assert slicer.get_bins(0) == [1, 3]
    assert slicer.get_indices(0) == [1, 2]
    assert slicer.get_values(0) == [1, 2]
    assert slicer.get_binned(0) == (False, True)


def test_set_array_normalizes_missing_cursor_state(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("a", "b"),
        coords={"a": np.arange(3), "b": np.arange(4)},
    )
    parent = QtCore.QObject()
    slicer = ArraySlicer(data, parent=parent)
    slicer._bins.clear()
    slicer._indices.clear()
    slicer._values.clear()
    slicer._binned.clear()

    slicer.set_array(data.copy(deep=False), validate=False)

    assert slicer.get_bins(0) == [1, 1]
    assert slicer.get_indices(0) == [1, 1]
    assert slicer.get_values(0) == [1, 1]
    assert slicer.get_binned(0) == (False, False)


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


def test_state_restore_preserves_valid_off_grid_cursor_values(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("x", "y"),
        coords={
            "x": np.arange(3, dtype=np.float32),
            "y": np.arange(4, dtype=np.float32),
        },
    )
    parent = QtCore.QObject()
    slicer = ArraySlicer(data, parent=parent)
    slicer.set_value(0, 1, 1.2, update=False)
    saved_state = slicer.state

    slicer.swap_axes(0, 1)
    slicer.state = saved_state

    assert slicer.get_indices(0) == saved_state["indices"][0]
    np.testing.assert_allclose(slicer.get_values(0), saved_state["values"][0])


def test_state_restore_suppresses_partial_cursor_restore_signal(qtbot) -> None:
    data = xr.DataArray(
        np.arange(107 * 36 * 35, dtype=np.float32).reshape(107, 36, 35),
        dims=("eV", "y", "x"),
        coords={
            "eV": np.arange(107, dtype=np.float32),
            "y": np.arange(36, dtype=np.float32),
            "x": np.arange(35, dtype=np.float32),
        },
    )

    parent = QtCore.QObject()
    slicer = ArraySlicer(data, parent=parent)
    slicer.add_cursor(update=False)
    saved_state = {
        "dims": ("x", "y", "eV"),
        "bins": [[1, 1, 1], [1, 1, 1]],
        "indices": [[26, 29, 36], [9, 9, 36]],
        "values": [[26.0, 29.0, 36.0], [9.0, 9.0, 36.0]],
        "snap_to_data": False,
        "twin_coord_names": (),
        "cursor_color_params": None,
    }
    emissions = []

    slicer.sigIndexChanged.connect(lambda *args: emissions.append(args))

    slicer.state = saved_state

    assert emissions == []
    assert slicer._obj.dims == saved_state["dims"]
    assert slicer._obj.shape == (35, 36, 107)
    assert slicer.get_indices(0) == saved_state["indices"][0]
    assert slicer.get_indices(1) == saved_state["indices"][1]
    assert (
        slicer.point_value(1, binned=True)
        == data.transpose("x", "y", "eV").data[9, 9, 36]
    )


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


def test_bin_slice_descending_axis_boundary_window_two(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((4, 3), dtype=np.float32),
        dims=("x", "y"),
        coords={"x": np.array([4.0, 3.0, 2.0, 1.0]), "y": np.arange(3)},
    )
    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [0, 1], update=False)
    slicer.set_bin(0, 0, 2, update=False)

    assert slicer._bin_slice(0, 0) == slice(0, 2)


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


def test_associated_coord_point_value_handles_bins_and_missing(qtbot) -> None:
    x = np.arange(3, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(5, dtype=float)
    plane = x[:, None] * 10.0 + y[None, :] ** 2
    plane[1, 1] = np.nan
    xz = x[:, None] * 1000.0 + z[None, :]
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype=np.float32),
        dims=("x", "y", "z"),
        coords={
            "x": x,
            "y": y,
            "z": z,
            "temp": ("x", x + 300.0),
            "plane": (("x", "y"), plane),
            "xz": (("x", "z"), xz),
            "label": ("x", ["a", "b", "c"]),
        },
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [1, 1, 3], update=False)

    assert float(np.asarray(slicer.associated_coord_point_value("temp", 0))) == 301.0
    assert np.isnan(float(np.asarray(slicer.associated_coord_point_value("plane", 0))))
    assert float(np.asarray(slicer.associated_coord_point_value("xz", 0))) == 1003.0
    assert slicer.associated_coord_point_value("missing", 0) is None

    slicer.set_bin(0, 1, 3, update=False)

    np.testing.assert_allclose(
        slicer.associated_coord_point_value("plane", 0),
        np.nanmean([10.0, np.nan, 14.0]),
    )
    assert float(np.asarray(slicer.associated_coord_point_value("xz", 0))) == 1003.0


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


def test_qsel_args_from_indexers_validates_live_bin_coordinates() -> None:
    data = xr.DataArray(
        np.zeros(5, dtype=np.float32),
        dims=("x",),
        coords={"x": np.array([0.0, 1.0, 1.0, 3.0, 2.0])},
    )

    with pytest.raises(ValueError, match="Dimension `missing` not found"):
        qsel_args_from_indexers(data, {"missing": 0}, ())

    with pytest.raises(ValueError, match="Bin does not contain"):
        qsel_args_from_indexers(data, {"x": slice(2, 5)}, ("x",))


@pytest.mark.parametrize("has_unindexed_coord", [False, True])
def test_qsel_args_from_indexers_uses_positions_for_unindexed_dimension(
    has_unindexed_coord: bool,
) -> None:
    data = xr.DataArray(
        np.array([0.0, 10.0, 100.0, 1000.0, 10000.0]),
        dims=("x",),
    )
    if has_unindexed_coord:
        data = data.assign_coords(x=[10.0, 20.0, 30.0, 40.0, 50.0]).drop_indexes("x")
    original = data.copy(deep=True)

    args = qsel_args_from_indexers(data, {"x": slice(1, 4)}, ("x",))

    assert args == {"x": 2.0, "x_width": 3.0}
    np.testing.assert_allclose(
        data.qsel(args).values,
        data.isel(x=slice(1, 4)).mean("x").values,
    )
    xr.testing.assert_identical(data, original)


def test_qsel_args_from_indexers_rejects_invalid_bin_indexers() -> None:
    data = xr.DataArray(np.arange(5.0), dims=("x",))

    with pytest.raises(ValueError, match="must be an index slice"):
        qsel_args_from_indexers(data, {"x": 2}, ("x",))

    with pytest.raises(ValueError, match="is empty"):
        qsel_args_from_indexers(data, {"x": slice(2, 2)}, ("x",))


def test_qsel_args_desc_uniform_descending_axis_emits_positive_width() -> None:
    data = xr.DataArray(
        np.arange(5, dtype=np.float32),
        dims=("x",),
        coords={"x": np.array([5.0, 4.0, 3.0, 2.0, 1.0])},
    )

    args = qsel_args_from_indexers(data, {"x": slice(0, 2)}, ("x",))

    assert args["x"] == 4.5
    assert args["x_width"] > 0.0

    actual = data.qsel(args)
    expected = data.sortby("x").sel(
        x=slice(
            args["x"] - args["x_width"] / 2,
            args["x"] + args["x_width"] / 2,
        )
    )
    np.testing.assert_allclose(actual.values, expected.mean().values)


def test_xslice_descending_hidden_axis_with_boundary_bin_succeeds() -> None:
    values = np.arange(10, dtype=np.float32).reshape(2, 5)
    data = xr.DataArray(
        values,
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.array([5.0, 4.0, 3.0, 2.0, 1.0])},
    )
    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [0, 0], update=False)
    slicer.set_bin(0, 1, 2, update=False)

    selected = slicer.xslice(0, (0,))

    np.testing.assert_allclose(selected.values, np.array([0.5, 5.5], dtype=np.float32))


def test_isel_code_uses_call_kwargs_formatting(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((4, 5, 6), dtype=np.float32),
        dims=("x", "y", "z"),
        coords={"x": np.arange(4), "y": np.arange(5), "z": np.arange(6)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())
    slicer.set_indices(0, [1, 2, 3], update=False)

    assert slicer.isel_code(0, (0, 1)) == ".isel(z=3)"


def test_selection_plan_averages_all_hidden_binned_dimensions(qtbot) -> None:
    data = xr.DataArray(
        np.arange(360.0).reshape(3, 4, 5, 6),
        dims=("x", "y", "z", "t"),
        coords={
            "x": np.arange(3.0),
            "y": np.arange(4.0),
            "z": [0.0, 0.2, 0.8, 1.6, 3.0],
            "t": np.arange(6.0),
        },
    )
    owner = QtCore.QObject()
    slicer = ArraySlicer(data, parent=owner)
    slicer.set_indices(0, [1, 2, 2, 3], update=False)
    slicer.set_bin(0, 2, 3, update=False)
    slicer.set_bin(0, 3, 2, update=False)
    expected = data.isel(z=slice(1, 4), t=slice(2, 4)).qsel.mean(("z", "t"))

    xr.testing.assert_identical(slicer.xslice(0, (0, 1)), expected)
    replayed = data
    for operation in slicer.selection_operations(0, (0, 1)):
        replayed = operation.apply(replayed)
    xr.testing.assert_identical(replayed, expected)
    for code in (slicer.qsel_code(0, (0, 1)), slicer.isel_code(0, (0, 1))):
        namespace = {"data": data}
        exec(f"selected = data{code}", namespace)  # noqa: S102
        xr.testing.assert_identical(namespace["selected"], expected)
        assert "z_idx" not in code


@pytest.mark.parametrize("width", [1, 3])
def test_nonuniform_plot_selection_replays_displayed_bin(
    qtbot, manager_context, tmp_path, width
) -> None:
    data = xr.DataArray(
        np.arange(60.0).reshape(3, 4, 5),
        dims=("x", "y", "z"),
        coords={
            "x": np.arange(3.0),
            "y": np.arange(4.0),
            "z": [0.0, 0.2, 0.8, 1.6, 3.0],
        },
    )
    path = tmp_path / "nonuniform.h5"
    data.to_netcdf(path, engine="h5netcdf")
    tool = erlab.interactive.itool(
        data,
        execute=False,
        manager=False,
        file_path=path,
        load_func=(
            xr.load_dataarray,
            {"engine": "h5netcdf"},
            FileDataSelection(kind="dataarray"),
        ),
    )
    qtbot.addWidget(tool)
    tool.slicer_area.set_index(axis=2, value=2, cursor=0)
    tool.slicer_area.set_bin(axis=2, value=width, cursor=0)
    image = tool.slicer_area.images[0]
    expected = data.isel(z=2) if width == 1 else data.isel(z=slice(1, 4)).qsel.mean("z")
    source_spec = image.make_tool_source_spec()
    xr.testing.assert_identical(image.current_data, expected)
    xr.testing.assert_identical(
        source_spec.apply(tool.slicer_area._tool_source_parent_data()), expected
    )
    code = image.get_selection_code()
    namespace = {"data": data}
    exec(f"selected = {code}", namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["selected"], expected)
    assert "z_idx" not in code
    assert "z_idx" not in source_spec.model_dump_json()

    with manager_context() as manager:
        manager.add_imagetool(tool, show=False)
        image.open_in_new_window()
        child_uid = manager._node_for_target(0)._childtool_indices[0]
        child = manager.get_imagetool(child_uid)
        xr.testing.assert_identical(child.slicer_area.displayed_data, expected)
        child_node = manager.node_from_slicer_area(child.slicer_area)
        assert child_node is not None
        assert child_node.reload_source_data()
        xr.testing.assert_identical(child.slicer_area.displayed_data, expected)
