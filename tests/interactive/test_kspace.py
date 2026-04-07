import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize
import xarray as xr

import erlab
from erlab.constants import AxesConfiguration
from erlab.interactive.kspace import KspaceTool, ktool
from erlab.io.exampledata import generate_hvdep_cuts

_NORMAL_EMISSION_CASES = [
    pytest.param(
        AxesConfiguration.Type1,
        {"xi": 7.25},
        {"delta": 12.5, "xi": 2.5, "beta": -3.75},
        [4.75, -3.75],
        id="Type1",
    ),
    pytest.param(
        AxesConfiguration.Type2,
        {"xi": -6.5},
        {"delta": -8.0, "xi": -1.75, "beta": 2.25},
        [-4.75, 2.25],
        id="Type2",
    ),
    pytest.param(
        AxesConfiguration.Type1DA,
        {"xi": 5.25, "chi": 6.5},
        {"delta": 9.0, "chi": 2.0, "xi": 1.75},
        [3.5, -4.5],
        id="Type1DA",
    ),
    pytest.param(
        AxesConfiguration.Type2DA,
        {"xi": 6.75, "chi": -4.0},
        {"delta": -7.5, "chi": 1.25, "xi": 2.5},
        [-5.25, 4.25],
        id="Type2DA",
    ),
]


def _make_ktool_data(
    anglemap, configuration: AxesConfiguration, coords: dict[str, float]
) -> xr.DataArray:
    data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 3)).copy(
        deep=True
    )
    data.attrs["configuration"] = int(configuration)
    return data.assign_coords(**coords)


def _solve_normal_emission_angles(
    configuration: AxesConfiguration,
    coords: dict[str, float],
    offsets: dict[str, float],
    initial_guess: list[float],
) -> tuple[float, float]:
    forward = erlab.analysis.kspace.get_kconv_forward(configuration)

    if configuration in (AxesConfiguration.Type1, AxesConfiguration.Type2):
        angle_params = {
            "delta": offsets["delta"],
            "xi": coords["xi"],
            "xi0": offsets["xi"],
            "beta0": offsets["beta"],
        }
    else:
        angle_params = {
            "delta": offsets["delta"],
            "chi": coords["chi"],
            "chi0": offsets["chi"],
            "xi": coords["xi"],
            "xi0": offsets["xi"],
        }

    result = scipy.optimize.root(
        lambda angles: np.array(
            [float(v) for v in forward(angles[0], angles[1], 1.0, **angle_params)]
        ),
        x0=np.asarray(initial_guess, dtype=float),
    )

    assert result.success, result.message
    return float(result.x[0]), float(result.x[1])


def test_ktool_compatible(anglemap) -> None:
    cut = anglemap.qsel(beta=-8.3)
    data_4d = anglemap.expand_dims("x", 2)
    data_3d_without_alpha = data_4d.qsel(alpha=-8.3)

    for data in (cut, data_4d, data_3d_without_alpha):
        with pytest.raises(
            ValueError, match=r"Data is not compatible with the interactive tool."
        ):
            data.kspace.interactive()


@pytest.mark.parametrize("wf", ["wf_auto", "wf_manual"])
@pytest.mark.parametrize("kind", ["map", "const_energy"])
@pytest.mark.parametrize("assignment", ["before", "after"])
def test_ktool(qtbot, anglemap, wf, kind, assignment) -> None:
    offset_dict = {"delta": 30.0, "xi": 20.0, "beta": 10.0}

    anglemap = anglemap.copy()

    def assign_attrs():
        anglemap.kspace.offsets = offset_dict
        if wf != "wf_auto":
            anglemap.kspace.work_function = 4.0

    if kind != "map":
        anglemap = anglemap.qsel(eV=-0.1)

    if assignment == "before":
        assign_attrs()

    win = ktool(
        anglemap,
        avec=erlab.lattice.abc2avec(6.97, 6.97, 8.685, 90, 90, 120),
        rotate_bz=30.0,
        cmap="terrain_r",
        execute=False,
    )
    qtbot.addWidget(win)
    win.centering_combo.setCurrentText("I")
    if not win.data.kspace._has_hv:
        win.kz_spin.setValue(0.5)

    for k, v in offset_dict.items():
        if assignment == "before":
            assert np.isclose(win._offset_spins[k].value(), v)
        else:
            win._offset_spins[k].setValue(v)

    if wf != "wf_auto":
        if assignment == "before":
            assert np.isclose(win._offset_spins["wf"].value(), 4.0)
        else:
            win._offset_spins["wf"].setValue(4.0)

    if assignment != "before":
        assign_attrs()
    anglemap_kconv = anglemap.kspace.convert()

    def _check_code_kconv(w: KspaceTool):
        code = w.copy_code()
        assert ".kspace.set_normal(" in code
        assert ".kspace.offsets =" not in code
        namespace = {"anglemap": anglemap}
        exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
        xr.testing.assert_identical(anglemap_kconv, namespace["anglemap_kconv"])

    _check_code_kconv(win)

    # Test ROI
    win.add_circle_btn.click()
    roi = win._roi_list[0]
    roi.getMenu()
    roi.set_position((0.1, 0.1), 0.2)
    assert roi.get_position() == (0.1, 0.1, 0.2)

    roi_control_widget = roi._pos_menu.actions()[0].defaultWidget()
    roi_control_widget.x_spin.setValue(0.0)
    roi_control_widget.y_spin.setValue(0.2)
    roi_control_widget.r_spin.setValue(0.3)
    assert roi.get_position() == (0.0, 0.2, 0.3)

    assert win.preview_symmetry_group.isEnabled()
    win.preview_symmetry_fold_spin.setValue(4)
    win.preview_symmetry_group.setChecked(True)
    win.update()

    # Show imagetool
    win.show_converted()
    win._itool.show()
    xr.testing.assert_identical(win._itool.slicer_area.data, anglemap_kconv)
    win._itool.close()

    # Test save & restore
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, KspaceTool)

        assert win.tool_status == win_restored.tool_status
        assert str(win_restored.info_text) == str(win.info_text)


@pytest.mark.parametrize(
    ("configuration", "coords", "reference_offsets", "initial_guess"),
    _NORMAL_EMISSION_CASES,
)
def test_ktool_normal_emission_updates_offsets(
    qtbot,
    anglemap,
    configuration: AxesConfiguration,
    coords: dict[str, float],
    reference_offsets: dict[str, float],
    initial_guess: list[float],
) -> None:
    data = _make_ktool_data(anglemap, configuration, coords)
    alpha_normal, beta_normal = _solve_normal_emission_angles(
        configuration, coords, reference_offsets, initial_guess
    )

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    assert win.normal_emission_group.title() == "Normal Emission"

    win._offset_spins["delta"].setValue(reference_offsets["delta"])
    win._normal_emission_spins["alpha"].setValue(alpha_normal)
    win._normal_emission_spins["beta"].setValue(beta_normal)

    for key, expected in reference_offsets.items():
        assert np.isclose(win._offset_spins[key].value(), expected)


@pytest.mark.parametrize(
    ("configuration", "coords", "reference_offsets", "initial_guess"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            {"xi": 7.25},
            {"delta": 12.5, "xi": 2.5, "beta": -3.75},
            [4.75, -3.75],
            id="Type1",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            {"xi": 6.75, "chi": -4.0},
            {"delta": -7.5, "chi": 1.25, "xi": 2.5},
            [-5.25, 4.25],
            id="Type2DA",
        ),
    ],
)
def test_ktool_normal_emission_spins_follow_offsets(
    qtbot,
    anglemap,
    configuration: AxesConfiguration,
    coords: dict[str, float],
    reference_offsets: dict[str, float],
    initial_guess: list[float],
) -> None:
    data = _make_ktool_data(anglemap, configuration, coords)
    alpha_normal, beta_normal = _solve_normal_emission_angles(
        configuration, coords, reference_offsets, initial_guess
    )

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    for key, value in reference_offsets.items():
        win._offset_spins[key].setValue(value)

    assert win._normal_emission_spins["alpha"].value() == pytest.approx(
        alpha_normal, abs=1e-3
    )
    assert win._normal_emission_spins["beta"].value() == pytest.approx(
        beta_normal, abs=1e-3
    )


def test_ktool_normal_emission_zero_offsets_are_finite_for_da(qtbot, anglemap) -> None:
    data = _make_ktool_data(
        anglemap,
        AxesConfiguration.Type2DA,
        {"xi": 0.0, "chi": 0.0},
    )

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    assert np.isfinite(win._normal_emission_spins["alpha"].value())
    assert np.isfinite(win._normal_emission_spins["beta"].value())
    assert win._normal_emission_spins["alpha"].value() == pytest.approx(0.0)
    assert win._normal_emission_spins["beta"].value() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("configuration", "coords", "reference_offsets"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            {"xi": 7.25},
            {"delta": 12.5, "xi": 2.5, "beta": -3.75},
            id="Type1",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            {"xi": 6.75, "chi": -4.0},
            {"delta": -7.5, "chi": 1.25, "xi": 2.5},
            id="Type2DA",
        ),
    ],
)
def test_ktool_copy_code_uses_set_normal(
    qtbot,
    anglemap,
    configuration: AxesConfiguration,
    coords: dict[str, float],
    reference_offsets: dict[str, float],
) -> None:
    data = _make_ktool_data(anglemap, configuration, coords)
    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    for key, value in reference_offsets.items():
        win._offset_spins[key].setValue(value)

    code = win.copy_code()
    assert ".kspace.set_normal(" in code
    assert ".kspace.offsets =" not in code

    input_name = str(win._argnames["data"])
    if not input_name.isidentifier():
        input_name = "data"

    namespace = {input_name: data.copy(deep=True)}
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102

    expected = win._assign_params(data.copy(deep=True)).kspace.convert(
        bounds=win.bounds, resolution=win.resolution
    )
    for key, value in reference_offsets.items():
        assert namespace[input_name].kspace.offsets[key] == pytest.approx(value)

    xr.testing.assert_allclose(expected, namespace[f"{input_name}_kconv"])


def test_ktool_update_rate_limited(qtbot, anglemap, monkeypatch) -> None:
    win = ktool(anglemap, execute=False)
    qtbot.addWidget(win)

    # Allow any startup-triggered delayed update to finish before counting.
    wait_ms = int(1000 / win._UPDATE_LIMIT_HZ) + 50
    qtbot.wait(wait_ms)

    call_count = 0
    original_get_data = win.get_data

    def _counting_get_data():
        nonlocal call_count
        call_count += 1
        return original_get_data()

    monkeypatch.setattr(win, "get_data", _counting_get_data)

    spin = win._offset_spins["delta"]
    base = spin.value()
    spin.setValue(base + 0.01)
    spin.setValue(base + 0.02)
    spin.setValue(base + 0.03)

    qtbot.wait_until(lambda: call_count == 1, timeout=2000)
    qtbot.wait(wait_ms)
    assert call_count == 1

    win.update()
    assert call_count == 2


def test_ktool_kinetic_energy_axis_preview(qtbot, anglemap) -> None:
    data = anglemap.copy().assign_coords(hv=6.2)
    data = data.assign_coords(eV=data.hv - data.kspace.work_function + data.eV)

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    min0, max0 = win.center_spin.minimum(), win.center_spin.maximum()
    win._offset_spins["wf"].setValue(win._offset_spins["wf"].value() - 0.2)
    assert np.isclose(win.center_spin.minimum(), min0 - 0.2)
    assert np.isclose(win.center_spin.maximum(), max0 - 0.2)

    win.width_spin.setValue(5)
    energy_axis = data.eV.values - float(data.hv.values) + win._work_function
    center = float(energy_axis[len(energy_axis) // 2])
    win.center_spin.setValue(center)
    center = win.center_spin.value()

    ang, kpreview = win.get_data()

    idx = int(np.argmin(np.abs(energy_axis - center)))
    start = max(0, idx - win.width_spin.value() // 2)
    stop = min(
        energy_axis.size,
        idx + (win.width_spin.value() - 1) // 2 + 1,
    )

    expected_ang = (
        data.copy()
        .assign_coords(eV=energy_axis)
        .isel(eV=slice(start, stop))
        .mean("eV", skipna=True, keep_attrs=True)
        .assign_coords(eV=center)
    )
    expected_ang = win._assign_params(expected_ang)
    expected_k = expected_ang.kspace.convert(
        bounds=win.bounds, resolution=win.resolution, silent=True
    )

    xr.testing.assert_allclose(ang, expected_ang)
    xr.testing.assert_allclose(kpreview, expected_k)


def test_ktool_bounds_estimate_uses_current_inner_potential(qtbot) -> None:
    data = generate_hvdep_cuts((15, 30, 20), hvrange=(20.0, 30.0), noise=False)
    data.kspace.inner_potential = 10.0

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    initial_kz_bounds = (
        win._bound_spins["kz0"].value(),
        win._bound_spins["kz1"].value(),
    )
    win._offset_spins["V0"].setValue(20.0)
    win.calculate_bounds()

    expected_data = win._assign_params(win.data.copy())
    expected_bounds = expected_data.kspace.estimate_bounds()
    for axis in win.data.kspace.momentum_axes:
        for idx in range(2):
            spin = win._bound_spins[f"{axis}{idx}"]
            assert np.isclose(
                spin.value(),
                np.round(expected_bounds[axis][idx], spin.decimals()),
            )

    assert not np.isclose(win._bound_spins["kz0"].value(), initial_kz_bounds[0])
    assert not np.isclose(win._bound_spins["kz1"].value(), initial_kz_bounds[1])


def test_ktool_resolution_estimate_uses_current_work_function(qtbot, anglemap) -> None:
    data = anglemap.copy().assign_coords(hv=6.0)
    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    initial_kx_resolution = win._resolution_spins["kx"].value()
    win._offset_spins["wf"].setValue(3.0)
    win.calculate_resolution()

    expected_data = win._assign_params(win.data.copy())
    for axis in win.data.kspace.momentum_axes:
        spin = win._resolution_spins[axis]
        expected = expected_data.kspace.estimate_resolution(
            axis, from_numpoints=win.res_npts_check.isChecked()
        )
        assert np.isclose(spin.value(), np.round(expected, spin.decimals()))

    assert not np.isclose(win._resolution_spins["kx"].value(), initial_kx_resolution)


def test_ktool_shallow_copy_paths_do_not_mutate_tool_data_attrs(
    qtbot, anglemap
) -> None:
    data = anglemap.copy().assign_coords(hv=21.2)
    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    original_attrs = win.data.attrs.copy()

    delta_spin = win._offset_spins["delta"]
    delta_spin.blockSignals(True)
    delta_spin.setValue(delta_spin.value() + 0.7)
    delta_spin.blockSignals(False)

    wf_spin = win._offset_spins["wf"]
    wf_spin.blockSignals(True)
    wf_spin.setValue(wf_spin.value() + 0.15)
    wf_spin.blockSignals(False)

    win.calculate_bounds()
    win.calculate_resolution()
    win.get_data()

    assert win.data.attrs == original_attrs


def test_ktool_nonphysical_kinetic_energy_raises_with_tool_context(
    qtbot, anglemap
) -> None:
    data = anglemap.copy().assign_coords(hv=6.0)
    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    wf_spin = win._offset_spins["wf"]
    wf_spin.blockSignals(True)
    wf_spin.setValue(9.9)
    wf_spin.blockSignals(False)

    with pytest.raises(
        ValueError,
        match=r"Nonphysical kinetic energy detected while estimating momentum "
        r"resolution in ktool: min\(E_k\)=",
    ):
        win.calculate_resolution()


def test_ktool_descending_energy_axis_preview(qtbot, anglemap) -> None:
    data = anglemap.copy().isel(eV=slice(None, None, -1))
    data = data.assign_coords(eV=data.eV.values[::-1])

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    assert np.isclose(win.center_spin.minimum(), float(data.eV.min()))
    assert np.isclose(win.center_spin.maximum(), float(data.eV.max()))

    win.width_spin.setValue(5)
    center = float(data.eV.values[len(data.eV) // 2])
    win.center_spin.setValue(center)
    center = win.center_spin.value()

    ang, _ = win.get_data()

    idx = int(np.argmin(np.abs(data.eV.values - center)))
    start = max(0, idx - win.width_spin.value() // 2)
    stop = min(data.eV.size, idx + (win.width_spin.value() - 1) // 2 + 1)
    expected_ang = (
        data.isel(eV=slice(start, stop))
        .mean("eV", skipna=True, keep_attrs=True)
        .assign_coords(eV=center)
    )
    expected_ang = win._assign_params(expected_ang)

    xr.testing.assert_allclose(ang, expected_ang)


@pytest.mark.parametrize(
    ("avec", "expected"),
    [
        pytest.param(erlab.lattice.abc2avec(3.5, 3.5, 5.0, 90.0, 90.0, 120.0), 6),
        pytest.param(erlab.lattice.abc2avec(3.5, 3.5, 5.0, 90.0, 90.0, 90.0), 4),
        pytest.param(erlab.lattice.abc2avec(3.5, 4.2, 5.0, 90.0, 90.0, 110.0), 2),
    ],
)
def test_ktool_preview_symmetry_default_fold(qtbot, anglemap, avec, expected) -> None:
    win = ktool(anglemap.qsel(eV=-0.1), avec=avec, execute=False)
    qtbot.addWidget(win)

    assert win.preview_symmetry_fold_spin.value() == expected


def test_ktool_preview_symmetry_is_preview_only(qtbot, anglemap) -> None:
    data = anglemap.qsel(eV=-0.1).copy(deep=True)
    data.values[2, 7] += 500.0

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    raw_k = win.get_data()[1]
    raw_preview = raw_k.T

    assert win.preview_symmetry_group.isEnabled()
    win.preview_symmetry_group.setChecked(True)
    win.update()

    displayed_preview = win.images[1].data_array
    assert displayed_preview is not None
    assert not np.allclose(displayed_preview.values, raw_preview.values, equal_nan=True)
    xr.testing.assert_allclose(win.get_data()[1], raw_k)

    win.show_converted()
    assert win._itool is not None
    xr.testing.assert_allclose(win._itool.slicer_area.data, raw_k)
    win._itool.close()


def test_ktool_preview_symmetry_rotates_and_averages(
    qtbot, anglemap, monkeypatch
) -> None:
    win = ktool(anglemap, execute=False)
    qtbot.addWidget(win)

    dummy_ang = anglemap.qsel(eV=-0.1)
    raw_k = xr.DataArray(
        np.zeros((5, 5), dtype=float),
        dims=("kx", "ky"),
        coords={
            "kx": np.arange(-2.0, 3.0, dtype=float),
            "ky": np.arange(-2.0, 3.0, dtype=float),
        },
    )
    raw_k.loc[{"kx": 1.0, "ky": 0.0}] = 1.0

    expected = xr.DataArray(
        np.zeros((5, 5), dtype=float),
        dims=("ky", "kx"),
        coords={
            "ky": np.arange(-2.0, 3.0, dtype=float),
            "kx": np.arange(-2.0, 3.0, dtype=float),
        },
    )
    for ky, kx in ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)):
        expected.loc[{"ky": ky, "kx": kx}] = 0.25

    monkeypatch.setattr(win, "get_data", lambda: (dummy_ang, raw_k))

    win.preview_symmetry_fold_spin.setValue(4)
    win.preview_symmetry_group.setChecked(True)
    win.update()

    displayed_preview = win.images[1].data_array
    assert displayed_preview is not None
    xr.testing.assert_allclose(displayed_preview, expected)


def test_ktool_preview_symmetry_disabled_for_non_in_plane_preview(qtbot) -> None:
    data = generate_hvdep_cuts((15, 30, 20), hvrange=(20.0, 30.0), noise=False)

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    assert not win.preview_symmetry_group.isEnabled()
    assert not win.preview_symmetry_group.isChecked()

    win.preview_symmetry_group.setChecked(True)
    win.update()

    assert not win.preview_symmetry_group.isChecked()


def test_ktool_exact_hv_bz_overlay_uses_cache(qtbot, monkeypatch) -> None:
    data = generate_hvdep_cuts((15, 30, 20), hvrange=(20.0, 30.0), noise=False)

    call_count = 0
    original = erlab.lattice.get_surface_bz

    def _counting_surface_bz(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(erlab.lattice, "get_surface_bz", _counting_surface_bz)

    win = ktool(
        data,
        avec=erlab.lattice.abc2avec(6.97, 6.97, 8.685, 90, 90, 120),
        rotate_bz=30.0,
        execute=False,
    )
    qtbot.addWidget(win)

    converted = win.images[1].data_array
    assert converted is not None
    kp_dim = next(d for d in converted.dims if d != "kz")
    other = "kx" if kp_dim == "ky" else "ky"

    assert converted[other].ndim == 2
    assert call_count >= 1

    call_count_before = call_count
    lines, vertices, midpoints = win.get_bz_lines()
    assert call_count == call_count_before
    assert len(lines) > 0
    assert all(np.isfinite(line).all() for line in lines)
    assert vertices.size == 0 or np.isfinite(vertices).all()
    assert midpoints.shape == (0, 2)

    win.update_bz()
    assert call_count == call_count_before

    win.points_check.setChecked(not win.points_check.isChecked())
    assert call_count == call_count_before


def test_get_bz_lines_returns_empty_without_converted_slice() -> None:
    stub = SimpleNamespace(
        _avec=np.eye(3),
        centering_combo=SimpleNamespace(currentText=lambda: "P"),
        images=[None, SimpleNamespace(data_array=None)],
    )

    lines, vertices, midpoints = KspaceTool.get_bz_lines(stub)

    assert lines == []
    assert vertices.shape == (0, 2)
    assert midpoints.shape == (0, 2)


def test_get_bz_lines_uses_legacy_hv_path_for_scalar_other_axis(monkeypatch) -> None:
    converted = xr.DataArray(
        np.zeros((3, 4), dtype=float),
        dims=("kz", "kx"),
        coords={
            "kz": [-1.0, 0.0, 1.0],
            "kx": [-0.5, 0.0, 0.5, 1.0],
            "ky": 0.25,
        },
    )
    expected_lines = [np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)]
    expected_vertices = np.array([[0.25, 0.5]], dtype=float)
    expected_midpoints = np.array([[0.5, 0.5]], dtype=float)

    def _mock_get_out_of_plane_bz(
        bvec, *, k_parallel: float, angle: float, bounds, return_midpoints: bool
    ):
        assert np.isclose(k_parallel, 0.25)
        assert np.isclose(angle, 15.0)
        assert bounds == (-0.5, 1.0, -1.0, 1.0)
        assert return_midpoints
        return expected_lines, expected_vertices, expected_midpoints

    monkeypatch.setattr(erlab.lattice, "get_out_of_plane_bz", _mock_get_out_of_plane_bz)

    stub = SimpleNamespace(
        _avec=np.eye(3),
        centering_combo=SimpleNamespace(currentText=lambda: "P"),
        images=[None, SimpleNamespace(data_array=converted)],
        _bz_cache_token=lambda data: ("cache", data.sizes["kx"], data.sizes["kz"]),
        _bz_cache_key=None,
        _bz_cache_value=None,
        rot_spin=SimpleNamespace(value=lambda: 15.0),
        data=SimpleNamespace(kspace=SimpleNamespace(_has_hv=True)),
    )

    lines, vertices, midpoints = KspaceTool.get_bz_lines(stub)

    assert len(lines) == 1
    assert np.allclose(lines[0], expected_lines[0])
    assert np.allclose(vertices, expected_vertices)
    assert np.allclose(midpoints, expected_midpoints)
