import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize
import xarray as xr

import erlab
from erlab.accessors.kspace import IncompleteDataError, MomentumAccessor
from erlab.constants import AxesConfiguration
from erlab.interactive.imagetool import _kspace_conversion, provenance
from erlab.interactive.imagetool import dialogs as imagetool_dialogs
from erlab.interactive.imagetool.dialogs import KspaceConversionDialog
from erlab.interactive.kspace import KspaceTool, ktool
from erlab.io.exampledata import generate_hvdep_cuts

_MISSING_KSPACE_PARAMETER_WARNINGS = {
    "Work function not found in data attributes, assuming 4.5 eV",
    "Inner potential not found in data attributes, assuming 10 eV",
}
_GIB = 1024**3

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


def _missing_kspace_parameter_warnings(
    caught: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    return [
        warning
        for warning in caught
        if str(warning.message) in _MISSING_KSPACE_PARAMETER_WARNINGS
    ]


def _make_ktool_data(
    anglemap, configuration: AxesConfiguration, coords: dict[str, float]
) -> xr.DataArray:
    data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 3)).copy(
        deep=True
    )
    data.attrs["configuration"] = int(configuration)
    return data.assign_coords(**coords)


def _make_da_ktool_data(anglemap) -> xr.DataArray:
    data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 3)).copy(
        deep=True
    )
    data.attrs["configuration"] = int(AxesConfiguration.Type1DA)
    data.kspace.work_function = 4.5
    return data.assign_coords(xi=0.0, chi=0.0)


def _add_kspace_conversion_dialog(qtbot, slicer_area) -> KspaceConversionDialog:
    dialog = KspaceConversionDialog(slicer_area)
    dialog.setParent(None)
    qtbot.addWidget(dialog)
    return dialog


def _add_hidden_tool(qtbot, widget):
    qtbot.addWidget(widget)
    widget.hide()
    return widget


def _memory_budget(
    *,
    total: int = 64 * _GIB,
    available: int = 48 * _GIB,
) -> _kspace_conversion.KspaceMemoryBudget:
    reserve = min(max(2 * _GIB, int(0.2 * total)), int(0.5 * available))
    return _kspace_conversion.KspaceMemoryBudget(
        total_bytes=total,
        available_bytes=available,
        reserve_bytes=reserve,
        safe_budget_bytes=max(0, available - reserve),
    )


def _conversion_estimate_stub(
    *,
    safe: bool = True,
    final_bytes: int = 8,
    peak_bytes: int = 8,
) -> _kspace_conversion.KspaceConversionEstimate:
    available_bytes = final_bytes if safe else max(0, final_bytes - 1)
    budget = _kspace_conversion.KspaceMemoryBudget(
        total_bytes=max(available_bytes, final_bytes),
        available_bytes=available_bytes,
        reserve_bytes=1,
        safe_budget_bytes=max(0, available_bytes - 1),
    )
    return _kspace_conversion.KspaceConversionEstimate(
        input_dims=(),
        output_dims=(),
        axis_sizes={},
        output_sizes={},
        bounds={},
        resolution={},
        total_points=1,
        final_bytes=final_bytes,
        peak_bytes=peak_bytes,
        memory=budget,
    )


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
    const_energy = anglemap.qsel(eV=-0.1)
    cut_without_beta = cut.drop_vars("beta")
    cut_with_beta_coord = cut_without_beta.assign_coords(
        beta=("alpha", np.linspace(-1.0, 1.0, cut.sizes["alpha"]))
    )
    cut_without_configuration = cut.copy(deep=True)
    del cut_without_configuration.attrs["configuration"]
    data_4d = anglemap.expand_dims("x", 2)
    data_3d_without_alpha = data_4d.qsel(alpha=-8.3)

    assert cut.kspace._interactive_compatible
    assert const_energy.kspace._interactive_compatible

    for data in (
        cut_without_beta,
        cut_with_beta_coord,
        cut_without_configuration,
        data_4d,
        data_3d_without_alpha,
    ):
        with pytest.raises(
            ValueError, match=r"Data is not compatible with the interactive tool."
        ):
            data.kspace.interactive()


def test_kspace_conversion_dialog_requires_configuration(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    data = anglemap.qsel(eV=-0.1).copy(deep=True)
    del data.attrs["configuration"]

    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)

    assert dialog._compatible is False
    assert not hasattr(dialog, "configuration_combo")
    dialog._handle_configuration_changed()
    dialog._update_memory_estimate()
    dialog._set_memory_estimate(_conversion_estimate_stub())
    monkeypatch.setattr(
        imagetool_dialogs.QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: None,
    )
    assert dialog._validate() == imagetool_dialogs.QtWidgets.QDialog.DialogCode.Rejected


def test_kspace_conversion_seed_helpers_cover_nonuniform_guidelines() -> None:
    data = xr.DataArray(
        np.zeros((3, 3)),
        dims=("alpha", "beta"),
        coords={"alpha": [-1.0, 0.0, 1.0], "beta": [10.0, 20.0, 40.0]},
    )
    slicer_area = SimpleNamespace(
        data=data,
        current_values=(0.0, 20.0),
        main_image=SimpleNamespace(
            display_axis=(0, 1),
            is_guidelines_visible=True,
            _guideline_offset=(0.5, 1.5),
            _guideline_angle=12.0,
        ),
        array_slicer=SimpleNamespace(
            _nonuniform_axes_set={1},
            coords_uniform={1: np.array([0.0, 1.0, 2.0])},
            coords={1: np.array([10.0, 20.0, 40.0])},
        ),
    )

    normal_emission, delta = (
        _kspace_conversion.initial_normal_emission_from_slicer_area(slicer_area)
    )

    assert normal_emission == pytest.approx((0.5, 30.0))
    assert delta == pytest.approx(-12.0)


def test_kspace_normal_emission_reports_missing_required_coords() -> None:
    offsets = {"delta": 0.0, "xi": 0.0, "beta": 0.0, "chi": 0.0}
    missing_xi = xr.DataArray(
        np.zeros((2, 2)),
        dims=("alpha", "beta"),
        attrs={"configuration": int(AxesConfiguration.Type1)},
    )
    missing_chi = missing_xi.assign_coords(xi=0.0)
    missing_chi.attrs["configuration"] = int(AxesConfiguration.Type1DA)

    with pytest.raises(IncompleteDataError, match="xi"):
        _kspace_conversion.normal_emission_angles(missing_xi, offsets)
    with pytest.raises(IncompleteDataError, match="chi"):
        _kspace_conversion.normal_emission_angles(missing_chi, offsets)


def test_kspace_conversion_operations_default_source_configuration(anglemap) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5

    operations = _kspace_conversion.kspace_conversion_operations(
        data,
        target_configuration=AxesConfiguration.Type1,
        source_configuration=None,
        work_function=4.5,
        inner_potential=None,
        normal_emission=(1.0, 2.0),
        delta=None,
        bounds=None,
        resolution=None,
        force_scalars=False,
    )

    assert [operation.op for operation in operations] == [
        "kspace_set_normal",
        "kspace_convert",
    ]
    assert operations[0].group is not None
    assert operations[0].group.focus == "normal_emission"
    assert operations[1].group is not None
    assert operations[1].group.focus == "bounds_resolution"


def test_kspace_conversion_operations_include_nondefault_angle_scales(
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    data.kspace.alpha_scale = 1.2
    data.kspace.beta_scale = 0.8

    operations = _kspace_conversion.kspace_conversion_operations(
        data,
        target_configuration=AxesConfiguration.Type1,
        source_configuration=None,
        work_function=4.5,
        inner_potential=None,
        normal_emission=(1.0, 2.0),
        delta=None,
        bounds=None,
        resolution=None,
        force_scalars=False,
    )

    set_normal = operations[0]
    assert isinstance(set_normal, provenance.KspaceSetNormalOperation)
    assert set_normal.alpha_scale == pytest.approx(1.2)
    assert set_normal.beta_scale == pytest.approx(0.8)
    code = set_normal.statement_code("data", output_name="data")
    assert "alpha_scale=1.2" in code
    assert "beta_scale=0.8" in code


@pytest.mark.parametrize("kind", ["cut", "map", "hv"])
def test_kspace_conversion_estimate_matches_safe_output(
    monkeypatch,
    anglemap,
    kind: str,
) -> None:
    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        lambda: _memory_budget(),
    )
    if kind == "cut":
        data = anglemap.isel(alpha=slice(0, 4), eV=slice(0, 5)).qsel(beta=-8.3)
        data = data.copy(deep=True)
    elif kind == "map":
        data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    else:
        data = generate_hvdep_cuts(
            shape=(3, 3, 3),
            configuration=AxesConfiguration.Type1,
        )
        data.kspace.inner_potential = 10.0
    data.kspace.work_function = 4.5

    estimate = _kspace_conversion.estimate_kspace_conversion(data)
    converted = data.kspace.convert()

    assert estimate.output_sizes == dict(converted.sizes)
    assert estimate.total_points == converted.size
    assert estimate.final_bytes == converted.size * 8


def test_kspace_conversion_memory_budget_uses_available_physical_memory(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        _kspace_conversion.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=64 * _GIB, available=10 * _GIB),
    )

    budget = _kspace_conversion.system_memory_budget()

    assert budget.reserve_bytes == 5 * _GIB
    assert budget.safe_budget_bytes == 5 * _GIB


def test_kspace_conversion_estimate_blocks_unsafe_final_array(
    monkeypatch,
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        lambda: _memory_budget(total=4, available=2),
    )

    with pytest.raises(_kspace_conversion.KspaceConversionMemoryError) as exc_info:
        _kspace_conversion.validate_kspace_conversion_memory(
            data,
            bounds={"kx": (-1.0, 1.0), "ky": (-1.0, 1.0)},
            resolution={"kx": 0.0001, "ky": 0.0001},
        )

    assert not exc_info.value.estimate.is_safe


def test_kspace_conversion_estimate_validates_bounds_and_resolution(
    monkeypatch,
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        lambda: _memory_budget(),
    )

    assert _kspace_conversion.validate_kspace_conversion_memory(
        data,
        bounds={"kx": (-1.0, 1.0), "ky": (-1.0, 1.0)},
        resolution={"kx": 1.0, "ky": 1.0},
    ).is_safe
    assert _conversion_estimate_stub(
        safe=True,
        final_bytes=8,
        peak_bytes=48,
    ).is_safe
    estimate_text = _kspace_conversion.kspace_conversion_estimate_text(
        _conversion_estimate_stub(final_bytes=8, peak_bytes=48)
    )
    assert estimate_text.startswith("Output: scalar\n")
    assert "Final array:" in estimate_text
    assert "Available memory:" not in estimate_text
    unsafe_estimate_text = _kspace_conversion.kspace_conversion_estimate_text(
        _conversion_estimate_stub(safe=False, final_bytes=8, peak_bytes=48),
    )
    assert "Open in ImageTool unavailable." in unsafe_estimate_text
    assert "Final array:" in unsafe_estimate_text
    assert "Available memory:" not in unsafe_estimate_text
    preview_estimate_text = _kspace_conversion.kspace_conversion_estimate_text(
        _conversion_estimate_stub(safe=False, final_bytes=8, peak_bytes=48),
        preview=True,
    )
    assert "Preview unavailable." in preview_estimate_text
    assert "Available memory:" not in preview_estimate_text
    with pytest.raises(ValueError, match="finite"):
        _kspace_conversion.estimate_kspace_conversion(
            data,
            bounds={"kx": (np.nan, 1.0)},
        )
    with pytest.raises(ValueError, match="increasing"):
        _kspace_conversion.estimate_kspace_conversion(
            data,
            bounds={"kx": (1.0, -1.0)},
        )
    with pytest.raises(ValueError, match="positive"):
        _kspace_conversion.estimate_kspace_conversion(
            data,
            resolution={"kx": 0.0},
        )


def test_kspace_conversion_estimate_uses_explicit_grid_directly(
    monkeypatch,
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5

    def _unexpected_estimate_bounds(self):
        del self
        raise AssertionError("explicit bounds should not estimate transformed bounds")

    def _unexpected_estimate_resolution(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("explicit resolution should not estimate resolution")

    monkeypatch.setattr(
        MomentumAccessor, "estimate_bounds", _unexpected_estimate_bounds
    )
    monkeypatch.setattr(
        MomentumAccessor,
        "estimate_resolution",
        _unexpected_estimate_resolution,
    )

    estimate = _kspace_conversion.estimate_kspace_conversion(
        data,
        bounds={"kx": (-1.0, 1.0), "ky": (-2.0, 2.0)},
        resolution={"kx": 1.0, "ky": 2.0},
        memory=_memory_budget(),
    )

    assert estimate.axis_sizes == {"kx": 3, "ky": 3}
    assert estimate.output_sizes == {"eV": 3, "kx": 3, "ky": 3}
    assert estimate.total_points == 27
    assert estimate.final_bytes == 27 * 8


def test_kspace_conversion_console_group_stamping_focuses_rows() -> None:
    operations = (
        provenance.KspaceConfigurationOperation(configuration=AxesConfiguration.Type2),
        provenance.KspaceInnerPotentialOperation(inner_potential=12.0),
        provenance.KspaceWorkFunctionOperation(work_function=4.2),
        provenance.KspaceSetNormalOperation(alpha=1.0, beta=2.0, delta=3.0),
        provenance.KspaceConvertOperation(bounds=None, resolution=None),
    )

    stamped = _kspace_conversion.stamp_kspace_conversion_groups(operations)

    assert [
        None if operation.group is None else operation.group.focus
        for operation in stamped
    ] == [
        "configuration",
        "inner_potential",
        "work_function",
        "normal_emission",
        "bounds_resolution",
    ]
    assert (
        _kspace_conversion._focus_for_operation(
            provenance.AverageOperation(dims=("x",))
        )
        is None
    )
    assert (
        _kspace_conversion.stamp_kspace_conversion_groups(
            (
                provenance.KspaceWorkFunctionOperation(work_function=4.2),
                provenance.KspaceWorkFunctionOperation(work_function=4.3),
                provenance.KspaceSetNormalOperation(alpha=1.0, beta=2.0),
                provenance.KspaceConvertOperation(bounds=None, resolution=None),
            )
        )[0].group
        is None
    )


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
    _add_hidden_tool(qtbot, win)
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
        assert "psutil" not in code
        assert "KspaceConversionMemory" not in code
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
    roi.sigRemoveRequested.emit(roi)
    assert win._roi_list == []

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
        _add_hidden_tool(qtbot, win_restored)
        assert isinstance(win_restored, KspaceTool)

        assert win.tool_status == win_restored.tool_status
        assert str(win_restored.info_text) == str(win.info_text)


def test_ktool_angle_energy_cut(qtbot, anglemap) -> None:
    cut = (
        anglemap.isel(alpha=slice(0, 4), eV=slice(0, 5)).qsel(beta=-8.3).copy(deep=True)
    )
    win = cut.kspace.interactive(data_name="cut", execute=False)
    _add_hidden_tool(qtbot, win)

    assert isinstance(win, KspaceTool)
    assert win.energy_group.isEnabled() is False
    assert win.bz_group.isEnabled() is False
    assert win.bz_group.isChecked() is False
    assert win.preview_symmetry_group.isEnabled() is False
    assert win.preview_symmetry_group.isChecked() is False

    assert win.images[0].data_array is not None
    assert win.images[0].data_array.dims == ("eV", "alpha")
    assert win.images[1].data_array is not None
    assert win.images[1].data_array.dims == ("eV", cut.kspace.slit_axis)

    expected = win._converted_output()

    win.show_converted()
    win._itool.show()
    xr.testing.assert_identical(win._itool.slicer_area.data, expected)
    win._itool.close()

    code = win.copy_code()
    namespace = {"cut": cut.copy(deep=True)}
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["cut_kconv"], expected)

    updated = cut.copy(deep=True)
    updated.data = np.asarray(updated.data) + 1.0
    win.update_data(updated)
    assert win.energy_group.isEnabled() is False
    assert win.bz_group.isEnabled() is False
    assert win.images[1].data_array is not None
    assert win.images[1].data_array.dims == ("eV", cut.kspace.slit_axis)


def test_ktool_unsafe_preview_clears_kspace_image(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    assert win.images[1].data_array is not None

    estimate = _conversion_estimate_stub(safe=False, peak_bytes=16)
    estimate_calls = 0
    convert_calls = 0

    def _unsafe_estimate(*args, **kwargs):
        del args, kwargs
        nonlocal estimate_calls
        estimate_calls += 1
        return estimate

    def _unexpected_convert(*args, **kwargs):
        del args, kwargs
        nonlocal convert_calls
        convert_calls += 1
        raise AssertionError("unsafe preview conversion should not run")

    monkeypatch.setattr(
        _kspace_conversion,
        "estimate_kspace_conversion",
        _unsafe_estimate,
    )
    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        lambda: estimate.memory,
    )
    monkeypatch.setattr(MomentumAccessor, "convert", _unexpected_convert)
    win.resolution_supergroup.setChecked(True)
    for spin in win._resolution_spins.values():
        spin.setValue(0.0001)

    win.update()

    qtbot.wait_until(lambda: win.images[1].data_array is None, timeout=2000)
    assert win.images[0].data_array is not None
    assert not win.preview_symmetry_group.isEnabled()
    assert not win.bz_group.isEnabled()
    assert estimate_calls == 1
    win._output_memory_estimate_timer.stop()
    win._flush_output_memory_estimate()
    assert win._memory_estimate_label.property("kspaceMemoryUnsafe") is True
    assert estimate_calls == 2
    assert convert_calls == 0


def test_ktool_preview_memory_estimate_uses_preview_data_synchronously(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    win.resolution_supergroup.setChecked(True)

    preview_data = win._preview_angle_data()
    output_data = win._prepared_output_data()
    large_budget = _memory_budget()
    preview_estimate = _kspace_conversion.estimate_kspace_conversion(
        preview_data,
        bounds=win.bounds,
        resolution=win.resolution,
        memory=large_budget,
    )
    output_estimate = _kspace_conversion.estimate_kspace_conversion(
        output_data,
        bounds=win.bounds,
        resolution=win.resolution,
        memory=large_budget,
    )
    assert output_estimate.total_points > preview_estimate.total_points

    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        lambda: large_budget,
    )
    estimated_dims: list[tuple[str, ...]] = []
    original_estimate = _kspace_conversion.estimate_kspace_conversion

    def _record_estimate(data, **kwargs):
        estimated_dims.append(tuple(data.dims))
        return original_estimate(data, **kwargs)

    monkeypatch.setattr(
        _kspace_conversion,
        "estimate_kspace_conversion",
        _record_estimate,
    )

    win.update()

    assert estimated_dims == [tuple(preview_data.dims)]
    assert estimated_dims[0] != tuple(output_data.dims)
    assert "eV" not in estimated_dims[0]
    win._output_memory_estimate_timer.stop()
    win._flush_output_memory_estimate()
    assert estimated_dims[:2] == [tuple(preview_data.dims), tuple(output_data.dims)]
    assert win._memory_estimate_label.property("kspaceMemoryUnsafe") is False


def test_ktool_preview_memory_estimate_runs_before_preview_conversion(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    win = ktool(anglemap, execute=False)
    _add_hidden_tool(qtbot, win)
    win._invalidate_preview_memory_estimate()
    events: list[str] = []

    def _estimate(*args, **kwargs):
        del args, kwargs
        events.append("estimate")
        return _conversion_estimate_stub(safe=True)

    original_convert = MomentumAccessor.convert

    def _record_convert(self, *args, **kwargs):
        events.append("convert")
        return original_convert(self, *args, **kwargs)

    monkeypatch.setattr(
        _kspace_conversion,
        "estimate_kspace_conversion",
        _estimate,
    )
    monkeypatch.setattr(MomentumAccessor, "convert", _record_convert)

    win.update()

    assert events[:2] == ["estimate", "convert"]


def test_ktool_preview_memory_estimate_cached_for_identical_state(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    win = ktool(anglemap, execute=False)
    _add_hidden_tool(qtbot, win)
    win._invalidate_preview_memory_estimate()
    win._output_memory_estimate_timer.stop()
    monkeypatch.setattr(win, "_queue_output_memory_estimate", lambda **kwargs: None)

    estimate_calls = 0
    budget_calls = 0

    def _estimate(*args, **kwargs):
        del args, kwargs
        nonlocal estimate_calls
        estimate_calls += 1
        return _conversion_estimate_stub(safe=True, peak_bytes=32)

    def _budget():
        nonlocal budget_calls
        budget_calls += 1
        return _memory_budget()

    monkeypatch.setattr(
        _kspace_conversion,
        "estimate_kspace_conversion",
        _estimate,
    )
    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        _budget,
    )

    win.update()
    win.update()
    win.update()

    assert estimate_calls == 1
    assert budget_calls == 2

    win._preview_memory_estimate = None
    win.update()
    assert estimate_calls == 2
    assert budget_calls == 2

    win.resolution_supergroup.setChecked(True)
    win.update()
    assert estimate_calls == 3


def test_ktool_full_output_unsafe_does_not_block_safe_preview(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    win._invalidate_preview_memory_estimate()
    messages: list[tuple[tuple, dict]] = []

    def _estimate(data, **kwargs):
        del kwargs
        if "eV" in data.dims:
            return _conversion_estimate_stub(safe=False, peak_bytes=64)
        return _conversion_estimate_stub(safe=True, peak_bytes=16)

    monkeypatch.setattr(
        _kspace_conversion,
        "estimate_kspace_conversion",
        _estimate,
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args, **kwargs: messages.append((args, kwargs)) or 0),
    )

    win.update()

    assert win.images[1].data_array is not None
    qtbot.wait_until(
        lambda: win._memory_estimate_label.property("kspaceMemoryUnsafe") is True,
        timeout=2000,
    )

    win.show_converted()

    assert messages
    assert win.images[1].data_array is not None
    assert getattr(win, "_itool", None) is None


def test_ktool_memory_label_uses_full_output_estimate(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    preview_data = win._preview_angle_data()
    output_data = win._assign_params(win.data.copy(deep=False))
    large_budget = _memory_budget()
    preview_estimate = _kspace_conversion.estimate_kspace_conversion(
        preview_data,
        bounds=win.bounds,
        resolution=win.resolution,
        memory=large_budget,
    )
    output_estimate = _kspace_conversion.estimate_kspace_conversion(
        output_data,
        bounds=win.bounds,
        resolution=win.resolution,
        memory=large_budget,
    )
    assert output_estimate.final_bytes > preview_estimate.final_bytes

    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        lambda: large_budget,
    )

    win.update()

    assert win.images[1].data_array is not None
    assert win._output_memory_estimate is None
    qtbot.wait_until(lambda: win._output_memory_estimate is not None, timeout=2000)
    assert win._output_memory_estimate is not None
    assert win._output_memory_estimate.final_bytes == output_estimate.final_bytes
    assert win._output_memory_estimate.final_bytes != preview_estimate.final_bytes


def test_ktool_clear_memory_refusal_preview_without_angle_data(qtbot, anglemap) -> None:
    win = ktool(anglemap, execute=False)
    _add_hidden_tool(qtbot, win)

    win._clear_kspace_preview_for_memory_refusal()

    assert win.images[0].data_array is not None
    assert win.images[1].data_array is None
    assert not win.preview_symmetry_group.isEnabled()
    assert not win.bz_group.isEnabled()


def test_ktool_preview_memory_estimate_recomputed_for_slice_change(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    win = ktool(anglemap, execute=False)
    _add_hidden_tool(qtbot, win)
    win._invalidate_preview_memory_estimate()
    win._output_memory_estimate_timer.stop()
    monkeypatch.setattr(win, "_queue_output_memory_estimate", lambda **kwargs: None)
    estimated_dims: list[tuple[str, ...]] = []

    def _estimate(data, **kwargs):
        del kwargs
        estimated_dims.append(tuple(data.dims))
        return _conversion_estimate_stub(safe=True)

    monkeypatch.setattr(
        _kspace_conversion,
        "estimate_kspace_conversion",
        _estimate,
    )

    win.update()
    win.update()
    assert estimated_dims == [("alpha", "beta")]

    win.width_spin.setValue(win.width_spin.value() + 1)
    win.update()

    assert len(estimated_dims) == 2
    assert estimated_dims[-1] == ("alpha", "beta")


def test_kspace_memory_estimate_labels_wrap(qtbot, anglemap) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    dialog_host = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, dialog_host)
    dialog = _add_kspace_conversion_dialog(qtbot, dialog_host.slicer_area)

    for label in (win._memory_estimate_label, dialog._memory_estimate_label):
        assert label.wordWrap()
        assert label.textFormat() == imagetool_dialogs.QtCore.Qt.TextFormat.PlainText
        assert label.alignment() & imagetool_dialogs.QtCore.Qt.AlignmentFlag.AlignTop
        assert label.minimumWidth() == 0
        assert (
            label.sizePolicy().horizontalPolicy()
            == imagetool_dialogs.QtWidgets.QSizePolicy.Policy.Expanding
        )


def test_ktool_unsafe_output_shows_blocking_error(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    data.kspace.work_function = 4.5
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    messages: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args, **kwargs: messages.append((args, kwargs)) or 0),
    )
    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        lambda: _memory_budget(total=4, available=2),
    )

    win.resolution_supergroup.setChecked(True)
    for spin in win._resolution_spins.values():
        spin.setValue(0.0001)

    win.show_converted()

    assert messages
    assert (
        messages[-1][1]["buttons"]
        == imagetool_dialogs.QtWidgets.QDialogButtonBox.StandardButton.Ok
    )
    assert getattr(win, "_itool", None) is None


@pytest.mark.parametrize(
    ("has_hv", "preview_data", "expected"),
    [
        (False, xr.DataArray(np.zeros((2, 3)), dims=("ky", "kx")), True),
        (False, xr.DataArray(np.zeros((2, 3)), dims=("eV", "kx")), False),
        (True, xr.DataArray(np.zeros((2, 3)), dims=("kz", "kx")), True),
        (True, xr.DataArray(np.zeros((2, 3)), dims=("ky", "kx")), False),
        (False, xr.DataArray(np.zeros((2, 3, 4)), dims=("eV", "ky", "kx")), False),
    ],
)
def test_ktool_preview_supports_bz(has_hv, preview_data, expected) -> None:
    stub = SimpleNamespace(data=SimpleNamespace(kspace=SimpleNamespace(_has_hv=has_hv)))

    assert KspaceTool._preview_supports_bz(stub, preview_data) is expected


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
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

    assert np.isfinite(win._normal_emission_spins["alpha"].value())
    assert np.isfinite(win._normal_emission_spins["beta"].value())
    assert win._normal_emission_spins["alpha"].value() == pytest.approx(0.0)
    assert win._normal_emission_spins["beta"].value() == pytest.approx(0.0)


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
def test_ktool_initial_normal_emission_seed(
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
    expected = data.copy(deep=True)
    expected.kspace.set_normal(alpha_normal, beta_normal)

    win = ktool(
        data,
        execute=False,
        initial_normal_emission=(alpha_normal, beta_normal),
    )
    _add_hidden_tool(qtbot, win)

    assert win._normal_emission_spins["alpha"].value() == pytest.approx(
        alpha_normal, abs=1e-3
    )
    assert win._normal_emission_spins["beta"].value() == pytest.approx(
        beta_normal, abs=1e-3
    )
    for key in expected.kspace._valid_offset_keys:
        assert win._offset_spins[key].value() == pytest.approx(
            expected.kspace.offsets[key]
        )


def test_ktool_initial_delta_overrides_delta(qtbot, anglemap) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type2, {"xi": -6.5})
    alpha_normal, beta_normal = _solve_normal_emission_angles(
        AxesConfiguration.Type2,
        {"xi": -6.5},
        {"delta": -8.0, "xi": -1.75, "beta": 2.25},
        [-4.75, 2.25],
    )
    expected = data.copy(deep=True)
    expected.kspace.set_normal(alpha_normal, beta_normal, delta=14.5)

    win = ktool(
        data,
        execute=False,
        initial_normal_emission=(alpha_normal, beta_normal),
        initial_delta=14.5,
    )
    _add_hidden_tool(qtbot, win)

    assert win._offset_spins["delta"].value() == pytest.approx(14.5)
    for key in expected.kspace._valid_offset_keys:
        assert win._offset_spins[key].value() == pytest.approx(
            expected.kspace.offsets[key]
        )


@pytest.mark.parametrize("kind", ["cut", "map", "hv"])
def test_kspace_conversion_dialog_code_and_result(qtbot, anglemap, kind) -> None:
    if kind == "cut":
        data = anglemap.qsel(eV=-0.1).copy(deep=True)
    elif kind == "map":
        data = anglemap.isel(alpha=slice(0, 5), beta=slice(0, 5), eV=slice(0, 5)).copy(
            deep=True
        )
    else:
        data = generate_hvdep_cuts((8, 10, 7), hvrange=(20.0, 30.0), noise=False)

    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)

    assert not {"delta", "xi", "beta", "chi"} & set(dialog._offset_spins)
    assert ("V0" in dialog._offset_spins) is data.kspace._has_hv

    operations = dialog.source_operations()
    assert isinstance(operations[-1], provenance.KspaceConvertOperation)
    assert any(isinstance(op, provenance.KspaceSetNormalOperation) for op in operations)
    assert all(
        operation.group is not None
        and operation.group.kind == _kspace_conversion.KSPACE_CONVERSION_GROUP_KIND
        for operation in operations
    )
    assert {operation.group.id for operation in operations if operation.group} == {
        operations[0].group.id
    }
    for index in range(len(operations)):
        assert provenance.operation_group_range(
            operations,
            index,
            kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
        ) == (0, len(operations))
    assert (
        any(
            isinstance(op, provenance.KspaceInnerPotentialOperation)
            for op in operations
        )
        is data.kspace._has_hv
    )

    code = dialog.make_code()
    assert ".copy(deep=False)" not in code
    assert "psutil" not in code
    assert "KspaceConversionMemory" not in code
    if not data.kspace._has_hv:
        assert ".kspace.inner_potential =" not in code

    namespace = {"data": data.copy(deep=True)}
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    xr.testing.assert_allclose(
        dialog.process_data(data.copy(deep=True)),
        namespace["data_kconv"],
    )


def test_kspace_conversion_dialog_unsafe_accept_shows_error(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    data = anglemap.qsel(eV=-0.1).copy(deep=True)
    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)
    messages: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args, **kwargs: messages.append((args, kwargs)) or 0),
    )
    monkeypatch.setattr(
        _kspace_conversion,
        "system_memory_budget",
        lambda: _memory_budget(total=4, available=2),
    )
    dialog.resolution_supergroup.setChecked(True)
    for spin in dialog._resolution_spins.values():
        spin.setValue(0.0001)

    with pytest.raises(_kspace_conversion.KspaceConversionMemoryError) as exc_info:
        dialog.preflight_data(data)

    assert dialog._handle_process_error(exc_info.value)

    assert messages
    assert (
        messages[-1][1]["buttons"]
        == imagetool_dialogs.QtWidgets.QDialogButtonBox.StandardButton.Ok
    )
    assert "Available physical memory" not in messages[-1][1]["informative_text"]
    assert "Peak estimate" not in messages[-1][1]["detailed_text"]
    assert "Reserve" not in messages[-1][1]["detailed_text"]
    assert "Swap" not in messages[-1][1]["detailed_text"]
    assert dialog._memory_estimate_label.property("kspaceMemoryUnsafe") is True


def test_kspace_conversion_dialog_estimate_defensive_paths(
    qtbot,
    monkeypatch,
    anglemap,
) -> None:
    data = _make_da_ktool_data(anglemap)
    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)
    dialog._set_control_configuration(int(AxesConfiguration.Type2DA))

    estimate = dialog.conversion_estimate_for_data(data)

    assert estimate.is_safe
    dialog.preflight_data(data)
    assert not dialog._handle_process_error(RuntimeError("not a memory guard"))

    def _raise_estimate(_data):
        raise RuntimeError("estimate failed")

    monkeypatch.setattr(dialog, "conversion_estimate_for_data", _raise_estimate)
    dialog._update_memory_estimate()

    assert dialog._memory_estimate_label.property("kspaceMemoryUnsafe") is True


def test_kspace_conversion_dialog_seeds_from_newest_ktool(qtbot, anglemap) -> None:
    data = anglemap.qsel(eV=-0.1).copy(deep=True)
    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    first = ktool(data, execute=False)
    second = ktool(data, execute=False)
    _add_hidden_tool(qtbot, first)
    _add_hidden_tool(qtbot, second)
    first._offset_spins["wf"].setValue(3.25)
    second._offset_spins["wf"].setValue(4.75)
    second._offset_spins["delta"].setValue(12.5)
    second._angle_scale_spins["alpha"].setValue(1.2)
    second._angle_scale_spins["beta"].setValue(0.8)
    second._sync_normal_emission_spins()
    second.bounds_supergroup.setChecked(True)
    for value, spin in zip(
        (-0.12345, 0.23456),
        second._bound_spins.values(),
        strict=False,
    ):
        spin.setValue(value)
    second.resolution_supergroup.setChecked(True)
    second.res_npts_check.setChecked(True)
    for value, spin in zip(
        (0.12345, 0.23456),
        second._resolution_spins.values(),
        strict=False,
    ):
        spin.setValue(value)
    win.slicer_area.add_tool_window(first)
    win.slicer_area.add_tool_window(second)

    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)

    assert dialog._offset_spins["wf"].value() == pytest.approx(4.75)
    assert dialog._normal_delta == pytest.approx(12.5)
    assert dialog.normal_emission == pytest.approx(
        second._current_normal_emission_angles()
    )
    assert dialog._control_data.kspace.alpha_scale == pytest.approx(1.2)
    assert dialog._control_data.kspace.beta_scale == pytest.approx(0.8)
    assert dialog.res_npts_check.isChecked() is True
    assert list(dialog._resolution_spins) == list(second._resolution_spins)
    assert dialog.bounds_supergroup.isChecked() is True
    assert [spin.value() for spin in dialog._bound_spins.values()] == pytest.approx(
        [spin.value() for spin in second._bound_spins.values()]
    )
    assert [
        spin.value() for spin in dialog._resolution_spins.values()
    ] == pytest.approx([spin.value() for spin in second._resolution_spins.values()])
    set_normal = next(
        operation
        for operation in dialog.source_operations()
        if isinstance(operation, provenance.KspaceSetNormalOperation)
    )
    assert set_normal.alpha_scale == pytest.approx(1.2)
    assert set_normal.beta_scale == pytest.approx(0.8)


def test_kspace_conversion_dialog_uses_seeded_scales_for_auto_bounds(
    qtbot, anglemap
) -> None:
    data = anglemap.qsel(eV=-0.1).copy(deep=True)
    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    child = ktool(data, execute=False)
    _add_hidden_tool(qtbot, child)
    child._angle_scale_spins["alpha"].setValue(1.7)
    child._angle_scale_spins["beta"].setValue(0.6)
    win.slicer_area.add_tool_window(child)

    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)

    expected_data = dialog._parameterized_data(dialog._control_data.copy(deep=False))
    expected_bounds = expected_data.kspace.estimate_bounds()

    assert [
        dialog._bound_spins[f"{axis}{index}"].value()
        for axis in dialog._control_data.kspace.momentum_axes
        for index in range(2)
    ] == pytest.approx(
        [
            expected_bounds[axis][index]
            for axis in dialog._control_data.kspace.momentum_axes
            for index in range(2)
        ],
        abs=5e-5,
    )


def test_kspace_conversion_dialog_seeds_hv_inner_potential_from_ktool(
    qtbot,
    monkeypatch,
) -> None:
    data = generate_hvdep_cuts((8, 10, 7), hvrange=(20.0, 30.0), noise=False)
    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    child = ktool(data, execute=False)
    _add_hidden_tool(qtbot, child)
    child._offset_spins["V0"].setValue(14.0)
    win.slicer_area.add_tool_window(child)

    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)

    assert dialog._offset_spins["V0"].value() == pytest.approx(14.0)

    manager = SimpleNamespace(
        target_from_slicer_area=lambda slicer_area: "target",
        _node_for_target=lambda target: SimpleNamespace(
            _childtool_indices=("missing", "child")
        ),
        get_childtool=lambda uid: (
            child if uid == "child" else (_ for _ in ()).throw(RuntimeError)
        ),
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    assert dialog._newest_child_ktool() is child
    dialog.focus_operation_group_control("inner_potential")

    operations = provenance.stamp_operation_group(
        (
            provenance.KspaceInnerPotentialOperation(inner_potential=13.0),
            provenance.KspaceSetNormalOperation(
                alpha=1.0,
                beta=2.0,
                delta=3.0,
                alpha_scale=1.1,
                beta_scale=0.9,
            ),
            provenance.KspaceConvertOperation(bounds=None, resolution=None),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )
    dialog.restore_transform_operations(operations)
    assert dialog._offset_spins["V0"].value() == pytest.approx(13.0)
    assert dialog._control_data.kspace.alpha_scale == pytest.approx(1.1)
    assert dialog._control_data.kspace.beta_scale == pytest.approx(0.9)
    with pytest.raises(ValueError, match="complete kspace conversion"):
        dialog.restore_transform_operation(operations[0])


def test_kspace_conversion_dialog_restore_scales_recalculates_auto_bounds(
    qtbot, anglemap
) -> None:
    data = anglemap.qsel(eV=-0.1).copy(deep=True)
    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)

    operations = provenance.stamp_operation_group(
        (
            provenance.KspaceSetNormalOperation(
                alpha=1.0,
                beta=2.0,
                delta=3.0,
                alpha_scale=1.6,
                beta_scale=0.7,
            ),
            provenance.KspaceConvertOperation(bounds=None, resolution=None),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )

    dialog.restore_transform_operations(operations)

    expected_data = dialog._parameterized_data(dialog._control_data.copy(deep=False))
    expected_bounds = expected_data.kspace.estimate_bounds()

    assert [
        dialog._bound_spins[f"{axis}{index}"].value()
        for axis in dialog._control_data.kspace.momentum_axes
        for index in range(2)
    ] == pytest.approx(
        [
            expected_bounds[axis][index]
            for axis in dialog._control_data.kspace.momentum_axes
            for index in range(2)
        ],
        abs=5e-5,
    )


def test_kspace_conversion_dialog_restores_unordered_setup_group(
    qtbot, monkeypatch, anglemap
) -> None:
    data = anglemap.qsel(eV=-0.1).copy(deep=True)
    win = erlab.interactive.itool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    dialog = _add_kspace_conversion_dialog(qtbot, win.slicer_area)
    axes = tuple(dialog._control_data.kspace.momentum_axes)
    bounds = dict.fromkeys(axes, (-0.03, 0.04))
    resolution = dict.fromkeys(axes, 0.02)
    operations = provenance.stamp_operation_group(
        (
            provenance.KspaceWorkFunctionOperation(work_function=4.2),
            provenance.KspaceSetNormalOperation(alpha=1.0, beta=2.0, delta=3.0),
            provenance.KspaceConfigurationOperation(
                configuration=AxesConfiguration.Type1
            ),
            provenance.KspaceConvertOperation(bounds=bounds, resolution=resolution),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )
    unmarked = provenance.strip_operation_groups(operations)

    with pytest.raises(ValueError, match="can only restore one"):
        imagetool_dialogs.DataTransformDialog.restore_transform_operations(
            SimpleNamespace(restore_transform_operation=lambda operation: None),
            operations[:2],
        )
    with pytest.raises(ValueError, match="Invalid kspace configuration"):
        dialog._set_configuration_combo(999)
    with imagetool_dialogs.QtCore.QSignalBlocker(dialog.configuration_combo):
        dialog.configuration_combo.setCurrentIndex(-1)
    assert dialog.current_configuration == dialog._control_data.kspace.configuration
    dialog._set_configuration_combo(int(AxesConfiguration.Type1))
    dialog._handle_configuration_changed()
    assert dialog.current_configuration == AxesConfiguration.Type1

    assert KspaceConversionDialog.operation_group_for_edit(unmarked, 0) is None
    duplicate_normal = provenance.stamp_operation_group(
        (
            provenance.KspaceSetNormalOperation(alpha=1.0, beta=2.0),
            provenance.KspaceSetNormalOperation(alpha=3.0, beta=4.0),
            provenance.KspaceConvertOperation(bounds=None, resolution=None),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )
    assert KspaceConversionDialog.operation_group_for_edit(duplicate_normal, 0) is None
    extra_convert = provenance.stamp_operation_group(
        (
            provenance.KspaceSetNormalOperation(alpha=1.0, beta=2.0),
            provenance.KspaceConvertOperation(bounds=None, resolution=None),
            provenance.KspaceConvertOperation(bounds=None, resolution=None),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )
    assert KspaceConversionDialog.operation_group_for_edit(extra_convert, 0) is None

    for index in range(len(operations)):
        assert KspaceConversionDialog.operation_group_for_edit(
            operations,
            index,
        ) == (0, len(operations))

    with pytest.raises(ValueError, match="complete kspace conversion"):
        dialog.restore_transform_operations(unmarked)
    dialog.restore_transform_operations(operations)

    assert dialog.current_configuration == AxesConfiguration.Type1
    assert dialog._offset_spins["wf"].value() == pytest.approx(4.2)
    assert dialog.normal_emission == pytest.approx((1.0, 2.0))
    assert dialog._normal_delta == pytest.approx(3.0)
    assert dialog.bounds is not None
    for axis, values in bounds.items():
        assert dialog.bounds[axis] == pytest.approx(values)
    assert dialog.resolution is not None
    for axis, value in resolution.items():
        assert dialog.resolution[axis] == pytest.approx(value)
    assert dialog._validate() == imagetool_dialogs.QtWidgets.QDialog.DialogCode.Accepted
    for focus in (
        "configuration",
        "work_function",
        "inner_potential",
        "normal_emission",
        "bounds_resolution",
        None,
    ):
        dialog.focus_operation_group_control(focus)

    monkeypatch.setattr(dialog, "_copy_data_name", lambda: "not a valid name")
    namespace = {"data": data.copy(deep=True)}
    code = dialog.make_code()
    assert code.splitlines()[-1].startswith("data_kconv = data.kspace.convert(")
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    xr.testing.assert_allclose(
        namespace["data_kconv"],
        dialog.process_data(data.copy(deep=True)),
    )

    spec = provenance.full_data(*operations)
    display_code = spec.display_code(parent_data=data)
    assert display_code is not None
    namespace = {"data": data.copy(deep=True)}
    exec(display_code, {"__builtins__": {}}, namespace)  # noqa: S102
    xr.testing.assert_allclose(
        namespace["derived"],
        spec.apply(data.copy(deep=True)),
    )

    class LayoutReturningNone:
        def __init__(self) -> None:
            self._count = 1

        def count(self) -> int:
            return self._count

        def takeAt(self, _index: int) -> None:
            self._count = 0
            return

    KspaceConversionDialog._clear_layout(LayoutReturningNone())


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
    _add_hidden_tool(qtbot, win)

    for key, value in reference_offsets.items():
        win._offset_spins[key].setValue(value)

    code = win.copy_code()
    assert ".kspace.set_normal(" in code
    assert ".kspace.offsets =" not in code

    input_name = str(win._argnames["data"])
    if not erlab.utils.misc._is_valid_identifier(input_name):
        input_name = "data"
    assert ".copy(deep=False)" not in code
    assert code.splitlines()[0].startswith(f"{input_name}.kspace.set_normal(")
    assert f"{input_name}_kconv = {input_name}.kspace.convert(" in code

    namespace = {input_name: data.copy(deep=True)}
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102

    expected_setup = win._assign_params(data.copy(deep=True))
    expected = expected_setup.kspace.convert(
        bounds=win.bounds,
        resolution=win.resolution,
    )

    xr.testing.assert_allclose(expected, namespace[f"{input_name}_kconv"])
    for key, value in expected_setup.kspace.offsets.items():
        assert namespace[input_name].kspace.offsets[key] == pytest.approx(
            value,
            abs=1e-5,
        )


@pytest.mark.parametrize(
    ("target_configuration", "expected_offsets", "expected_axes"),
    [
        (
            AxesConfiguration.Type2,
            {"delta", "xi", "beta", "wf"},
            {"ky"},
        ),
        (
            AxesConfiguration.Type2DA,
            {"delta", "chi", "xi", "wf"},
            {"kx", "ky"},
        ),
    ],
)
def test_ktool_configuration_combo_rebuilds_controls_and_code(
    qtbot,
    anglemap,
    target_configuration: AxesConfiguration,
    expected_offsets: set[str],
    expected_axes: set[str],
) -> None:
    data = _make_da_ktool_data(anglemap)
    win = ktool(data, data_name="scan", execute=False)
    _add_hidden_tool(qtbot, win)

    assert win.configuration_combo.count() == len(AxesConfiguration)
    index = win.configuration_combo.findData(int(target_configuration))
    assert index >= 0
    win.configuration_combo.setCurrentIndex(index)

    assert win.data.kspace.configuration == target_configuration
    assert set(win._offset_spins) == expected_offsets
    assert set(win._resolution_spins) == expected_axes
    assert win.tool_status.configuration == int(target_configuration)

    code = win.copy_code()
    assert f".kspace.as_configuration({int(target_configuration)})" in code
    assert code.count(".copy(deep=False)") == 0
    assert ".kspace.set_normal(" in code
    assert ".kspace.convert(" in code
    assert code.splitlines()[0] == (
        f"scan_kconv = scan.kspace.as_configuration({int(target_configuration)})"
    )
    namespace = {"scan": data.copy(deep=True)}
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    xr.testing.assert_allclose(namespace["scan_kconv"], win._converted_output())
    assert namespace["scan"].kspace.configuration == AxesConfiguration.Type1DA


def test_ktool_configuration_state_edges(qtbot, anglemap) -> None:
    data = _make_da_ktool_data(anglemap)
    win = ktool(data, data_name="scan", execute=False)
    _add_hidden_tool(qtbot, win)

    with imagetool_dialogs.QtCore.QSignalBlocker(win.configuration_combo):
        win.configuration_combo.setCurrentIndex(-1)
    assert win.current_configuration == win.data.kspace.configuration
    win._set_configuration_combo(win.data.kspace.configuration)
    with pytest.raises(ValueError, match="Invalid kspace configuration"):
        win._set_configuration_combo(999)
    win._handle_configuration_changed()

    status = win.tool_status
    win.tool_status = status.model_copy(
        update={
            "configuration": int(AxesConfiguration.Type2DA),
            "offsets": {**status.offsets, "unused": 1.0},
            "angle_scales": {**status.angle_scales, "unused": 1.0},
            "bounds": {**status.bounds, "unused": 1.0},
            "resolution": {**status.resolution, "unused": 1.0},
        }
    )

    assert win.data.kspace.configuration == AxesConfiguration.Type2DA

    class LayoutReturningNone:
        def __init__(self) -> None:
            self._count = 1

        def count(self) -> int:
            return self._count

        def takeAt(self, _index: int) -> None:
            self._count = 0
            return

    KspaceTool._clear_layout(LayoutReturningNone())


def test_ktool_configuration_state_round_trip_output_provenance_and_update_data(
    qtbot, anglemap
) -> None:
    data = _make_da_ktool_data(anglemap)
    win = ktool(data, data_name="scan", execute=False)
    _add_hidden_tool(qtbot, win)
    target_configuration = AxesConfiguration.Type2DA
    win.configuration_combo.setCurrentIndex(
        win.configuration_combo.findData(int(target_configuration))
    )

    converted = win.output_imagetool_data(KspaceTool.Output.CONVERTED)
    assert converted is not None
    spec = win.output_imagetool_provenance(KspaceTool.Output.CONVERTED, converted)
    assert spec is not None
    assert [operation.op for operation in spec.operations] == [
        "kspace_configuration",
        "kspace_set_normal",
        "kspace_convert",
    ]
    code = spec.display_code()
    assert code is not None
    assert ".kspace.as_configuration(4)" in code
    namespace = {"scan": data.copy(deep=True)}
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    xr.testing.assert_allclose(namespace["scan_kconv"], converted)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)
        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        _add_hidden_tool(qtbot, win_restored)
        assert isinstance(win_restored, KspaceTool)
        assert win_restored.data.kspace.configuration == target_configuration
        assert win_restored.tool_status.configuration == int(target_configuration)

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) + 1.0
    win.update_data(updated)
    assert win._source_configuration == int(AxesConfiguration.Type1DA)
    assert win.data.kspace.configuration == target_configuration
    xr.testing.assert_allclose(
        win.tool_data,
        updated.kspace.as_configuration(target_configuration),
    )


def test_ktool_output_provenance_uses_converted_output_name(qtbot) -> None:
    data = generate_hvdep_cuts((15, 30, 20), hvrange=(20.0, 30.0), noise=False)
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    converted = win.output_imagetool_data(KspaceTool.Output.CONVERTED)
    assert converted is not None

    spec = win.output_imagetool_provenance(KspaceTool.Output.CONVERTED, converted)
    assert spec is not None

    input_name = str(win._argnames["data"])
    if not erlab.utils.misc._is_valid_identifier(input_name):
        input_name = "data"

    assert spec.active_name == f"{input_name}_kconv"
    code = spec.display_code()
    assert code is not None
    assert f"{input_name}_kconv" in code
    assert ".kspace.set_normal(" in code
    assert "alpha_scale" not in code
    assert "beta_scale" not in code


def test_ktool_angle_scales_are_set_normal_provenance_kwargs(qtbot, anglemap) -> None:
    data = _make_da_ktool_data(anglemap)
    win = ktool(data, data_name="scan", execute=False)
    _add_hidden_tool(qtbot, win)

    win._angle_scale_spins["alpha"].setValue(1.25)
    win._angle_scale_spins["beta"].setValue(0.75)

    converted = win.output_imagetool_data(KspaceTool.Output.CONVERTED)
    assert converted is not None
    spec = win.output_imagetool_provenance(KspaceTool.Output.CONVERTED, converted)
    assert spec is not None

    set_normal = next(
        operation
        for operation in spec.operations
        if isinstance(operation, provenance.KspaceSetNormalOperation)
    )
    assert set_normal.alpha_scale == pytest.approx(1.25)
    assert set_normal.beta_scale == pytest.approx(0.75)

    code = spec.display_code()
    assert code is not None
    assert ".kspace.set_normal(" in code
    assert "alpha_scale=1.25" in code
    assert "beta_scale=0.75" in code
    assert "angle_scales" not in code
    namespace = {"scan": data.copy(deep=True)}
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    xr.testing.assert_allclose(namespace["scan_kconv"], converted)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)
        restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        _add_hidden_tool(qtbot, restored)
        assert isinstance(restored, KspaceTool)
        assert restored.data.kspace.alpha_scale == pytest.approx(1.25)
        assert restored.data.kspace.beta_scale == pytest.approx(0.75)
        assert restored.tool_status.angle_scales["alpha"] == pytest.approx(1.25)
        assert restored.tool_status.angle_scales["beta"] == pytest.approx(0.75)


def test_ktool_copy_code_aliases_expression_input_names(qtbot) -> None:
    data = generate_hvdep_cuts((15, 30, 20), hvrange=(20.0, 30.0), noise=False)
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    win.set_input_provenance_spec(
        erlab.interactive.imagetool.provenance.script(
            start_label="Start from watched variable 'my_data'",
            seed_code="derived = my_data.astype(np.float64)",
            active_name="derived",
        )
    )

    code = win.copy_code()

    assert "input_data_kconv = my_data.astype(np.float64)" in code
    assert ".copy(deep=False)" not in code
    assert "input_data_kconv.kspace.set_normal(" in code
    assert "input_data_kconv = input_data_kconv.kspace.convert(" in code
    assert "astype(np.float64)_kconv" not in code
    namespace = {"my_data": data.copy(deep=True), "np": np}
    exec(code, {"__builtins__": {}, "np": np}, namespace)  # noqa: S102
    assert "input_data_kconv" in namespace


def test_ktool_copy_code_ignores_parent_provenance_but_keeps_source(qtbot) -> None:
    data = generate_hvdep_cuts((15, 30, 20), hvrange=(20.0, 30.0), noise=False)
    source = provenance.selection(
        provenance.IselOperation(kwargs={"alpha": slice(2, 24)})
    )
    parent_provenance = provenance.selection(
        provenance.IselOperation(kwargs={"hv": slice(0, 5)})
    )
    source_data = source.apply(data)
    win = ktool(source_data, execute=False)
    _add_hidden_tool(qtbot, win)
    win.set_source_binding(source)
    win.set_input_provenance_parent_fetcher(lambda: parent_provenance)

    code = win.copy_code()

    assert "hv=slice" not in code
    assert "alpha=slice" in code
    namespace = {"data": data.copy(deep=True)}
    exec(code, {"__builtins__": {"slice": slice}}, namespace)  # noqa: S102
    expected = win._assign_params(source_data.copy(deep=True)).kspace.convert(
        bounds=win.bounds, resolution=win.resolution
    )
    xr.testing.assert_allclose(expected, namespace["derived_kconv"])


def test_ktool_update_rate_limited(qtbot, anglemap, monkeypatch) -> None:
    win = ktool(anglemap, execute=False)
    _add_hidden_tool(qtbot, win)

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


def test_ktool_deferred_restore_queues_preview_update(
    qtbot, anglemap, monkeypatch
) -> None:
    data = _make_ktool_data(anglemap, AxesConfiguration.Type1, {"xi": 0.0})
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)
    saved = win.to_dataset()
    calls: list[KspaceTool] = []
    original = KspaceTool._update_now

    def _tracked_update_now(self: KspaceTool) -> None:
        calls.append(self)
        original(self)

    monkeypatch.setattr(KspaceTool, "_update_now", _tracked_update_now)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved,
        _defer_restore_work=True,
    )
    _add_hidden_tool(qtbot, restored)
    assert isinstance(restored, KspaceTool)
    assert calls == []

    restored.show()

    qtbot.wait_until(lambda: calls == [restored], timeout=5000)


def test_ktool_deferred_restore_skips_default_calculations(
    qtbot, anglemap, monkeypatch
) -> None:
    data = _make_da_ktool_data(anglemap)
    win = ktool(data, avec=np.eye(2), data_name="scan", execute=False)
    _add_hidden_tool(qtbot, win)
    status = win.tool_status
    win.tool_status = status.model_copy(
        update={
            "offsets": {
                **status.offsets,
                "delta": status.offsets.get("delta", 0.0) + 0.25,
                "wf": 4.75,
            },
            "angle_scales": {"alpha": 1.25, "beta": 0.75},
            "bounds_enabled": True,
            "bounds": {
                key: -0.35 + 0.1 * index for index, key in enumerate(status.bounds)
            },
            "resolution_enabled": True,
            "resolution": {
                key: 0.031 + 0.002 * index
                for index, key in enumerate(status.resolution)
            },
            "bz_enabled": True,
            "cmap_name": "plasma",
            "cmap_gamma": 1.4,
        }
    )
    expected = win.tool_status
    saved = win.to_dataset()

    def fail_argname(*_args, **_kwargs) -> str:
        pytest.fail("deferred ktool restore should not inspect the caller frame")

    def fail_calculate_bounds(_self: KspaceTool) -> None:
        pytest.fail("deferred ktool restore should not calculate default bounds")

    def fail_calculate_resolution(_self: KspaceTool) -> None:
        pytest.fail("deferred ktool restore should not calculate default resolution")

    monkeypatch.setattr(erlab.interactive.utils.varname, "argname", fail_argname)
    monkeypatch.setattr(KspaceTool, "calculate_bounds", fail_calculate_bounds)
    monkeypatch.setattr(KspaceTool, "calculate_resolution", fail_calculate_resolution)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved,
        _defer_restore_work=True,
    )
    _add_hidden_tool(qtbot, restored)
    assert isinstance(restored, KspaceTool)

    restored_status = restored.tool_status
    assert restored_status.data_name == "scan"
    assert restored._argnames["data"] == "scan"
    assert restored_status.offsets == pytest.approx(expected.offsets)
    assert restored_status.angle_scales == pytest.approx(expected.angle_scales)
    assert restored_status.bounds_enabled is True
    assert restored_status.bounds == pytest.approx(expected.bounds)
    assert restored_status.resolution_enabled is True
    assert restored_status.resolution == pytest.approx(expected.resolution)
    assert restored_status.bz_enabled is True
    assert restored_status.cmap_name == "plasma"
    assert restored_status.cmap_gamma == pytest.approx(1.4)


def test_ktool_standalone_and_update_data_calculate_defaults_eager(
    qtbot, anglemap, monkeypatch
) -> None:
    data = _make_da_ktool_data(anglemap)
    calls: list[tuple[str, KspaceTool]] = []
    original_bounds = KspaceTool.calculate_bounds
    original_resolution = KspaceTool.calculate_resolution

    def record_calculate_bounds(self: KspaceTool) -> None:
        calls.append(("bounds", self))
        original_bounds(self)

    def record_calculate_resolution(self: KspaceTool) -> None:
        calls.append(("resolution", self))
        original_resolution(self)

    monkeypatch.setattr(KspaceTool, "calculate_bounds", record_calculate_bounds)
    monkeypatch.setattr(KspaceTool, "calculate_resolution", record_calculate_resolution)

    win = ktool(data, data_name="scan", execute=False)
    _add_hidden_tool(qtbot, win)
    assert calls == [("bounds", win), ("resolution", win)]

    calls.clear()
    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) + 1.0
    win.update_data(updated)
    assert calls == [("bounds", win), ("resolution", win)]


def test_ktool_kinetic_energy_axis_preview(qtbot, anglemap) -> None:
    data = anglemap.copy().assign_coords(hv=6.2)
    data = data.assign_coords(eV=data.hv - data.kspace.work_function + data.eV)

    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

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


def test_ktool_suppresses_missing_kspace_parameter_warnings(qtbot) -> None:
    data = generate_hvdep_cuts((15, 30, 20), hvrange=(20.0, 30.0), noise=False)
    data = data.copy(deep=True)
    data.attrs.pop("sample_workfunction", None)
    data.attrs.pop("inner_potential", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    assert not _missing_kspace_parameter_warnings(caught)
    assert win._offset_spins["wf"].value() == pytest.approx(4.5)
    assert win._offset_spins["V0"].value() == pytest.approx(10.0)
    code = win.copy_code()
    assert ".kspace.work_function =" not in code
    assert ".kspace.inner_potential =" not in code

    original_attrs = win.data.attrs.copy()
    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.01

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        win.calculate_bounds()
        win.calculate_resolution()
        win.get_data()
        win.copy_code()
        win.update_data(updated)

    assert not _missing_kspace_parameter_warnings(caught)
    assert win.data.attrs == original_attrs


def test_ktool_shallow_copy_paths_do_not_mutate_tool_data_attrs(
    qtbot, anglemap
) -> None:
    data = anglemap.copy().assign_coords(hv=21.2)
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

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


def test_ktool_angle_scale_controls_apply_to_preview_and_output(
    qtbot, monkeypatch, anglemap
) -> None:
    data = anglemap.isel(alpha=slice(0, 4), beta=slice(0, 4), eV=slice(0, 3)).copy(
        deep=True
    )
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    with (
        imagetool_dialogs.QtCore.QSignalBlocker(win._angle_scale_spins["alpha"]),
        imagetool_dialogs.QtCore.QSignalBlocker(win._angle_scale_spins["beta"]),
    ):
        win._angle_scale_spins["alpha"].setValue(1.25)
        win._angle_scale_spins["beta"].setValue(0.75)
    win._sync_angle_scales()
    win.bounds_supergroup.setChecked(True)
    win.resolution_supergroup.setChecked(True)
    for axis in win.data.kspace.momentum_axes:
        win._bound_spins[f"{axis}0"].setValue(-0.1)
        win._bound_spins[f"{axis}1"].setValue(0.1)
        win._resolution_spins[axis].setValue(0.1)

    preview = win._preview_angle_data()
    np.testing.assert_allclose(preview.alpha.values, data.alpha.values)
    np.testing.assert_allclose(preview.beta.values, data.beta.values)
    assert preview.kspace.alpha_scale == pytest.approx(1.25)
    assert preview.kspace.beta_scale == pytest.approx(0.75)

    manual_preview = preview.assign_coords(
        alpha=preview.alpha * 1.25,
        beta=preview.beta * 0.75,
    )
    manual_preview.attrs.pop("alpha_scale", None)
    manual_preview.attrs.pop("beta_scale", None)
    preview_k = win._preview_kspace_data(preview)
    expected_preview = manual_preview.kspace.convert(
        bounds=win.bounds,
        resolution=win.resolution,
        silent=True,
    )
    xr.testing.assert_allclose(preview_k, expected_preview)

    convert_inputs: list[xr.DataArray] = []
    original_convert = MomentumAccessor.convert

    def _record_convert(accessor, *args, **kwargs):
        convert_inputs.append(accessor._obj)
        return original_convert(accessor, *args, **kwargs)

    monkeypatch.setattr(MomentumAccessor, "convert", _record_convert)

    win._converted_output()

    assert convert_inputs
    output_input = convert_inputs[-1]
    np.testing.assert_allclose(output_input.alpha.values, data.alpha.values)
    np.testing.assert_allclose(output_input.beta.values, data.beta.values)
    assert output_input.kspace.alpha_scale == pytest.approx(1.25)
    assert output_input.kspace.beta_scale == pytest.approx(0.75)


def test_ktool_nonphysical_kinetic_energy_raises_with_tool_context(
    qtbot, anglemap
) -> None:
    data = anglemap.copy().assign_coords(hv=6.0)
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

    assert win.preview_symmetry_fold_spin.value() == expected


def test_ktool_preview_symmetry_is_preview_only(qtbot, anglemap) -> None:
    data = anglemap.qsel(eV=-0.1).copy(deep=True)
    data.values[2, 7] += 500.0

    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

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
    _add_hidden_tool(qtbot, win)

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
    stub._preview_supports_bz = lambda data: KspaceTool._preview_supports_bz(stub, data)

    lines, vertices, midpoints = KspaceTool.get_bz_lines(stub)

    assert len(lines) == 1
    assert np.allclose(lines[0], expected_lines[0])
    assert np.allclose(vertices, expected_vertices)
    assert np.allclose(midpoints, expected_midpoints)


def test_ktool_update_data_preserves_state(qtbot, anglemap) -> None:
    data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 5)).copy(
        deep=True
    )
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    win.center_spin.setValue(float(data.eV.values[2]))
    win.width_spin.setValue(3)
    win.bounds_supergroup.setChecked(True)
    win.resolution_supergroup.setChecked(True)
    win.preview_symmetry_group.setChecked(True)
    win.preview_symmetry_fold_spin.setValue(4)
    win._offset_spins["delta"].setValue(5.0)
    win._offset_spins["wf"].setValue(4.6)
    win._angle_scale_spins["alpha"].setValue(1.1)
    win._angle_scale_spins["beta"].setValue(0.9)
    if "beta" in win._offset_spins:
        win._offset_spins["beta"].setValue(-1.5)
    win.add_circle_btn.click()
    win._roi_list[0].set_position((0.1, 0.2), 0.25)

    status = win.tool_status
    new_data = data.copy(deep=True)
    new_data.data = np.asarray(new_data.data) * 1.1
    win.update_data(new_data)

    assert win.tool_status == status
    assert win.data.kspace.alpha_scale == pytest.approx(1.1)
    assert win.data.kspace.beta_scale == pytest.approx(0.9)
    expected_tool_data = new_data.copy(deep=True)
    expected_tool_data.kspace.alpha_scale = 1.1
    expected_tool_data.kspace.beta_scale = 0.9
    xr.testing.assert_identical(win.tool_data, expected_tool_data)
    assert win.images[0].data_array is not None
    assert win.images[1].data_array is not None


def test_ktool_undo_redo_colormap_state(qtbot, anglemap) -> None:
    win = ktool(anglemap.qsel(eV=-0.1), execute=False)
    _add_hidden_tool(qtbot, win)
    initial = win.tool_status

    win.gamma_widget.setValue(initial.cmap_gamma + 0.1)

    assert win._flush_pending_history_write()
    assert win.undoable
    assert win.tool_status.cmap_gamma == initial.cmap_gamma + 0.1

    win.undo()
    assert win.tool_status == initial
    assert win.redoable

    win.redo()
    assert win.tool_status.cmap_gamma == initial.cmap_gamma + 0.1


def test_ktool_update_data_with_single_energy_disables_energy_group(
    qtbot, anglemap
) -> None:
    initial = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 5)).copy(
        deep=True
    )
    data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(1, 2)).copy(
        deep=True
    )
    win = ktool(initial, execute=False)
    _add_hidden_tool(qtbot, win)

    win.update_data(data)
    fixed_energy = float(data.eV.values[0])
    assert win.energy_group.isEnabled() is False
    assert win.center_spin.minimum() == pytest.approx(fixed_energy - 0.1, abs=1e-3)
    assert win.center_spin.maximum() == pytest.approx(fixed_energy + 0.1, abs=1e-3)


def test_ktool_update_data_reconnects_energy_controls_after_single_energy(
    qtbot, anglemap, monkeypatch
) -> None:
    data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(1, 2)).copy(
        deep=True
    )
    updated = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 5)).copy(
        deep=True
    )
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    update_calls: list[None] = []
    original_update = win.update

    def _wrapped_update() -> None:
        update_calls.append(None)
        original_update()

    monkeypatch.setattr(win, "update", _wrapped_update)

    win.update_data(updated)
    assert win.energy_group.isEnabled() is True

    update_calls.clear()
    win.center_spin.setValue(float(updated.eV.values[2]))
    qtbot.wait_until(lambda: len(update_calls) > 0, timeout=5000)

    update_calls.clear()
    win.width_spin.setValue(3)
    qtbot.wait_until(lambda: len(update_calls) > 0, timeout=5000)


def test_ktool_update_data_rejects_noninteractive_replacement_for_cut(
    qtbot, anglemap
) -> None:
    data = (
        anglemap.isel(alpha=slice(0, 3), eV=slice(0, 5)).qsel(beta=-8.3).copy(deep=True)
    )
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    with pytest.raises(ValueError, match="not compatible with the interactive tool"):
        win.update_data(data.isel(eV=0))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "_valid_offset_keys",
            ("delta",),
            "incompatible offset coordinates",
        ),
        ("momentum_axes", ("kx",), "incompatible momentum axes"),
        ("configuration", 999, "incompatible analyzer configuration"),
        ("_has_hv", True, "incompatible photon-energy dimensions"),
    ],
)
def test_ktool_validate_update_data_rejects_incompatible_metadata(
    qtbot, anglemap, monkeypatch, field, value, match
) -> None:
    data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 5)).copy(
        deep=True
    )
    win = ktool(data, execute=False)
    _add_hidden_tool(qtbot, win)

    fake_kspace = SimpleNamespace(
        _interactive_compatible=True,
        _valid_offset_keys=tuple(win.data.kspace._valid_offset_keys),
        momentum_axes=tuple(win.data.kspace.momentum_axes),
        configuration=int(win.data.kspace.configuration),
        _has_hv=win.data.kspace._has_hv,
    )
    setattr(fake_kspace, field, value)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "parse_data",
        lambda _: SimpleNamespace(kspace=fake_kspace),
    )

    with pytest.raises(ValueError, match=match):
        win.validate_update_data(object())
