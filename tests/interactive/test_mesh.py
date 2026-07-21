import pathlib
import typing

import numpy as np
import pytest
import xarray as xr
from qtpy import QtWidgets

import erlab
from erlab.interactive._mesh import MeshTool, meshtool
from erlab.interactive.imagetool._provenance._model import (
    FileLoadSource,
    FileReplayCall,
    ScriptInput,
    ToolProvenanceSpec,
    file_load,
    parse_tool_provenance_spec,
    script,
)
from erlab.interactive.imagetool._provenance._operations import (
    RemoveMeshOperation,
    ScriptCodeOperation,
)


@pytest.fixture
def meshy_data() -> xr.DataArray:
    height = width = 32
    alpha = np.arange(height)
    ev = np.arange(width)

    yy, xx = np.meshgrid(alpha, ev, indexing="ij")
    base = np.exp(-((yy - height / 2) ** 2 + (xx - width / 2) ** 2) / 200)
    mesh_pattern = 1 + 0.1 * np.cos(2 * np.pi * ev / 8)
    mesh_pattern = np.tile(mesh_pattern, (height, 1))

    data = np.stack([base * mesh_pattern, base * mesh_pattern * 1.05], axis=0)
    return xr.DataArray(
        data,
        dims=("rep", "alpha", "eV"),
        coords={"rep": [0, 1], "alpha": alpha, "eV": ev},
        name="mesh_data",
    )


def test_meshtool_update_and_copy_code(qtbot, meshy_data) -> None:
    first_order = [[16, 16], [16, 20], [16, 12]]
    win: MeshTool = meshtool(meshy_data, data_name="mesh_data", execute=False)
    qtbot.addWidget(win)

    win.order_spin.setValue(1)
    win.n_pad_spin.setValue(0)
    win.roi_hw_spin.setValue(8)
    win.k_spin.setValue(0.1)
    win.feather_spin.setValue(1.0)
    win.method_combo.setCurrentText("constant")

    win.p0_spin0.setValue(first_order[1][0])
    win.p0_spin1.setValue(first_order[1][1])
    win.p1_spin0.setValue(first_order[2][0])
    win.p1_spin1.setValue(first_order[2][1])

    win.set_data_beforecalc()
    win.update()

    qtbot.wait_until(lambda: win._corrected is not None and win._mesh is not None)
    assert win._corrected is not None
    assert win._mesh is not None

    (
        corrected,
        mesh_da,
        _,
        log_mag,
        log_mag_corr,
        _,
        mask,
    ) = erlab.analysis.mesh.remove_mesh(
        meshy_data,
        first_order_peaks=first_order,
        order=1,
        n_pad=0,
        roi_hw=8,
        k=0.1,
        feather=1.0,
        full_output=True,
    )

    xr.testing.assert_identical(win._corrected, corrected)
    xr.testing.assert_identical(win._mesh, mesh_da)

    np.testing.assert_allclose(win.main_fft_image.image, log_mag)
    np.testing.assert_allclose(win.corr_fft_image.image, log_mag_corr)
    np.testing.assert_allclose(win.mask_fft_image.image, mask)

    code = win.copy_code()
    namespace: dict[str, object] = {
        "era": erlab.analysis,
        "mesh_data": meshy_data,
        "corrected": None,
        "mesh": None,
    }
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102

    xr.testing.assert_identical(win._corrected, namespace["corrected"])
    xr.testing.assert_identical(win._mesh, namespace["mesh"])


def test_meshtool_output_provenance_roundtrip_uses_tuple_assignment(
    qtbot, meshy_data
) -> None:
    win: MeshTool = meshtool(meshy_data, data_name="mesh_data", execute=False)
    qtbot.addWidget(win)
    win.order_spin.setValue(1)
    win.n_pad_spin.setValue(0)
    win.roi_hw_spin.setValue(8)
    win.p0_spin0.setValue(16)
    win.p0_spin1.setValue(20)
    win.p1_spin0.setValue(16)
    win.p1_spin1.setValue(12)
    win.update()
    qtbot.wait_until(lambda: win._corrected is not None and win._mesh is not None)

    for output_id, expected_name in (
        (MeshTool.Output.CORRECTED, "corrected"),
        (MeshTool.Output.MESH, "mesh"),
    ):
        spec = win.output_imagetool_provenance(
            output_id,
            typing.cast("xr.DataArray", win.output_imagetool_data(output_id)),
        )
        assert spec is not None
        payload = spec.model_dump(mode="json")
        assert payload["active_name"] == expected_name

        reparsed = parse_tool_provenance_spec(payload)
        assert reparsed is not None
        assert isinstance(reparsed.operations[-1], RemoveMeshOperation)
        assert reparsed.operations[-1].output == expected_name
        xr.testing.assert_identical(
            reparsed.operations[-1].apply(meshy_data),
            typing.cast("xr.DataArray", win.output_imagetool_data(output_id)),
        )
        code = reparsed.display_code()
        assert code is not None
        assert "corrected, mesh =" in code
        assert ")[0]" not in code
        assert ")[1]" not in code
        assert max(map(len, code.splitlines())) <= 88
        namespace: dict[str, object] = {
            "era": erlab.analysis,
            "mesh_data": meshy_data,
        }
        exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
        output = namespace[expected_name]
        assert isinstance(output, xr.DataArray)
        xr.testing.assert_identical(
            typing.cast("xr.DataArray", win.output_imagetool_data(output_id)),
            output,
        )

    corrected_operation = win._mesh_provenance_operation(output="corrected")
    assert corrected_operation.statement_code(
        "mesh", output_name="corrected"
    ).startswith("corrected, mesh_2 =")
    mesh_operation = win._mesh_provenance_operation(output="mesh")
    assert mesh_operation.statement_code("corrected", output_name="mesh").startswith(
        "corrected_2, mesh ="
    )

    corrected_collision_code = corrected_operation.statement_code(
        "mesh",
        output_name="mesh_output",
    )
    corrected_namespace: dict[str, object] = {
        "era": erlab.analysis,
        "mesh": meshy_data,
    }
    exec(  # noqa: S102
        corrected_collision_code,
        {"__builtins__": {}},
        corrected_namespace,
    )
    xr.testing.assert_identical(
        typing.cast("xr.DataArray", corrected_namespace["mesh_output"]),
        corrected_operation.apply(meshy_data),
    )

    mesh_collision_code = mesh_operation.statement_code(
        "corrected",
        output_name="corrected_output",
    )
    mesh_namespace: dict[str, object] = {
        "era": erlab.analysis,
        "corrected": meshy_data,
    }
    exec(mesh_collision_code, {"__builtins__": {}}, mesh_namespace)  # noqa: S102
    xr.testing.assert_identical(
        typing.cast("xr.DataArray", mesh_namespace["corrected_output"]),
        mesh_operation.apply(meshy_data),
    )


def test_remove_mesh_replay_preserves_reserved_tuple_binding(
    meshy_data, tmp_path
) -> None:
    operation = RemoveMeshOperation(
        first_order_peaks=((16, 16), (16, 20), (16, 12)),
        order=1,
        n_pad=0,
        roi_hw=8,
        output="corrected",
    )
    data_path = tmp_path / "mesh_removal_input.nc"
    existing_mesh_path = tmp_path / "existing_mesh.nc"
    existing_mesh = xr.zeros_like(meshy_data) + 2.0
    meshy_data.to_netcdf(data_path)
    existing_mesh.to_netcdf(existing_mesh_path)

    def file_spec(path: pathlib.Path) -> ToolProvenanceSpec:
        load_code = f"derived = xr.load_dataarray({str(path)!r})"
        return file_load(
            start_label=f"Load {path.name}",
            seed_code=load_code,
            file_load_source=FileLoadSource(
                path=str(path),
                loader_label="xarray.load_dataarray",
                loader_text="xarray.load_dataarray",
                kwargs_text="",
                replay_call=FileReplayCall(
                    kind="callable",
                    target="xarray.load_dataarray",
                    selected_index=0,
                ),
                load_code=load_code,
            ),
        )

    provenance = script(
        operation,
        ScriptCodeOperation(
            label="Combine corrected data with an existing mesh array",
            code="result = corrected + mesh",
        ),
        start_label="Start from mesh-removal inputs",
        seed_code="derived = data",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="data",
                label="Mesh-removal input",
                provenance_spec=file_spec(data_path),
            ),
            ScriptInput(
                name="mesh",
                label="Existing mesh array",
                provenance_spec=file_spec(existing_mesh_path),
            ),
        ),
    )
    code = provenance.display_code()
    assert code is not None
    assert "corrected, mesh_2 = era.mesh.remove_mesh(" in code

    namespace: dict[str, object] = {
        "era": erlab.analysis,
        "xr": xr,
    }
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    expected = operation.apply(meshy_data) + existing_mesh
    xr.testing.assert_identical(
        typing.cast("xr.DataArray", namespace["result"]),
        expected,
    )


def test_meshtool_autofind_and_persistence(
    qtbot, meshy_data, tmp_path, monkeypatch
) -> None:
    win: MeshTool = meshtool(meshy_data, data_name="mesh_input", execute=False)
    qtbot.addWidget(win)

    win.n_pad_spin.setValue(0)
    win.set_data_beforecalc()

    valid_peaks = np.array([[16, 16], [16, 20], [16, 12]], dtype=np.intp)
    monkeypatch.setattr(
        erlab.analysis.mesh, "find_peaks", lambda *args, **kwargs: valid_peaks.copy()
    )
    expected_peaks = valid_peaks - win.n_pad_spin.value()
    win.auto_find_peaks()

    assert win.p0_spin0.value() == expected_peaks[1, 0]
    assert win.p0_spin1.value() == expected_peaks[1, 1]
    assert win.p1_spin0.value() == expected_peaks[2, 0]
    assert win.p1_spin1.value() == expected_peaks[2, 1]

    win.order_spin.setValue(2)
    win._update_higher_order_targets()
    expected_higher = erlab.analysis.mesh.higher_order_peaks(
        win.tool_status.first_order_peaks,
        order=2,
        shape=win.main_fft_image.image.shape,
        include_center=False,
        only_upper=False,
    )[2:]

    assert len(win.higher_order_targets) == len(expected_higher)
    for target, expected in zip(win.higher_order_targets, expected_higher, strict=True):
        pos = target.pos()
        assert (pos.y(), pos.x()) == pytest.approx(expected)

    warnings: list[str] = []

    def _fake_warning(parent, title, text):
        warnings.append(text)
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_fake_warning))
    win.save_mesh()
    assert warnings
    assert "No mesh data" in warnings[-1]

    state_file = tmp_path / "meshtool_state.h5"
    saved_code = win.copy_code()
    win.to_file(state_file)

    restored = erlab.interactive.utils.ToolWindow.from_file(state_file)
    qtbot.addWidget(restored)
    assert isinstance(restored, MeshTool)

    assert restored.tool_status == win.tool_status
    assert restored.data_name == "mesh_input"
    assert restored.copy_code() == saved_code
    xr.testing.assert_identical(restored.tool_data, win.tool_data)

    namespace: dict[str, object] = {
        "era": erlab.analysis,
        "mesh_input": meshy_data,
    }
    exec(restored.copy_code(), {"__builtins__": {}}, namespace)  # noqa: S102
    corrected, mesh = erlab.analysis.mesh.remove_mesh(
        meshy_data,
        **restored.get_params_dict(),
    )
    xr.testing.assert_identical(
        typing.cast("xr.DataArray", namespace["corrected"]), corrected
    )
    xr.testing.assert_identical(typing.cast("xr.DataArray", namespace["mesh"]), mesh)


def test_meshtool_deferred_restore_queues_initial_preview(
    qtbot, meshy_data, monkeypatch
) -> None:
    win: MeshTool = meshtool(meshy_data, data_name="mesh_input", execute=False)
    qtbot.addWidget(win)
    saved = win.to_dataset()
    calls: list[tuple[MeshTool, bool]] = []
    original = MeshTool.set_data_beforecalc

    def _tracked_set_data_beforecalc(self: MeshTool, initial: bool = False) -> None:
        calls.append((self, initial))
        original(self, initial=initial)

    monkeypatch.setattr(
        MeshTool,
        "set_data_beforecalc",
        _tracked_set_data_beforecalc,
    )
    monkeypatch.setattr(
        erlab.interactive.utils.varname,
        "argname",
        lambda *_args, **_kwargs: pytest.fail(
            "deferred meshtool restore should not inspect the caller frame"
        ),
    )

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved,
        _defer_restore_work=True,
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, MeshTool)
    assert calls == []

    restored.show()

    qtbot.wait_until(lambda: calls == [(restored, True)], timeout=5000)


def test_meshtool_autofind_invalid_peaks_warns_and_keeps_values(
    qtbot, meshy_data, monkeypatch
) -> None:
    win: MeshTool = meshtool(meshy_data, execute=False)
    qtbot.addWidget(win)

    win.n_pad_spin.setValue(0)
    win.set_data_beforecalc()

    win.p0_spin0.setValue(8)
    win.p0_spin1.setValue(12)
    win.p1_spin0.setValue(10)
    win.p1_spin1.setValue(20)

    def _bad_find_peaks(*args, **kwargs):
        return np.array([[16, 16], [0, 0], [0, 0]], dtype=np.intp)

    monkeypatch.setattr(erlab.analysis.mesh, "find_peaks", _bad_find_peaks)

    warnings: list[tuple[str, str]] = []

    def _fake_warning(parent, title, text):
        warnings.append((title, text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_fake_warning))

    win.auto_find_peaks()

    assert warnings
    assert warnings[-1][0] == "Peak detection failed"
    assert "distinct first-order peaks" in warnings[-1][1]
    assert win.p0_spin0.value() == 8
    assert win.p0_spin1.value() == 12
    assert win.p1_spin0.value() == 10
    assert win.p1_spin1.value() == 20


def test_meshtool_autofind_invalid_peaks_reraises_in_manager(
    qtbot, meshy_data, monkeypatch
) -> None:
    win: MeshTool = meshtool(meshy_data, execute=False)
    qtbot.addWidget(win)

    win.n_pad_spin.setValue(0)
    win.set_data_beforecalc()

    win.p0_spin0.setValue(8)
    win.p0_spin1.setValue(12)
    win.p1_spin0.setValue(10)
    win.p1_spin1.setValue(20)

    def _bad_find_peaks(*args, **kwargs):
        return np.array([[16, 16], [0, 0], [0, 0]], dtype=np.intp)

    def _unexpected_warning(*args, **kwargs):
        raise AssertionError("Standalone warning dialog should not be used in manager.")

    monkeypatch.setattr(erlab.analysis.mesh, "find_peaks", _bad_find_peaks)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", staticmethod(_unexpected_warning)
    )
    monkeypatch.setattr(win, "_is_in_manager", lambda: True)

    with pytest.raises(ValueError, match="distinct first-order peaks"):
        win.auto_find_peaks()

    assert win.p0_spin0.value() == 8
    assert win.p0_spin1.value() == 12
    assert win.p1_spin0.value() == 10
    assert win.p1_spin1.value() == 20


def test_meshtool_update_data_preserves_state(qtbot, meshy_data) -> None:
    win: MeshTool = meshtool(meshy_data, execute=False)
    qtbot.addWidget(win)

    win.order_spin.setValue(2)
    win.n_pad_spin.setValue(1)
    win.roi_hw_spin.setValue(6)
    win.k_spin.setValue(0.2)
    win.feather_spin.setValue(0.5)
    win.undo_edge_correction_check.setChecked(True)
    win.method_combo.setCurrentText("gaussian")
    win.p0_spin0.setValue(12)
    win.p0_spin1.setValue(18)
    win.p1_spin0.setValue(20)
    win.p1_spin1.setValue(10)

    status = win.tool_status
    new_data = meshy_data.copy(deep=True)
    new_data.data = np.asarray(new_data.data) * 1.2
    win.update()
    assert win._corrected is not None
    assert win._mesh is not None

    win.update_data(new_data)

    assert win.tool_status == status
    xr.testing.assert_identical(win.tool_data, new_data)
    assert win._corrected is None
    assert win._mesh is None
    assert win.main_image.data_array is not None


def test_meshtool_undo_redo_state_change(qtbot, meshy_data) -> None:
    win: MeshTool = meshtool(meshy_data, execute=False)
    qtbot.addWidget(win)

    initial = win.tool_status
    win.roi_hw_spin.setValue(initial.roi_hw + 1)

    assert win._flush_pending_history_write()
    assert win.undoable is True
    assert win.tool_status.roi_hw == initial.roi_hw + 1

    win.undo()

    assert win.tool_status == initial
    assert win.redoable is True

    win.redo()

    assert win.tool_status.roi_hw == initial.roi_hw + 1
