# ruff: noqa: F403, F405

from ._common import *


def test_figure_composer_bz_overlay_editor_updates_state(qtbot) -> None:
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    )
    tool = _bz_tool(operation)
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )

    tool.operation_panel.select_section("slice")
    page = tool.operation_panel.editor_stack.currentWidget()
    assert page is not None
    mode_combo = page.findChild(QtWidgets.QComboBox, "figureComposerBZModeCombo")
    angle_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZAngleSpin")
    kz_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZKzSpin")
    kz_absolute_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerBZKzAbsoluteSpin"
    )
    k_parallel_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerBZKParallelSpin"
    )
    bounds_edit = page.findChild(QtWidgets.QLineEdit, "figureComposerBZBoundsEdit")
    assert mode_combo is not None
    assert angle_spin is not None
    assert kz_spin is not None
    assert kz_absolute_spin is not None
    assert k_parallel_spin is not None
    assert bounds_edit is not None
    assert angle_spin.suffix() == "°"
    assert kz_spin.suffix() == " π/c"
    assert kz_absolute_spin.suffix() == " Å⁻¹"
    assert k_parallel_spin.suffix() == " Å⁻¹"

    _activate_combo_text(mode_combo, "Out-of-plane")
    angle_spin.setValue(30.0)
    kz_spin.setValue(0.5)
    assert np.isclose(kz_absolute_spin.value(), np.pi / 2.0)
    kz_absolute_spin.setValue(0.25)
    assert np.isclose(kz_spin.value(), 0.25 / np.pi, atol=5e-5)
    k_parallel_spin.setValue(0.25)
    bounds_edit.setText("-2, 2, -3, 3")
    bounds_edit.editingFinished.emit()

    tool.operation_panel.select_section("lattice")
    page = tool.operation_panel.editor_stack.currentWidget()
    assert page is not None
    a_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZAEdit")
    b_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZBEdit")
    c_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZCEdit")
    alpha_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZAlphaEdit")
    beta_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZBetaEdit")
    gamma_spin = page.findChild(QtWidgets.QDoubleSpinBox, "figureComposerBZGammaEdit")
    centering_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerBZCenteringCombo"
    )
    assert a_spin is not None
    assert b_spin is not None
    assert c_spin is not None
    assert alpha_spin is not None
    assert beta_spin is not None
    assert gamma_spin is not None
    assert centering_combo is not None
    assert a_spin.suffix() == " Å"
    assert b_spin.suffix() == " Å"
    assert c_spin.suffix() == " Å"
    assert alpha_spin.suffix() == "°"
    assert beta_spin.suffix() == "°"
    assert gamma_spin.suffix() == "°"
    c_spin.setValue(4.5)
    _activate_combo_text(centering_combo, "F")

    tool.operation_panel.select_section("style")
    page = tool.operation_panel.editor_stack.currentWidget()
    assert page is not None
    color_edit = page.findChild(QtWidgets.QLineEdit, "figureComposerBZColorEdit")
    line_style_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerBZLineStyleCombo"
    )
    line_width_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerBZLineWidthSpin"
    )
    vertices_check = page.findChild(
        QtWidgets.QCheckBox, "figureComposerBZVerticesCheck"
    )
    vertex_kw_edit = page.findChild(QtWidgets.QLineEdit, "figureComposerBZVertexKwEdit")
    assert color_edit is not None
    assert line_style_combo is not None
    assert line_width_spin is not None
    assert vertices_check is not None
    assert vertex_kw_edit is not None

    color_edit.setText("tab:red")
    color_edit.editingFinished.emit()
    _activate_combo_text(line_style_combo, "-.")
    line_width_spin.setValue(1.5)
    vertices_check.setChecked(True)
    vertex_kw_edit.setText("s=12, color='tab:green'")
    vertex_kw_edit.editingFinished.emit()

    updated = tool.tool_status.operations[0]
    assert updated.bz_mode == "out_of_plane"
    assert updated.bz_angle == 30.0
    assert np.isclose(updated.bz_kz_pi_over_c, 0.25 * 4.5 / np.pi)
    assert updated.bz_k_parallel == 0.25
    assert updated.bz_bounds == (-2.0, 2.0, -3.0, 3.0)
    assert updated.bz_c == 4.5
    assert updated.bz_centering_type == "F"
    assert updated.bz_vertices is True
    assert updated.bz_vertex_kw == {"s": 12, "color": "tab:green"}
    assert (
        figurecomposer_bz_overlay._section_summary(tool, "slice", updated)
        == "OOP, k=0.25 Å⁻¹"
    )
    assert updated.line_kw == {
        "color": "tab:red",
        "linestyle": "-.",
        "linewidth": 1.5,
    }


def test_figure_composer_bz_overlay_state_round_trip() -> None:
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        mode="out_of_plane",
    ).model_copy(
        update={
            "bz_a": 2.0,
            "bz_b": 3.0,
            "bz_c": 4.0,
            "bz_centering_type": "I",
            "bz_angle": 15.0,
            "bz_k_parallel": 0.25,
            "bz_bounds": (-1.0, 1.0, -2.0, 2.0),
            "bz_vertices": True,
            "bz_midpoint_kw": {"color": "tab:blue"},
            "line_kw": {"color": "tab:red"},
        }
    )
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )

    restored = FigureRecipeState.model_validate(recipe.model_dump(mode="json"))

    assert restored.operations[0].kind == FigureOperationKind.BZ_OVERLAY
    assert restored.operations[0].bz_centering_type == "I"
    assert restored.operations[0].bz_bounds == (-1.0, 1.0, -2.0, 2.0)
    assert restored.operations[0].bz_midpoint_kw == {"color": "tab:blue"}
    assert restored.operations[0].line_kw == {"color": "tab:red"}


def test_figure_composer_bz_overlay_render_matches_plotting_helper(qtbot) -> None:
    bounds = (-4.0, 4.0, -4.0, 4.0)
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    ).model_copy(
        update={
            "bz_bounds": bounds,
            "bz_angle": 45.0,
            "bz_vertices": True,
            "line_kw": {"color": "tab:purple", "linewidth": 1.25},
        }
    )
    tool = _bz_tool(operation)
    qtbot.addWidget(tool)

    figure = Figure(figsize=(3, 3))
    FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    bvec = erlab.lattice.to_reciprocal(erlab.lattice.abc2avec(1, 1, 1, 90, 90, 90))
    segments, vertices, _midpoints = erlab.lattice.get_in_plane_bz(
        bvec,
        kz=0.0,
        angle=45.0,
        bounds=bounds,
        return_midpoints=True,
    )
    axis = figure.axes[0]
    _assert_bz_lines_match_segments(axis.lines, segments)
    assert axis.collections
    np.testing.assert_allclose(axis.collections[0].get_offsets(), vertices)
    assert all(line.get_color() == "tab:purple" for line in axis.lines)
    assert all(line.get_linewidth() == 1.25 for line in axis.lines)


def test_figure_composer_bz_overlay_out_of_plane_render_and_code(qtbot) -> None:
    bounds = (-4.0, 4.0, -4.0, 4.0)
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        mode="out_of_plane",
    ).model_copy(
        update={
            "bz_bounds": bounds,
            "bz_k_parallel": 0.25,
            "bz_midpoints": True,
            "bz_vertex_kw": {"s": 11},
            "bz_midpoint_kw": {"s": 13},
            "line_kw": {"color": "tab:orange"},
        }
    )
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 1.0], "ky": [-1.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figure = Figure(figsize=(4, 2))
    FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    bvec = erlab.lattice.to_reciprocal(erlab.lattice.abc2avec(1, 1, 1, 90, 90, 90))
    segments, _vertices, midpoints = erlab.lattice.get_out_of_plane_bz(
        bvec,
        k_parallel=0.25,
        angle=0.0,
        bounds=bounds,
        return_midpoints=True,
    )
    for axis in figure.axes:
        _assert_bz_lines_match_segments(axis.lines, segments)
        assert all(line.get_color() == "tab:orange" for line in axis.lines)
        assert len(axis.collections) == 1
        np.testing.assert_allclose(axis.collections[0].get_offsets(), midpoints)
        assert axis.collections[0].get_sizes()[0] == 13

    code = tool.generated_code()
    assert "for ax in axs.flat:" in code
    namespace = _exec_generated_code(code, {"data": tool.source_data()["data"]})
    assert len(namespace["fig"].axes) == 2
    assert all(axis.lines for axis in namespace["fig"].axes)


def test_figure_composer_bz_overlay_text_parsers() -> None:
    assert figurecomposer_bz_overlay._mode_from_text("not a mode") == "in_plane"
    assert figurecomposer_bz_overlay._format_bounds(None) == ""
    assert figurecomposer_bz_overlay._bounds_from_text("") is None
    assert figurecomposer_bz_overlay._bounds_from_text("-1, 1, -2, 2") == (
        -1.0,
        1.0,
        -2.0,
        2.0,
    )

    for text in ("'not bounds'", "1, 2, 3", "1, 2, 3, object()"):
        with pytest.raises(ValueError, match="four comma-separated numbers"):
            figurecomposer_bz_overlay._bounds_from_text(text)
    with pytest.raises(ValueError, match="four comma-separated numbers"):
        figurecomposer_bz_overlay._bounds_from_text("1, 2, 3, 'bad'")

    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    ).model_copy(update={"bz_kz_pi_over_c": 0.5, "bz_c": 4.0})
    tool = _bz_tool(operation)
    assert figurecomposer_bz_overlay._section_summary(tool, "slice", operation) == (
        f"IP, kz=0.5 π/c; {np.pi / 8:g} Å⁻¹"
    )


def test_figure_composer_bz_overlay_helper_edges(qtbot, monkeypatch) -> None:
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    )
    tool = _bz_tool(operation)
    qtbot.addWidget(tool)
    empty_tool = FigureComposerTool.from_sources(
        {"data": xr.DataArray(np.zeros((2, 2)), dims=("kx", "ky"), name="data")},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(empty_tool)
    assert (
        figurecomposer_bz_overlay._current_bz_operation(empty_tool, operation)
        is operation
    )

    nonprimitive = operation.model_copy(update={"bz_centering_type": "I"})
    assert figurecomposer_bz_overlay._bz_bvec(nonprimitive).shape == (3, 3)
    assert (
        figurecomposer_bz_overlay._single_axis_code(
            tool,
            operation.model_copy(
                update={"axes": FigureAxesSelectionState(expression="axs[:1]")}
            ),
        )
        is None
    )

    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="axis-b",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
    )
    grid_tool = FigureComposerTool(
        xr.DataArray(np.zeros((2, 2)), dims=("kx", "ky"), name="data"),
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=root),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    assert (
        figurecomposer_bz_overlay._single_axis_code(
            grid_tool,
            operation.model_copy(
                update={"axes": FigureAxesSelectionState(axes_ids=("axis-a", "axis-b"))}
            ),
        )
        is None
    )

    monkeypatch.setattr(
        figurecomposer_bz_overlay,
        "_axes_from_selection",
        lambda *_args, **_kwargs: (),
    )
    figurecomposer_bz_overlay._render_bz_overlay(tool, operation, None)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    created = figurecomposer_bz_overlay._create_operation(tool)
    assert created.kind == FigureOperationKind.BZ_OVERLAY
    assert figurecomposer_bz_overlay._section_summary(tool, "unknown", operation) == ""


def test_figure_composer_bz_overlay_generated_code_static_centering(qtbot) -> None:
    primitive = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 0),))
    ).model_copy(update={"bz_bounds": (-20.0, 20.0, -20.0, 20.0)})
    primitive_tool = _bz_tool(primitive)
    qtbot.addWidget(primitive_tool)

    primitive_code = primitive_tool.generated_code()
    assert "to_primitive" not in primitive_code
    primitive_namespace = _exec_generated_code(
        primitive_code, {"data": primitive_tool.source_data()["data"]}
    )
    assert primitive_namespace["fig"].axes[0].lines

    nonprimitive = primitive.model_copy(update={"bz_centering_type": "F"})
    nonprimitive_tool = _bz_tool(nonprimitive)
    qtbot.addWidget(nonprimitive_tool)

    nonprimitive_code = nonprimitive_tool.generated_code()
    assert (
        'avec = erlab.lattice.to_primitive(avec, centering_type="F")'
        in nonprimitive_code
    )
    assert 'if centering_type != "P"' not in nonprimitive_code
    assert "if " not in "\n".join(
        line for line in nonprimitive_code.splitlines() if "to_primitive" in line
    )
    nonprimitive_namespace = _exec_generated_code(
        nonprimitive_code, {"data": nonprimitive_tool.source_data()["data"]}
    )
    assert nonprimitive_namespace["fig"].axes[0].lines


def test_figure_composer_bz_overlay_invalid_axes_block_generated_code(qtbot) -> None:
    operation = FigureOperationState.bz_overlay(
        axes=FigureAxesSelectionState(axes=((0, 1),))
    )
    tool = _bz_tool(operation)
    qtbot.addWidget(tool)

    with pytest.raises(ValueError, match="target axes"):
        tool.generated_code()


def test_figure_composer_bz_overlay_plain_momentum_seeding() -> None:
    in_plane = xr.DataArray(
        np.zeros((3, 4)),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 0.0, 1.0], "ky": [-2.0, -1.0, 0.0, 1.0]},
    )
    kx_kz = xr.DataArray(
        np.zeros((3, 4)),
        dims=("kx", "kz"),
        coords={"kx": [-0.5, 0.0, 0.5], "kz": [-2.0, -1.0, 0.0, 1.0]},
    )
    ky_kz = xr.DataArray(
        np.zeros((2, 3)),
        dims=("ky", "kz"),
        coords={"ky": [0.25, 0.75], "kz": [-1.0, 0.0, 1.0]},
    )

    in_plane_operation = bz_overlay_operation_from_momentum_data(in_plane)
    kx_kz_operation = bz_overlay_operation_from_momentum_data(kx_kz)
    ky_kz_operation = bz_overlay_operation_from_momentum_data(ky_kz)

    assert in_plane_operation is not None
    assert in_plane_operation.bz_mode == "in_plane"
    assert in_plane_operation.bz_bounds == (-1.0, 1.0, -2.0, 1.0)
    assert kx_kz_operation is not None
    assert kx_kz_operation.bz_mode == "out_of_plane"
    assert kx_kz_operation.bz_bounds == (-0.5, 0.5, -2.0, 1.0)
    assert ky_kz_operation is not None
    assert ky_kz_operation.bz_mode == "out_of_plane"
    assert ky_kz_operation.bz_bounds == (0.25, 0.75, -1.0, 1.0)


def test_figure_composer_bz_overlay_seeding_edge_cases() -> None:
    no_momentum = xr.DataArray(
        np.zeros(2),
        dims=("x",),
        coords={"x": [0.0, 1.0]},
    )
    bad_coord = xr.DataArray(
        np.zeros(2),
        dims=("kx",),
        coords={"kx": ["bad", "coord"]},
    )
    nan_coord = xr.DataArray(
        np.zeros(2),
        dims=("kx",),
        coords={"kx": [np.nan, np.nan]},
    )

    assert figurecomposer_seeding._coordinate_bounds(no_momentum, "kx") is None
    assert figurecomposer_seeding._coordinate_bounds(bad_coord, "kx") is None
    assert figurecomposer_seeding._coordinate_bounds(nan_coord, "kx") is None
    assert figurecomposer_seeding._momentum_bounds(nan_coord, "kx", "ky") is None
    assert figurecomposer_seeding._representative_coord_value(no_momentum, "kx") is None
    assert figurecomposer_seeding._representative_coord_value(bad_coord, "kx") is None
    assert figurecomposer_seeding._representative_coord_value(nan_coord, "kx") is None

    in_plane_bad_bounds = xr.DataArray(
        np.zeros((2, 2)),
        dims=("kx", "ky"),
        coords={"kx": [np.nan, np.nan], "ky": [0.0, 1.0]},
    )
    out_of_plane_bad_bounds = xr.DataArray(
        np.zeros((2, 2)),
        dims=("kx", "kz"),
        coords={"kx": [np.nan, np.nan], "kz": [0.0, 1.0]},
    )
    kz_only = xr.DataArray(
        np.zeros(2),
        dims=("kz",),
        coords={"kz": [0.0, 1.0]},
    )

    assert bz_overlay_operation_from_momentum_data(in_plane_bad_bounds) is None
    assert bz_overlay_operation_from_momentum_data(no_momentum) is None
    assert bz_overlay_operation_from_momentum_data(out_of_plane_bad_bounds) is None
    assert bz_overlay_operation_from_momentum_data(kz_only) is None
    assert (
        figurecomposer_seeding._flat_ktool_out_of_plane_value(
            kz_only, FigureOperationState.bz_overlay(mode="out_of_plane")
        )
        is None
    )

    class DisabledStatus:
        bz_enabled = False

    class EnabledStatus:
        bz_enabled = True

    class DisabledKtool:
        tool_status = DisabledStatus()

    class EnabledKtool:
        tool_status = EnabledStatus()

    assert bz_overlay_operation_from_ktool(DisabledKtool(), no_momentum) is None
    assert bz_overlay_operation_from_ktool(EnabledKtool(), no_momentum) is None


def test_figure_composer_bz_overlay_ktool_seeding_snapshot() -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("kz", "kx"),
        coords={"kz": [-1.0, 0.0, 1.0], "kx": [-0.5, 0.0, 0.5, 1.0], "ky": 0.25},
    )

    class FakeStatus:
        bz_enabled = True
        lattice_params = (2.0, 3.0, 4.0, 90.0, 100.0, 110.0)
        centering = "I"
        rot = 15.0
        kz = 0.5
        points = True

    class FakeKtool:
        tool_status = FakeStatus()

    operation = bz_overlay_operation_from_ktool(FakeKtool(), data)

    assert operation is not None
    assert operation.bz_mode == "out_of_plane"
    assert operation.bz_a == 2.0
    assert operation.bz_b == 3.0
    assert operation.bz_c == 4.0
    assert operation.bz_centering_type == "I"
    assert operation.bz_angle == 15.0
    assert operation.bz_kz_pi_over_c == 0.5
    assert operation.bz_k_parallel == 0.25
    assert operation.bz_vertices is True
    assert operation.bz_midpoints is True


def test_figure_composer_bz_overlay_ktool_seeding_uses_fixed_coord_median() -> None:
    ky_values = np.array(
        [
            [np.nan, -0.25, 0.0, 0.25],
            [0.5, 0.75, 1.0, 1.25],
            [1.5, 1.75, 2.0, 2.25],
        ]
    )
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("kz", "kx"),
        coords={
            "kz": [-1.0, 0.0, 1.0],
            "kx": [-0.5, 0.0, 0.5, 1.0],
            "ky": (("kz", "kx"), ky_values),
        },
    )

    class FakeStatus:
        bz_enabled = True
        lattice_params = (2.0, 3.0, 4.0, 90.0, 100.0, 110.0)
        centering = "I"
        rot = 15.0
        kz = 0.5
        points = True

    class FakeKtool:
        tool_status = FakeStatus()

    operation = bz_overlay_operation_from_ktool(FakeKtool(), data)

    assert operation is not None
    assert operation.bz_mode == "out_of_plane"
    assert operation.bz_k_parallel == np.nanmedian(ky_values)


def test_figure_composer_photon_energy_overlay_editor_updates_state(qtbot) -> None:
    data = _photon_energy_source()
    other_data = _photon_energy_source(name="other_kconv")
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0,)})
    tool = _photon_energy_tool(
        operation,
        data,
        extra_source_data={"other_kconv": other_data},
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )

    source_combo = next(
        (
            combo
            for combo in tool.step_source_controls.findChildren(
                QtWidgets.QComboBox, "figureComposerPhotonEnergySourceCombo"
            )
            if combo.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
        ),
        None,
    )
    assert source_combo is not None
    other_index = source_combo.findData("other_kconv")
    assert other_index >= 0
    _activate_combo_index(source_combo, other_index)

    tool.operation_panel.select_section("photon")
    page = tool.operation_panel.editor_stack.currentWidget()
    assert page is not None
    energies_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyValuesEdit"
    )
    binding_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyBindingEnergyEdit"
    )
    assert energies_edit is not None
    assert binding_edit is not None
    assert energies_edit.placeholderText() == "Enter photon energies"
    energies_edit.setText("30, 45, 60")
    energies_edit.editingFinished.emit()
    binding_edit.setText("-0.3")
    binding_edit.editingFinished.emit()

    tool.operation_panel.select_section("style")
    page = tool.operation_panel.editor_stack.currentWidget()
    assert page is not None
    color_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyColorEdit"
    )
    line_style_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerPhotonEnergyLineStyleCombo"
    )
    line_width_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerPhotonEnergyLineWidthSpin"
    )
    legend_check = page.findChild(
        QtWidgets.QCheckBox, "figureComposerPhotonEnergyLegendCheck"
    )
    legend_kw_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyLegendKwEdit"
    )
    label_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerPhotonEnergyLabelTemplateEdit"
    )
    assert color_edit is not None
    assert line_style_combo is not None
    assert line_width_spin is not None
    assert legend_check is not None
    assert legend_kw_edit is not None
    assert label_edit is not None

    color_edit.setText("tab:red")
    color_edit.editingFinished.emit()
    _activate_combo_text(line_style_combo, "--")
    line_width_spin.setValue(1.5)
    legend_check.setChecked(False)
    legend_kw_edit.setText("title='Photon energy', frameon=False")
    legend_kw_edit.editingFinished.emit()
    label_edit.setText("hv={hv:g} eV")
    label_edit.editingFinished.emit()

    updated = tool.tool_status.operations[0]
    assert updated.hv_overlay_source == "other_kconv"
    assert updated.photon_energies == (30.0, 45.0, 60.0)
    assert updated.binding_energy == -0.3
    assert updated.show_legend is False
    assert updated.legend_kw == {"title": "Photon energy", "frameon": False}
    assert updated.label_template == "hv={hv:g} eV"
    assert _operation_section_button(tool, "photon").text() == "hν: 30, 45, 60; eV=-0.3"
    assert updated.line_kw == {
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1.5,
    }


def test_figure_composer_photon_energy_overlay_state_round_trip() -> None:
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(
        update={
            "photon_energies": (30.0, 45.0, 60.0),
            "show_legend": False,
            "legend_kw": {"title": "Photon energy", "frameon": False},
            "label_template": "hv={hv:g}",
            "line_kw": {"color": "tab:green", "linewidth": 1.25},
        }
    )
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(operation,),
        primary_source="hvdep_kconv",
    )

    restored = FigureRecipeState.model_validate(recipe.model_dump(mode="json"))

    assert restored.operations[0].kind == FigureOperationKind.PHOTON_ENERGY_OVERLAY
    assert restored.operations[0].hv_overlay_source == "hvdep_kconv"
    assert restored.operations[0].photon_energies == (30.0, 45.0, 60.0)
    assert restored.operations[0].binding_energy == -0.3
    assert restored.operations[0].show_legend is False
    assert restored.operations[0].legend_kw == {
        "title": "Photon energy",
        "frameon": False,
    }
    assert restored.operations[0].label_template == "hv={hv:g}"
    assert restored.operations[0].line_kw == {
        "color": "tab:green",
        "linewidth": 1.25,
    }


@pytest.mark.parametrize(("configuration", "x_dim"), [(1, "kx"), (2, "ky")])
def test_figure_composer_photon_energy_overlay_render_and_code(
    qtbot, configuration: int, x_dim: str
) -> None:
    data = _photon_energy_source(configuration)
    hv_values = (30.0, 45.0, 60.0)
    binding_energy = -0.3
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=binding_energy,
    ).model_copy(
        update={
            "photon_energies": hv_values,
            "line_kw": {"color": "tab:blue", "linewidth": 1.5},
            "legend_kw": {"title": "Photon energy", "frameon": False},
        }
    )
    tool = _photon_energy_tool(operation, data)
    qtbot.addWidget(tool)

    figure = Figure(figsize=(3, 3))
    FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    expected = data.kspace.hv_to_kz(list(hv_values)).qsel(eV=binding_energy)
    axis = figure.axes[0]
    _assert_photon_energy_lines_match(axis.lines, expected, x_dim)
    assert axis.get_legend() is not None
    assert axis.get_legend().get_title().get_text() == "Photon energy"
    assert axis.get_legend().get_frame_on() is False
    assert all(line.get_color() == "tab:blue" for line in axis.lines)
    assert all(line.get_linewidth() == 1.5 for line in axis.lines)

    code = tool.generated_code()
    assert "import erlab" in code
    assert "kz_values = hvdep_kconv.kspace.hv_to_kz([30, 45, 60]).qsel(eV=-0.3)" in code
    assert "for i in range(len(kz_values.hv)):" in code
    assert f"axs[0, 0].plot(kz.{x_dim}, kz, label=rf" in code
    assert "ax.legend()" not in code
    assert 'axs[0, 0].legend(title="Photon energy", frameon=False)' in code

    namespace = _exec_generated_code(code, {"hvdep_kconv": data})
    generated_axis = namespace["fig"].axes[0]
    _assert_photon_energy_lines_match(generated_axis.lines, expected, x_dim)
    assert generated_axis.get_legend() is not None
    assert generated_axis.get_legend().get_title().get_text() == "Photon energy"
    assert generated_axis.get_legend().get_frame_on() is False


def test_figure_composer_photon_energy_overlay_multiple_axes_code(qtbot) -> None:
    data = _photon_energy_source()
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0, 45.0), "show_legend": False})
    tool = _photon_energy_tool(
        operation,
        data,
        setup=FigureSubplotsState(nrows=1, ncols=2),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "for ax in axs.flat:\n    for i in range(len(kz_values.hv)):" in code
    assert "        ax.plot(kz.kx, kz, label=rf" in code
    assert "legend()" not in code

    namespace = _exec_generated_code(code, {"hvdep_kconv": data})
    expected = data.kspace.hv_to_kz([30.0, 45.0]).qsel(eV=-0.3)
    for axis in namespace["fig"].axes:
        _assert_photon_energy_lines_match(axis.lines, expected, "kx")
        assert axis.get_legend() is None


def test_figure_composer_photon_energy_overlay_custom_label_code(qtbot) -> None:
    data = _photon_energy_source()
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(
        update={
            "photon_energies": (30.0, 45.0),
            "label_template": "hv={hv:g} eV",
            "show_legend": False,
        }
    )
    tool = _photon_energy_tool(operation, data)
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert ".format(hv=kz.hv)" in code
    assert 'rf"$h\\nu' not in code
    assert "legend()" not in code

    namespace = _exec_generated_code(code, {"hvdep_kconv": data})
    expected = data.kspace.hv_to_kz([30.0, 45.0]).qsel(eV=-0.3)
    axis = namespace["fig"].axes[0]
    assert axis.get_legend() is None
    assert [line.get_label() for line in axis.lines] == [
        f"hv={expected.isel(hv=index).hv:g} eV" for index in range(expected.sizes["hv"])
    ]


def test_figure_composer_photon_energy_overlay_expression_axes_code(qtbot) -> None:
    data = _photon_energy_source()
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(expression="axs[0, :]"),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0, 45.0)})
    tool = _photon_energy_tool(
        operation,
        data,
        setup=FigureSubplotsState(nrows=1, ncols=2),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "for ax in axs[0, :]:\n    for i in range(len(kz_values.hv)):" in code
    assert "    ax.legend()" in code

    namespace = _exec_generated_code(code, {"hvdep_kconv": data})
    expected = data.kspace.hv_to_kz([30.0, 45.0]).qsel(eV=-0.3)
    for axis in namespace["fig"].axes:
        _assert_photon_energy_lines_match(axis.lines, expected, "kx")
        assert axis.get_legend() is not None


def test_figure_composer_photon_energy_overlay_blocks_invalid_inputs(qtbot) -> None:
    data = _photon_energy_source()
    missing_values = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    )
    missing_values_tool = _photon_energy_tool(missing_values, data)
    qtbot.addWidget(missing_values_tool)
    with pytest.raises(ValueError, match="at least one photon energy"):
        missing_values_tool.generated_code()

    unselected_energy = missing_values.model_copy(
        update={"photon_energies": (30.0,), "binding_energy": None}
    )
    unselected_tool = _photon_energy_tool(unselected_energy, data)
    qtbot.addWidget(unselected_tool)
    with pytest.raises(ValueError, match="Select a binding energy"):
        unselected_tool.generated_code()

    invalid_axes = missing_values.model_copy(
        update={"axes": FigureAxesSelectionState(axes=((0, 1),))}
    )
    invalid_axes_tool = _photon_energy_tool(invalid_axes, data)
    qtbot.addWidget(invalid_axes_tool)
    with pytest.raises(ValueError, match="target axes"):
        invalid_axes_tool.generated_code()

    invalid_source = xr.DataArray(
        np.zeros((2, 2)),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 1.0], "ky": [-1.0, 1.0]},
        name="hvdep_kconv",
    )
    invalid_source_operation = missing_values.model_copy(
        update={"photon_energies": (30.0,)}
    )
    invalid_source_tool = _photon_energy_tool(invalid_source_operation, invalid_source)
    qtbot.addWidget(invalid_source_tool)
    with pytest.raises(ValueError, match="kx-kz or ky-kz"):
        invalid_source_tool.generated_code()

    no_source_operation = missing_values.model_copy(update={"hv_overlay_source": None})
    no_source_tool = _photon_energy_tool(no_source_operation, data)
    qtbot.addWidget(no_source_tool)
    with pytest.raises(ValueError, match="Select a source"):
        no_source_tool.generated_code()

    missing_source_operation = missing_values.model_copy(
        update={"hv_overlay_source": "missing"}
    )
    missing_source_tool = _photon_energy_tool(missing_source_operation, data)
    qtbot.addWidget(missing_source_tool)
    with pytest.raises(ValueError, match="Source 'missing' is missing"):
        missing_source_tool.generated_code()

    figure = Figure(figsize=(3, 3))
    FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(
        missing_values_tool, figure, sync_visible=False
    )
    assert (
        "at least one photon energy"
        in missing_values_tool._operation_render_errors[missing_values.operation_id]
    )


def test_figure_composer_photon_energy_overlay_helper_edges(qtbot, monkeypatch) -> None:
    assert figurecomposer_photon_energy._optional_float_from_text("") is None
    assert figurecomposer_photon_energy._optional_float_from_text(" -0.3 ") == -0.3
    with pytest.raises(ValueError, match="one number"):
        figurecomposer_photon_energy._optional_float_from_text("bad")
    assert figurecomposer_photon_energy._photon_energies_from_text("30, 45") == (
        30.0,
        45.0,
    )

    data = _photon_energy_source()
    operation = FigureOperationState.photon_energy_overlay(
        source=None,
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0,)})
    tool = _photon_energy_tool(operation, data)
    qtbot.addWidget(tool)

    assert (
        figurecomposer_photon_energy._section_summary(tool, "sources", operation)
        == "none"
    )
    with pytest.raises(ValueError, match="Select a source"):
        figurecomposer_photon_energy._source_data(tool, operation)

    kz_values = xr.DataArray(
        np.zeros((2, 3)),
        dims=("hv", "angle"),
        coords={"hv": [30.0, 45.0], "angle": [0.0, 1.0, 2.0]},
    )
    with pytest.raises(ValueError, match="kx-kz or ky-kz"):
        figurecomposer_photon_energy._photon_x_dim(data, kz_values)

    class FakeKspace:
        slit_axis = "kx"

        @staticmethod
        def hv_to_kz(values: Sequence[float]) -> xr.DataArray:
            return xr.DataArray(
                np.zeros((len(values), 2, 2)),
                dims=("hv", "kx", "temperature"),
                coords={
                    "hv": list(values),
                    "kx": [-1.0, 1.0],
                    "temperature": [20.0, 30.0],
                },
            )

    class FakeKParallelKzData:
        dims = ("kx", "kz")
        kspace = FakeKspace()

        def squeeze(self, *, drop: bool):
            return self

    extra_dim_operation = operation.model_copy(update={"binding_energy": None})
    with pytest.raises(ValueError, match="hv and one momentum dimension"):
        figurecomposer_photon_energy._kz_values(
            FakeKParallelKzData(), extra_dim_operation
        )

    assert (
        figurecomposer_photon_energy._single_axis_code(
            tool,
            operation.model_copy(
                update={"axes": FigureAxesSelectionState(expression="axs[:1]")}
            ),
        )
        is None
    )

    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="axis-b",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
    )
    grid_tool = _photon_energy_tool(
        operation,
        data,
        setup=FigureSubplotsState(
            layout_mode="gridspec",
            gridspec=FigureGridSpecLayoutState(root=root),
        ),
    )
    qtbot.addWidget(grid_tool)
    assert (
        figurecomposer_photon_energy._single_axis_code(
            grid_tool,
            operation.model_copy(
                update={"axes": FigureAxesSelectionState(axes_ids=("axis-a", "axis-b"))}
            ),
        )
        is None
    )

    render_operation = operation.model_copy(update={"hv_overlay_source": "hvdep_kconv"})
    monkeypatch.setattr(
        figurecomposer_photon_energy,
        "_axes_from_selection",
        lambda *_args, **_kwargs: (),
    )
    render_tool = _photon_energy_tool(render_operation, data)
    qtbot.addWidget(render_tool)
    figurecomposer_photon_energy._render_photon_energy_overlay(
        render_tool, render_operation, None
    )


def test_figure_composer_photon_energy_overlay_add_step_seeds_source_and_binding(
    qtbot,
) -> None:
    data = _photon_energy_source()
    plot_operation = FigureOperationState.plot_slices(
        label="hvdep",
        sources=("hvdep_kconv",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(-0.3,),
    )
    tool = FigureComposerTool.from_sources(
        {"hvdep_kconv": data},
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(plot_operation,),
        setup=FigureSubplotsState(),
        primary_source="hvdep_kconv",
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )

    action = next(
        action
        for action in tool.operation_panel.add_step_menu.actions()
        if action.data() == FigureOperationKind.PHOTON_ENERGY_OVERLAY.value
    )
    action.trigger()

    operation = tool.tool_status.operations[-1]
    assert operation.kind == FigureOperationKind.PHOTON_ENERGY_OVERLAY
    assert operation.hv_overlay_source == "hvdep_kconv"
    assert operation.binding_energy == -0.3
    assert operation.photon_energies == ()


def test_figure_composer_photon_energy_overlay_seed_fallbacks(qtbot) -> None:
    data = _photon_energy_source()
    unused_data = _photon_energy_source(name="unused")
    tool = FigureComposerTool.from_sources(
        {"unused": unused_data, "hvdep_kconv": data},
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(),
        setup=FigureSubplotsState(),
        primary_source="unused",
    )
    qtbot.addWidget(tool)

    assert figurecomposer_photon_energy._seed_source(tool) == "hvdep_kconv"
    assert figurecomposer_photon_energy._seed_binding_energy(tool) is None

    primary_tool = FigureComposerTool.from_sources(
        {"hvdep_kconv": data},
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(),
        setup=FigureSubplotsState(),
        primary_source="hvdep_kconv",
    )
    qtbot.addWidget(primary_tool)
    assert figurecomposer_photon_energy._seed_source(primary_tool) == "hvdep_kconv"

    line_operation = FigureOperationState.line(
        label="line", source="hvdep_kconv", axes=FigureAxesSelectionState()
    )
    line_tool = FigureComposerTool.from_sources(
        {"hvdep_kconv": data},
        sources=(FigureSourceState(name="hvdep_kconv", label="hvdep_kconv"),),
        operations=(line_operation,),
        setup=FigureSubplotsState(),
        primary_source="hvdep_kconv",
    )
    qtbot.addWidget(line_tool)
    line_tool.operation_panel.operation_list.setCurrentItem(
        line_tool.operation_panel.operation_list.topLevelItem(0)
    )

    assert figurecomposer_photon_energy._seed_source(line_tool) == "hvdep_kconv"
    assert figurecomposer_photon_energy._seed_binding_energy(line_tool) is None


def test_figure_composer_photon_energy_overlay_tracks_source_ownership(
    qtbot,
) -> None:
    data = _photon_energy_source()
    existing_data = _photon_energy_source(name="hvdep_kconv")
    unused_data = _photon_energy_source(name="unused")
    operation = FigureOperationState.photon_energy_overlay(
        source="hvdep_kconv",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        binding_energy=-0.3,
    ).model_copy(update={"photon_energies": (30.0,)})
    source_tool = _photon_energy_tool(
        operation,
        data,
        extra_source_data={"unused": unused_data},
    )
    destination = _photon_energy_tool(
        FigureOperationState.photon_energy_overlay(
            source="hvdep_kconv",
            axes=FigureAxesSelectionState(axes=((0, 0),)),
            binding_energy=-0.3,
        ).model_copy(update={"photon_energies": (45.0,)}),
        existing_data,
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()

    assert not source_tool._document.source_is_removable("hvdep_kconv")
    assert source_tool._document.source_is_removable("unused")

    _select_operation_rows(source_tool, (0,))
    source_tool.operation_panel.copy_button.click()
    payload = source_tool._clipboard_step_payload()
    assert payload is not None
    _operations, sources, source_data, selection_base_data = payload
    assert [source.name for source in sources] == ["hvdep_kconv"]
    xr.testing.assert_identical(source_data["hvdep_kconv"], data)
    assert selection_base_data == {}

    destination._paste_operations_from_clipboard()

    pasted = destination.tool_status.operations[-1]
    assert pasted.hv_overlay_source == "hvdep_kconv_copy"
    assert [source.name for source in destination.tool_status.sources] == [
        "hvdep_kconv",
        "hvdep_kconv_copy",
    ]
    xr.testing.assert_identical(destination.source_data()["hvdep_kconv"], existing_data)
    xr.testing.assert_identical(destination.source_data()["hvdep_kconv_copy"], data)
