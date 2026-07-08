# ruff: noqa: F403, F405

from ._common import *


def test_figure_composer_operation_modules_use_editor_signal_contract() -> None:
    modules = (
        figurecomposer_custom_code,
        figurecomposer_line_profile,
        figurecomposer_method,
        figurecomposer_photon_energy,
        figurecomposer_plot_slices,
        figurecomposer_set_palette,
    )
    direct_connects: list[str] = []
    for module in modules:
        module_file = module.__file__
        assert module_file is not None
        tree = ast.parse(Path(module_file).read_text())
        direct_connects.extend(
            f"{Path(module_file).name}:{node.lineno}"
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
            )
        )
    assert direct_connects == []


def test_figure_composer_text_helpers_parse_user_inputs() -> None:
    assert figurecomposer_text._float_pair_from_text("") is None
    assert figurecomposer_text._float_pair_from_text("1, 2.5") == (1.0, 2.5)
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._float_pair_from_text("1")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._float_pair_from_text("1, bad")

    assert figurecomposer_text._plot_limit_from_text("") is None
    assert figurecomposer_text._plot_limit_from_text("1.5") == 1.5
    assert figurecomposer_text._plot_limit_from_text("None") is None
    assert figurecomposer_text._plot_limit_from_text("[2]") == 2.0
    assert figurecomposer_text._plot_limit_from_text("(1, 2)") == (1.0, 2.0)
    assert figurecomposer_text._plot_limit_from_text("0, None") == (0.0, None)
    assert figurecomposer_text._plot_limit_from_text("(None, 2)") == (None, 2.0)
    assert figurecomposer_text._limit_pair_from_text("0, None") == (0.0, None)
    assert figurecomposer_text._limit_pair_from_text("") is None
    assert figurecomposer_text._format_plot_limit((0.0, None)) == "0, None"
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="one"):
        figurecomposer_text._plot_limit_from_text("(1, 2, 3)")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="one"):
        figurecomposer_text._plot_limit_from_text("(1, 'bad')")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("1")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("(1,)")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("(1, 'bad')")

    assert figurecomposer_text._float_tuple_from_text("1, 2") == (1.0, 2.0)
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="numbers"):
        figurecomposer_text._float_tuple_from_text("1, bad")
    assert figurecomposer_text._literal_sequence_from_text("") == ()
    assert figurecomposer_text._literal_sequence_from_text("[1, 2]") == (1, 2)
    assert figurecomposer_text._literal_sequence_from_text("(1)") == (1,)
    assert figurecomposer_text._literal_sequence_from_text("'x'") == ("x",)
    assert figurecomposer_text._literal_sequence_from_text("1, 2") == (1, 2)

    assert figurecomposer_text._string_tuple_from_text("") == ()
    assert figurecomposer_text._string_tuple_from_text("alpha, beta") == (
        "alpha",
        "beta",
    )
    assert figurecomposer_text._string_tuple_from_text("('alpha')") == ("alpha",)
    assert figurecomposer_text._string_tuple_from_text("['alpha', 2]") == (
        "alpha",
        "2",
    )
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="text"):
        figurecomposer_text._string_tuple_from_text("(1)")
    assert figurecomposer_text._text_tuple_from_text("a\n\nb") == ("a", "b")
    assert figurecomposer_text._text_tuple_from_text("a\n\nb", preserve_empty=True) == (
        "a",
        "",
        "b",
    )

    assert figurecomposer_text._dict_from_text("") == {}
    assert figurecomposer_text._dict_from_text(
        "a=1, b=slice(0, 2)", allow_slice=True
    ) == {
        "a": 1,
        "b": slice(0, 2),
    }
    assert figurecomposer_text._dict_from_text("{'a': 1}") == {"a": 1}
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("a=")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("a=object()")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("1")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("{1, 2}")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="explicit"):
        figurecomposer_text._dict_from_text("**{'a': 1}")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="explicit"):
        figurecomposer_text._dict_from_text("{**{'a': 1}}")
    with pytest.raises(figurecomposer_text.FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("{alpha: 1}")

    assert figurecomposer_text._format_pair(None) == ""
    assert figurecomposer_text._format_limit_pair(None) == ""
    assert figurecomposer_text._format_plot_limit(2.0) == "2"
    assert (
        figurecomposer_text._format_dim_sizes(
            xr.DataArray(np.zeros((2, 3)), dims=("alpha", "beta"))
        )
        == "alpha=2, beta=3"
    )
    assert figurecomposer_text._format_axes_tuple((), nrows=2, ncols=2) == "none"
    assert figurecomposer_text._selection_value_count(np.arange(6).reshape(2, 3)) == 6
    assert figurecomposer_text._selection_value_count([1, 2]) == 2
    assert figurecomposer_text._selection_value_count(slice(None)) is None


def test_figure_composer_redraw_and_preview_cache_edges(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    render_calls: list[dict[str, object]] = []
    single_shot_calls: list[tuple[int, Callable[[], None]]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda _tool, **kwargs: render_calls.append(kwargs),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda _owner, delay, callback: single_shot_calls.append((delay, callback)),
    )

    tool.auto_redraw_check.setChecked(False)
    assert not tool._maybe_redraw_plot(show_window=True)
    assert tool._auto_redraw_dirty
    assert tool._preview_pixmap_stale
    assert render_calls == []

    tool._queue_preview_render_update()
    assert tool._auto_redraw_dirty
    assert single_shot_calls == []

    tool._run_queued_preview_render_update(tool._preview_render_update_generation)
    assert tool._auto_redraw_dirty
    assert tool._preview_pixmap_stale
    assert render_calls == []

    tool.auto_redraw_check.setChecked(True)
    assert render_calls == [{}]
    assert not tool._auto_redraw_dirty

    tool._redraw_plot_requested()
    assert render_calls[-1] == {}

    assert not tool._saved_tool_window_visible(xr.Dataset())
    assert tool._saved_tool_window_visible(xr.Dataset(attrs={"tool_visible": True}))

    tool.auto_redraw_check.setChecked(False)
    visible_ds = xr.Dataset(attrs={"tool_visible": True})
    tool._queue_post_restore_redraw_if_needed(visible_ds)
    assert single_shot_calls == []
    tool.auto_redraw_check.setChecked(True)
    tool._queue_post_restore_redraw_if_needed(visible_ds)
    assert single_shot_calls[-1][0] == 0
    single_shot_calls[-1][1]()
    assert render_calls[-1] == {"show_window": True}

    assert tool._persisted_preview_cache_pixmap() is None
    preview = QtGui.QPixmap(
        figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.width() + 20,
        figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.height() + 20,
    )
    preview.fill(QtGui.QColor("red"))
    tool._preview_pixmap_cache = preview
    persisted = tool._persisted_preview_cache_pixmap()
    assert persisted is not None
    assert (
        persisted.width()
        <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.width()
    )
    saved = tool._append_persistence_payload(xr.Dataset())
    assert figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR in saved.attrs

    restored = FigureComposerTool(data)
    qtbot.addWidget(restored)
    restored._restore_persisted_preview_cache(
        xr.Dataset(
            attrs={figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR: "bad"}
        )
    )
    assert restored._preview_pixmap_cache is None
    restored._restore_persisted_preview_cache(saved)
    assert restored._preview_pixmap_cache is not None
    assert (
        restored._preview_pixmap_stale
        == saved.attrs[figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_STALE_ATTR]
    )


def test_figure_composer_preview_draw_error_ignores_missing_operation(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._set_preview_draw_error(tool, RuntimeError("boom"))

    assert tool._operation_render_errors == {}


def test_figure_composer_pipeline_codegen_executes(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    profile = xr.DataArray(
        np.arange(2.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(
                FigureSourceState(name="data", label="data"),
                FigureSourceState(name="profile", label="profile"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="left",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.plot_slices(
                    label="right",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                    slice_dim="eV",
                    slice_values=(1.0,),
                ),
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(update={"line_x": "kx", "xlim": (0.25, 0.75)}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="clean_labels",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data, "profile": profile},
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (2,))
    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    profile_coordinate_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_coordinate_combo is not None
    assert profile_values_combo is not None
    assert profile_coordinate_combo.itemData(0) is None
    assert profile_values_combo.itemData(0) is None
    assert profile_coordinate_combo.findData("kx") >= 0
    assert profile_values_combo.findData("kx") >= 0
    _activate_combo_index(
        profile_coordinate_combo, profile_coordinate_combo.findData("kx")
    )
    assert tool.tool_status.operations[2].line_x == "kx"
    _activate_combo_index(profile_coordinate_combo, 0)
    assert tool.tool_status.operations[2].line_x is None
    _activate_combo_index(profile_values_combo, profile_values_combo.findData("kx"))
    assert tool.tool_status.operations[2].line_y == "kx"
    selection_page = tool.step_editor_stack.currentWidget()
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_values_combo is not None
    _activate_combo_index(profile_values_combo, 0)
    assert tool.tool_status.operations[2].line_y is None
    selection_page = tool.step_editor_stack.currentWidget()
    profile_coordinate_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_coordinate_combo is not None
    assert profile_values_combo is not None
    assert profile_coordinate_combo.toolTip()
    assert profile_values_combo.toolTip()
    assert all(
        widget.toolTip() for widget in selection_page.findChildren(QtWidgets.QLineEdit)
    )
    assert all(
        widget.toolTip() for widget in selection_page.findChildren(QtWidgets.QCheckBox)
    )
    tool._select_step_section("view")
    view_page = tool.step_editor_stack.currentWidget()
    data_values_axis_combo = view_page.findChild(
        QtWidgets.QComboBox, "figureComposerDataValuesAxisCombo"
    )
    assert data_values_axis_combo is not None
    assert data_values_axis_combo.toolTip()
    assert all(
        widget.toolTip() for widget in view_page.findChildren(QtWidgets.QLineEdit)
    )
    assert all(
        widget.toolTip() for widget in view_page.findChildren(QtWidgets.QCheckBox)
    )

    _select_operation_rows(tool, (3,))
    assert tool.operation_list.item(3).text() == "eplt.clean_labels"
    assert tool.step_section_buttons["method"].text() == "eplt.clean_labels"
    tool._select_step_section("method")
    erlab_method_page = tool.step_editor_stack.currentWidget()
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QComboBox)
    )
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QPlainTextEdit)
    )
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QLineEdit)
    )

    namespace = {"data": data, "profile": profile}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["axs"].shape == (1, 2)
    assert namespace["axs"][0, 0].get_xlim() == pytest.approx((0.25, 0.75))
