# ruff: noqa: F403, F405

import subprocess
import textwrap

from erlab.interactive._figurecomposer._document import FigureDocument
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError

from ._common import *


def test_figure_document_orders_source_dependencies_and_rejects_cycles() -> None:
    source = xr.DataArray(np.arange(3.0), dims="x", name="source")
    sources = (
        FigureSourceState(name="base"),
        FigureSourceState(name="selected", selection_source="base"),
    )
    document = FigureDocument(
        FigureRecipeState(sources=sources, primary_source="base"),
        source_data={"base": source, "selected": source},
    )

    assert document.source_dependency_names(("selected",)) == ("base", "selected")

    document.recipe = document.recipe.model_copy(
        update={
            "sources": (
                sources[0].model_copy(update={"selection_source": "selected"}),
                sources[1],
            )
        }
    )
    with pytest.raises(FigureComposerInputError, match="selected -> base -> selected"):
        document.source_dependency_names(("selected",), reject_cycles=True)


def test_figure_document_sources_import_without_qt_widgets() -> None:
    code = textwrap.dedent(
        """
        import sys

        from erlab.interactive._figurecomposer._document import FigureDocument
        from erlab.interactive._figurecomposer._sources import _source_alias_error
        from erlab.interactive._figurecomposer._state import (
            FigureExportState,
            FigureRecipeState,
            FigureSubplotsState,
        )

        recipe = FigureRecipeState(
            setup=FigureSubplotsState(
                figsize=(6.4, 4.8), dpi=100.0, layout=None
            ),
            export=FigureExportState(
                dpi="figure", transparent=False, bbox_inches="tight"
            ),
        )
        assert FigureDocument(recipe)
        assert _source_alias_error("data") is None
        assert "erlab.interactive._figurecomposer._widgets" not in sys.modules
        assert "erlab.interactive._stylesheets" not in sys.modules
        assert "erlab.interactive._figurecomposer._tool" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_figure_document_renames_source_references_atomically() -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="data"),
                FigureSourceState(name="selected", selection_source="data"),
            ),
            operations=(
                FigureOperationState.plot_array(label="array", source="selected"),
                FigureOperationState.custom(
                    label="python", code="result = data + selected", trusted=True
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data, "selected": data},
        source_selection_base_data={"selected": data},
    )

    assert document.source_usage_count("data") == 2
    assert not document.source_is_removable("data")
    assert document.rename_source("data", "renamed")
    assert tuple(document.source_by_name()) == ("renamed", "selected")
    assert document.source_by_name()["renamed"].label == "renamed"
    assert document.source_by_name()["selected"].selection_source == "renamed"
    assert document.recipe.primary_source == "renamed"
    assert tuple(document.source_data) == ("renamed", "selected")
    assert document.recipe.operations[1].code == "result = renamed + selected"

    recipe = document.recipe.model_copy(
        update={
            "operations": (
                FigureOperationState.custom(
                    label="ambiguous", code="renamed = renamed + 1", trusted=True
                ),
            )
        }
    )
    document.recipe = recipe
    before_data = dict(document.source_data)
    with pytest.raises(FigureComposerInputError, match="also binds"):
        document.rename_source("renamed", "other")
    assert document.recipe == recipe
    assert document.source_data == before_data


def test_figure_document_duplicate_move_reorder_and_remove_sources() -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="base"),
                FigureSourceState(name="selected", selection_source="base"),
                FigureSourceState(name="other"),
            ),
            operations=(FigureOperationState.line(label="line", source="other"),),
            primary_source="base",
        ),
        source_data={"base": data, "selected": data, "other": data},
        source_selection_base_data={"selected": data},
    )

    assert document.duplicate_sources(("selected",)) == ("selected_copy",)
    assert document.source_by_name()["selected_copy"].selection_source == "base"
    assert document.source_data["selected_copy"] is not data
    assert document.can_move_sources(("selected_copy",), 1)
    assert document.move_sources(("selected_copy",), 1)
    with pytest.raises(ValueError, match="must be -1 or 1"):
        document.can_move_sources(("selected_copy",), 0)
    with pytest.raises(ValueError, match="must be -1 or 1"):
        document.move_sources(("selected_copy",), 2)
    assert tuple(document.source_by_name()) == (
        "base",
        "selected",
        "other",
        "selected_copy",
    )
    assert document.reorder_sources(("other", "base", "selected", "selected_copy"))
    with pytest.raises(ValueError, match="exact permutation"):
        document.reorder_sources(("other", "base"))

    assert document.remove_sources(("base", "selected", "selected_copy")) == (
        "selected",
        "selected_copy",
        "base",
    )
    assert tuple(document.source_by_name()) == ("other",)
    assert document.recipe.primary_source == "other"
    assert not document.remove_source("other")


def test_figure_document_add_and_replace_recompute_dependents_atomically() -> None:
    base = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("u", "v"),
        coords={"u": [0.0, 1.0], "v": [0.0, 1.0, 2.0]},
        name="base",
    )
    selected = base.qsel(u=1.0)
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="base", node_uid="node"),
                FigureSourceState(
                    name="selected",
                    selection_source="base",
                    qsel={"u": 1.0},
                ),
            ),
            primary_source="base",
        ),
        source_data={"base": base, "selected": selected},
        source_selection_base_data={"selected": base},
    )

    refreshed = base + 10.0
    result = document.add_sources(
        (FigureSourceState(name="incoming", node_uid="node"),),
        {"incoming": refreshed},
    )
    assert result.updated == (("incoming", "base"),)
    xr.testing.assert_identical(document.source_data["selected"], refreshed.qsel(u=1.0))

    before_recipe = document.recipe
    before_data = dict(document.source_data)
    incompatible = refreshed.isel(u=0, drop=True)
    result = document.replace_source(
        "base", FigureSourceState(name="bad", node_uid="other"), incompatible
    )
    assert not result
    assert result.skipped
    assert document.recipe == before_recipe
    for name, expected in before_data.items():
        xr.testing.assert_identical(document.source_data[name], expected)

    replacement = base + 20.0
    result = document.replace_source(
        "base", FigureSourceState(name="replacement", node_uid="other"), replacement
    )
    assert result.updated == ("base",)
    xr.testing.assert_identical(
        document.source_data["selected"], replacement.qsel(u=1.0)
    )


def test_figure_document_updates_source_selection_per_compatible_source() -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims="x",
        coords={"x": [0.0, 1.0, 2.0]},
        name="data",
    )
    scalar = xr.DataArray(1.0, name="scalar")
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="data"),
                FigureSourceState(name="scalar"),
            ),
            primary_source="data",
        ),
        source_data={"data": data, "scalar": scalar},
    )

    result = document.update_source_selection_dimension(
        ("data", "scalar", "missing"), "x", "isel", 1
    )

    assert result.updated == ("data",)
    assert {name for name, _detail in result.skipped} == {"scalar", "missing"}
    assert document.source_by_name()["data"].isel == {"x": 1}
    xr.testing.assert_identical(document.source_data["data"], data.isel(x=1))
    xr.testing.assert_identical(document.source_data["scalar"], scalar)

    recipe = document.recipe
    source_data = dict(document.source_data)
    result = document.update_source_selection_dimension(
        ("data",), "x", "qsel", slice(0.0, 1.0), 0.1
    )
    assert not result
    assert result.skipped
    assert document.recipe == recipe
    for name, expected in source_data.items():
        xr.testing.assert_identical(document.source_data[name], expected)


def test_operation_metadata_covers_every_operation_kind() -> None:
    plot_array = FigureOperationState.plot_array(
        label="array", source="first"
    ).model_copy(update={"sources": ("first", "first", "second")})
    plot_slices = FigureOperationState.plot_slices(
        label="slices",
        sources=("second", "first", "second"),
        map_selections=(FigureDataSelectionState(source="ignored"),),
    )
    line = FigureOperationState.line(label="line", source="ignored").model_copy(
        update={
            "map_selections": (
                FigureDataSelectionState(source="second"),
                FigureDataSelectionState(source="first"),
                FigureDataSelectionState(source="second"),
            )
        }
    )
    photon_energy = FigureOperationState.photon_energy_overlay(source="first")
    method = FigureOperationState.method(
        family=FigureMethodFamily.AXES, name="errorbar"
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_x": FigureMethodPlotValueState(source="first"),
            "method_plot_y": FigureMethodPlotValueState(source="second"),
            "method_plot_xerr": FigureMethodPlotValueState(source="second"),
            "method_plot_yerr": FigureMethodPlotValueState(source="third"),
        }
    )
    custom = FigureOperationState.custom(
        label="custom", code="result = second + first", trusted=False
    )
    operations = (
        FigureOperationState(kind=FigureOperationKind.SET_PALETTE, label="palette"),
        plot_array,
        plot_slices,
        line,
        FigureOperationState.bz_overlay(),
        photon_energy,
        method,
        custom,
    )
    expected = {
        FigureOperationKind.SET_PALETTE: (),
        FigureOperationKind.PLOT_ARRAY: ("first", "second"),
        FigureOperationKind.PLOT_SLICES: ("second", "first"),
        FigureOperationKind.LINE: ("second", "first"),
        FigureOperationKind.BZ_OVERLAY: (),
        FigureOperationKind.PHOTON_ENERGY_OVERLAY: ("first",),
        FigureOperationKind.METHOD: ("first", "second", "third"),
        FigureOperationKind.CUSTOM: (),
    }

    assert {operation.kind for operation in operations} == set(FigureOperationKind)
    assert {
        operation.kind: (
            figurecomposer_operation_metadata.declared_operation_source_names(operation)
        )
        for operation in operations
    } == expected
    assert figurecomposer_operation_metadata.recipe_operation_source_names(
        custom, ("first", "second", "unused")
    ) == ("first", "second")
    assert figurecomposer_operation_metadata.declared_operation_source_names(
        line.model_copy(
            update={"map_selections": (FigureDataSelectionState(source="second"),)}
        )
    ) == ("ignored",)
    assert (
        figurecomposer_operation_metadata.declared_operation_source_names(
            photon_energy.model_copy(update={"hv_overlay_source": None})
        )
        == ()
    )
    for non_source_method in (
        method.model_copy(update={"method_plot_data_mode": "entered"}),
        method.model_copy(update={"method_name": "legend"}),
        method.model_copy(update={"method_family": FigureMethodFamily.FIGURE}),
    ):
        assert (
            figurecomposer_operation_metadata.declared_operation_source_names(
                non_source_method
            )
            == ()
        )

    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="first"),
                FigureSourceState(name="second"),
            ),
            operations=(custom,),
        )
    )
    assert document.operation_source_names(custom) == ("first", "second")
    assert document.direct_sources_used_by_recipe() == {"first", "second"}
    assert document.direct_sources_used_by_recipe(executable_only=True) == set()


def test_figure_composer_operation_modules_use_editor_signal_contract() -> None:
    modules = (
        figurecomposer_custom_code_operation,
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
    assert tool.operation_list.topLevelItem(3).text(0) == "eplt.clean_labels"
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


def test_figure_composer_generated_code_skips_disabled_operations(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.custom(
                    label="disabled Python",
                    code="raise RuntimeError('must not run')",
                    trusted=True,
                ).model_copy(update={"enabled": False}),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()

    assert "must not run" not in code
