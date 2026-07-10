# ruff: noqa: F403, F405

import erlab.interactive._figurecomposer._codegen as figurecomposer_codegen

from ._common import *


def test_figure_composer_custom_code_helpers_cover_codegen_paths(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    operation = FigureOperationState.custom(
        label="code",
        code=(
            "values = np.asarray(data.values)\n"
            "arr = xr.DataArray(values)\n"
            "eplt.clean_labels(ax)"
        ),
        trusted=True,
    )
    assert figurecomposer_custom_code._section_summary(tool, "missing", operation) == ""
    assert figurecomposer_custom_code._required_imports(tool, operation) == (
        "import numpy as np",
        "import xarray as xr",
        "import erlab.plotting as eplt",
    )
    assert figurecomposer_custom_code._custom_code_names("bad code !!") == frozenset()
    assert figurecomposer_custom_code._custom_axes_alias_lines(tool) == []
    assert (
        figurecomposer_custom_code._code_lines(
            tool, operation.model_copy(update={"trusted": False})
        )
        == []
    )
    assert (
        figurecomposer_custom_code._required_imports(
            tool, operation.model_copy(update={"code": ""})
        )
        == ()
    )

    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=1,
                        ncols=1,
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
                        ),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    assert figurecomposer_custom_code._custom_axes_alias_lines(grid_tool) == [
        "axs = {",
        "    'axis-a': ax0,",
        "}",
    ]
    assert figurecomposer_custom_code._custom_first_axis_code(grid_tool) == "ax0"


def test_figure_composer_source_rename_refactors_custom_python(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    code = (
        "# Keep source comments and formatting intact.\n"
        "fig.source_label = 'μ data'; "
        "fig.renamed_source_mean = float(data.mean())  # data remains in comments\n"
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.custom(
                    label="summary",
                    code=code,
                    trusted=True,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    assert tool._rename_source_alias("data", "renamed")
    renamed_code = tool.tool_status.operations[0].code
    assert renamed_code == code.replace("float(data.mean())", "float(renamed.mean())")

    namespace: dict[str, typing.Any] = {"renamed": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["fig"].renamed_source_mean == 1.5
    assert namespace["fig"].source_label == "μ data"


@pytest.mark.parametrize(
    "code",
    (
        "data = data.mean()",
        "def summarize(data):\n    return data.mean()\nfig.result = summarize(data)",
        "data += 1",
        "del data",
        "fig.result = data.mean(\n",
    ),
)
def test_figure_composer_source_rename_rejects_ambiguous_python(
    qtbot, code: str
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    operation = FigureOperationState.custom(
        label="summary",
        code=code,
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    assert not tool._rename_source_alias("data", "renamed")
    assert tool.tool_status.sources[0].name == "data"
    assert tool.tool_status.operations[0].code == code
    assert not tool.source_status_label.isHidden()


def test_figure_composer_custom_code_editor_is_multiline_and_debounced(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    operation = FigureOperationState.custom(
        label="code",
        code="ax.set_title('old')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None
    assert isinstance(code_edit.highlighter, erlab.interactive.utils.PythonHighlighter)
    assert code_edit.lineWrapMode() == QtWidgets.QTextEdit.LineWrapMode.NoWrap

    render_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    first_code = "ax.set_title('first')"
    second_code = "ax.set_title('second')\nax.set_xlabel('energy')"
    code_edit.setPlainText(first_code)
    qtbot.wait(300)
    assert tool.tool_status.operations[0].code == "ax.set_title('old')"
    assert render_calls == []

    code_edit.setPlainText(second_code)

    assert tool.tool_status.operations[0].code == "ax.set_title('old')"
    assert render_calls == []

    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].code == second_code,
        timeout=1000,
    )
    qtbot.waitUntil(lambda: len(render_calls) == 1, timeout=1000)
    assert render_calls == [(tool,)]


def test_figure_composer_custom_code_editor_skips_render_until_valid_python(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    operation = FigureOperationState.custom(
        label="code",
        code="ax.set_title('old')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None

    render_calls: list[tuple[object, ...]] = []
    info_changed: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    invalid_code = "ax.set_title("
    code_edit.setPlainText(invalid_code)

    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].code == invalid_code,
        timeout=2000,
    )
    assert render_calls == []
    assert info_changed == [None]

    valid_code = "ax.set_title('valid')"
    code_edit.setPlainText(valid_code)

    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].code == valid_code,
        timeout=2000,
    )
    qtbot.waitUntil(lambda: len(render_calls) == 1, timeout=1000)
    assert render_calls == [(tool,)]
    assert info_changed == [None, None, None]


def test_figure_composer_custom_code_pending_edit_survives_step_switch(
    qtbot,
) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    first_operation = FigureOperationState.custom(
        label="first",
        code="ax.set_title('old')",
        trusted=True,
    )
    second_operation = FigureOperationState.custom(
        label="second",
        code="ax.set_xlabel('other')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(first_operation, second_operation),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None

    new_code = "ax.set_title('pending')\nax.set_ylabel('counts')"
    code_edit.setPlainText(new_code)
    tool.operation_list.setCurrentRow(1)

    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].code == new_code,
        timeout=1000,
    )
    assert tool.tool_status.operations[1].code == "ax.set_xlabel('other')"


def test_figure_composer_custom_code_pending_edit_flushes_on_close(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    operation = FigureOperationState.custom(
        label="code",
        code="ax.set_title('old')",
        trusted=True,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None

    new_code = "ax.set_title('pending close')\nax.set_ylabel('counts')"
    code_edit.setPlainText(new_code)

    assert tool.tool_status.operations[0].code == "ax.set_title('old')"
    tool.close()

    assert tool.tool_status.operations[0].code == new_code


def test_figure_composer_custom_code_uses_public_nonuniform_dims(qtbot) -> None:
    public = xr.DataArray(
        np.arange(8.0).reshape(4, 2),
        dims=("sample_temp", "alpha"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
        },
        name="map",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)
    operation = FigureOperationState.custom(
        label="code",
        code=(
            "assert 'sample_temp' in data.dims\n"
            "assert 'sample_temp_idx' not in data.dims"
        ),
        trusted=True,
    )
    tool = FigureComposerTool(
        internal,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="map"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_preview(tool, show_window=False)

    item = tool.operation_list.item(0)
    assert item is not None
    assert "(render error)" not in item.text()


def test_figure_composer_recipe_codegen_and_loaded_custom_code_trust(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    assert tool.tool_status.setup.figsize == FigureSubplotsState().figsize
    code = tool.generated_code()
    assert "tools[" not in code
    assert "_manager" not in code
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert "fig" in namespace

    status = tool.tool_status
    restored = FigureComposerTool(data)
    qtbot.addWidget(restored)
    restored.tool_status = status
    assert restored.tool_status.model_dump() == status.model_dump()

    custom_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=status.sources,
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code="ax.set_title('trusted')",
                    trusted=True,
                ),
            ),
            primary_source=status.primary_source,
        ),
    )
    qtbot.addWidget(custom_tool)

    loaded = erlab.interactive.utils.ToolWindow.from_dataset(custom_tool.to_dataset())
    qtbot.addWidget(loaded)
    assert loaded.tool_status.operations[0].trusted is False


def test_figure_composer_custom_code_codegen_namespace(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code=(
                        "ax.set_title(str(np.array([1])[0]))\n"
                        "fig.__dict__['_eplt_name'] = eplt.__name__"
                    ),
                    trusted=True,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "import numpy as np" in code
    assert "import erlab.plotting as eplt" in code
    assert "ax = axs[0, 0]" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].axes[0].get_title() == "1"
    assert namespace["fig"].__dict__["_eplt_name"] == "erlab.plotting"


def test_figure_composer_source_name_map_does_not_rewrite_custom_locals(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="sample_map",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data_0", label="sample map"),),
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code=("data_0 = 1\nfig.__dict__['custom_data_0'] = data_0"),
                    trusted=True,
                ),
            ),
            primary_source="data_0",
        ),
        source_data={"data_0": data},
    )
    qtbot.addWidget(tool)

    code = figurecomposer_codegen.generated_code(
        tool, source_name_map={"data_0": "sample_map"}
    )

    assert "data_0 = sample_map" in code
    assert "fig.__dict__['custom_data_0'] = data_0" in code
    assert "fig.__dict__['custom_data_0'] = sample_map" not in code
    namespace: dict[str, typing.Any] = {"sample_map": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].__dict__["custom_data_0"] == 1


def test_figure_composer_source_name_map_assigns_custom_aliases_simultaneously(
    qtbot,
) -> None:
    source_a = xr.DataArray(np.array([1.0]), dims=("x",), name="a")
    source_b = xr.DataArray(np.array([2.0]), dims=("x",), name="b")
    tool = FigureComposerTool(
        source_a,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="a", label="a"),
                FigureSourceState(name="b", label="b"),
            ),
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code=(
                        "fig.__dict__['custom_values'] = "
                        "(float(a.values[0]), float(b.values[0]))"
                    ),
                    trusted=True,
                ),
            ),
            primary_source="a",
        ),
        source_data={"a": source_a, "b": source_b},
    )
    qtbot.addWidget(tool)

    code = figurecomposer_codegen.generated_code(
        tool, source_name_map={"a": "b", "b": "a"}
    )

    assert "a, b = b, a" in code
    namespace: dict[str, typing.Any] = {"a": source_b, "b": source_a}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].__dict__["custom_values"] == (1.0, 2.0)


def test_figure_composer_source_name_replacement_fallback_edges() -> None:
    assert figurecomposer_codegen._source_alias_assignment_lines({}) == []
    assert (
        figurecomposer_codegen._replace_source_load_names("bad code !", {"data": "map"})
        == "bad code !"
    )
    assert (
        figurecomposer_codegen._replace_source_load_names(
            "value = data", {"data": "not valid"}
        )
        == "value = data"
    )
    assert (
        figurecomposer_custom_code._renamed_source_loads(
            "value = 1", {"data": "renamed"}
        )
        == "value = 1"
    )
    assert (
        figurecomposer_custom_code._renamed_source_loads(
            "value = data", {"data": "data"}
        )
        == "value = data"
    )


def test_figure_composer_source_alias_editor_rejects_ambiguous_python(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.custom(
                    label="summary",
                    code="data = data.mean()",
                    trusted=True,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool._set_selected_source_names_silent({"data"}, "data")
    tool._refresh_source_selection_editor()
    alias_edit = next(
        item.widget()
        for row in range(tool.source_selection_controls_layout.rowCount())
        if (
            (
                item := tool.source_selection_controls_layout.itemAt(
                    row, QtWidgets.QFormLayout.ItemRole.FieldRole
                )
            )
            is not None
            and isinstance(item.widget(), QtWidgets.QLineEdit)
            and item.widget().objectName() == "figureComposerSourceAliasEdit"
        )
    )

    alias_edit.setText("renamed")
    alias_edit.editingFinished.emit()

    assert alias_edit.text() == "data"
    assert tool.tool_status.sources[0].name == "data"
    assert not tool.source_status_label.isHidden()


def test_figure_composer_custom_code_codegen_gridspec_axes_alias(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=1,
                        ncols=1,
                        axes=(
                            FigureGridSpecAxesState(
                                axes_id="main-axis",
                                label="main_axis",
                                span=FigureGridSpecSpanState(
                                    row_start=0,
                                    row_stop=1,
                                    col_start=0,
                                    col_stop=1,
                                ),
                            ),
                        ),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code=(
                        "ax.set_title('main')\naxs['main-axis'].set_xlabel('energy')"
                    ),
                    trusted=True,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "axs = {" in code
    assert "'main-axis': main_axis" in code
    assert "ax = main_axis" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].axes[0].get_title() == "main"
    assert namespace["fig"].axes[0].get_xlabel() == "energy"


def test_figure_composer_untrusted_custom_code_reports_render_error(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    operation = FigureOperationState.custom(
        label="custom",
        code="ax.set_title('loaded')",
        trusted=False,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    item = tool.operation_list.item(0)
    assert item is not None
    assert "(render error)" in item.text()
    assert "Custom code is not trusted" in item.toolTip()
    assert "Enable Trusted to render it" in tool.step_source_status_label.text()

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("code")
    trusted_check = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QCheckBox, "figureComposerCustomCodeTrustedCheck"
    )
    assert trusted_check is not None
    assert not trusted_check.isChecked()

    trusted_check.setChecked(True)

    assert tool.tool_status.operations[0].trusted is True
    qtbot.waitUntil(lambda: not tool._operation_render_errors, timeout=1000)
    item = tool.operation_list.item(0)
    assert item is not None
    assert "(render error)" not in item.text()
    assert "Custom code is not trusted" not in item.toolTip()
    assert "Render error" not in tool.step_source_status_label.text()
