# ruff: noqa: F403, F405

import textwrap

import erlab.interactive._figurecomposer._codegen as figurecomposer_codegen
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError

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
    assert (
        figurecomposer_custom_code_operation._section_summary(
            tool, "missing", operation
        )
        == ""
    )
    assert figurecomposer_custom_code_operation._required_imports(tool, operation) == (
        "import numpy as np",
        "import xarray as xr",
        "import erlab.plotting as eplt",
    )
    assert figurecomposer_custom_code._custom_code_names("bad code !!") == frozenset()
    assert "data" in figurecomposer_custom_code._custom_code_names("data = data.mean()")
    assert "data" in figurecomposer_custom_code._custom_code_names("data += 1")
    assert "data" in figurecomposer_custom_code._custom_code_names(
        "def update():\n    global data\n    data += 1"
    )
    assert "data" in figurecomposer_custom_code._custom_code_names(
        "def remove():\n    global data\n    del data"
    )
    assert "data" not in figurecomposer_custom_code._custom_code_names(
        "data = 1\nfig.result = data"
    )
    assert "data" in figurecomposer_custom_code._custom_code_names(
        "def f():\n    return data\nresult = f()\ndata = 1"
    )
    assert "data" in figurecomposer_custom_code._custom_code_names(
        "if condition:\n    data = 1\nfig.result = data"
    )
    assert (
        figurecomposer_custom_code._custom_code_names(
            "import data\ndef identity(data):\n    return data\nvalue = data"
        )
        == frozenset()
    )
    assert figurecomposer_custom_code._custom_code_bound_names(
        "value = 1\n"
        "import numpy as array_module\n"
        "def identity(item):\n"
        "    return item\n"
        "def overwrite():\n"
        "    global replay_input\n"
        "    del replay_input"
    ) == frozenset({"value", "array_module", "identity", "overwrite", "replay_input"})
    assert (
        figurecomposer_custom_code._custom_code_bound_names("bad code !!")
        == frozenset()
    )
    assert figurecomposer_custom_code_operation._custom_axes_alias_lines(tool) == []
    assert (
        figurecomposer_custom_code_operation._code_lines(
            tool, operation.model_copy(update={"trusted": False})
        )
        == []
    )
    assert (
        figurecomposer_custom_code_operation._required_imports(
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
    assert figurecomposer_custom_code_operation._custom_axes_alias_lines(grid_tool) == [
        "axs = {",
        "    'axis-a': ax0,",
        "}",
    ]
    assert (
        figurecomposer_custom_code_operation._custom_first_axis_code(grid_tool) == "ax0"
    )


def test_figure_composer_custom_code_names_cover_nested_class_and_flow() -> None:
    nested_class_source = (
        "def build():\n"
        "    class Result:\n"
        "        data = data.mean()\n"
        "    return Result\n"
        "build()"
    )
    nested_class_import = nested_class_source.replace(
        "data = data.mean()", "np = np.asarray([1.0])"
    )
    assert "data" in figurecomposer_custom_code._custom_code_names(nested_class_source)
    assert "np" in figurecomposer_custom_code._custom_code_names(nested_class_import)

    class_local = nested_class_source.replace(
        "data = data.mean()", "data = 1\n        result = data"
    )
    class_import = nested_class_source.replace(
        "data = data.mean()",
        "import numpy as np\n        result = np.asarray([1.0])",
    )
    assert "data" not in figurecomposer_custom_code._custom_code_names(class_local)
    assert "np" not in figurecomposer_custom_code._custom_code_names(class_import)

    for module_bound_nested_scope in (
        "data = 1\nclass C:\n    data = data + 1",
        "import data\nclass C:\n    data = data.value",
        "data = 1\ndef f():\n    return data",
        "data = 1\ndef f():\n    global data\n    data += 1",
        (
            "data = 2\n"
            "class Outer:\n"
            "    data = 1\n"
            "    class Inner:\n"
            "        data = data + 1"
        ),
    ):
        assert "data" not in figurecomposer_custom_code._custom_code_names(
            module_bound_nested_scope
        )

    for late_or_class_local_binding in (
        "class C:\n    data = data + 1\ndata = 1",
        "def f():\n    global data\n    data += 1\nf()\ndata = 1",
        ("class Outer:\n    data = 1\n    class Inner:\n        data = data + 1"),
    ):
        assert "data" in figurecomposer_custom_code._custom_code_names(
            late_or_class_local_binding
        )

    plain_global_assignment = "def set_data():\n    global data\n    data = 1"
    assert "data" not in figurecomposer_custom_code._custom_code_names(
        plain_global_assignment
    )

    for conditional_binding in (
        "match flag:\n    case 0:\n        data = 1\nprint(data)",
        "flag and (data := 1)\nprint(data)",
        "(data := 1) if flag else 0\nprint(data)",
    ):
        assert "data" in figurecomposer_custom_code._custom_code_names(
            conditional_binding
        )

    for definite_binding in (
        "(data := 1) and print(data)\nprint(data)",
        "(data := 1) if flag else (data := 2)\nprint(data)",
    ):
        assert "data" not in figurecomposer_custom_code._custom_code_names(
            definite_binding
        )

    assert "data" in figurecomposer_custom_code._custom_code_names(
        "values = [data for _ in range(1)]"
    )
    for local_comprehension_data in (
        "data = 1\nvalues = [data for _ in range(1)]",
        "values = [data for data in range(1)]",
    ):
        assert "data" not in figurecomposer_custom_code._custom_code_names(
            local_comprehension_data
        )

    assert {"data", "other"} <= figurecomposer_custom_code._custom_code_names(
        "f, g = (lambda: data, lambda: other)"
    )
    assert {"data", "other"} <= figurecomposer_custom_code._custom_code_names(
        "x = [data for _ in ()]; y = [other for _ in ()]"
    )
    assert not {
        "data",
        "other",
    } & figurecomposer_custom_code._custom_code_names(
        "data = 1\nother = 2\nf, g = (lambda: data, lambda: other)"
    )


def test_figure_composer_custom_code_names_cover_structured_python() -> None:
    code = textwrap.dedent(
        """\
        @decorator
        def function(
            positional: annotation,
            /,
            regular: annotation = default,
            *args: annotation,
            required_keyword: annotation,
            keyword: annotation = default,
            **kwargs: annotation,
        ) -> annotation:
            return child_source

        @decorator
        async def async_function(
            positional: annotation,
            /,
            regular: annotation = default,
            *args: annotation,
            required_keyword: annotation,
            keyword: annotation = default,
            **kwargs: annotation,
        ) -> annotation:
            return child_source

        callable_value = lambda item=default: child_source
        keyword_callable_value = lambda *, required, optional=default: child_source

        @decorator
        class Result(base, metaclass=metaclass):
            class_value = class_value + child_source

        import package.submodule
        from package import imported as local_import
        from package import *
        annotation_only: annotation
        container.attribute: annotation
        value: annotation = source
        result += source
        container.attribute += source
        (walrus := source)
        del result
        del container.attribute

        if condition:
            branch = source
        else:
            branch = source
        conditional = source if condition else alternate
        condition and (short_circuit := source)

        match subject:
            case [head, *tail] as matched if guard:
                match_result = source
            case {"item": mapping_value, **remaining}:
                mapping_result = source

        for first, *tail in iterable:
            loop_result = source
        else:
            loop_else = source
        while condition:
            while_result = source
        else:
            while_else = source
        with context:
            bare_with_result = source
        with context as (left, right):
            with_result = source

        try:
            success = source
        except exception_type as error:
            failure = source
        else:
            after_success = source
        finally:
            cleanup = source
        try:
            star_success = source
        except* exception_type as grouped:
            star_failure = source
        try:
            bare_success = source
        except:
            bare_failure = source

        list_result = [
            item for item in iterable if condition for child in child_iterable if child
        ]
        set_result = {item for item in iterable if condition}
        dict_result = {item: source for item in iterable}
        generator_result = (item for item in iterable)

        def mutation_scope():
            global mutation_target, object_target
            mutation_target += source
            object_target.value += source
            del mutation_target, object_target.value
            def nested():
                return None
            async def nested_async():
                return None
            class Nested:
                pass
            ignored = lambda: None
        """
    )

    names = figurecomposer_custom_code._custom_code_names(code)

    assert {
        "annotation",
        "alternate",
        "base",
        "child_iterable",
        "child_source",
        "class_value",
        "condition",
        "container",
        "context",
        "decorator",
        "default",
        "exception_type",
        "guard",
        "iterable",
        "metaclass",
        "mutation_target",
        "source",
        "subject",
    } <= names
    assert "local_import" not in names

    if sys.version_info >= (3, 12):
        assert "source" in figurecomposer_custom_code._custom_code_names(
            "def generic[T](value: T) -> T:\n    return source"
        )
        assert "source" in figurecomposer_custom_code._custom_code_names(
            "class Generic[T]:\n    value: T = source"
        )


def test_figure_composer_custom_code_analyzer_handles_async_blocks() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """\
            async def process():
                async for item in async_iterable:
                    current = item
                else:
                    current = fallback
                async with async_context as context_value:
                    current = context_value
            """
        )
    )
    [function] = tree.body
    assert isinstance(function, ast.AsyncFunctionDef)
    analyzer = figurecomposer_custom_code._TopLevelExternalNameAnalyzer()

    analyzer._analyze_block(function.body, set())

    assert {"async_context", "async_iterable", "fallback"} <= analyzer.external


def test_figure_composer_custom_code_analyzer_handles_partial_ast_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class_tree = ast.parse("class Result:\n    value = source")
    root = figurecomposer_custom_code.symtable.symtable(
        "class Result:\n    value = source", "<test>", "exec"
    )
    monkeypatch.setattr(
        figurecomposer_custom_code, "_scope_symbol_tables", lambda *_args: {}
    )
    assert (
        figurecomposer_custom_code._class_local_external_names(class_tree, root, {})
        == set()
    )

    analyzer = figurecomposer_custom_code._TopLevelExternalNameAnalyzer()
    analyzer.bound.add("already_bound")
    analyzer.visit(ast.BoolOp(op=ast.And(), values=[]))
    assert analyzer.bound == {"already_bound"}
    assert analyzer.external == set()

    incomplete_comprehension = ast.ListComp(
        elt=ast.Name(id="source", ctx=ast.Load()), generators=[]
    )
    analyzer.visit(incomplete_comprehension)
    assert analyzer.external == {"source"}
    assert analyzer.scope_bindings[id(incomplete_comprehension)] == frozenset(
        {"already_bound"}
    )


@pytest.mark.parametrize(
    "code",
    (
        "data = data.mean()",
        "data += 1",
        "def update():\n    global data\n    data += 1",
        "def remove():\n    global data\n    del data",
        (
            "def build():\n"
            "    class Result:\n"
            "        data = data.mean()\n"
            "    return Result\n"
            "build()"
        ),
    ),
)
def test_figure_composer_custom_code_read_write_source_is_dependency(
    qtbot, code: str
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    operation = FigureOperationState.custom(
        label="read-write source",
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

    assert tool._document.operation_source_names(operation) == ("data",)
    assert tool._document.source_usage_count("data") == 1
    assert not tool.remove_source("data")


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

    assert tool._document.rename_source("data", "renamed")
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

    with pytest.raises(FigureComposerInputError):
        tool._document.rename_source("data", "renamed")
    assert tool.tool_status.sources[0].name == "data"
    assert tool.tool_status.operations[0].code == code


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
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
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
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
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
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("code")

    current_page = tool.step_editor_stack.currentWidget()
    assert current_page is not None
    code_edit = current_page.findChild(
        erlab.interactive.utils.PythonCodeEditor, "figureComposerCustomCodeEdit"
    )
    assert code_edit is not None

    new_code = "ax.set_title('pending')\nax.set_ylabel('counts')"
    code_edit.setPlainText(new_code)
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))

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
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
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
    internal = erlab.utils.array._make_dims_uniform(public)
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

    item = tool.operation_list.topLevelItem(0)
    assert item is not None
    assert _operation_status_codes(tool, 0) == ()


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


def test_figure_composer_custom_code_codegen_nested_class_import(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.custom(
                    label="nested class",
                    code=(
                        "def build():\n"
                        "    class Result:\n"
                        "        np = np.asarray([1.0, 2.0])\n"
                        "        total = float(np.sum())\n"
                        "    return Result\n"
                        "fig.custom_total = build().total"
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
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].custom_total == 3.0


def test_figure_composer_custom_code_sources_drive_usage_and_full_replay(
    qtbot, tmp_path
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="custom_source",
    )
    path = tmp_path / "custom-source.nc"
    data.to_netcdf(path)
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(
                    name="custom_source",
                    provenance_spec=_file_load_provenance(path).model_dump(mode="json"),
                ),
            ),
            operations=(
                FigureOperationState.custom(
                    label="custom",
                    code=(
                        "custom_source = custom_source.mean()\n"
                        "fig.source_total = float(custom_source)\n"
                        "import statistics\n"
                        "fig.source_mean = statistics.fmean([float(custom_source)])"
                    ),
                    trusted=True,
                ),
                FigureOperationState.line(label="line", source="custom_source"),
            ),
            primary_source="custom_source",
        ),
        source_data={"custom_source": data},
    )
    qtbot.addWidget(tool)

    assert tool._document.source_usage_count("custom_source") == 2
    assert not tool.remove_source("custom_source")
    _select_operation_rows(tool, (0,))
    tool._copy_selected_operations()
    clipboard_payload = tool._clipboard_step_payload()
    assert clipboard_payload is not None
    assert tuple(source.name for source in clipboard_payload[1]) == ("custom_source",)
    spec = tool.current_provenance_spec()
    assert spec is not None
    [replay_input] = spec.script_inputs
    assert replay_input.name != "custom_source"

    code = spec.display_code()
    assert code is not None
    assert code.index("fig, axs") < code.index("import statistics")
    assert f"custom_source = {replay_input.name}" in code
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].source_total == 1.5
    assert namespace["fig"].source_mean == 1.5
    np.testing.assert_allclose(namespace["fig"].axes[0].lines[-1].get_ydata(), data)


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
    assert (
        figurecomposer_custom_code._renamed_source_loads(
            "bad code !!", {"data": "renamed"}
        )
        == "bad code !!"
    )
    assert (
        figurecomposer_custom_code._renamed_source_loads(
            "bad code !!  # data", {"data": "renamed"}
        )
        == "bad code !!  # data"
    )


def test_figure_composer_source_rename_preserves_multibyte_custom_code() -> None:
    code = (
        "label = 'μ source'\n"
        "total = data.mean() + other.mean()\n"
        "# data and other stay comments\n"
    )

    renamed = figurecomposer_custom_code._renamed_source_loads(
        code, {"data": "renamed_data", "other": "renamed_other", "unused": "x"}
    )

    assert renamed == (
        "label = 'μ source'\n"
        "total = renamed_data.mean() + renamed_other.mean()\n"
        "# data and other stay comments\n"
    )


@pytest.mark.parametrize(
    "code",
    (
        "import package as data\nresult = data.value",
        "def summarize(data):\n    return data.mean()",
        (
            "def outer():\n"
            "    data = 1\n"
            "    def inner():\n"
            "        nonlocal data\n"
            "        return data\n"
            "    return inner()"
        ),
        "def read_global():\n    global data\n    return data",
    ),
)
def test_figure_composer_source_rename_rejects_every_local_binding_kind(
    code: str,
) -> None:
    with pytest.raises(ValueError, match="also binds 'data'"):
        figurecomposer_custom_code._renamed_source_loads(code, {"data": "renamed"})


def test_figure_composer_source_rename_rejects_unparseable_source_tokens() -> None:
    with pytest.raises(ValueError, match="valid code"):
        figurecomposer_custom_code._renamed_source_loads(
            "text = '''\ndata", {"data": "renamed"}
        )

    with pytest.raises(ValueError, match="valid code"):
        figurecomposer_custom_code._renamed_source_loads(
            "    invalid\n  data", {"data": "renamed"}
        )


def test_figure_composer_source_rename_ignores_non_load_declarations() -> None:
    code = "def configure():\n    global data"

    assert (
        figurecomposer_custom_code._renamed_source_loads(code, {"data": "renamed"})
        == code
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
    tool.source_panel.set_selected_names(("data",), current_name="data")
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    alias_edit = tool.source_panel.source_alias_edit

    alias_edit.setText("renamed")
    alias_edit.editingFinished.emit()

    assert alias_edit.text() == "data"
    assert tool.tool_status.sources[0].name == "data"
    assert not tool.source_panel.source_validation_label.isHidden()


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
                        "axs = dict(axs)\n"
                        "ax = ax.twinx()\n"
                        "np = np.asarray([1.0, 2.0])\n"
                        "ax.set_title('main')\n"
                        "axs['main-axis'].set_xlabel('energy')\n"
                        "fig.custom_total = float(np.sum())"
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
    assert "axs = {" in code
    assert "'main-axis': main_axis" in code
    assert "ax = main_axis" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].axes[1].get_title() == "main"
    assert namespace["fig"].axes[0].get_xlabel() == "energy"
    assert namespace["fig"].custom_total == 3.0


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

    item = tool.operation_list.topLevelItem(0)
    assert item is not None
    assert _operation_status_codes(tool, 0) == ("render_error",)
    assert "Custom code is not trusted" in item.toolTip(
        figurecomposer_tool_module._OPERATION_LIST_STATUS_COLUMN
    )
    assert tool.step_source_status_label.isHidden()

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("code")
    trusted_check = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QCheckBox, "figureComposerCustomCodeTrustedCheck"
    )
    assert trusted_check is not None
    assert not trusted_check.isChecked()

    trusted_check.setChecked(True)

    assert tool.tool_status.operations[0].trusted is True
    qtbot.waitUntil(lambda: not tool._operation_render_errors, timeout=1000)
    item = tool.operation_list.topLevelItem(0)
    assert item is not None
    assert _operation_status_codes(tool, 0) == ()
    assert "Custom code is not trusted" not in item.toolTip(
        figurecomposer_tool_module._OPERATION_LIST_STATUS_COLUMN
    )
    assert tool.step_source_status_label.isHidden()
