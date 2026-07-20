import ast
import importlib
import multiprocessing
import pathlib
import runpy
import subprocess
import sys
import textwrap
import types
import warnings

import erlab
import erlab.interactive.imagetool.plot_items
import erlab.interactive.imagetool.viewer


def test_interactive_modules_do_not_construct_global_qt_values() -> None:
    interactive_root = pathlib.Path(erlab.__file__).resolve().parent / "interactive"
    offenders: list[str] = []

    def qt_constructors(node: ast.AST | None) -> list[ast.Call]:
        if node is None:
            return []
        return [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr.startswith("Q")
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id in {"QtCore", "QtGui", "QtWidgets"}
        ]

    def inspect_definition_body(path: pathlib.Path, body: list[ast.stmt]) -> None:
        for node in body:
            values: list[ast.AST | None] = []
            if isinstance(node, ast.Assign | ast.AnnAssign):
                values.append(node.value)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                values.extend(node.args.defaults)
                values.extend(
                    default for default in node.args.kw_defaults if default is not None
                )
            elif isinstance(node, ast.ClassDef):
                inspect_definition_body(path, node.body)
                continue
            for value in values:
                for call in qt_constructors(value):
                    function = call.func
                    if not isinstance(function, ast.Attribute) or not isinstance(
                        function.value, ast.Name
                    ):
                        raise TypeError("Invalid Qt constructor expression")
                    relative = path.relative_to(interactive_root)
                    offenders.append(
                        f"{relative}:{call.lineno} {function.value.id}.{function.attr}"
                    )

    for path in interactive_root.rglob("*.py"):
        inspect_definition_body(path, ast.parse(path.read_text()).body)

    assert offenders == []


def test_core_shim_export_identity_and_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        core = importlib.import_module("erlab.interactive.imagetool.core")
        importlib.reload(core)

    assert any(issubclass(w.category, FutureWarning) for w in caught)

    assert core.ImageSlicerArea is erlab.interactive.imagetool.viewer.ImageSlicerArea
    assert core._parse_input is erlab.interactive.imagetool.viewer_state._parse_input
    assert (
        core._PolyROIEditDialog
        is erlab.interactive.imagetool.plot_items._PolyROIEditDialog
    )


def test_canonical_modules_are_importable() -> None:
    assert erlab.interactive.imagetool.viewer.ImageSlicerArea
    assert erlab.interactive.imagetool.plot_items.ItoolPlotItem


def test_figure_composer_facade_loads_state_without_loading_tool() -> None:
    code = (
        "import sys\n"
        "import erlab.interactive._figurecomposer as figurecomposer\n"
        "assert 'erlab.interactive._figurecomposer._tool' not in sys.modules\n"
        "assert figurecomposer.FigureRecipeState.__name__ == 'FigureRecipeState'\n"
        "assert 'erlab.interactive._figurecomposer._tool' not in sys.modules\n"
        "assert figurecomposer.FigureComposerTool.__module__ == "
        "'erlab.interactive._figurecomposer._tool'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_figure_composer_semantics_do_not_load_qt_widgets() -> None:
    code = textwrap.dedent(
        """
        import sys

        from erlab.interactive._figurecomposer import _seeding
        from erlab.interactive._figurecomposer._model._document import FigureDocument
        from erlab.interactive._figurecomposer._model._state import (
            FigureExportState,
            FigureOperationState,
            FigureRecipeState,
            FigureSourceState,
            FigureSubplotsState,
        )
        from erlab.interactive._figurecomposer._operations._method._catalog import (
            AXES_METHODS,
        )
        from erlab.interactive._figurecomposer._operations._method._state import (
            _default_method_args,
        )
        from erlab.interactive._figurecomposer._operations._plot_slices._model import (
            _slice_values_mode_text,
        )

        assert "erlab.interactive._stylesheets" not in sys.modules
        recipe = FigureRecipeState(
            setup=FigureSubplotsState(
                figsize=(6.4, 4.8), dpi=100.0, layout=None
            ),
            sources=(FigureSourceState(name="data"),),
            operations=(),
            export=FigureExportState(
                dpi="figure", transparent=False, bbox_inches="tight"
            ),
            primary_source="data",
        )
        document = FigureDocument(recipe)
        plot_operation = FigureOperationState.plot_array(
            label="plot", source="data"
        )
        custom_operation = FigureOperationState.custom(
            label="custom", code="result = data.mean()", trusted=True
        )

        assert document.operation_source_names(plot_operation) == ("data",)
        assert document.operation_source_names(custom_operation) == ("data",)
        assert (
            _seeding._operation_with_source_names(plot_operation, {})
            is plot_operation
        )
        assert isinstance(_default_method_args(AXES_METHODS["plot"], None), tuple)
        assert _slice_values_mode_text("all")
        assert "qtpy.QtWidgets" not in sys.modules
        assert "pyqtgraph" not in sys.modules
        assert "erlab.interactive._stylesheets" not in sys.modules
        assert (
            "erlab.interactive._figurecomposer._ui._axes_widgets"
            not in sys.modules
        )
        assert (
            "erlab.interactive._figurecomposer._ui._color_widgets"
            not in sys.modules
        )
        assert (
            "erlab.interactive._figurecomposer._ui._figure_window"
            not in sys.modules
        )
        assert "erlab.interactive._figurecomposer._tool" not in sys.modules
        assert (
            "erlab.interactive._figurecomposer._operations._registry"
            not in sys.modules
        )
        assert (
            "erlab.interactive._figurecomposer._operations._custom_code"
            not in sys.modules
        )
        assert "matplotlib.backends.backend_qtagg" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_manager_import_does_not_load_figure_dialog_widgets() -> None:
    code = (
        "import importlib.util\n"
        "import sys\n"
        "import erlab.interactive.imagetool.manager._mainwindow\n"
        "assert 'erlab.interactive.imagetool.manager._figurecomposer._dialogs' "
        "not in sys.modules\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.manager._figure_dialogs') is None\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.manager._figure_manager') is None\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.manager._workspace_io') is None\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.manager._workspace_state') is None\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.manager._xarray') is None\n"
        "assert 'erlab.interactive._figurecomposer._ui._axes_widgets' "
        "not in sys.modules\n"
        "assert 'matplotlib.backends.backend_qtagg' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_provenance_layers_load_in_one_direction() -> None:
    code = textwrap.dedent(
        """
        import importlib.util
        import sys

        from erlab.interactive.imagetool._provenance import _model

        prefix = "erlab.interactive.imagetool._provenance"
        assert f"{prefix}._code" in sys.modules
        assert f"{prefix}._graph" not in sys.modules
        assert f"{prefix}._execution" not in sys.modules
        assert f"{prefix}._operations" not in sys.modules
        assert "qtpy.QtWidgets" not in sys.modules
        assert "pyqtgraph" not in sys.modules

        operation = _model.parse_tool_provenance_operation(
            {"op": "rename", "name": "renamed"}
        )
        assert operation.name == "renamed"
        assert f"{prefix}._operations" in sys.modules
        assert f"{prefix}._graph" not in sys.modules
        assert f"{prefix}._execution" not in sys.modules

        assert importlib.util.find_spec(
            "erlab.interactive.imagetool._provenance_framework"
        ) is None
        assert importlib.util.find_spec(
            "erlab.interactive.imagetool._replay_graph"
        ) is None
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_interactive_utils_does_not_load_imagetool_provenance() -> None:
    code = textwrap.dedent(
        """
        import sys

        import erlab.interactive.utils

        prefix = "erlab.interactive.imagetool._provenance"
        assert not any(name.startswith(prefix) for name in sys.modules)
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plot_items_loads_figure_composer_only_for_action() -> None:
    code = textwrap.dedent(
        """
        import sys
        import types

        import erlab.interactive.imagetool.plot_items as plot_items

        adapter_name = "erlab.interactive.imagetool._figurecomposer_adapter"
        composer_prefix = "erlab.interactive._figurecomposer"
        assert adapter_name not in sys.modules
        assert not any(name.startswith(composer_prefix) for name in sys.modules)

        created = []
        manager = types.SimpleNamespace(
            target_from_slicer_area=lambda _area: 0,
            _node_for_target=lambda _target: types.SimpleNamespace(),
            _script_input_name_for_node=lambda _node: "data",
            create_figure_from_slicer_area=lambda _area, **kwargs: (
                created.append(kwargs["operation"])
            ),
        )
        slicer_area = types.SimpleNamespace(
            _in_manager=True,
            _manager_instance=manager,
            data=types.SimpleNamespace(ndim=2, dims=("x", "y")),
            n_cursors=1,
            manual_limits={},
            colormap_properties={
                "cmap": "viridis",
                "levels_locked": False,
                "reverse": False,
                "zero_centered": False,
                "high_contrast": False,
                "gamma": 1.0,
            },
        )
        plot = types.SimpleNamespace(
            slicer_area=slicer_area,
            is_image=True,
            display_axis=(0, 1),
            axis_dims=("y", "x"),
            axis_dims_uniform=("y", "x"),
            array_slicer=types.SimpleNamespace(
                _nonuniform_axes=set(),
                qsel_args=lambda _cursor, _axes: {},
            ),
            slicer_data_items=(),
            vb=types.SimpleNamespace(
                state={
                    "autoRange": (True, True),
                    "viewRange": ((0.0, 1.0), (0.0, 1.0)),
                    "aspectLocked": False,
                }
            ),
        )

        plot_items.ItoolPlotItem.plot_with_matplotlib(plot)

        assert adapter_name in sys.modules
        assert f"{composer_prefix}._model._state" in sys.modules
        assert len(created) == 1
        assert created[0].kind.value == "plot_array"
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_manager_entrypoint_freeze_support_runs_before_manager_import(
    monkeypatch,
) -> None:
    events: list[str] = []

    manager_module = types.ModuleType("erlab.interactive.imagetool.manager")

    def _getattr(name: str):
        if name == "main":
            events.append("manager_import")
            return lambda: events.append("main")
        raise AttributeError(name)

    manager_module.__getattr__ = _getattr  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "erlab.interactive.imagetool.manager", manager_module
    )
    monkeypatch.setattr(
        multiprocessing, "freeze_support", lambda: events.append("freeze")
    )

    entrypoint = (
        pathlib.Path(__file__).resolve().parents[3]
        / "src/erlab/interactive/imagetool/manager/__main__.py"
    )
    runpy.run_path(str(entrypoint), run_name="__main__")

    assert events == ["freeze", "manager_import", "main"]
