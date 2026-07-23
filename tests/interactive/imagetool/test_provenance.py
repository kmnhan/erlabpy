import ast
import collections.abc
import itertools
import json
import pathlib
import subprocess
import sys
import types
import typing

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

import erlab
from erlab.interactive.imagetool._provenance import _code
from erlab.interactive.imagetool._provenance._code import (
    _MAPPING_MARKER,
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    _TUPLE_MARKER,
    _expression_receiver_code,
    _format_selection_step,
    _migrate_legacy_nonuniform_restore_code,
    _provenance_value_code,
    _replace_code_identifiers,
    _restore_nonuniform_dims_expression,
    _simplify_display_code,
    _statement_load_count,
    _statement_store_count,
    _validate_active_name,
    _validate_script_replay_code,
    rebase_default_replay_input,
    uses_default_replay_input,
)
from erlab.interactive.imagetool._provenance._execution import (
    _load_file_source_data,
    _parse_replay_input,
    _resolve_importable_callable,
    _select_replay_input,
    can_reload_without_trust,
    file_load_source_status,
    replay_file_provenance,
    replay_script_provenance,
    script_provenance_replayable,
    script_provenance_requires_trust,
)
from erlab.interactive.imagetool._provenance._model import (
    _OPERATION_TYPES,
    ConsoleCall,
    ConsoleOperationPattern,
    DerivationEntry,
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    OperationGroupMarker,
    ReplayStage,
    ReplayStep,
    ScriptInput,
    ScriptInputDependencyRef,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _as_script_replay_spec,
    _assignment_code,
    _callable_paths,
    _coerce_float_sequence,
    _console_mapping_values,
    _console_values_equal,
    _encode_provenance_hashable,
    _ensure_float_tuple,
    _format_derivation_value,
    _is_whole_array_rename_entry,
    _normalize_provenance_hashable,
    _ProvenanceDisplayContext,
    _ProvenanceReorderBlock,
    _ProvenanceReorderBlockRef,
    _ProvenanceReorderSection,
    _ProvenanceReorderSectionRef,
    _ProvenanceStepRef,
    _SourceViewOperation,
    compose_display_provenance,
    compose_full_provenance,
    decode_provenance_value,
    direct_replay_input_name,
    encode_provenance_value,
    file_load,
    full_data,
    has_file_load_source,
    iter_operation_refs,
    mark_promoted_1d_source,
    operation_from_console_call,
    operation_group_range,
    operations_expression_code,
    parse_tool_provenance_operation,
    parse_tool_provenance_spec,
    public_data,
    rebase_script_input_node_uids,
    replay_input_name,
    require_live_source_spec,
    restamp_operation_groups,
    script,
    script_input_dependency_refs,
    selection,
    stamp_operation_group,
    strip_operation_groups,
    strip_partial_operation_groups,
    to_replay_provenance_spec,
)
from erlab.interactive.imagetool._provenance._operations import (
    AffineCoordOperation,
    AssignAttrsOperation,
    AssignCoord1DOperation,
    AssignCoordsOperation,
    AssignScalarCoordOperation,
    AverageOperation,
    BoxcarFilterOperation,
    CoarsenOperation,
    CorrectWithEdgeOperation,
    DivideByCoordOperation,
    FillNaOperation,
    GaussianFilterOperation,
    ImageDerivativeOperation,
    ImageToolSelectionSourceBinding,
    InterpolationOperation,
    IselOperation,
    KspaceConfigurationOperation,
    KspaceConvertOperation,
    KspaceInnerPotentialOperation,
    KspaceSetNormalOperation,
    KspaceWorkFunctionOperation,
    LeadingEdgeOperation,
    MaskWithPolygonOperation,
    ModelFitOperation,
    NormalizeOperation,
    QSelAggregationOperation,
    QSelOperation,
    RemoveMeshOperation,
    RenameDimsCoordsOperation,
    RenameOperation,
    RestoreNonuniformDimsOperation,
    RotateOperation,
    ScriptCodeOperation,
    SelectCoordOperation,
    SelOperation,
    SliceAlongPathOperation,
    SortByOperation,
    SortCoordOrderOperation,
    SqueezeOperation,
    SwapDimsOperation,
    SymmetrizeNfoldOperation,
    SymmetrizeOperation,
    ThinOperation,
    TransposeOperation,
    UniformInterpolationOperation,
    _ModelFitParameterSpec,
)


def _exec_generated_code(
    code: str, namespace: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    exec_namespace = {
        "np": np,
        "xr": xr,
        "erlab": erlab,
        "era": erlab.analysis,
        **namespace,
    }
    exec(code, exec_namespace)  # noqa: S102
    return exec_namespace


def _generated_call_names(code: str) -> tuple[str, ...]:
    def _call_name(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
            if isinstance(node, ast.Call):
                return ".".join(reversed(parts))
        if not isinstance(node, ast.Name):
            return None
        parts.append(node.id)
        return ".".join(reversed(parts))

    return tuple(
        name
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        for name in [_call_name(node.func)]
        if name is not None
    )


def _assert_no_meaningless_reassignments(code: str) -> None:
    statements = ast.parse(code).body
    for statement in statements:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Name)
        ):
            assert statement.targets[0].id != statement.value.id

    for statement, next_statement in itertools.pairwise(statements):
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or not isinstance(next_statement, ast.Assign)
            or len(next_statement.targets) != 1
            or not isinstance(next_statement.targets[0], ast.Name)
        ):
            continue
        target = statement.targets[0].id
        assert not (
            next_statement.targets[0].id == target
            and _statement_load_count(next_statement, target) == 1
        )


def _base_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(24).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={
            "x": [0.0, 1.0, 2.0],
            "y": [10.0, 11.0, 12.0, 13.0],
            "z": [5.0, 6.0],
            "x_alt": ("x", ["a", "b", "c"]),
        },
        name="data",
    )


def _kspace_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(27.0).reshape(3, 3, 3),
        dims=("alpha", "beta", "eV"),
        coords={
            "alpha": [-1.0, 0.0, 1.0],
            "beta": [-1.0, 0.0, 1.0],
            "eV": [-0.2, 0.0, 0.2],
            "xi": 0.0,
            "hv": 21.2,
        },
        attrs={
            "configuration": int(erlab.constants.AxesConfiguration.Type1),
            "inner_potential": 10.0,
            "sample_workfunction": 4.5,
        },
        name="anglemap",
    )


def _hashable_data() -> xr.DataArray:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)),
        dims=(1, ("beta", 0)),
        name="data",
    )
    return data.assign_coords(
        {"coord_1": xr.DataArray([100.0, 101.0, 102.0], dims=[1])}
    )


def _string_key_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(12).reshape((3, 4)),
        dims=("k-space", ("beta", 0)),
        coords={"k-space": [0.0, 1.0, 2.0]},
        name="data",
    )


def test_provenance_import_keeps_analysis_targets_lazy() -> None:
    code = (
        "import sys\n"
        "from erlab.interactive.imagetool import provenance\n"
        "assert 'AverageOperation' not in provenance.__all__\n"
        "assert not hasattr(provenance, 'AverageOperation')\n"
        "assert 'erlab.interactive.imagetool._provenance._operations' "
        "not in sys.modules\n"
        "loaded = sorted("
        "name for name in sys.modules if name.startswith('erlab.analysis.') "
        "or name.startswith('scipy.interpolate') or name.startswith('scipy.linalg')"
        ")\n"
        "if loaded:\n"
        "    print('\\n'.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_console_operation_lookup_loads_catalog_independently() -> None:
    code = (
        "from erlab.interactive.imagetool._provenance._model import (\n"
        "    ConsoleCall, operation_from_console_call,\n"
        ")\n"
        "operation = operation_from_console_call(ConsoleCall(\n"
        "    dataarray_method='isel',\n"
        "    kwargs={'x': 0},\n"
        "    display_code='data.isel(x=0)',\n"
        "    has_extra_tracked_inputs=False,\n"
        "))\n"
        "assert operation is not None\n"
        "assert operation.op == 'isel'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_provenance_constructors_expose_required_value_models() -> None:
    from erlab.interactive.imagetool import provenance

    selection = provenance.FileDataSelection(kind="dataarray")
    replay_call = provenance.FileReplayCall(
        kind="callable",
        target="xarray.load_dataarray",
        selection=selection,
    )
    load_source = provenance.FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="(none)",
        replay_call=replay_call,
    )
    stage = provenance.ReplayStage.from_source_spec(provenance.full_data())
    spec = provenance.file_load(
        start_label="Load scan",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=load_source,
        replay_stages=(stage,),
    )
    script_input = provenance.ScriptInput(
        name="data_0",
        provenance_spec=spec.model_dump(mode="json"),
    )

    assert script_input.parsed_provenance_spec() == spec
    assert provenance.DerivationEntry("Load scan", None).label == "Load scan"


def test_provenance_model_registers_operations_in_fresh_process() -> None:
    code = (
        "import importlib.util\n"
        "from erlab.interactive.imagetool._provenance._model import (\n"
        "    FileLoadSource, FileReplayCall, ToolProvenanceSpec, file_load,\n"
        "    full_data, parse_tool_provenance_operation,\n"
        "    parse_tool_provenance_spec, script, selection,\n"
        ")\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.provenance_operations') is None\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.provenance_framework') is None\n"
        "payload = {\n"
        "    'schema_version': 2,\n"
        "    'kind': 'full_data',\n"
        "    'operations': [\n"
        "        {'op': 'average', 'dims': ['x']},\n"
        "        {'op': 'rename', 'name': 'avg'},\n"
        "    ],\n"
        "}\n"
        "parsed = parse_tool_provenance_spec(payload)\n"
        "from erlab.interactive.imagetool._provenance._operations import (\n"
        "    AverageOperation, IselOperation, RenameOperation,\n"
        "    ScriptCodeOperation,\n"
        ")\n"
        "assert isinstance(parsed, ToolProvenanceSpec)\n"
        "assert isinstance(parsed.operations[0], AverageOperation)\n"
        "assert parsed.operations[-1] == RenameOperation(name='avg')\n"
        "assert isinstance(\n"
        "    parse_tool_provenance_operation("
        "{'op': 'rename', 'name': 'renamed'}),\n"
        "    RenameOperation,\n"
        ")\n"
        "assert selection(IselOperation(kwargs={'x': 0})).kind "
        "== 'selection'\n"
        "assert script(\n"
        "    ScriptCodeOperation(label='Step', code='derived = data'),\n"
        "    start_label='Run script',\n"
        "    active_name='derived',\n"
        ").kind == 'script'\n"
        "source = FileLoadSource(\n"
        "    path='scan.h5',\n"
        "    loader_label='xarray.load_dataarray',\n"
        "    loader_text='xarray.load_dataarray',\n"
        "    kwargs_text='',\n"
        "    replay_call=FileReplayCall(\n"
        "        kind='callable',\n"
        "        target='xarray.load_dataarray',\n"
        "        selected_index=0,\n"
        "    ),\n"
        ")\n"
        "file_spec = file_load(\n"
        "    start_label='Load data',\n"
        "    seed_code='derived = data',\n"
        "    file_load_source=source,\n"
        ")\n"
        "assert file_spec.kind == 'file'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_script_replay_keeps_unused_aliases_lazy() -> None:
    code = (
        "import sys\n"
        "import numpy as np\n"
        "import xarray as xr\n"
        "from erlab.interactive.imagetool._provenance._execution import "
        "replay_script_provenance\n"
        "from erlab.interactive.imagetool._provenance._model import "
        "ScriptInput, script\n"
        "from erlab.interactive.imagetool._provenance._operations import "
        "ScriptCodeOperation\n"
        "data = xr.DataArray(np.arange(2.0), dims=('x',))\n"
        "spec = script(\n"
        "    ScriptCodeOperation(label='Subtract', "
        "code='derived = data_0 - data_1'),\n"
        "    start_label='Run script',\n"
        "    active_name='derived',\n"
        "    script_inputs=(\n"
        "        ScriptInput(name='data_0', label='A'),\n"
        "        ScriptInput(name='data_1', label='B'),\n"
        "    ),\n"
        ")\n"
        "replay_script_provenance(spec, {'data_0': data, 'data_1': data})\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('erlab.analysis.')\n"
        "    or name.startswith('erlab.plotting')\n"
        "    or name.startswith('matplotlib')\n"
        ")\n"
        "if loaded:\n"
        "    print('\\n'.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def _file_replay_source(
    path: typing.Any = "scan.h5",
    *,
    replay_call: typing.Any = None,
) -> typing.Any:
    if replay_call is None:
        replay_call = FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={},
            selected_index=0,
        )
    return FileLoadSource(
        path=path,
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="(none)",
        replay_call=replay_call,
        load_code=None,
    )


def _file_provenance_spec(path: typing.Any = "scan.h5") -> typing.Any:
    return file_load(
        start_label="Load data from file 'scan.h5'",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(path),
    )


def _representative_structured_operations() -> tuple[ToolProvenanceOperation, ...]:
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})
    vertices = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    return (
        QSelOperation(kwargs={"x": 1.0}),
        IselOperation(kwargs={"x": slice(0, 2)}),
        SelOperation(kwargs={"y": slice(10.0, 12.0)}),
        SortCoordOrderOperation(),
        SortByOperation(variables=("x",), ascending=False),
        SelectCoordOperation(coord_name="x"),
        TransposeOperation(dims=("y", "x", "z")),
        SqueezeOperation(),
        RenameOperation(name="renamed"),
        RestoreNonuniformDimsOperation(),
        RotateOperation(angle=0.0, axes=("x", "y"), center=(0.0, 10.0)),
        AverageOperation(dims=("x",)),
        QSelAggregationOperation(dims=("x",), func="sum"),
        InterpolationOperation(dim="x", values=[0.25, 0.75]),
        UniformInterpolationOperation(sizes={"x": 4}),
        LeadingEdgeOperation(fraction=0.5, dim="x"),
        DivideByCoordOperation(coord_name="x"),
        GaussianFilterOperation(sigma={"x": 0.5}),
        BoxcarFilterOperation(size={"x": 3}),
        FillNaOperation(value=0.0),
        ImageDerivativeOperation(
            method="diffn",
            kwargs={"coord": "x", "order": 2},
        ),
        RemoveMeshOperation(
            first_order_peaks=((5, 5), (5, 7), (5, 3)),
            order=1,
            n_pad=0,
            roi_hw=2,
            output="corrected",
        ),
        NormalizeOperation(dims=("x",), mode="minmax"),
        CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        ),
        ThinOperation(mode="per_dim", factors={"x": 2}),
        SymmetrizeOperation(dim="x", center=1.0),
        SymmetrizeNfoldOperation(
            fold=4,
            axes=("x", "y"),
            center={"x": 1.0, "y": 10.0},
        ),
        CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False),
        SwapDimsOperation(mapping={"x": "x_alt"}),
        RenameDimsCoordsOperation(mapping={"x": "energy"}),
        AffineCoordOperation(coord_name="x", scale=2.0, offset=1.0),
        AssignCoordsOperation(coord_name="x", values=[0.0, 1.0, 2.0]),
        AssignScalarCoordOperation(coord_name="temperature", value=20.0),
        AssignCoord1DOperation(
            coord_name="temperature",
            dim="x",
            values=[1.0, 2.0, 3.0],
        ),
        AssignAttrsOperation(attrs={"sample": "test"}),
        KspaceConfigurationOperation(configuration=2),
        KspaceWorkFunctionOperation(work_function=4.2),
        KspaceInnerPotentialOperation(inner_potential=12.0),
        KspaceSetNormalOperation(alpha=1.5, beta=-0.5, delta=2.0),
        KspaceConvertOperation(
            bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
            resolution={"kx": 0.02, "ky": 0.02},
        ),
        SliceAlongPathOperation(
            vertices={"x": [0.0, 1.0], "y": [10.0, 11.0]},
            step_size=0.1,
            dim_name="path",
        ),
        MaskWithPolygonOperation(vertices=vertices, dims=("x", "y")),
        ModelFitOperation(
            fit_dim="x",
            model="PolynomialModel",
            model_kwargs={"degree": 1},
            parameters={
                "c0": _ModelFitParameterSpec(value=0.0),
                "c1": _ModelFitParameterSpec(value=1.0),
            },
            method="leastsq",
            parameter="c1",
        ),
    )


@pytest.mark.parametrize(
    "operation",
    _representative_structured_operations(),
    ids=lambda operation: operation.op,
)
def test_structured_operations_generate_public_code(
    operation: ToolProvenanceOperation,
) -> None:
    try:
        code = f"derived = {operation.expression_code('data')}"
    except NotImplementedError:
        code = operation.statement_code("data", output_name="derived")

    assert "erlab.interactive.imagetool" not in code
    assert "decode_provenance_value" not in code


@pytest.mark.parametrize(
    "operation_factory",
    [
        lambda: BoxcarFilterOperation(size={}),
        lambda: BoxcarFilterOperation(size={"x": 0}),
        lambda: BoxcarFilterOperation(size={"x": True}),
        lambda: BoxcarFilterOperation(size={"x": 1.5}),
        lambda: BoxcarFilterOperation(size={"x": 3}, cval=np.inf),
        lambda: UniformInterpolationOperation(sizes={}),
        lambda: UniformInterpolationOperation(sizes={"x": 0}),
        lambda: UniformInterpolationOperation(sizes={"x": True}),
        lambda: UniformInterpolationOperation(sizes={"x": 1.5}),
        lambda: FillNaOperation(value=np.nan),
        lambda: ImageDerivativeOperation(
            method="diffn",
            kwargs={"coord": "x"},
        ),
        lambda: ImageDerivativeOperation(
            method="diffn",
            kwargs={"coord": "x", "order": 0},
        ),
        lambda: ImageDerivativeOperation(
            method="curvature",
            kwargs={"a0": 0.0, "factor": 1.0},
        ),
        lambda: ImageDerivativeOperation(
            method="scaled_laplace",
            kwargs={"factor": np.inf},
        ),
        lambda: ImageDerivativeOperation(
            method="scaled_laplace",
            kwargs={"factor": True},
        ),
        lambda: ImageDerivativeOperation(
            method="diffn",
            kwargs={"coord": ["x"], "order": 2},
        ),
        lambda: ImageDerivativeOperation(
            method="minimum_gradient",
            kwargs={1: 2},
        ),
        lambda: RemoveMeshOperation(
            first_order_peaks=((5, 5), (5, 7), (5, 3)),
            k=np.inf,
        ),
        lambda: RemoveMeshOperation(
            first_order_peaks=((5, 5), (5, 7), (5, 3)),
            feather=np.inf,
        ),
        lambda: RemoveMeshOperation(
            first_order_peaks=((-1, 5), (5, 7), (5, 3)),
        ),
        lambda: RemoveMeshOperation(
            first_order_peaks=((5, 5), (5, 7), (5, 3)),
            order=True,
        ),
        lambda: RemoveMeshOperation(
            first_order_peaks=((5, 5), (5, 7), (5, 3)),
            n_pad=1.5,
        ),
        lambda: RemoveMeshOperation(
            first_order_peaks=((5, 5), (5, 7), (5, 3)),
            roi_hw=False,
        ),
        lambda: RemoveMeshOperation(
            first_order_peaks=((5, 5), (5, 7), (True, 3)),
        ),
    ],
)
def test_tool_output_operations_reject_invalid_parameters(operation_factory) -> None:
    with pytest.raises((TypeError, ValidationError)):
        operation_factory()


def test_remove_mesh_operation_defers_data_dependent_peak_validation() -> None:
    operation = RemoveMeshOperation(
        first_order_peaks=((5, 5), (5, 7), (5, 3)),
    )
    payload = operation.model_dump(mode="json")
    payload["first_order_peaks"] = [[5, 5], [5, 5], [5, 3]]

    parsed = parse_tool_provenance_operation(payload)

    assert isinstance(parsed, RemoveMeshOperation)
    with pytest.raises(ValueError, match="distinct"):
        parsed.apply(xr.DataArray(np.ones((11, 11)), dims=("alpha", "eV")))


def test_uniform_interpolation_uses_current_coordinate_bounds() -> None:
    operation = UniformInterpolationOperation(sizes={"x": 5})
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape(3, 2),
        dims=("x", "y"),
        coords={"x": [10.0, 20.0, 40.0], "y": [0.0, 1.0]},
    )
    expected = data.interp(x=np.linspace(10.0, 40.0, 5))

    xr.testing.assert_identical(
        operation.apply(data),
        expected,
    )
    namespace = _exec_generated_code(
        operation.replay_code("data", output_name="derived"),
        {"data": data},
    )
    generated = namespace["derived"]
    assert isinstance(generated, xr.DataArray)
    xr.testing.assert_identical(generated, expected)


def test_assignment_code_wraps_generator_calls_without_changing_behavior() -> None:
    output_name = "generated_result_with_a_name_long_enough_to_require_wrapping"
    code = _assignment_code(
        output_name,
        "sum(value for value in range(10))",
    )
    assert "\n" in code
    namespace: dict[str, object] = {}
    exec(code, {"sum": sum, "range": range}, namespace)  # noqa: S102
    assert namespace[output_name] == 45


def test_operation_group_markers_round_trip_and_strip_partial_groups() -> None:
    operations = stamp_operation_group(
        (
            AverageOperation(dims=("x",)),
            SqueezeOperation(),
        ),
        kind="demo",
        group_id="group-1",
        focuses=("first", "second"),
    )

    assert operation_group_range(operations, 0, kind="demo") == (0, 2)
    assert operation_group_range(operations, 1, kind="demo") == (0, 2)
    assert operations[0].group is not None
    assert operations[0].group.focus == "first"
    assert "group" not in AverageOperation(dims=("x",)).model_dump(mode="json")

    parsed = tuple(
        parse_tool_provenance_operation(operation.model_dump(mode="json"))
        for operation in operations
    )
    assert parsed == operations

    assert strip_partial_operation_groups(operations) == operations
    partial = strip_partial_operation_groups(operations[:1])
    assert partial[0].group is None

    scrambled = strip_partial_operation_groups((operations[1], operations[0]))
    assert all(operation.group is None for operation in scrambled)

    restamped = restamp_operation_groups(operations)
    assert strip_operation_groups(restamped) == strip_operation_groups(operations)
    assert operation_group_range(restamped, 0, kind="demo") == (0, 2)
    assert restamped[0].group is not None
    assert operations[0].group is not None
    assert restamped[0].group.id != operations[0].group.id

    adjacent = restamp_operation_groups(operations + operations)
    assert operation_group_range(adjacent, 0, kind="demo") == (0, 2)
    assert operation_group_range(adjacent, 2, kind="demo") == (2, 4)
    assert adjacent[0].group is not None
    assert adjacent[2].group is not None
    assert adjacent[0].group.id != adjacent[2].group.id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "", "id": "group", "index": 0, "size": 1},
        {"kind": "demo", "id": "", "index": 0, "size": 1},
        {"kind": "demo", "id": "group", "index": -1, "size": 1},
        {"kind": "demo", "id": "group", "index": 0, "size": 0},
        {"kind": "demo", "id": "group", "index": 1, "size": 1},
    ],
)
def test_operation_group_marker_rejects_invalid_metadata(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        OperationGroupMarker(**kwargs)


def test_operation_group_helpers_reject_broken_ranges() -> None:
    operations = stamp_operation_group(
        (
            AverageOperation(dims=("x",)),
            SqueezeOperation(),
        ),
        kind="demo",
        group_id="group-1",
    )
    plain = AverageOperation(dims=("y",))

    assert stamp_operation_group((), kind="demo") == ()
    with pytest.raises(ValueError, match="focuses must match"):
        stamp_operation_group(
            operations,
            kind="demo",
            focuses=("first",),
        )
    assert strip_operation_groups((plain,)) == (plain,)

    assert operation_group_range(operations, -1) is None
    assert operation_group_range(operations, len(operations)) is None
    assert operation_group_range((plain,), 0) is None
    assert operation_group_range(operations, 0, kind="other") is None
    assert operation_group_range((operations[1],), 0) is None

    neighbor = plain.model_copy(update={"group": operations[0].group})
    assert operation_group_range((*operations, neighbor), 0) is None

    restamped = restamp_operation_groups((operations[1],))
    assert restamped[0].group is None


def test_tool_provenance_codec_and_combinators() -> None:
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})
    encoded = encode_provenance_value(
        {"sel": slice(1.0, 2.0), "data": _base_data(), "edge_fit": edge_fit}
    )
    decoded = decode_provenance_value(encoded)

    assert decoded["sel"] == slice(1.0, 2.0)
    xr.testing.assert_identical(decoded["data"], _base_data())
    xr.testing.assert_identical(decoded["edge_fit"], edge_fit)

    hashable_encoded = encode_provenance_value(
        {1: slice(0.0, 1.0), ("beta", 0): {"nested": [1, 2, 3]}}
    )
    assert _MAPPING_MARKER in hashable_encoded
    mapping_entries = hashable_encoded[_MAPPING_MARKER]
    assert mapping_entries[0][0] == 1
    assert mapping_entries[1][0] == {_TUPLE_MARKER: ["beta", 0]}
    assert decode_provenance_value(hashable_encoded) == {
        1: slice(0.0, 1.0),
        ("beta", 0): {"nested": [1, 2, 3]},
    }

    spec = full_data(AverageOperation(dims=("y",))).append_final_rename("avg")
    trimmed = spec.drop_trailing_rename()
    replaced = spec.append_replacement_operations(
        ThinOperation(mode="global", factor=2)
    )

    assert [op.op for op in spec.operations] == ["average", "rename"]
    assert [op.op for op in trimmed.operations] == ["average"]
    assert [op.op for op in replaced.operations] == ["average", "thin"]

    with pytest.raises(ValidationError, match="Instance is frozen"):
        spec.kind = "selection"
    with pytest.raises(TypeError, match="ToolProvenanceOperation instances only"):
        full_data({"op": "average", "dims": ["y"]})
    with pytest.raises(TypeError, match="ToolProvenanceOperation instances only"):
        spec.append_replacement_operations(
            {"op": "thin", "mode": "global", "factor": 2}
        )


def test_tool_provenance_parse_final_payload_and_migrate_legacy_schema() -> None:
    payload = {
        "schema_version": 1,
        "kind": "full_data",
        "operations": [
            {"op": "average", "dims": {_TUPLE_MARKER: ["x"]}},
            {"op": "rename", "name": "avg"},
        ],
    }

    spec = parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 3
    assert [op.op for op in spec.operations] == ["average", "rename"]
    assert [entry.label for entry in spec.derivation_entries()] == [
        "Start from current parent ImageTool data",
        'Average(dims=("x",))',
        "rename('avg')",
    ]
    assert [entry.label for entry in spec.display_entries()] == [
        "Start from current parent ImageTool data",
        'Average(dims=("x",))',
    ]
    assert spec.derivation_code() == "derived = data.qsel.mean('x')"
    display_code = typing.cast("str", spec.display_code())
    assert ".rename(" not in display_code
    namespace = _exec_generated_code(display_code, {"data": _base_data()})
    xr.testing.assert_identical(
        namespace["derived"].rename(None),
        _base_data().qsel.mean("x").rename(None),
    )

    dumped = spec.model_dump(mode="json")
    assert dumped["schema_version"] == 3
    assert "active_name" in dumped
    assert dumped["active_name"] is None
    assert dumped["operations"][0]["op"] == "average"
    assert dumped["operations"][0]["dims"] == {_TUPLE_MARKER: ["x"]}
    assert dumped["steps"] == []
    assert spec.to_replay_spec().active_name == "derived"

    with pytest.raises(ValidationError, match="Unknown provenance operation"):
        parse_tool_provenance_spec(
            {
                "kind": "full_data",
                "operations": [
                    {
                        "op": "transform",
                        "name": "average",
                        "kwargs": {"dims": ["x"]},
                    }
                ],
            }
        )

    with pytest.raises(
        ValidationError, match="script provenance specs must define `active_name`"
    ):
        parse_tool_provenance_spec(
            {
                "kind": "script",
                "start_label": "Start from current parent ImageTool data",
                "seed_code": "derived = data",
                "operations": [],
            }
        )

    with pytest.raises(
        TypeError, match="Serialized provenance operations must be a sequence"
    ):
        parse_tool_provenance_spec({"kind": "full_data", "operations": 1})

    with pytest.raises(
        TypeError, match="Serialized provenance operations must be a sequence"
    ):
        parse_tool_provenance_spec(
            {"kind": "full_data", "operations": {"op": "average", "dims": ["x"]}}
        )


def test_tool_provenance_migrates_legacy_nonuniform_restore_code() -> None:
    public = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 1.0], "y": [0.0, 1.0]},
    )
    uniform = erlab.utils.array._make_dims_uniform(public)
    legacy_call = "erlab.interactive.imagetool.slicer.restore_nonuniform_dims(data)"
    payload = {
        "schema_version": 2,
        "kind": "script",
        "start_label": "Restore legacy ImageTool data",
        "active_name": "derived",
        "operations": [
            {
                "op": "script_code",
                "label": "Restore nonuniform dimensions",
                "code": f"derived = {legacy_call}",
            }
        ],
    }

    spec = parse_tool_provenance_spec(payload)

    assert spec is not None
    migrated_code = spec.derivation_code()
    assert "erlab.interactive.imagetool.slicer.restore_nonuniform_dims" not in (
        migrated_code
    )
    assert "erlab.utils.array._restore_nonuniform_dims(data)" in migrated_code
    namespace = _exec_generated_code(migrated_code, {"data": uniform})
    xr.testing.assert_identical(namespace["derived"], public)

    seed_spec = parse_tool_provenance_spec(
        {
            "schema_version": 2,
            "kind": "script",
            "start_label": "Restore legacy ImageTool data",
            "seed_code": f"derived = {legacy_call}",
            "active_name": "derived",
            "operations": [],
        }
    )
    assert seed_spec is not None
    assert seed_spec.seed_code is not None
    assert "erlab.utils.array._restore_nonuniform_dims(data)" in seed_spec.seed_code

    file_source = FileLoadSource.model_validate(
        {
            "path": "scan.nc",
            "loader_label": "Load scan.nc",
            "loader_text": "xarray.load_dataarray",
            "kwargs_text": "{}",
            "load_code": f"derived = {legacy_call}",
        }
    )
    assert file_source.load_code is not None
    assert "erlab.utils.array._restore_nonuniform_dims(data)" in file_source.load_code

    preserved_code = (
        f'# café: keep "{legacy_call}" as documentation\n'
        f"derived  =  {legacy_call}  # keep formatting\n"
    )
    assert _migrate_legacy_nonuniform_restore_code(preserved_code) == (
        f'# café: keep "{legacy_call}" as documentation\n'
        "derived  =  erlab.utils.array._restore_nonuniform_dims(data)  "
        "# keep formatting\n"
    )


def test_tool_provenance_migrates_legacy_parent_data_script_context() -> None:
    """Keep schema-v2 ScriptCodeOperation payloads replayable without new usage."""
    payload = {
        "schema_version": 2,
        "kind": "script",
        "start_label": "Run saved script",
        "seed_code": "derived = data",
        "active_name": "result",
        "operations": [
            {
                "op": "script_code",
                "label": "Use saved parent context",
                "code": "result = parent_data + derived",
            }
        ],
        "script_context_bindings": [
            {"operation_index": 0, "names": ["parent_data"]},
            {"operation_index": 0, "names": ["derived", "parent_data"]},
        ],
    }

    spec = parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 3
    assert spec.steps[0].context_names == ("parent_data", "derived")
    assert "script_context_bindings" not in spec.model_dump(mode="json")
    data = xr.DataArray([1.0, 2.0], dims=("x",))
    xr.testing.assert_identical(
        replay_script_provenance(spec, {"data": data}),
        data + data,
    )


def test_tool_provenance_migrates_legacy_top_level_operation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema_version": 2,
        "kind": "script",
        "start_label": "Run saved script",
        "seed_code": "derived = data",
        "active_name": "derived",
        "operations": [{"op": "average", "dims": ["x"]}],
    }
    seen_contexts: list[tuple[xr.DataArray, xr.DataArray]] = []
    original_apply = AverageOperation.apply

    def apply_schema_v2(
        self: AverageOperation,
        data: xr.DataArray,
        *,
        parent_data: xr.DataArray,
    ) -> xr.DataArray:
        seen_contexts.append((data, parent_data))
        return original_apply(self, data)

    monkeypatch.setattr(AverageOperation, "_apply_schema_v2", apply_schema_v2)

    spec = parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.steps[0].legacy_context is not None
    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))
    xr.testing.assert_identical(
        replay_script_provenance(spec, {"data": data}),
        data.mean("x"),
    )
    assert len(seen_contexts) == 1
    xr.testing.assert_identical(seen_contexts[0][0], data)
    xr.testing.assert_identical(seen_contexts[0][1], data)


def test_tool_provenance_replays_external_schema_v2_operation_contract() -> None:
    class LegacyPluginOperation(ToolProvenanceOperation):
        op: typing.Literal["test_legacy_plugin_parent_data"] = (
            "test_legacy_plugin_parent_data"
        )

        def apply(  # type: ignore[override]
            self,
            data: xr.DataArray,
            *,
            parent_data: xr.DataArray,
        ) -> xr.DataArray:
            return data + parent_data

    _OPERATION_TYPES["test_legacy_plugin_parent_data"] = LegacyPluginOperation
    try:
        spec = parse_tool_provenance_spec(
            {
                "schema_version": 2,
                "kind": "script",
                "start_label": "Run saved plugin operation",
                "seed_code": "derived = data",
                "active_name": "derived",
                "operations": [{"op": "test_legacy_plugin_parent_data"}],
            }
        )

        assert spec is not None
        assert spec.steps[0].legacy_context is not None
        data = xr.DataArray([1.0, 2.0], dims=("x",))
        xr.testing.assert_identical(
            replay_script_provenance(spec, {"data": data}),
            data + data,
        )
    finally:
        _OPERATION_TYPES.pop("test_legacy_plugin_parent_data", None)


def test_tool_provenance_discards_saved_cosmetic_coordinate_sorting() -> None:
    spec = parse_tool_provenance_spec(
        {
            "schema_version": 2,
            "kind": "script",
            "start_label": "Run saved script",
            "seed_code": "derived = data",
            "active_name": "derived",
            "replay_stages": [
                {
                    "source_kind": "selection",
                    "operations": [
                        {"op": "isel", "kwargs": {"x": 0}},
                        {"op": "sort_coord_order"},
                    ],
                }
            ],
            "operations": [{"op": "sort_coord_order"}],
        }
    )

    assert spec is not None
    assert [operation.op for operation in spec.operations] == ["isel"]
    assert all(step.operation.op != "sort_coord_order" for step in spec.steps)
    assert "sort_coord_order" not in str(spec.model_dump(mode="json"))


def test_registered_provenance_define_operation_code_api() -> None:

    structured_operation_types = [
        operation_type
        for op, operation_type in _OPERATION_TYPES.items()
        if op != "script_code"
    ]
    assert [
        operation_type
        for operation_type in structured_operation_types
        if "derivation_label" not in operation_type.__dict__
    ] == []
    assert [
        operation_type
        for operation_type in structured_operation_types
        if (
            operation_type.expression_code is ToolProvenanceOperation.expression_code
            and operation_type.statement_code is ToolProvenanceOperation.statement_code
        )
    ] == []


@pytest.mark.parametrize(
    "operation",
    [
        IselOperation(kwargs={"x": slice(0, 2)}),
        SelOperation(kwargs={"y": 11.0}),
        DivideByCoordOperation(coord_name="scale"),
        GaussianFilterOperation(sigma={"x": 0.5}),
        NormalizeOperation(
            dims=("x",),
            mode="minmax",
        ),
        CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        ),
        ThinOperation(
            mode="per_dim",
            factors={"y": 2},
        ),
    ],
)
def test_operation_replay_code_uses_requested_names(
    operation: ToolProvenanceOperation,
) -> None:
    data = _base_data().drop_vars("x_alt").assign_coords(scale=("x", [1.0, 2.0, 3.0]))

    code = operation.replay_code("data", output_name="result", source_name="source")
    assert "derived" not in code

    namespace = _exec_generated_code(
        code,
        {
            "data": data.copy(deep=True),
            "source": data.copy(deep=True),
        },
    )
    result = namespace["result"]
    expected = operation.apply(data)
    if isinstance(
        operation,
        DivideByCoordOperation,
    ):
        result = result.rename(None)
        expected = expected.rename(None)
        assert ".rename(" not in code
    xr.testing.assert_identical(result, expected)


def test_operation_replay_code_passes_source_context() -> None:
    parent = _base_data()
    child = parent.transpose("z", "x", "y")
    operation = SortCoordOrderOperation()

    code = operation.replay_code("child", output_name="result", source_name="parent")
    assert "parent.coords.keys()" in code
    assert "derived" not in code

    namespace = _exec_generated_code(
        code,
        {
            "child": child.copy(deep=True),
            "parent": parent.copy(deep=True),
        },
    )
    xr.testing.assert_identical(
        namespace["result"],
        operation._apply_schema_v2(child, parent_data=parent),
    )


def test_operation_code_base_edges() -> None:
    class ExternalStatementOperation(ToolProvenanceOperation):
        def apply(self, data: xr.DataArray) -> xr.DataArray:
            return data + 1

        def derivation_label(self) -> str:
            return "External statement operation"

        def expression_code(
            self, input_name: str, *, source_name: str | None = None
        ) -> str:
            raise NotImplementedError

        def statement_code(
            self,
            input_name: str,
            *,
            output_name: str,
            source_name: str | None = None,
        ) -> str:
            return f"{output_name} = {input_name} + 1"

    data = xr.DataArray(np.arange(4.0), dims=("x",))

    with pytest.raises(NotImplementedError):
        ToolProvenanceOperation().expression_code("data")
    with pytest.raises(NotImplementedError):
        ToolProvenanceOperation().statement_code(
            "data",
            output_name="derived",
        )
    with pytest.raises(NotImplementedError):
        KspaceWorkFunctionOperation(work_function=4.2).replay_code(
            "data",
            output_name=None,
        )
    assert (
        ExternalStatementOperation().replay_code(
            "data",
            output_name="result",
            reserved_names={"external_helper"},
        )
        == "result = data + 1"
    )

    assert (
        IselOperation(kwargs={"x": 0}).replay_code("data", output_name=None)
        == "data.isel(x=0)"
    )
    assert _expression_receiver_code("data +") == "(data +)"
    assert (
        _simplify_display_code(
            "derived = data\nfor item in []:\n    pass",
            inline_targets={"derived"},
        )
        == "for item in []:\n    pass"
    )
    xr.testing.assert_identical(
        NormalizeOperation(dims=()).apply(data),
        data,
    )


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (
            NormalizeOperation(dims=("x",), mode="area"),
            'data / data.mean("x")',
        ),
        (
            NormalizeOperation(dims=("x",), mode="min"),
            'data - data.min("x")',
        ),
        (
            NormalizeOperation(dims=("x",), mode="min_area"),
            '(data - data.min("x")) / data.mean("x")',
        ),
    ],
)
def test_normalize_operation_expression_modes(
    operation: NormalizeOperation,
    expected: str,
) -> None:
    assert operation.expression_code("data") == expected


def test_roi_operation_derivation_labels() -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 1.0]])

    path_operation = SliceAlongPathOperation(
        vertices={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        step_size=0.1,
        dim_name="s",
    )
    mask_operation = MaskWithPolygonOperation(
        vertices=vertices,
        dims=("x", "y"),
    )

    assert path_operation.derivation_label().startswith("Slice Along ROI Path(")
    assert mask_operation.derivation_label().startswith("Mask with ROI(")


def test_operations_expression_code_chains_without_relay_assignments() -> None:
    data = _base_data().assign_coords(scale=("x", [1.0, 2.0, 3.0]))
    operations = (
        DivideByCoordOperation(coord_name="scale"),
        IselOperation(kwargs={"x": slice(0, 2)}),
    )

    code = operations_expression_code(operations, "data")
    assert code.startswith("(data / data.scale).isel(")
    assert ".rename(" not in code
    assert "derived" not in code

    namespace = _exec_generated_code(
        f"result = {code}",
        {"data": data.copy(deep=True)},
    )
    expected = operations[1].apply(
        operations[0].apply(data),
    )
    xr.testing.assert_identical(namespace["result"].rename(None), expected.rename(None))


def test_generated_provenance_code_elides_meaningless_reassignments() -> None:
    data = _base_data()
    parent = script(
        ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = data + 1",
        ),
        start_label="Start from current tool input data",
        active_name="result",
    )
    structured = compose_full_provenance(
        parent,
        full_data(AverageOperation(dims=("x",))),
    )
    scripted = compose_full_provenance(
        parent,
        script(
            ScriptCodeOperation(label="Mean", code="result = derived.mean()"),
            start_label="Use parent result",
            active_name="derived",
        ),
    )
    assert structured is not None
    assert scripted is not None

    cases = (
        (full_data(AverageOperation(dims=("x",))), "derived", data.qsel.mean("x")),
        (structured, "derived", (data + 1).qsel.mean("x")),
        (scripted, "result", (data + 1).mean()),
    )
    for spec, output_name, expected in cases:
        for code in (spec.derivation_code(), spec.display_code()):
            assert code is not None
            _assert_no_meaningless_reassignments(code)
            assert "erlab.interactive.imagetool" not in code
            namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
            xr.testing.assert_identical(namespace[output_name], expected)


def test_generated_provenance_code_preserves_effect_order_across_aliases() -> None:
    spec = script(
        ScriptCodeOperation(
            label="Read NumPy error state",
            code=("derived = xr.DataArray([numpy.geterr()['divide'] == 'warn'])"),
        ),
        ScriptCodeOperation(
            label="Change NumPy error state",
            code="np.seterr(divide='ignore')",
        ),
        ScriptCodeOperation(
            label="Copy result",
            code="derived = derived.copy()",
        ),
        start_label="Run script",
        active_name="derived",
    )
    previous_error_state = np.seterr(divide="warn")
    try:
        expected = replay_script_provenance(spec, {}, trusted_user_code=True)
        for code in (spec.derivation_code(), spec.display_code()):
            assert code is not None
            np.seterr(divide="warn")
            namespace = _exec_generated_code(
                code,
                {"np": np, "numpy": np, "xr": xr},
            )
            xr.testing.assert_identical(namespace["derived"], expected)
            assert code.index("numpy.geterr") < code.index("np.seterr")
    finally:
        np.seterr(**previous_error_state)


def test_generated_provenance_code_preserves_seed_load_before_effectful_call() -> None:
    original = xr.DataArray([1.0], dims=("x",))
    replacement = xr.DataArray([2.0], dims=("x",))
    spec = script(
        ScriptCodeOperation(
            label="Run effectful callable",
            code="derived = replace_data()(derived)",
        ),
        start_label="Start from input data",
        seed_code="derived = data",
        active_name="derived",
    )

    def execute(code: str) -> dict[str, typing.Any]:
        namespace: dict[str, typing.Any] = {"data": original}

        def replace_data() -> collections.abc.Callable[[xr.DataArray], xr.DataArray]:
            namespace["data"] = replacement
            return lambda value: value

        namespace["replace_data"] = replace_data
        exec(code, namespace, namespace)  # noqa: S102
        return namespace

    for code in (spec.derivation_code(), spec.display_code()):
        assert code is not None
        namespace = execute(code)

        xr.testing.assert_identical(namespace["derived"], original)
        xr.testing.assert_identical(namespace["data"], replacement)


@pytest.mark.parametrize("record_mapping", [False, True])
def test_nonuniform_restore_statement_code_is_safe_to_reingest(
    record_mapping: bool,
) -> None:
    public = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 1.0], "y": [0.0, 1.0]},
    )
    internal = erlab.utils.array._make_dims_uniform(public)
    operation = RestoreNonuniformDimsOperation(
        dimension_mapping=(
            erlab.utils.array._nonuniform_dim_mapping(internal)
            if record_mapping
            else None
        )
    )

    with pytest.raises(NotImplementedError):
        operations_expression_code((operation,), "data")

    code = operation.replay_code("data", output_name="derived")
    spec = script(
        start_label="Restore ImageTool dimensions",
        seed_code=code,
        active_name="derived",
    )

    assert "lambda" not in code
    assert not script_provenance_requires_trust(spec, external_input_names={"data"})
    xr.testing.assert_identical(
        replay_script_provenance(spec, {"data": internal}),
        public,
    )


def test_nonuniform_restore_expression_is_linear_and_restores_applicable_dims() -> None:
    public = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 1.0], "y": [0.0, 0.4, 2.0]},
    )
    internal = erlab.utils.array._make_dims_uniform(public)
    recorded_mapping = erlab.utils.array._nonuniform_dim_mapping(internal)
    mapping = {**recorded_mapping, "missing_idx": "missing"}
    code = _restore_nonuniform_dims_expression("data", mapping)

    attributes = tuple(
        node.attr
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Attribute)
    )
    assert attributes.count("swap_dims") == 1
    assert attributes.count("drop_vars") == 1
    assert len(code) < 5_000

    partially_restored = erlab.utils.array._restore_nonuniform_dims(
        internal, {"x_idx": "x"}
    )
    namespace = _exec_generated_code(
        f"result = {code}",
        {"data": partially_restored},
    )
    xr.testing.assert_identical(namespace["result"], public)


def test_nonuniform_restore_expression_preserves_inapplicable_coordinates() -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0], "x_idx": ("x", [10, 11, 12])},
    )
    code = _restore_nonuniform_dims_expression("data", {"x_idx": "x"})

    namespace = _exec_generated_code(f"result = {code}", {"data": data})

    xr.testing.assert_identical(namespace["result"], data)


def test_statement_operation_replay_code_mutates_working_copy() -> None:
    data = _kspace_data()
    operation = KspaceWorkFunctionOperation(work_function=4.2)

    code = operation.replay_code("data", output_name="result", source_name="data")

    assert "result = data.copy(deep=False)" in code
    assert "result.kspace.work_function = 4.2" in code
    assert "sample_workfunction" not in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    assert namespace["result"].kspace.work_function == pytest.approx(4.2)
    assert namespace["data"].kspace.work_function == pytest.approx(4.5)


def test_statement_operation_derivation_entry_omits_same_name_noop() -> None:
    operation = KspaceSetNormalOperation(alpha=1.5, beta=-0.5, delta=2.0)

    code = operation.derivation_entry().code

    assert code == "derived.kspace.set_normal(alpha=1.5, beta=-0.5, delta=2.0)"


def test_tool_provenance_mixed_statement_and_expression_display_code() -> None:
    data = _kspace_data()
    operations = (
        KspaceWorkFunctionOperation(work_function=4.2),
        KspaceSetNormalOperation(alpha=1.5, beta=-0.5, delta=2.0),
        KspaceConvertOperation(
            bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
            resolution={"kx": 0.02, "ky": 0.02},
        ),
    )
    spec = full_data(*operations).to_replay_spec()

    code = typing.cast("str", spec.display_code())

    assert "derived = data.copy(deep=False)" in code
    assert code.count(".copy(deep=False)") == 1
    assert "derived.kspace.work_function = 4.2" in code
    assert "derived.kspace.set_normal(" in code
    assert "derived = derived.kspace.convert(" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    expected = data.copy(deep=False)
    expected.kspace.work_function = 4.2
    expected.kspace.set_normal(alpha=1.5, beta=-0.5, delta=2.0)
    expected = expected.kspace.convert(
        bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
        resolution={"kx": 0.02, "ky": 0.02},
        silent=True,
    )
    xr.testing.assert_allclose(namespace["derived"], expected)
    assert namespace["data"].kspace.work_function == pytest.approx(4.5)


def test_tool_provenance_parse_legacy_file_script_metadata(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray([1.0, 2.0, 3.0], dims="x")
    path = tmp_path / "scan.h5"
    data.to_netcdf(path, engine="h5netcdf")
    payload = {
        "schema_version": 1,
        "kind": "script",
        "start_label": "Load data from file 'scan.h5'",
        "seed_code": f"import xarray\n\nderived = xarray.load_dataarray({str(path)!r})",
        "active_name": "derived",
        "file_load_source": {
            "path": str(path),
            "loader_label": "Load Function",
            "loader_text": "xarray.load_dataarray",
            "kwargs_text": "(none)",
            "load_code": (
                f"import xarray\n\ndata = xarray.load_dataarray({str(path)!r})"
            ),
        },
        "operations": [
            {
                "op": "script_code",
                "label": 'Average(dims=("x",))',
                "code": 'derived = derived.qsel.average("x")',
                "copyable": True,
            }
        ],
    }

    spec = parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 3
    assert spec.kind == "script"
    assert spec.file_load_source is not None
    assert spec.file_load_source.path == str(path)
    assert spec.file_load_source.replay_call is None
    assert [operation.op for operation in spec.operations] == ["average"]
    assert spec.display_rows()[1].edit_ref == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    code = spec.derivation_code()
    assert code is not None
    assert ".qsel.mean(" in code
    assert "_itool_replay_" not in code
    namespace = _exec_generated_code(code, {})
    xr.testing.assert_identical(namespace["derived"], data.qsel.mean("x"))


def test_tool_provenance_apply_selection_and_xarray_operations() -> None:
    data = _base_data()
    unsorted = erlab.utils.array.sort_coord_order(
        data,
        tuple(reversed(tuple(data.coords))),
        dims_first=False,
    )
    assert tuple(selection().apply(unsorted).coords) == (
        *unsorted.dims,
        *(coord for coord in unsorted.coords if coord not in unsorted.dims),
    )

    nonuniform_public = xr.DataArray(
        np.arange(24).reshape((4, 3, 2)),
        dims=("alpha", "eV", "beta"),
        coords={
            "alpha": [0.0, 0.6, 1.7, 3.0],
            "eV": [-0.2, 0.0, 0.2],
            "beta": [1.0, 2.0],
        },
        name="data",
    )
    nonuniform = erlab.utils.array._make_dims_uniform(nonuniform_public)
    selection_spec = selection(
        QSelOperation(kwargs={"beta": 2.0}),
        IselOperation(kwargs={"alpha": slice(1, 3)}),
        SortCoordOrderOperation(),
    )
    assert [operation.op for operation in selection_spec.operations] == ["qsel", "isel"]
    xr.testing.assert_identical(
        selection_spec.apply(nonuniform),
        nonuniform_public.qsel(beta=2.0).isel({"alpha": slice(1, 3)}),
    )

    transformed = full_data(
        IselOperation(kwargs={"z": 0}),
        SelOperation(kwargs={"y": slice(11.0, 12.0)}),
        TransposeOperation(dims=("y", "x")),
        SqueezeOperation(),
        RenameOperation(name="done"),
    )
    xr.testing.assert_identical(
        transformed.apply(data),
        data.isel({"z": 0})
        .sel({"y": slice(11.0, 12.0)})
        .transpose("y", "x")
        .squeeze()
        .rename("done"),
    )

    xr.testing.assert_identical(
        full_data(AverageOperation(dims=("y",))).apply(data),
        data.qsel.mean("y"),
    )
    xr.testing.assert_identical(
        full_data(
            QSelAggregationOperation(
                dims=("y",),
                func="sum",
            )
        ).apply(data),
        data.qsel.sum("y"),
    )
    xr.testing.assert_identical(
        full_data(
            CoarsenOperation(
                dim={"y": 2},
                boundary="trim",
                side="left",
                coord_func="mean",
                reducer="mean",
            )
        ).apply(data),
        data.coarsen(y=2, boundary="trim", side="left", coord_func="mean").mean(),
    )
    xr.testing.assert_identical(
        full_data(ThinOperation(mode="global", factor=2)).apply(data),
        data.thin(2),
    )
    xr.testing.assert_identical(
        full_data(ThinOperation(mode="per_dim", factors={"x": 2})).apply(data),
        data.thin({"x": 2}),
    )
    xr.testing.assert_identical(
        full_data(SwapDimsOperation(mapping={"x": "x_alt"})).apply(data),
        data.swap_dims({"x": "x_alt"}),
    )
    xr.testing.assert_identical(
        full_data(
            RenameDimsCoordsOperation(mapping={"x": "kx", "x_alt": "label"})
        ).apply(data),
        data.rename({"x": "kx", "x_alt": "label"}),
    )

    assigned = full_data(
        AssignCoordsOperation(
            coord_name="y", values=np.array([100.0, 101.0, 102.0, 103.0])
        )
    ).apply(data)
    expected_assigned = erlab.utils.array.sort_coord_order(
        data.assign_coords(
            {"y": data["y"].copy(data=np.array([100.0, 101.0, 102.0, 103.0]))}
        ),
        keys=data.coords.keys(),
        dims_first=False,
    )
    xr.testing.assert_identical(assigned, expected_assigned)


def test_tool_provenance_rename_dims_coords_round_trip_and_code() -> None:
    data = _string_key_data().assign_coords(
        {"coord-1": xr.DataArray([100.0, 101.0, 102.0], dims=["k-space"])}
    )
    operation = RenameDimsCoordsOperation(
        mapping={"k-space": "kx", "coord-1": "temperature"}
    )
    expected = data.rename({"k-space": "kx", "coord-1": "temperature"})

    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation
    xr.testing.assert_identical(operation.apply(data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    spec = full_data(operation).to_replay_spec()
    code = spec.display_code(parent_data=data)
    assert code is not None
    assert ".rename(" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


def test_tool_provenance_interpolation_operation_round_trip_and_code() -> None:
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=("k-space", "y"),
        coords={"k-space": [0.0, 1.0, 2.0], "y": [10.0, 20.0]},
        name="data",
    )
    values = np.linspace(0.0, 2.0, 5)
    operation = InterpolationOperation(dim="k-space", values=values, method="linear")
    expected = data.interp({"k-space": values}, method="linear")

    xr.testing.assert_identical(operation.apply(data), expected)
    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    assert "Interpolate" in entry.label
    assert '.interp({"k-space": np.linspace' in entry.code
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    code = full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".interp") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


def test_tool_provenance_leading_edge_operation_round_trip_and_code() -> None:
    ev = np.linspace(0.0, 4.0, 5)
    data = xr.DataArray(
        np.vstack([4.0 - ev, 8.0 - 2.0 * ev, 2.0 - 0.5 * ev]),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": ev},
        name="data",
    )
    operation = LeadingEdgeOperation(
        fraction=0.5,
        dim="eV",
        direction="positive",
    )
    expected = erlab.analysis.interpolate.leading_edge(data)

    xr.testing.assert_identical(operation.apply(data), expected)
    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data), expected)

    payload = full_data(operation).model_dump(mode="json")
    json.dumps(payload)
    reparsed_spec = parse_tool_provenance_spec(payload)
    assert reparsed_spec is not None
    xr.testing.assert_identical(reparsed_spec.apply(data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    assert "leading_edge" in entry.code
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    code = full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".leading_edge") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


@pytest.mark.parametrize(
    ("values", "expected_call"),
    [
        (np.array([100.0, 101.0, 102.0, 103.0]), "np.arange"),
        (np.linspace(0.25, 1.0, 4), "np.linspace"),
        (np.ones(4), "np.linspace"),
        (np.array([0.0, 0.5, 2.0, 3.0]), "np.array"),
    ],
)
def test_tool_provenance_assign_coords_replay_display_code(
    values: np.ndarray, expected_call: str
) -> None:
    data = _base_data()

    spec = full_data(
        AssignCoordsOperation(coord_name="y", values=values)
    ).to_replay_spec()

    code = spec.display_code(parent_data=data)
    assert code is not None
    call_names = _generated_call_names(code)
    assert any(call.endswith(".assign_coords") for call in call_names)
    assert "erlab.utils.array.sort_coord_order" not in call_names
    assert expected_call in call_names

    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_allclose(
        namespace["derived"],
        data.assign_coords({"y": data["y"].copy(data=values)}),
    )


def test_tool_provenance_assign_coords_single_value_uses_linspace() -> None:
    data = xr.DataArray(
        np.arange(1),
        dims=("x",),
        coords={"x": np.array([0.0])},
        name="data",
    )
    values = np.array([5.0])

    spec = full_data(
        AssignCoordsOperation(coord_name="x", values=values)
    ).to_replay_spec()

    code = spec.display_code(parent_data=data)
    assert code is not None
    call_names = _generated_call_names(code)
    assert "np.linspace" in call_names
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_allclose(
        namespace["derived"],
        data.assign_coords({"x": data["x"].copy(data=values)}),
    )


def test_tool_provenance_assign_scalar_coord_operation() -> None:
    data = _base_data()
    operation = AssignScalarCoordOperation(coord_name="temperature", value=21.5)
    expected = erlab.utils.array.sort_coord_order(
        data.assign_coords({"temperature": 21.5}),
        keys=data.coords.keys(),
        dims_first=False,
    )

    xr.testing.assert_identical(operation.apply(data), expected)
    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation

    code = full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".assign_coords") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"], data.assign_coords(temperature=21.5)
    )


def test_tool_provenance_nonfinite_coord_and_attr_code() -> None:
    data = _base_data()

    scalar_spec = full_data(
        AssignScalarCoordOperation(coord_name="temperature", value=np.nan)
    )
    scalar_code = typing.cast("str", scalar_spec.derivation_code())
    assert "np.nan" in scalar_code
    scalar_namespace = _exec_generated_code(scalar_code, {"data": data.copy(deep=True)})
    assert np.isnan(scalar_namespace["derived"].coords["temperature"].item())

    attrs_spec = full_data(
        AssignAttrsOperation(
            attrs={
                "offset": np.inf,
                "bad": np.nan,
                "complex": complex(float("nan"), float("inf")),
            }
        )
    )
    attrs_code = typing.cast("str", attrs_spec.derivation_code())
    assert "np.nan" in attrs_code
    assert "np.inf" in attrs_code
    assert "complex(np.nan, np.inf)" in attrs_code
    attrs_namespace = _exec_generated_code(attrs_code, {"data": data.copy(deep=True)})
    assert np.isinf(attrs_namespace["derived"].attrs["offset"])
    assert np.isnan(attrs_namespace["derived"].attrs["bad"])
    assert np.isnan(attrs_namespace["derived"].attrs["complex"].real)
    assert np.isinf(attrs_namespace["derived"].attrs["complex"].imag)

    coord_spec = full_data(
        AssignCoord1DOperation(
            coord_name="temperature",
            dim="x",
            values=np.array([np.nan, np.inf, -np.inf]),
        )
    )
    coord_code = typing.cast("str", coord_spec.derivation_code())
    assert "np.nan" in coord_code
    assert "np.inf" in coord_code
    coord_namespace = _exec_generated_code(coord_code, {"data": data.copy(deep=True)})
    np.testing.assert_equal(
        coord_namespace["derived"].coords["temperature"].values,
        np.array([np.nan, np.inf, -np.inf]),
    )


def test_tool_provenance_assign_1d_coord_operation() -> None:
    data = _base_data()
    values = np.array(["low", "mid", "high"])
    operation = AssignCoord1DOperation(
        coord_name="label",
        dim="x",
        values=values,
    )
    expected = erlab.utils.array.sort_coord_order(
        data.assign_coords({"label": ("x", values)}),
        keys=data.coords.keys(),
        dims_first=False,
    )

    xr.testing.assert_identical(operation.apply(data), expected)
    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation

    code = full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".assign_coords") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"], data.assign_coords(label=("x", values))
    )


def test_tool_provenance_assign_attrs_operation() -> None:
    data = _base_data().assign_attrs(source="old", count=1)
    attrs = {"source": "new", "flag": True, "meta": {"scan": 1}}
    operation = AssignAttrsOperation(attrs=attrs)
    expected = data.assign_attrs(attrs)

    xr.testing.assert_identical(operation.apply(data), expected)
    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation

    code = full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".assign_attrs") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


def test_kspace_configuration_operation_round_trip_and_code() -> None:
    data = _kspace_data()
    operation = KspaceConfigurationOperation(
        configuration=erlab.constants.AxesConfiguration.Type2
    )
    expected = data.kspace.as_configuration(erlab.constants.AxesConfiguration.Type2)

    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))

    assert parsed == operation
    xr.testing.assert_identical(operation.apply(data), expected)
    assert operation.derivation_entry().label == "Set kspace configuration(2 Type2)"
    code = operation.replay_code("anglemap", output_name="converted")
    assert code == "converted = anglemap.kspace.as_configuration(2)"
    namespace = _exec_generated_code(code, {"anglemap": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["converted"], expected)


@pytest.mark.parametrize(
    ("operation", "attr_name", "expected"),
    [
        (
            KspaceWorkFunctionOperation(work_function=4.2),
            "sample_workfunction",
            4.2,
        ),
        (
            KspaceInnerPotentialOperation(inner_potential=12.0),
            "inner_potential",
            12.0,
        ),
    ],
)
def test_kspace_scalar_statement_operations_round_trip_and_code(
    operation: ToolProvenanceOperation,
    attr_name: str,
    expected: float,
) -> None:
    data = _kspace_data()
    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))

    result = parsed.apply(data)

    assert result.attrs[attr_name] == pytest.approx(expected)
    assert data.attrs[attr_name] != pytest.approx(expected)
    code = parsed.replay_code("data", output_name="result")
    assert "result = data.copy(deep=False)" in code
    if attr_name == "sample_workfunction":
        assert "result.kspace.work_function =" in code
        assert "result.attrs" not in code
    else:
        assert "result.kspace.inner_potential =" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    assert namespace["result"].attrs[attr_name] == pytest.approx(expected)
    assert namespace["data"].attrs[attr_name] != pytest.approx(expected)


def test_kspace_set_normal_operation_round_trip_and_code() -> None:
    data = _kspace_data()
    operation = KspaceSetNormalOperation(
        alpha=1.5,
        beta=-0.5,
        delta=2.0,
    )

    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    result = parsed.apply(data)

    assert result.kspace.offsets["delta"] == pytest.approx(2.0)
    assert result.kspace.offsets != data.kspace.offsets
    code = parsed.replay_code("data", output_name="result")
    assert "result = data.copy(deep=False)" in code
    assert "result.kspace.set_normal(alpha=1.5, beta=-0.5, delta=2.0)" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["result"], result)
    for key, value in data.kspace.offsets.items():
        assert namespace["data"].kspace.offsets[key] == pytest.approx(value)


def test_kspace_convert_operation_round_trip_and_code() -> None:
    data = _kspace_data()
    operation = KspaceConvertOperation(
        bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
        resolution={"kx": 0.02, "ky": 0.02},
    )
    expected = data.kspace.convert(
        bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
        resolution={"kx": 0.02, "ky": 0.02},
        silent=True,
    )

    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))

    assert parsed == operation
    xr.testing.assert_allclose(parsed.apply(data), expected)
    code = parsed.replay_code("data", output_name="result")
    assert "result = data.kspace.convert(" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_allclose(namespace["result"], expected)


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            ConsoleCall(
                accessor_path=("kspace", "as_configuration"),
                args=(2,),
                display_code="data.kspace.as_configuration(2)",
                has_extra_tracked_inputs=False,
            ),
            KspaceConfigurationOperation(configuration=2),
        ),
        (
            ConsoleCall(
                accessor_path=("kspace", "set_normal"),
                args=(1.5, -0.5),
                kwargs={"delta": 2.0},
                display_code="data.kspace.set_normal(1.5, -0.5, delta=2.0)",
                has_extra_tracked_inputs=False,
            ),
            KspaceSetNormalOperation(alpha=1.5, beta=-0.5, delta=2.0),
        ),
        (
            ConsoleCall(
                accessor_path=("kspace", "convert"),
                args=(),
                kwargs={
                    "bounds": {"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
                    "resolution": {"kx": 0.02, "ky": 0.02},
                },
                display_code=(
                    "data.kspace.convert(bounds=bounds, resolution=resolution)"
                ),
                has_extra_tracked_inputs=False,
            ),
            KspaceConvertOperation(
                bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
                resolution={"kx": 0.02, "ky": 0.02},
            ),
        ),
    ],
)
def test_kspace_operations_match_console_calls(
    call: ConsoleCall,
    operation: ToolProvenanceOperation,
) -> None:
    assert operation_from_console_call(call) == operation


def _expected_affine_coord(
    data: xr.DataArray, coord_name: str, scale: float, offset: float
) -> xr.DataArray:
    coord = data.coords[coord_name]
    return erlab.utils.array.sort_coord_order(
        data.assign_coords(
            {coord_name: coord.copy(data=scale * coord.values + offset)}
        ),
        keys=data.coords.keys(),
        dims_first=False,
    )


@pytest.mark.parametrize(
    ("data", "coord_name", "scale", "offset"),
    [
        (_base_data(), "y", 2.0, -1.0),
        (
            _base_data().assign_coords(temp=("x", [100.0, 200.0, 300.0])),
            "temp",
            0.5,
            1.0,
        ),
        (_base_data().assign_coords(temp=20.0), "temp", 1.5, -2.0),
        (
            _base_data().assign_coords({"beam current": ("x", [1.0, 2.0, 4.0])}),
            "beam current",
            3.0,
            0.25,
        ),
    ],
)
def test_tool_provenance_affine_coord_operation(
    data: xr.DataArray, coord_name: str, scale: float, offset: float
) -> None:
    operation = AffineCoordOperation(
        coord_name=coord_name,
        scale=scale,
        offset=offset,
    )

    expected = _expected_affine_coord(data, coord_name, scale, offset)
    xr.testing.assert_identical(operation.apply(data), expected)

    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data), expected)

    code = full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    call_names = _generated_call_names(code)
    assert any(call.endswith(".assign_coords") for call in call_names)
    assert "erlab.utils.array.sort_coord_order" not in call_names
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"],
        data.assign_coords(
            {
                coord_name: data.coords[coord_name].copy(
                    data=scale * data.coords[coord_name].values + offset
                )
            }
        ),
    )


@pytest.mark.parametrize(
    ("scale", "offset", "expected_fragments", "forbidden_fragments"),
    [
        (1.0, -3.7, ("['y'].values - 3.7",), ("1.0 *", "+ -3.7")),
        (2.0, -3.7, ("2.0 *", "['y'].values - 3.7"), ("+ -3.7",)),
        (
            1.0,
            0.0,
            ("['y'].values)",),
            ("1.0 *", "+ 0.0", "- 0.0"),
        ),
    ],
)
def test_tool_provenance_affine_coord_display_code_formats_no_ops(
    scale: float,
    offset: float,
    expected_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
) -> None:
    data = _base_data()
    operation = AffineCoordOperation(
        coord_name="y",
        scale=scale,
        offset=offset,
    )

    code = full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    for fragment in expected_fragments:
        assert fragment in code
    for fragment in forbidden_fragments:
        assert fragment not in code

    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"], _expected_affine_coord(data, "y", scale, offset)
    )


@pytest.mark.parametrize(
    ("scale", "offset"),
    [
        (np.nan, 0.0),
        (np.inf, 0.0),
        (1.0, np.nan),
        (1.0, -np.inf),
    ],
)
def test_tool_provenance_affine_coord_rejects_nonfinite_values(
    scale: float, offset: float
) -> None:
    with pytest.raises(
        ValidationError, match="affine coordinate scale and offset must be finite"
    ):
        AffineCoordOperation(
            coord_name="y",
            scale=scale,
            offset=offset,
        )


def test_tool_provenance_divide_by_coord_operation() -> None:
    data = _base_data().assign_coords(mesh_current=("x", [1.0, 2.0, 4.0]))

    spec = full_data(DivideByCoordOperation(coord_name="mesh_current"))
    expected = (data / data.mesh_current).rename(data.name)
    xr.testing.assert_identical(spec.apply(data), expected)
    code = spec.derivation_code()
    assert code is not None
    assert "derived.mesh_current" in code
    assert ".rename(" not in code
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], data / data.mesh_current)

    reparsed = parse_tool_provenance_spec(spec.model_dump(mode="json"))
    assert reparsed == spec
    xr.testing.assert_identical(reparsed.apply(data), expected)


def test_tool_provenance_divide_by_coord_fallback_code_and_broadcast() -> None:
    data = _base_data().assign_coords(
        {
            "mesh current": ("x", [1.0, 2.0, 4.0]),
            "mesh_map": (
                ("x", "y"),
                np.arange(12, dtype=float).reshape(3, 4) + 1.0,
            ),
            "mean": ("x", [1.0, 2.0, 4.0]),
        }
    )

    spaced_spec = full_data(DivideByCoordOperation(coord_name="mesh current"))
    spaced_code = spaced_spec.derivation_code()
    assert spaced_code is not None
    assert 'derived.coords["mesh current"]' in spaced_code
    assert ".rename(" not in spaced_code
    namespace = _exec_generated_code(spaced_code, {"data": data})
    xr.testing.assert_identical(
        namespace["derived"], data / data.coords["mesh current"]
    )

    conflict_spec = full_data(DivideByCoordOperation(coord_name="mean"))
    conflict_code = conflict_spec.derivation_code()
    assert conflict_code is not None
    assert 'derived.coords["mean"]' in conflict_code
    assert ".rename(" not in conflict_code
    namespace = _exec_generated_code(conflict_code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], data / data.coords["mean"])

    broadcast_spec = full_data(DivideByCoordOperation(coord_name="mesh_map"))
    xr.testing.assert_identical(
        broadcast_spec.apply(data), (data / data.coords["mesh_map"]).rename(data.name)
    )


def test_tool_provenance_divide_by_coord_rejects_zero_values() -> None:
    data = _base_data().assign_coords(mesh_current=("x", [1.0, 0.0, 4.0]))
    spec = full_data(DivideByCoordOperation(coord_name="mesh_current"))

    with pytest.raises(ValueError, match="zero values"):
        spec.apply(data)


def test_tool_provenance_public_data_replays_on_restored_nonuniform_dims() -> None:
    public = xr.DataArray(
        np.arange(20).reshape((5, 4)),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4)},
        name="data",
    )
    uniform = erlab.utils.array._make_dims_uniform(public)

    spec = public_data(
        CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        )
    )
    reparsed = parse_tool_provenance_spec(spec.model_dump(mode="json"))

    assert reparsed is not None
    assert reparsed.kind == "public_data"
    xr.testing.assert_identical(
        reparsed.apply(uniform),
        public.coarsen(x=2, boundary="trim", side="left", coord_func="mean").mean(),
    )

    display_code = reparsed.display_code(parent_data=uniform)
    assert display_code is not None
    assert "coarsen(x=2" in display_code
    assert "x_idx" not in display_code

    restored_spec = full_data(
        AverageOperation(dims=("y",)),
        RestoreNonuniformDimsOperation(
            dimension_mapping=erlab.utils.array._nonuniform_dim_mapping(uniform)
        ),
    )
    reparsed_restored = parse_tool_provenance_spec(
        restored_spec.model_dump(mode="json")
    )

    assert reparsed_restored is not None
    xr.testing.assert_identical(
        reparsed_restored.apply(uniform),
        public.qsel.mean("y"),
    )
    restored_code = reparsed_restored.display_code(parent_data=uniform)
    assert restored_code is not None
    assert "restore_nonuniform_dims" not in restored_code
    assert ".swap_dims(" in restored_code
    assert ".drop_vars(" in restored_code
    restored_namespace = _exec_generated_code(restored_code, {"data": uniform})
    xr.testing.assert_identical(restored_namespace["derived"], public.qsel.mean("y"))


def test_recorded_nonuniform_mapping_is_not_restored_after_rotation_drops_coord() -> (
    None
):
    public = xr.DataArray(
        np.arange(20.0).reshape(5, 4),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4.0)},
        name="scan",
    )
    internal = erlab.utils.array._make_dims_uniform(public)
    mapping = erlab.utils.array._nonuniform_dim_mapping(internal)
    rotation = RotateOperation(
        angle=10.0,
        axes=("x_idx", "y"),
        center=(0.0, 0.0),
    )
    rotated = rotation.apply(internal)
    expected = erlab.utils.array._restore_nonuniform_dims(rotated)
    spec = full_data(
        rotation,
        RestoreNonuniformDimsOperation(dimension_mapping=mapping),
    )

    assert "x_idx" in rotated.dims
    assert "x" not in rotated.coords
    assert expected.dims == ("x_idx", "y")
    xr.testing.assert_identical(spec.apply(internal), expected)

    code = typing.cast("str", spec.display_code(parent_data=internal))
    namespace = _exec_generated_code(code, {"data": internal})

    assert "erlab.utils.array._restore_nonuniform_dims" not in code
    assert "erlab.interactive.imagetool.slicer" not in code
    xr.testing.assert_identical(namespace["derived"], expected)


def test_tool_provenance_preserves_hashable_dims_and_mapping_keys() -> None:
    data = _hashable_data()
    string_key_data = _string_key_data()

    qsel_spec = full_data(QSelOperation(kwargs={"k-space": 1.0, "k-space_width": 1.0}))
    assert qsel_spec.derivation_code() == (
        "derived = data.qsel({'k-space': 1.0, 'k-space_width': 1.0})"
    )
    xr.testing.assert_identical(
        qsel_spec.apply(string_key_data),
        string_key_data.qsel({"k-space": 1.0, "k-space_width": 1.0}),
    )

    isel_spec = full_data(IselOperation(kwargs={1: slice(1, 3)}))
    assert isel_spec.derivation_code() == "derived = data.isel({1: slice(1, 3)})"
    xr.testing.assert_identical(isel_spec.apply(data), data.isel({1: slice(1, 3)}))

    transpose_spec = full_data(TransposeOperation(dims=(("beta", 0), 1)))
    assert transpose_spec.derivation_code() == (
        "derived = data.transpose(*(('beta', 0), 1))"
    )
    xr.testing.assert_identical(
        transpose_spec.apply(data), data.transpose(("beta", 0), 1)
    )

    average_spec = full_data(AverageOperation(dims=("k-space",)))
    assert average_spec.derivation_code() == "derived = data.qsel.mean('k-space')"
    xr.testing.assert_identical(
        average_spec.apply(string_key_data), string_key_data.qsel.mean("k-space")
    )

    tuple_average_spec = full_data(AverageOperation(dims=(("beta", 0),)))
    assert tuple_average_spec.derivation_code() == (
        "derived = data.qsel.mean((('beta', 0),))"
    )

    aggregate_spec = full_data(QSelAggregationOperation(dims=("k-space",), func="sum"))
    assert aggregate_spec.derivation_code() == "derived = data.qsel.sum('k-space')"
    xr.testing.assert_identical(
        aggregate_spec.apply(string_key_data), string_key_data.qsel.sum("k-space")
    )

    mean_aggregate_spec = full_data(
        QSelAggregationOperation(dims=(("beta", 0),), func="mean")
    )
    assert mean_aggregate_spec.derivation_code() == (
        "derived = data.qsel.mean((('beta', 0),))"
    )

    dumped = aggregate_spec.model_dump(mode="json")
    assert dumped["operations"][0] == {
        "op": "qsel_aggregate",
        "dims": {_TUPLE_MARKER: ["k-space"]},
        "func": "sum",
    }
    reparsed = parse_tool_provenance_spec(dumped)
    assert reparsed.operations[0] == aggregate_spec.operations[0]

    coarsen_spec = full_data(
        CoarsenOperation(
            dim={1: 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        )
    )
    assert coarsen_spec.derivation_code() == (
        "derived = data.coarsen(dim={1: 2}, boundary='trim').mean()"
    )
    xr.testing.assert_identical(
        coarsen_spec.apply(data),
        data.coarsen(
            dim={1: 2}, boundary="trim", side="left", coord_func="mean"
        ).mean(),
    )

    thin_spec = full_data(ThinOperation(mode="per_dim", factors={1: 2}))
    assert thin_spec.derivation_code() == "derived = data.thin({1: 2})"
    xr.testing.assert_identical(thin_spec.apply(data), data.thin({1: 2}))

    swap_spec = full_data(SwapDimsOperation(mapping={1: "coord_1"}))
    assert swap_spec.derivation_code() == "derived = data.swap_dims({1: 'coord_1'})"
    xr.testing.assert_identical(swap_spec.apply(data), data.swap_dims({1: "coord_1"}))

    dumped = tuple_average_spec.model_dump(mode="json")
    assert dumped["operations"][0]["dims"] == {
        _TUPLE_MARKER: [{_TUPLE_MARKER: ["beta", 0]}]
    }

    coarsen_dump = coarsen_spec.model_dump(mode="json")
    assert coarsen_dump["operations"][0]["dim"] == {_MAPPING_MARKER: [[1, 2]]}


def test_tool_provenance_display_entries_streamline_live_source() -> None:
    data = _base_data()

    hidden_spec = full_data(
        IselOperation(kwargs={}),
        SortCoordOrderOperation(),
        TransposeOperation(dims=data.dims),
        SqueezeOperation(),
    )
    assert [entry.label for entry in hidden_spec.display_entries(parent_data=data)] == [
        "Start from current parent ImageTool data"
    ]
    assert hidden_spec.display_code(parent_data=data) is None

    squeezed_spec = full_data(
        IselOperation(kwargs={"z": slice(0, 1)}),
        SqueezeOperation(),
    )
    squeezed_entries = squeezed_spec.display_entries(parent_data=data)
    assert squeezed_entries[0].label == "Start from current parent ImageTool data"
    assert squeezed_entries[-1].label == "squeeze()"
    squeezed_code = typing.cast("str", squeezed_spec.display_code(parent_data=data))
    squeezed_namespace = _exec_generated_code(
        squeezed_code,
        {"data": data.copy(deep=True)},
    )
    squeezed = squeezed_namespace["derived"]
    assert isinstance(squeezed, xr.DataArray)
    xr.testing.assert_identical(
        squeezed,
        data.isel(z=slice(0, 1)).squeeze(),
    )

    hidden_dim_spec = full_data(SqueezeOperation(dims=("z",)))
    assert [
        entry.label for entry in hidden_dim_spec.display_entries(parent_data=data)
    ] == ["Start from current parent ImageTool data"]
    assert hidden_dim_spec.display_code(parent_data=data) is None

    specific_squeezed_spec = full_data(
        IselOperation(kwargs={"z": slice(0, 1)}),
        SqueezeOperation(dims=("z",), drop=True),
    )
    specific_squeezed_entries = specific_squeezed_spec.display_entries(parent_data=data)
    assert specific_squeezed_entries[-1].label == 'squeeze(dim=("z",), drop=True)'
    specific_squeezed_code = typing.cast(
        "str", specific_squeezed_spec.display_code(parent_data=data)
    )
    specific_namespace = _exec_generated_code(
        specific_squeezed_code,
        {"data": data.copy(deep=True)},
    )
    specific_squeezed = specific_namespace["derived"]
    assert isinstance(specific_squeezed, xr.DataArray)
    xr.testing.assert_identical(
        specific_squeezed,
        data.isel(z=slice(0, 1)).squeeze(dim=("z",), drop=True),
    )

    chained_squeeze_data = xr.DataArray(
        np.ones((1, 1)),
        dims=("x", "z"),
        name="scan",
    )
    chained_squeeze_spec = full_data(
        SqueezeOperation(dims=("z",)),
        SqueezeOperation(),
    )
    chained_squeeze_entries = chained_squeeze_spec.display_entries(
        parent_data=chained_squeeze_data
    )
    assert [entry.label for entry in chained_squeeze_entries] == [
        "Start from current parent ImageTool data",
        'squeeze(dim=("z",))',
        "squeeze()",
    ]
    chained_squeeze_code = typing.cast(
        "str",
        chained_squeeze_spec.display_code(parent_data=chained_squeeze_data),
    )
    chained_namespace = _exec_generated_code(
        chained_squeeze_code,
        {"data": chained_squeeze_data.copy(deep=True)},
    )
    chained_squeezed = chained_namespace["derived"]
    assert isinstance(chained_squeezed, xr.DataArray)
    xr.testing.assert_identical(
        chained_squeezed,
        chained_squeeze_data.squeeze(dim=("z",)).squeeze(),
    )


def test_tool_provenance_display_streamlining_is_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _base_data()
    calls: list[str] = []

    def record_apply(
        self: ToolProvenanceOperation,
        data: xr.DataArray,
    ) -> xr.DataArray:
        calls.append(self.op)
        return data

    monkeypatch.setattr(
        SymmetrizeNfoldOperation,
        "apply",
        record_apply,
    )
    monkeypatch.setattr(
        KspaceConvertOperation,
        "apply",
        record_apply,
    )
    spec = full_data(
        SymmetrizeNfoldOperation(fold=4, axes=("x", "y")),
        KspaceConvertOperation(),
    )

    rows = spec.display_rows(parent_data=data)
    entries = spec.display_entries(parent_data=data)
    code = spec.display_code(parent_data=data)

    assert calls == []
    assert any(row.entry.label.startswith("Rotational Symmetrize") for row in rows)
    assert any(entry.label.startswith("Convert to momentum") for entry in entries)
    assert code is not None
    assert "symmetrize_nfold" in code
    assert ".kspace.convert(" in code

    staged_spec = script(
        start_label="Run script",
        seed_code="derived = data",
        active_name="derived",
        replay_stages=(ReplayStage.from_source_spec(spec),),
    )
    assert staged_spec.display_rows(parent_data=data)
    assert staged_spec.display_code(parent_data=data) is not None
    assert calls == []

    spec.apply(data)
    assert calls == ["symmetrize_nfold", "kspace_convert"]


def test_tool_provenance_unknown_display_context_keeps_noop_candidates() -> None:
    spec = full_data(
        TransposeOperation(dims=("x", "y", "z")),
        SqueezeOperation(),
        RestoreNonuniformDimsOperation(),
    )
    assert [entry.label for entry in spec.display_entries()] == [
        "Start from current parent ImageTool data",
        "transpose(('x', 'y', 'z'))",
        "squeeze()",
        "Restore nonuniform dimensions",
    ]
    unknown_code = typing.cast("str", spec.display_code())
    assert ".transpose(" in unknown_code
    assert ".squeeze()" in unknown_code
    assert "def _restore_image_tool_dimensions" in unknown_code
    assert "erlab.utils.array._restore_nonuniform_dims" not in unknown_code

    data = _base_data()
    assert [entry.label for entry in spec.display_entries(parent_data=data)] == [
        "Start from current parent ImageTool data",
    ]
    assert spec.display_code(parent_data=data) is None


def test_tool_provenance_display_metadata_context_branches() -> None:
    data = _base_data()
    context_cls = _ProvenanceDisplayContext

    context = context_cls.from_source("full_data", data)
    assert context.dims == data.dims
    assert context.sizes == dict(data.sizes)
    nonuniform_data = xr.DataArray(np.ones(2), dims=("x_idx",), name="scan")
    assert context_cls.from_source("public_data", nonuniform_data).dims is None
    assert context_cls.dims_may_restore_nonuniform(("x_idx",))
    assert not context_cls.dims_may_restore_nonuniform(("x",))

    indexer_size = context_cls._isel_indexer_size
    assert indexer_size(slice(None, None, 2), 5) == (True, 3)
    assert indexer_size(True, 5) == (False, None)
    assert indexer_size(np.bool_(False), 5) == (False, None)
    assert indexer_size(1, 5) == (True, None)
    assert indexer_size(np.int64(1), 5) == (True, None)
    assert indexer_size(xr.DataArray([0], dims=("index",)), 5) == (False, None)
    assert indexer_size(np.array(1), 5) == (True, None)
    assert indexer_size(np.array([0, 2]), 5) == (True, 2)
    assert indexer_size(np.array([[0, 1]]), 5) == (False, None)
    assert indexer_size(np.array([True, False]), 5) == (False, None)
    assert indexer_size(range(0, 5, 2), 5) == (True, 3)
    assert indexer_size([True, False], 5) == (False, None)
    assert indexer_size([0, 2], 5) == (True, 2)
    assert indexer_size(object(), 5) == (False, None)

    assert context.advance(RenameOperation(name="renamed")) == context
    assert context.advance(_SourceViewOperation(source_kind="full_data")) == context
    assert context.advance(_SourceViewOperation(source_kind="public_data")) == context
    nonuniform_context = context_cls(("x_idx", "y"), {"x_idx": 3, "y": 4})
    assert (
        nonuniform_context.advance(_SourceViewOperation(source_kind="public_data")).dims
        is None
    )
    assert context.advance(QSelOperation()) == context
    assert context.advance(QSelOperation(kwargs={"x": 1.0})).dims is None
    assert context.advance(SelOperation()) == context
    assert context.advance(SelOperation(kwargs={"x": 1.0})).dims is None
    assert context.advance(IselOperation()) == context
    assert context.advance(IselOperation(kwargs={"missing": 0})).dims is None
    assert (
        context.advance(
            IselOperation(kwargs={"x": xr.DataArray([0], dims=("index",))})
        ).dims
        is None
    )
    assert context.advance(IselOperation(kwargs={"z": 0})).dims == (
        "x",
        "y",
    )
    vector_index_context = context.advance(IselOperation(kwargs={"x": [0, 2]}))
    assert vector_index_context.dims == data.dims
    assert vector_index_context.sizes == {"x": 2, "y": 4, "z": 2}
    assert context.advance(TransposeOperation(dims=("x", "missing", "z"))).dims is None
    assert context.advance(TransposeOperation()).dims == ("z", "y", "x")

    unknown_context = context_cls()
    assert unknown_context.advance(TransposeOperation()).dims is None
    assert unknown_context.advance(SqueezeOperation()).dims is None

    assert context.advance(SqueezeOperation(dims=("missing",))).dims is None
    mixed_squeeze_context = context_cls.from_source(
        "full_data", xr.DataArray(np.ones((1, 2)), dims=("x", "z"), name="scan")
    )
    assert mixed_squeeze_context.advance(SqueezeOperation(dims=("x", "z"))).dims is None
    assert (
        mixed_squeeze_context.advance(SqueezeOperation(dims=("z",)))
        == mixed_squeeze_context
    )
    assert mixed_squeeze_context.advance(SqueezeOperation(dims=("x",))).dims == ("z",)
    assert mixed_squeeze_context.advance(SqueezeOperation()).dims == ("z",)
    assert context.advance(RestoreNonuniformDimsOperation()) == context
    assert nonuniform_context.advance(RestoreNonuniformDimsOperation()).dims is None
    assert context.advance(AverageOperation(dims=("x",))).dims is None

    staged_spec = script(
        start_label="Run script",
        seed_code="derived = data",
        active_name="derived",
        replay_stages=(
            ReplayStage(
                source_kind="full_data",
                operations=(
                    RestoreNonuniformDimsOperation(),
                    SqueezeOperation(),
                ),
            ),
        ),
    )
    assert [
        entry.label for entry in staged_spec._code_fallback_entries(parent_data=data)
    ] == ["Run script"]


def test_tool_provenance_display_entries_keep_ambiguous_script_steps() -> None:

    spec = script(
        ScriptCodeOperation(label="isel()", code="derived = derived.isel()"),
        ScriptCodeOperation(
            label="Sort coordinates to parent order",
            code=(
                "derived = erlab.utils.array.sort_coord_order("
                "derived, data.coords.keys())"
            ),
        ),
        ScriptCodeOperation(
            label="Custom coordinate-order step",
            code=(
                "derived = erlab.utils.array.sort_coord_order("
                "derived, data.coords.keys(), dims_first=False)"
            ),
        ),
        ScriptCodeOperation(
            label="transpose(('x', 'y', 'z'))",
            code="derived = derived.transpose(*('x', 'y', 'z'))",
        ),
        ScriptCodeOperation(
            label="squeeze()",
            code="derived = derived.squeeze()",
        ),
        start_label="Start from current analysis-tool input data",
        seed_code="derived = data",
    )

    code = spec.display_code()
    assert code is not None
    call_names = _generated_call_names(code)
    assert not any(call.endswith(".isel") for call in call_names)
    assert call_names.count("erlab.utils.array.sort_coord_order") == 1
    assert any(call.rsplit(".", maxsplit=1)[-1] == "transpose" for call in call_names)
    assert any(call.rsplit(".", maxsplit=1)[-1] == "squeeze" for call in call_names)
    namespace = _exec_generated_code(code, {"data": _base_data().copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(
        derived,
        _base_data().transpose(*("x", "y", "z")).squeeze(),
    )


def test_tool_provenance_display_keeps_name_rename_before_script_code() -> None:
    data = _base_data().rename("source")
    spec = script(
        RenameOperation(name="renamed"),
        ScriptCodeOperation(
            label="Use DataArray name",
            code="derived = derived.rename(derived.name + '_used')",
        ),
        start_label="Run script",
        seed_code="derived = data",
        active_name="derived",
    )

    code = typing.cast("str", spec.display_code())

    assert ".rename(" in code
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], data.rename("renamed_used"))


def test_imagetool_selection_source_binding_materializes_current_coordinates() -> None:
    original = _base_data()
    shifted = original.assign_coords(y=[20.0, 21.0, 22.0, 23.0])

    binned = ImageToolSelectionSourceBinding(
        selection_indexers={"y": slice(1, 4)},
        selection_binned_dims=("y",),
    )
    old_spec = binned.materialize(original)
    new_spec = binned.materialize(shifted)

    assert old_spec.operations[0].decoded_kwargs == {"y": 12.0, "y_width": 3.0}
    assert new_spec.operations[0].decoded_kwargs == {"y": 22.0, "y_width": 3.0}
    xr.testing.assert_identical(
        new_spec.apply(shifted), shifted.qsel(y=22.0, y_width=3.0)
    )

    unbinned = ImageToolSelectionSourceBinding(selection_indexers={"y": 2})
    unbinned_spec = unbinned.materialize(shifted)
    assert unbinned_spec.operations[0].decoded_kwargs == {"y": 22.0}
    xr.testing.assert_identical(unbinned_spec.apply(shifted), shifted.qsel(y=22.0))

    cropped = ImageToolSelectionSourceBinding(crop_sel_indexers={"y": slice(1, 3)})
    cropped_spec = cropped.materialize(shifted)
    assert cropped_spec.operations[0].decoded_kwargs == {"y": slice(21.0, 22.0)}
    xr.testing.assert_identical(
        cropped_spec.apply(shifted),
        shifted.sel(y=slice(21.0, 22.0)),
    )


def test_imagetool_selection_source_binding_round_trips_and_reuses_operations() -> None:
    data = _base_data()
    binding = ImageToolSelectionSourceBinding(
        selection_mode="isel",
        selection_indexers={"z": 1},
        crop_sel_indexers={"x": slice(0, 3)},
        crop_isel_indexers={"y": slice(1, 3)},
        transpose_dims=("y", "x"),
        squeeze=True,
    )

    reparsed = ImageToolSelectionSourceBinding.model_validate(
        binding.model_dump(mode="json")
    )
    assert reparsed == binding

    spec = reparsed.materialize(data)
    assert [op.op for op in spec.operations] == [
        "isel",
        "sel",
        "isel",
        "transpose",
        "squeeze",
    ]
    xr.testing.assert_identical(
        spec.apply(data),
        erlab.utils.array.sort_coord_order(
            data.isel(z=1).sel(x=slice(0.0, 2.0)).isel(y=slice(1, 3)),
            data.coords.keys(),
        )
        .transpose("y", "x")
        .squeeze(),
    )


def test_imagetool_selection_source_binding_validates_crop_indexers() -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
    )

    with pytest.raises(ValueError, match="Dimension `missing` not found"):
        ImageToolSelectionSourceBinding(
            crop_sel_indexers={"missing": slice(0, 1)}
        ).materialize(data)

    with pytest.raises(ValueError, match="Selection for dimension `x` is empty"):
        ImageToolSelectionSourceBinding(
            crop_sel_indexers={"x": slice(1, 1)}
        ).materialize(data)

    binding = ImageToolSelectionSourceBinding(
        crop_sel_indexers={"x": typing.cast("typing.Any", 1)}
    )
    spec = binding.materialize(data)
    sel_kwargs = next(op.decoded_kwargs for op in spec.operations if op.op == "sel")

    assert sel_kwargs == {"x": 1.0}


def test_tool_provenance_rejects_unsupported_hashables() -> None:
    class _UnsupportedHashable:
        def __hash__(self) -> int:
            return 0

    with pytest.raises(TypeError, match="provenance hashable fields only support"):
        AverageOperation(dims=(_UnsupportedHashable(),))


def test_tool_provenance_validation_helpers_and_error_branches() -> None:
    base_operation = ToolProvenanceOperation()

    assert _format_derivation_value([1, 2]) == "(1, 2)"
    assert _format_selection_step("isel", {}) == "derived = derived.isel()"
    assert _simplify_display_code("if") == "if"
    assert _simplify_display_code("") == ""
    assert (
        _simplify_display_code("for item in []:\n    pass")
        == "for item in []:\n    pass"
    )
    assert _simplify_display_code("derived = data\nresult = derived + 1") == (
        "result = data + 1"
    )
    simplified = _simplify_display_code(
        "derived = data\nscale = 2\nresult = derived + scale"
    )
    simplified_namespace = _exec_generated_code(simplified, {"data": 3})
    assert simplified_namespace["result"] == 5
    assert simplified_namespace["derived"] == 3
    assert (
        _simplify_display_code(
            "derived = data\nscale = 2\nleft, right = (data * scale, data + 1)",
            inline_targets={"derived"},
        )
        == "scale = 2\nleft, right = (data * scale, data + 1)"
    )
    assert "record()" in _simplify_display_code(
        "derived = record()\nresult = data",
        inline_targets={"derived"},
    )
    assert (
        _simplify_display_code(
            "left, right = (data - 1, data + 1)\n"
            "derived = left\n"
            "derived = derived.sel(x=0)"
        )
        == "left, right = (data - 1, data + 1)\nderived = left.sel(x=0)"
    )
    invalidated_namespace = _exec_generated_code(
        _simplify_display_code("derived = data + 1\ndata = other\nresult = derived"),
        {"data": 3, "other": 10},
    )
    assert invalidated_namespace["result"] == 4
    rebased = rebase_default_replay_input(
        "derived = data\nscale = 2\nresult = derived + scale",
        "source_data",
    )
    rebased_namespace = _exec_generated_code(rebased, {"source_data": 3})
    assert rebased_namespace["result"] == 5
    assert rebased_namespace["derived"] == 3
    assert uses_default_replay_input("result = data + 1")
    assert not uses_default_replay_input("result = source_data + 1")
    helper_code = (
        "def normalize(data):\n"
        "    return data / data.max()\n"
        "\n"
        "derived = normalize(data_0)"
    )
    assert not uses_default_replay_input(helper_code)
    assert rebase_default_replay_input(helper_code, "source_data") == helper_code
    mixed_helper_code = (
        "def normalize(data):\n"
        "    return data / data.max()\n"
        "\n"
        "derived = normalize(data)"
    )
    rebased_helper_code = rebase_default_replay_input(mixed_helper_code, "source_data")
    assert "def normalize(data):\n    return data / data.max()" in rebased_helper_code
    assert "derived = normalize(source_data)" in rebased_helper_code
    free_input_helper_code = (
        "def transform():\n    return data + 1\n\nderived = transform()"
    )
    assert uses_default_replay_input(free_input_helper_code)
    rebased_free_helper_code = rebase_default_replay_input(
        free_input_helper_code,
        "source_data",
    )
    assert not uses_default_replay_input(rebased_free_helper_code)
    free_helper_namespace = {"source_data": 3}
    exec(  # noqa: S102
        rebased_free_helper_code,
        free_helper_namespace,
        free_helper_namespace,
    )
    assert free_helper_namespace["derived"] == 4
    same_line_lambdas = (
        "first = lambda: data + 1; second = lambda: data + 2\n"
        "derived = first() + second()"
    )
    rebased_lambdas = rebase_default_replay_input(same_line_lambdas, "source_data")
    lambda_namespace = {"source_data": 1}
    exec(rebased_lambdas, lambda_namespace, lambda_namespace)  # noqa: S102
    assert lambda_namespace["derived"] == 5
    replaced_helper_code = _replace_code_identifiers(
        "def normalize(data):\n    return data / data.max()\n\nderived = data",
        {"data": "source_data", "derived": "result"},
    )
    assert "def normalize(data):\n    return data / data.max()" in replaced_helper_code
    assert "result = source_data" in replaced_helper_code
    renamed_import_code = _replace_code_identifiers(
        "import numpy as result\nresult = result.float64(2)",
        {"result": "script_result"},
    )
    assert _exec_generated_code(renamed_import_code, {})["script_result"] == 2

    with pytest.raises(ValueError, match="Expected 2 items"):
        _ensure_float_tuple([1.0], expected_len=2)
    with pytest.raises(TypeError, match="expected an array-like sequence"):
        _coerce_float_sequence("not-a-sequence")
    with pytest.raises(TypeError, match="active_name must be a string"):
        _validate_active_name(1)
    with pytest.raises(ValueError, match="active_name must be a valid"):
        _validate_active_name("for")
    with pytest.raises(TypeError, match="expected a sequence"):
        ToolProvenanceOperation._coerce_hashable_tuple_field("x")
    with pytest.raises(ValueError, match="Expected 2 items"):
        ToolProvenanceOperation._coerce_hashable_tuple_field([1], expected_len=2)
    assert ToolProvenanceOperation._coerce_hashable_mapping_field(None) == {}
    with pytest.raises(TypeError, match="expected a mapping"):
        ToolProvenanceOperation._coerce_hashable_mapping_field([("x", 1)])
    with pytest.raises(NotImplementedError):
        base_operation.apply(_base_data())
    with pytest.raises(NotImplementedError):
        base_operation.derivation_entry()
    with pytest.raises(TypeError, match="must be mappings"):
        parse_tool_provenance_operation(1)
    with pytest.raises(TypeError, match="must include a string `op`"):
        parse_tool_provenance_operation({"op": 1})
    with pytest.raises(TypeError, match="array-like"):
        AssignCoordsOperation(coord_name="x", values=object())
    with pytest.raises(TypeError, match=r"xarray\.Dataset"):
        CorrectWithEdgeOperation(edge_fit=object())

    assert ToolProvenanceSpec(kind="full_data", operations=None).operations == ()
    with pytest.raises(ValidationError, match="must define `start_label`"):
        ToolProvenanceSpec(kind="script", active_name="derived")
    with pytest.raises(ValidationError, match="Only script or file provenance specs"):
        ToolProvenanceSpec(kind="full_data", start_label="bad")
    with pytest.raises(ValidationError, match="live-applicable operations"):
        file_load(
            start_label="Load source",
            seed_code="derived = xr.load_dataarray('scan.h5')",
            file_load_source=_file_replay_source(),
            steps=(
                ReplayStep(
                    operation=ScriptCodeOperation(
                        label="Run script-only step",
                        code="derived = derived + 1",
                    )
                ),
            ),
        )
    with pytest.raises(TypeError, match="Script and file provenance use"):
        script(start_label="Start", active_name="derived")._display_operations()


def test_rebase_default_replay_input_respects_lexical_scopes() -> None:
    helper_code = (
        "from __future__ import annotations\n\n"
        "def transform(source):\n"
        "    return data + source\n\n"
        "derived = transform(1)"
    )
    rebased = rebase_default_replay_input(helper_code, "source")
    namespace = {"source": 10}
    exec(rebased, namespace, namespace)  # noqa: S102
    assert namespace["derived"] == 11
    assert not uses_default_replay_input(rebased)

    if sys.version_info >= (3, 12):
        class_comprehension = (
            "class Container:\n"
            "    data = 1\n"
            "    values = [data for _ in range(1)]\n"
            "    functions = [lambda: data for _ in range(1)]\n"
            "derived = (Container.values, Container.functions[0]())"
        )
        rebased_comprehension = rebase_default_replay_input(
            class_comprehension,
            "source",
        )
        namespace = {"source": 10}
        exec(rebased_comprehension, namespace, namespace)  # noqa: S102
        assert namespace["derived"] == ([10], 10)

        generic_code = (
            "def transform[T](value: T) -> T:\n"
            "    return data\n"
            "\n"
            "class Container[T]:\n"
            "    value = data\n"
            "\n"
            "derived = transform(Container.value)"
        )
        rebased_generic = rebase_default_replay_input(generic_code, "source")
        namespace = {"source": 10}
        exec(rebased_generic, namespace, namespace)  # noqa: S102
        assert namespace["derived"] == 10

        type_alias_code = "type Payload[T] = tuple[T, type(data)]"
        rebased_type_alias = rebase_default_replay_input(type_alias_code, "source")
        namespace = {"source": 10}
        exec(rebased_type_alias, namespace, namespace)  # noqa: S102
        assert typing.get_args(namespace["Payload"].__value__)[1] is int
        assert not uses_default_replay_input(rebased_type_alias)

    if sys.version_info >= (3, 14):
        annotation_code = (
            "def transform(value: (lambda: data)()):\n"
            "    return value\n"
            "\n"
            "derived = transform(1)"
        )
        rebased_annotation = rebase_default_replay_input(annotation_code, "source")
        assert not uses_default_replay_input(rebased_annotation)


def test_select_coord_operation_round_trips_and_applies() -> None:
    data = _base_data().assign_coords(temp=("x", [100.0, 200.0, 300.0]))
    operation = SelectCoordOperation(coord_name="temp")

    xr.testing.assert_identical(operation.apply(data), data.coords["temp"])

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], data.coords["temp"])

    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation

    spec = public_data(operation)
    assert require_live_source_spec(spec) == spec
    xr.testing.assert_identical(spec.apply(data), data.coords["temp"])


def test_tool_provenance_remaining_operation_and_display_branches(monkeypatch) -> None:
    data = _base_data()

    xr.testing.assert_identical(
        full_data(TransposeOperation()).apply(data),
        data.transpose(*reversed(data.dims)),
    )
    assert TransposeOperation().derivation_entry().code == (
        "derived = derived.transpose()"
    )
    assert SortCoordOrderOperation().derivation_entry().copyable is True
    assert SelOperation(kwargs={"x": 1.0}).derivation_entry().label.startswith("sel(")
    rename_entry = RenameOperation(name="renamed").derivation_entry()
    assert rename_entry.copyable is True
    assert rename_entry.code is not None
    namespace = _exec_generated_code(
        rename_entry.code,
        {"derived": data.copy(deep=True)},
    )
    xr.testing.assert_identical(namespace["derived"], data.rename("renamed"))
    assert full_data().derivation_code() is None
    assert script(start_label="Start", active_name="derived").display_code() is None
    edge_entry = CorrectWithEdgeOperation(
        edge_fit=xr.Dataset(), shift_coords=True
    ).derivation_entry()
    assert edge_entry.copyable is True
    assert edge_entry.code is not None

    with pytest.raises(TypeError, match="script_code operations"):
        ScriptCodeOperation(label="Step", code="derived = data").apply(data)
    with pytest.raises(ValidationError, match="thin global mode requires factor"):
        ThinOperation(mode="global")
    with pytest.raises(ValidationError, match="thin per_dim mode requires factors"):
        ThinOperation(mode="per_dim")
    assert ThinOperation(mode="global", factor=2).derivation_entry().code == (
        "derived = derived.thin(2)"
    )

    monkeypatch.setattr(
        erlab.interactive.utils,
        "generate_code",
        lambda *_args, assign=None, **_kwargs: (
            "generated()" if assign is None else f"{assign} = generated()"
        ),
    )
    assert (
        RotateOperation(angle=45.0, axes=("x", "y"), center=(0.0, 0.0))
        .derivation_entry()
        .code
        == "derived = generated()"
    )
    assert (
        SymmetrizeOperation(dim="x", center=0.0).derivation_entry().code
        == "derived = generated()"
    )
    assert (
        SymmetrizeNfoldOperation(fold=4, axes=("x", "y")).derivation_entry().code
        == "derived = generated()"
    )
    symmetrize_nfold_payload = SymmetrizeNfoldOperation(
        fold=4, axes=("x", "y"), center=(0.0, 0.0)
    ).model_dump(mode="json")
    assert parse_tool_provenance_operation(
        symmetrize_nfold_payload
    ) == SymmetrizeNfoldOperation(fold=4, axes=("x", "y"), center=(0.0, 0.0))

    assign_entry = AssignCoordsOperation(
        coord_name="x", values=np.array([2.0, 1.0, 0.0])
    ).derivation_entry()
    assert assign_entry.copyable is True
    assert "assign_coords" in typing.cast("str", assign_entry.code)

    ambiguous = full_data(
        SelOperation(kwargs={"missing": 0}),
        SqueezeOperation(),
    )
    assert [entry.label for entry in ambiguous.display_entries(parent_data=data)] == [
        "Start from current parent ImageTool data",
        "sel(missing=0)",
        "squeeze()",
    ]

    parent = script(
        start_label="Start from watched variable 'my_1d'",
        seed_code="derived = my_1d",
    )
    promoted = mark_promoted_1d_source(data.copy(deep=False))
    assert (
        compose_display_provenance(
            parent,
            selection(IselOperation(kwargs={"x": 0})),
            parent_data=promoted,
        )
        is not parent
    )
    assert (
        compose_display_provenance(
            parent,
            selection(AverageOperation(dims=("x",))),
            parent_data=promoted,
        )
        is not parent
    )
    assert (
        direct_replay_input_name(
            script(start_label="Start", seed_code="prepared = data")
        )
        is None
    )
    assert (
        direct_replay_input_name(script(start_label="Start", seed_code="derived = for"))
        is None
    )
    assert compose_full_provenance(parent, None) == parent


def test_append_display_operation_preserves_final_rename_for_live_sources() -> None:
    operation = NormalizeOperation(dims=("x",), mode="min")
    spec = full_data().append_final_rename("filtered")

    displayed = spec.append_display_operation(operation)

    assert [op.op for op in displayed.operations] == [
        "normalize",
        "rename",
    ]
    assert displayed.operations[-1] == RenameOperation(name="filtered")


def test_append_display_operation_rejects_non_live_sources() -> None:
    operation = NormalizeOperation(dims=("x",), mode="min")
    spec = script(
        start_label="Evaluate console expression",
        active_name="derived",
    )

    with pytest.raises(TypeError, match="live sources"):
        spec.append_display_operation(operation)


def test_tool_provenance_apply_analysis_operations(monkeypatch) -> None:
    data = _base_data()
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})
    calls: list[tuple[str, dict[str, object]]] = []

    def _record(name: str):
        def _inner(data_arg, *args, **kwargs):
            calls.append((name, {"args": args, "kwargs": kwargs}))
            return data_arg.assign_attrs(last_op=name)

        return _inner

    monkeypatch.setattr(erlab.analysis.transform, "rotate", _record("rotate"))
    monkeypatch.setattr(erlab.analysis.transform, "symmetrize", _record("symmetrize"))
    monkeypatch.setattr(
        erlab.analysis.transform, "symmetrize_nfold", _record("symmetrize_nfold")
    )
    monkeypatch.setattr(
        erlab.analysis.gold, "correct_with_edge", _record("correct_with_edge")
    )
    monkeypatch.setattr(
        erlab.analysis.interpolate, "slice_along_path", _record("slice_along_path")
    )
    monkeypatch.setattr(
        erlab.analysis.mask, "mask_with_polygon", _record("mask_with_polygon")
    )

    rotate_spec = full_data(
        RotateOperation(
            angle=45.0, axes=("x", "y"), center=(0.5, 1.5), reshape=False, order=3
        )
    )
    assert rotate_spec.apply(data).attrs["last_op"] == "rotate"

    symmetrize_spec = full_data(
        SymmetrizeOperation(
            dim="x", center=1.0, subtract=True, mode="valid", part="below"
        )
    )
    assert symmetrize_spec.apply(data).attrs["last_op"] == "symmetrize"

    symmetrize_nfold_spec = full_data(
        SymmetrizeNfoldOperation(
            fold=4,
            axes=("x", "y"),
            center={"x": 1.0, "y": 11.0},
            reshape=True,
            order=2,
        )
    )
    assert symmetrize_nfold_spec.apply(data).attrs["last_op"] == "symmetrize_nfold"

    edge_spec = full_data(
        CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False)
    )
    assert edge_spec.apply(data).attrs["last_op"] == "correct_with_edge"
    entries = edge_spec.derivation_entries()
    assert entries[-1].copyable is True
    assert entries[-1].code is not None
    assert edge_spec.derivation_code() is not None

    path_spec = full_data(
        SliceAlongPathOperation(
            vertices={"x": [0.0, 1.0], "y": [10.0, 12.0]},
            step_size=0.5,
            dim_name="path",
        )
    )
    assert path_spec.apply(data).attrs["last_op"] == "slice_along_path"

    mask_spec = full_data(
        MaskWithPolygonOperation(
            vertices=np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]),
            dims=("x", "y"),
            invert=True,
            drop=False,
        )
    )
    assert mask_spec.apply(data).attrs["last_op"] == "mask_with_polygon"

    call_names = [name for name, _ in calls]
    assert call_names == [
        "rotate",
        "symmetrize",
        "symmetrize_nfold",
        "correct_with_edge",
        "slice_along_path",
        "mask_with_polygon",
    ]
    assert calls[0][1]["kwargs"] == {
        "angle": 45.0,
        "axes": ("x", "y"),
        "center": (0.5, 1.5),
        "reshape": False,
        "order": 3,
    }


def test_tool_provenance_roundtrip_correct_with_edge(monkeypatch) -> None:
    data = _base_data()
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})

    monkeypatch.setattr(
        erlab.analysis.gold,
        "correct_with_edge",
        lambda data_arg, *_args, **_kwargs: data_arg.assign_attrs(
            last_op="correct_with_edge"
        ),
    )

    spec = full_data(CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False))
    payload = spec.model_dump(mode="json")

    reparsed_operation = parse_tool_provenance_operation(payload["operations"][0])
    assert isinstance(
        reparsed_operation,
        CorrectWithEdgeOperation,
    )
    xr.testing.assert_identical(reparsed_operation.decoded_edge_fit, edge_fit)

    reparsed_spec = parse_tool_provenance_spec(payload)
    assert reparsed_spec is not None
    assert reparsed_spec.apply(data).attrs["last_op"] == "correct_with_edge"
    entries = reparsed_spec.derivation_entries()
    assert entries[-1].copyable is True
    assert entries[-1].code is not None
    namespace = _exec_generated_code(
        typing.cast("str", reparsed_spec.derivation_code()),
        {"data": data.copy(deep=True)},
    )
    assert namespace["derived"].attrs["last_op"] == "correct_with_edge"


def test_correct_with_edge_code_handles_nonfinite_dataset(monkeypatch) -> None:
    data = _base_data()
    edge_fit = xr.Dataset({"edge": ("x", [np.nan, np.inf, -np.inf])})

    def correct_with_edge(data_arg, edge_fit_arg, *, shift_coords=True):
        xr.testing.assert_identical(edge_fit_arg, edge_fit)
        return data_arg.assign_attrs(shift_coords=shift_coords)

    monkeypatch.setattr(erlab.analysis.gold, "correct_with_edge", correct_with_edge)
    spec = full_data(CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False))
    code = typing.cast("str", spec.derivation_code())

    assert "np.nan" in code
    assert "np.inf" in code
    assert "imagetool" not in code
    assert "xr.Dataset.from_dict" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    assert namespace["derived"].attrs["shift_coords"] is False


def test_tool_provenance_roundtrip_correct_with_edge_fit_dataset(
    gold, gold_fit_res
) -> None:
    spec = full_data(
        CorrectWithEdgeOperation(edge_fit=gold_fit_res, shift_coords=False)
    )

    payload = spec.model_dump(mode="json")
    json.dumps(payload)

    reparsed_operation = parse_tool_provenance_operation(payload["operations"][0])
    assert isinstance(reparsed_operation, CorrectWithEdgeOperation)
    decoded = reparsed_operation.decoded_edge_fit
    xr.testing.assert_identical(
        decoded.drop_vars("modelfit_results"),
        gold_fit_res.drop_vars("modelfit_results"),
    )
    assert (
        decoded.modelfit_results.item().success
        == gold_fit_res.modelfit_results.item().success
    )

    reparsed_spec = parse_tool_provenance_spec(payload)
    assert reparsed_spec is not None
    assert reparsed_spec.derivation_code() is None
    xr.testing.assert_allclose(
        reparsed_spec.apply(gold),
        erlab.analysis.gold.correct_with_edge(gold, gold_fit_res, shift_coords=False),
    )


def test_tool_provenance_script_specs_reject_live_source() -> None:

    script_spec = script(
        ScriptCodeOperation(
            label="Fit current tool data",
            code="result = data.mean()",
        ),
        start_label="Start from current analysis-tool input data",
        seed_code="prepared = data.copy()",
    )
    reparsed_script = parse_tool_provenance_spec(script_spec.model_dump(mode="json"))

    assert reparsed_script is not None
    assert reparsed_script.derivation_entries()[0].label == (
        "Start from current analysis-tool input data"
    )
    assert reparsed_script.derivation_code() == (
        "prepared = data.copy()\nresult = data.mean()"
    )
    with pytest.raises(
        TypeError, match="source_spec must be a live ToolProvenanceSpec"
    ):
        require_live_source_spec(reparsed_script)


def test_tool_provenance_parse_restores_legacy_structured_script_steps() -> None:
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "seed_code": "derived = data",
        "active_name": "derived",
        "operations": [
            {
                "op": "script_code",
                "label": "Concatenate selected ImageTools",
                "code": "derived = left + right",
            },
            {
                "op": "script_code",
                "label": 'Sort By(variables=("x",), ascending=False)',
                "code": 'derived = derived.sortby("x", ascending=False)',
            },
            {
                "op": "script_code",
                "label": "Average(dims=('y',))",
                "code": 'derived = derived.qsel.mean("y")',
            },
        ],
    }

    parsed = parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == [
        "script_code",
        "sortby",
        "qsel_aggregate",
    ]
    rows = [
        row
        for row in parsed.display_rows()
        if row.replay_ref is not None and row.replay_ref.kind == "operation"
    ]
    assert [row.edit_ref is not None for row in rows] == [True, True, True]
    data = _base_data()
    replayed = replay_script_provenance(
        parsed,
        {
            "data": data.copy(deep=True),
            "left": data.copy(deep=True),
            "right": data.copy(deep=True),
        },
    )
    xr.testing.assert_identical(
        replayed,
        (data + data).sortby("x", ascending=False).qsel.mean("y"),
    )


def test_tool_provenance_parse_keeps_non_active_self_assignments_as_script() -> None:
    data = xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=("x",),
        coords={"x": [2.0, 1.0, 0.0]},
        name="scan",
    )
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "seed_code": "derived = data",
        "active_name": "derived",
        "operations": [
            {
                "op": "script_code",
                "label": "Prepare temporary data",
                "code": 'temp = derived.sortby("x", ascending=False)',
            },
            {
                "op": "script_code",
                "label": "Sort temporary data",
                "code": 'temp = temp.sortby("x")',
            },
        ],
    }

    parsed = parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == [
        "script_code",
        "script_code",
    ]
    replayed = replay_script_provenance(parsed, {"data": data})
    xr.testing.assert_identical(replayed, data)


def test_tool_provenance_parse_keeps_script_input_self_assignment_as_script() -> None:
    data = xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=("x",),
        coords={"x": [2.0, 1.0, 0.0]},
        name="scan",
    )
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "active_name": "derived",
        "script_inputs": [{"name": "data_0", "label": "ImageTool 0"}],
        "operations": [
            {
                "op": "script_code",
                "label": "Copy input",
                "code": "derived = data_0",
            },
            {
                "op": "script_code",
                "label": "Sort input alias",
                "code": 'data_0 = data_0.sortby("x")',
            },
        ],
    }

    parsed = parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == [
        "script_code",
        "script_code",
    ]
    replayed = replay_script_provenance(parsed, {"data_0": data})
    xr.testing.assert_identical(replayed, data)


def test_tool_provenance_parse_restores_active_alias_structured_steps() -> None:
    data = xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=("x",),
        coords={"x": [2.0, 1.0, 0.0]},
        name="scan",
    )
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "seed_code": "result = data",
        "active_name": "result",
        "operations": [
            {
                "op": "script_code",
                "label": "Sort active result",
                "code": 'result = result.sortby("x")',
            },
        ],
    }

    parsed = parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == ["sortby"]
    replayed = replay_script_provenance(parsed, {"data": data})
    xr.testing.assert_identical(replayed, data.sortby("x"))


def test_tool_provenance_parse_legacy_script_call_parser_edges() -> None:
    def parse_codes(
        *codes: str | None,
        seed_code: str | None = "derived = data",
        active_name: str = "derived",
        script_inputs: tuple[dict[str, str], ...] = (),
        copyable: bool = True,
    ) -> ToolProvenanceSpec:
        parsed = parse_tool_provenance_spec(
            {
                "kind": "script",
                "start_label": "Run ImageTool manager action",
                "seed_code": seed_code,
                "active_name": active_name,
                "script_inputs": script_inputs,
                "operations": [
                    {
                        "op": "script_code",
                        "label": "Legacy script step",
                        "code": code,
                        "copyable": copyable,
                    }
                    for code in codes
                ],
            }
        )
        assert parsed is not None
        return parsed

    parsed = parse_codes(
        "derived = derived.isel(x=slice(0, 2))",
        "derived = derived.coarsen(x=2).mean()",
        "derived = derived.qsel(x=1.0)",
    )
    assert [operation.op for operation in parsed.operations] == [
        "isel",
        "coarsen",
        "qsel",
    ]
    assert parsed.operations[0] == IselOperation(kwargs={"x": slice(0, 2)})
    assert parsed.operations[1] == CoarsenOperation(
        dim={"x": 2},
        boundary="exact",
        side="left",
        coord_func="mean",
        reducer="mean",
    )
    assert parsed.operations[2] == QSelOperation(kwargs={"x": 1.0})

    conservative_codes = (
        "derived = derived.isel(x=slice(0, stop))",
        'derived = derived.sortby(np.array(["x"]))',
        "derived = derived.sortby(np.array([unknown]))",
        "derived = derived.sortby(unknown)",
        "derived = derived.isel(**extra)",
        "derived = make_data()",
        'derived = factory().sortby("x")',
        'derived = factory().data.sortby("x")',
        'derived = other.sortby("x")',
        "derived = derived.unknown.accessor()",
        "derived = derived.coarsen(x=step).mean()",
        'derived = derived.sortby("x")\nextra = derived',
        'derived, other = derived.sortby("x"), other',
        "derived = 1",
        'other = derived.sortby("x")',
    )
    parsed = parse_codes(*conservative_codes)
    assert all(
        isinstance(operation, ScriptCodeOperation) for operation in parsed.operations
    )
    assert [operation.code for operation in parsed.operations] == list(
        conservative_codes
    )

    parsed = parse_codes(
        'derived = derived.sortby("x")',
        seed_code="derived =",
        script_inputs=({"name": "derived", "label": "ImageTool"},),
    )
    assert [operation.op for operation in parsed.operations] == ["sortby"]

    parsed = parse_codes('derived = derived.sortby("x")', seed_code=None)
    assert [operation.op for operation in parsed.operations] == ["script_code"]

    parsed = parse_codes('derived = derived.sortby("x")', copyable=False)
    assert [operation.op for operation in parsed.operations] == ["script_code"]

    parsed = parse_codes(None)
    assert [operation.op for operation in parsed.operations] == ["script_code"]

    parsed = parse_codes("derived =")
    assert [operation.op for operation in parsed.operations] == ["script_code"]


def test_tool_provenance_parse_restores_mixed_active_legacy_steps() -> None:
    data = _base_data().assign_coords(x=[2.0, 1.0, 0.0])
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "seed_code": "result = data",
        "active_name": "result",
        "operations": [
            {
                "op": "script_code",
                "label": "Prepare temporary data",
                "code": "temp = result + 1.0",
            },
            {
                "op": "script_code",
                "label": "Sort active result",
                "code": 'result = result.sortby("x", ascending=False)',
            },
            {
                "op": "script_code",
                "label": "Average active result",
                "code": 'result = result.qsel.mean("y")',
            },
        ],
    }

    parsed = parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == [
        "script_code",
        "sortby",
        "qsel_aggregate",
    ]
    replayed = replay_script_provenance(parsed, {"data": data})
    xr.testing.assert_identical(
        replayed,
        data.sortby("x", ascending=False).qsel.mean("y"),
    )


def test_current_structured_operations_round_trip_without_script_fallback() -> None:
    operations = _representative_structured_operations()
    assert {operation.op for operation in operations} == set(_OPERATION_TYPES) - {
        "script_code",
        "source_view",
    }
    assert (
        parse_tool_provenance_operation(
            {"op": "source_view", "source_kind": "selection"}
        ).op
        == "source_view"
    )

    def assert_round_trip_operations(
        spec: ToolProvenanceSpec,
        expected_ops: tuple[str, ...],
    ) -> None:
        parsed = parse_tool_provenance_spec(spec.model_dump(mode="json"))
        assert parsed is not None
        parsed_ops = parsed.operations
        assert tuple(operation.op for operation in parsed_ops) == expected_ops
        assert not any(isinstance(op, ScriptCodeOperation) for op in parsed_ops)

    for operation in operations:
        expected = () if operation.op == "sort_coord_order" else (operation.op,)
        assert_round_trip_operations(full_data(operation), expected)
        assert_round_trip_operations(public_data(operation), expected)
        assert_round_trip_operations(selection(operation), expected)
        assert_round_trip_operations(
            full_data(operation).to_replay_spec(),
            expected,
        )
        assert_round_trip_operations(
            file_load(
                start_label="Load data",
                seed_code="derived = data",
                file_load_source=_file_replay_source(),
                replay_stages=(
                    ReplayStage(
                        source_kind="full_data",
                        operations=(operation,),
                    ),
                ),
            ),
            expected,
        )


def test_tool_replay_provenance_helpers_compose_parent_provenance() -> None:

    parent = selection(IselOperation(kwargs={"x": slice(0, 2)}))
    local = script(
        ScriptCodeOperation(
            label="Compute tool output",
            code="result = derived.mean()",
        ),
        start_label="Start from current tool input data",
    )

    composed = compose_full_provenance(parent, local)

    assert composed is not None
    assert composed.derivation_entries()[0].label == (
        "Start from selected parent ImageTool data"
    )
    assert composed.derivation_entries()[-1].label == "Compute tool output"
    code = composed.derivation_code()
    assert code is not None
    assert "_itool_replay_" not in code
    data = xr.DataArray([1.0, 2.0, 3.0], dims="x")
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["result"], data.isel(x=slice(0, 2)).mean())

    assert compose_display_provenance(parent, full_data()) == (
        to_replay_provenance_spec(parent)
    )


def test_tool_provenance_compose_full_uses_parent_active_name_for_live_local() -> None:
    data = _base_data()
    parent = script(
        ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = data + 1",
        ),
        start_label="Start from current tool input data",
        active_name="result",
    )
    local = full_data(AverageOperation(dims=("x",)))

    composed = compose_full_provenance(parent, local)

    assert composed is not None
    code = composed.derivation_code()
    assert code == "derived = (data + 1).qsel.mean('x')"
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, (data + 1).qsel.mean("x"))


def test_tool_provenance_compose_full_preserves_structured_live_steps() -> None:
    data = _base_data()
    parent = script(
        ScriptCodeOperation(
            label="Concatenate selected ImageTools",
            code="derived = data_0 + data_1",
        ),
        start_label="Run ImageTool manager action",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="ImageTool 0: scan"),
            ScriptInput(name="data_1", label="ImageTool 1: scan"),
        ),
    )
    local = full_data(
        SortByOperation(variables=("x",), ascending=False),
        AverageOperation(dims=("y",)),
    )

    composed = compose_full_provenance(parent, local)

    assert composed is not None
    assert [operation.op for operation in composed.operations] == [
        "script_code",
        "sortby",
        "average",
    ]
    rows = [
        row
        for row in composed.display_rows()
        if row.replay_ref is not None and row.replay_ref.kind == "operation"
    ]
    assert [row.edit_ref is not None for row in rows] == [True, True, True]
    derived = replay_script_provenance(
        composed,
        {
            "data_0": data.copy(deep=True),
            "data_1": data.copy(deep=True),
        },
    )
    xr.testing.assert_identical(
        derived,
        (data + data).sortby("x", ascending=False).qsel.mean("y"),
    )


def test_tool_provenance_script_context_binding_replays_current_output() -> None:
    data = _base_data()
    parent = script(
        ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = derived + 1",
        ),
        start_label="Start from current tool input data",
        seed_code="derived = data",
        active_name="result",
    )
    local = script(
        ScriptCodeOperation(
            label="Offset copied result",
            code="result = derived + 2",
        ),
        start_label="Start from current ImageTool data",
        active_name="result",
    )
    current = replay_script_provenance(parent, {"data": data})
    expected = replay_script_provenance(
        local,
        {"data": current, "derived": current},
    )

    composed = compose_full_provenance(
        parent,
        local,
        script_context_names=("data", "derived", "data"),
    )

    assert composed is not None
    assert [entry.label for entry in composed.derivation_entries()] == [
        "Start from current tool input data",
        "Compute intermediate result",
        "Offset copied result",
    ]
    assert [
        operation.derivation_entry().label for operation in composed.operations
    ] == [
        "Compute intermediate result",
        "Offset copied result",
    ]
    assert [
        binding.model_dump(mode="json") for binding in composed.script_context_bindings
    ] == [{"operation_index": 1, "names": ["data", "derived"]}]

    replayed = replay_script_provenance(composed, {"data": data})
    xr.testing.assert_identical(replayed, expected)
    code = typing.cast("str", composed.derivation_code())
    assert "Start from current ImageTool data" not in code
    assert "derived = derived" not in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    result = namespace["result"]
    assert isinstance(result, xr.DataArray)
    xr.testing.assert_identical(result, expected)


def test_tool_provenance_script_context_binding_validation() -> None:
    operation = ScriptCodeOperation(
        label="Offset copied result",
        code="result = derived + 2",
    )

    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
        steps=(
            ReplayStep(
                operation=operation,
                context_names=("data", "derived", "data"),
            ),
        ),
    )
    assert [
        binding.model_dump(mode="json") for binding in spec.script_context_bindings
    ] == [{"operation_index": 0, "names": ["data", "derived"]}]
    assert (
        ToolProvenanceSpec(
            kind="script",
            start_label="Run script",
            active_name="result",
            steps=(ReplayStep(operation=operation),),
        ).script_context_bindings
        == ()
    )

    invalid_payloads: tuple[tuple[typing.Any, type[BaseException], str], ...] = (
        ("data", TypeError, "names must be a sequence"),
        ([None], ValidationError, "must not be None"),
    )
    for context_names, exc_type, message in invalid_payloads:
        with pytest.raises(exc_type, match=message):
            ReplayStep(operation=operation, context_names=context_names)

    with pytest.raises(ValidationError, match="cannot define script context"):
        ToolProvenanceSpec(
            kind="file",
            start_label="Load source",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=_file_replay_source(),
            steps=(ReplayStep(operation=operation, context_names=("data",)),),
        )


def test_tool_provenance_script_context_bindings_follow_operation_edits() -> None:
    first = ScriptCodeOperation(
        label="Compute intermediate result",
        code="result = derived + 1",
    )
    second = ScriptCodeOperation(
        label="Offset copied result",
        code="result = derived + 2",
    )
    average = AverageOperation(dims=("x",))
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
        steps=(
            ReplayStep(operation=first),
            ReplayStep(
                operation=second,
                context_names=("data", "derived"),
            ),
            ReplayStep(operation=average, context_names=("data",)),
        ),
    )

    def binding_payloads(
        value: ToolProvenanceSpec,
    ) -> list[dict[str, typing.Any]]:
        return [
            binding.model_dump(mode="json") for binding in value.script_context_bindings
        ]

    expanded = spec._replace_operation_ref(
        _ProvenanceStepRef("operation", operation_index=0),
        (first, SqueezeOperation()),
    )
    assert binding_payloads(expanded) == [
        {"operation_index": 2, "names": ["data", "derived"]},
        {"operation_index": 3, "names": ["data"]},
    ]

    replaced_at_binding = spec._replace_operation_ref(
        _ProvenanceStepRef("operation", operation_index=1),
        (
            SqueezeOperation(),
            AssignAttrsOperation(attrs={"edited": True}),
        ),
    )
    assert binding_payloads(replaced_at_binding) == [
        {"operation_index": 1, "names": ["data", "derived"]},
        {"operation_index": 3, "names": ["data"]},
    ]

    deleted_last = spec._replace_operation_ref(
        _ProvenanceStepRef("operation", operation_index=2),
        (),
    )
    assert binding_payloads(deleted_last) == [
        {"operation_index": 1, "names": ["data", "derived"]},
    ]

    through_second = spec._prefix_through_ref(
        _ProvenanceStepRef("operation", operation_index=1)
    )
    assert binding_payloads(through_second) == [
        {"operation_index": 1, "names": ["data", "derived"]},
    ]
    before_second = spec._prefix_before_ref(
        _ProvenanceStepRef("operation", operation_index=1)
    )
    assert binding_payloads(before_second) == []
    start_only = spec._prefix_through_ref(_ProvenanceStepRef("start"))
    assert start_only.operations == ()
    assert start_only.script_context_bindings == ()


def test_legacy_operations_model_copy_preserves_replay_step_metadata() -> None:
    spec = parse_tool_provenance_spec(
        {
            "schema_version": 2,
            "kind": "script",
            "start_label": "Run saved script",
            "seed_code": "derived = data",
            "active_name": "result",
            "replay_stages": [
                {
                    "source_kind": "public_data",
                    "operations": [{"op": "average", "dims": ["x"]}],
                }
            ],
            "operations": [
                {
                    "op": "script_code",
                    "label": "Offset result",
                    "code": "result = parent_data + 1",
                }
            ],
            "script_context_bindings": [
                {"operation_index": 0, "names": ["parent_data"]}
            ],
        }
    )
    assert spec is not None
    average, offset = spec.operations
    original_average_metadata = spec.steps[0].model_dump(
        exclude={"operation"}, mode="python"
    )
    original_offset_metadata = spec.steps[1].model_dump(
        exclude={"operation"}, mode="python"
    )

    appended = spec.model_copy(
        update={"operations": (average, offset, SqueezeOperation())}
    )
    assert appended.steps[0].model_dump(exclude={"operation"}, mode="python") == (
        original_average_metadata
    )
    assert appended.steps[1].model_dump(exclude={"operation"}, mode="python") == (
        original_offset_metadata
    )
    assert appended.steps[2] == ReplayStep(operation=SqueezeOperation())

    reordered = appended.model_copy(
        update={"operations": (offset, SqueezeOperation(), average)}
    )
    assert reordered.steps[0].context_names == ("parent_data",)
    assert reordered.steps[1] == ReplayStep(operation=SqueezeOperation())
    assert reordered.steps[2].legacy_context == spec.steps[0].legacy_context

    replaced_and_appended = spec.model_copy(
        update={
            "operations": (
                IselOperation(kwargs={"x": 0}),
                offset,
                SqueezeOperation(),
            )
        }
    )
    assert replaced_and_appended.steps[0].legacy_context == (
        spec.steps[0].legacy_context
    )
    assert replaced_and_appended.steps[1].context_names == ("parent_data",)
    assert replaced_and_appended.steps[2] == ReplayStep(operation=SqueezeOperation())


def test_legacy_operations_model_copy_rejects_ambiguous_duplicate_metadata() -> None:
    operation = AverageOperation(dims=("x",))
    spec = script(
        start_label="Run script",
        seed_code="derived = data",
        active_name="derived",
        steps=(
            ReplayStep(operation=operation, input_policy="current"),
            ReplayStep(operation=operation, input_policy="restored"),
        ),
    )

    with pytest.raises(ValueError, match="duplicate operations"):
        spec.model_copy(
            update={
                "operations": (
                    AverageOperation(dims=("x",)),
                    AverageOperation(dims=("x",)),
                    SqueezeOperation(),
                )
            }
        )


def test_tool_provenance_script_flat_step_prefix_and_fallback_rows() -> None:
    stage = ReplayStage(
        source_kind="full_data",
        operations=(
            AverageOperation(dims=("x",)),
            IselOperation(kwargs={"missing": 0}),
        ),
    )
    spec = script(
        ScriptCodeOperation(label="Offset", code="result = result + 1"),
        start_label="Run script",
        seed_code="result = data",
        active_name="result",
        replay_stages=(stage,),
    )
    stage_ref = _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )

    through_stage = spec._prefix_through_ref(stage_ref)
    assert through_stage.operations == (AverageOperation(dims=("x",)),)
    assert through_stage.script_context_bindings == ()
    assert through_stage.active_name == "result"

    before_stage = spec._prefix_before_ref(
        _ProvenanceStepRef(
            "operation",
            operation_index=1,
        )
    )
    assert before_stage.operations == (AverageOperation(dims=("x",)),)
    assert before_stage.script_context_bindings == ()
    assert before_stage.active_name == "result"

    data = xr.DataArray(np.arange(3.0), dims=("x",), name="scan")
    entries = spec._code_fallback_entries(parent_data=data)
    labels = [entry.label for entry in entries]
    assert 'Average(dims=("x",))' in labels
    assert "isel(missing=0)" in labels
    assert "Offset" in labels
    rows = spec.display_rows(parent_data=data)
    assert [row.entry.label for row in rows[1:3]] == [
        'Average(dims=("x",))',
        "isel(missing=0)",
    ]
    assert rows[3].entry.label == "Offset"

    assert (
        ToolProvenanceSpec(
            kind="script",
            start_label="Run script",
            active_name="result",
        )._script_seed_output_name()
        is None
    )


def test_tool_provenance_script_prefix_tracks_active_output_name() -> None:
    spec = script(
        UniformInterpolationOperation(sizes={"x": 5}),
        GaussianFilterOperation(sigma={"x": 1.0}),
        ImageDerivativeOperation(
            method="diffn",
            kwargs={"coord": "x", "order": 2},
        ),
        TransposeOperation(),
        start_label="Start from current dtool input data",
        seed_code="derived = data",
        active_name="result",
    )

    expected_before = ("derived", "processed_data", "processed_data", "result")
    expected_through = ("processed_data", "processed_data", "result", "result")
    for operation_index in range(len(spec.operations)):
        ref = _ProvenanceStepRef(
            "operation",
            operation_index=operation_index,
        )
        before = spec._prefix_before_ref(ref)
        through = spec._prefix_through_ref(ref)

        assert before.active_name == expected_before[operation_index]
        assert through.active_name == expected_through[operation_index]
        assert script_provenance_replayable(
            before,
            external_input_names={"data"},
        )
        assert script_provenance_replayable(
            through,
            external_input_names={"data"},
        )


def test_tool_provenance_operation_group_replacement_preserves_script_context() -> None:
    grouped = stamp_operation_group(
        (
            ScriptCodeOperation(
                label="Offset copied result",
                code="result = derived + 2",
            ),
            AverageOperation(dims=("x",)),
        ),
        kind="demo",
        group_id="group-1",
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
        operations=(
            ScriptCodeOperation(
                label="Compute intermediate result",
                code="result = derived + 1",
            ),
            *grouped,
        ),
        script_context_bindings=[
            {"operation_index": 1, "names": ["data", "derived"]},
        ],
    )

    replaced = spec._replace_operation_range_ref(
        _ProvenanceStepRef("operation", operation_index=2),
        1,
        3,
        (SqueezeOperation(),),
    )

    assert [operation.op for operation in replaced.operations] == [
        "script_code",
        "squeeze",
    ]
    assert [
        binding.model_dump(mode="json") for binding in replaced.script_context_bindings
    ] == [{"operation_index": 1, "names": ["data", "derived"]}]


def test_tool_provenance_range_ref_helpers_cover_invalid_and_replay_refs() -> None:
    operations = stamp_operation_group(
        (
            AverageOperation(dims=("x",)),
            SqueezeOperation(),
        ),
        kind="demo",
    )
    spec = full_data(*operations)
    ref = _ProvenanceStepRef("operation", operation_index=0)

    with pytest.raises(ValueError, match="operation provenance row"):
        spec._replace_operation_range_ref(
            _ProvenanceStepRef("start"),
            0,
            1,
            (),
        )
    with pytest.raises(ValueError, match="non-empty operation range"):
        spec._replace_operation_range_ref(ref, 1, 1, ())

    deleted = spec._replace_operation_range_ref(ref, 0, 2, ())
    assert deleted.operations == ()

    file_spec = _file_provenance_spec().append_replay_stage(full_data())
    stage_spec = file_spec.append_replay_stage(full_data(*operations))
    stage_ref = _ProvenanceStepRef(
        "operation",
        operation_index=1,
    )
    replaced = stage_spec._replace_operation_range_ref(
        stage_ref,
        0,
        2,
        (ThinOperation(mode="per_dim", factors={"x": 2}),),
    )
    assert replaced.operations == (ThinOperation(mode="per_dim", factors={"x": 2}),)


def test_tool_provenance_reorder_blocks_replays_generated_code() -> None:
    data = xr.DataArray(
        [[1.0, 2.0], [3.0, 6.0], [10.0, 20.0]],
        dims=("x", "y"),
        name="scan",
    )
    spec = full_data(
        IselOperation(kwargs={"x": slice(0, 2)}),
        NormalizeOperation(dims=("x",), mode="area"),
    )
    sections = spec._reorder_sections()
    assert len(sections) == 1
    section = sections[0]
    reordered = spec._reorder_operation_blocks(
        sections,
        {section.ref: tuple(reversed(tuple(block.ref for block in section.blocks)))},
    )

    assert [operation.op for operation in reordered.operations] == [
        "normalize",
        "isel",
    ]
    original_result = spec.apply(data)
    reordered_result = reordered.apply(data)
    assert not np.allclose(original_result.values, reordered_result.values)

    code = reordered.display_code(parent_data=data)
    assert code is not None
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], reordered_result)


def test_tool_provenance_reorder_keeps_groups_and_script_bindings_atomic() -> None:
    grouped = stamp_operation_group(
        (
            ScriptCodeOperation(label="Copy result", code="result = derived + 2"),
            AverageOperation(dims=("x",)),
        ),
        kind="demo",
        group_id="reorder-group",
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
        operations=(
            ScriptCodeOperation(label="Initial result", code="result = derived + 1"),
            *grouped,
            AssignAttrsOperation(attrs={"reordered": True}),
        ),
        script_context_bindings=[
            {"operation_index": 1, "names": ["data", "derived"]},
        ],
    )
    sections = spec._reorder_sections()
    assert len(sections) == 1
    section = sections[0]
    assert [(block.ref.start, block.ref.stop) for block in section.blocks] == [
        (0, 1),
        (1, 3),
        (3, 4),
    ]

    grouped_block = section.blocks[1].ref
    reordered = spec._reorder_operation_blocks(
        sections,
        {
            section.ref: (
                grouped_block,
                section.blocks[2].ref,
                section.blocks[0].ref,
            )
        },
    )

    assert [operation.op for operation in reordered.operations] == [
        "script_code",
        "average",
        "assign_attrs",
        "script_code",
    ]
    assert operation_group_range(reordered.operations, 0, kind="demo") == (0, 2)
    assert [
        binding.model_dump(mode="json") for binding in reordered.script_context_bindings
    ] == [{"operation_index": 0, "names": ["data", "derived"]}]


def test_tool_provenance_reorder_sections_respect_hidden_and_fixed_boundaries() -> None:
    hidden_boundary = full_data(
        IselOperation(kwargs={"x": slice(0, 2)}),
        ScriptCodeOperation(
            label="Hidden boundary",
            code="derived = derived.copy(deep=False)",
            visible=False,
        ),
        NormalizeOperation(dims=("x",)),
    )
    assert hidden_boundary._reorder_sections() == ()

    fixed_middle = full_data(
        IselOperation(kwargs={"x": slice(0, 2)}),
        NormalizeOperation(dims=("x",)),
        AssignAttrsOperation(attrs={"done": True}),
    )
    assert (
        fixed_middle._reorder_sections(
            fixed_refs=(_ProvenanceStepRef("operation", operation_index=1),)
        )
        == ()
    )

    staged = (
        _file_provenance_spec()
        .append_replay_stage(
            full_data(
                IselOperation(kwargs={"x": slice(0, 2)}),
                NormalizeOperation(dims=("x",)),
            )
        )
        .append_replay_stage(
            full_data(
                AssignAttrsOperation(attrs={"first": True}),
                AssignAttrsOperation(attrs={"second": True}),
            )
        )
    )
    sections = staged._reorder_sections()
    assert len(sections) == 1
    section = sections[0]
    assert [(block.ref.start, block.ref.stop) for block in section.blocks] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    orders = {section.ref: tuple(reversed([block.ref for block in section.blocks]))}
    reordered = staged._reorder_operation_blocks(
        sections,
        orders,
    )
    assert reordered.operations == tuple(reversed(staged.operations))
    assert [step.input_policy for step in reordered.steps] == [
        "current",
        "current",
        "current",
        "current",
    ]

    invalid_orders = dict(orders)
    invalid_orders[section.ref] = (
        section.blocks[0].ref,
        section.blocks[0].ref,
    )
    with pytest.raises(ValueError, match="every block exactly once"):
        staged._reorder_operation_blocks(
            sections,
            invalid_orders,
        )


def test_tool_provenance_reorder_planning_is_independent_of_display_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = full_data(
        IselOperation(kwargs={"x": slice(0, 2)}),
        NormalizeOperation(dims=("x",)),
    )

    def fail_display_projection(*args, **kwargs):
        raise AssertionError(
            "structural reorder planning must not use display projection"
        )

    monkeypatch.setattr(ToolProvenanceSpec, "display_rows", fail_display_projection)
    monkeypatch.setattr(
        ToolProvenanceSpec,
        "_streamlined_operation_refs",
        fail_display_projection,
    )
    sections = spec._reorder_sections()
    assert len(sections) == 1
    assert [block.ref.start for block in sections[0].blocks] == [0, 1]


def test_tool_provenance_reorder_planning_preserves_structural_boundaries() -> None:
    malformed_group = AssignAttrsOperation(attrs={"malformed": True}).model_copy(
        update={
            "group": OperationGroupMarker(
                kind="test",
                id="incomplete",
                index=0,
                size=2,
            )
        }
    )
    spec = _file_provenance_spec().model_copy(
        update={
            "operations": (
                malformed_group,
                AssignAttrsOperation(attrs={"root_a": True}),
                AssignAttrsOperation(attrs={"root_b": True}),
            )
        }
    )
    hidden_boundary = ScriptCodeOperation(
        label="Internal boundary",
        code="derived = derived.copy(deep=False)",
        visible=False,
    )
    for stage in (
        full_data(RenameOperation(name="initial_boundary")),
        full_data(AssignAttrsOperation(attrs={"stage_0": True})),
        full_data(AssignAttrsOperation(attrs={"stage_1": True})),
        full_data(
            RenameOperation(name="stage_boundary"),
            AssignAttrsOperation(attrs={"stage_2a": True}),
            AssignAttrsOperation(attrs={"stage_2b": True}),
        ),
        full_data(AssignAttrsOperation(attrs={"stage_3": True})),
        full_data(AssignAttrsOperation(attrs={"stage_4": True})),
    ):
        spec = spec.append_replay_stage(stage)

    sections = spec._reorder_sections()
    assert sections

    hidden_operations = (
        hidden_boundary,
        ScriptCodeOperation(label="Empty selection", code="derived = derived.isel()"),
        IselOperation(),
        RenameOperation(name="renamed"),
    )
    assert all(
        spec._operation_reorder_entry(operation) is None
        for operation in hidden_operations
    )


def test_tool_provenance_reorder_rejects_malformed_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = full_data(
        AssignAttrsOperation(attrs={"a": True}),
        AssignAttrsOperation(attrs={"b": True}),
        AssignAttrsOperation(attrs={"c": True}),
    )
    section = spec._reorder_sections()[0]
    identity = {section.ref: tuple(block.ref for block in section.blocks)}

    with pytest.raises(ValueError, match="sections must be unique"):
        spec._reorder_operation_blocks((section, section), identity)
    with pytest.raises(ValueError, match="does not match its sections"):
        spec._reorder_operation_blocks((section,), {})

    out_of_range_ref = _ProvenanceReorderSectionRef(0, 4)
    out_of_range = _ProvenanceReorderSection(
        out_of_range_ref,
        "Invalid range",
        section.blocks,
    )
    with pytest.raises(ValueError, match="section is out of range"):
        spec._reorder_operation_blocks(
            (out_of_range,),
            {out_of_range_ref: identity[section.ref]},
        )

    overlapping = (
        _ProvenanceReorderSection(
            _ProvenanceReorderSectionRef(0, 2),
            "First overlap",
            section.blocks[:2],
        ),
        _ProvenanceReorderSection(
            _ProvenanceReorderSectionRef(1, 3),
            "Second overlap",
            section.blocks[1:],
        ),
    )
    with pytest.raises(ValueError, match="sections must not overlap"):
        spec._reorder_operation_blocks(
            overlapping,
            {
                item.ref: tuple(block.ref for block in item.blocks)
                for item in overlapping
            },
        )

    unknown_block = _ProvenanceReorderBlock(
        _ProvenanceReorderBlockRef(0, 2),
        tuple(entry for block in section.blocks[:2] for entry in block.entries),
    )
    unknown_section = _ProvenanceReorderSection(
        _ProvenanceReorderSectionRef(0, 2),
        "Unknown block",
        (unknown_block,),
    )
    with pytest.raises(ValueError, match="movable displayed steps"):
        spec._reorder_operation_blocks(
            (unknown_section,),
            {unknown_section.ref: (unknown_block.ref,)},
        )

    reversed_section = _ProvenanceReorderSection(
        section.ref,
        section.label,
        tuple(reversed(section.blocks)),
    )
    with pytest.raises(ValueError, match="partition their section"):
        spec._reorder_operation_blocks(
            (reversed_section,),
            {
                reversed_section.ref: tuple(
                    block.ref for block in reversed_section.blocks
                )
            },
        )

    incomplete_section = _ProvenanceReorderSection(
        section.ref,
        section.label,
        section.blocks[:2],
    )
    with pytest.raises(ValueError, match="partition their section"):
        spec._reorder_operation_blocks(
            (incomplete_section,),
            {
                incomplete_section.ref: tuple(
                    block.ref for block in incomplete_section.blocks
                )
            },
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            ToolProvenanceSpec,
            "_reorderable_operation_blocks",
            lambda self, **_kwargs: (unknown_block,),
        )
        patch.setattr(
            ToolProvenanceSpec,
            "_reorder_sections",
            lambda self, **_kwargs: (),
        )
        with pytest.raises(ValueError, match="Ungrouped provenance blocks"):
            spec._reorder_operation_blocks(
                (unknown_section,),
                {unknown_section.ref: (unknown_block.ref,)},
            )

    grouped_spec = full_data(
        *stamp_operation_group(
            spec.operations[:2],
            kind="test",
            group_id="validation-group",
        )
    )
    partial_group_block = _ProvenanceReorderBlock(
        _ProvenanceReorderBlockRef(0, 1),
        (grouped_spec.operations[0].derivation_entry(),),
    )
    partial_group_section = _ProvenanceReorderSection(
        _ProvenanceReorderSectionRef(0, 1),
        "Partial group",
        (partial_group_block,),
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            ToolProvenanceSpec,
            "_reorderable_operation_blocks",
            lambda self, **_kwargs: (partial_group_block,),
        )
        patch.setattr(
            ToolProvenanceSpec,
            "_reorder_sections",
            lambda self, **_kwargs: (),
        )
        with pytest.raises(ValueError, match="complete group"):
            grouped_spec._reorder_operation_blocks(
                (partial_group_section,),
                {partial_group_section.ref: (partial_group_block.ref,)},
            )


def test_tool_provenance_script_context_names_are_validation_only() -> None:
    parent = script(
        ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = derived + 1",
        ),
        start_label="Start from current tool input data",
        seed_code="derived = data",
        active_name="result",
    )
    local_script = script(
        ScriptCodeOperation(
            label="Offset copied result",
            code="result = derived + 2",
        ),
        start_label="Run pasted script",
        active_name="result",
    )
    local_structured = full_data(AverageOperation(dims=("x",)))

    with pytest.raises(ValueError, match="script context names"):
        compose_full_provenance(
            parent,
            local_script,
            script_context_names=(typing.cast("str", None),),
        )
    composed = compose_full_provenance(
        parent,
        local_structured,
        script_context_names=("data", "derived"),
    )

    assert composed is not None
    assert composed.script_context_bindings == ()


def test_file_load_source_replay_call_round_trips() -> None:
    xarray_source = FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text='engine="h5netcdf"',
        replay_call=FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selection=FileDataSelection(kind="dataarray"),
            cast_float64=True,
        ),
        load_code='import xarray\n\ndata = xarray.load_dataarray("/tmp/scan.h5")',
    )
    parsed_xarray = FileLoadSource.model_validate(xarray_source.model_dump(mode="json"))
    assert parsed_xarray == xarray_source
    assert parsed_xarray.replay_call.kind == "callable"
    assert parsed_xarray.replay_call.target == "xarray.load_dataarray"
    assert parsed_xarray.replay_call.kwargs == {"engine": "h5netcdf"}
    assert parsed_xarray.replay_call.selection == FileDataSelection(kind="dataarray")
    assert parsed_xarray.replay_call.cast_float64 is True

    legacy_call = FileReplayCall.model_validate(
        {
            "kind": "callable",
            "target": "xarray.load_dataarray",
            "selected_index": 2,
        }
    )
    assert legacy_call.selection == FileDataSelection(kind="parsed_index", value=2)
    assert "selected_index" not in legacy_call.model_dump(mode="json")

    legacy_tree_call = FileReplayCall.model_validate(
        {
            "kind": "callable",
            "target": "xarray.load_datatree",
            "selection": {
                "kind": "datatree_path",
                "value": "/branch/image",
            },
        }
    )
    assert legacy_tree_call.selection == FileDataSelection(
        kind="datatree_variable",
        value=("/branch", "image"),
    )
    assert legacy_tree_call.model_dump(mode="json")["selection"] == {
        "kind": "datatree_variable",
        "value": ["/branch", "image"],
    }

    erlab_source = FileLoadSource(
        path="data_002.h5",
        loader_label="Loader",
        loader_text="example",
        kwargs_text="(none)",
        replay_call=FileReplayCall(
            kind="erlab_loader",
            target="example",
            kwargs={},
            selection=FileDataSelection(kind="dataarray"),
        ),
        load_code="erlab.io.set_loader('example')\ndata = erlab.io.load(2)",
    )
    parsed_erlab = FileLoadSource.model_validate(erlab_source.model_dump(mode="json"))
    assert parsed_erlab == erlab_source
    assert parsed_erlab.replay_call.kind == "erlab_loader"
    assert parsed_erlab.replay_call.target == "example"


def test_provenance_file_source_capabilities_cover_script_backed_files(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.h5"
    xr.DataArray(np.arange(3.0), dims=("x",)).to_netcdf(path, engine="h5netcdf")
    replay_call = FileReplayCall(
        kind="callable",
        target="xarray.load_dataarray",
        kwargs={"engine": "h5netcdf"},
        selected_index=0,
    )
    load_source = _file_replay_source(path, replay_call=replay_call)
    seed_code = (
        "import xarray\n\n"
        f"derived = xarray.load_dataarray({str(path)!r}, engine='h5netcdf')"
    )
    file_spec = file_load(
        start_label="Load data",
        seed_code=seed_code,
        file_load_source=load_source,
    ).append_replay_stage(full_data(AverageOperation(dims=("x",))))
    script_spec = script(
        start_label="Load data",
        seed_code=seed_code,
        active_name="derived",
        file_load_source=load_source,
        steps=file_spec.steps,
    )

    for spec in (file_spec, script_spec):
        assert has_file_load_source(spec)
        assert file_load_source_status(spec) == "loadable"
        assert can_reload_without_trust(spec)
        operation_refs = tuple(iter_operation_refs(spec))
        assert [ref.operation_index for ref, _op in operation_refs] == [0]
        assert isinstance(operation_refs[0][1], AverageOperation)

    missing_spec = script_spec.model_copy(
        update={
            "file_load_source": load_source.model_copy(
                update={"path": str(tmp_path / "missing.h5")}
            )
        }
    )
    assert file_load_source_status(missing_spec) == "missing-file"
    assert not can_reload_without_trust(missing_spec)

    no_replay_call_spec = script_spec.model_copy(
        update={
            "file_load_source": load_source.model_copy(update={"replay_call": None})
        }
    )
    assert file_load_source_status(no_replay_call_spec) == "no-replay-call"
    assert not can_reload_without_trust(no_replay_call_spec)

    missing_loader = "definitely-missing-erlab-loader"
    missing_loader_spec = script_spec.model_copy(
        update={
            "file_load_source": load_source.model_copy(
                update={
                    "replay_call": replay_call.model_copy(
                        update={"kind": "erlab_loader", "target": missing_loader}
                    )
                }
            )
        }
    )
    assert file_load_source_status(missing_loader_spec) == "missing-loader"
    assert not can_reload_without_trust(missing_loader_spec)

    missing_callable_spec = script_spec.model_copy(
        update={
            "file_load_source": load_source.model_copy(
                update={
                    "replay_call": replay_call.model_copy(
                        update={
                            "kind": "callable",
                            "target": "definitely_missing_erlab_callable.load",
                        }
                    )
                }
            )
        }
    )
    assert file_load_source_status(missing_callable_spec) == "missing-loader"
    assert not can_reload_without_trust(missing_callable_spec)

    plain_script = script(
        ScriptCodeOperation(
            label="Make data",
            code="derived = xr.DataArray([1.0])",
        ),
        start_label="Run script",
        active_name="derived",
    )
    assert not has_file_load_source(plain_script)
    assert file_load_source_status(plain_script) == "no-file-load-source"
    assert can_reload_without_trust(plain_script)


def test_provenance_replay_stage_source_view_and_empty_refs() -> None:
    data = _base_data()

    selection_view = _SourceViewOperation(source_kind="selection")
    public_view = _SourceViewOperation(source_kind="public_data")
    full_view = _SourceViewOperation(source_kind="full_data")

    xr.testing.assert_identical(
        selection_view.apply(data),
        erlab.utils.array._restore_nonuniform_dims(data),
    )
    assert (
        selection_view.derivation_label() == "Start from selected parent ImageTool data"
    )
    assert (
        public_view.derivation_label()
        == "Start from current parent ImageTool public data"
    )
    assert full_view.derivation_label() == "Start from current parent ImageTool data"
    assert full_view.expression_code("data") == "data.copy(deep=False)"
    with pytest.raises(NotImplementedError):
        selection_view.expression_code("data")
    assert tuple(iter_operation_refs(None)) == ()


def test_file_provenance_validation_rejects_invalid_payloads() -> None:
    replay_stage = ReplayStage(source_kind="full_data")
    file_source = _file_replay_source()

    with pytest.raises(ValidationError, match="file selection index"):
        FileReplayCall(
            kind="callable", target="xarray.load_dataarray", selected_index=-1
        )
    with pytest.raises(ValidationError, match="target"):
        FileReplayCall(kind="callable", target="", selected_index=0)

    bad_kwargs_call = FileReplayCall.model_construct(
        kind="callable",
        target="xarray.load_dataarray",
        kwargs={1: "bad"},
        selection=FileDataSelection(kind="dataarray"),
    )
    with pytest.raises(TypeError, match="string keys"):
        bad_kwargs_call._validate_replay_call()

    assert ReplayStage(source_kind="full_data", operations=None).operations == ()
    with pytest.raises(TypeError, match="replay stage operations"):
        ReplayStage(source_kind="full_data", operations=1)
    with pytest.raises(TypeError, match="script-only operations"):
        ReplayStage(
            source_kind="full_data",
            operations=[
                ScriptCodeOperation(label="Generated", code="derived = derived")
            ],
        )
    with pytest.raises(TypeError, match="source must not be None"):
        ReplayStage.from_source_spec(typing.cast("typing.Any", None))

    assert ToolProvenanceSpec(kind="full_data", replay_stages=None).steps == ()
    with pytest.raises(TypeError, match="Serialized replay stages"):
        ToolProvenanceSpec(kind="full_data", replay_stages=1)
    migrated = ToolProvenanceSpec(
        kind="script",
        start_label="Start",
        active_name="derived",
        replay_stages=[replay_stage],
    )
    assert migrated.steps == ()

    with pytest.raises(ValidationError, match="must define `start_label`"):
        ToolProvenanceSpec(
            kind="file",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `seed_code`"):
        ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            active_name="derived",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `active_name`"):
        ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `file_load_source`"):
        ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
        )
    with pytest.raises(ValidationError, match="must define `replay_call`"):
        ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source.model_copy(update={"replay_call": None}),
        )
    with pytest.raises(ValidationError, match="cannot define operations"):
        ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source,
            operations=[AverageOperation(dims=("x",))],
        )
    with pytest.raises(TypeError, match="Replay steps can only"):
        full_data().append_replay_stage(full_data())


def test_file_provenance_display_entries_keep_steps_after_stage_failure() -> None:
    spec = (
        _file_provenance_spec()
        .append_replay_stage(full_data(SelOperation(kwargs={"missing": 0})))
        .append_replay_stage(full_data(SqueezeOperation()))
    )

    assert [entry.label for entry in spec.derivation_entries()] == [
        "Load data from file 'scan.h5'",
        "sel(missing=0)",
        "squeeze()",
    ]
    entries = spec.display_entries(parent_data=_base_data())
    assert [entry.label for entry in entries] == [
        "Load data from file 'scan.h5'",
        "sel(missing=0)",
        "squeeze()",
    ]
    code = spec.display_code(parent_data=_base_data())
    assert code is not None
    assert ".sel(missing=0)" in code
    assert ".squeeze()" in code


def test_tool_provenance_display_rows_expose_edit_and_replay_refs() -> None:
    file_spec = _file_provenance_spec().append_replay_stage(
        full_data(QSelAggregationOperation(dims=("x",)))
    )
    file_rows = file_spec.display_rows(parent_data=_base_data())

    assert file_rows[0].edit_ref == _ProvenanceStepRef("file_load")
    assert file_rows[0].replay_ref == _ProvenanceStepRef("file_load")
    assert file_rows[1].edit_ref == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    assert file_rows[1].replay_ref == file_rows[1].edit_ref

    live_spec = full_data(
        CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        )
    )
    live_rows = live_spec.display_rows(parent_data=_base_data(), scope="source")

    assert live_rows[0].scope == "source"
    assert live_rows[0].edit_ref is None
    assert live_rows[0].replay_ref == _ProvenanceStepRef("start")
    assert live_rows[1].scope == "source"
    assert live_rows[1].edit_ref == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )

    script_spec = script(
        ScriptCodeOperation(label="Run code", code="derived = data"),
        QSelAggregationOperation(dims=("x",)),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="source", label="Input"),),
    )
    script_rows = script_spec.display_rows()

    assert script_rows[1].edit_ref is None
    assert script_rows[1].replay_ref == _ProvenanceStepRef(
        "script_input",
        script_input_index=0,
    )
    assert script_rows[2].edit_ref == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    assert script_rows[2].replay_ref == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    assert script_rows[3].edit_ref == _ProvenanceStepRef(
        "operation",
        operation_index=1,
    )


def test_file_provenance_compose_fallbacks_and_replay_aliases() -> None:
    file_spec = _file_provenance_spec()

    assert replay_input_name(None) is None
    assert script(start_label="Start", active_name="derived").derivation_code() is None
    assert (
        script(
            start_label="Start",
            seed_code="derived = data",
            active_name="derived",
        ).display_code()
        is None
    )
    assert compose_full_provenance(None, None) is None
    local_replay = compose_full_provenance(
        None, full_data(AverageOperation(dims=("x",)))
    )
    assert local_replay is not None
    assert local_replay.kind == "script"

    assert compose_full_provenance(file_spec, full_data()) == file_spec
    assert _as_script_replay_spec(full_data()).kind == "script"

    script_local = script(
        ScriptCodeOperation(label="Offset", code="result = derived + 1"),
        start_label="Run generated code",
        seed_code="derived = data",
        active_name="result",
    )
    file_with_script = compose_full_provenance(file_spec, script_local)
    assert file_with_script is not None
    assert file_with_script.kind == "script"
    assert file_with_script.file_load_source == file_spec.file_load_source
    assert file_with_script.derivation_code() == (
        "import xarray\nresult = xarray.load_dataarray('scan.h5') + 1"
    )

    file_with_stage = file_spec.append_replay_stage(
        full_data(AverageOperation(dims=("x",)))
    )
    staged_with_script = compose_full_provenance(
        file_with_stage,
        script_local,
    )
    assert staged_with_script is not None
    assert staged_with_script.kind == "script"
    assert staged_with_script.file_load_source == file_spec.file_load_source
    assert any(
        isinstance(step.operation, AverageOperation)
        for step in staged_with_script.steps
    )
    assert all(
        not isinstance(step.operation, ScriptCodeOperation)
        for step in staged_with_script.steps[:1]
    )
    staged_rows = staged_with_script.display_rows()
    assert staged_rows[0].edit_ref == _ProvenanceStepRef("file_load")
    assert staged_rows[1].edit_ref == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )

    script_parent = script(
        ScriptCodeOperation(
            label="Crop",
            code="derived = derived.isel(x=0)",
        ),
        start_label="Run parent script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    staged_local = script(
        ScriptCodeOperation(label="Offset", code="result = derived + 1"),
        start_label="Run local script",
        active_name="result",
        replay_stages=(
            ReplayStage.from_source_spec(selection(AverageOperation(dims=("x",)))),
        ),
    )
    script_with_ordered_stage = compose_full_provenance(
        script_parent,
        staged_local,
    )
    assert script_with_ordered_stage is not None
    assert script_with_ordered_stage.kind == "script"
    assert [operation.op for operation in script_with_ordered_stage.operations] == [
        "script_code",
        "average",
        "script_code",
    ]
    ordered_rows = script_with_ordered_stage.display_rows()
    assert all(row.entry.label != staged_local.start_label for row in ordered_rows)
    assert any(
        row.edit_ref
        == _ProvenanceStepRef(
            "operation",
            operation_index=1,
        )
        for row in ordered_rows
    )

    alias_local = script(
        AverageOperation(dims=("x",)),
        start_label="Start from current ktool input data",
        seed_code="scan_kconv = derived",
        active_name="scan_kconv",
    )
    alias_composed = compose_full_provenance(file_spec, alias_local)
    assert alias_composed is not None
    assert alias_composed.kind == "script"
    assert isinstance(alias_composed.operations[0], ScriptCodeOperation)
    assert alias_composed.operations[0].visible is False
    assert isinstance(alias_composed.operations[1], AverageOperation)
    assert all(
        row.entry.label != "Start from current ktool input data"
        for row in alias_composed.display_rows()
    )
    assert any(
        isinstance(row.edit_ref, _ProvenanceStepRef)
        and row.edit_ref.operation_index == 1
        for row in alias_composed.display_rows()
    )

    watched_parent = script(
        start_label="Start from watched variable 'watched_data'",
        seed_code="derived = watched_data",
        active_name="derived",
    )
    default_seed_local = script(
        ScriptCodeOperation(label="Mean", code="result = derived.mean()"),
        start_label="Use current parent output",
        seed_code="derived = data",
        active_name="result",
    )
    watched_composed = compose_full_provenance(watched_parent, default_seed_local)
    assert watched_composed is not None
    assert watched_composed.derivation_code() == "result = watched_data.mean()"
    assert watched_composed.display_code() == "result = watched_data.mean()"

    result_parent = script(
        ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = data + 1",
        ),
        start_label="Start",
        active_name="result",
    )
    no_seed_local = script(
        ScriptCodeOperation(label="Mean", code="result = derived.mean()"),
        start_label="Use parent result",
        active_name="derived",
    )
    result_composed = compose_full_provenance(result_parent, no_seed_local)
    assert result_composed is not None
    assert result_composed.derivation_code() == "result = (data + 1).mean()"
    assert all(
        row.entry.label != "Use parent result" for row in result_composed.display_rows()
    )

    promoted = mark_promoted_1d_source(_base_data().copy(deep=False))
    assert (
        compose_display_provenance(
            watched_parent,
            selection(IselOperation(), SortCoordOrderOperation()),
            parent_data=promoted,
        )
        is not None
    )
    assert compose_display_provenance(
        watched_parent, None
    ) == to_replay_provenance_spec(watched_parent)


@pytest.mark.parametrize("parent_kind", ["script", "file"])
@pytest.mark.parametrize("source_builder", [public_data, selection])
def test_compose_operation_free_restored_source_preserves_nonuniform_dimensions(
    parent_kind: str,
    source_builder: collections.abc.Callable[[], ToolProvenanceSpec],
    tmp_path: pathlib.Path,
) -> None:
    public = xr.DataArray(
        np.arange(20.0).reshape(5, 4),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4)},
        name="scan",
    )
    internal = erlab.utils.array._make_dims_uniform(public)

    if parent_kind == "file":
        path = tmp_path / "nonuniform.nc"
        internal.to_netcdf(path)
        parent = file_load(
            start_label="Load nonuniform data",
            seed_code=(
                f"import xarray\n\nderived = xarray.load_dataarray({str(path)!r})"
            ),
            file_load_source=_file_replay_source(path),
        )
        replay_inputs: dict[str, xr.DataArray] = {}
    else:
        parent = script(
            start_label="Start from data",
            seed_code="derived = data",
            active_name="derived",
        )
        replay_inputs = {"data": internal}

    composed = compose_full_provenance(parent, source_builder())

    assert composed is not None
    assert len(composed.steps) == 1
    assert isinstance(composed.steps[0].operation, _SourceViewOperation)
    if composed.kind == "file":
        replayed = replay_file_provenance(composed)
    else:
        replayed = replay_script_provenance(composed, replay_inputs)
    xr.testing.assert_identical(replayed, public)

    code = composed.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, replay_inputs)
    xr.testing.assert_identical(
        namespace[typing.cast("str", composed.active_name)], public
    )


def test_script_provenance_supports_named_console_inputs() -> None:
    left = script(
        ScriptCodeOperation(
            label="Offset left input",
            code="data_0 = data_0 + 1.0",
        ),
        start_label="Load left",
        seed_code="data_0 = xr.DataArray([1.0, 2.0], dims=['x'])",
        active_name="data_0",
    )
    right = script(
        start_label="Load right",
        seed_code="data_1 = xr.DataArray([0.5, 1.5], dims=['x'])",
        active_name="data_1",
    )
    spec = script(
        ScriptCodeOperation(
            label="Subtract console inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="ImageTool 0",
                node_uid="left",
                provenance_spec=left,
            ),
            ScriptInput(
                name="data_1",
                label="ImageTool 1",
                node_uid="right",
                provenance_spec=right,
            ),
        ),
    )

    reparsed = parse_tool_provenance_spec(spec.model_dump(mode="json"))

    assert reparsed == spec
    assert [entry.label for entry in spec.display_entries()] == [
        "Run ImageTool manager console code",
        "Use data_0 from ImageTool 0",
        "Use data_1 from ImageTool 1",
        "Subtract console inputs",
    ]
    rows = spec.display_rows()
    assert rows[1].children[0].entry.label == "Load left"
    assert rows[1].children[0].replay_ref == _ProvenanceStepRef("start")
    assert rows[1].children[0].script_input_path == (0,)
    assert rows[1].children[1].entry.label == "Offset left input"
    assert rows[1].children[1].edit_ref == _ProvenanceStepRef(
        "operation", operation_index=0
    )
    assert rows[1].children[1].replay_ref == _ProvenanceStepRef(
        "operation", operation_index=0
    )
    assert rows[1].children[1].script_input_path == (0,)
    assert rows[2].children[0].entry.label == "Load right"
    assert rows[2].children[0].script_input_path == (1,)
    assert rows[3].edit_ref == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    assert rows[3].replay_ref == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code, {})
    xr.testing.assert_identical(
        namespace["derived"],
        xr.DataArray([1.5, 1.5], dims=["x"]),
    )


def test_script_input_code_reuses_shared_file_replay_prefix(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "polarization.nc"
    source = xr.DataArray(
        np.arange(12.0).reshape(2, 2, 3),
        dims=("pol", "energy", "k"),
        coords={"pol": ["LH", "LV"], "energy": [0.0, 1.0], "k": [0, 1, 2]},
    )
    source.to_netcdf(path)
    file_spec = file_load(
        start_label="Load both polarizations",
        seed_code=f"import xarray\n\nderived = xarray.load_dataarray({str(path)!r})",
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
        ),
    )
    shared_stage = full_data(AverageOperation(dims=("k",)))
    left_stage = selection(
        SelOperation(kwargs={"pol": "LH"}),
        SqueezeOperation(),
    )
    right_stage = selection(
        SelOperation(kwargs={"pol": "LV"}),
        SqueezeOperation(),
    )
    left_spec = compose_full_provenance(
        compose_full_provenance(file_spec, shared_stage),
        left_stage,
    )
    right_spec = compose_full_provenance(
        compose_full_provenance(file_spec, shared_stage),
        right_stage,
    )
    assert left_spec is not None
    assert right_spec is not None
    spec = script(
        ScriptCodeOperation(
            label="Subtract polarizations",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="ImageTool 0: LH",
                provenance_spec=left_spec,
            ),
            ScriptInput(
                name="data_1",
                label="ImageTool 1: LV",
                provenance_spec=right_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())

    assert code.count("xarray.load_dataarray") == 1
    assert code.count(".qsel.mean") == 1
    assert sum(line.startswith("data_0 =") for line in code.splitlines()) == 1
    assert sum(line.startswith("data_1 =") for line in code.splitlines()) == 1
    assert code.count(".copy(deep=True)") == 2
    assert "restore_nonuniform_dims" not in code
    namespace = _exec_generated_code(code, {})
    expected = left_stage.apply(shared_stage.apply(source)) - right_stage.apply(
        shared_stage.apply(source)
    )
    xr.testing.assert_identical(namespace["derived"], expected)


def test_script_input_code_keeps_distinct_structured_replay_nodes() -> None:
    first = file_load(
        start_label="Load first",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(
            "scan.h5",
            replay_call=FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )
    second = file_load(
        start_label="Load second",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(
            "scan.h5",
            replay_call=FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=1,
            ),
        ),
    )
    spec = script(
        ScriptCodeOperation(
            label="Subtract inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="ImageTool 0", provenance_spec=first),
            ScriptInput(
                name="data_1",
                label="ImageTool 1",
                provenance_spec=second,
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count("xarray.load_dataarray") == 2


def test_script_input_dependency_refs_recurse_and_rebase() -> None:
    left_snapshot_id = "left-snapshot"
    right_snapshot_id = "right-snapshot"
    extra_snapshot_id = "extra-snapshot"
    nested = script(
        ScriptCodeOperation(
            label="Subtract console inputs",
            code="diff = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="diff",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="ImageTool 0",
                node_uid="old-left",
                node_snapshot_token=left_snapshot_id,
                data_role="source",
            ),
            ScriptInput(
                name="data_1",
                label="ImageTool 1",
                node_uid="old-right",
                node_snapshot_token=right_snapshot_id,
            ),
        ),
    )
    spec = script(
        ScriptCodeOperation(
            label="Add nested input",
            code="derived = diff + data_2",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="diff",
                label="console variable 'diff'",
                node_uid="",
                provenance_spec=nested,
            ),
            ScriptInput(
                name="data_2",
                label="ImageTool 2",
                node_uid="old-extra",
                node_snapshot_token=extra_snapshot_id,
            ),
        ),
    )

    refs = script_input_dependency_refs(spec)
    assert [
        (
            ref.name,
            ref.label,
            ref.node_uid,
            ref.node_snapshot_token,
            ref.data_role,
        )
        for ref in refs
    ] == [
        ("data_0", "ImageTool 0", "old-left", left_snapshot_id, "source"),
        ("data_1", "ImageTool 1", "old-right", right_snapshot_id, "displayed"),
        ("data_2", "ImageTool 2", "old-extra", extra_snapshot_id, "displayed"),
    ]

    rebased = rebase_script_input_node_uids(
        spec,
        {
            "old-left": "new-left",
            "old-right": "new-right",
            "old-extra": "new-extra",
        },
    )

    assert [source.name for source in rebased.script_inputs] == ["diff", "data_2"]
    assert rebased.script_inputs[1].node_uid == "new-extra"
    assert typing.cast("str", rebased.operations[-1].derivation_entry().code) == (
        "derived = diff + data_2"
    )
    assert [
        (
            ref.name,
            ref.label,
            ref.node_uid,
            ref.node_snapshot_token,
            ref.data_role,
        )
        for ref in script_input_dependency_refs(rebased)
    ] == [
        ("data_0", "ImageTool 0", "new-left", left_snapshot_id, "source"),
        ("data_1", "ImageTool 1", "new-right", right_snapshot_id, "displayed"),
        ("data_2", "ImageTool 2", "new-extra", extra_snapshot_id, "displayed"),
    ]
    assert script_input_dependency_refs(None) == ()
    assert rebase_script_input_node_uids(spec, {}) is spec
    with pytest.raises(TypeError, match="Expected provenance spec"):
        rebase_script_input_node_uids(None, {})


def test_low_level_provenance_replay_helper_branches() -> None:

    assert not _is_whole_array_rename_entry(DerivationEntry("rename", "derived ="))
    assert not _is_whole_array_rename_entry(
        DerivationEntry("rename", "derived = derived.rename('a', 'b')")
    )
    assert not _is_whole_array_rename_entry(
        DerivationEntry("rename", "derived = derived.rename(mapping={'x': 'y'})")
    )
    assert _is_whole_array_rename_entry(
        DerivationEntry(
            "rename",
            "derived = derived.rename(new_name_or_name_dict=None)",
        )
    )

    assert _provenance_value_code({"left": (1,)}) == "{'left': (1,)}"
    with pytest.raises(TypeError, match="Cannot generate replay code"):
        _provenance_value_code(object())
    with pytest.raises(TypeError, match="hashable fields"):
        _normalize_provenance_hashable(object())
    assert _encode_provenance_hashable(("x", 1)) == {_TUPLE_MARKER: ["x", 1]}
    with pytest.raises(ValueError, match="Expected 2 items"):
        _ensure_float_tuple([1], expected_len=2)
    for value in (1, "abc"):
        with pytest.raises(TypeError, match="array-like sequence"):
            _coerce_float_sequence(value)
    assert _format_selection_step("sel", {}) == "derived = derived.sel()"
    assert _validate_active_name(None) is None
    for value, error in ((1, TypeError), ("class", ValueError)):
        with pytest.raises(error):
            _validate_active_name(value)

    code = """
@decorator
def helper(value=data_0, *, scale=data_1) -> data_2:
    return value

async def async_helper(value=data_3):
    return value

lambda_value = lambda value=data_4: value

@class_decorator
class Child(Base, metaclass=data_5):
    pass
"""
    statements = ast.parse(code).body
    assert _statement_load_count(statements[0], "data_0") == 1
    assert _statement_load_count(statements[0], "data_1") == 1
    assert _statement_load_count(statements[0], "data_2") == 1
    assert (
        _statement_store_count(statements[0], "helper", count_definition_names=True)
        == 1
    )
    assert _statement_load_count(statements[1], "data_3") == 1
    assert _statement_load_count(statements[2], "data_4") == 1
    assert _statement_load_count(statements[3], "class_decorator") == 1
    assert _statement_load_count(statements[3], "data_5") == 1
    assert (
        _statement_store_count(statements[3], "Child", count_definition_names=True) == 1
    )
    assert "source_data" in _replace_code_identifiers(
        code,
        {"data_0": "source_data"},
    )
    rebased = rebase_default_replay_input(
        """
def normalize(value=data) -> data:
    return value

lambda_value = lambda value=data: value

class Child(data):
    pass

derived = data
""",
        "source_data",
    )
    assert "source_data" in rebased
    assert (
        rebase_default_replay_input("derived = data", "not valid python(")
        == "derived = data"
    )
    assert _simplify_display_code("derived =") == "derived ="
    assert _simplify_display_code("") == ""
    assert _simplify_display_code("for item in data:\n    pass") == (
        "for item in data:\n    pass"
    )
    assert _simplify_display_code("left = right = data\nresult = left") == (
        "left = right = data\nresult = left"
    )
    assert _simplify_display_code("tmp = data\nresult = tmp") == "result = data"
    assert (
        _simplify_display_code(
            "tmp = data\nother = 1\nresult = tmp",
            inline_targets={"missing"},
        )
        == "tmp = data\nother = 1\nresult = tmp"
    )

    wrapped = staticmethod(lambda: None)
    assert any(path.endswith(".<lambda>") for path in _callable_paths(wrapped))
    assert _callable_paths(types.SimpleNamespace(__module__=1, __name__=2)) == set()

    with pytest.raises(ValidationError):
        ScriptInput(name=None, label="Input")
    with pytest.raises(ValidationError):
        ScriptInput(name="data_0", label="Input", node_snapshot_token="")
    with pytest.raises(TypeError, match="script input provenance"):
        ScriptInput(name="data_0", label="Input", provenance_spec=object())
    with pytest.raises(TypeError, match="Serialized replay stages"):
        ToolProvenanceSpec(kind="full_data", replay_stages="bad")
    with pytest.raises(TypeError, match="Serialized script inputs"):
        ToolProvenanceSpec(
            kind="script",
            start_label="Run",
            active_name="derived",
            script_inputs="bad",
        )

    assert (
        script(start_label="Run", active_name="derived")._graph_code(display=True)
        is None
    )
    assert (
        full_data(
            ScriptCodeOperation(label="Opaque", code=None, copyable=False)
        ).derivation_code()
        is None
    )
    with pytest.raises(ValueError, match="not valid Python"):
        _validate_script_replay_code("derived =")
    _validate_script_replay_code(
        "try:\n    import numpy as np\nexcept ImportError:\n    pass\nderived = data"
    )
    with pytest.raises(TypeError, match="unsupported Import"):
        _validate_script_replay_code(
            "try:\n    import seaborn\nexcept ImportError:\n    pass\nderived = data"
        )
    with pytest.raises(TypeError, match="unsupported Try"):
        _validate_script_replay_code(
            "try:\n"
            "    import seaborn\n"
            "except ImportError as exc:\n"
            "    pass\n"
            "derived = data"
        )
    with pytest.raises(TypeError, match="unsupported Try"):
        _validate_script_replay_code(
            "try:\n"
            "    import seaborn\n"
            "except (ImportError, ModuleNotFoundError):\n"
            "    pass\n"
            "derived = data"
        )
    for code_snippet, message in (
        ("derived = __name__", "dunder names"),
        ("derived = data.__class__", "dunder attributes"),
        ("derived = open('path')", "cannot call"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_script_replay_code(code_snippet)


def test_script_input_label_is_preserved_and_defaults_to_name() -> None:

    script_input = ScriptInput(
        name="data_0",
        label="  ImageTool 0:\n\n  processed data  ",
    )

    assert script_input.name == "data_0"
    assert script_input.label == "ImageTool 0: processed data"
    assert script_input.model_dump()["label"] == "ImageTool 0: processed data"
    assert ScriptInput(name="data_0").label == "data_0"
    assert ScriptInput(name="data_0", label=None).label == "data_0"
    assert ScriptInput(name="data_0").data_role == "displayed"
    source_input = ScriptInput(name="data_0", data_role="source")
    assert source_input.model_dump(mode="json")["data_role"] == "source"

    node_marker = "snapshot"
    assert ScriptInputDependencyRef(
        "data_0", "ImageTool 0", "node", node_marker
    ) == ScriptInputDependencyRef(
        name="data_0",
        label="ImageTool 0",
        node_uid="node",
        node_snapshot_token=node_marker,
    )
    legacy_dependency = ScriptInputDependencyRef("data_0", "", "")
    assert legacy_dependency.label == ""
    assert legacy_dependency.node_uid == ""
    assert legacy_dependency.data_role == "displayed"

    with pytest.raises(ValidationError):
        ScriptInput(name="data_0", data_role="invalid")

    with pytest.raises(TypeError, match="script input label"):
        ScriptInput(name="data_0", label=1)
    with pytest.raises(ValidationError):
        ScriptInput(name="data_0", label="\n  \t")


def test_replay_script_provenance_uses_resolved_inputs_without_mutating() -> None:
    left = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0, 1]})
    right = xr.DataArray([0.5, 1.5], dims=("x",), coords={"x": [0, 1]})
    spec = script(
        ScriptCodeOperation(
            label="Mutate local input",
            code="data_0[0] = 10.0\nderived = data_0 - data_1",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="ImageTool 0"),
            ScriptInput(name="data_1", label="ImageTool 1"),
        ),
    )

    assert script_provenance_replayable(spec)
    result = replay_script_provenance(
        spec,
        {"data_0": left, "data_1": right},
    )

    xr.testing.assert_identical(
        result,
        xr.DataArray([9.5, 0.5], dims=("x",), coords={"x": [0, 1]}),
    )
    xr.testing.assert_identical(
        left,
        xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0, 1]}),
    )


def test_untrusted_script_replay_imports_use_executor_owned_modules() -> None:
    data = xr.DataArray([1.0], dims=("x",))

    def script_with_code(code: str) -> ToolProvenanceSpec:
        return script(
            ScriptCodeOperation(label="Import", code=code),
            start_label="Run script",
            active_name="derived",
            script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
        )

    safe_import = script(
        ScriptCodeOperation(
            label="Use NumPy",
            code="import numpy as np\nderived = data_0 + np.float64(1.0)",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    safe_erlab_import = script_with_code("import erlab\nderived = data_0.copy()")
    safe_lmfit_import = script_with_code(
        "import lmfit\nderived = data_0 + lmfit.Parameter('value', value=1.0).value"
    )
    optional_approved_import = script_with_code(
        "try:\n"
        "    import xarray as xr\n"
        "except ImportError:\n"
        "    pass\n"
        "derived = data_0 + xr.DataArray(1.0)"
    )
    nested_analysis_alias = script_with_code(
        "def identity(value):\n"
        "    _ = era\n"
        "    return value\n"
        "derived = identity(data_0)"
    )
    optional_external_import = script_with_code(
        "try:\n    import seaborn\nexcept ImportError:\n    pass\nderived = data_0"
    )
    unsafe_from_import = script_with_code(
        "from numpy import __builtins__ as exposed\n"
        "derived = data_0 + exposed['eval']('40 + 2')"
    )
    unsafe_dotted_import = script_with_code(
        "import numpy.testing._private.utils as exposed\n"
        "derived = data_0 + int(exposed.os.path.exists('/'))"
    )
    unsafe_internal_import = script_with_code(
        "import erlab.interactive.imagetool._provenance._code as exposed\n"
        "derived = data_0 + exposed.np.float64(1.0)"
    )
    unsafe_dunder_alias = script_with_code(
        "import numpy as __builtins__\nderived = data_0"
    )
    poisoned_import_policy = script_with_code(
        "framework = erlab.interactive.imagetool._provenance._code\n"
        "framework._SCRIPT_REPLAY_PREBOUND_IMPORTS = {\n"
        "    'numpy': framework,\n"
        "}\n"
        "import numpy as imported_numpy\n"
        "derived = data_0 + imported_numpy.float64(1.0)"
    )

    assert script_provenance_replayable(safe_import)
    assert not script_provenance_requires_trust(safe_import)
    xr.testing.assert_identical(
        replay_script_provenance(safe_import, {"data_0": data}),
        data + 1.0,
    )
    xr.testing.assert_identical(
        replay_script_provenance(safe_erlab_import, {"data_0": data}),
        data,
    )
    assert script_provenance_replayable(safe_lmfit_import)
    assert not script_provenance_requires_trust(safe_lmfit_import)
    xr.testing.assert_identical(
        replay_script_provenance(safe_lmfit_import, {"data_0": data}),
        data + 1.0,
    )
    assert script_provenance_replayable(optional_approved_import)
    assert not script_provenance_requires_trust(optional_approved_import)
    xr.testing.assert_identical(
        replay_script_provenance(optional_approved_import, {"data_0": data}),
        data + 1.0,
    )
    assert script_provenance_replayable(nested_analysis_alias)
    xr.testing.assert_identical(
        replay_script_provenance(nested_analysis_alias, {"data_0": data}),
        data,
    )
    assert "__import__" not in _SCRIPT_REPLAY_ALLOWED_BUILTINS
    assert not hasattr(_code, "_SCRIPT_REPLAY_PREBOUND_IMPORTS")

    try:
        xr.testing.assert_identical(
            replay_script_provenance(
                poisoned_import_policy,
                {"data_0": data},
            ),
            data + 1.0,
        )
        xr.testing.assert_identical(
            replay_script_provenance(safe_import, {"data_0": data}),
            data + 1.0,
        )
    finally:
        if hasattr(_code, "_SCRIPT_REPLAY_PREBOUND_IMPORTS"):
            del _code._SCRIPT_REPLAY_PREBOUND_IMPORTS

    for unsafe, message in (
        (optional_external_import, "unsupported Import"),
        (unsafe_from_import, "unsupported ImportFrom"),
        (unsafe_dotted_import, "unsupported Import"),
        (unsafe_internal_import, "unsupported Import"),
        (unsafe_dunder_alias, "unsupported Import"),
    ):
        assert not script_provenance_replayable(unsafe)
        assert script_provenance_requires_trust(unsafe)
        with pytest.raises(TypeError, match=message):
            replay_script_provenance(unsafe, {"data_0": data})

    xr.testing.assert_identical(
        replay_script_provenance(
            optional_external_import,
            {"data_0": data},
            trusted_user_code=True,
        ),
        data,
    )
    xr.testing.assert_identical(
        replay_script_provenance(
            unsafe_from_import,
            {"data_0": data},
            trusted_user_code=True,
        ),
        data + 42.0,
    )
    xr.testing.assert_identical(
        replay_script_provenance(
            unsafe_dotted_import,
            {"data_0": data},
            trusted_user_code=True,
        ),
        data + 1.0,
    )


def test_replay_script_provenance_rejects_unsupported_or_incomplete_code() -> None:
    data = xr.DataArray([1.0], dims=("x",))
    unsupported = script(
        ScriptCodeOperation(
            label="Unsupported",
            code="import os\nderived = data_0",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    incomplete = script(
        ScriptCodeOperation(label="Incomplete", code=None),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    missing_seed = script(
        AverageOperation(dims=("x",)),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    active_input = script(
        AverageOperation(dims=("x",)),
        start_label="Run script",
        active_name="data_0",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    rename_input = script(
        RenameOperation(name="renamed"),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    external_active_input = script(
        AverageOperation(dims=("x",)),
        start_label="Run script",
        active_name="data_0",
    )
    function_local_active = script(
        ScriptCodeOperation(
            label="Helper",
            code="def helper(data):\n    derived = data\n",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    missing_helper = script(
        ScriptCodeOperation(
            label="Missing helper",
            code="derived = helper(data_0)",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    missing_helper_global = script(
        ScriptCodeOperation(
            label="Missing helper global",
            code=(
                "def helper(data):\n    return data + scale\n\nderived = helper(data_0)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    captured_helper_global = script(
        ScriptCodeOperation(
            label="Captured helper global",
            code=(
                "scale = 2.0\n"
                "\n"
                "def helper(data):\n"
                "    return data + scale\n"
                "\n"
                "derived = helper(data_0)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    late_helper_global = script(
        ScriptCodeOperation(
            label="Late helper global",
            code=(
                "def helper(data):\n"
                "    return data + scale\n"
                "\n"
                "derived = helper(data_0)\n"
                "scale = 2.0"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    redefined_helper = script(
        ScriptCodeOperation(
            label="Redefined helper",
            code=(
                "def helper(data):\n"
                "    return data + missing\n"
                "\n"
                "def helper(data):\n"
                "    return data\n"
                "\n"
                "derived = helper(data_0)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )

    assert not script_provenance_replayable(unsupported)
    assert not script_provenance_replayable(incomplete)
    assert not script_provenance_replayable(missing_seed)
    assert not script_provenance_replayable(function_local_active)
    assert not script_provenance_replayable(missing_helper)
    assert not script_provenance_replayable(missing_helper_global)
    assert not script_provenance_replayable(late_helper_global)
    assert script_provenance_replayable(active_input)
    assert script_provenance_replayable(rename_input)
    assert script_provenance_replayable(captured_helper_global)
    assert script_provenance_replayable(redefined_helper)
    with pytest.raises(TypeError, match="unsupported Import"):
        replay_script_provenance(unsupported, {"data_0": data})
    with pytest.raises(ValueError, match="non-replayable"):
        replay_script_provenance(incomplete, {"data_0": data})
    with pytest.raises(TypeError, match="no replay code"):
        replay_script_provenance(missing_seed, {"data_0": data})
    with pytest.raises(TypeError, match="no replay code"):
        replay_script_provenance(function_local_active, {"data_0": data})
    with pytest.raises(TypeError, match="unresolved name 'helper'"):
        replay_script_provenance(missing_helper, {"data_0": data})
    with pytest.raises(TypeError, match="unresolved name 'scale'"):
        replay_script_provenance(missing_helper_global, {"data_0": data})
    with pytest.raises(TypeError, match="unresolved name 'scale'"):
        replay_script_provenance(late_helper_global, {"data_0": data})
    xr.testing.assert_identical(
        replay_script_provenance(active_input, {"data_0": data}),
        data.qsel.average("x"),
    )
    xr.testing.assert_identical(
        replay_script_provenance(rename_input, {"data_0": data}),
        data.rename("renamed"),
    )
    xr.testing.assert_identical(
        replay_script_provenance(external_active_input, {"data_0": data}),
        data.qsel.average("x"),
    )
    xr.testing.assert_identical(
        replay_script_provenance(captured_helper_global, {"data_0": data}),
        data + 2.0,
    )
    xr.testing.assert_identical(
        replay_script_provenance(redefined_helper, {"data_0": data}),
        data,
    )


def test_replay_script_provenance_accepts_console_module_aliases() -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    spec = script(
        ScriptCodeOperation(
            label="Rotate",
            code=(
                "derived = era.transform.rotate("
                "data_0, 0.0, axes=('x', 'y'), reshape=False)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="ImageTool 0"),),
    )

    assert script_provenance_replayable(spec)
    xr.testing.assert_identical(
        replay_script_provenance(spec, {"data_0": data}),
        erlab.analysis.transform.rotate(data, 0.0, axes=("x", "y"), reshape=False),
    )


def test_console_pattern_expands_named_xarray_mapping_arguments() -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0.0, 1.0]})

    qsel_operation = operation_from_console_call(
        ConsoleCall(
            accessor_path=("qsel",),
            kwargs={"indexers": {"x": 1.0}},
            display_code="data.qsel(indexers={'x': 1.0})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    isel_operation = operation_from_console_call(
        ConsoleCall(
            dataarray_method="isel",
            kwargs={"indexers": {"x": 1}},
            display_code="data.isel(indexers={'x': 1})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    sel_operation = operation_from_console_call(
        ConsoleCall(
            dataarray_method="sel",
            kwargs={"indexers": {"x": 1.0}},
            display_code="data.sel(indexers={'x': 1.0})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    interp_operation = operation_from_console_call(
        ConsoleCall(
            dataarray_method="interp",
            kwargs={"coords": {"x": [0.25, 0.75]}},
            display_code="data.interp(coords={'x': [0.25, 0.75]})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    rename_operation = operation_from_console_call(
        ConsoleCall(
            dataarray_method="rename",
            kwargs={"new_name_or_name_dict": {"x": "energy"}},
            display_code="data.rename(new_name_or_name_dict={'x': 'energy'})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    multidim_coord_operation = operation_from_console_call(
        ConsoleCall(
            dataarray_method="assign_coords",
            kwargs={"foo": (("x", "y"), np.ones((2, 2)))},
            display_code="data.assign_coords(foo=(('x', 'y'), values))",
            has_extra_tracked_inputs=False,
            receiver_data=xr.DataArray(np.ones((2, 2)), dims=("x", "y")),
        )
    )

    assert qsel_operation == QSelOperation(kwargs={"x": 1.0})
    assert isel_operation == IselOperation(kwargs={"x": 1})
    assert sel_operation == SelOperation(kwargs={"x": 1.0})
    assert interp_operation == InterpolationOperation(dim="x", values=[0.25, 0.75])
    assert rename_operation == RenameDimsCoordsOperation(mapping={"x": "energy"})
    assert multidim_coord_operation is None
    assert isinstance(qsel_operation, QSelOperation)
    assert isinstance(isel_operation, IselOperation)
    assert isinstance(sel_operation, SelOperation)
    assert isinstance(interp_operation, InterpolationOperation)
    assert isinstance(rename_operation, RenameDimsCoordsOperation)
    xr.testing.assert_identical(
        qsel_operation.apply(data),
        data.qsel(indexers={"x": 1.0}),
    )
    xr.testing.assert_identical(
        isel_operation.apply(data),
        data.isel(indexers={"x": 1}),
    )
    xr.testing.assert_identical(
        sel_operation.apply(data),
        data.sel(indexers={"x": 1.0}),
    )
    xr.testing.assert_identical(
        interp_operation.apply(data),
        data.interp(coords={"x": [0.25, 0.75]}),
    )
    xr.testing.assert_identical(
        rename_operation.apply(data),
        data.rename(new_name_or_name_dict={"x": "energy"}),
    )


def test_console_pattern_matches_public_parameter_aliases() -> None:
    edge_fit = xr.Dataset({"center": ("x", [0.0, 1.0])})

    operation = operation_from_console_call(
        ConsoleCall(
            func=erlab.analysis.gold.correct_with_edge,
            kwargs={"modelresult": edge_fit, "shift_coords": False},
            display_code="era.gold.correct_with_edge(data, modelresult=edge_fit)",
            has_extra_tracked_inputs=False,
        )
    )

    assert isinstance(operation, CorrectWithEdgeOperation)
    assert not operation.shift_coords
    xr.testing.assert_identical(operation.decoded_edge_fit, edge_fit)


def test_console_pattern_matches_new_replayable_operations() -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "eV"),
        coords={"x": [0.0, 1.0], "eV": [0.0, 1.0, 2.0]},
    )

    aggregate_operation = operation_from_console_call(
        ConsoleCall(
            accessor_path=("qsel", "sum"),
            args=("x",),
            display_code='data.qsel.sum("x")',
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    leading_edge_operation = operation_from_console_call(
        ConsoleCall(
            func=erlab.analysis.interpolate.leading_edge,
            kwargs={"fraction": 0.25, "dim": "eV", "direction": "negative"},
            display_code=(
                "era.interpolate.leading_edge("
                "data, fraction=0.25, dim='eV', direction='negative')"
            ),
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )

    assert aggregate_operation == QSelAggregationOperation(dims=("x",), func="sum")
    assert leading_edge_operation == LeadingEdgeOperation(
        fraction=0.25,
        dim="eV",
        direction="negative",
    )
    xr.testing.assert_identical(
        aggregate_operation.apply(data),
        data.qsel.sum("x"),
    )


def test_sortby_operation_apply_code_and_console_calls() -> None:
    tuple_key = ("beta", 0)
    data = xr.DataArray(
        np.arange(12.0).reshape(4, 3),
        dims=("x", "y"),
        coords={
            "x": [2.0, 1.0, 2.0, 0.0],
            "y": [0.0, 1.0, 2.0],
            "sample temp": ("x", [20.0, 10.0, 15.0, 30.0]),
        },
        name="scan",
    )
    tuple_key_data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={
            "x": [0.0, 1.0, 2.0, 3.0],
            tuple_key: ("x", [20.0, 10.0, 15.0, 30.0]),
        },
        name="tuple_coord_scan",
    )

    single = SortByOperation(variables=("x",))
    multi = SortByOperation(
        variables=("x", "sample temp"),
        ascending=False,
    )
    non_identifier = SortByOperation(variables=("sample temp",))
    tuple_key_operation = SortByOperation(variables=(tuple_key,))

    assert SortByOperation(variables="x") == single
    with pytest.raises(TypeError, match="sortby variables must be coordinate names"):
        SortByOperation(variables=lambda darr: darr.x)
    assert multi.derivation_label().startswith("Sort By(")
    for operation in (single, multi, tuple_key_operation):
        assert (
            parse_tool_provenance_operation(operation.model_dump(mode="json"))
            == operation
        )

    xr.testing.assert_identical(single.apply(data), data.sortby("x"))
    xr.testing.assert_identical(
        multi.apply(data),
        data.sortby(["x", "sample temp"], ascending=False),
    )
    xr.testing.assert_identical(
        non_identifier.apply(data),
        data.sortby("sample temp"),
    )
    xr.testing.assert_identical(
        tuple_key_operation.apply(tuple_key_data),
        tuple_key_data.sortby(tuple_key),
    )

    code = f"derived = {multi.expression_code('data')}"
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"],
        data.sortby(["x", "sample temp"], ascending=False),
    )
    code = f"derived = {tuple_key_operation.expression_code('data')}"
    namespace = _exec_generated_code(code, {"data": tuple_key_data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], tuple_key_data.sortby(tuple_key))

    assert (
        operation_from_console_call(
            ConsoleCall(
                dataarray_method="sortby",
                args=("x",),
                display_code='data.sortby("x")',
                has_extra_tracked_inputs=False,
                receiver_data=data,
            )
        )
        == single
    )
    assert (
        operation_from_console_call(
            ConsoleCall(
                dataarray_method="sortby",
                kwargs={"variables": ["x", "sample temp"], "ascending": False},
                display_code=(
                    'data.sortby(variables=["x", "sample temp"], ascending=False)'
                ),
                has_extra_tracked_inputs=False,
                receiver_data=data,
            )
        )
        == multi
    )
    assert (
        operation_from_console_call(
            ConsoleCall(
                dataarray_method="sortby",
                args=(tuple_key,),
                display_code="data.sortby(('beta', 0))",
                has_extra_tracked_inputs=False,
                receiver_data=tuple_key_data,
            )
        )
        == tuple_key_operation
    )

    for args, kwargs in (
        ((), {"variables": "x", "ascending": "false"}),
        (("x",), {"variables": "y"}),
        ((), {}),
    ):
        assert (
            SortByOperation.from_console_call(
                ConsoleCall(
                    dataarray_method="sortby",
                    args=args,
                    kwargs=kwargs,
                    display_code="data.sortby(...)",
                    has_extra_tracked_inputs=False,
                    receiver_data=data,
                )
            )
            is None
        )

    for variables in (lambda darr: darr.x, data.x, [lambda darr: darr.x], [data.x], []):
        assert (
            SortByOperation.from_console_call(
                ConsoleCall(
                    dataarray_method="sortby",
                    args=(variables,),
                    display_code="data.sortby(...)",
                    has_extra_tracked_inputs=False,
                    receiver_data=data,
                )
            )
            is None
        )
    assert (
        SortByOperation.from_console_call(
            ConsoleCall(
                dataarray_method="sortby",
                args=("x",),
                display_code='data.sortby("x")',
                has_extra_tracked_inputs=True,
                receiver_data=data,
            )
        )
        is None
    )


def test_console_pattern_rejects_ambiguous_calls_and_expands_defaults() -> None:

    assert _console_values_equal(np.nan, np.nan)
    assert _console_mapping_values(
        (None,), {"coords": None, "x": 1}, mapping_kwargs=("coords",)
    ) == {"x": 1}
    assert _console_mapping_values(
        ({"x": 1},), {"coords": {"y": 2}}, mapping_kwargs=("coords",)
    ) == {"x": 1, "y": 2}
    assert _console_mapping_values((1, 2), {}) is None
    assert _console_mapping_values((1,), {}) is None
    assert (
        _console_mapping_values((), {"coords": 1}, mapping_kwargs=("coords",)) is None
    )

    pattern = ConsoleOperationPattern(
        target="builtins.abs",
        fields=("value",),
        field_aliases={"old_value": "value"},
        defaults={"scale": 1},
        ignored_defaults={"drop": False},
    )

    assert (
        pattern.match(
            ConsoleCall(
                display_code="abs(3)",
                has_extra_tracked_inputs=True,
            )
        )
        is None
    )
    assert (
        pattern.match(
            ConsoleCall(
                display_code="abs(3)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        pattern.match(
            ConsoleCall(
                func=len,
                args=(3,),
                display_code="len(3)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        pattern.match(
            ConsoleCall(
                func=abs,
                kwargs={"old_value": 3, "value": 4},
                display_code="abs(old_value=3, value=4)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        pattern.match(
            ConsoleCall(
                func=abs,
                args=(3,),
                kwargs={"drop": True},
                display_code="abs(3, drop=True)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert pattern.match(
        ConsoleCall(
            func=abs,
            args=(3,),
            kwargs={"scale": 2, "drop": False},
            display_code="abs(3, scale=2, drop=False)",
            has_extra_tracked_inputs=False,
        )
    ) == {"value": 3, "scale": 2}

    assert (
        ConsoleOperationPattern(dataarray_method="isel").match(
            ConsoleCall(
                dataarray_method="sel",
                display_code="data.sel()",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        ConsoleOperationPattern().match(
            ConsoleCall(
                dataarray_method="isel",
                display_code="data.isel()",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        ConsoleOperationPattern(accessor_path=("qsel",)).match(
            ConsoleCall(
                accessor_path=("qsel", "mean"),
                display_code="data.qsel.mean()",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        ConsoleOperationPattern(fields=("required",)).match(
            ConsoleCall(
                display_code="data.call()",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        ConsoleOperationPattern().match(
            ConsoleCall(
                kwargs={"unexpected": 1},
                display_code="data.call(unexpected=1)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        ConsoleOperationPattern(kwargs_field="mapping").match(
            ConsoleCall(
                args=(1,),
                display_code="data.call(1)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )


def test_console_operations_match_branch_specific_calls() -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )

    def call(**kwargs: typing.Any) -> typing.Any:
        kwargs.setdefault("display_code", "data.call()")
        kwargs.setdefault("has_extra_tracked_inputs", False)
        kwargs.setdefault("receiver_data", data)
        return ConsoleCall(**kwargs)

    assert TransposeOperation.from_console_call(
        call(
            dataarray_method="transpose",
            args=("y", "x"),
            kwargs={"transpose_coords": True, "missing_dims": "raise"},
        )
    ) == TransposeOperation(dims=("y", "x"))
    assert (
        TransposeOperation.from_console_call(
            call(dataarray_method="transpose", kwargs={"transpose_coords": False})
        )
        is None
    )

    assert (
        SqueezeOperation.from_console_call(
            call(dataarray_method="squeeze", kwargs={"dim": None, "axis": None})
        )
        == SqueezeOperation()
    )
    assert SqueezeOperation.from_console_call(
        call(dataarray_method="squeeze", kwargs={"drop": True})
    ) == SqueezeOperation(drop=True)
    assert SqueezeOperation.from_console_call(
        call(dataarray_method="squeeze", args=("z",), kwargs={"drop": True})
    ) == SqueezeOperation(dims=("z",), drop=True)
    assert SqueezeOperation.from_console_call(
        call(dataarray_method="squeeze", kwargs={"dim": ("z",)})
    ) == SqueezeOperation(dims=("z",))
    assert (
        SqueezeOperation.from_console_call(
            call(dataarray_method="squeeze", args=("z",), kwargs={"dim": "z"})
        )
        is None
    )
    assert (
        SqueezeOperation.from_console_call(
            call(dataarray_method="squeeze", args=("x", "z"), kwargs={})
        )
        is None
    )
    assert (
        SqueezeOperation.from_console_call(
            call(dataarray_method="squeeze", kwargs={"axis": 0})
        )
        is None
    )
    assert (
        SqueezeOperation.from_console_call(
            call(dataarray_method="squeeze", kwargs={"drop": "yes"})
        )
        is None
    )
    assert (
        SqueezeOperation.from_console_call(
            call(dataarray_method="squeeze", kwargs={"unknown": True})
        )
        is None
    )

    assert RenameOperation.from_console_call(
        call(dataarray_method="rename", args=("renamed",))
    ) == RenameOperation(name="renamed")
    assert (
        RenameOperation.from_console_call(
            call(dataarray_method="rename", args=("renamed",), kwargs={"bad": 1})
        )
        is None
    )

    assert AverageOperation.from_console_call(
        call(accessor_path=("qsel", "average"), kwargs={"dim": "x"})
    ) == AverageOperation(dims=("x",))
    assert (
        AverageOperation.from_console_call(
            call(accessor_path=("qsel", "average"), args=("x",), kwargs={"dim": "x"})
        )
        is None
    )

    assert QSelAggregationOperation.from_console_call(
        call(accessor_path=("qsel", "mean"), kwargs={"dim": ("x", "y")})
    ) == QSelAggregationOperation(dims=("x", "y"), func="mean")
    assert (
        QSelAggregationOperation.from_console_call(
            call(accessor_path=("qsel", "median"), kwargs={"dim": "x"})
        )
        is None
    )

    assert InterpolationOperation.from_console_call(
        call(
            dataarray_method="interp",
            args=({"x": [0.25, 0.75]},),
            kwargs={"method": "nearest", "assume_sorted": False, "kwargs": None},
        )
    ) == InterpolationOperation(dim="x", values=[0.25, 0.75], method="nearest")
    for bad_call in (
        call(dataarray_method="interp", kwargs={"method": "cubic", "x": [0.5]}),
        call(dataarray_method="interp", kwargs={"x": [0.5], "assume_sorted": True}),
        call(dataarray_method="interp", args=({"x": [0.5], "y": [0.5]},)),
        call(dataarray_method="interp", kwargs={"x": [[0.0, 1.0]]}),
    ):
        assert InterpolationOperation.from_console_call(bad_call) is None

    assert CoarsenOperation.from_console_call(
        call(
            dataarray_method="coarsen",
            args=({"x": 2},),
            kwargs={"_reducer": "mean", "boundary": "trim"},
        )
    ) == CoarsenOperation(
        dim={"x": 2},
        boundary="trim",
        side="left",
        coord_func="mean",
        reducer="mean",
    )
    for bad_call in (
        call(dataarray_method="coarsen", kwargs={"x": 2}),
        call(dataarray_method="coarsen", args=(2,), kwargs={"_reducer": "mean"}),
        call(
            dataarray_method="coarsen",
            args=({"x": "bad"},),
            kwargs={"_reducer": "mean"},
        ),
        call(
            dataarray_method="coarsen",
            args=({"x": 2},),
            kwargs={"_reducer": "mean", "extra": object()},
        ),
    ):
        assert CoarsenOperation.from_console_call(bad_call) is None

    assert ThinOperation.from_console_call(
        call(dataarray_method="thin", args=(2,))
    ) == ThinOperation(mode="global", factor=2)
    assert ThinOperation.from_console_call(
        call(dataarray_method="thin", args=(None,), kwargs={"x": 2})
    ) == ThinOperation(mode="per_dim", factors={"x": 2})
    assert ThinOperation.from_console_call(
        call(dataarray_method="thin", args=({"x": 2},), kwargs={"y": 2})
    ) == ThinOperation(mode="per_dim", factors={"x": 2, "y": 2})
    for bad_call in (
        call(dataarray_method="thin", args=(1, 2)),
        call(dataarray_method="thin", args=(1,), kwargs={"x": 2}),
    ):
        assert ThinOperation.from_console_call(bad_call) is None

    assert RenameDimsCoordsOperation.from_console_call(
        call(dataarray_method="rename", kwargs={"new_name_or_name_dict": {"x": "kx"}})
    ) == RenameDimsCoordsOperation(mapping={"x": "kx"})
    assert (
        RenameDimsCoordsOperation.from_console_call(
            call(dataarray_method="rename", args=(None,))
        )
        is None
    )

    assigned = AssignCoordsOperation.from_console_call(
        call(dataarray_method="assign_coords", kwargs={"x": np.array([2.0, 3.0])})
    )
    assert isinstance(assigned, AssignCoordsOperation)
    np.testing.assert_allclose(assigned.decoded_values, [2.0, 3.0])
    assert AssignScalarCoordOperation.from_console_call(
        call(dataarray_method="assign_coords", kwargs={"temperature": 21.5})
    ) == AssignScalarCoordOperation(coord_name="temperature", value=21.5)
    assert AssignCoord1DOperation.from_console_call(
        call(
            dataarray_method="assign_coords",
            kwargs={"temperature": ("x", [100, 101])},
        )
    ) == AssignCoord1DOperation(coord_name="temperature", dim="x", values=[100, 101])
    for bad_call in (
        call(dataarray_method="assign_coords", args=(1, 2)),
        call(dataarray_method="assign_coords", kwargs={"z": np.array([1.0, 2.0])}),
        call(
            dataarray_method="assign_coords",
            kwargs={"temperature": (["x"], [1, 2])},
        ),
        call(
            dataarray_method="assign_coords",
            kwargs={"temperature": ("x", np.ones((2, 2)))},
        ),
    ):
        assert AssignCoordsOperation.from_console_call(bad_call) is None
        assert AssignCoord1DOperation.from_console_call(bad_call) is None
    assert (
        AssignCoordsOperation.from_console_call(
            call(
                dataarray_method="assign_coords",
                kwargs={"x": ("x", [2.0, 3.0])},
            )
        )
        is None
    )


def test_file_replay_parses_supported_inputs_and_errors(tmp_path, monkeypatch) -> None:
    image = xr.DataArray(
        np.arange(6).reshape((2, 3)),
        dims=("row", "col"),
        name="image",
    )
    line = xr.DataArray(np.arange(3), dims=("energy",), name="line")
    five_dim = xr.DataArray(
        np.zeros((1, 2, 3, 1, 4)),
        dims=("a", "b", "c", "d", "e"),
        name="five_dim",
    )

    parsed_array = _parse_replay_input(np.arange(6).reshape((2, 3)))
    assert len(parsed_array) == 1
    assert isinstance(parsed_array[0], xr.DataArray)

    dataset = xr.Dataset(
        {
            "line": line,
            "image": image,
            "five_dim": five_dim,
            "scalar": xr.DataArray(1.0),
        }
    )
    assert [darr.name for darr in _parse_replay_input(dataset)] == [
        "line",
        "image",
        "five_dim",
    ]

    tree = xr.DataTree.from_dict({"leaf": xr.Dataset({"image": image})})
    assert [darr.name for darr in _parse_replay_input(tree)] == ["image"]
    xr.testing.assert_identical(
        _select_replay_input(
            dataset,
            FileDataSelection(kind="dataset_variable", value="image"),
        ),
        image,
    )
    xr.testing.assert_identical(
        _select_replay_input(
            tree,
            FileDataSelection(kind="datatree_variable", value=("/leaf", "image")),
        ),
        image,
    )
    keyed_tree = xr.DataTree.from_dict({"leaf": xr.Dataset({1: image.rename(1)})})
    keyed_selection = FileDataSelection(
        kind="datatree_variable",
        value=("/leaf", 1),
    )
    xr.testing.assert_identical(
        _select_replay_input(keyed_tree, keyed_selection),
        image.rename(1),
    )
    assert (
        FileDataSelection.model_validate(keyed_selection.model_dump(mode="json"))
        == keyed_selection
    )
    assert _select_replay_input(
        np.arange(6).reshape((2, 3)),
        FileDataSelection(kind="dataarray"),
    ).shape == (2, 3)
    xr.testing.assert_identical(
        _select_replay_input(
            [line, image],
            FileDataSelection(kind="sequence_index", value=1),
        ),
        image,
    )

    with pytest.raises(ValueError, match="No valid data"):
        _parse_replay_input([])
    with pytest.raises(ValueError, match="No valid data"):
        _parse_replay_input(xr.Dataset({"scalar": xr.DataArray(1.0)}))
    with pytest.raises(TypeError, match="Unsupported input type list"):
        _parse_replay_input([object()])
    with pytest.raises(KeyError, match="Selected file variable"):
        _select_replay_input(
            dataset,
            FileDataSelection(kind="dataset_variable", value="missing"),
        )
    with pytest.raises(KeyError, match="Selected file DataTree variable"):
        _select_replay_input(
            tree,
            FileDataSelection(kind="datatree_variable", value=("/missing", "image")),
        )
    with pytest.raises(TypeError, match="require the loader to return a sequence"):
        _select_replay_input(
            image,
            FileDataSelection(kind="sequence_index", value=0),
        )
    with pytest.raises(IndexError, match="sequence index 2 is out of range"):
        _select_replay_input(
            [image],
            FileDataSelection(kind="sequence_index", value=2),
        )

    assert _resolve_importable_callable("xarray.load_dataarray") is xr.load_dataarray
    with pytest.raises(ValueError, match="must be dotted"):
        _resolve_importable_callable("load")
    with pytest.raises(ModuleNotFoundError):
        _resolve_importable_callable("missing_erlab_replay_loader.load")
    with pytest.raises(AttributeError):
        _resolve_importable_callable("xarray.missing_loader.load")
    with pytest.raises(TypeError, match="not callable"):
        _resolve_importable_callable("math.pi")

    broken_module = tmp_path / "broken_loader.py"
    broken_module.write_text(
        "import missing_erlab_replay_dependency\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="missing_erlab_replay_dependency"):
        _resolve_importable_callable("broken_loader.load")

    source_file = tmp_path / "source.h5"
    image.to_netcdf(source_file, engine="h5netcdf")
    dataset_file = tmp_path / "dataset.h5"
    dataset.to_netcdf(dataset_file, engine="h5netcdf")
    xr.testing.assert_identical(
        _load_file_source_data(
            _file_replay_source(
                dataset_file,
                replay_call=FileReplayCall(
                    kind="callable",
                    target="xarray.load_dataset",
                    kwargs={"engine": "h5netcdf"},
                    selection=FileDataSelection(
                        kind="dataset_variable",
                        value="image",
                    ),
                ),
            )
        ),
        image,
    )
    datatree_file = tmp_path / "tree.h5"
    tree.to_netcdf(datatree_file, engine="h5netcdf")
    xr.testing.assert_identical(
        _load_file_source_data(
            _file_replay_source(
                datatree_file,
                replay_call=FileReplayCall(
                    kind="callable",
                    target="xarray.load_datatree",
                    kwargs={"engine": "h5netcdf"},
                    selection=FileDataSelection(
                        kind="datatree_variable",
                        value=("/leaf", "image"),
                    ),
                ),
            )
        ),
        image,
    )
    with pytest.raises(IndexError, match="out of range"):
        _load_file_source_data(
            _file_replay_source(
                source_file,
                replay_call=FileReplayCall(
                    kind="callable",
                    target="xarray.load_dataarray",
                    kwargs={"engine": "h5netcdf"},
                    selected_index=1,
                ),
            )
        )
    with pytest.raises(ValueError, match="replay metadata"):
        _load_file_source_data(
            FileLoadSource(
                path=source_file,
                loader_label="Load Function",
                loader_text="xarray.load_dataarray",
                kwargs_text="(none)",
            )
        )
    with pytest.raises(TypeError, match="Expected structured file provenance"):
        replay_file_provenance(full_data())
    with pytest.raises(TypeError, match="Expected structured file provenance"):
        replay_file_provenance(typing.cast("typing.Any", None))


def test_file_replay_uses_erlab_loader(example_loader, example_data_dir) -> None:
    del example_loader
    file_path = example_data_dir / "data_002.h5"
    spec = file_load(
        start_label="Load data from file 'data_002.h5'",
        seed_code="import erlab\n\nderived = erlab.io.load(2)",
        file_load_source=_file_replay_source(
            file_path,
            replay_call=FileReplayCall(
                kind="erlab_loader",
                target="example",
                kwargs={},
                selected_index=0,
            ),
        ),
    )

    xr.testing.assert_identical(
        replay_file_provenance(spec),
        erlab.io.loaders["example"].load(file_path),
    )


def test_file_provenance_composes_structured_stages_and_replays_modified_source(
    tmp_path,
) -> None:
    path = tmp_path / "source.h5"
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )
    data.to_netcdf(path, engine="h5netcdf")

    file_spec = file_load(
        start_label=f"Load data from file {path.name!r}",
        seed_code=(
            "import xarray\n\n"
            f"derived = xarray.load_dataarray({str(path)!r}, "
            'engine="h5netcdf").astype("float64")'
        ),
        file_load_source=FileLoadSource(
            path=path,
            loader_label="Load Function",
            loader_text="xarray.load_dataarray",
            kwargs_text='engine="h5netcdf"',
            replay_call=FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                kwargs={"engine": "h5netcdf"},
                selected_index=0,
                cast_float64=True,
            ),
            load_code=(
                "import xarray\n\n"
                f"data = xarray.load_dataarray({str(path)!r}, "
                'engine="h5netcdf").astype("float64")'
            ),
        ),
    )
    first_stage = full_data(
        AverageOperation(dims=("x",)),
        RenameOperation(name="avg"),
    )
    second_stage = selection(
        IselOperation(kwargs={"y": slice(0, 2)}),
        RenameDimsCoordsOperation(mapping={"y": "energy"}),
        AssignCoordsOperation(
            coord_name="energy",
            values=np.array([10.0, 20.0]),
        ),
    )

    composed = compose_full_provenance(file_spec, first_stage)
    composed = compose_full_provenance(composed, second_stage)

    assert composed is not None
    assert composed.kind == "file"
    assert [step.input_policy for step in composed.steps] == [
        "current",
        "current",
        "restored",
        "restored",
        "restored",
    ]
    assert all(
        not isinstance(step.operation, ScriptCodeOperation) for step in composed.steps
    )
    assert composed.display_entries()[0].label == "Load data from file 'source.h5'"
    assert any("Average" in entry.label for entry in composed.display_entries())

    code = composed.display_code()
    assert code is not None
    assert "import xarray" in code
    assert "xarray.load_dataarray" in code
    assert '.rename("avg")' not in code
    assert ".rename(y=" in code
    assert "data =" not in code
    namespace = _exec_generated_code(code, {})
    assert isinstance(namespace["derived"], xr.DataArray)

    updated = data + 100
    updated.to_netcdf(path, engine="h5netcdf")
    live_expected = second_stage.apply(first_stage.apply(updated.astype(np.float64)))
    xr.testing.assert_identical(replay_file_provenance(composed), live_expected)


def test_tool_provenance_compose_display_provenance_streamlines_live_source() -> None:
    parent = script(
        start_label="Start from watched variable 'my_data_name'",
        seed_code="derived = my_data_name",
    )
    source = selection(
        IselOperation(kwargs={"z": 0}),
        SortCoordOrderOperation(),
        SqueezeOperation(),
    )

    composed = compose_display_provenance(
        parent,
        source,
        parent_data=_base_data(),
    )

    assert composed is not None
    assert [operation.op for operation in composed.operations] == ["isel"]
    code = composed.display_code()
    assert code is not None
    namespace = _exec_generated_code(
        code,
        {"my_data_name": _base_data().copy(deep=True)},
    )
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, _base_data().isel(z=0))


def test_tool_provenance_display_compose_keeps_default_seed_without_parent() -> None:
    source = selection(
        IselOperation(kwargs={"z": 0}),
        SortCoordOrderOperation(),
        SqueezeOperation(),
    )

    composed = compose_display_provenance(
        None,
        source,
        parent_data=_base_data(),
    )

    assert composed is not None
    code = composed.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, {"data": _base_data().copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, _base_data().isel(z=0))


def test_tool_provenance_direct_replay_input_name_requires_simple_seed() -> None:

    watched = script(
        start_label="Start from watched variable 'my_data'",
        seed_code="derived = my_data",
    )
    assert direct_replay_input_name(watched) == "my_data"
    watched_cast = script(
        start_label="Start from watched variable 'my_data'",
        seed_code="derived = my_data.astype(np.float64)",
    )
    assert direct_replay_input_name(watched_cast) == ("my_data.astype(np.float64)")

    assert (
        direct_replay_input_name(
            script(
                start_label="Start from current parent ImageTool data",
                seed_code="derived = data",
            )
        )
        is None
    )
    assert (
        direct_replay_input_name(
            script(
                start_label="Start from watched variable 'my_data'",
                seed_code="derived = data.sel(x=0)",
            )
        )
        is None
    )


def test_tool_provenance_compose_display_replay_omits_synthetic_1d_squeeze() -> None:
    parent = script(
        start_label="Start from watched variable 'my_1d'",
        seed_code="derived = my_1d",
    )
    source = selection(
        SortCoordOrderOperation(),
        SqueezeOperation(),
    )
    parent_data = xr.DataArray(
        np.arange(5).reshape((5, 1)),
        dims=("x", "stack_dim"),
        coords={"x": np.arange(5), "stack_dim": [0]},
    )
    parent_data = mark_promoted_1d_source(parent_data)

    composed = compose_display_provenance(
        parent,
        source,
        parent_data=parent_data,
    )

    assert composed is not None
    code = composed.display_code()
    assert code is not None
    watched_data = xr.DataArray(
        np.arange(5),
        dims=("x",),
        coords={"x": np.arange(5)},
    )
    namespace = _exec_generated_code(code, {"my_1d": watched_data.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, watched_data)
    assert ".squeeze()" not in code

    explicit_source = selection(
        SortCoordOrderOperation(),
        SqueezeOperation(drop=True),
    )
    explicit_composed = compose_display_provenance(
        parent,
        explicit_source,
        parent_data=parent_data,
    )
    assert explicit_composed is not None
    explicit_code = explicit_composed.display_code()
    assert explicit_code is not None
    assert ".squeeze(drop=True)" in explicit_code
    explicit_namespace = _exec_generated_code(
        explicit_code,
        {"my_1d": watched_data.copy(deep=True)},
    )
    explicit_derived = explicit_namespace["derived"]
    assert isinstance(explicit_derived, xr.DataArray)
    xr.testing.assert_identical(explicit_derived, watched_data.squeeze(drop=True))


def test_model_fit_operation_replays_selected_parameter_as_dataarray() -> None:
    x = np.linspace(-1.0, 1.0, 11)
    y = np.array([0, 1])
    data = xr.DataArray(
        np.stack((1.0 + 2.0 * x, 3.0 + 4.0 * x)),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    operation = ModelFitOperation(
        fit_dim="x",
        model="PolynomialModel",
        model_kwargs={"degree": 1},
        parameters={
            "c0": _ModelFitParameterSpec(value=(0.5, 2.5)),
            "c1": _ModelFitParameterSpec(
                value=(1.5, 3.5),
                minimum=(-np.inf, 0.0),
                maximum=(3.0, np.inf),
            ),
        },
        method="leastsq",
        parameter="c1",
        broadcast_dim="y",
    )

    expected = operation.apply(data)
    assert expected.name == "c1_values"
    np.testing.assert_allclose(expected.values, [2.0, 4.0])

    code = f"derived = {operation.expression_code('data')}"
    assert "imagetool" not in code
    assert "fit_result" not in code
    assert "-np.inf" in code
    assert "np.inf" in code
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], expected)

    parsed = parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert isinstance(parsed, ModelFitOperation)
    assert parsed == operation

    stderr_operation = operation.model_copy(update={"output": "stderr"})
    stderr = stderr_operation.apply(data)
    assert stderr.name == "c1_stderr"
    assert isinstance(stderr, xr.DataArray)
    assert np.isfinite(stderr.values).all()


def test_model_fit_operation_replays_fixed_and_expression_parameters() -> None:
    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(
        1.0 + 4.0 * x + 2.0 * x**2,
        dims=("x",),
        coords={"x": x},
    )
    operation = ModelFitOperation(
        fit_dim="x",
        model="PolynomialModel",
        model_kwargs={"degree": 2},
        parameters={
            "c0": _ModelFitParameterSpec(value=1.0, vary=False),
            "c1": _ModelFitParameterSpec(expr="2 * c2"),
            "c2": _ModelFitParameterSpec(value=1.0),
        },
        method="leastsq",
        parameter="c2",
    )

    expected = operation.apply(data)
    np.testing.assert_allclose(expected, 2.0)

    code = f"derived = {operation.expression_code('data')}"
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], expected)


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        ({"expr": " "}, "expressions must not be empty"),
        ({"value": 1.0, "expr": "c0"}, "cannot define values or bounds"),
        ({"vary": False, "expr": "c0"}, "cannot define vary=False"),
        ({}, "must define a value or expression"),
        ({"value": ()}, "arrays must not be empty"),
        ({"value": np.nan}, "must not contain NaN"),
        ({"value": np.inf}, "values must be finite"),
        (
            {"value": (1.0, 2.0), "minimum": (0.0,)},
            "arrays must have equal lengths",
        ),
    ],
)
def test_model_fit_parameter_spec_rejects_invalid_state(
    parameter: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ModelFitParameterSpec.model_validate(parameter)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"fit_dim": ""}, ValueError, "dimension must not be empty"),
        ({"broadcast_dim": "x"}, ValueError, "dimensions must differ"),
        ({"method": " "}, ValueError, "method must not be empty"),
        ({"parameter": ""}, ValueError, "output parameter must not be empty"),
        ({"parameters": {}}, ValueError, "parameters must not be empty"),
        (
            {"parameters": {"": _ModelFitParameterSpec(value=1.0)}},
            ValueError,
            "parameter names must not be empty",
        ),
        (
            {"model_kwargs": {1: 1}},
            TypeError,
            "constructor kwargs must use string keys",
        ),
    ],
)
def test_model_fit_operation_rejects_invalid_state(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "fit_dim": "x",
        "model": "PolynomialModel",
        "model_kwargs": {"degree": 1},
        "parameters": {"c1": _ModelFitParameterSpec(value=1.0)},
        "method": "leastsq",
        "parameter": "c1",
    }
    arguments.update(overrides)

    with pytest.raises(error, match=message):
        ModelFitOperation.model_validate(arguments)


def test_model_fit_operation_rejects_ambiguous_parameter_shapes() -> None:
    parameters = {
        "c0": _ModelFitParameterSpec(value=(0.0, 1.0)),
        "c1": _ModelFitParameterSpec(value=1.0),
    }
    with pytest.raises(ValueError, match="broadcast dimension"):
        ModelFitOperation(
            fit_dim="x",
            model="PolynomialModel",
            model_kwargs={"degree": 1},
            parameters=parameters,
            method="leastsq",
            parameter="c1",
        )
    with pytest.raises(ValueError, match="Unsupported model-fit model"):
        ModelFitOperation(
            fit_dim="x",
            model="CustomModel",
            parameters={"c1": _ModelFitParameterSpec(value=1.0)},
            method="leastsq",
            parameter="c1",
        )

    operation = ModelFitOperation(
        fit_dim="x",
        model="PolynomialModel",
        model_kwargs={"degree": 1},
        parameters=parameters,
        method="leastsq",
        parameter="c1",
        broadcast_dim="y",
    )
    data = xr.DataArray(
        np.ones((3, 5)),
        dims=("y", "x"),
        coords={"y": [0, 1, 2], "x": np.arange(5)},
    )
    with pytest.raises(ValueError, match="does not match dimension"):
        operation.apply(data)

    data_without_broadcast_dim = xr.DataArray(
        np.ones(5),
        dims=("x",),
        coords={"x": np.arange(5)},
    )
    with pytest.raises(ValueError, match="was not found in data"):
        operation.apply(data_without_broadcast_dim)

    data_without_fit_dim = xr.DataArray(
        np.ones(2),
        dims=("y",),
        coords={"y": np.arange(2)},
    )
    with pytest.raises(ValueError, match="was not found in data"):
        operation.apply(data_without_fit_dim)
