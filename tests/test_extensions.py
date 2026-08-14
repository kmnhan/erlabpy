from __future__ import annotations

import hashlib
import sys
import types
import typing

import numpy as np
import pytest
import xarray as xr

import erlab
import erlab.extensions._api as extension_api
from erlab.extensions import (
    ExtensionExecutionError,
    ExtensionSignatureError,
    LoaderDescriptor,
    ParameterDescriptor,
    ParameterKind,
    RoutineDescriptor,
    load_script,
    loader,
    routine,
    run_loader,
    run_routine,
)
from erlab.extensions._api import _coerce_call_parameters

if typing.TYPE_CHECKING:
    import importlib.machinery
    import pathlib


def test_routine_decorator_preserves_normal_call_behavior() -> None:
    def normalize(data: xr.DataArray, scale: float = 2.0) -> xr.DataArray:
        return data / scale

    decorated = routine(name="Normalize", category="Lab")(normalize)

    assert decorated is normalize
    xr.testing.assert_identical(
        decorated(xr.DataArray([2.0, 4.0])), xr.DataArray([1.0, 2.0])
    )


def test_loader_decorator_preserves_normal_call_behavior(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "values.txt"
    path.write_text("1 2 3")

    @loader(name="Text")
    def load_text(path: pathlib.Path) -> xr.DataArray:
        return xr.DataArray([float(value) for value in path.read_text().split()])

    xr.testing.assert_identical(load_text(path), xr.DataArray([1.0, 2.0, 3.0]))


def test_loader_decorator_accepts_one_extension_as_a_string(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "loader_extension.py"
    source.write_text(
        "from pathlib import Path\n"
        "import xarray as xr\n"
        "from erlab.extensions import loader\n\n"
        "@loader(extensions='txt')\n"
        "def load_text(path: Path) -> xr.DataArray:\n"
        "    return xr.DataArray([1.0])\n"
    )

    descriptor = load_script(source).loaders["load_text"][0]

    assert descriptor.extensions == (".txt",)


def test_loader_decorator_rejects_invalid_extensions() -> None:
    with pytest.raises(ValueError, match="must contain a suffix"):
        loader(extensions="")
    with pytest.raises(TypeError, match="must be strings"):
        loader(extensions=typing.cast("typing.Any", (1,)))


def test_load_script_validates_parameters_and_exact_source(
    tmp_path: pathlib.Path,
) -> None:
    script = tmp_path / "extension.py"
    script.write_text(
        """from __future__ import annotations
import enum
import pathlib
import typing
import xarray as xr
from erlab.extensions import routine

class Mode(enum.Enum):
    ADD = "add"
    MULTIPLY = "multiply"

@routine(name="Adjust", id="stable-adjust", category="Lab")
def adjust(
    data: xr.DataArray,
    mode: Mode = Mode.ADD,
    amount: float = 1.0,
    label: str | None = None,
    choice: typing.Literal[1, 2] = 1,
    output: pathlib.Path | None = None,
) -> xr.DataArray:
    result = data + amount if mode is Mode.ADD else data * amount
    return result.assign_attrs(label=label, choice=choice, output=str(output))
"""
    )
    loaded = load_script(script)
    descriptor = loaded.routines["stable-adjust"][0]

    assert descriptor.function_name == "adjust"
    assert tuple(parameter.kind for parameter in descriptor.parameters) == (
        ParameterKind.ENUM,
        ParameterKind.NUMBER,
        ParameterKind.STRING,
        ParameterKind.LITERAL,
        ParameterKind.PATH,
    )
    xr.testing.assert_identical(
        loaded.adjust(
            xr.DataArray([2.0]),
            mode="multiply",
            amount=3.0,
            output="result.nc",
        ),
        xr.DataArray([6.0], attrs={"label": None, "choice": 1, "output": "result.nc"}),
    )
    result = run_routine(
        xr.DataArray([2.0]),
        script=script,
        source_hash=loaded.source_hash,
        routine_id="stable-adjust",
        parameters={
            "mode": "multiply",
            "amount": 3.0,
            "label": "test",
            "choice": 2,
            "output": "result.nc",
        },
    )
    assert result.item() == 6.0
    assert result.attrs == {"label": "test", "choice": 2, "output": "result.nc"}

    with pytest.raises(ExtensionExecutionError, match="must be one of"):
        run_routine(
            xr.DataArray([2.0]),
            script=script,
            routine_id="stable-adjust",
            parameters={"choice": 3},
        )
    bool_script = tmp_path / "bool_extension.py"
    bool_script.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def choose(data: xr.DataArray, enabled: bool) -> xr.DataArray:
    return data if enabled else -data
"""
    )
    with pytest.raises(ExtensionExecutionError, match="must be a bool"):
        run_routine(
            xr.DataArray([2.0]),
            script=bool_script,
            routine_id="choose",
            parameters={"enabled": 1},
        )
    with pytest.raises(ExtensionExecutionError, match="must be finite"):
        run_routine(
            xr.DataArray([2.0]),
            script=script,
            routine_id="stable-adjust",
            parameters={"amount": float("nan")},
        )

    script.write_text(script.read_text() + "\n# changed\n")
    with pytest.raises(erlab.extensions.ExtensionImportError, match="does not match"):
        load_script(script, expected_source_hash=loaded.source_hash)


def test_loaded_script_preserves_natural_function_call_forms(
    tmp_path: pathlib.Path,
) -> None:
    script = tmp_path / "natural_calls.py"
    script.write_text(
        """import enum
import xarray as xr
from erlab.extensions import routine

class Mode(enum.Enum):
    ADD = "add"
    MULTIPLY = "multiply"

@routine()
def adjust(
    data: xr.DataArray,
    amount: float = 1.0,
    mode: Mode = Mode.ADD,
) -> xr.DataArray:
    return data + amount if mode is Mode.ADD else data * amount
"""
    )
    loaded = load_script(script)
    data = xr.DataArray([2.0])

    xr.testing.assert_identical(
        loaded.adjust(data, 3.0, "multiply"), xr.DataArray([6.0])
    )
    xr.testing.assert_identical(
        loaded.adjust(data=data, amount=3.0, mode="multiply"),
        xr.DataArray([6.0]),
    )


def test_load_script_registers_module_during_import(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "dataclass_extension.py"
    script.write_text(
        """from __future__ import annotations
from dataclasses import dataclass
import xarray as xr
from erlab.extensions import routine

@dataclass
class Settings:
    scale: float = 2.0

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * Settings().scale
"""
    )
    module_name = "_erlab_test_dataclass_extension"
    try:
        loaded = load_script(script, module_name=module_name)
        assert sys.modules[module_name] is loaded.module
        xr.testing.assert_identical(
            loaded.routines["scale"][1](xr.DataArray([1.0, 2.0])),
            xr.DataArray([2.0, 4.0]),
        )
    finally:
        sys.modules.pop(module_name, None)


def test_load_script_executes_verified_source_snapshot(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "extension.py"
    script.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def value(data: xr.DataArray) -> xr.DataArray:
    return data + 1
"""
    )
    revision = hashlib.sha256(script.read_bytes()).hexdigest()
    spec_from_file_location = extension_api.importlib.util.spec_from_file_location

    def replace_source_during_import(
        name: str, location: pathlib.Path
    ) -> importlib.machinery.ModuleSpec | None:
        script.write_text(script.read_text().replace("data + 1", "data + 9"))
        return spec_from_file_location(name, location)

    monkeypatch.setattr(
        extension_api.importlib.util,
        "spec_from_file_location",
        replace_source_during_import,
    )

    loaded = load_script(script, expected_source_hash=revision)

    xr.testing.assert_identical(
        loaded.routines["value"][1](xr.DataArray([1])), xr.DataArray([2])
    )


def test_load_script_rejects_unsupported_signature(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def bad(data: xr.DataArray, values: list[float]) -> xr.DataArray:
    return data
"""
    )
    with pytest.raises(ExtensionSignatureError, match="unsupported annotation"):
        load_script(script)


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            "def bad() -> xr.DataArray:\n    return xr.DataArray()",
            "must have an input parameter",
        ),
        (
            "def bad(data: xr.DataArray, *values: float) -> xr.DataArray:\n"
            "    return data",
            "cannot use .*args",
        ),
        (
            "def bad(data: xr.DataArray = xr.DataArray()) -> xr.DataArray:\n"
            "    return data",
            "must require its input",
        ),
        (
            "def bad(data: xr.DataArray, value: float, /) -> xr.DataArray:\n"
            "    return data",
            "cannot use positional-only user parameters",
        ),
        (
            "def bad(data: xr.Dataset) -> xr.DataArray:\n    return xr.DataArray()",
            "first parameter as xarray.DataArray",
        ),
        (
            "def bad(data: xr.DataArray) -> xr.Dataset:\n    return xr.Dataset()",
            "must return xarray.DataArray",
        ),
        (
            "def bad(data: 'MissingType') -> xr.DataArray:\n    return xr.DataArray()",
            "Could not resolve annotations",
        ),
    ],
)
def test_load_script_rejects_invalid_routine_signatures(
    tmp_path: pathlib.Path,
    declaration: str,
    message: str,
) -> None:
    script = tmp_path / "invalid_routine.py"
    script.write_text(
        "import xarray as xr\n"
        "from erlab.extensions import routine\n\n"
        "@routine()\n"
        f"{declaration}\n"
    )

    with pytest.raises(ExtensionSignatureError, match=message):
        load_script(script)


@pytest.mark.parametrize(
    ("supporting_code", "annotation", "default", "message"),
    [
        ("", "typing.Literal[()]", None, "Literal values"),
        (
            "class Mode(enum.Enum):\n    VALUE = object()\n",
            "Mode",
            None,
            "Enum values",
        ),
        (
            "class Mode(enum.Enum):\n    VALUE = 'value'\n",
            "Mode",
            "'value'",
            "not a Mode member",
        ),
        ("", "pathlib.Path", "'value'", "pathlib.Path default"),
        ("", "bool", "1", "bool default"),
        ("", "float", "'value'", "numeric default"),
        ("", "str", "1", "string default"),
    ],
)
def test_load_script_rejects_other_invalid_parameter_defaults(
    tmp_path: pathlib.Path,
    supporting_code: str,
    annotation: str,
    default: str | None,
    message: str,
) -> None:
    script = tmp_path / "invalid_parameter.py"
    default_clause = "" if default is None else f" = {default}"
    script.write_text(
        "import enum\n"
        "import pathlib\n"
        "import typing\n"
        "import xarray as xr\n"
        "from erlab.extensions import routine\n\n"
        f"{supporting_code}\n"
        "@routine()\n"
        "def invalid_parameter(\n"
        "    data: xr.DataArray,\n"
        f"    value: {annotation}{default_clause},\n"
        ") -> xr.DataArray:\n"
        "    return data\n"
    )

    with pytest.raises(ExtensionSignatureError, match=message):
        load_script(script)


def test_load_script_rejects_invalid_loader_input_and_duplicate_ids(
    tmp_path: pathlib.Path,
) -> None:
    invalid_loader = tmp_path / "invalid_loader_input.py"
    invalid_loader.write_text(
        "import xarray as xr\n"
        "from erlab.extensions import loader\n\n"
        "@loader()\n"
        "def invalid(path: str) -> xr.DataArray:\n"
        "    return xr.DataArray()\n"
    )
    with pytest.raises(
        ExtensionSignatureError, match=r"first parameter as pathlib\.Path"
    ):
        load_script(invalid_loader)

    duplicate = tmp_path / "duplicate.py"
    duplicate.write_text(
        "import xarray as xr\n"
        "from erlab.extensions import routine\n\n"
        "@routine(id='same')\n"
        "def first(data: xr.DataArray) -> xr.DataArray:\n"
        "    return data\n\n"
        "@routine(id='same')\n"
        "def second(data: xr.DataArray) -> xr.DataArray:\n"
        "    return data\n"
    )
    with pytest.raises(ExtensionSignatureError, match="defined more than once"):
        load_script(duplicate)


def test_load_script_reports_source_and_import_failures(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.py"
    with pytest.raises(erlab.extensions.ExtensionImportError, match="Could not read"):
        load_script(missing)

    source = tmp_path / "empty.py"
    source.write_text("VALUE = 1\n")
    with pytest.raises(ExtensionSignatureError, match="no decorated capabilities"):
        load_script(source)

    monkeypatch.setattr(
        extension_api.importlib.util, "spec_from_file_location", lambda *_: None
    )
    with pytest.raises(erlab.extensions.ExtensionImportError, match="import spec"):
        load_script(source)


def test_load_script_rejects_an_existing_module_name(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "extension.py"
    source.write_text(
        "from erlab.extensions import routine\n"
        "@routine()\n"
        "def identity(data):\n"
        "    return data\n"
    )
    module_name = "_erlab_test_existing_extension"
    previous = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, previous)

    with pytest.raises(erlab.extensions.ExtensionImportError, match="already in use"):
        load_script(source, module_name=module_name)

    assert sys.modules[module_name] is previous


def test_load_script_removes_a_new_module_after_import_failure(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "broken.py"
    source.write_text("raise RuntimeError('broken import')\n")
    module_name = "_erlab_test_broken_extension"

    with pytest.raises(erlab.extensions.ExtensionImportError, match="broken import"):
        load_script(source, module_name=module_name)

    assert module_name not in sys.modules


@pytest.mark.parametrize(
    ("annotation", "default", "message"),
    [
        ("int", "'two'", "int default"),
        ("int", "None", "not optional"),
        ("float", "float('nan')", "must be finite"),
        ("typing.Literal['a', 'b']", "'c'", "outside its Literal choices"),
    ],
)
def test_load_script_rejects_invalid_parameter_defaults(
    tmp_path: pathlib.Path,
    annotation: str,
    default: str,
    message: str,
) -> None:
    script = tmp_path / "bad_default.py"
    script.write_text(
        f"""import typing
import xarray as xr
from erlab.extensions import routine

@routine()
def bad_default(
    data: xr.DataArray,
    value: {annotation} = {default},
) -> xr.DataArray:
    return data
"""
    )

    with pytest.raises(ExtensionSignatureError, match=message):
        load_script(script)


def test_run_routine_validates_result(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "bad_result.py"
    script.write_text(
        """import typing
import xarray as xr
from erlab.extensions import routine

@routine()
def bad(data: xr.DataArray) -> xr.DataArray:
    return typing.cast(xr.DataArray, data.values)
"""
    )
    with pytest.raises(ExtensionExecutionError, match="expected DataArray"):
        run_routine(xr.DataArray(np.arange(3)), script=script, routine_id="bad")


def test_load_script_runs_a_decorated_loader(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "loader_extension.py"
    source.write_text(
        """from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(name="Text Values", extensions=("txt",))
def text_values(path: Path, scale: float = 1.0) -> xr.DataArray | xr.Dataset:
    return xr.DataArray([float(value) * scale for value in path.read_text().split()])
"""
    )
    values = tmp_path / "values.txt"
    values.write_text("1 2 3")

    loaded = load_script(source)
    descriptor = loaded.loaders["text_values"][0]
    result = run_loader(
        values,
        script=source,
        source_hash=loaded.source_hash,
        loader_id="text_values",
        parameters={"scale": 2.0},
    )

    assert descriptor.extensions == (".txt",)
    xr.testing.assert_identical(result, xr.DataArray([2.0, 4.0, 6.0]))


def test_load_script_rejects_invalid_loader_return_annotation(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "invalid_loader.py"
    source.write_text(
        """from pathlib import Path
from erlab.extensions import loader

@loader()
def invalid(path: Path) -> list[float]:
    return []
"""
    )

    with pytest.raises(ExtensionSignatureError, match="must return"):
        load_script(source)


def test_script_can_use_an_installed_dependency(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "lab_dependency.py").write_text(
        "def scale(values, factor):\n    return values * factor\n"
    )
    source = tmp_path / "lab_extension.py"
    source.write_text(
        "import xarray as xr\n"
        "from erlab.extensions import routine\n"
        "from lab_dependency import scale\n\n"
        "@routine()\n"
        "def apply_scale(data: xr.DataArray, factor: float = 2.0) -> xr.DataArray:\n"
        "    return scale(data, factor)\n"
    )
    monkeypatch.syspath_prepend(tmp_path)

    loaded = load_script(source)

    xr.testing.assert_identical(
        loaded.apply_scale(xr.DataArray([1.0, 2.0]), factor=3.0),
        xr.DataArray([3.0, 6.0]),
    )


def test_source_resolver_lookup_survives_manager_removal(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("value = 1\n")

    def closing_resolver(_extension_id: str, _source_hash: str) -> pathlib.Path:
        extension_api._remove_resolvers("first")
        raise KeyError

    extension_api._set_source_resolver(
        "first", lambda _extension_id, _source_hash: source_path
    )
    extension_api._set_source_resolver("closing", closing_resolver)
    try:
        assert extension_api._resolved_source("lab", "source") == source_path
    finally:
        extension_api._remove_resolvers("first")
        extension_api._remove_resolvers("closing")


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"unknown": 1}, "Unknown extension parameters"),
        (
            {
                "path": "value.txt",
                "mode": "add",
                "scale": 1.0,
                "label": "value",
                "note": None,
            },
            "Missing extension parameters: count",
        ),
        (
            {
                "count": None,
                "path": "value.txt",
                "mode": "add",
                "scale": 1.0,
                "label": "value",
                "note": None,
            },
            "does not accept None",
        ),
        (
            {
                "count": 1,
                "path": 3,
                "mode": "add",
                "scale": 1.0,
                "label": "value",
                "note": None,
            },
            "must be a path",
        ),
        (
            {
                "count": 1,
                "path": "value.txt",
                "mode": "missing",
                "scale": 1.0,
                "label": "value",
                "note": None,
            },
            "not a valid Mode value",
        ),
        (
            {
                "count": True,
                "path": "value.txt",
                "mode": "add",
                "scale": 1.0,
                "label": "value",
                "note": None,
            },
            "must be an int",
        ),
        (
            {
                "count": 1,
                "path": "value.txt",
                "mode": "add",
                "scale": "one",
                "label": "value",
                "note": None,
            },
            "must be a number",
        ),
        (
            {
                "count": 1,
                "path": "value.txt",
                "mode": "add",
                "scale": 1.0,
                "label": 2,
                "note": None,
            },
            "must be a string",
        ),
    ],
)
def test_run_routine_rejects_invalid_parameter_values(
    tmp_path: pathlib.Path,
    parameters: dict[str, typing.Any],
    message: str,
) -> None:
    source = tmp_path / "parameters.py"
    source.write_text(
        "import enum\n"
        "import pathlib\n"
        "import xarray as xr\n"
        "from erlab.extensions import routine\n\n"
        "class Mode(enum.Enum):\n"
        "    ADD = 'add'\n"
        "    SUBTRACT = 'subtract'\n\n"
        "@routine()\n"
        "def calculate(\n"
        "    data: xr.DataArray,\n"
        "    count: int,\n"
        "    path: pathlib.Path,\n"
        "    mode: Mode,\n"
        "    scale: float,\n"
        "    label: str,\n"
        "    note: str | None,\n"
        ") -> xr.DataArray:\n"
        "    return data\n"
    )

    with pytest.raises(ExtensionExecutionError, match=message):
        run_routine(
            xr.DataArray([1.0]),
            script=source,
            routine_id="calculate",
            parameters=parameters,
        )


def test_coerce_call_parameters_rejects_an_unvalidated_annotation() -> None:
    def invalid(data: xr.DataArray, values: list[int]) -> xr.DataArray:
        return data

    with pytest.raises(ExtensionSignatureError, match="unsupported annotation"):
        _coerce_call_parameters(invalid, {"values": [1]})


def test_public_models_validate_persisted_values() -> None:
    with pytest.raises(ValueError, match="parameter ID cannot be empty"):
        ParameterDescriptor(
            id=" ", kind=ParameterKind.STRING, required=False, default="value"
        )
    with pytest.raises(ValueError, match=r"nested\.value"):
        extension_api._require_finite_parameter_values(
            {"nested": {"value": float("inf")}}
        )
    with pytest.raises(ValueError, match=r"nested\[1\]"):
        extension_api._require_finite_parameter_values({"nested": [0.0, float("nan")]})

    descriptor_arguments = {
        "id": " ",
        "name": "Name",
        "category": "Other",
        "summary": "",
        "function_name": "function",
    }
    with pytest.raises(ValueError, match="descriptor text cannot be empty"):
        RoutineDescriptor(**descriptor_arguments)
    with pytest.raises(ValueError, match="descriptor text cannot be empty"):
        LoaderDescriptor(**descriptor_arguments)
    valid_loader_arguments = {
        **descriptor_arguments,
        "id": "loader",
    }
    with pytest.raises(ValueError, match="contain a suffix"):
        LoaderDescriptor(**valid_loader_arguments, extensions=(".",))
    with pytest.raises(ValueError, match="must be unique"):
        LoaderDescriptor(**valid_loader_arguments, extensions=(".txt", ".txt"))


def test_loaded_script_exposes_capabilities_and_module_attributes(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "mixed.py"
    source.write_text(
        "import pathlib\n"
        "import xarray as xr\n"
        "from erlab.extensions import loader, routine\n\n"
        "VALUE = 3\n\n"
        "@routine()\n"
        "def calculate(data: xr.DataArray) -> xr.DataArray:\n"
        "    return data\n\n"
        "@loader()\n"
        "def load_data(path: pathlib.Path) -> xr.Dataset:\n"
        "    return xr.Dataset()\n"
    )

    loaded = load_script(source)

    assert [descriptor.id for descriptor in loaded.capabilities] == [
        "calculate",
        "load_data",
    ]
    assert loaded.VALUE == 3


def test_run_functions_validate_lookup_and_results(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "mixed.py"
    source.write_text(
        "import pathlib\n"
        "import typing\n"
        "import xarray as xr\n"
        "from erlab.extensions import loader, routine\n\n"
        "@routine()\n"
        "def calculate(data: xr.DataArray) -> xr.DataArray:\n"
        "    return data\n\n"
        "@loader()\n"
        "def load_data(path: pathlib.Path) -> xr.DataArray:\n"
        "    return typing.cast(xr.DataArray, path.name)\n"
    )

    with pytest.raises(TypeError, match="data must be"):
        run_routine(typing.cast("xr.DataArray", np.arange(2)), routine_id="calculate")
    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="script or both"):
        run_routine(xr.DataArray([1]), routine_id="calculate")
    with pytest.raises(
        erlab.extensions.ExtensionNotFoundError, match="Routine 'missing'"
    ):
        run_routine(xr.DataArray([1]), script=source, routine_id="missing")
    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="script or both"):
        run_loader("data.txt", loader_id="load_data")
    with pytest.raises(
        erlab.extensions.ExtensionNotFoundError, match="Loader 'missing'"
    ):
        run_loader("data.txt", script=source, loader_id="missing")
    with pytest.raises(ExtensionExecutionError, match="expected an xarray object"):
        run_loader("data.txt", script=source, loader_id="load_data")


def test_capability_resolvers_run_direct_callables_and_report_absence() -> None:
    owner = "test-public-capability-resolver"

    def resolve(
        extension_id: str,
        source_hash: str,
        kind: str,
        capability_id: str,
    ) -> typing.Callable[..., typing.Any]:
        if (extension_id, source_hash, capability_id) != ("lab", "source", "value"):
            raise KeyError
        if kind == "routine":
            return lambda data: data + 1
        return lambda path: xr.DataArray([len(path.name)])

    extension_api._set_capability_resolver(owner, resolve)
    try:
        xr.testing.assert_identical(
            run_routine(
                xr.DataArray([1]),
                extension_id="lab",
                source_hash="source",
                routine_id="value",
            ),
            xr.DataArray([2]),
        )
        result = run_loader(
            "value.txt",
            extension_id="lab",
            source_hash="source",
            loader_id="value",
        )
        xr.testing.assert_identical(result, xr.DataArray([9]))
    finally:
        extension_api._remove_resolvers(owner)

    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="No extension"):
        extension_api._resolved_source("lab", "source")
