from __future__ import annotations

import hashlib
import json
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
    ParameterKind,
    load_entry_point,
    load_script,
    loader,
    routine,
    run_loader,
    run_routine,
)
from erlab.extensions._api import _resolve_loader_method
from erlab.extensions._entry_points import _entry_point_revision

if typing.TYPE_CHECKING:
    import importlib.machinery
    import pathlib


def _entry_point_preview_loader(
    path: typing.Any, *, scale: float = 1.0
) -> xr.DataArray:
    return xr.DataArray([float(path.read_text()) * scale])


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


def test_load_script_validates_parameters_and_exact_revision(
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
        revision=loaded.revision,
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
        load_script(script, expected_revision=loaded.revision)


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


def test_load_entry_point_exposes_a_pinned_package_routine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("lab_package")

    @routine()
    def normalize(data: xr.DataArray) -> xr.DataArray:
        return data / data.max()

    module.normalize = normalize
    load_calls: list[None] = []

    class EntryPoint:
        group = "erlab.extensions"
        name = "lab"
        value = "lab_package"
        dist = None

        @staticmethod
        def load():
            load_calls.append(None)
            return module

    class EntryPoints(tuple):
        def select(self, **parameters):
            return tuple(
                entry
                for entry in self
                if all(
                    getattr(entry, key, None) == value
                    for key, value in parameters.items()
                )
            )

    entry_point = EntryPoint()
    monkeypatch.setattr(
        extension_api.importlib.metadata,
        "entry_points",
        lambda: EntryPoints((entry_point,)),
    )
    revision = _entry_point_revision(entry_point)

    loaded = load_entry_point("erlab.extensions", "lab", expected_revision=revision)

    xr.testing.assert_identical(
        loaded.normalize(xr.DataArray([1.0, 2.0])), xr.DataArray([0.5, 1.0])
    )
    assert load_calls == [None]


def test_load_entry_point_exposes_declared_external_loader_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class PreviewLoader(erlab.io.dataloader.LoaderBase):
        name = "preview"
        extensions: typing.ClassVar[set[str]] = {".txt"}
        skip_validate = True

        @property
        def file_dialog_methods(self):
            return {
                "Preview Data (*.txt)": (
                    _entry_point_preview_loader,
                    {"scale": 2.0},
                )
            }

        def load_single(self, file_path, without_values=False):
            del without_values
            return xr.DataArray([float(file_path.read_text())])

    class EntryPoint:
        group = "erlab.io.loaders"
        name = "preview"
        value = "lab_package:PreviewLoader"
        dist = None

        @staticmethod
        def load():
            return PreviewLoader

    class EntryPoints(tuple):
        def select(self, **parameters):
            return tuple(
                entry
                for entry in self
                if all(
                    getattr(entry, key, None) == value
                    for key, value in parameters.items()
                )
            )

    entry_point = EntryPoint()
    monkeypatch.setattr(
        extension_api.importlib.metadata,
        "entry_points",
        lambda: EntryPoints((entry_point,)),
    )
    revision = _entry_point_revision(entry_point)
    path = tmp_path / "value.txt"
    path.write_text("3")

    loaded = load_entry_point("erlab.io.loaders", "preview", expected_revision=revision)
    method = loaded.resolve_loader(f"{__name__}._entry_point_preview_loader")

    xr.testing.assert_identical(method(path, scale=2.0), xr.DataArray([6.0]))
    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="not declared"):
        loaded.resolve_loader("lab_package.missing")
    with pytest.raises(
        erlab.extensions.ExtensionNotFoundError, match="does not match revision"
    ):
        load_entry_point(
            "erlab.io.loaders",
            "preview",
            expected_revision="a" * 64,
        )


def test_load_entry_point_rejects_a_preloaded_editable_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    package_path = tmp_path / "lab_package"
    package_path.mkdir()
    (package_path / "plugin.py").write_text("VALUE = 1\n")
    load_calls: list[None] = []

    class Distribution:
        metadata: typing.ClassVar[dict[str, str]] = {"Name": "preloaded-lab"}
        version = "1"

        @staticmethod
        def read_text(name: str) -> str | None:
            if name != "direct_url.json":
                return None
            return json.dumps(
                {
                    "url": tmp_path.as_uri(),
                    "dir_info": {"editable": True},
                }
            )

    class EntryPoint:
        group = "erlab.extensions"
        name = "preloaded"
        value = "lab_package.plugin:normalize"
        dist = Distribution()

        @staticmethod
        def load():
            load_calls.append(None)

    class EntryPoints(tuple):
        def select(self, **parameters):
            return tuple(
                entry
                for entry in self
                if all(
                    getattr(entry, key, None) == value
                    for key, value in parameters.items()
                )
            )

    entry_point = EntryPoint()
    monkeypatch.setattr(
        extension_api.importlib.metadata,
        "entry_points",
        lambda: EntryPoints((entry_point,)),
    )
    monkeypatch.setitem(sys.modules, "lab_package.plugin", types.ModuleType("plugin"))

    with pytest.raises(erlab.extensions.ExtensionImportError, match="Restart Python"):
        load_entry_point(
            entry_point.group,
            entry_point.name,
            expected_revision=_entry_point_revision(entry_point),
        )

    assert load_calls == []


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

    loaded = load_script(script, expected_revision=revision)

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
        revision=loaded.revision,
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


def test_loader_method_resolution_preserves_dependency_import_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "dependency_extension"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "loader.py").write_text(
        "import dependency_that_is_not_installed\n\ndef load(path):\n    return path\n"
    )
    monkeypatch.syspath_prepend(tmp_path)

    with pytest.raises(ModuleNotFoundError, match="dependency_that_is_not_installed"):
        _resolve_loader_method(
            lambda path: path,
            "dependency_extension.loader.load",
        )


def test_revision_resolver_lookup_survives_manager_removal(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "revision.py"
    source_path.write_text("value = 1\n")

    def closing_resolver(_extension_id: str, _revision: str) -> pathlib.Path:
        extension_api._remove_resolvers("first")
        raise KeyError

    extension_api._set_revision_resolver(
        "first", lambda _extension_id, _revision: source_path
    )
    extension_api._set_revision_resolver("closing", closing_resolver)
    try:
        assert extension_api._resolved_revision("lab", "revision") == source_path
    finally:
        extension_api._remove_resolvers("first")
        extension_api._remove_resolvers("closing")
