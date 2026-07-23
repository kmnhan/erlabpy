import json
import os
import pathlib
import queue
import subprocess
import sys
import textwrap
import threading
import types
import typing

import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._dialogs as manager_dialogs
from erlab.interactive.imagetool._load_source import (
    _deserialize_loader_kwargs,
    _load_code_from_file_details,
    _resolve_identified_path,
    _scan_number_load_call_args,
    _serialize_loader_kwargs,
    _spreadsheet_metadata_source_code,
    _spreadsheet_metadata_source_from_config,
)
from erlab.interactive.imagetool._provenance._execution import _load_file_source_object
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
)
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _CoordinateAttrsPickerDialog,
    _NameFilterDialog,
    _NameMapEditorDialog,
    _text_to_loader_extension_value,
)
from erlab.interactive.imagetool.manager._spreadsheet_metadata import (
    _MAPPING_NAME_COLUMN,
    _MAPPING_VALUE_ROLE,
    _column_display_entries,
    _SpreadsheetDiscoveryWorker,
    _SpreadsheetMetadataDialog,
)


def test_load_code_from_file_details_uses_erlab_io_loader_syntax(
    tmp_path: pathlib.Path,
    example_loader,
) -> None:
    file_path = tmp_path / "example.pxt"
    dataarray_selection = FileDataSelection(kind="dataarray")
    code = _load_code_from_file_details(
        file_path,
        ("merlin", {"bad-key": 1, "single": True}, dataarray_selection),
    )

    expected = (
        "erlab.io.set_loader('merlin')\n"
        f"data = erlab.io.load({str(file_path)!r}, "
        '**{"bad-key": 1, "single": True})'
    )
    assert code == expected
    assigned_code = _load_code_from_file_details(
        file_path,
        ("merlin", {"bad-key": 1, "single": True}, dataarray_selection),
        assign="source_scan",
    )
    assert assigned_code == expected.replace("data = ", "source_scan = ")
    with pytest.raises(ValueError, match="assign"):
        _load_code_from_file_details(
            file_path, ("merlin", {}, dataarray_selection), assign="bad-name"
        )

    extension_code = _load_code_from_file_details(
        file_path,
        (
            "example",
            {"loader_extensions": {"additional_coords": {"gui_extra": 7.0}}},
            dataarray_selection,
        ),
    )
    assert extension_code == (
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load({str(file_path)!r}, "
        'loader_extensions={"additional_coords": {"gui_extra": 7.0}})'
    )


def test_load_code_from_file_details_uses_stable_file_selectors(
    tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path / "scan.h5"

    dataset_code = _load_code_from_file_details(
        file_path,
        (
            xr.load_dataset,
            {"engine": "h5netcdf"},
            FileDataSelection(kind="dataset_variable", value="second"),
        ),
    )
    assert dataset_code == (
        "import xarray\n\n"
        f'data = xarray.load_dataset({str(file_path)!r}, engine="h5netcdf")'
        "['second']"
    )
    assert "_parse_input" not in dataset_code

    datatree_code = _load_code_from_file_details(
        file_path,
        (
            xr.load_datatree,
            {"engine": "h5netcdf"},
            FileDataSelection(kind="datatree_variable", value=("/diag", "image")),
        ),
    )
    assert datatree_code == (
        "import xarray\n\n"
        f'data = xarray.load_datatree({str(file_path)!r}, engine="h5netcdf")'
        "['/diag'].dataset['image']"
    )
    assert "_parse_input" not in datatree_code

    keyed_datatree_code = _load_code_from_file_details(
        file_path,
        (
            xr.load_datatree,
            {"engine": "h5netcdf"},
            FileDataSelection(
                kind="datatree_variable",
                value=("/diag", 1),
            ),
        ),
    )
    assert keyed_datatree_code is not None
    assert "['/diag'].dataset[1]" in keyed_datatree_code


def test_load_code_from_file_details_uses_public_merlin_bcs_import(
    tmp_path: pathlib.Path,
) -> None:
    from erlab.io.plugins.merlin import load_bcs

    dataarray_selection = FileDataSelection(kind="dataarray")
    file_path = tmp_path / "scan.txt"

    code = _load_code_from_file_details(file_path, (load_bcs, {}, dataarray_selection))

    assert code == (
        "import erlab.io.plugins.merlin\n\n"
        f"data = erlab.io.plugins.merlin.load_bcs({str(file_path)!r})"
    )
    assert "_merlin_bcs" not in code


def test_load_code_from_file_details_prefers_scan_number_for_erlab_loader(
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    file_path = example_data_dir / "data_002.h5"
    dataarray_selection = FileDataSelection(kind="dataarray")
    code = _load_code_from_file_details(file_path, ("example", {}, dataarray_selection))
    assert code == (
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load(2, data_dir={str(example_data_dir)!r})"
    )

    multi_file_code = _load_code_from_file_details(
        example_data_dir / "data_001_S001.h5", ("example", {}, dataarray_selection)
    )
    assert multi_file_code == (
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load(1, data_dir={str(example_data_dir)!r})"
    )

    single_file_code = _load_code_from_file_details(
        file_path, ("example", {"single": True}, dataarray_selection)
    )
    assert single_file_code == (
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load({str(file_path)!r}, single=True)"
    )

    extension_code = _load_code_from_file_details(
        file_path,
        (
            "example",
            {"loader_extensions": {"additional_coords": {"gui_extra": 7.0}}},
            dataarray_selection,
        ),
    )
    assert extension_code == (
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load(2, data_dir={str(example_data_dir)!r}, "
        'loader_extensions={"additional_coords": {"gui_extra": 7.0}})'
    )

    del example_loader
    bound_loader_code = _load_code_from_file_details(
        file_path, (erlab.io.loaders["example"].load, {}, dataarray_selection)
    )
    assert bound_loader_code == code


def test_spreadsheet_metadata_loader_kwargs_roundtrip_and_code(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    source = erlab.io.metadata.ExcelMetadataSource(
        tmp_path / "metadata.xlsx",
        sheet_name="Measurements",
        file_name_column="File\nName",
        coordinate_mapping={"Temperature\n(K)": "sample_temp"},
        attribute_mapping={"Mode": "mode"},
        overwrite=True,
        row_range=(20, 27),
    )
    serialized = _serialize_loader_kwargs({"single": True, "metadata": source})
    restored = _deserialize_loader_kwargs(json.loads(json.dumps(serialized)))

    restored_source = restored["metadata"]
    assert isinstance(restored_source, erlab.io.metadata.ExcelMetadataSource)
    assert restored_source.path == source.path
    assert restored_source.sheet_name == "Measurements"
    assert restored_source.file_name_column == "File\nName"
    assert restored_source.coordinate_mapping == {"Temperature\n(K)": "sample_temp"}
    assert restored_source.attribute_mapping == {"Mode": "mode"}
    assert restored_source.overwrite
    assert restored_source.row_range == (20, 27)

    calls: list[tuple[object, dict[str, typing.Any]]] = []
    monkeypatch.setattr(erlab.io, "set_loader", lambda _name: None)
    monkeypatch.setattr(
        erlab.io,
        "load",
        lambda identifier, **kwargs: (
            calls.append((identifier, kwargs)) or xr.DataArray([1.0], dims="x")
        ),
    )
    namespace = {"erlab": erlab}
    file_path = example_data_dir / "data_002.h5"
    for kwargs in (
        {"metadata": source},
        {"metadata": source, "file_number": 42},
    ):
        code = _load_code_from_file_details(
            file_path,
            (
                "example",
                kwargs,
                FileDataSelection(kind="dataarray"),
            ),
        )
        if code is None:
            raise RuntimeError("Spreadsheet metadata load code was not generated")
        exec(code, namespace)  # noqa: S102

    assert namespace["data"].identical(xr.DataArray([1.0], dims="x"))
    assert [identifier for identifier, _kwargs in calls] == [
        str(file_path),
        str(file_path),
    ]
    assert "file_number" not in calls[0][1]
    assert calls[1][1]["file_number"] == 42
    for _identifier, generated_kwargs in calls:
        generated_source = generated_kwargs["metadata"]
        assert isinstance(generated_source, erlab.io.metadata.ExcelMetadataSource)
        assert generated_source.coordinate_mapping == {
            "Temperature\n(K)": "sample_temp"
        }

    replay_calls: list[tuple[pathlib.Path, dict[str, typing.Any]]] = []

    class _ReplayLoader(erlab.io.dataloader.LoaderBase):
        name = "metadata_replay"
        description = "Spreadsheet metadata replay test loader"

        def load(self, path: pathlib.Path, **kwargs):
            replay_calls.append((path, kwargs))
            return xr.DataArray([2.0], dims="x")

    monkeypatch.setitem(erlab.io.loaders._loaders, "metadata_replay", _ReplayLoader())
    monkeypatch.setitem(
        erlab.io.loaders._alias_mapping, "metadata_replay", "metadata_replay"
    )
    replay_source = FileLoadSource(
        path=str(file_path),
        loader_label="Loader",
        loader_text="metadata_replay",
        kwargs_text="",
        replay_call=FileReplayCall(
            kind="erlab_loader",
            target="metadata_replay",
            kwargs=serialized,
            selection=FileDataSelection(kind="dataarray"),
        ),
    )
    replayed = _load_file_source_object(replay_source)

    assert replayed.identical(xr.DataArray([2.0], dims="x"))
    assert replay_calls[0][0] == file_path
    assert isinstance(
        replay_calls[0][1]["metadata"],
        erlab.io.metadata.ExcelMetadataSource,
    )


def test_google_spreadsheet_metadata_loader_kwargs_roundtrip_and_code() -> None:
    source = erlab.io.metadata.GoogleSheetsMetadataSource(
        "https://docs.google.com/spreadsheets/d/test-sheet/edit?usp=sharing",
        sheet_name="Measurements",
        file_name_column="File Name",
        coordinate_mapping={"Temperature": "sample_temp"},
        overwrite=True,
        row_range=(2, 10),
        timeout=2.5,
    )

    serialized = _serialize_loader_kwargs({"metadata": source})
    restored = _deserialize_loader_kwargs(json.loads(json.dumps(serialized)))[
        "metadata"
    ]

    assert isinstance(restored, erlab.io.metadata.GoogleSheetsMetadataSource)
    assert restored.share_url == source.share_url
    assert restored.sheet_name == "Measurements"
    assert restored.file_name_column == "File Name"
    assert restored.coordinate_mapping == {"Temperature": "sample_temp"}
    assert restored.overwrite
    assert restored.row_range == (2, 10)
    assert restored.timeout == 2.5

    namespace = {"erlab": erlab}
    exec(  # noqa: S102
        f"metadata_source = {_spreadsheet_metadata_source_code(source)}",
        namespace,
    )
    generated = namespace["metadata_source"]
    assert isinstance(generated, erlab.io.metadata.GoogleSheetsMetadataSource)
    assert generated.share_url == source.share_url
    assert generated.sheet_name == "Measurements"
    assert generated.coordinate_mapping == {"Temperature": "sample_temp"}
    assert generated.row_range == (2, 10)
    assert generated.timeout == 2.5


@pytest.mark.parametrize("row_range", [42, "2-3", [2]])
def test_spreadsheet_metadata_rejects_invalid_saved_row_range(
    row_range: object,
) -> None:
    with pytest.raises(ValueError, match="row range"):
        _spreadsheet_metadata_source_from_config(
            {"type": "excel", "path": "metadata.xlsx", "row_range": row_range}
        )


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        ({"type": "excel", "path": None}, TypeError, "Excel metadata path"),
        (
            {"type": "google_sheets", "share_url": None},
            TypeError,
            "Google Sheets metadata link",
        ),
        ({"type": "unsupported"}, ValueError, "Unknown spreadsheet metadata"),
    ],
)
def test_spreadsheet_metadata_rejects_invalid_saved_source_config(
    config: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _spreadsheet_metadata_source_from_config(config)


def test_custom_spreadsheet_metadata_source_is_not_serialized() -> None:
    class CustomSpreadsheetMetadataSource(erlab.io.metadata.SpreadsheetMetadataSource):
        @property
        def source_name(self) -> str:
            return "custom"

        def _read_rows(self) -> list[list[object]]:
            return [["File Name", "Temperature"]]

    source = CustomSpreadsheetMetadataSource(
        file_name_column="File Name",
        coordinate_mapping={"Temperature": "sample_temp"},
    )

    serialized = _serialize_loader_kwargs({"metadata": source})
    assert serialized["metadata"] is source
    assert _deserialize_loader_kwargs(serialized)["metadata"] is source
    with pytest.raises(TypeError, match="Cannot generate code"):
        _spreadsheet_metadata_source_code(source)


def test_spreadsheet_column_labels_preserve_exact_line_break_names() -> None:
    assert _column_display_entries(("File\nName", "File Name", "Temperature")) == [
        ("File Name (line breaks, column 1)", "File\nName"),
        ("File Name (literal spaces, column 2)", "File Name"),
        ("Temperature", "Temperature"),
    ]


def test_spreadsheet_metadata_dialog_loads_google_structure_asynchronously(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    column_requests: list[str | int | None] = []

    def get_column_names(self) -> list[str]:
        column_requests.append(self.sheet_name)
        if self.sheet_name == "Overview":
            return ["File", "Comment"]
        return ["File\nName", "Temperature\n(K)", "Mode"]

    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_sheet_names",
        lambda _self: ["Overview", "2604 NSRRC"],
    )
    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_selected_sheet_name",
        lambda _self: "2604 NSRRC",
    )
    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_column_names",
        get_column_names,
    )

    share_url = "https://docs.google.com/spreadsheets/d/test-sheet/edit?usp=sharing"
    dialog = _SpreadsheetMetadataDialog(
        None,
        erlab.io.metadata.GoogleSheetsMetadataSource(
            share_url,
            sheet_name="2604 NSRRC",
        ),
    )
    qtbot.addWidget(dialog)
    qtbot.waitUntil(
        lambda: not dialog._busy and not dialog._workers and bool(dialog._columns)
    )

    assert dialog.google_url_line.text() == share_url
    assert dialog.sheet_combo.currentData() == "2604 NSRRC"
    assert dialog.file_name_combo.currentText() == "File Name"
    assert dialog.file_name_combo.currentData() == "File\nName"

    dialog.sheet_combo.setCurrentIndex(dialog.sheet_combo.findData("Overview"))
    qtbot.waitUntil(lambda: not dialog._busy and dialog._columns == ("File", "Comment"))
    dialog.sheet_combo.setCurrentIndex(dialog.sheet_combo.findData("2604 NSRRC"))
    qtbot.waitUntil(lambda: not dialog._busy and dialog._columns[0] == "File\nName")
    assert column_requests == ["2604 NSRRC", "Overview", "2604 NSRRC"]

    dialog.add_mapping_row("Temperature\n(K)", "coordinate", "sample_temp")
    dialog.add_mapping_row("Mode", "attribute", "scan_mode")
    assert dialog.row_start_label.buddy() is dialog.row_start_spin
    assert dialog.row_end_label.buddy() is dialog.row_end_spin
    assert dialog.row_start_spin.isEnabled()
    assert dialog.row_end_spin.isEnabled()
    assert not dialog.row_start_spin.prefix()
    assert not dialog.row_end_spin.prefix()
    dialog.row_start_spin.setValue(20)
    dialog.row_end_spin.setValue(27)
    dialog.overwrite_check.setChecked(True)
    dialog.accept()

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    source = dialog.selected_source()
    assert isinstance(source, erlab.io.metadata.GoogleSheetsMetadataSource)
    assert source.sheet_name == "2604 NSRRC"
    assert source.file_name_column == "File\nName"
    assert source.coordinate_mapping == {"Temperature\n(K)": "sample_temp"}
    assert source.attribute_mapping == {"Mode": "scan_mode"}
    assert source.row_range == (20, 27)
    assert source.overwrite


def test_spreadsheet_metadata_dialog_previews_current_rows_asynchronously(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    example_loader,
) -> None:
    rows: list[list[typing.Any]] = [
        ["File", "Temperature", "Mode"],
        [16, 35.0, "cut"],
    ]
    read_count = 0

    def read_rows(_self) -> list[list[typing.Any]]:
        nonlocal read_count
        read_count += 1
        return [row.copy() for row in rows]

    monkeypatch.setattr(
        erlab.io.metadata.ExcelMetadataSource,
        "_read_rows",
        read_rows,
    )
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *args, **_kwargs: messages.append(args),
    )

    dialog = _SpreadsheetMetadataDialog(
        None,
        sample_path=tmp_path / "scan_016.h5",
        loader=erlab.io.loaders["example"],
    )
    qtbot.addWidget(dialog)
    dialog.excel_path_line.setText(str(tmp_path / "metadata.xlsx"))
    dialog.sheet_combo.blockSignals(True)
    dialog.sheet_combo.addItem("Measurements", "Measurements")
    dialog.sheet_combo.blockSignals(False)
    dialog._set_columns(("File", "Temperature", "Mode"))
    dialog.file_name_combo.setCurrentIndex(dialog.file_name_combo.findData("File"))
    dialog.add_mapping_row("Temperature", "coordinate", "sample_temp")
    dialog.add_mapping_row("Mode", "attribute", "mode")

    assert not dialog.test_match_button.isHidden()
    assert dialog.test_match_button.isEnabled()
    dialog.test_match_button.click()
    qtbot.waitUntil(lambda: not dialog._busy and not dialog._workers)

    preview = dialog._last_preview
    assert preview is not None
    assert preview.lookup == 16
    assert preview.spreadsheet_row == 2
    assert preview.values is not None
    assert preview.values.coordinate_values == {"sample_temp": 35.0}
    assert preview.values.attribute_values == {"mode": "cut"}
    assert read_count == 1

    rows[1][1] = 42.0
    dialog.test_match_button.click()
    qtbot.waitUntil(lambda: not dialog._busy and not dialog._workers)
    preview = dialog._last_preview
    assert preview is not None
    assert preview.values is not None
    assert preview.values.coordinate_values == {"sample_temp": 42.0}
    assert read_count == 2

    rows.append(["f_0015~20", 50.0, "map"])
    dialog.test_match_button.click()
    qtbot.waitUntil(lambda: not dialog._busy and not dialog._workers)
    assert dialog._last_preview is None
    assert dialog._last_preview_error is not None
    assert read_count == 3
    assert messages == []


def test_spreadsheet_metadata_dialog_ignores_result_after_cancel(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    source_returned = threading.Event()
    selected_sheet_requests: list[None] = []

    def get_sheet_names(_self) -> list[str]:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release spreadsheet discovery")
        source_returned.set()
        return ["Measurements"]

    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_sheet_names",
        get_sheet_names,
    )
    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_selected_sheet_name",
        lambda _self: selected_sheet_requests.append(None) or "Measurements",
    )
    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)
    dialog.source_type_combo.setCurrentIndex(
        dialog.source_type_combo.findData("google_sheets")
    )
    dialog.google_url_line.setText(
        "https://docs.google.com/spreadsheets/d/test-sheet/edit?usp=sharing"
    )
    dialog.refresh_button.click()
    qtbot.waitUntil(started.is_set)

    dialog.reject()
    release.set()
    qtbot.waitUntil(source_returned.is_set)
    qtbot.waitUntil(
        lambda: (
            not any(
                thread.name.startswith("SpreadsheetDiscovery-")
                for thread in threading.enumerate()
            )
        )
    )
    dialog._drain_discovery_results()

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert dialog.sheet_combo.count() == 0
    assert selected_sheet_requests == []


def test_cancelled_spreadsheet_discovery_skips_queued_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[None] = []
    result_queue: queue.SimpleQueue[tuple[_SpreadsheetDiscoveryWorker, str, object]] = (
        queue.SimpleQueue()
    )
    source = erlab.io.metadata.GoogleSheetsMetadataSource(
        "https://docs.google.com/spreadsheets/d/test-sheet/edit?usp=sharing"
    )
    monkeypatch.setattr(
        source,
        "get_sheet_names",
        lambda: requests.append(None) or ["Measurements"],
    )
    worker = _SpreadsheetDiscoveryWorker(
        1,
        "sheets",
        source,
        result_queue=result_queue,
    )
    worker.cancel()
    worker.start()

    result_worker, status, _result = result_queue.get(timeout=5)
    assert result_worker is worker
    assert status == "cancelled"
    assert requests == []


def test_spreadsheet_metadata_dialog_can_be_destroyed_during_discovery(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    source_returned = threading.Event()

    def get_sheet_names(_self) -> list[str]:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release spreadsheet discovery")
        source_returned.set()
        return ["Measurements"]

    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_sheet_names",
        get_sheet_names,
    )
    dialog = _SpreadsheetMetadataDialog(None)
    dialog.source_type_combo.setCurrentIndex(
        dialog.source_type_combo.findData("google_sheets")
    )
    dialog.google_url_line.setText(
        "https://docs.google.com/spreadsheets/d/test-sheet/edit?usp=sharing"
    )
    dialog.refresh_button.click()
    qtbot.waitUntil(started.is_set)

    try:
        dialog.reject()
        dialog.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(
            dialog,
            QtCore.QEvent.Type.DeferredDelete,
        )

        assert not erlab.interactive.utils.qt_is_valid(dialog)
        assert not source_returned.is_set()
    finally:
        release.set()

    qtbot.waitUntil(source_returned.is_set)
    qtbot.waitUntil(
        lambda: (
            not any(
                thread.name.startswith("SpreadsheetDiscovery-")
                for thread in threading.enumerate()
            )
        )
    )


def test_spreadsheet_discovery_limits_parallel_work(qtbot) -> None:
    release = threading.Event()
    started: list[int] = []
    started_lock = threading.Lock()
    result_queue: queue.SimpleQueue[tuple[_SpreadsheetDiscoveryWorker, str, object]] = (
        queue.SimpleQueue()
    )

    class Source:
        def __init__(self, index: int) -> None:
            self.index = index

        def get_sheet_names(self) -> list[str]:
            with started_lock:
                started.append(self.index)
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release spreadsheet discovery")
            return ["Measurements"]

        def get_selected_sheet_name(self) -> str:
            return "Measurements"

    workers = [
        _SpreadsheetDiscoveryWorker(
            index,
            "sheets",
            typing.cast("typing.Any", Source(index)),
            result_queue=result_queue,
        )
        for index in range(3)
    ]
    for worker in workers:
        worker.start()

    qtbot.waitUntil(lambda: len(started) == 2)
    assert len(started) == 2

    release.set()
    results = [result_queue.get(timeout=5) for _worker in workers]
    assert sorted(started) == [0, 1, 2]
    assert {status for _worker, status, _result in results} == {"succeeded"}


def test_spreadsheet_discovery_cancels_queued_work(qtbot) -> None:
    release = threading.Event()
    started: list[int] = []
    returned: list[int] = []
    state_lock = threading.Lock()
    result_queue: queue.SimpleQueue[tuple[_SpreadsheetDiscoveryWorker, str, object]] = (
        queue.SimpleQueue()
    )

    class Source:
        def __init__(self, index: int) -> None:
            self.index = index

        def get_sheet_names(self) -> list[str]:
            with state_lock:
                started.append(self.index)
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release spreadsheet discovery")
            with state_lock:
                returned.append(self.index)
            return ["Measurements"]

        def get_selected_sheet_name(self) -> str:
            return "Measurements"

    workers = [
        _SpreadsheetDiscoveryWorker(
            index,
            "sheets",
            typing.cast("typing.Any", Source(index)),
            result_queue=result_queue,
        )
        for index in range(3)
    ]
    for worker in workers:
        worker.start()

    qtbot.waitUntil(lambda: len(started) == 2)
    for worker in workers:
        worker.cancel()
    release.set()
    results = [result_queue.get(timeout=5) for _worker in workers]

    assert {status for _worker, status, _result in results} == {"cancelled"}
    assert sorted(returned) == sorted(started)
    assert len(started) == 2

    qtbot.waitUntil(
        lambda: (
            not any(
                thread.name.startswith("SpreadsheetDiscovery-")
                for thread in threading.enumerate()
            )
        )
    )


def test_spreadsheet_discovery_preview_requires_file_name() -> None:
    result_queue: queue.SimpleQueue[tuple[_SpreadsheetDiscoveryWorker, str, object]] = (
        queue.SimpleQueue()
    )
    worker = _SpreadsheetDiscoveryWorker(
        1,
        "preview",
        typing.cast("typing.Any", object()),
        result_queue=result_queue,
    )

    worker.run()

    result_worker, status, result = result_queue.get_nowait()
    assert result_worker is worker
    assert status == "failed"
    assert result == "RuntimeError: A file name is required for metadata preview"


def test_spreadsheet_discovery_does_not_delay_application_exit() -> None:
    script = textwrap.dedent(
        """
        import threading
        import queue

        from qtpy import QtCore, QtWidgets

        from erlab.interactive.imagetool.manager._spreadsheet_metadata import (
            _SpreadsheetDiscoveryWorker,
        )

        started = threading.Event()

        class Source:
            def get_sheet_names(self):
                started.set()
                threading.Event().wait(30)
                return ["Measurements"]

            def get_selected_sheet_name(self):
                return "Measurements"

        application = QtWidgets.QApplication([])
        worker = _SpreadsheetDiscoveryWorker(
            1,
            "sheets",
            Source(),
            result_queue=queue.SimpleQueue(),
        )
        worker.start()

        def quit_when_started():
            if started.is_set():
                application.quit()
            else:
                QtCore.QTimer.singleShot(1, quit_when_started)

        QtCore.QTimer.singleShot(0, quit_when_started)
        application.exec()
        print("event-loop-returned", flush=True)
        """
    )
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["QT_API"] = (
        "pyqt6" if hasattr(QtCore, "PYQT_VERSION_STR") else "pyside6"
    )
    for name in tuple(environment):
        if name.startswith(("COV_CORE_", "COVERAGE_")):
            environment.pop(name)

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=3,
    )

    assert result.returncode == 0
    assert result.stdout == "event-loop-returned\n"
    assert "RuntimeError" not in result.stderr


def test_spreadsheet_metadata_dialog_restores_excel_source(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = erlab.io.metadata.ExcelMetadataSource(
        tmp_path / "metadata.xlsx",
        sheet_name="Measurements",
        file_name_column="File\nName",
        coordinate_mapping={"Temperature": "sample_temp"},
        attribute_mapping={"Mode": "scan_mode"},
        overwrite=True,
        row_range=(20, 27),
    )
    monkeypatch.setattr(
        erlab.io.metadata.ExcelMetadataSource,
        "get_sheet_names",
        lambda _self: ["Overview", "Measurements"],
    )
    monkeypatch.setattr(
        erlab.io.metadata.ExcelMetadataSource,
        "get_selected_sheet_name",
        lambda self: typing.cast("str", self.sheet_name),
    )
    monkeypatch.setattr(
        erlab.io.metadata.ExcelMetadataSource,
        "get_column_names",
        lambda _self: ["File\nName", "Temperature", "Mode"],
    )

    dialog = _SpreadsheetMetadataDialog(None, source)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(
        lambda: not dialog._busy and not dialog._workers and bool(dialog._columns)
    )

    assert dialog.source_type_combo.currentData() == "excel"
    assert dialog.excel_path_line.text() == str(source.path)
    assert dialog.sheet_combo.currentData() == "Measurements"
    assert dialog.file_name_combo.currentData() == "File\nName"
    assert dialog.row_range() == (20, 27)
    assert dialog.overwrite_check.isChecked()
    assert dialog.mapping_table.topLevelItemCount() == 2

    dialog.mapping_table.setCurrentItem(dialog.mapping_table.topLevelItem(0))
    dialog.remove_mapping_button.click()
    assert dialog.mapping_table.topLevelItemCount() == 1


def test_spreadsheet_metadata_dialog_replaces_missing_excel_without_losing_settings(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    old_path = tmp_path / "missing.xlsx"
    new_path = tmp_path / "replacement.xlsx"
    source = erlab.io.metadata.ExcelMetadataSource(
        old_path,
        sheet_name="Measurements",
        file_name_column="File",
        coordinate_mapping={"Temperature": "sample_temp"},
        attribute_mapping={"Mode": "scan_mode"},
        overwrite=True,
        row_range=(20, 27),
    )
    sheet_requests: list[pathlib.Path] = []

    def get_sheet_names(self) -> list[str]:
        sheet_requests.append(self.path)
        return ["Overview", "Replacement Data"]

    monkeypatch.setattr(
        erlab.io.metadata.ExcelMetadataSource,
        "get_sheet_names",
        get_sheet_names,
    )
    monkeypatch.setattr(
        erlab.io.metadata.ExcelMetadataSource,
        "get_selected_sheet_name",
        lambda _self: (_ for _ in ()).throw(
            ValueError("the previously selected worksheet is missing")
        ),
    )
    monkeypatch.setattr(
        erlab.io.metadata.ExcelMetadataSource,
        "get_column_names",
        lambda _self: ["File", "Temperature", "Mode"],
    )

    dialog = _SpreadsheetMetadataDialog(None, source, load_on_open=False)
    qtbot.addWidget(dialog)

    assert not dialog._workers
    assert sheet_requests == []
    assert dialog.mapping_table.topLevelItemCount() == 2
    assert dialog.row_range() == (20, 27)
    assert dialog.overwrite_check.isChecked()

    dialog.excel_path_line.setText(str(new_path))
    dialog.refresh_button.click()
    qtbot.waitUntil(
        lambda: not dialog._busy and not dialog._workers and bool(dialog._columns)
    )

    assert sheet_requests == [new_path]
    assert dialog.sheet_combo.currentData() == "Overview"
    assert dialog.file_name_combo.currentData() == "File"
    assert dialog._mappings() == (
        {"Temperature": "sample_temp"},
        {"Mode": "scan_mode"},
    )

    dialog.accept()
    replacement = dialog.selected_source()
    assert isinstance(replacement, erlab.io.metadata.ExcelMetadataSource)
    assert replacement.path == new_path
    assert replacement.sheet_name == "Overview"
    assert replacement.row_range == (20, 27)
    assert replacement.overwrite


def test_spreadsheet_metadata_dialog_invalid_replacement_link_keeps_settings(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_url = "https://docs.google.com/spreadsheets/d/old-sheet/edit"
    new_url = "https://docs.google.com/spreadsheets/d/new-sheet/edit"
    source = erlab.io.metadata.GoogleSheetsMetadataSource(
        old_url,
        sheet_name="Measurements",
        file_name_column="File",
        coordinate_mapping={"Temperature": "sample_temp"},
        row_range=(20, 27),
    )
    errors: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *args, **_kwargs: errors.append(args),
    )
    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_sheet_names",
        lambda _self: ["Overview", "Measurements"],
    )
    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_selected_sheet_name",
        lambda self: typing.cast("str", self.sheet_name),
    )
    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_column_names",
        lambda _self: ["File", "Temperature"],
    )

    dialog = _SpreadsheetMetadataDialog(None, source, load_on_open=False)
    qtbot.addWidget(dialog)
    dialog.google_url_line.setText("not a Google Sheets link")
    dialog.refresh_button.click()

    assert errors == []
    assert dialog._last_discovery_error is not None
    assert dialog.mapping_table.topLevelItemCount() == 1
    assert dialog.row_range() == (20, 27)

    dialog.google_url_line.setText(new_url)
    dialog.refresh_button.click()
    qtbot.waitUntil(
        lambda: not dialog._busy and not dialog._workers and bool(dialog._columns)
    )
    assert dialog._last_discovery_error is None
    assert dialog.sheet_combo.currentData() == "Measurements"
    assert dialog._mappings() == ({"Temperature": "sample_temp"}, {})

    dialog.accept()
    replacement = dialog.selected_source()
    assert isinstance(replacement, erlab.io.metadata.GoogleSheetsMetadataSource)
    assert replacement.share_url == new_url
    assert replacement.sheet_name == "Measurements"
    assert replacement.row_range == (20, 27)


def test_spreadsheet_metadata_dialog_reports_background_read_error(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        erlab.io.metadata.GoogleSheetsMetadataSource,
        "get_sheet_names",
        lambda _self: (_ for _ in ()).throw(RuntimeError("read failed")),
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *args, **_kwargs: messages.append(args),
    )
    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)
    dialog.source_type_combo.setCurrentIndex(
        dialog.source_type_combo.findData("google_sheets")
    )
    dialog.google_url_line.setText(
        "https://docs.google.com/spreadsheets/d/test-sheet/edit?usp=sharing"
    )

    dialog.refresh_button.click()
    qtbot.waitUntil(lambda: not dialog._busy and not dialog._workers)

    assert messages == []
    assert dialog._last_discovery_error is not None
    assert dialog.sheet_combo.count() == 0
    assert not dialog.progress_bar.isVisible()


def test_spreadsheet_metadata_dialog_reports_worker_start_error(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("worker thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)
    dialog.source_type_combo.setCurrentIndex(
        dialog.source_type_combo.findData("google_sheets")
    )
    dialog.google_url_line.setText(
        "https://docs.google.com/spreadsheets/d/test-sheet/edit?usp=sharing"
    )

    dialog.refresh_button.click()

    assert not dialog._busy
    assert not dialog._workers
    assert dialog._last_discovery_error == ("RuntimeError: worker thread unavailable")
    assert not dialog._discovery_timer.isActive()


def test_spreadsheet_metadata_dialog_validates_mappings(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *args, **_kwargs: messages.append(args),
    )
    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)

    with pytest.raises(RuntimeError, match="has not been configured"):
        dialog.selected_source()

    dialog.accept()
    assert len(messages) == 1
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected

    dialog._request_sheets()
    assert len(messages) == 1
    assert dialog._last_discovery_error is not None
    dialog._sheet_changed()
    assert len(messages) == 1
    dialog.sheet_combo.blockSignals(True)
    dialog.sheet_combo.addItem("Measurements", "Measurements")
    dialog.sheet_combo.blockSignals(False)
    dialog._sheet_changed()
    assert len(messages) == 1
    assert dialog._last_discovery_error is not None

    dialog.excel_path_line.setText("metadata.xlsx")
    dialog._set_columns(("File", "Temperature"))
    dialog.file_name_combo.setCurrentIndex(dialog.file_name_combo.findData("File"))
    dialog.sheet_combo.clear()
    dialog.sheet_combo.blockSignals(True)
    dialog.sheet_combo.addItem("Measurements", "Measurements")
    dialog.sheet_combo.blockSignals(False)
    dialog.accept()
    assert len(messages) == 2

    dialog._busy = True
    dialog.accept()
    dialog._busy = False
    assert len(messages) == 2

    dialog.add_mapping_row("Missing", name="missing")
    with pytest.raises(ValueError, match="no available column"):
        dialog._mappings()
    dialog.remove_mapping_button.click()

    dialog.add_mapping_row("Temperature")
    with pytest.raises(ValueError, match="no destination name"):
        dialog._mappings()

    item = dialog.mapping_table.topLevelItem(0)
    assert item is not None
    item.setText(2, "sample_temp")
    dialog.add_mapping_row("Temperature", name="other_temp")
    with pytest.raises(ValueError, match="mapped more than once"):
        dialog._mappings()
    duplicate_item = dialog.mapping_table.topLevelItem(1)
    assert duplicate_item is not None
    duplicate_item.setData(
        0,
        _MAPPING_VALUE_ROLE,
        "File",
    )
    duplicate_item.setText(2, "sample_temp")
    with pytest.raises(
        ValueError, match=r"Destination name 'sample_temp'.*rows 1 and 2"
    ):
        dialog._mappings()
    dialog.accept()
    assert len(messages) == 3

    dialog.remove_mapping_button.click()
    assert dialog.mapping_table.topLevelItemCount() == 1


def test_spreadsheet_metadata_mapping_reorder_controls_mapping_order(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)
    dialog.excel_path_line.setText("metadata.xlsx")
    dialog._set_columns(("File", "Temperature", "Mode", "Energy", "Comment"))
    dialog.file_name_combo.setCurrentIndex(dialog.file_name_combo.findData("File"))
    dialog.sheet_combo.blockSignals(True)
    dialog.sheet_combo.addItem("Measurements", "Measurements")
    dialog.sheet_combo.blockSignals(False)
    dialog.add_mapping_row("Temperature", "coordinate", "sample_temp")
    dialog.add_mapping_row("Mode", "attribute", "mode")
    dialog.add_mapping_row("Energy", "coordinate", "hv")
    dialog.add_mapping_row("Comment", "attribute", "comment")

    assert dialog.mapping_table.selectionBehavior() == (
        QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    )
    assert not dialog.mapping_table.findChildren(QtWidgets.QComboBox)
    assert (
        dialog.mapping_table.dragDropMode()
        == QtWidgets.QAbstractItemView.DragDropMode.InternalMove
    )
    assert dialog.mapping_table.defaultDropAction() == QtCore.Qt.DropAction.MoveAction
    assert dialog.mapping_table.showDropIndicator()
    assert all(
        bool(
            typing.cast(
                "QtWidgets.QTreeWidgetItem", dialog.mapping_table.topLevelItem(row)
            ).flags()
            & QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        and not bool(
            typing.cast(
                "QtWidgets.QTreeWidgetItem", dialog.mapping_table.topLevelItem(row)
            ).flags()
            & QtCore.Qt.ItemFlag.ItemIsDropEnabled
        )
        for row in range(dialog.mapping_table.topLevelItemCount())
    )

    energy = dialog.mapping_table.takeTopLevelItem(2)
    comment = dialog.mapping_table.takeTopLevelItem(2)
    assert energy is not None
    assert comment is not None
    dialog.mapping_table.insertTopLevelItem(0, energy)
    dialog.mapping_table.insertTopLevelItem(1, comment)
    dialog.mapping_table.setCurrentItem(comment)
    dialog.mapping_table._queue_rows_reordered()
    assert [
        typing.cast(
            "QtWidgets.QTreeWidgetItem", dialog.mapping_table.topLevelItem(row)
        ).text(2)
        for row in range(dialog.mapping_table.topLevelItemCount())
    ] == ["hv", "comment", "sample_temp", "mode"]

    dialog._move_current_mapping(1)
    assert [
        typing.cast(
            "QtWidgets.QTreeWidgetItem", dialog.mapping_table.topLevelItem(row)
        ).text(2)
        for row in range(dialog.mapping_table.topLevelItemCount())
    ] == [
        "hv",
        "sample_temp",
        "comment",
        "mode",
    ]
    monkeypatch.setattr(QtWidgets.QMenu, "popup", lambda *_args: None)
    current_item = dialog.mapping_table.currentItem()
    assert current_item is not None
    dialog._show_mapping_context_menu(
        dialog.mapping_table.visualItemRect(current_item).center()
    )
    context_menu = dialog._mapping_context_menu
    assert context_menu is not None
    actions = {action.objectName(): action for action in context_menu.actions()}
    assert actions["spreadsheet_metadata_move_mapping_up"].isEnabled()
    assert actions["spreadsheet_metadata_move_mapping_down"].isEnabled()
    actions["spreadsheet_metadata_move_mapping_up"].trigger()

    dialog.accept()
    source = dialog.selected_source()
    assert list(source.coordinate_mapping) == ["Energy", "Temperature"]
    assert list(source.attribute_mapping) == ["Comment", "Mode"]


def test_spreadsheet_metadata_mapping_reorder_defensive_paths(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        dialog.mapping_table,
        "_show_active_combo_popup",
        lambda: None,
    )
    dialog._move_current_mapping(1)
    dialog._set_columns(("Temperature", "Mode"))
    dialog.add_mapping_row("Temperature", name="sample_temp")
    item = dialog.mapping_table.topLevelItem(0)
    assert item is not None
    dialog.add_mapping_row()
    assert all(
        dialog.mapping_table.itemWidget(item, column) is None for column in range(3)
    )
    assert not dialog.mapping_table.findChildren(QtWidgets.QComboBox)

    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    dialog.mapping_table.setCurrentIndex(dialog.mapping_table.indexFromItem(item, 0))
    dialog.mapping_table.setFocus()
    qtbot.keyClick(dialog.mapping_table, QtCore.Qt.Key.Key_F2)
    qtbot.waitUntil(
        lambda: bool(dialog.mapping_table.findChildren(QtWidgets.QComboBox))
    )
    source_editor = dialog.mapping_table.findChild(QtWidgets.QComboBox)
    assert source_editor is not None
    assert (
        dialog.mapping_table.visualItemRect(item).height()
        >= source_editor.sizeHint().height()
    )
    source_editor.setCurrentIndex(source_editor.findData("Mode"))
    source_editor.activated.emit(source_editor.currentIndex())
    propagated_return = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Return,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    dialog.mapping_table.keyPressEvent(propagated_return)
    assert propagated_return.isAccepted()
    assert item.text(0) == "Mode"
    qtbot.waitUntil(
        lambda: (
            dialog.mapping_table.currentIndex().column() == 1
            and any(
                editor.isVisible()
                for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
            )
        )
    )
    kind_editor = next(
        editor
        for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        if editor.isVisible()
    )
    kind_editor.setCurrentIndex(kind_editor.findData("attribute"))
    kind_editor.activated.emit(kind_editor.currentIndex())
    assert item.text(1) == "Attr"
    qtbot.waitUntil(
        lambda: (
            dialog.mapping_table.currentIndex().column() == 2
            and any(
                editor.isVisible()
                for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
            )
        )
    )
    name_editor = next(
        editor
        for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        if editor.isVisible()
    )
    assert name_editor.isEditable()
    name_editor.setEditText("new_name")
    line_edit = name_editor.lineEdit()
    assert line_edit is not None
    qtbot.keyClick(line_edit, QtCore.Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: item.text(2) == "new_name")
    qtbot.waitUntil(
        lambda: (
            dialog.mapping_table.currentIndex().row() == 1
            and dialog.mapping_table.currentIndex().column() == 0
            and any(
                editor.isVisible()
                for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
            )
        )
    )
    active_editor = next(
        editor
        for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        if editor.isVisible()
    )
    active_editor.hidePopup()
    qtbot.mouseClick(
        dialog.remove_mapping_button,
        QtCore.Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(lambda: dialog.mapping_table.topLevelItemCount() == 1)
    qtbot.waitUntil(lambda: not dialog.mapping_table.findChildren(QtWidgets.QComboBox))

    dialog.mapping_table.setCurrentIndex(QtCore.QModelIndex())
    dialog._remove_selected_mapping()
    assert dialog.mapping_table.topLevelItemCount() == 1

    requested: list[QtCore.QPoint] = []
    dialog.mapping_table.context_menu_requested.disconnect(
        dialog._show_mapping_context_menu
    )
    dialog.mapping_table.context_menu_requested.connect(requested.append)
    dialog.mapping_table.keyPressEvent(None)
    menu_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Menu,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    dialog.mapping_table.keyPressEvent(menu_event)
    assert menu_event.isAccepted()
    assert len(requested) == 1


def test_spreadsheet_metadata_mapping_keyboard_cursor_navigation(qtbot) -> None:
    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)
    table = dialog.mapping_table
    no_modifiers = QtCore.Qt.KeyboardModifier.NoModifier
    action = QtWidgets.QAbstractItemView.CursorAction

    assert not table.moveCursor(action.MoveNext, no_modifiers).isValid()

    dialog.add_mapping_row("Temperature", name="sample_temp")
    dialog.add_mapping_row("Mode", "attribute", "mode")

    def check_move(
        row: int,
        column: int,
        cursor_action: QtWidgets.QAbstractItemView.CursorAction,
        expected: tuple[int, int],
    ) -> None:
        item = table.topLevelItem(row)
        assert item is not None
        table.setCurrentIndex(table.indexFromItem(item, column))
        result = table.moveCursor(cursor_action, no_modifiers)
        assert (result.row(), result.column()) == expected

    check_move(0, 0, action.MoveNext, (0, 1))
    check_move(0, 2, action.MoveNext, (1, 0))
    check_move(0, 1, action.MovePrevious, (0, 0))
    check_move(1, 0, action.MovePrevious, (0, 2))
    check_move(0, 1, action.MoveLeft, (0, 0))
    check_move(0, 0, action.MoveLeft, (0, 0))
    check_move(0, 1, action.MoveRight, (0, 2))
    check_move(0, 2, action.MoveRight, (0, 2))
    check_move(1, 1, action.MoveUp, (0, 1))
    check_move(0, 1, action.MoveUp, (0, 1))
    check_move(0, 1, action.MoveDown, (1, 1))
    check_move(1, 1, action.MoveDown, (1, 1))

    first_item = table.topLevelItem(0)
    last_item = table.topLevelItem(1)
    assert first_item is not None
    assert last_item is not None
    table.setCurrentIndex(table.indexFromItem(first_item, 0))
    table.moveCursor(action.MovePrevious, no_modifiers)
    table.setCurrentIndex(table.indexFromItem(last_item, 2))
    table.moveCursor(action.MoveNext, no_modifiers)
    table.moveCursor(action.MovePageUp, no_modifiers)


def test_spreadsheet_metadata_add_mapping_keyboard_opens_source_popup(qtbot) -> None:
    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)
    dialog._set_columns(("Comment", "Mode"))
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)

    dialog.add_mapping_button.setFocus()
    qtbot.keyClick(dialog.add_mapping_button, QtCore.Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: dialog.mapping_table.topLevelItemCount() == 1)
    qtbot.waitUntil(
        lambda: any(
            editor.isVisible()
            for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        )
    )
    editor = next(
        editor
        for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        if editor.isVisible()
    )
    assert not editor.view().isVisible()

    qtbot.keyClick(editor, QtCore.Qt.Key.Key_Return)
    qtbot.waitUntil(editor.view().isVisible)


def test_spreadsheet_metadata_mapping_uses_fixed_destination_suggestions(qtbot) -> None:
    expected = ("sample_temp", "hv", "chi", "xi", "delta", "alpha", "beta")

    def show_popup_with_arrow(editor: QtWidgets.QComboBox) -> None:
        option = QtWidgets.QStyleOptionComboBox()
        editor.initStyleOption(option)
        arrow_rect = editor.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox,
            option,
            QtWidgets.QStyle.SubControl.SC_ComboBoxArrow,
            editor,
        )
        qtbot.mouseClick(
            editor,
            QtCore.Qt.MouseButton.LeftButton,
            pos=arrow_rect.center(),
        )
        qtbot.waitUntil(editor.view().isVisible)

    dialog = _SpreadsheetMetadataDialog(None)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    dialog._set_columns(("Temperature", "Mode"))
    dialog.add_mapping_row("Temperature", "coordinate", "custom_coord")
    coordinate_item = dialog.mapping_table.currentItem()
    assert coordinate_item is not None
    dialog.mapping_table.editItem(coordinate_item, _MAPPING_NAME_COLUMN)
    qtbot.waitUntil(
        lambda: any(
            editor.isVisible()
            for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        )
    )
    editor = next(
        child
        for child in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        if child.isVisible()
    )
    assert editor.isEditable()
    coordinate_suggestions = tuple(
        editor.itemText(index) for index in range(editor.count())
    )
    assert coordinate_suggestions == expected
    assert editor.currentText() == "custom_coord"
    assert editor.currentIndex() == -1
    show_popup_with_arrow(editor)
    editor.hidePopup()
    line_edit = editor.lineEdit()
    assert line_edit is not None
    qtbot.keyClick(line_edit, QtCore.Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: not dialog.mapping_table.findChildren(QtWidgets.QComboBox))

    dialog.add_mapping_row("Mode", "attribute", "hv")
    attribute_item = dialog.mapping_table.currentItem()
    assert attribute_item is not None
    dialog.mapping_table.editItem(attribute_item, _MAPPING_NAME_COLUMN)
    qtbot.waitUntil(
        lambda: any(
            editor.isVisible()
            for editor in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        )
    )
    editor = next(
        child
        for child in dialog.mapping_table.findChildren(QtWidgets.QComboBox)
        if child.isVisible()
    )
    assert editor.isEditable()
    attribute_suggestions = tuple(
        editor.itemText(index) for index in range(editor.count())
    )
    assert attribute_suggestions == expected
    assert editor.currentText() == "hv"
    assert editor.currentIndex() == editor.findText("hv")
    show_popup_with_arrow(editor)


@pytest.mark.parametrize("accept_file", [False, True])
def test_spreadsheet_metadata_excel_browse_accept_and_cancel(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    accept_file: bool,
) -> None:
    selected_path = tmp_path / "metadata.xlsx"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(
            lambda *_args, **_kwargs: (
                (str(selected_path), "Excel Workbooks (*.xlsx *.xlsm)")
                if accept_file
                else ("", "")
            )
        ),
    )
    dialog = _SpreadsheetMetadataDialog(None, initial_directory=tmp_path)
    qtbot.addWidget(dialog)
    requests: list[None] = []
    monkeypatch.setattr(dialog, "_request_sheets", lambda: requests.append(None))

    dialog.excel_browse_button.click()

    assert dialog.excel_path_line.text() == (str(selected_path) if accept_file else "")
    assert len(requests) == int(accept_file)


@pytest.mark.parametrize("accepted", [False, True])
def test_loader_options_spreadsheet_metadata_dialog_accept_and_cancel(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    example_loader,
    accepted: bool,
) -> None:
    source = erlab.io.metadata.ExcelMetadataSource(
        tmp_path / "metadata.xlsx",
        sheet_name="Measurements",
        file_name_column="File",
        coordinate_mapping={"Temperature": "sample_temp"},
    )
    dialog_kwargs: list[dict[str, typing.Any]] = []

    class _Dialog:
        def __init__(self, *_args, **_kwargs) -> None:
            dialog_kwargs.append(_kwargs)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return (
                QtWidgets.QDialog.DialogCode.Accepted
                if accepted
                else QtWidgets.QDialog.DialogCode.Rejected
            )

        def selected_source(self):
            return source

    monkeypatch.setattr(manager_dialogs, "_SpreadsheetMetadataDialog", _Dialog)
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[tmp_path / "scan.h5"],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")

    dialog.metadata_button.click()
    assert dialog_kwargs[0]["sample_path"] == tmp_path / "scan.h5"
    assert dialog_kwargs[0]["loader"] is erlab.io.loaders["example"]
    kwargs = dialog.checked_filter()[2]
    if accepted:
        assert kwargs["metadata"] is source
        dialog.metadata_clear_button.click()
        assert "metadata" not in dialog.checked_filter()[2]
    else:
        assert "metadata" not in kwargs


def test_loader_options_restore_spreadsheet_metadata_without_raw_literal(
    qtbot,
    tmp_path: pathlib.Path,
    example_loader,
) -> None:
    source = erlab.io.metadata.ExcelMetadataSource(
        tmp_path / "metadata.xlsx",
        sheet_name="Measurements",
        file_name_column="File",
        coordinate_mapping={"Temperature": "sample_temp"},
    )
    dialog = _NameFilterDialog(
        None,
        {
            "Example Raw Data (*.h5)": (
                erlab.io.loaders["example"].load,
                {"single": True, "metadata": source},
            )
        },
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")

    assert "metadata" not in dialog.kwargs_line.text()
    assert dialog.checked_filter()[2] == {"single": True, "metadata": source}
    assert dialog.metadata_clear_button.isEnabled()


def test_loader_options_spreadsheet_summary_expands_for_wrapped_text(
    qtbot,
    tmp_path: pathlib.Path,
    example_loader,
) -> None:
    source = erlab.io.metadata.ExcelMetadataSource(
        tmp_path / "metadata.xlsx",
        sheet_name="A long acquisition sheet name that wraps",
        file_name_column="File",
    )
    name_filter = "Example Raw Data (*.h5)"
    dialog = _NameFilterDialog(
        None,
        {name_filter: (erlab.io.loaders["example"].load, {"metadata": source})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter(name_filter)
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    dialog.resize(dialog.minimumSizeHint().width(), dialog.height())
    qtbot.wait(10)

    summary = dialog.metadata_summary
    required_height = summary.heightForWidth(summary.width())
    assert required_height > summary.fontMetrics().lineSpacing()
    assert summary.height() >= required_height


def test_scan_number_load_call_args_rejects_ambiguous_loader_matches(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path / "scan_007.h5"
    file_path.touch()
    assert _resolve_identified_path("scan_007.h5", tmp_path) == file_path.resolve()
    assert _scan_number_load_call_args(file_path, "missing_loader", {}) is None
    assert _scan_number_load_call_args(file_path, "coverage_loader", {1: "bad"}) is None

    class _FakeLoader:
        infer_result: typing.Any = (7, {})
        identify_result: typing.Any = ([str(file_path)],)
        infer_error = False
        identify_error = False

        def infer_index(self, stem: str) -> typing.Any:
            assert stem == file_path.stem
            if self.infer_error:
                raise RuntimeError("infer failed")
            return self.infer_result

        def identify(
            self, scan_num: int, data_dir: pathlib.Path, **kwargs
        ) -> typing.Any:
            assert scan_num == 7
            assert data_dir == tmp_path
            if self.identify_error:
                raise RuntimeError("identify failed")
            return self.identify_result

    loader = _FakeLoader()
    monkeypatch.setattr(erlab.io, "loaders", {"coverage_loader": loader})

    loader.infer_error = True
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.infer_error = False
    loader.infer_result = (7, None)
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) == [
        "7",
        f"data_dir={str(tmp_path)!r}",
    ]

    loader.infer_result = (7, ["not", "mapping"])
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.infer_result = (7, {"data_dir": tmp_path})
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.infer_result = (7, {})
    loader.identify_error = True
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.identify_error = False
    loader.identify_result = None
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.identify_result = ([str(tmp_path / "other.h5")],)
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None


def test_loader_extension_literal_parser() -> None:
    assert _text_to_loader_extension_value("coordinate_attrs", "['theta', 'phi']") == {
        "coordinate_attrs": ["theta", "phi"]
    }
    assert _text_to_loader_extension_value("additional_coords", "{'scan': 1}") == {
        "additional_coords": {"scan": 1}
    }
    assert _text_to_loader_extension_value(
        "name_map", "{'theta': ['Theta', 'Angle']}"
    ) == {"name_map": {"theta": ["Theta", "Angle"]}}

    with pytest.raises(ValueError, match="not a valid literal"):
        _text_to_loader_extension_value("additional_coords", "dict(scan=1)")


def test_name_filter_dialog_loader_extensions_toggle_resizes(
    qtbot, example_loader
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _NameFilterDialog(
        parent,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.show()
    QtWidgets.QApplication.processEvents()

    collapsed_height = dialog.height()
    assert dialog.extensions_toggle.isVisible()
    assert "extend_loader" in dialog.extensions_toggle.toolTip()
    assert "<tt>" in dialog.extensions_toggle.toolTip()
    assert not dialog.extensions_group.isVisible()

    dialog.extensions_toggle.setChecked(True)
    QtWidgets.QApplication.processEvents()
    expanded_height = dialog.height()
    assert expanded_height > collapsed_height
    extensions_layout = typing.cast(
        "QtWidgets.QFormLayout", dialog.extensions_group.layout()
    )
    for field in dialog.loader_extension_fields.values():
        label = extensions_layout.labelForField(field)
        assert label is not None
        assert field.toolTip()
        assert "<tt>" in field.toolTip()
        assert label.toolTip() == field.toolTip()
    assert dialog.name_map_editor_button is not None
    assert (
        dialog.loader_extension_lines["name_map"].sizePolicy().horizontalPolicy()
        == QtWidgets.QSizePolicy.Policy.Ignored
    )
    assert dialog.coordinate_attrs_picker_button is not None
    assert (
        dialog.loader_extension_lines["coordinate_attrs"]
        .sizePolicy()
        .horizontalPolicy()
        == QtWidgets.QSizePolicy.Policy.Ignored
    )

    dialog.extensions_toggle.setChecked(False)
    QtWidgets.QApplication.processEvents()
    assert not dialog.extensions_group.isVisible()
    assert dialog.height() < expanded_height

    dialog.loader_extension_lines["additional_coords"].setText("{'scan': 1}")
    filter_name, _func, kwargs = dialog.checked_filter()
    assert filter_name == "Example Raw Data (*.h5)"
    assert kwargs["loader_extensions"] == {"additional_coords": {"scan": 1}}


def _tree_item_by_text(
    tree: QtWidgets.QTreeWidget, column: int, text: str
) -> QtWidgets.QTreeWidgetItem:
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        if item is not None and item.text(column) == text:
            return item
    raise AssertionError(f"Could not find tree item {text!r}")


def _table_row_by_text(table: QtWidgets.QTableWidget, column: int, text: str) -> int:
    for row in range(table.rowCount()):
        item = table.item(row, column)
        if item is not None and item.text() == text:
            return row
    raise AssertionError(f"Could not find table row {text!r}")


def _dialog_label_text(dialog: QtWidgets.QDialog) -> str:
    return "\n".join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))


def test_loader_extension_literal_helpers_handle_edge_cases() -> None:
    assert manager_dialogs._string_tuple_from_literal("coordinate_attrs", "") == ()
    with pytest.raises(TypeError, match="not a string"):
        manager_dialogs._string_tuple_from_literal("coordinate_attrs", "'theta'")
    with pytest.raises(TypeError, match=r"coordinate_attrs must be an iterable$"):
        manager_dialogs._string_tuple_from_literal("coordinate_attrs", "1")

    assert manager_dialogs._name_map_from_literal("") == {}
    with pytest.raises(TypeError, match="name_map must be a dict"):
        manager_dialogs._name_map_from_literal("['theta']")

    assert list(
        manager_dialogs._iter_name_map_pairs({"theta": ["Theta", "Angle"]})
    ) == [("theta", "Theta"), ("theta", "Angle")]
    assert manager_dialogs._name_map_from_pairs(
        [("theta", "Theta"), ("theta", "Theta"), ("theta", "Angle")]
    ) == {"theta": ["Theta", "Angle"]}
    assert manager_dialogs._name_map_literal({}) == ""
    assert manager_dialogs._coordinate_attrs_literal(()) == ""


def test_coordinate_attrs_sample_attrs_handles_loader_variants() -> None:
    class _PlainLoader:
        def __init__(self) -> None:
            self.paths: list[pathlib.Path] = []

        def load_single(self, path: pathlib.Path) -> types.SimpleNamespace:
            self.paths.append(path)
            return types.SimpleNamespace(attrs={"LensMode": "Angular"})

    plain_loader = _PlainLoader()
    plain_path = pathlib.Path("plain.h5")
    assert manager_dialogs._coordinate_attrs_sample_attrs(
        typing.cast("erlab.io.dataloader.LoaderBase", plain_loader),
        plain_path,
    ) == {"LensMode": "Angular"}
    assert plain_loader.paths == [plain_path]

    class _RetryLoader:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def load_single(
            self, _path: pathlib.Path, *, without_values: bool = False
        ) -> object:
            self.calls.append(without_values)
            if without_values:
                raise TypeError("without_values is unavailable")
            return object()

    retry_loader = _RetryLoader()
    assert (
        manager_dialogs._coordinate_attrs_sample_attrs(
            typing.cast("erlab.io.dataloader.LoaderBase", retry_loader),
            pathlib.Path("retry.h5"),
        )
        == {}
    )
    assert retry_loader.calls == [True, False]

    class _FailingLoader:
        def load_single(self, _path: pathlib.Path) -> object:
            raise TypeError("load failed")

    with pytest.raises(TypeError, match="load failed"):
        manager_dialogs._coordinate_attrs_sample_attrs(
            typing.cast("erlab.io.dataloader.LoaderBase", _FailingLoader()),
            pathlib.Path("failed.h5"),
        )


def test_name_map_editor_emits_custom_mapping_and_omits_blank_rows(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    editor = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        example_data_dir / "data_002.h5",
        {},
    )
    qtbot.addWidget(editor)

    table = editor.findChild(QtWidgets.QTableWidget)
    assert table is not None

    lens_mode_row = _table_row_by_text(table, 0, "LensMode")
    lens_mode_target = table.item(lens_mode_row, 1)
    assert lens_mode_target is not None
    assert lens_mode_target.text() == ""
    assert lens_mode_target.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
    lens_mode_target.setText("lens_mode")

    temp_row = _table_row_by_text(table, 0, "TB")
    temp_target = table.item(temp_row, 1)
    assert temp_target is not None
    assert temp_target.text() == "sample_temp"
    assert not (temp_target.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
    assert not (temp_target.flags() & QtCore.Qt.ItemFlag.ItemIsEditable)

    editor.accept()
    assert editor.selected_name_map() == {"lens_mode": "LensMode"}


def test_name_map_editor_prefills_and_preserves_unmatched_mappings(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    editor = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        example_data_dir / "data_002.h5",
        {"lens_mode": "LensMode", "legacy": "MissingRaw"},
    )
    qtbot.addWidget(editor)

    table = editor.findChild(QtWidgets.QTableWidget)
    assert table is not None
    lens_mode_row = _table_row_by_text(table, 0, "LensMode")
    lens_mode_target = table.item(lens_mode_row, 1)
    assert lens_mode_target is not None
    assert lens_mode_target.text() == "lens_mode"

    editor.accept()
    assert editor.selected_name_map() == {
        "legacy": "MissingRaw",
        "lens_mode": "LensMode",
    }


def test_name_map_editor_disabled_sample_states(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    no_sample = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        None,
        {"legacy": "MissingRaw"},
    )
    qtbot.addWidget(no_sample)
    assert no_sample.findChild(QtWidgets.QTableWidget) is None
    assert "No sample file is available." in _dialog_label_text(no_sample)
    assert no_sample.selected_name_map() == {"legacy": "MissingRaw"}

    def _raise_sample_attrs(*_args: object) -> dict[str, typing.Any]:
        raise RuntimeError("sample failed")

    monkeypatch.setattr(
        manager_dialogs,
        "_coordinate_attrs_sample_attrs",
        _raise_sample_attrs,
    )
    failed_sample = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        pathlib.Path("bad.h5"),
        {"legacy": "MissingRaw"},
    )
    qtbot.addWidget(failed_sample)
    assert failed_sample.findChild(QtWidgets.QTableWidget) is None
    failed_text = _dialog_label_text(failed_sample)
    assert "Could not inspect the selected file." in failed_text
    assert "RuntimeError: sample failed" in failed_text

    monkeypatch.setattr(
        manager_dialogs,
        "_coordinate_attrs_sample_attrs",
        lambda *_args: {},
    )
    empty_sample = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        pathlib.Path("empty.h5"),
        {"legacy": "MissingRaw"},
    )
    qtbot.addWidget(empty_sample)
    assert empty_sample.findChild(QtWidgets.QTableWidget) is None
    assert "No attributes were found in the sample." in _dialog_label_text(empty_sample)


def test_name_filter_dialog_name_map_editor_updates_literal(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    file_path = example_data_dir / "data_002.h5"
    editor_calls: list[tuple[typing.Any, ...]] = []

    class _FakeNameMapEditorDialog:
        def __init__(
            self,
            parent,
            loader,
            sample_path,
            current_name_map,
        ) -> None:
            editor_calls.append((parent, loader, sample_path, current_name_map))

        def exec(self) -> bool:
            return True

        def selected_name_map(self) -> dict[str, str]:
            return {"lens_mode": "LensMode"}

    monkeypatch.setattr(
        manager_dialogs,
        "_NameMapEditorDialog",
        _FakeNameMapEditorDialog,
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[file_path],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["name_map"].setText("{'old_name': 'Old Raw'}")

    assert dialog.name_map_editor_button is not None
    dialog.name_map_editor_button.click()

    assert editor_calls == [
        (
            dialog.options_widget,
            erlab.io.loaders["example"],
            file_path,
            {"old_name": "Old Raw"},
        )
    ]
    assert dialog.loader_extension_lines["name_map"].text() == (
        "{'lens_mode': 'LensMode'}"
    )


def test_name_filter_dialog_invalid_name_map_editor_literal_shows_error(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args: critical_calls.append(args) or 0),
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["name_map"].setText("dict(scan=1)")

    assert dialog.name_map_editor_button is not None
    dialog.name_map_editor_button.click()

    assert critical_calls
    assert critical_calls[0][1:4] == (
        "Error",
        "Invalid loader arguments.",
        "Value for 'name_map' is not a valid literal",
    )


def test_name_filter_dialog_editor_cancel_leaves_literals(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    class _CancelNameMapEditorDialog:
        def __init__(self, *_args: object) -> None:
            pass

        def exec(self) -> bool:
            return False

        def selected_name_map(self) -> dict[str, str]:
            raise AssertionError("selected_name_map must not be called")

    class _CancelCoordinateAttrsPickerDialog:
        def __init__(self, *_args: object) -> None:
            pass

        def exec(self) -> bool:
            return False

        def selected_coordinate_attrs(self) -> tuple[str, ...]:
            raise AssertionError("selected_coordinate_attrs must not be called")

    monkeypatch.setattr(
        manager_dialogs,
        "_NameMapEditorDialog",
        _CancelNameMapEditorDialog,
    )
    monkeypatch.setattr(
        manager_dialogs,
        "_CoordinateAttrsPickerDialog",
        _CancelCoordinateAttrsPickerDialog,
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[pathlib.Path("sample.h5")],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["name_map"].setText("{'old_name': 'Old Raw'}")
    dialog.loader_extension_lines["coordinate_attrs"].setText("['old_coord']")

    assert dialog.name_map_editor_button is not None
    assert dialog.coordinate_attrs_picker_button is not None
    dialog.name_map_editor_button.click()
    dialog.coordinate_attrs_picker_button.click()

    assert dialog.loader_extension_lines["name_map"].text() == "{'old_name': 'Old Raw'}"
    assert dialog.loader_extension_lines["coordinate_attrs"].text() == "['old_coord']"


def test_name_filter_dialog_editor_helpers_ignore_non_loader_functions(qtbot) -> None:
    def non_loader(*_args: object, **_kwargs: object) -> None:
        return None

    dialog = _NameFilterDialog(
        None,
        {"Plain Files (*.txt)": (non_loader, {})},
        sample_paths=[pathlib.Path("plain.txt")],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Plain Files (*.txt)")
    dialog.loader_extension_lines["name_map"].setText("{'old_name': 'Old Raw'}")
    dialog.loader_extension_lines["coordinate_attrs"].setText("['old_coord']")

    assert dialog.name_map_editor_button is not None
    assert dialog.coordinate_attrs_picker_button is not None
    dialog.name_map_editor_button.click()
    dialog.coordinate_attrs_picker_button.click()

    assert dialog.loader_extension_lines["name_map"].text() == "{'old_name': 'Old Raw'}"
    assert dialog.loader_extension_lines["coordinate_attrs"].text() == "['old_coord']"


def test_name_filter_dialog_invalid_coordinate_attrs_picker_literal_shows_error(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args: critical_calls.append(args) or 0),
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["coordinate_attrs"].setText("'LensMode'")

    assert dialog.coordinate_attrs_picker_button is not None
    dialog.coordinate_attrs_picker_button.click()

    assert critical_calls
    assert critical_calls[0][1:4] == (
        "Error",
        "Invalid loader arguments.",
        "coordinate_attrs must be an iterable, not a string",
    )


def test_coordinate_attrs_picker_shows_mapped_and_builtin_attrs(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    picker = _CoordinateAttrsPickerDialog(
        None,
        erlab.io.loaders["example"],
        example_data_dir / "data_002.h5",
        ("LensMode",),
        {},
    )
    qtbot.addWidget(picker)

    tree = picker.findChild(QtWidgets.QTreeWidget)
    assert tree is not None

    lens_mode_item = _tree_item_by_text(tree, 0, "LensMode")
    assert lens_mode_item.text(1) == ""
    assert lens_mode_item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert lens_mode_item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable

    temp_item = _tree_item_by_text(tree, 0, "TB")
    assert temp_item.text(1) == "sample_temp"
    assert temp_item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert not (temp_item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
    assert not (temp_item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable)

    picker.accept()
    assert picker.selected_coordinate_attrs() == ("LensMode",)


def test_coordinate_attrs_picker_uses_literal_name_map(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    picker = _CoordinateAttrsPickerDialog(
        None,
        erlab.io.loaders["example"],
        example_data_dir / "data_002.h5",
        (),
        {"lens_mode": "LensMode"},
    )
    qtbot.addWidget(picker)

    tree = picker.findChild(QtWidgets.QTreeWidget)
    assert tree is not None
    lens_mode_item = _tree_item_by_text(tree, 0, "LensMode")
    assert lens_mode_item.text(1) == "lens_mode"

    lens_mode_item.setCheckState(0, QtCore.Qt.CheckState.Checked)
    picker.accept()
    assert picker.selected_coordinate_attrs() == ("lens_mode",)


def test_name_filter_dialog_coordinate_attrs_picker_updates_literal(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    file_path = example_data_dir / "data_002.h5"
    picker_calls: list[tuple[typing.Any, ...]] = []

    class _FakeCoordinateAttrsPickerDialog:
        def __init__(
            self,
            parent,
            loader,
            sample_path,
            current_values,
            name_map,
        ) -> None:
            picker_calls.append((parent, loader, sample_path, current_values, name_map))

        def exec(self) -> bool:
            return True

        def selected_coordinate_attrs(self) -> tuple[str, ...]:
            return ("LensMode", "extra_coord")

    monkeypatch.setattr(
        manager_dialogs,
        "_CoordinateAttrsPickerDialog",
        _FakeCoordinateAttrsPickerDialog,
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[file_path],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["coordinate_attrs"].setText("['old_coord']")
    dialog.loader_extension_lines["name_map"].setText("{'extra_coord': 'Extra Raw'}")

    assert dialog.coordinate_attrs_picker_button is not None
    dialog.coordinate_attrs_picker_button.click()

    assert picker_calls == [
        (
            dialog.options_widget,
            erlab.io.loaders["example"],
            file_path,
            ("old_coord",),
            {"extra_coord": "Extra Raw"},
        )
    ]
    assert (
        dialog.loader_extension_lines["coordinate_attrs"].text()
        == "['LensMode', 'extra_coord']"
    )


def test_coordinate_attrs_picker_failure_leaves_literal_editor(
    qtbot,
    example_loader,
) -> None:
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[pathlib.Path("missing.h5")],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["coordinate_attrs"].setText("['LensMode']")

    picker = _CoordinateAttrsPickerDialog(
        dialog,
        erlab.io.loaders["example"],
        pathlib.Path("missing.h5"),
        ("LensMode",),
        {},
    )
    qtbot.addWidget(picker)

    assert picker.findChild(QtWidgets.QTreeWidget) is None
    assert picker.selected_coordinate_attrs() == ("LensMode",)
    assert dialog.loader_extension_lines["coordinate_attrs"].text() == "['LensMode']"


def test_coordinate_attrs_picker_disabled_sample_states(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    no_sample = _CoordinateAttrsPickerDialog(
        None,
        erlab.io.loaders["example"],
        None,
        ("legacy_coord",),
        {},
    )
    qtbot.addWidget(no_sample)
    assert no_sample.findChild(QtWidgets.QTreeWidget) is None
    assert "No sample file is available." in _dialog_label_text(no_sample)
    assert no_sample.selected_coordinate_attrs() == ("legacy_coord",)

    monkeypatch.setattr(
        manager_dialogs,
        "_coordinate_attrs_sample_attrs",
        lambda *_args: {},
    )
    empty_sample = _CoordinateAttrsPickerDialog(
        None,
        erlab.io.loaders["example"],
        pathlib.Path("empty.h5"),
        ("legacy_coord",),
        {},
    )
    qtbot.addWidget(empty_sample)
    assert empty_sample.findChild(QtWidgets.QTreeWidget) is None
    assert "No attributes were found in the sample." in _dialog_label_text(empty_sample)
    assert empty_sample.selected_coordinate_attrs() == ("legacy_coord",)


def test_name_filter_dialog_without_loader_extensions_returns_kwargs(
    qtbot,
    example_loader,
) -> None:
    loader_dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(loader_dialog)
    loader_dialog.check_filter("Example Raw Data (*.h5)")
    assert loader_dialog.checked_filter() == (
        "Example Raw Data (*.h5)",
        erlab.io.loaders["example"].load,
        {},
    )

    def non_loader(*_args, **_kwargs) -> None:
        return None

    non_loader_dialog = _NameFilterDialog(
        None,
        {"Plain Files (*.txt)": (non_loader, {"plain": True})},
    )
    qtbot.addWidget(non_loader_dialog)
    non_loader_dialog.check_filter("Plain Files (*.txt)")
    assert not non_loader_dialog.extensions_toggle.isVisible()
    assert non_loader_dialog.checked_filter() == (
        "Plain Files (*.txt)",
        non_loader,
        {"plain": True},
    )


def test_name_filter_dialog_invalid_loader_extensions_shows_error(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args: critical_calls.append(args) or 0),
    )

    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["additional_coords"].setText("dict(scan=1)")

    dialog.accept()

    assert critical_calls
    assert critical_calls[0][1:4] == (
        "Error",
        "Invalid loader arguments.",
        "Value for 'additional_coords' is not a valid literal",
    )
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected


def test_choose_from_datatree_dialog_tree_helper_branches(qtbot) -> None:
    class _FakeManager(QtWidgets.QWidget):
        next_idx = 7

    manager = _FakeManager()
    qtbot.addWidget(manager)
    tree = xr.DataTree.from_dict(
        {
            "root/imagetool": xr.Dataset(attrs={"itool_title": "Root"}),
            "root/childtools/child/tool": xr.Dataset(attrs={"tool_title": "Child"}),
        }
    )

    dialog = _ChooseFromDataTreeDialog(
        manager,
        tree,
    )
    qtbot.addWidget(dialog)

    root_item = dialog._tree_widget.topLevelItem(0)
    assert root_item is not None
    assert root_item.text(0) == "7: Root"
    child_item = root_item.child(0)
    assert child_item is not None
    assert child_item.text(0) == "Child"

    dialog._on_item_changed(root_item, 1)
    dialog._uncheck_children()
    assert child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked

    dialog._check_all()
    assert root_item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert child_item.checkState(0) == QtCore.Qt.CheckState.Checked

    child_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    dialog._on_item_changed(child_item, 0)
    assert root_item.checkState(0) == QtCore.Qt.CheckState.Unchecked

    dialog._populate_tree(typing.cast("xr.DataTree", {"bad": object()}))
    with pytest.raises(ValueError, match="supported payload"):
        dialog._node_payload(xr.DataTree())
