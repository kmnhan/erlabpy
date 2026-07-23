import pathlib
import types
import typing

import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._dialogs as manager_dialogs
import erlab.interactive.imagetool.manager._window_layout as window_layout
from erlab.interactive.imagetool._load_source import (
    _load_code_from_file_details,
    _resolve_identified_path,
    _scan_number_load_call_args,
)
from erlab.interactive.imagetool._provenance._model import FileDataSelection
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _CoordinateAttrsPickerDialog,
    _NameFilterDialog,
    _NameMapEditorDialog,
    _text_to_loader_extension_value,
)


@pytest.mark.parametrize(
    ("mode", "primary_count", "reverse", "expected_positions"),
    [
        (
            "grid_row",
            3,
            False,
            [(-100, 20), (-66, 20), (-32, 20), (-100, 51), (-66, 51)],
        ),
        (
            "grid_row",
            3,
            True,
            [(-32, 20), (-66, 20), (-100, 20), (-32, 51), (-66, 51)],
        ),
        (
            "grid_column",
            2,
            False,
            [(-100, 20), (-100, 51), (-66, 20), (-66, 51), (-32, 20)],
        ),
        (
            "grid_column",
            2,
            True,
            [(-32, 20), (-32, 51), (-66, 20), (-66, 51), (-100, 20)],
        ),
        (
            "row",
            2,
            False,
            [(-100, 20), (-79, 20), (-58, 20), (-38, 20), (-18, 20)],
        ),
        (
            "column",
            2,
            True,
            [(-100, 20), (-100, 33), (-100, 46), (-100, 58), (-100, 70)],
        ),
    ],
)
def test_window_layout_rects_follow_visual_order(
    mode,
    primary_count: int,
    reverse: bool,
    expected_positions: list[tuple[int, int]],
) -> None:
    rects = window_layout.window_layout_rects(
        QtCore.QRect(-100, 20, 101, 61),
        5,
        mode,
        primary_count,
        1,
        reverse,
    )

    assert [(rect.x(), rect.y()) for rect in rects] == expected_positions
    assert all(rect.width() > 0 and rect.height() > 0 for rect in rects)


def test_window_layout_rects_reject_impossible_spacing() -> None:
    with pytest.raises(ValueError, match="Spacing"):
        window_layout.window_layout_rects(
            QtCore.QRect(0, 0, 10, 10), 3, "row", 1, 5, False
        )
    with pytest.raises(ValueError, match="positive"):
        window_layout.window_layout_rects(
            QtCore.QRect(0, 0, 10, 10), 0, "row", 1, 0, False
        )
    with pytest.raises(ValueError, match="positive"):
        window_layout.window_layout_rects(
            QtCore.QRect(0, 0, 10, 10), 2, "row", 0, 0, False
        )
    with pytest.raises(ValueError, match="negative"):
        window_layout.window_layout_rects(
            QtCore.QRect(0, 0, 10, 10), 2, "row", 1, -1, False
        )


def test_window_layout_minimum_size_validation(qtbot) -> None:
    window = QtWidgets.QWidget()
    qtbot.addWidget(window)
    window.setMinimumSize(100, 80)

    assert not window_layout.frame_rects_fit_windows(
        [window], [QtCore.QRect(0, 0, 99, 80)]
    )
    assert window_layout.frame_rects_fit_windows(
        [window], [QtCore.QRect(0, 0, 100, 80)]
    )


def test_window_layout_dialog_uses_semantic_icon_controls(qtbot) -> None:
    def icon_image(button: QtWidgets.QToolButton) -> QtGui.QImage:
        return (
            button.icon()
            .pixmap(button.iconSize())
            .toImage()
            .convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        )

    def alpha_centroid(button: QtWidgets.QToolButton) -> tuple[float, float]:
        image = icon_image(button)
        total_alpha = weighted_x = weighted_y = 0
        for y in range(image.height()):
            for x in range(image.width()):
                alpha = image.pixelColor(x, y).alpha()
                total_alpha += alpha
                weighted_x += x * alpha
                weighted_y += y * alpha
        return weighted_x / total_alpha, weighted_y / total_alpha

    def max_channel_delta(first: QtGui.QImage, second: QtGui.QImage) -> int:
        return max(
            abs(
                first.pixelColor(x, y).getRgb()[channel]
                - second.pixelColor(x, y).getRgb()[channel]
            )
            for y in range(first.height())
            for x in range(first.width())
            for channel in range(4)
        )

    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = window_layout._WindowLayoutDialog(parent)
    qtbot.addWidget(dialog)
    dialog.set_window_count(7)

    assert dialog.primary_count == 3
    assert dialog.spacing == 0
    assert all(not button.icon().isNull() for button in dialog.layout_buttons.values())
    assert all(button.accessibleName() for button in dialog.layout_buttons.values())
    assert all(
        button.toolButtonStyle() == QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        and button.size() == QtCore.QSize(32, 32)
        for button in (
            *dialog.layout_buttons.values(),
            *dialog.direction_buttons.values(),
        )
    )
    assert {
        button.parentWidget().layout().spacing()
        for button in (
            *dialog.layout_buttons.values(),
            *dialog.direction_buttons.values(),
        )
    } == {4}
    direction_layout = dialog.direction_buttons[True].parentWidget().layout()
    assert direction_layout.indexOf(dialog.direction_buttons[True]) == 0
    assert direction_layout.indexOf(dialog.direction_buttons[False]) == 1
    assert dialog.layout().itemAt(0).layout().labelAlignment() == (
        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
    )
    assert all(
        button.property("layoutMode") == mode
        for mode, button in dialog.layout_buttons.items()
    )
    for button in dialog.layout_buttons.values():
        image = icon_image(button)
        centroid = alpha_centroid(button)
        assert centroid[0] == pytest.approx((image.width() - 1) / 2, abs=0.2)
        assert centroid[1] == pytest.approx((image.height() - 1) / 2, abs=0.2)

    dialog.layout_buttons["grid_column"].click()
    assert dialog.layout_mode == "grid_column"
    assert dialog.primary_count_spin.isEnabled()
    assert all(
        not button.icon().isNull() for button in dialog.direction_buttons.values()
    )
    assert all(button.accessibleName() for button in dialog.direction_buttons.values())
    for button in dialog.direction_buttons.values():
        image = icon_image(button)
        centroid = alpha_centroid(button)
        assert centroid[0] == pytest.approx((image.width() - 1) / 2, abs=0.2)
        assert centroid[1] == pytest.approx((image.height() - 1) / 2, abs=0.2)

    forward_icons = {
        mode: icon_image(button) for mode, button in dialog.layout_buttons.items()
    }
    dialog.direction_buttons[True].click()
    for mode, button in dialog.layout_buttons.items():
        expected = (
            forward_icons[mode]
            if mode == "column"
            else forward_icons[mode].flipped(QtCore.Qt.Orientation.Horizontal)
        )
        assert max_channel_delta(icon_image(button), expected) <= 2
        image = icon_image(button)
        centroid = alpha_centroid(button)
        assert centroid[0] == pytest.approx((image.width() - 1) / 2, abs=0.2)
        assert centroid[1] == pytest.approx((image.height() - 1) / 2, abs=0.2)
    dialog.direction_buttons[False].click()

    dialog.layout_buttons["column"].click()
    assert dialog.layout_mode == "column"
    assert not dialog.primary_count_spin.isEnabled()
    dialog.direction_buttons[True].click()
    assert dialog.reverse_order
    dialog.primary_count_spin.setValue(7)
    dialog.set_window_count(4)
    assert dialog.primary_count == 4

    old_keys = {
        mode: button.icon().cacheKey() for mode, button in dialog.layout_buttons.items()
    }
    palette = dialog.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#ff00aa"))
    dialog.setPalette(palette)
    assert any(
        button.icon().cacheKey() != old_keys[mode]
        for mode, button in dialog.layout_buttons.items()
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
