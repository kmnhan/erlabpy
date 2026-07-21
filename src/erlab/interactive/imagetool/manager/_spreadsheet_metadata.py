"""Spreadsheet metadata configuration for ImageTool Manager file loads."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._figurecomposer._ui._reorder_list import (
    ReorderList,
    event_requests_context_menu,
)
from erlab.io.metadata import (
    ExcelMetadataSource,
    GoogleSheetsMetadataSource,
    SpreadsheetMetadataSource,
)

if typing.TYPE_CHECKING:
    import pathlib


_MAPPING_SOURCE_COLUMN = 0
_MAPPING_KIND_COLUMN = 1
_MAPPING_NAME_COLUMN = 2
_MAPPING_VALUE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


class _MappingEditDelegate(QtWidgets.QStyledItemDelegate):
    """Create combo boxes only while a mapping cell is being edited."""

    combo_accepted = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._columns: tuple[str, ...] = ()

    def set_columns(self, columns: tuple[str, ...]) -> None:
        self._columns = columns

    def createEditor(
        self,
        parent: QtWidgets.QWidget | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtWidgets.QWidget | None:
        if index.column() == _MAPPING_SOURCE_COLUMN:
            if not self._columns:
                return None
            editor = QtWidgets.QComboBox(parent)
            editor.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
            )
            editor.activated.connect(
                lambda _index, combo=editor: self._commit_combo(combo)
            )
            return editor
        if index.column() == _MAPPING_KIND_COLUMN:
            editor = QtWidgets.QComboBox(parent)
            editor.addItem("Coord", "coordinate")
            editor.addItem("Attr", "attribute")
            editor.activated.connect(
                lambda _index, combo=editor: self._commit_combo(combo)
            )
            return editor
        return super().createEditor(parent, option, index)

    def setEditorData(
        self, editor: QtWidgets.QWidget | None, index: QtCore.QModelIndex
    ) -> None:
        if isinstance(editor, QtWidgets.QComboBox):
            value = index.data(_MAPPING_VALUE_ROLE)
            if index.column() == _MAPPING_SOURCE_COLUMN:
                _set_column_combo_items(
                    editor,
                    self._columns,
                    value if isinstance(value, str) else None,
                )
            else:
                editor.setCurrentIndex(editor.findData(value))
            return
        super().setEditorData(editor, index)

    def setModelData(
        self,
        editor: QtWidgets.QWidget | None,
        model: QtCore.QAbstractItemModel | None,
        index: QtCore.QModelIndex,
    ) -> None:
        if model is None:
            return
        if isinstance(editor, QtWidgets.QComboBox):
            value = editor.currentData()
            if not isinstance(value, str):
                return
            model.setData(index, value, _MAPPING_VALUE_ROLE)
            model.setData(
                index, editor.currentText(), QtCore.Qt.ItemDataRole.DisplayRole
            )
            if index.column() == _MAPPING_SOURCE_COLUMN:
                model.setData(
                    index,
                    value.replace("\n", "\\n"),
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )
            return
        super().setModelData(editor, model, index)

    def _commit_combo(self, editor: QtWidgets.QComboBox) -> None:
        self.combo_accepted.emit()
        self.commitData.emit(editor)
        self.closeEditor.emit(editor)


class _MappingTable(ReorderList):
    """Editable mapping table with Figure Composer-style row reordering."""

    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(_MAPPING_SOURCE_COLUMN, parent)
        self.setColumnCount(3)
        self.setHeaderLabels(["Spreadsheet column", "Use as", "Name in loaded data"])
        self.setRootIsDecorated(False)
        self.setItemsExpandable(False)
        self.setIndentation(0)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.setTabKeyNavigation(True)
        self._return_edit_suppressed = False
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def suppress_return_edit(self) -> None:
        """Ignore a Return event propagated by an editor that just accepted it."""
        self._return_edit_suppressed = True
        erlab.interactive.utils.single_shot(
            self, 0, self._clear_return_edit_suppression
        )

    def _clear_return_edit_suppression(self) -> None:
        self._return_edit_suppressed = False

    def moveCursor(
        self,
        cursor_action: QtWidgets.QAbstractItemView.CursorAction,
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> QtCore.QModelIndex:
        count = self.topLevelItemCount()
        if count == 0:
            return QtCore.QModelIndex()
        current = self.currentIndex()
        row = max(0, current.row())
        column = max(0, current.column())
        last_column = self.columnCount() - 1

        if cursor_action == QtWidgets.QAbstractItemView.CursorAction.MoveNext:
            position = row * self.columnCount() + column + 1
            if position >= count * self.columnCount():
                return super().moveCursor(cursor_action, modifiers)
            row, column = divmod(position, self.columnCount())
        elif cursor_action == QtWidgets.QAbstractItemView.CursorAction.MovePrevious:
            position = row * self.columnCount() + column - 1
            if position < 0:
                return super().moveCursor(cursor_action, modifiers)
            row, column = divmod(position, self.columnCount())
        elif cursor_action == QtWidgets.QAbstractItemView.CursorAction.MoveLeft:
            column = max(0, column - 1)
        elif cursor_action == QtWidgets.QAbstractItemView.CursorAction.MoveRight:
            column = min(last_column, column + 1)
        elif cursor_action == QtWidgets.QAbstractItemView.CursorAction.MoveUp:
            row = max(0, row - 1)
        elif cursor_action == QtWidgets.QAbstractItemView.CursorAction.MoveDown:
            row = min(count - 1, row + 1)
        else:
            return super().moveCursor(cursor_action, modifiers)

        item = self.topLevelItem(row)
        return (
            QtCore.QModelIndex() if item is None else self.indexFromItem(item, column)
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if event_requests_context_menu(event):
            item = self.currentItem()
            rect = self.visualItemRect(item) if item is not None else QtCore.QRect()
            self.context_menu_requested.emit(rect.center())
            event.accept()
            return
        if self._return_edit_suppressed and event.key() in (
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
        ):
            event.accept()
            return
        if event.key() in (
            QtCore.Qt.Key.Key_F2,
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
        ):
            index = self.currentIndex()
            item = self.itemFromIndex(index)
            if item is not None:
                self.editItem(item, index.column())
                if index.column() in (
                    _MAPPING_SOURCE_COLUMN,
                    _MAPPING_KIND_COLUMN,
                ):
                    erlab.interactive.utils.single_shot(
                        self, 0, self._show_active_combo_popup
                    )
                event.accept()
                return
        super().keyPressEvent(event)

    def _show_active_combo_popup(self) -> None:
        editor = QtWidgets.QApplication.focusWidget()
        if isinstance(editor, QtWidgets.QComboBox) and self.isAncestorOf(editor):
            editor.showPopup()


class _SpreadsheetDiscoverySignals(QtCore.QObject):
    succeeded = QtCore.Signal(int, str, object)
    failed = QtCore.Signal(int, str, str)
    finished = QtCore.Signal(object)


class _SpreadsheetDiscoveryWorker(QtCore.QRunnable):
    """Read worksheet structure without blocking the GUI thread."""

    def __init__(
        self,
        request_id: int,
        operation: typing.Literal["sheets", "columns"],
        source: SpreadsheetMetadataSource,
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.signals = _SpreadsheetDiscoverySignals()
        self.request_id = request_id
        self.operation = operation
        self.source = source

    @QtCore.Slot()
    def run(self) -> None:
        try:
            if self.operation == "sheets":
                sheet_names = self.source.get_sheet_names()
                try:
                    selected_sheet: str | None = self.source.get_selected_sheet_name()
                except ValueError:
                    # Worksheet discovery must remain usable when a previously
                    # selected worksheet has been removed or renamed.
                    selected_sheet = None
                result: object = (sheet_names, selected_sheet)
            else:
                result = self.source.get_column_names()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.signals.failed.emit(self.request_id, self.operation, detail)
        else:
            self.signals.succeeded.emit(
                self.request_id,
                self.operation,
                result,
            )
        finally:
            self.signals.finished.emit(self)


def _column_display_entries(columns: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return single-line labels paired with exact spreadsheet column names."""
    labels = [column.replace("\n", " ") for column in columns]
    counts = {label: labels.count(label) for label in labels}
    entries: list[tuple[str, str]] = []
    for index, (column, label) in enumerate(zip(columns, labels, strict=True), start=1):
        if counts[label] > 1:
            qualifier = "line breaks" if "\n" in column else "literal spaces"
            label = f"{label} ({qualifier}, column {index})"
        entries.append((label, column))
    return entries


def _set_column_combo_items(
    combo: QtWidgets.QComboBox,
    columns: tuple[str, ...],
    selected: str | None,
) -> None:
    """Populate a column combo while retaining exact names as item data."""
    combo.clear()
    for label, column in _column_display_entries(columns):
        combo.addItem(label, column)
        combo.setItemData(
            combo.count() - 1,
            column.replace("\n", "\\n"),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
    if selected is not None:
        index = combo.findData(selected)
        if index < 0:
            combo.addItem(f"{selected.replace(chr(10), ' ')} (not found)", selected)
            index = combo.count() - 1
        combo.setCurrentIndex(index)
    combo.setEnabled(bool(columns))


def _column_display_label(columns: tuple[str, ...], column: str | None) -> str:
    if column is None:
        return ""
    for label, exact_column in _column_display_entries(columns):
        if exact_column == column:
            return label
    return f"{column.replace(chr(10), ' ')} (not found)"


class _SpreadsheetMetadataDialog(QtWidgets.QDialog):
    """Configure an Excel or Google Sheets metadata source for a file load."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        source: SpreadsheetMetadataSource | None = None,
        *,
        initial_directory: pathlib.Path | None = None,
        load_on_open: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spreadsheet Metadata")
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(760, 560)

        self._selected_source: SpreadsheetMetadataSource | None = source
        self._initial_directory = initial_directory
        self._load_on_open = load_on_open
        self._columns: tuple[str, ...] = ()
        self._next_mapping_row_id = 0
        self._mapping_context_menu: QtWidgets.QMenu | None = None
        self._request_id = 0
        self._busy = False
        self._google_timeout = (
            source.timeout if isinstance(source, GoogleSheetsMetadataSource) else 10.0
        )
        self._preferred_sheet: str | int | None = None
        self._workers: set[_SpreadsheetDiscoveryWorker] = set()
        self._threadpool = QtCore.QThreadPool(self)
        self._threadpool.setMaxThreadCount(2)
        self._threadpool.setExpiryTimeout(0)

        layout = QtWidgets.QVBoxLayout(self)
        self._setup_source_group(layout)
        self._setup_mapping_group(layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._populate_from_source(source)

    def _setup_source_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        source_group = QtWidgets.QGroupBox("Workbook", self)
        source_layout = QtWidgets.QFormLayout(source_group)

        self.source_type_combo = QtWidgets.QComboBox(source_group)
        self.source_type_combo.setObjectName("spreadsheet_metadata_source_type")
        self.source_type_combo.addItem("Excel workbook", "excel")
        self.source_type_combo.addItem("Google Sheets", "google_sheets")
        self.source_type_combo.currentIndexChanged.connect(self._source_type_changed)
        source_layout.addRow("Source", self.source_type_combo)

        self.source_stack = QtWidgets.QStackedWidget(source_group)
        self.source_stack.setContentsMargins(0, 0, 0, 0)

        excel_page = QtWidgets.QWidget(self.source_stack)
        excel_layout = QtWidgets.QHBoxLayout(excel_page)
        excel_layout.setContentsMargins(0, 0, 0, 0)
        self.excel_path_line = QtWidgets.QLineEdit(excel_page)
        self.excel_path_line.setObjectName("spreadsheet_metadata_excel_path")
        self.excel_path_line.setPlaceholderText("Path to an .xlsx or .xlsm workbook")
        self.excel_path_line.textChanged.connect(self._source_input_edited)
        self.excel_path_line.editingFinished.connect(self._load_sheets_if_ready)
        excel_layout.addWidget(self.excel_path_line, 1)
        self.excel_browse_button = QtWidgets.QPushButton("Browse…", excel_page)
        self.excel_browse_button.setObjectName("spreadsheet_metadata_excel_browse")
        self.excel_browse_button.clicked.connect(self._browse_excel)
        excel_layout.addWidget(self.excel_browse_button)
        self.source_stack.addWidget(excel_page)

        google_page = QtWidgets.QWidget(self.source_stack)
        google_layout = QtWidgets.QHBoxLayout(google_page)
        google_layout.setContentsMargins(0, 0, 0, 0)
        self.google_url_line = QtWidgets.QLineEdit(google_page)
        self.google_url_line.setObjectName("spreadsheet_metadata_google_url")
        self.google_url_line.setPlaceholderText("Paste a shareable Google Sheets link")
        self.google_url_line.textChanged.connect(self._source_input_edited)
        self.google_url_line.editingFinished.connect(self._load_sheets_if_ready)
        google_layout.addWidget(self.google_url_line, 1)
        self.source_stack.addWidget(google_page)
        source_layout.addRow("Location", self.source_stack)

        sheet_field = QtWidgets.QWidget(source_group)
        sheet_layout = QtWidgets.QHBoxLayout(sheet_field)
        sheet_layout.setContentsMargins(0, 0, 0, 0)
        self.sheet_combo = QtWidgets.QComboBox(sheet_field)
        self.sheet_combo.setObjectName("spreadsheet_metadata_sheet")
        self.sheet_combo.setEnabled(False)
        self.sheet_combo.currentIndexChanged.connect(self._sheet_changed)
        sheet_layout.addWidget(self.sheet_combo, 1)
        self.refresh_button = QtWidgets.QPushButton("Load Worksheets", sheet_field)
        self.refresh_button.setObjectName("spreadsheet_metadata_refresh")
        self.refresh_button.clicked.connect(self._request_sheets)
        sheet_layout.addWidget(self.refresh_button)
        source_layout.addRow("Worksheet", sheet_field)

        range_field = QtWidgets.QWidget(source_group)
        range_layout = QtWidgets.QHBoxLayout(range_field)
        range_layout.setContentsMargins(0, 0, 0, 0)
        self.row_start_label = QtWidgets.QLabel("First row", range_field)
        self.row_start_label.setObjectName("spreadsheet_metadata_first_row_label")
        range_layout.addWidget(self.row_start_label)
        self.row_start_spin = QtWidgets.QSpinBox(range_field)
        self.row_start_spin.setObjectName("spreadsheet_metadata_first_row")
        self.row_start_spin.setRange(2, 999_999_999)
        self.row_start_spin.setValue(2)
        self.row_start_label.setBuddy(self.row_start_spin)
        range_layout.addWidget(self.row_start_spin)
        self.row_end_label = QtWidgets.QLabel("Last row", range_field)
        self.row_end_label.setObjectName("spreadsheet_metadata_last_row_label")
        range_layout.addWidget(self.row_end_label)
        self.row_end_spin = QtWidgets.QSpinBox(range_field)
        self.row_end_spin.setObjectName("spreadsheet_metadata_last_row")
        self.row_end_spin.setRange(2, 999_999_999)
        self.row_end_spin.setValue(1000)
        self.row_end_label.setBuddy(self.row_end_spin)
        self.row_start_spin.valueChanged.connect(self.row_end_spin.setMinimum)
        range_layout.addWidget(self.row_end_spin)
        range_layout.addStretch()
        source_layout.addRow("Row range", range_field)

        status_field = QtWidgets.QWidget(source_group)
        status_layout = QtWidgets.QHBoxLayout(status_field)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar = QtWidgets.QProgressBar(status_field)
        self.progress_bar.setObjectName("spreadsheet_metadata_progress")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMaximumWidth(90)
        self.progress_bar.hide()
        status_layout.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel(
            "Choose a workbook to begin.", status_field
        )
        self.status_label.setObjectName("spreadsheet_metadata_status")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label, 1)
        source_layout.addRow("Status", status_field)
        layout.addWidget(source_group)

    def _setup_mapping_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        mapping_group = QtWidgets.QGroupBox("Column Mapping", self)
        mapping_layout = QtWidgets.QVBoxLayout(mapping_group)

        file_name_layout = QtWidgets.QFormLayout()
        self.file_name_combo = QtWidgets.QComboBox(mapping_group)
        self.file_name_combo.setObjectName("spreadsheet_metadata_file_name_column")
        self.file_name_combo.setEnabled(False)
        file_name_layout.addRow("File name column", self.file_name_combo)
        mapping_layout.addLayout(file_name_layout)

        self.mapping_table = _MappingTable(mapping_group)
        self.mapping_table.setObjectName("spreadsheet_metadata_mapping_table")
        self.mapping_table.context_menu_requested.connect(
            self._show_mapping_context_menu
        )
        self.mapping_delegate = _MappingEditDelegate(self.mapping_table)
        self.mapping_delegate.combo_accepted.connect(
            self.mapping_table.suppress_return_edit
        )
        self.mapping_table.setItemDelegate(self.mapping_delegate)
        header = typing.cast("QtWidgets.QHeaderView", self.mapping_table.header())
        header.setStretchLastSection(False)
        header.setSectionResizeMode(
            _MAPPING_SOURCE_COLUMN, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        header.setSectionResizeMode(
            _MAPPING_KIND_COLUMN, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        kind_size_hint = QtWidgets.QComboBox()
        kind_size_hint.addItems(["Coord", "Attr"])
        editor_size = kind_size_hint.sizeHint()
        self._mapping_row_height = editor_size.height()
        header.resizeSection(
            _MAPPING_KIND_COLUMN,
            max(
                header.sectionSizeHint(_MAPPING_KIND_COLUMN),
                editor_size.width(),
            ),
        )
        kind_size_hint.deleteLater()
        header.setSectionResizeMode(
            _MAPPING_NAME_COLUMN, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        mapping_layout.addWidget(self.mapping_table, 1)

        mapping_actions = QtWidgets.QHBoxLayout()
        self.add_mapping_button = QtWidgets.QPushButton("Add", mapping_group)
        self.add_mapping_button.setObjectName("spreadsheet_metadata_add_mapping")
        self.add_mapping_button.clicked.connect(self._add_mapping)
        mapping_actions.addWidget(self.add_mapping_button)
        self.remove_mapping_button = QtWidgets.QPushButton("Remove", mapping_group)
        self.remove_mapping_button.setObjectName("spreadsheet_metadata_remove_mapping")
        self.remove_mapping_button.clicked.connect(self._remove_selected_mapping)
        self.remove_mapping_button.setEnabled(False)
        mapping_actions.addWidget(self.remove_mapping_button)
        mapping_actions.addStretch()
        mapping_layout.addLayout(mapping_actions)
        self.mapping_table.itemSelectionChanged.connect(self._sync_mapping_controls)

        self.overwrite_check = QtWidgets.QCheckBox(
            "Overwrite existing coordinates and attributes", mapping_group
        )
        self.overwrite_check.setObjectName("spreadsheet_metadata_overwrite")
        mapping_layout.addWidget(self.overwrite_check)
        layout.addWidget(mapping_group, 1)

    def _populate_from_source(self, source: SpreadsheetMetadataSource | None) -> None:
        if isinstance(source, GoogleSheetsMetadataSource):
            self.source_type_combo.setCurrentIndex(
                self.source_type_combo.findData("google_sheets")
            )
            self.google_url_line.setText(source.share_url)
            self._preferred_sheet = source.sheet_name
        elif isinstance(source, ExcelMetadataSource):
            self.source_type_combo.setCurrentIndex(
                self.source_type_combo.findData("excel")
            )
            self.excel_path_line.setText(str(source.path))
            self._preferred_sheet = source.sheet_name

        self.file_name_combo.setProperty(
            "selectedSpreadsheetColumn",
            None if source is None else source.file_name_column,
        )
        if source is not None:
            if source.row_range is not None:
                self.row_start_spin.setValue(source.row_range[0])
                self.row_end_spin.setValue(source.row_range[1])
            self.overwrite_check.setChecked(source.overwrite)
            for column, name in source.coordinate_mapping.items():
                self.add_mapping_row(column, "coordinate", name)
            for column, name in source.attribute_mapping.items():
                self.add_mapping_row(column, "attribute", name)
            if self._load_on_open:
                erlab.interactive.utils.single_shot(self, 0, self._request_sheets)
            else:
                self._set_busy(
                    False,
                    "Could not read the spreadsheet. Replace the location or load "
                    "worksheets to retry.",
                )
        else:
            self._source_type_changed()

    @QtCore.Slot()
    def _browse_excel(self) -> None:
        start = self.excel_path_line.text().strip()
        if not start and self._initial_directory is not None:
            start = str(self._initial_directory)
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Metadata Workbook",
            start,
            "Excel Workbooks (*.xlsx *.xlsm)",
        )
        if path:
            self.excel_path_line.setText(path)
            self._request_sheets()

    @QtCore.Slot()
    def _source_type_changed(self) -> None:
        self.source_stack.setCurrentIndex(self.source_type_combo.currentIndex())
        self._source_input_edited()

    @QtCore.Slot()
    def _source_input_edited(self) -> None:
        self._request_id += 1
        self._threadpool.clear()
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        self.sheet_combo.blockSignals(False)
        self.sheet_combo.setEnabled(False)
        self._set_columns(())
        self._set_busy(False, "Load the workbook to choose a worksheet.")

    @QtCore.Slot()
    def _load_sheets_if_ready(self) -> None:
        if self._source_location_text():
            self._request_sheets()

    def _source_location_text(self) -> str:
        if self.source_type_combo.currentData() == "excel":
            return self.excel_path_line.text().strip()
        return self.google_url_line.text().strip()

    def _make_source(
        self,
        *,
        sheet_name: str | int | None = None,
        configured: bool = False,
    ) -> SpreadsheetMetadataSource:
        common: dict[str, typing.Any] = {}
        if sheet_name is not None:
            common["sheet_name"] = sheet_name
        if configured:
            coordinate_mapping, attribute_mapping = self._mappings()
            common.update(
                file_name_column=self._current_column(self.file_name_combo),
                coordinate_mapping=coordinate_mapping,
                attribute_mapping=attribute_mapping,
                overwrite=self.overwrite_check.isChecked(),
                row_range=self.row_range(),
            )
        if self.source_type_combo.currentData() == "excel":
            return ExcelMetadataSource(self.excel_path_line.text().strip(), **common)
        return GoogleSheetsMetadataSource(
            self.google_url_line.text().strip(),
            timeout=self._google_timeout,
            **common,
        )

    @QtCore.Slot()
    def _request_sheets(self) -> None:
        try:
            selected_sheet = self.sheet_combo.currentData()
            if isinstance(selected_sheet, str):
                self._preferred_sheet = selected_sheet
            # Discover the workbook independently of the previous worksheet. This
            # keeps replacement workbooks usable when their worksheet names differ.
            source = self._make_source()
        except Exception as exc:
            self._show_error(f"{type(exc).__name__}: {exc}")
            return
        self._set_columns(())
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        self.sheet_combo.blockSignals(False)
        self.sheet_combo.setEnabled(False)
        self._start_request("sheets", source, "Loading worksheets…")

    @QtCore.Slot()
    def _sheet_changed(self) -> None:
        sheet_name = self.sheet_combo.currentData()
        if not isinstance(sheet_name, str):
            return
        try:
            source = self._make_source(sheet_name=sheet_name)
        except Exception as exc:
            self._show_error(f"{type(exc).__name__}: {exc}")
            return
        self._set_columns(())
        self._start_request("columns", source, "Loading column names…")

    def _start_request(
        self,
        operation: typing.Literal["sheets", "columns"],
        source: SpreadsheetMetadataSource,
        status: str,
    ) -> None:
        self._request_id += 1
        request_id = self._request_id
        self._threadpool.clear()
        worker = _SpreadsheetDiscoveryWorker(request_id, operation, source)
        self._workers.add(worker)
        worker.signals.succeeded.connect(self._request_succeeded)
        worker.signals.failed.connect(self._request_failed)
        worker.signals.finished.connect(self._worker_finished)
        self._set_busy(True, status)
        self._threadpool.start(worker)

    @QtCore.Slot(int, str, object)
    def _request_succeeded(
        self, request_id: int, operation: str, result: object
    ) -> None:
        if request_id != self._request_id:
            return
        if operation == "sheets":
            names, selected = typing.cast("tuple[list[str], str | None]", result)
            self.sheet_combo.blockSignals(True)
            self.sheet_combo.clear()
            for name in names:
                self.sheet_combo.addItem(name.replace("\n", " "), name)
            preferred = self._preferred_sheet
            if isinstance(preferred, int) and 0 <= preferred < len(names):
                preferred = names[preferred]
            if not isinstance(preferred, str) or preferred not in names:
                preferred = selected
            selected_index = self.sheet_combo.findData(preferred)
            self.sheet_combo.setCurrentIndex(max(0, selected_index))
            self.sheet_combo.setEnabled(bool(names))
            current_sheet = self.sheet_combo.currentData()
            self._preferred_sheet = (
                current_sheet if isinstance(current_sheet, str) else None
            )
            self.sheet_combo.blockSignals(False)
            self._sheet_changed()
            return
        columns = tuple(typing.cast("list[str]", result))
        self._set_columns(columns)
        self._set_busy(False, f"Loaded {len(columns)} column names.")

    @QtCore.Slot(int, str, str)
    def _request_failed(self, request_id: int, _operation: str, detail: str) -> None:
        if request_id == self._request_id:
            self._show_error(detail)

    @QtCore.Slot(object)
    def _worker_finished(self, worker: object) -> None:
        if isinstance(worker, _SpreadsheetDiscoveryWorker):
            self._workers.discard(worker)

    def _show_error(self, detail: str) -> None:
        self._set_busy(False, "Could not read the spreadsheet.")
        erlab.interactive.utils.MessageDialog.critical(
            self,
            "Spreadsheet Metadata",
            "Could not read the spreadsheet.",
            detail,
        )

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        self.progress_bar.setVisible(busy)
        self.status_label.setText(status)
        ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(not busy)

    def _set_columns(self, columns: tuple[str, ...]) -> None:
        previous_file_name = self._current_column(self.file_name_combo)
        if previous_file_name is None:
            selected = self.file_name_combo.property("selectedSpreadsheetColumn")
            previous_file_name = selected if isinstance(selected, str) else None
        self._columns = columns
        _set_column_combo_items(self.file_name_combo, columns, previous_file_name)
        self.mapping_delegate.set_columns(columns)
        for row in range(self.mapping_table.topLevelItemCount()):
            item = self.mapping_table.topLevelItem(row)
            if item is None:
                continue
            column = item.data(_MAPPING_SOURCE_COLUMN, _MAPPING_VALUE_ROLE)
            item.setText(
                _MAPPING_SOURCE_COLUMN,
                _column_display_label(
                    columns, column if isinstance(column, str) else None
                ),
            )
            item.setToolTip(
                _MAPPING_SOURCE_COLUMN,
                column.replace("\n", "\\n") if isinstance(column, str) else "",
            )

    def row_range(self) -> tuple[int, int]:
        return self.row_start_spin.value(), self.row_end_spin.value()

    @QtCore.Slot()
    def add_mapping_row(
        self,
        column: str | None = None,
        kind: typing.Literal["coordinate", "attribute"] = "coordinate",
        name: str = "",
    ) -> None:
        self._next_mapping_row_id += 1
        item = QtWidgets.QTreeWidgetItem(
            [
                _column_display_label(self._columns, column),
                "Coord" if kind == "coordinate" else "Attr",
                name,
            ]
        )
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEditable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        item.setData(
            _MAPPING_SOURCE_COLUMN,
            QtCore.Qt.ItemDataRole.UserRole,
            f"mapping-{self._next_mapping_row_id}",
        )
        item.setData(
            _MAPPING_SOURCE_COLUMN,
            _MAPPING_VALUE_ROLE,
            column,
        )
        item.setData(
            _MAPPING_KIND_COLUMN,
            _MAPPING_VALUE_ROLE,
            kind,
        )
        item.setToolTip(
            _MAPPING_SOURCE_COLUMN,
            column.replace("\n", "\\n") if column is not None else "",
        )
        item.setSizeHint(
            _MAPPING_SOURCE_COLUMN,
            QtCore.QSize(0, self._mapping_row_height),
        )
        self.mapping_table.addTopLevelItem(item)
        self.mapping_table.setCurrentItem(item)
        item.setSelected(True)
        self._sync_mapping_controls()

    @QtCore.Slot()
    def _add_mapping(self) -> None:
        self.add_mapping_row()
        item = self.mapping_table.currentItem()
        if item is not None and self._columns:
            self.mapping_table.editItem(item, _MAPPING_SOURCE_COLUMN)

    @QtCore.Slot()
    def _sync_mapping_controls(self) -> None:
        self.remove_mapping_button.setEnabled(
            self.mapping_table.currentItem() is not None
        )

    @QtCore.Slot()
    def _remove_selected_mapping(self) -> None:
        item = self.mapping_table.currentItem()
        if item is None:
            return
        row = self.mapping_table.indexOfTopLevelItem(item)
        self.mapping_table.takeTopLevelItem(row)
        remaining = self.mapping_table.topLevelItemCount()
        if remaining:
            self.mapping_table.setCurrentItem(
                self.mapping_table.topLevelItem(min(row, remaining - 1))
            )
        self._sync_mapping_controls()

    @QtCore.Slot(QtCore.QPoint)
    def _show_mapping_context_menu(self, position: QtCore.QPoint) -> None:
        item = self.mapping_table.itemAt(position)
        if item is not None:
            self.mapping_table.setCurrentItem(item)
        current_item = self.mapping_table.currentItem()
        row = (
            -1
            if current_item is None
            else self.mapping_table.indexOfTopLevelItem(current_item)
        )
        menu = QtWidgets.QMenu("Column Mapping", self.mapping_table)
        self._mapping_context_menu = menu

        def clear_menu(_destroyed: object | None = None) -> None:
            if self._mapping_context_menu is menu:
                self._mapping_context_menu = None

        menu.destroyed.connect(clear_menu)
        menu.aboutToHide.connect(menu.deleteLater)
        for text, object_name, offset in (
            ("Move Up", "spreadsheet_metadata_move_mapping_up", -1),
            ("Move Down", "spreadsheet_metadata_move_mapping_down", 1),
        ):
            action = QtGui.QAction(text, menu)
            action.setObjectName(object_name)
            action.setEnabled(
                row >= 0 and 0 <= row + offset < self.mapping_table.topLevelItemCount()
            )
            action.triggered.connect(
                lambda _checked=False, direction=offset: self._move_current_mapping(
                    direction
                )
            )
            menu.addAction(action)
        menu.addSeparator()
        remove_action = QtGui.QAction("Remove", menu)
        remove_action.setObjectName("spreadsheet_metadata_remove_mapping_action")
        remove_action.setEnabled(row >= 0)
        remove_action.triggered.connect(self._remove_selected_mapping)
        menu.addAction(remove_action)
        viewport = typing.cast("QtWidgets.QWidget", self.mapping_table.viewport())
        menu.popup(viewport.mapToGlobal(position))

    def _move_current_mapping(self, offset: int) -> None:
        item = self.mapping_table.currentItem()
        row = -1 if item is None else self.mapping_table.indexOfTopLevelItem(item)
        target = row + offset
        if row < 0 or not 0 <= target < self.mapping_table.topLevelItemCount():
            return
        item = self.mapping_table.takeTopLevelItem(row)
        if item is None:  # pragma: no cover - guarded by the valid row above.
            return
        self.mapping_table.insertTopLevelItem(target, item)
        self.mapping_table.setCurrentItem(item)
        item.setSelected(True)

    @staticmethod
    def _current_column(combo: QtWidgets.QComboBox) -> str | None:
        value = combo.currentData()
        return value if isinstance(value, str) else None

    def _mappings(self) -> tuple[dict[str, str], dict[str, str]]:
        coordinate_mapping: dict[str, str] = {}
        attribute_mapping: dict[str, str] = {}
        source_rows: dict[tuple[str, str], int] = {}
        destination_rows: dict[tuple[str, str], int] = {}
        for row in range(self.mapping_table.topLevelItemCount()):
            item = self.mapping_table.topLevelItem(row)
            if item is None:  # pragma: no cover - bounded QTreeWidget access.
                continue
            column = item.data(_MAPPING_SOURCE_COLUMN, _MAPPING_VALUE_ROLE)
            kind = item.data(_MAPPING_KIND_COLUMN, _MAPPING_VALUE_ROLE)
            name = item.text(_MAPPING_NAME_COLUMN).strip()
            if column is None or column not in self._columns:
                raise ValueError(f"Mapping row {row + 1} has no available column")
            if kind not in ("coordinate", "attribute"):
                raise ValueError(f"Mapping row {row + 1} has no mapping type")
            if not name:
                raise ValueError(f"Mapping row {row + 1} has no destination name")
            source_key = kind, column
            if source_key in source_rows:
                raise ValueError(
                    f"Spreadsheet column {column!r} is mapped more than once as "
                    f"a {kind} (mapping rows {source_rows[source_key]} and {row + 1})"
                )
            destination_key = kind, name
            if destination_key in destination_rows:
                raise ValueError(
                    f"Destination name {name!r} is used more than once as a {kind} "
                    f"(mapping rows {destination_rows[destination_key]} and {row + 1})"
                )
            source_rows[source_key] = row + 1
            destination_rows[destination_key] = row + 1
            target = coordinate_mapping if kind == "coordinate" else attribute_mapping
            target[column] = name
        return coordinate_mapping, attribute_mapping

    def selected_source(self) -> SpreadsheetMetadataSource:
        if self._selected_source is None:
            raise RuntimeError("Spreadsheet metadata has not been configured")
        return self._selected_source

    def accept(self) -> None:
        if self._busy:
            return
        file_name_column = self._current_column(self.file_name_combo)
        if not self._columns or file_name_column not in self._columns:
            self._show_error("Choose a file-name column from the loaded worksheet.")
            return
        try:
            source = self._make_source(
                sheet_name=typing.cast("str", self.sheet_combo.currentData()),
                configured=True,
            )
        except Exception as exc:
            self._show_error(f"{type(exc).__name__}: {exc}")
            return
        if not source.coordinate_mapping and not source.attribute_mapping:
            self._show_error(
                "ValueError: Add at least one coordinate or attribute mapping"
            )
            return
        self._selected_source = source
        super().accept()

    def reject(self) -> None:
        self._request_id += 1
        self._threadpool.clear()
        super().reject()
