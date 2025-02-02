"""Dialogs for the ImageToolManager."""

from __future__ import annotations

import typing
import weakref

from qtpy import QtWidgets

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool.manager import ImageToolManager


class _RenameDialog(QtWidgets.QDialog):
    def __init__(self, manager: ImageToolManager, original_names: list[str]) -> None:
        super().__init__(manager)
        self.setWindowTitle("Rename selected tools")
        self._manager = weakref.ref(manager)

        self._layout = QtWidgets.QGridLayout()
        self.setLayout(self._layout)

        self._new_name_lines: list[QtWidgets.QLineEdit] = []

        for i, name in enumerate(original_names):
            line_new = QtWidgets.QLineEdit(name)
            line_new.setPlaceholderText("New name")
            self._layout.addWidget(QtWidgets.QLabel(name), i, 0)
            self._layout.addWidget(QtWidgets.QLabel("→"), i, 1)
            self._layout.addWidget(line_new, i, 2)
            self._new_name_lines.append(line_new)

        fm = self._new_name_lines[0].fontMetrics()
        max_width = max(
            fm.boundingRect(line.text()).width() for line in self._new_name_lines
        )
        for line in self._new_name_lines:
            line.setMinimumWidth(max_width + 10)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self._layout.addWidget(button_box, len(original_names), 0, 1, 3)

    def new_names(self) -> list[str]:
        return [w.text() for w in self._new_name_lines]

    def accept(self) -> None:
        manager = self._manager()
        if manager is not None:
            for index, new_name in zip(
                manager.list_view.selected_tool_indices, self.new_names(), strict=True
            ):
                manager.rename_tool(index, new_name)
        super().accept()


class _StoreDialog(QtWidgets.QDialog):
    def __init__(self, manager: ImageToolManager, target_indices: list[int]) -> None:
        super().__init__(manager)
        self.setWindowTitle("Store with IPython")
        self._manager = weakref.ref(manager)
        self._target_indices: list[int] = target_indices

        self._layout = QtWidgets.QFormLayout()
        self.setLayout(self._layout)

        self._var_name_lines: list[QtWidgets.QLineEdit] = []

        self._layout.addRow("Data to store", QtWidgets.QLabel("Stored name"))

        for tool_idx in target_indices:
            data = manager.get_tool(tool_idx).slicer_area._data
            wrapper = manager._tool_wrappers[tool_idx]
            default_name = data.name
            if not (isinstance(default_name, str) and default_name.isidentifier()):
                if wrapper.name.isidentifier():
                    default_name = wrapper.name
                else:
                    default_name = f"data_{tool_idx}"

            line_new = QtWidgets.QLineEdit(default_name)
            line_new.setPlaceholderText("Enter variable name")
            self._layout.addRow(wrapper.label_text, line_new)
            self._var_name_lines.append(line_new)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self._layout.addRow(button_box)

    def var_name_map(self) -> dict[int, str]:
        return {
            idx: w.text()
            for idx, w in zip(self._target_indices, self._var_name_lines, strict=True)
        }

    def accept(self) -> None:
        manager = self._manager()
        if manager is not None:
            for idx, var_name in self.var_name_map().items():
                manager.console._console_widget.store_data_as(idx, var_name)
        super().accept()


class _NameFilterDialog(QtWidgets.QDialog):
    def __init__(self, parent: ImageToolManager, valid_name_filters: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Loader")

        self._valid_name_filters = valid_name_filters

        layout = QtWidgets.QVBoxLayout(self)
        self._button_group = QtWidgets.QButtonGroup(self)

        for i, name in enumerate(valid_name_filters):
            radio_button = QtWidgets.QRadioButton(name)
            self._button_group.addButton(radio_button, i)
            layout.addWidget(radio_button)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def check_filter(self, name_filter: str | None) -> None:
        self._button_group.buttons()[
            self._valid_name_filters.index(name_filter)
            if name_filter in self._valid_name_filters
            else 0
        ].setChecked(True)

    def checked_filter(self) -> str:
        return self._valid_name_filters[self._button_group.checkedId()]
