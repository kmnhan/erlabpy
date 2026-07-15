"""Legend-label help widgets for Figure Composer operation editors."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive._figurecomposer._labels import (
    label_text_help_placeholder_rows,
    label_text_tooltip,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from erlab.interactive._figurecomposer._labels import LabelPlaceholderHelpRow


_DIALOG_ATTR = "_figure_composer_legend_label_help_dialog"


class LegendLabelHelpDialog(QtWidgets.QDialog):
    """Modeless help for Figure Composer legend-label placeholder syntax."""

    def __init__(
        self,
        contexts: Sequence[Mapping[str, typing.Any]],
        *,
        item_name: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("figureComposerLegendLabelsHelpDialog")
        self.setWindowTitle("Legend Labels")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setProperty("figureComposerLegendLabelItemName", item_name)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        summary = QtWidgets.QLabel(self)
        summary.setObjectName("figureComposerLegendLabelsHelpSummary")
        summary.setWordWrap(True)
        summary.setText(
            f"Enter one label for all {item_name}s, or comma-separated labels. "
            "Use Python f-string placeholders when labels should follow coordinates "
            "or attrs."
        )
        layout.addWidget(summary)

        examples_group = QtWidgets.QGroupBox("Examples", self)
        examples_layout = QtWidgets.QVBoxLayout(examples_group)
        examples_layout.setContentsMargins(8, 6, 8, 8)
        examples_layout.setSpacing(4)
        for index, (example, meaning) in enumerate(
            (
                ("{value:g}", "current coordinate value"),
                ("{value + 1.5:.1f} K", "basic arithmetic and format specs"),
                ("{source}, {number}", "source name and one-based line number"),
                (r"$E-E_F = {eV:g}$ eV", "plain text and LaTeX around a placeholder"),
            )
        ):
            row = _example_row(example, meaning, index, examples_group)
            examples_layout.addWidget(row)
        layout.addWidget(examples_group)

        rows = label_text_help_placeholder_rows(contexts, item_name=item_name)
        if rows:
            table_group = QtWidgets.QGroupBox("Available placeholders", self)
            table_layout = QtWidgets.QVBoxLayout(table_group)
            table_layout.setContentsMargins(8, 6, 8, 8)
            table = self._placeholder_table(rows)
            table_layout.addWidget(table)
            layout.addWidget(table_group)

        note = QtWidgets.QLabel(self)
        note.setObjectName("figureComposerLegendLabelsHelpNote")
        note.setWordWrap(True)
        note.setText(
            "Names that are not valid Python identifiers are exposed as aliases, "
            "for example sample temp becomes sample_temp. Plain text and LaTeX "
            "braces are kept as typed."
        )
        layout.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            parent=self,
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(520, 420)

    def _placeholder_table(
        self, rows: Sequence[LabelPlaceholderHelpRow]
    ) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(len(rows), 3, self)
        table.setObjectName("figureComposerLegendLabelsHelpTable")
        table.setHorizontalHeaderLabels(("Placeholder", "Kind", "Meaning"))
        vertical_header = table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for row_index, row in enumerate(rows):
            placeholder = f"{{{row.placeholder}}}" if row.placeholder else ""
            self._set_table_item(table, row_index, 0, placeholder, row)
            self._set_table_item(table, row_index, 1, _display_kind(row.kind), row)
            self._set_table_item(table, row_index, 2, row.description, row)
        table.resizeColumnsToContents()
        horizontal_header = table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setStretchLastSection(True)
        table.setMaximumHeight(min(220, 34 + 24 * max(1, len(rows))))
        return table

    @staticmethod
    def _set_table_item(
        table: QtWidgets.QTableWidget,
        row_index: int,
        column_index: int,
        text: str,
        row: LabelPlaceholderHelpRow,
    ) -> None:
        item = QtWidgets.QTableWidgetItem(text)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, row.placeholder)
        item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, row.kind)
        item.setData(QtCore.Qt.ItemDataRole.UserRole + 2, row.description)
        table.setItem(row_index, column_index, item)


def legend_label_input_widget(
    edit: QtWidgets.QLineEdit,
    contexts: Sequence[Mapping[str, typing.Any]],
    *,
    item_name: str,
    button_object_name: str,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QWidget:
    """Wrap a legend-label line edit with a compact syntax help button."""
    tooltip = label_text_tooltip(contexts, item_name=item_name)
    edit.setToolTip(tooltip)
    wrapper = QtWidgets.QWidget(parent)
    wrapper.setObjectName(f"{button_object_name}Row")
    wrapper.setFocusProxy(edit)
    wrapper.setToolTip(tooltip)
    layout = QtWidgets.QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(edit, 1)

    button = QtWidgets.QToolButton(wrapper)
    button.setObjectName(button_object_name)
    button.setAccessibleName("Legend Label Help")
    button.setProperty("figureComposerLegendLabelItemName", item_name)
    button.setToolTip("Show legend label examples and coordinate or attr aliases.")
    button.setAutoRaise(True)
    button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setIcon(QtGui.QIcon.fromTheme("help-faq"))
    if button.icon().isNull():
        button.setIcon(QtGui.QIcon.fromTheme("help-about"))
    style = typing.cast("QtWidgets.QStyle", button.style())
    if button.icon().isNull():
        button.setIcon(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_TitleBarContextHelpButton
            )
        )
    if button.icon().isNull():
        button.setText("?")
    button.clicked.connect(
        lambda _checked=False, button=button: _show_legend_label_help(
            button, contexts, item_name=item_name
        )
    )
    layout.addWidget(button)
    return wrapper


def _show_legend_label_help(
    button: QtWidgets.QToolButton,
    contexts: Sequence[Mapping[str, typing.Any]],
    *,
    item_name: str,
) -> None:
    dialog = getattr(button, _DIALOG_ATTR, None)
    if isinstance(dialog, LegendLabelHelpDialog) and dialog.isVisible():
        dialog.raise_()
        dialog.activateWindow()
        return

    parent = button.window()
    dialog = LegendLabelHelpDialog(contexts, item_name=item_name, parent=parent)
    setattr(button, _DIALOG_ATTR, dialog)
    dialog.finished.connect(
        lambda _result=0, button=button: setattr(button, _DIALOG_ATTR, None)
    )
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def _display_kind(kind: str) -> str:
    if kind == "coord":
        return "coordinate"
    return kind


def _example_row(
    example: str,
    meaning: str,
    index: int,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QWidget:
    row = QtWidgets.QWidget(parent)
    row.setObjectName("figureComposerLegendLabelsHelpExample")
    row.setProperty("legend_label_example_index", index)
    row.setProperty("legend_label_example", example)
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    example_label = QtWidgets.QLabel(example, row)
    example_label.setObjectName("figureComposerLegendLabelsHelpExampleText")
    example_label.setProperty("legend_label_example", example)
    example_label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    example_label.setFont(font)
    example_label.setMinimumWidth(170)
    layout.addWidget(example_label, 0)

    meaning_label = QtWidgets.QLabel(meaning, row)
    meaning_label.setObjectName("figureComposerLegendLabelsHelpExampleMeaning")
    meaning_label.setWordWrap(True)
    meaning_label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    layout.addWidget(meaning_label, 1)
    return row
