from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.ptable._inspector import ElementInspector
from erlab.interactive.ptable._shared import (
    ElementRecord,
    _adjacent_atomic_number,
    _blend_colors,
    _css_rgba,
    _element_records,
    _has_keyboard_modifier,
    _is_toggle_selection_modifier,
    _navigation_atomic_numbers,
    _parse_positive_float,
    _parse_symbol_selection_query,
    _rectangular_selection_range,
    _search_records,
    _selection_range,
    _set_background,
    _set_foreground,
    _theme_colors,
)
from erlab.interactive.ptable._table import (
    PeriodicTableView,
    PeriodicTableWidget,
    _CategoryReferenceDialog,
    _popup_font,
)

__all__ = ["PeriodicTableWindow", "ptable"]

if TYPE_CHECKING:
    from collections.abc import Callable


class _SearchLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.completion_key_handler: Callable[[QtGui.QKeyEvent], bool] | None = None
        self.focus_in_handler: Callable[[], None] | None = None

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if (
            event is not None
            and self.completion_key_handler is not None
            and self.completion_key_handler(event)
        ):
            event.accept()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event: QtGui.QFocusEvent | None) -> None:
        super().focusInEvent(event)
        if self.focus_in_handler is not None:
            self.focus_in_handler()


class _SuffixLineEdit(QtWidgets.QLineEdit):
    def __init__(self, suffix: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._suffix = suffix
        self._suffix_label = QtWidgets.QLabel(suffix, self)
        self._suffix_label.setObjectName("suffix")
        self._suffix_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._suffix_label.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.textChanged.connect(self._handle_text_changed)
        self._update_suffix_margins()
        self._update_suffix_label()

    @property
    def suffix(self) -> str:
        return self._suffix

    def setClearButtonEnabled(self, enable: bool) -> None:
        super().setClearButtonEnabled(enable)
        self._update_suffix_margins()
        self._update_suffix_label()
        self.updateGeometry()

    def sizeHint(self) -> QtCore.QSize:
        return self._stabilized_size_hint(super().sizeHint())

    def minimumSizeHint(self) -> QtCore.QSize:
        return self._stabilized_size_hint(super().minimumSizeHint())

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if event is None:
            return
        if event.type() in {
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
            QtCore.QEvent.Type.PaletteChange,
        }:
            self._update_suffix_margins()
            self._update_suffix_label()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._update_suffix_label()

    def _clear_button_reserve(self) -> int:
        clear_button = self._clear_button()
        if clear_button is None or not clear_button.isVisible():
            return 0
        return clear_button.width()

    def _max_clear_button_reserve(self) -> int:
        if not self.isClearButtonEnabled():
            return 0
        clear_button = self._clear_button()
        if clear_button is not None:
            return clear_button.sizeHint().width()
        option = QtWidgets.QStyleOptionFrame()
        option.initFrom(self)
        style = self.style()
        if style is None:
            style = QtWidgets.QApplication.style()
        if style is None:  # pragma: no cover
            return 0
        return style.sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_LineEdit,
            option,
            QtCore.QSize(),
            self,
        ).height()

    def _handle_text_changed(self, _text: str) -> None:
        self._update_suffix_margins()
        self._update_suffix_label()
        self.updateGeometry()

    def _clear_button(self) -> QtWidgets.QAbstractButton | None:
        buttons = [
            child
            for child in self.findChildren(QtWidgets.QAbstractButton)
            if child.parent() is self
        ]
        if not buttons:
            return None
        return max(buttons, key=lambda button: button.geometry().right())

    def _suffix_right_padding(self) -> int:
        return max(8, self.fontMetrics().horizontalAdvance("e"))

    def _suffix_text_gap(self) -> int:
        return max(4, self.fontMetrics().horizontalAdvance(" "))

    def _update_suffix_margins(self) -> None:
        self._suffix_label.setFont(self.font())
        self._suffix_label.adjustSize()
        margins = self.textMargins()
        right_margin = self._suffix_label.sizeHint().width() + self._suffix_text_gap()
        self.setTextMargins(
            margins.left(),
            margins.top(),
            right_margin,
            margins.bottom(),
        )

    def _update_suffix_label(self) -> None:
        self._suffix_label.setFont(self.font())
        suffix_palette = QtGui.QPalette(self._suffix_label.palette())
        suffix_palette.setColor(
            QtGui.QPalette.ColorRole.WindowText,
            self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText),
        )
        self._suffix_label.setPalette(suffix_palette)
        self._suffix_label.adjustSize()

        contents = self.contentsRect()
        suffix_size = self._suffix_label.sizeHint()
        clear_button = self._clear_button()
        if clear_button is not None and self.text() != "":
            suffix_x = (
                clear_button.geometry().left()
                - self._suffix_text_gap()
                - suffix_size.width()
            )
        else:
            suffix_x = (
                contents.right() - self._suffix_right_padding() - suffix_size.width()
            )
        suffix_y = contents.top() + (contents.height() - suffix_size.height()) // 2
        self._suffix_label.move(suffix_x, suffix_y)
        self._suffix_label.show()

    def _stabilized_size_hint(self, size: QtCore.QSize) -> QtCore.QSize:
        extra_width = self._max_clear_button_reserve() - self._clear_button_reserve()
        if extra_width > 0:
            size.setWidth(size.width() + extra_width)
        return size


class _SearchPopup(QtWidgets.QListWidget):
    completion_index_activated = QtCore.Signal(QtCore.QModelIndex)
    completion_index_highlighted = QtCore.Signal(QtCore.QModelIndex)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._max_visible_items = 10

        self.setObjectName("ptable-search-popup")
        self.setFont(_popup_font())
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        viewport = self.viewport()
        if viewport is not None:
            viewport.setMouseTracking(True)
        self.setUniformItemSizes(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.hide()

        # The popup viewport owns mouse interaction, so the view emits model indexes
        # directly and leaves "completer" semantics to the controller object.
        self.pressed.connect(self._handle_pressed_index)
        self.currentRowChanged.connect(self._emit_highlighted_index)

    def setMaxVisibleItems(self, value: int) -> None:
        self._max_visible_items = max(1, int(value))

    def maxVisibleItems(self) -> int:
        return self._max_visible_items

    def viewportEvent(self, event: QtCore.QEvent | None) -> bool:
        if event is not None:
            if event.type() == QtCore.QEvent.Type.MouseMove:
                mouse_event = event if isinstance(event, QtGui.QMouseEvent) else None
                if mouse_event is not None:
                    self._set_current_index_from_point(mouse_event.position().toPoint())
            elif event.type() == QtCore.QEvent.Type.Leave and self.currentRow() >= 0:
                self.setCurrentRow(-1)
        return super().viewportEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None:
            super().mousePressEvent(event)
            return
        point = event.position().toPoint()
        index = self.indexAt(point)
        if event.button() == QtCore.Qt.MouseButton.LeftButton and index.isValid():
            self._handle_pressed_index(index)
            event.accept()
            return
        super().mousePressEvent(event)

    def _handle_pressed_index(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        if index.row() != self.currentRow():
            self.setCurrentRow(index.row())
        self.completion_index_activated.emit(index)

    def _set_current_index_from_point(self, point: QtCore.QPoint) -> None:
        index = self.indexAt(point)
        target_row = index.row() if index.isValid() else -1
        if target_row != self.currentRow():
            self.setCurrentRow(target_row)

    def _emit_highlighted_index(self, row: int) -> None:
        if row < 0:
            self.completion_index_highlighted.emit(QtCore.QModelIndex())
            return
        model = self.model()
        if model is None:
            return
        self.completion_index_highlighted.emit(model.index(row, 0))


class _SearchCompleter(QtCore.QObject):
    activated = QtCore.Signal((str,), (QtCore.QModelIndex,))
    highlighted = QtCore.Signal((QtCore.QModelIndex,))

    def __init__(
        self,
        popup: _SearchPopup,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._popup = popup
        self._complete_handler: Callable[[], None] | None = None

        self._popup.completion_index_activated.connect(
            self.activated[QtCore.QModelIndex].emit
        )
        self._popup.completion_index_highlighted.connect(
            self.highlighted[QtCore.QModelIndex].emit
        )

    def popup(self) -> _SearchPopup:
        return self._popup

    def model(self) -> QtCore.QAbstractItemModel | None:
        return self._popup.model()

    def complete(self) -> None:
        if self._complete_handler is not None:
            self._complete_handler()

    def set_complete_handler(self, handler: Callable[[], None]) -> None:
        self._complete_handler = handler

    def setCaseSensitivity(self, _: object) -> None:
        return None

    def setCompletionMode(self, _: object) -> None:
        return None

    def setFilterMode(self, _: object) -> None:
        return None

    def setCompletionPrefix(self, _: str) -> None:
        return None

    def setMaxVisibleItems(self, value: int) -> None:
        self._popup.setMaxVisibleItems(value)

    def maxVisibleItems(self) -> int:
        return self._popup.maxVisibleItems()


@dataclass(frozen=True)
class _SearchCompletion:
    display_text: str
    atomic_numbers: tuple[int, ...]
    search_text: str


_NOTATION_VALUES = frozenset({"orbital", "iupac"})
_DEFAULT_NOTATION = "orbital"
_NOTATION_SETTINGS_KEY = "notation"
_ENERGY_RELATION_TOOLTIP = (
    "<i>E</i><sub>kin</sub> = <i>h\u03bd</i> \u2212 <i>E</i><sub>edge</sub> "
    "\u2212 \u03a6"
)
_PHOTON_ENERGY_TOOLTIP = (
    "<qt>Photon energy <i>h\u03bd</i> in eV. Kinetic energies are computed as "
    f"{_ENERGY_RELATION_TOOLTIP}.</qt>"
)
_WORKFUNCTION_TOOLTIP = (
    "<qt>Spectrometer work function \u03a6 in eV. Kinetic energies are computed as "
    f"{_ENERGY_RELATION_TOOLTIP}. Leave blank to treat \u03a6 as 0 eV.</qt>"
)


def _get_ptable_settings() -> QtCore.QSettings:
    return QtCore.QSettings(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        "erlabpy",
        "ptable",
    )


def _resolve_notation(notation: str | None) -> str:
    if notation is None:
        saved = _get_ptable_settings().value(_NOTATION_SETTINGS_KEY, _DEFAULT_NOTATION)
        if isinstance(saved, str) and saved in _NOTATION_VALUES:
            return saved
        return _DEFAULT_NOTATION
    if notation not in _NOTATION_VALUES:
        raise ValueError("notation must be either 'orbital' or 'iupac'")
    return notation


class PeriodicTableWindow(QtWidgets.QMainWindow):
    _MAX_HARMONIC = 10
    _LEGACY_MINIMUM_WINDOW_SIZE = QtCore.QSize(1180, 760)
    _LEGACY_TOP_SPLITTER_RATIO = 560 / (560 + 340)

    def __init__(
        self,
        *,
        photon_energy: float | None = None,
        workfunction: float | None = None,
        max_harmonic: int = 1,
        notation: str | None = None,
    ) -> None:
        super().__init__()
        notation = _resolve_notation(notation)
        if not 1 <= max_harmonic <= self._MAX_HARMONIC:
            raise ValueError(f"max_harmonic must be between 1 and {self._MAX_HARMONIC}")

        self._theme = _theme_colors()
        self._selected_atomic_numbers: list[int] = []
        self._current_atomic_number: int | None = None
        self._selection_anchor_atomic_number: int | None = None
        self._plot_atomic_number: int | None = None
        self._plot_target_user_selected = False
        self._hover_atomic_number: int | None = None
        self._search_matches: set[int] = set()
        self._search_completions: tuple[_SearchCompletion, ...] = ()
        self._search_completion_row: int | None = None

        self.setWindowTitle("Periodic Table of the Elements")
        self.resize(1600, 920)

        self.central = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central)
        root_layout = QtWidgets.QVBoxLayout(self.central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.header = QtWidgets.QFrame(self.central)
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        root_layout.addWidget(self.header)

        self.search_edit = _SearchLineEdit(self.header)
        self.search_edit.setObjectName("ptable-search")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setPlaceholderText("Search by symbol, name, or symbol list")
        self.search_edit.setMinimumWidth(256)
        self.search_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.search_edit.completion_key_handler = self._handle_search_key_press
        self.search_edit.focus_in_handler = self._handle_search_focus_in
        self.search_popup = _SearchPopup(self)
        self.search_completer = _SearchCompleter(self.search_popup, self)
        self.search_completer.set_complete_handler(self._show_search_popup)
        self.search_completer.setCaseSensitivity(
            QtCore.Qt.CaseSensitivity.CaseInsensitive
        )
        self.search_completer.setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.PopupCompletion
        )
        self.search_completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.search_completer.setMaxVisibleItems(10)
        self.search_completer.activated[str].connect(self._handle_search_completion)
        self.search_completer.activated[QtCore.QModelIndex].connect(
            self._handle_search_completion_index
        )
        self.search_completer.highlighted[QtCore.QModelIndex].connect(
            self._handle_search_highlighted_index
        )
        self.search_edit.textChanged.connect(self._handle_search_changed)
        self.search_edit.returnPressed.connect(self._handle_search_return_pressed)
        header_layout.addWidget(self.search_edit)
        header_layout.addStretch(1)
        self.find_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Find, self)
        self.find_shortcut.activated.connect(self._focus_search_edit)
        self.close_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+W"), self)
        self.close_shortcut.activated.connect(self.hide)

        self.notation_frame = QtWidgets.QFrame(self.header)
        notation_layout = QtWidgets.QHBoxLayout(self.notation_frame)
        notation_layout.setContentsMargins(0, 0, 0, 0)

        self.notation_label = QtWidgets.QLabel("Notation:", self.notation_frame)
        notation_layout.addWidget(self.notation_label)

        self.notation_combo = QtWidgets.QComboBox(self.notation_frame)
        self.notation_combo.setObjectName("ptable-notation")
        self.notation_combo.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.notation_combo.setToolTip("Choose how energy levels are labeled.")
        self.notation_combo.addItem("Orbital", "orbital")
        self.notation_combo.addItem("X-ray level", "iupac")
        self.notation_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.notation_combo.setCurrentIndex(self.notation_combo.findData(notation))
        notation_layout.addWidget(self.notation_combo)
        header_layout.addWidget(self.notation_frame)

        self.inspector = ElementInspector()
        self.inspector.setObjectName("ptable-inspector")
        self.inspector.setMinimumHeight(300)
        self.inspector.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.inspector.setAutoFillBackground(True)
        self.inspector.plot_target_changed.connect(self._handle_plot_target_changed)
        root_layout.addWidget(self.inspector, 1)

        self.photon_energy_label = QtWidgets.QLabel(
            "h\u03bd", self.inspector.levels_controls_frame
        )
        photon_energy_label_font = self.photon_energy_label.font()
        photon_energy_label_font.setItalic(True)
        self.photon_energy_label.setFont(photon_energy_label_font)
        self.photon_energy_label.setToolTip(_PHOTON_ENERGY_TOOLTIP)
        self.inspector.levels_controls_layout.addWidget(self.photon_energy_label)

        self.photon_energy_edit = _SuffixLineEdit(
            "eV", self.inspector.levels_controls_frame
        )
        self.photon_energy_edit.setObjectName("ptable-photon-energy")
        self.photon_energy_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.photon_energy_edit.setMinimumWidth(77)
        self.photon_energy_edit.setClearButtonEnabled(True)
        self.photon_energy_edit.setToolTip(_PHOTON_ENERGY_TOOLTIP)
        self.photon_energy_label.setBuddy(self.photon_energy_edit)
        if photon_energy is not None:
            self.photon_energy_edit.setText(str(photon_energy))
        self.photon_energy_edit.textChanged.connect(self._handle_energy_inputs_changed)
        self.inspector.levels_controls_layout.addWidget(self.photon_energy_edit, 1)

        self.workfunction_label = QtWidgets.QLabel(
            "\u03a6", self.inspector.levels_controls_frame
        )
        self.workfunction_label.setToolTip(_WORKFUNCTION_TOOLTIP)
        self.inspector.levels_controls_layout.addWidget(self.workfunction_label)

        self.workfunction_edit = _SuffixLineEdit(
            "eV", self.inspector.levels_controls_frame
        )
        self.workfunction_edit.setObjectName("ptable-workfunction")
        self.workfunction_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.workfunction_edit.setMinimumWidth(77)
        self.workfunction_edit.setClearButtonEnabled(True)
        self.workfunction_edit.setPlaceholderText("0")
        self.workfunction_edit.setToolTip(_WORKFUNCTION_TOOLTIP)
        self.workfunction_label.setBuddy(self.workfunction_edit)
        if workfunction is not None:
            self.workfunction_edit.setText(str(workfunction))
        self.workfunction_edit.textChanged.connect(self._handle_energy_inputs_changed)
        self.inspector.levels_controls_layout.addWidget(self.workfunction_edit, 1)

        self.harmonic_frame = QtWidgets.QFrame(self.inspector.levels_controls_frame)
        harmonic_layout = QtWidgets.QHBoxLayout(self.harmonic_frame)
        harmonic_layout.setContentsMargins(0, 0, 0, 0)
        harmonic_layout.setSpacing(4)

        self.max_harmonic_label = QtWidgets.QLabel(
            "Harmonics up to", self.harmonic_frame
        )
        harmonic_layout.addWidget(self.max_harmonic_label)

        self.max_harmonic_spin = QtWidgets.QSpinBox(self.harmonic_frame)
        self.max_harmonic_spin.setObjectName("ptable-max-harmonic")
        self.max_harmonic_spin.setRange(1, self._MAX_HARMONIC)
        self.max_harmonic_spin.setValue(max_harmonic)
        self.max_harmonic_spin.setMinimumWidth(64)
        self.max_harmonic_spin.valueChanged.connect(self._handle_max_harmonic_changed)
        harmonic_layout.addWidget(self.max_harmonic_spin)

        self.inspector.levels_controls_layout.addWidget(self.harmonic_frame)

        self.splitter = self.inspector.splitter
        self.top_panel = self.inspector.top_panel

        self.table_panel = QtWidgets.QFrame(self.inspector.table_container)
        self.table_panel.setObjectName("ptable-table-panel")
        self.table_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.table_panel.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        table_layout = QtWidgets.QVBoxLayout(self.table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)

        self.periodic_table = PeriodicTableWidget()
        self.category_legend = self.periodic_table.category_legend
        self.reference_dialog = _CategoryReferenceDialog(self)
        self.category_legend.set_reference_dialog(self.reference_dialog)
        self.table_view = PeriodicTableView(self.periodic_table, self.table_panel)
        self.periodic_table.hovered.connect(self._handle_card_hovered)
        self.periodic_table.selected.connect(self._handle_card_selected)
        self.table_view.hovered_atomic_number_changed.connect(
            self._handle_table_hover_changed
        )
        self.table_view.background_clicked.connect(self._handle_background_clicked)
        self.table_view.navigate_requested.connect(self._handle_navigation_requested)
        self.table_view.clear_requested.connect(self._handle_clear_requested)
        table_layout.addWidget(self.table_view, 1)
        self.inspector.set_table_panel(self.table_panel)
        self._update_table_view_minimum_scale()

        self.notation_combo.currentIndexChanged.connect(self._handle_notation_changed)
        application = QtWidgets.QApplication.instance()
        self._application = (
            application if isinstance(application, QtWidgets.QApplication) else None
        )
        if self._application is not None:
            self._application.installEventFilter(self)
            self._application.focusChanged.connect(
                self._handle_application_focus_changed
            )

        self._sync_vertical_minimum_height()
        self._apply_theme()
        self._refresh_inputs()
        self._refresh_window_state(ensure_visible=False)

    @property
    def selected_atomic_number(self) -> int | None:
        if (
            self._current_atomic_number is not None
            and self._current_atomic_number in self._selected_atomic_numbers
        ):
            return self._current_atomic_number
        return None

    @property
    def selected_atomic_numbers(self) -> tuple[int, ...]:
        return tuple(self._selected_atomic_numbers)

    @property
    def current_notation(self) -> str:
        current_data = self.notation_combo.currentData()
        if isinstance(current_data, str) and current_data in _NOTATION_VALUES:
            return current_data
        return _DEFAULT_NOTATION

    @property
    def current_record(self) -> ElementRecord | None:
        atomic_number = self._display_atomic_number()
        if atomic_number is None:
            return None
        return _element_records()[atomic_number]

    @property
    def photon_energy(self) -> float | None:
        return _parse_positive_float(self.photon_energy_edit.text().strip())

    @property
    def max_harmonic(self) -> int:
        return int(self.max_harmonic_spin.value())

    @property
    def workfunction(self) -> float:
        text = self.workfunction_edit.text().strip()
        if text == "":
            return 0.0
        try:
            value = float(text)
        except ValueError:
            return 0.0
        return max(value, 0.0)

    def _set_line_edit_invalid_state(
        self, widget: QtWidgets.QLineEdit, invalid: bool
    ) -> None:
        widget.setProperty("invalid", invalid)
        palette = QtGui.QPalette(widget.palette())
        palette.setColor(
            QtGui.QPalette.ColorRole.Base,
            self._theme.invalid_bg if invalid else self._theme.input_bg,
        )
        palette.setColor(QtGui.QPalette.ColorRole.Text, self._theme.text)
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, self._theme.text)
        palette.setColor(
            QtGui.QPalette.ColorRole.PlaceholderText,
            self._theme.muted_text,
        )
        widget.setPalette(palette)

    def _apply_theme(self) -> None:
        self._theme = _theme_colors()
        theme = self._theme
        _set_background(self.central, theme.window)
        _set_foreground(self.header, theme.text)
        for widget in (
            self.search_edit,
            self.photon_energy_edit,
            self.workfunction_edit,
        ):
            palette = QtGui.QPalette(widget.palette())
            palette.setColor(QtGui.QPalette.ColorRole.Base, theme.input_bg)
            palette.setColor(QtGui.QPalette.ColorRole.Text, theme.text)
            palette.setColor(QtGui.QPalette.ColorRole.WindowText, theme.text)
            palette.setColor(
                QtGui.QPalette.ColorRole.PlaceholderText,
                theme.muted_text,
            )
            widget.setPalette(palette)
        harmonic_palette = QtGui.QPalette(self.max_harmonic_spin.palette())
        harmonic_palette.setColor(QtGui.QPalette.ColorRole.Base, theme.input_bg)
        harmonic_palette.setColor(QtGui.QPalette.ColorRole.Button, theme.panel_alt)
        harmonic_palette.setColor(QtGui.QPalette.ColorRole.Text, theme.text)
        harmonic_palette.setColor(QtGui.QPalette.ColorRole.WindowText, theme.text)
        harmonic_palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.Base,
            theme.panel_alt,
        )
        harmonic_palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.Button,
            theme.panel_alt,
        )
        harmonic_palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.Text,
            theme.disabled_text,
        )
        harmonic_palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.WindowText,
            theme.disabled_text,
        )
        self.max_harmonic_spin.setPalette(harmonic_palette)
        harmonic_label_palette = QtGui.QPalette(self.max_harmonic_label.palette())
        harmonic_label_palette.setColor(QtGui.QPalette.ColorRole.WindowText, theme.text)
        harmonic_label_palette.setColor(QtGui.QPalette.ColorRole.Text, theme.text)
        harmonic_label_palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.WindowText,
            theme.disabled_text,
        )
        harmonic_label_palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.Text,
            theme.disabled_text,
        )
        self.max_harmonic_label.setPalette(harmonic_label_palette)
        _set_foreground(self.photon_energy_label, theme.text)
        _set_foreground(self.workfunction_label, theme.text)
        _set_foreground(self.notation_label, theme.text)
        notation_combo_palette = QtGui.QPalette(self.notation_combo.palette())
        notation_combo_palette.setColor(QtGui.QPalette.ColorRole.Base, theme.input_bg)
        notation_combo_palette.setColor(
            QtGui.QPalette.ColorRole.Button, theme.panel_alt
        )
        notation_combo_palette.setColor(QtGui.QPalette.ColorRole.Text, theme.text)
        notation_combo_palette.setColor(QtGui.QPalette.ColorRole.WindowText, theme.text)
        self.notation_combo.setPalette(notation_combo_palette)
        _set_background(self.table_panel, theme.table_surface)
        _set_background(self.inspector, theme.panel)
        self.inspector.setPalette(QtGui.QPalette(self.inspector.palette()))
        self._apply_search_popup_theme()
        self.periodic_table.apply_theme(theme)
        self.table_view.apply_theme(theme)
        self.inspector.apply_theme(theme)
        self._refresh_inputs()

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if event is None:
            return
        if event.type() in {
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._apply_theme()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if self.search_popup.isVisible():
            self._show_search_popup()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if self._application is not None:
            self._application.removeEventFilter(self)
            with contextlib.suppress(TypeError):
                self._application.focusChanged.disconnect(
                    self._handle_application_focus_changed
                )
        self.category_legend.cleanup()
        self._hide_search_popup(reset_navigation=False)
        super().closeEvent(event)

    def eventFilter(
        self,
        watched: QtCore.QObject | None,
        event: QtCore.QEvent | None,
    ) -> bool:
        if self.search_popup.isVisible() and event is not None:
            if event.type() == QtCore.QEvent.Type.WindowDeactivate:
                self._hide_search_popup(reset_navigation=True)
            elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
                mouse_event = event if isinstance(event, QtGui.QMouseEvent) else None
                if self._is_search_popup_target(watched):
                    return super().eventFilter(watched, event)
                if mouse_event is not None:
                    global_pos = mouse_event.globalPosition().toPoint()
                    if not (
                        self.search_popup.frameGeometry().contains(global_pos)
                        or QtCore.QRect(
                            self.search_edit.mapToGlobal(QtCore.QPoint(0, 0)),
                            self.search_edit.size(),
                        ).contains(global_pos)
                    ):
                        self._hide_search_popup(reset_navigation=True)
                elif not self._is_search_popup_target(watched):
                    self._hide_search_popup(reset_navigation=True)
        return super().eventFilter(watched, event)

    def _handle_search_key_press(self, event: QtGui.QKeyEvent) -> bool:
        if event.key() == QtCore.Qt.Key.Key_Down:
            return self._move_search_completion(1)
        if event.key() == QtCore.Qt.Key.Key_Up:
            return self._move_search_completion(-1)
        if event.key() == QtCore.Qt.Key.Key_Escape and self.search_popup.isVisible():
            self._hide_search_popup(reset_navigation=True)
            return True
        if (
            event.key() in {QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter}
            and self._search_completions
        ):
            self._handle_search_return_pressed()
            return True
        return False

    def _apply_search_popup_theme(self) -> None:
        theme = self._theme
        selected_bg = _blend_colors(
            theme.search_accent,
            theme.panel_alt,
            0.38 if theme.is_dark else 0.2,
        )
        hovered_bg = _blend_colors(
            theme.search_accent,
            theme.panel,
            0.24 if theme.is_dark else 0.12,
        )
        self.search_popup.setStyleSheet(
            f"""
            QListWidget#ptable-search-popup {{
                background: {_css_rgba(theme.input_bg)};
                border: 1px solid {_css_rgba(theme.border)};
                border-radius: 6px;
                padding: 4px 0px;
                outline: none;
            }}
            QListWidget#ptable-search-popup::item {{
                padding: 6px 10px;
                border: 0px;
                color: {_css_rgba(theme.text)};
            }}
            QListWidget#ptable-search-popup::item:hover {{
                background: {_css_rgba(hovered_bg)};
            }}
            QListWidget#ptable-search-popup::item:selected,
            QListWidget#ptable-search-popup::item:selected:active,
            QListWidget#ptable-search-popup::item:selected:!active {{
                background: {_css_rgba(selected_bg)};
                color: {_css_rgba(theme.text)};
            }}
            """
        )

    def _is_search_popup_target(self, watched: QtCore.QObject | None) -> bool:
        widget = watched if isinstance(watched, QtWidgets.QWidget) else None
        return bool(
            widget is not None
            and (
                widget is self.search_edit
                or self.search_edit.isAncestorOf(widget)
                or widget is self.search_popup
                or self.search_popup.isAncestorOf(widget)
            )
        )

    def _search_completion_at(self, row: int) -> _SearchCompletion | None:
        if 0 <= row < len(self._search_completions):
            return self._search_completions[row]
        return None

    def _all_search_match_atomic_numbers(self) -> set[int]:
        return {
            atomic_number
            for completion in self._search_completions
            for atomic_number in completion.atomic_numbers
        }

    def _build_search_completions(self, text: str) -> list[_SearchCompletion]:
        completions: list[_SearchCompletion] = []
        multi_records = _parse_symbol_selection_query(text)
        if multi_records:
            symbols = ", ".join(record.symbol for record in multi_records)
            completions.append(
                _SearchCompletion(
                    display_text=f"Select: {symbols}",
                    atomic_numbers=tuple(
                        record.atomic_number for record in multi_records
                    ),
                    search_text=symbols,
                )
            )
        completions.extend(
            _SearchCompletion(
                display_text=f"{record.symbol} - {record.name}",
                atomic_numbers=(record.atomic_number,),
                search_text=record.symbol,
            )
            for record in _search_records(text)
        )
        return completions

    def _show_search_popup(self) -> None:
        if (
            not self._search_completions
            or not self.search_edit.text().strip()
            or not self.search_edit.hasFocus()
        ):
            self._hide_search_popup(reset_navigation=False)
            return
        was_visible = self.search_popup.isVisible()
        row_height = self.search_popup.sizeHintForRow(0)
        if row_height <= 0:
            row_height = self.search_popup.fontMetrics().height() + 10
        visible_rows = min(
            self.search_popup.count(),
            self.search_popup.maxVisibleItems(),
        )
        popup_height = (
            (row_height * max(1, visible_rows))
            + (self.search_popup.frameWidth() * 2)
            + 8
        )
        top_left = self.search_edit.mapTo(
            self,
            QtCore.QPoint(0, self.search_edit.height() + 2),
        )
        self.search_popup.setGeometry(
            top_left.x(),
            top_left.y(),
            self.search_edit.width(),
            popup_height,
        )
        self.search_popup.show()
        self.search_popup.raise_()
        if not was_visible:
            self._refresh_window_state(ensure_visible=False)

    def _hide_search_popup(self, *, reset_navigation: bool) -> None:
        was_visible = self.search_popup.isVisible()
        self.search_popup.hide()
        if reset_navigation:
            self._search_completion_row = None
            self._search_matches = self._all_search_match_atomic_numbers()
        if was_visible:
            self._refresh_window_state(ensure_visible=False)

    def _handle_search_focus_in(self) -> None:
        if self._search_completions:
            self._show_search_popup()

    def _focus_search_edit(self) -> None:
        self.search_edit.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        self.search_edit.selectAll()
        if self._search_completions and self.search_edit.text().strip():
            self._show_search_popup()

    def _handle_application_focus_changed(
        self,
        _old: QtWidgets.QWidget | None,
        new: QtWidgets.QWidget | None,
    ) -> None:
        if not self.search_popup.isVisible() or new is None:
            return
        if self._is_search_popup_target(new):
            return
        self._hide_search_popup(reset_navigation=True)

    def _refresh_inputs(self) -> None:
        photon_text = self.photon_energy_edit.text().strip()
        workfunction_text = self.workfunction_edit.text().strip()
        self._set_line_edit_invalid_state(
            self.photon_energy_edit,
            photon_text != "" and _parse_positive_float(photon_text) is None,
        )
        if workfunction_text == "":
            work_invalid = False
        else:
            try:
                work_invalid = float(workfunction_text) < 0.0
            except ValueError:
                work_invalid = True
        self._set_line_edit_invalid_state(self.workfunction_edit, work_invalid)
        harmonics_enabled = (
            photon_text != "" and _parse_positive_float(photon_text) is not None
        )
        self.harmonic_frame.setEnabled(harmonics_enabled)

    def _legacy_table_viewport_size(self) -> QtCore.QSize:
        root_layout = self.central.layout()
        if not isinstance(root_layout, QtWidgets.QVBoxLayout):  # pragma: no cover
            return QtCore.QSize()

        margins = root_layout.contentsMargins()
        available_width = max(
            self._LEGACY_MINIMUM_WINDOW_SIZE.width() - margins.left() - margins.right(),
            0,
        )
        available_height = max(
            self._LEGACY_MINIMUM_WINDOW_SIZE.height()
            - margins.top()
            - margins.bottom()
            - self.header.sizeHint().height()
            - root_layout.spacing(),
            0,
        )

        table_width = max(
            available_width
            - self.inspector.side_panel.minimumWidth()
            - (2 * self.table_panel.frameWidth()),
            0,
        )

        table_height = max(
            round(
                (available_height - self.splitter.handleWidth())
                * self._LEGACY_TOP_SPLITTER_RATIO
            )
            - (2 * self.table_panel.frameWidth()),
            0,
        )
        return QtCore.QSize(table_width, table_height)

    def _update_table_view_minimum_scale(self) -> None:
        self.table_view.set_minimum_scale_for_viewport(
            self._legacy_table_viewport_size()
        )

    def _sync_vertical_minimum_height(self) -> None:
        minimum_height = self.splitter.minimumSizeHint().height()
        self.splitter.setMinimumHeight(minimum_height)
        self.setMinimumHeight(minimum_height)

    def _refresh_window_state(self, *, ensure_visible: bool) -> None:
        preview = self._should_preview_hover()
        self.periodic_table.set_search_matches(
            self._search_matches if self.search_popup.isVisible() else set()
        )
        self.periodic_table.set_selected_atomic_numbers(
            set(self._selected_atomic_numbers)
        )
        self.periodic_table.set_current_atomic_number(self._current_atomic_number)
        self.periodic_table.set_hovered_atomic_number(self._hover_atomic_number)
        selected_records = tuple(
            _element_records()[num] for num in self._selected_atomic_numbers
        )
        current_record = self.current_record
        plot_atomic_number = (
            self._display_atomic_number() if preview else self._plot_atomic_number
        )
        plot_record = (
            None
            if plot_atomic_number is None
            else _element_records()[plot_atomic_number]
        )
        self.inspector.set_elements(
            selected_records,
            active_record=current_record,
            plot_record=plot_record,
            notation=self.current_notation,
            photon_energy=self.photon_energy,
            workfunction=self.workfunction,
            max_harmonic=self.max_harmonic,
            preview=preview,
        )
        self._sync_vertical_minimum_height()
        if ensure_visible:
            self.table_view.ensure_atomic_number_visible(self._current_atomic_number)

    def _display_atomic_number(self) -> int | None:
        if self._should_preview_hover():
            return self._hover_atomic_number
        return self.selected_atomic_number

    def _should_preview_hover(self) -> bool:
        return (
            self._hover_atomic_number is not None
            and len(self._selected_atomic_numbers) <= 1
        )

    def _set_selection_state(
        self,
        atomic_numbers: list[int],
        *,
        current_atomic_number: int | None,
        anchor_atomic_number: int | None = None,
    ) -> None:
        deduped_atomic_numbers = list(dict.fromkeys(atomic_numbers))
        if current_atomic_number not in deduped_atomic_numbers:
            current_atomic_number = None
        if anchor_atomic_number not in deduped_atomic_numbers:
            anchor_atomic_number = current_atomic_number

        self._selected_atomic_numbers = deduped_atomic_numbers
        self._current_atomic_number = current_atomic_number
        self._selection_anchor_atomic_number = anchor_atomic_number

        if len(deduped_atomic_numbers) == 0:
            self._plot_atomic_number = None
            self._plot_target_user_selected = False
            return
        if len(deduped_atomic_numbers) == 1:
            self._plot_atomic_number = deduped_atomic_numbers[0]
            self._plot_target_user_selected = False
            return
        if (
            self._plot_target_user_selected
            and self._plot_atomic_number in self._selected_atomic_numbers
        ):
            return
        self._plot_atomic_number = (
            current_atomic_number
            if current_atomic_number is not None
            else deduped_atomic_numbers[-1]
        )
        self._plot_target_user_selected = False

    def _select_single_atomic_number(self, atomic_number: int) -> None:
        self._set_selection_state(
            [atomic_number],
            current_atomic_number=atomic_number,
            anchor_atomic_number=atomic_number,
        )

    def _toggle_atomic_number(self, atomic_number: int) -> None:
        updated_selection = list(self._selected_atomic_numbers)
        if atomic_number in updated_selection:
            updated_selection.remove(atomic_number)
            current_atomic_number = (
                self._current_atomic_number
                if self._current_atomic_number in updated_selection
                else updated_selection[-1]
                if updated_selection
                else None
            )
            anchor_atomic_number = (
                self._selection_anchor_atomic_number
                if self._selection_anchor_atomic_number in updated_selection
                else current_atomic_number
            )
        else:
            updated_selection.append(atomic_number)
            current_atomic_number = atomic_number
            anchor_atomic_number = self._selection_anchor_atomic_number or atomic_number
        self._set_selection_state(
            updated_selection,
            current_atomic_number=current_atomic_number,
            anchor_atomic_number=anchor_atomic_number,
        )

    def _range_select_to(
        self,
        atomic_number: int,
        *,
        rectangular: bool,
    ) -> None:
        anchor_atomic_number = (
            self._selection_anchor_atomic_number
            or self._current_atomic_number
            or atomic_number
        )
        selection = (
            _rectangular_selection_range(anchor_atomic_number, atomic_number)
            if rectangular
            else _selection_range(anchor_atomic_number, atomic_number)
        )
        self._set_selection_state(
            list(selection),
            current_atomic_number=atomic_number,
            anchor_atomic_number=anchor_atomic_number,
        )

    def _clear_selection(self) -> None:
        self._set_selection_state(
            [], current_atomic_number=None, anchor_atomic_number=None
        )

    def _handle_card_hovered(self, atomic_number: int) -> None:
        self._set_hover_atomic_number(atomic_number)

    def _handle_card_unhovered(self, atomic_number: int) -> None:
        if self._hover_atomic_number == atomic_number:
            self._set_hover_atomic_number(None)

    def _handle_table_hover_changed(self, atomic_number: object) -> None:
        self._set_hover_atomic_number(
            atomic_number if isinstance(atomic_number, int) else None
        )

    def _set_hover_atomic_number(self, atomic_number: int | None) -> None:
        if self._hover_atomic_number == atomic_number:
            return
        self._hover_atomic_number = atomic_number
        self._refresh_window_state(ensure_visible=False)

    def _clear_hover_preview(self) -> None:
        self.table_view.clear_hover_tracking()
        self._set_hover_atomic_number(None)

    def _handle_card_selected(self, atomic_number: int, modifiers: object) -> None:
        if _has_keyboard_modifier(modifiers, QtCore.Qt.KeyboardModifier.ShiftModifier):
            self._range_select_to(atomic_number, rectangular=True)
        elif _is_toggle_selection_modifier(modifiers):
            self._toggle_atomic_number(atomic_number)
        else:
            self._select_single_atomic_number(atomic_number)
        self.table_view.setFocus()
        self._clear_hover_preview()
        self._refresh_window_state(ensure_visible=False)

    def _handle_background_clicked(self, _: object) -> None:
        self._clear_hover_preview()
        self._clear_selection()
        self._refresh_window_state(ensure_visible=False)

    def _handle_clear_requested(self) -> None:
        self._clear_selection()
        self._refresh_window_state(ensure_visible=False)

    def _handle_navigation_requested(self, key: int, modifiers: object) -> None:
        navigation = _navigation_atomic_numbers()
        current_atomic_number = self._current_atomic_number
        target_atomic_number: int | None
        if current_atomic_number is None:
            if not navigation:
                return
            target_atomic_number = (
                navigation[-1]
                if key in {int(QtCore.Qt.Key.Key_Left), int(QtCore.Qt.Key.Key_Up)}
                else navigation[0]
            )
        else:
            target_atomic_number = _adjacent_atomic_number(current_atomic_number, key)
        if target_atomic_number is None:
            return

        if _has_keyboard_modifier(modifiers, QtCore.Qt.KeyboardModifier.ShiftModifier):
            self._range_select_to(target_atomic_number, rectangular=False)
        else:
            self._select_single_atomic_number(target_atomic_number)
        self._clear_hover_preview()
        self._refresh_window_state(ensure_visible=True)

    def _handle_search_changed(self, text: str) -> None:
        if text.strip() == "":
            self._search_matches = set()
            self._search_completions = ()
            self._search_completion_row = None
            self._update_search_completions([])
            self._refresh_window_state(ensure_visible=False)
            return

        completions = self._build_search_completions(text)
        self._search_completions = tuple(completions)
        self._search_matches = self._all_search_match_atomic_numbers()
        self._search_completion_row = None
        self._update_search_completions(completions)
        self._refresh_window_state(ensure_visible=False)

    def _update_search_completions(self, completions: list[_SearchCompletion]) -> None:
        self.search_popup.blockSignals(True)
        self.search_popup.clear()
        for completion in completions:
            item = QtWidgets.QListWidgetItem(completion.display_text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, completion.atomic_numbers)
            self.search_popup.addItem(item)
        self.search_popup.setCurrentRow(-1)
        self.search_popup.blockSignals(False)
        if (
            completions
            and self.search_edit.text().strip()
            and self.search_edit.hasFocus()
        ):
            self.search_completer.complete()
            return
        self._hide_search_popup(reset_navigation=False)

    def _handle_search_completion(self, text: str) -> None:
        completion = self._search_completion_at(self.search_popup.currentRow()) or next(
            (entry for entry in self._search_completions if entry.display_text == text),
            None,
        )
        if completion is not None:
            self._activate_search_completion(completion)

    def _handle_search_completion_index(self, index: QtCore.QModelIndex) -> None:
        completion = self._search_completion_at(index.row())
        if completion is None:
            return
        self._search_completion_row = index.row()
        self._activate_search_completion(completion)

    def _handle_search_highlighted_index(self, index: QtCore.QModelIndex) -> None:
        completion = self._search_completion_at(index.row())
        if completion is None:
            self._search_completion_row = None
            previous_matches = set(self._search_matches)
            self._search_matches = self._all_search_match_atomic_numbers()
            if self._search_matches != previous_matches:
                self._refresh_window_state(ensure_visible=False)
            return
        self._search_completion_row = index.row()
        previous_matches = set(self._search_matches)
        self._search_matches = set(completion.atomic_numbers)
        if self._search_matches != previous_matches:
            self._refresh_window_state(ensure_visible=False)

    def _move_search_completion(self, step: int) -> bool:
        if not self._search_completions:
            return False
        row_count = len(self._search_completions)
        if self._search_completion_row is not None:
            current_row = self._search_completion_row
        else:
            current_row = -1 if step > 0 else row_count
        target_row = max(0, min(row_count - 1, current_row + step))
        self._search_completion_row = target_row
        completion = self._search_completions[target_row]
        previous_matches = set(self._search_matches)
        self._search_matches = set(completion.atomic_numbers)
        if self._search_matches != previous_matches:
            self._refresh_window_state(ensure_visible=False)
        self.search_popup.blockSignals(True)
        self.search_popup.setCurrentRow(target_row)
        self.search_popup.blockSignals(False)
        self.search_popup.scrollToItem(self.search_popup.item(target_row))
        if not self.search_popup.isVisible():
            self.search_completer.complete()
        return True

    def _sync_search_popup_index(self) -> None:
        return None

    def _handle_search_return_pressed(self) -> None:
        completion = (
            self._search_completion_at(self._search_completion_row)
            if self._search_completion_row is not None
            else None
        )
        if completion is None and self._search_completions:
            completion = self._search_completions[0]
        if completion is not None:
            self._activate_search_completion(completion)

    def _activate_search_completion(self, completion: _SearchCompletion) -> None:
        blocker = QtCore.QSignalBlocker(self.search_edit)
        self.search_edit.setText(completion.search_text)
        del blocker
        self._clear_hover_preview()
        self._search_completions = (completion,)
        self._search_matches = set(completion.atomic_numbers)
        self._search_completion_row = None
        self._update_search_completions([completion])
        self._hide_search_popup(reset_navigation=False)
        if len(completion.atomic_numbers) == 1:
            self._select_single_atomic_number(completion.atomic_numbers[0])
        else:
            current_atomic_number = completion.atomic_numbers[-1]
            self._set_selection_state(
                list(completion.atomic_numbers),
                current_atomic_number=current_atomic_number,
                anchor_atomic_number=current_atomic_number,
            )
        self.table_view.setFocus()
        self._refresh_window_state(ensure_visible=True)

    def _handle_plot_target_changed(self, atomic_number: int) -> None:
        if (
            atomic_number not in self._selected_atomic_numbers
            or atomic_number == self._plot_atomic_number
        ):
            return
        self._plot_atomic_number = atomic_number
        self._plot_target_user_selected = len(self._selected_atomic_numbers) > 1
        self._refresh_window_state(ensure_visible=False)

    def _handle_energy_inputs_changed(self, _: str) -> None:
        self._refresh_inputs()
        self._refresh_window_state(ensure_visible=False)

    def _handle_max_harmonic_changed(self, _: int) -> None:
        self._refresh_window_state(ensure_visible=False)

    def _handle_notation_changed(self, _: int) -> None:
        settings = _get_ptable_settings()
        settings.setValue(_NOTATION_SETTINGS_KEY, self.current_notation)
        settings.sync()
        self._refresh_window_state(ensure_visible=False)


def ptable(
    *,
    photon_energy: float | None = None,
    workfunction: float | None = None,
    max_harmonic: int = 1,
    notation: str | None = None,
    execute: bool | None = None,
) -> PeriodicTableWindow:
    r"""Open the periodic table window.

    The periodic table provides an interactive reference for XPS-relevant elemental
    properties, including core-level binding energies and photoionization cross
    sections.

    See :ref:`guide-ptable` for an overview of the features and user interface.

    Parameters
    ----------
    photon_energy
        Optional photon energy in eV. When given, the inspector also shows kinetic
        energies and marks the photon energy on the cross-section plot.
    workfunction
        Optional work function in eV used for converting an absorption edge to kinetic
        energy.
    max_harmonic
        Highest harmonic order to include when kinetic energies are shown. Harmonics are
        integer multiples of ``photon_energy`` from ``1`` through ``max_harmonic``.
    notation
        Initial energy-level notation. When omitted, the most recently used notation is
        restored. Explicit values should be either ``"orbital"`` or ``"iupac"``.
    execute
        Passed through to :func:`erlab.interactive.utils.setup_qapp`.

    Notes
    -----
    The periodic table combines several reference datasets.

    - Element symbols, names, atomic masses, and the absorption-edge values shown in the
      core-level table are loaded at runtime from :mod:`xraydb`
      :cite:p:`newville2023xraydb`, using the underlying X-ray level compilation
      described by :cite:t:`elam2002xraydb`. These absorption edges are closely related
      to, and in many XPS workflows interpreted similarly to, core-level binding
      energies.
    - Photoionization cross sections shown by
      :func:`erlab.analysis.xps.get_cross_section` use the bundled
      ``yeh_lindau_1985_pics.npz`` archive, obtained from the `Elettra WebCrossSections
      <https://vuo.elettra.eu/services/elements/WebElements.html>`_ service based on of
      the original Yeh-Lindau subshell tables :cite:p:`yeh1985photoionization`.
    - Ground-state shell configurations for elements :math:`Z=1\ldots108` are taken
      from the NIST Atomic Spectra Database :cite:p:`kramida2024asd`.
    """
    with erlab.interactive.utils.setup_qapp(execute):
        win = PeriodicTableWindow(
            photon_energy=photon_energy,
            workfunction=workfunction,
            max_harmonic=max_harmonic,
            notation=notation,
        )
        win.show()
        win.raise_()
        win.activateWindow()
    return win
