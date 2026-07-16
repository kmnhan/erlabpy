"""Browse and select manager-owned Figure Composer windows."""

from __future__ import annotations

import re
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.manager._widgets import _manager_settings

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager


_VIEW_MODE_SETTINGS_KEY = "figures/view_mode"
_GALLERY_SIZE_SETTINGS_KEY = "figures/gallery_thumbnail_size"
_VIEW_MODE_LIST = "list"
_VIEW_MODE_GALLERY = "gallery"
_VIEW_MODES = (_VIEW_MODE_LIST, _VIEW_MODE_GALLERY)
_GALLERY_SIZE_MEDIUM = "medium"
_GALLERY_THUMBNAIL_SIZES = {
    "small": (112, 84),
    _GALLERY_SIZE_MEDIUM: (152, 114),
    "large": (216, 162),
}


class _FigureCollectionPane(QtWidgets.QWidget):
    """Own the widgets used to browse managed Figure Composer windows."""

    selection_changed = QtCore.Signal()
    item_changed = QtCore.Signal(object)
    item_activated = QtCore.Signal(object)
    context_menu_requested = QtCore.Signal(QtCore.QPoint)
    view_mode_requested = QtCore.Signal(str)
    gallery_size_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("manager_figures_tab")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view_controls = QtWidgets.QWidget(self)
        self.view_controls.setObjectName("manager_figures_view_controls")
        controls_layout = QtWidgets.QHBoxLayout(self.view_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(self._horizontal_spacing())

        self.view_button_group = QtWidgets.QButtonGroup(self.view_controls)
        self.view_button_group.setExclusive(True)
        self.list_button = self._view_mode_button(
            "List",
            erlab.interactive.utils.qtawesome.icon("ph.list-bullets"),
            _VIEW_MODE_LIST,
        )
        self.gallery_button = self._view_mode_button(
            "Gallery",
            erlab.interactive.utils.qtawesome.icon("ph.grid-four"),
            _VIEW_MODE_GALLERY,
        )
        for button in (self.list_button, self.gallery_button):
            self.view_button_group.addButton(button)
            controls_layout.addWidget(button)

        self.gallery_size_label = QtWidgets.QLabel("Thumbnail", self.view_controls)
        self.gallery_size_combo = QtWidgets.QComboBox(self.view_controls)
        self.gallery_size_combo.setObjectName("manager_figures_gallery_size")
        self.gallery_size_combo.setToolTip("Choose gallery thumbnail size.")
        self.gallery_size_label.setBuddy(self.gallery_size_combo)
        for label, key in (
            ("Small", "small"),
            ("Medium", "medium"),
            ("Large", "large"),
        ):
            self.gallery_size_combo.addItem(label, key)
        self.gallery_size_combo.currentIndexChanged.connect(self._gallery_size_changed)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.gallery_size_label)
        controls_layout.addWidget(self.gallery_size_combo)
        layout.addWidget(self.view_controls)

        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setObjectName("manager_figures_list")
        self.list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.list_widget.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.list_widget.itemSelectionChanged.connect(self.selection_changed)
        self.list_widget.itemChanged.connect(self._emit_item_changed)
        self.list_widget.itemDoubleClicked.connect(self._emit_item_activated)
        self.list_widget.customContextMenuRequested.connect(self.context_menu_requested)
        layout.addWidget(self.list_widget)
        self._updating_controls = False

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _emit_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        self.item_changed.emit(item)

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _emit_item_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        self.item_activated.emit(item)

    def _horizontal_spacing(self) -> int:
        style = self.style() or QtWidgets.QApplication.style()
        if style is None:
            raise RuntimeError("No active Qt style")
        return style.pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing
        )

    def _view_mode_button(
        self, text: str, icon: QtGui.QIcon, mode: str
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self.view_controls)
        button.setObjectName(f"manager_figures_{mode}_view_button")
        button.setIcon(icon)
        button.setCheckable(True)
        button.setAutoRaise(True)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setAccessibleName(f"{text} view")
        button.setToolTip(f"Show figures in {text.lower()} view.")
        button.clicked.connect(
            lambda _checked=False, requested_mode=mode: self.view_mode_requested.emit(
                requested_mode
            )
        )
        return button

    @QtCore.Slot(int)
    def _gallery_size_changed(self, _index: int) -> None:
        if self._updating_controls:
            return
        size_name = self.gallery_size_combo.currentData()
        if isinstance(size_name, str):
            self.gallery_size_requested.emit(size_name)

    def apply_view(self, mode: str, size_name: str) -> None:
        self._updating_controls = True
        try:
            gallery_mode = mode == _VIEW_MODE_GALLERY
            self.list_button.setChecked(mode == _VIEW_MODE_LIST)
            self.gallery_button.setChecked(gallery_mode)
            self.gallery_size_label.setVisible(gallery_mode)
            self.gallery_size_combo.setVisible(gallery_mode)
            self.gallery_size_combo.setEnabled(gallery_mode)
            size_index = self.gallery_size_combo.findData(size_name)
            if size_index >= 0:
                self.gallery_size_combo.setCurrentIndex(size_index)
        finally:
            self._updating_controls = False

        self.list_widget.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        if gallery_mode:
            self.list_widget.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
            self.list_widget.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
            self.list_widget.setMovement(QtWidgets.QListView.Movement.Static)
            self.list_widget.setFlow(QtWidgets.QListView.Flow.LeftToRight)
            self.list_widget.setWrapping(True)
            self.list_widget.setSpacing(self._horizontal_spacing())
            self.list_widget.setUniformItemSizes(True)
            self.list_widget.setIconSize(self.thumbnail_size(size_name))
            thumbnail_size = self.thumbnail_size(size_name)
            label_height = self.list_widget.fontMetrics().height() * 2
            spacing = self._horizontal_spacing()
            self.list_widget.setGridSize(
                QtCore.QSize(
                    thumbnail_size.width() + spacing * 4,
                    thumbnail_size.height() + label_height + spacing * 3,
                )
            )
            self.list_widget.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
            return
        self.list_widget.setViewMode(QtWidgets.QListView.ViewMode.ListMode)
        self.list_widget.setResizeMode(QtWidgets.QListView.ResizeMode.Fixed)
        self.list_widget.setMovement(QtWidgets.QListView.Movement.Static)
        self.list_widget.setFlow(QtWidgets.QListView.Flow.TopToBottom)
        self.list_widget.setWrapping(False)
        self.list_widget.setSpacing(0)
        self.list_widget.setUniformItemSizes(False)
        self.list_widget.setIconSize(QtCore.QSize())
        self.list_widget.setGridSize(QtCore.QSize())

    @staticmethod
    def thumbnail_size(size_name: str) -> QtCore.QSize:
        width, height = _GALLERY_THUMBNAIL_SIZES[size_name]
        return QtCore.QSize(width, height)


class _FigureCollectionController(QtCore.QObject):
    """Own the manager's Figure Composer collection browser."""

    def __init__(self, host: ImageToolManager, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self._host = host
        self._parent_widget = parent
        self._pane: _FigureCollectionPane | None = None
        self._menu: QtWidgets.QMenu | None = None
        self._refreshing = False
        self._view_mode = self._read_view_mode_setting()
        self._gallery_size_name = self._read_gallery_size_setting()

    @property
    def pane(self) -> _FigureCollectionPane | None:
        return self._pane

    @staticmethod
    def _settings_string(key: str, default: str) -> str:
        value = _manager_settings().value(key, default)
        return value if isinstance(value, str) else default

    def _read_view_mode_setting(self) -> str:
        mode = self._settings_string(_VIEW_MODE_SETTINGS_KEY, _VIEW_MODE_GALLERY)
        return mode if mode in _VIEW_MODES else _VIEW_MODE_GALLERY

    def _read_gallery_size_setting(self) -> str:
        size_name = self._settings_string(
            _GALLERY_SIZE_SETTINGS_KEY, _GALLERY_SIZE_MEDIUM
        )
        if size_name in _GALLERY_THUMBNAIL_SIZES:
            return size_name
        return _GALLERY_SIZE_MEDIUM

    def selected_uids(self) -> list[str]:
        pane = self._pane
        if pane is None:
            return []
        output: list[str] = []
        for item in pane.list_widget.selectedItems():
            uid = self.uid_from_item(item)
            if uid is not None and self._host._is_figure_uid(uid):
                output.append(uid)
        return output

    def next_display_name(self) -> str:
        highest = 0
        for uid in self._host._figure_uids():
            match = re.fullmatch(
                r"Figure (\d+)", self._host._child_node(uid).display_text
            )
            if match is not None:
                highest = max(highest, int(match.group(1)))
        return f"Figure {highest + 1}"

    def duplicated_display_name(self, display_name: str) -> str:
        if re.fullmatch(r"Figure \d+", display_name):
            return self.next_display_name()
        existing_names = {
            self._host._child_node(uid).display_text
            for uid in self._host._figure_uids()
        }
        base_name = f"{display_name} copy"
        if base_name not in existing_names:
            return base_name
        suffix = 2
        while f"{base_name} {suffix}" in existing_names:
            suffix += 1
        return f"{base_name} {suffix}"

    def sync(self, *, select_uid: str | None = None) -> None:
        if self._host._figure_ui_refresh_is_deferred():
            self._host._defer_figure_ui_refresh(select_uid)
            return
        figure_uids = self._host._figure_uids()
        selected_uids = (
            {select_uid} if select_uid is not None else set(self.selected_uids())
        )
        self._set_available(bool(figure_uids))
        pane = self._pane
        if pane is None:
            return
        pane.apply_view(self._view_mode, self._gallery_size_name)
        self._refreshing = True
        pane.list_widget.blockSignals(True)
        try:
            pane.list_widget.clear()
            for uid in figure_uids:
                item = self._list_item(uid)
                pane.list_widget.addItem(item)
                if uid in selected_uids:
                    item.setSelected(True)
                    pane.list_widget.setCurrentItem(item)
        finally:
            pane.list_widget.blockSignals(False)
            self._refreshing = False
        if select_uid is not None:
            self._host.left_tabs.setCurrentWidget(pane)

    def select_uid(self, uid: str) -> None:
        self.sync(select_uid=uid)
        self._host._deselect_tree()
        self._host._update_actions()
        self._host._update_info()

    def clear_selection_from_tree(self) -> None:
        pane = self._pane
        if (
            self._refreshing
            or pane is None
            or not self._host._tree_has_selection()
            or not pane.list_widget.selectedItems()
        ):
            return
        pane.list_widget.blockSignals(True)
        try:
            pane.list_widget.clearSelection()
        finally:
            pane.list_widget.blockSignals(False)
        self._host._update_actions()
        self._host._update_info()

    def item_for_uid(self, uid: str) -> QtWidgets.QListWidgetItem | None:
        pane = self._pane
        if pane is None:
            return None
        for row in range(pane.list_widget.count()):
            item = pane.list_widget.item(row)
            if item is not None and self.uid_from_item(item) == uid:
                return item
        return None

    @staticmethod
    def uid_from_item(item: QtWidgets.QListWidgetItem | None) -> str | None:
        if item is None:
            return None
        uid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return uid if isinstance(uid, str) else None

    def update_gallery_icon(self, uid: str) -> None:
        if self._host._figure_ui_refresh_is_deferred():
            self._host._defer_figure_gallery_icon_update(uid)
            return
        pane = self._pane
        if (
            pane is None
            or not erlab.interactive.utils.qt_is_valid(pane)
            or self._refreshing
            or self._view_mode != _VIEW_MODE_GALLERY
            or not self._host._is_figure_uid(uid)
        ):
            return
        item = self.item_for_uid(uid)
        if item is not None:
            pane.list_widget.blockSignals(True)
            try:
                item.setIcon(self._gallery_icon(uid))
            finally:
                pane.list_widget.blockSignals(False)

    def close_menu(self) -> None:
        menu = self._menu
        if menu is None:
            return
        if not erlab.interactive.utils.qt_is_valid(menu):
            self._menu = None
            return
        menu.close()
        if self._menu is menu:
            self._release_menu(menu)

    def _set_available(self, available: bool) -> None:
        if not available:
            self._destroy_pane()
            tab_bar = self._host.left_tabs.tabBar()
            if tab_bar is not None:
                tab_bar.setVisible(False)
            self._host.left_tabs.updateGeometry()
            return
        pane = self._ensure_pane()
        tab_index = self._host.left_tabs.indexOf(pane)
        if tab_index < 0:
            tab_index = self._host.left_tabs.addTab(pane, "Figures")
        pane.show()
        self._host.left_tabs.setTabVisible(tab_index, True)
        tab_bar = self._host.left_tabs.tabBar()
        if tab_bar is not None:
            tab_bar.setVisible(True)
        self._host.left_tabs.updateGeometry()

    def _ensure_pane(self) -> _FigureCollectionPane:
        if self._pane is not None:
            return self._pane
        pane = _FigureCollectionPane(self._host.left_tabs)
        pane.selection_changed.connect(self._selection_changed)
        pane.item_changed.connect(self._item_changed)
        pane.item_activated.connect(self._show_item)
        pane.context_menu_requested.connect(self._show_menu)
        pane.view_mode_requested.connect(self._set_view_mode)
        pane.gallery_size_requested.connect(self._set_gallery_size)
        self._host._install_selection_shortcuts(pane.list_widget)
        pane.apply_view(self._view_mode, self._gallery_size_name)
        self._pane = pane
        return pane

    def _destroy_pane(self) -> None:
        self.close_menu()
        pane = self._pane
        if pane is None:
            return
        self._refreshing = True
        pane.list_widget.blockSignals(True)
        tab_index = self._host.left_tabs.indexOf(pane)
        if tab_index >= 0:
            if self._host.left_tabs.currentIndex() == tab_index:
                self._host.left_tabs.setCurrentIndex(0)
            self._host.left_tabs.removeTab(tab_index)
        self._pane = None
        pane.hide()
        pane.deleteLater()
        self._refreshing = False

    @QtCore.Slot(str)
    def _set_view_mode(self, mode: str) -> None:
        if mode not in _VIEW_MODES or mode == self._view_mode:
            return
        self._view_mode = mode
        _manager_settings().setValue(_VIEW_MODE_SETTINGS_KEY, mode)
        self.sync()

    @QtCore.Slot(str)
    def _set_gallery_size(self, size_name: str) -> None:
        if (
            size_name not in _GALLERY_THUMBNAIL_SIZES
            or size_name == self._gallery_size_name
        ):
            return
        self._gallery_size_name = size_name
        _manager_settings().setValue(_GALLERY_SIZE_SETTINGS_KEY, size_name)
        self.sync()

    @QtCore.Slot()
    def _selection_changed(self) -> None:
        pane = self._pane
        if self._refreshing or pane is None:
            return
        if pane.list_widget.selectedItems():
            self._host._clear_tree_selection()
        self._host._update_actions()
        self._host._update_info()

    @QtCore.Slot(object)
    def _item_changed(self, item: object) -> None:
        if self._refreshing or not isinstance(item, QtWidgets.QListWidgetItem):
            return
        uid = self.uid_from_item(item)
        if uid is None or not self._host._is_figure_uid(uid):
            return
        self._host._child_node(uid).name = item.text()
        self._host._update_info(uid=uid)

    @QtCore.Slot(object)
    def _show_item(self, item: object) -> None:
        if not isinstance(item, QtWidgets.QListWidgetItem):
            return
        uid = self.uid_from_item(item)
        if uid is not None and self._host._is_figure_uid(uid):
            self._host.show_childtool(uid)

    @QtCore.Slot(QtCore.QPoint)
    def _show_menu(self, position: QtCore.QPoint) -> None:
        pane = self._pane
        if pane is None:
            return
        self.close_menu()
        menu = QtWidgets.QMenu("Figures", pane.list_widget)
        self._menu = menu
        menu.aboutToHide.connect(lambda *, popup=menu: self._release_menu(popup))
        menu.addAction(self._host.show_action)
        menu.addAction(self._host.hide_action)
        menu.addSeparator()
        menu.addAction(self._host.duplicate_action)
        menu.addAction(self._host.remove_action)
        menu.addAction(self._host.rename_action)
        menu.addSeparator()
        menu.addAction(self._host.edit_note_action)
        menu.addAction(self._host.copy_note_action)
        viewport = pane.list_widget.viewport()
        if viewport is None:
            self._release_menu(menu)
            return
        menu.popup(viewport.mapToGlobal(position))

    def _release_menu(self, menu: QtWidgets.QMenu) -> None:
        if self._menu is menu:
            self._menu = None
        if erlab.interactive.utils.qt_is_valid(menu):
            menu.deleteLater()

    def _list_item(self, uid: str) -> QtWidgets.QListWidgetItem:
        item = QtWidgets.QListWidgetItem(self._host._child_node(uid).display_text)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, uid)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        if self._view_mode == _VIEW_MODE_GALLERY:
            item.setIcon(self._gallery_icon(uid))
            item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignHCenter
                | QtCore.Qt.AlignmentFlag.AlignBottom
            )
        return item

    def _gallery_icon(self, uid: str) -> QtGui.QIcon:
        if not self._host._is_figure_uid(uid):
            return QtGui.QIcon(self._placeholder_pixmap())
        node = self._host._child_node(uid)
        tool_window = node.tool_window
        if tool_window is None or not erlab.interactive.utils.qt_is_valid(tool_window):
            pending_preview = node.pending_workspace_tool_preview_image()
            if pending_preview is not None:
                return QtGui.QIcon(self._thumbnail_pixmap(pending_preview[1]))
            return QtGui.QIcon(self._placeholder_pixmap())
        thumbnail = self._tool_thumbnail_pixmap(tool_window)
        if thumbnail is None or thumbnail.isNull():
            return QtGui.QIcon(self._placeholder_pixmap())
        return QtGui.QIcon(thumbnail)

    def _tool_thumbnail_pixmap(
        self, tool_window: erlab.interactive.utils.ToolWindow[typing.Any]
    ) -> QtGui.QPixmap | None:
        if not erlab.interactive.utils.qt_is_valid(tool_window):
            return None
        thumbnail = tool_window._preview_thumbnail_pixmap(
            _FigureCollectionPane.thumbnail_size(self._gallery_size_name)
        )
        if thumbnail is not None and not thumbnail.isNull():
            return self._thumbnail_pixmap(thumbnail)
        preview_pixmap = tool_window.preview_pixmap
        if preview_pixmap is None or preview_pixmap.isNull():
            return None
        return self._thumbnail_pixmap(preview_pixmap)

    def _placeholder_pixmap(self) -> QtGui.QPixmap:
        size = _FigureCollectionPane.thumbnail_size(self._gallery_size_name)
        pixmap = QtGui.QPixmap(size)
        pixmap.fill(self._parent_widget.palette().color(QtGui.QPalette.ColorRole.Base))
        painter = QtGui.QPainter(pixmap)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = QtCore.QRectF(pixmap.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.setPen(
                self._parent_widget.palette().color(QtGui.QPalette.ColorRole.Mid)
            )
            painter.drawRoundedRect(rect, 3.0, 3.0)
        finally:
            painter.end()
        return pixmap

    def _thumbnail_pixmap(self, source_pixmap: QtGui.QPixmap) -> QtGui.QPixmap:
        size = _FigureCollectionPane.thumbnail_size(self._gallery_size_name)
        canvas = QtGui.QPixmap(size)
        canvas.fill(self._parent_widget.palette().color(QtGui.QPalette.ColorRole.Base))
        device_pixel_ratio = source_pixmap.devicePixelRatioF()
        if device_pixel_ratio <= 0.0:
            device_pixel_ratio = 1.0
        source_size = QtCore.QSizeF(
            source_pixmap.width() / device_pixel_ratio,
            source_pixmap.height() / device_pixel_ratio,
        )
        if source_size.isEmpty():
            return canvas
        target_size = QtCore.QSizeF(source_size)
        target_size.scale(
            QtCore.QSizeF(size), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        target_rect = QtCore.QRectF(
            QtCore.QPointF(
                (size.width() - target_size.width()) / 2.0,
                (size.height() - target_size.height()) / 2.0,
            ),
            target_size,
        )
        painter = QtGui.QPainter(canvas)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(
                target_rect,
                source_pixmap,
                QtCore.QRectF(QtCore.QPointF(0.0, 0.0), source_size),
            )
        finally:
            painter.end()
        return canvas
