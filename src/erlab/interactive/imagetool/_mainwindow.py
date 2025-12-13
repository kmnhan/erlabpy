"""Complete ImageTool window with menubar and keyboard shortcuts.

This module implements :class:`BaseImageTool` and :class:`ImageTool` that contains all
functionality of the ImageTool window, including GUI controls and keyboard shortcuts.

"""

from __future__ import annotations

__all__ = ["BaseImageTool", "ImageTool"]

import json
import os
import typing

import numpy as np
import numpy.typing as npt
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab.interactive.imagetool.slicer import ArraySlicer

_ITOOL_DATA_NAME: str = "<erlab-itool-data>"
#: Name to use for the data variable in cached datasets


class BaseImageTool(QtWidgets.QMainWindow):
    """Base class for an ImageTool window.

    This class combines the :class:`ImageSlicerArea
    <erlab.interactive.imagetool.core.ImageSlicerArea>` and the controls into a single
    window.

    Use this class only if you want to extend ImageTool without the menubar or keyboard
    shortcuts. Otherwise, use :class:`ImageTool
    <erlab.interactive.imagetool.ImageTool>`.

    Parameters
    ----------
    data
        The data to be displayed.
    parent
        The parent widget.
    **kwargs
        Additional keyword arguments to the underlying :class:`ImageSlicerArea
        <erlab.interactive.imagetool.core.ImageSlicerArea>`.

    """

    def __init__(
        self, data=None, parent: QtWidgets.QWidget | None = None, **kwargs
    ) -> None:
        super().__init__(parent=parent)
        self._slicer_area = erlab.interactive.imagetool.core.ImageSlicerArea(
            self, data, **kwargs
        )
        self.setCentralWidget(self.slicer_area)

        self.docks: tuple[QtWidgets.QDockWidget, ...] = tuple(
            QtWidgets.QDockWidget(name, self) for name in ("Cursor", "Color", "Binning")
        )
        for d in self.docks:
            d.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        self.docks[0].setWidget(
            self.widget_box(
                erlab.interactive.imagetool.controls.ItoolCrosshairControls(
                    self.slicer_area, orientation=QtCore.Qt.Orientation.Vertical
                )
            )
        )
        self.docks[1].setWidget(
            self.widget_box(
                erlab.interactive.imagetool.controls.ItoolColormapControls(
                    self.slicer_area
                )
            )
        )
        self.docks[2].setWidget(
            self.widget_box(
                erlab.interactive.imagetool.controls.ItoolBinningControls(
                    self.slicer_area
                )
            )
        )

        for d in self.docks:
            self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, d)
        self.resize(720, 720)

        # Allow writing history after initialization
        self.slicer_area._write_history = True

    @property
    def slicer_area(self) -> erlab.interactive.imagetool.core.ImageSlicerArea:
        """
        The underlying :class:`ImageSlicerArea
        <erlab.interactive.imagetool.core.ImageSlicerArea>`.
        """  # noqa: D205
        return self._slicer_area

    @property
    def array_slicer(self) -> ArraySlicer:
        """
        The underlying :class:`ArraySlicer
        <erlab.interactive.imagetool.slicer.ArraySlicer>`.
        """  # noqa: D205
        return self.slicer_area.array_slicer

    def to_dataset(self) -> xr.Dataset:
        name = self.slicer_area._data.name
        if name is None:
            name = ""

        return self.slicer_area._data.to_dataset(
            name=_ITOOL_DATA_NAME, promote_attrs=False
        ).assign_attrs(
            {
                "itool_state": json.dumps(self.slicer_area.state),
                "itool_title": self.windowTitle(),
                "itool_name": str(name),
                "itool_rect": self.geometry().getRect(),
                "itool_visible": bool(self.isVisible()),
                "erlab_version": erlab.__version__,
            }
        )

    def to_file(self, filename: str | os.PathLike) -> None:
        """Save the data, state, title, and geometry of the tool to a file.

        The saved netcdf dataset can be used to recreate the ImageTool with the class
        method :meth:`from_file`.

        Parameters
        ----------
        filename
            The name of the target netcdf file.

        """
        self.to_dataset().to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, **kwargs) -> typing.Self:
        """Restore a window from a dataset saved using :meth:`to_dataset`.

        Parameters
        ----------
        ds
            The dataset.
        **kwargs
            Additional keyword arguments passed to the constructor.

        """
        saved_version = ds.attrs.get("erlab_version", "0.0.0")
        if erlab.utils.misc.is_newer_version(saved_version):  # pragma: no cover
            erlab.utils.misc.emit_user_level_warning(
                f"This ImageTool was saved with a newer version of erlab "
                f"({saved_version}) than the current version "
                f"({erlab.__version__}). Some features may not be supported.",
            )
        name = ds.attrs["itool_name"]
        name = None if name == "" else name
        tool = cls(
            ds[_ITOOL_DATA_NAME].rename(name),
            state=json.loads(ds.attrs["itool_state"]),
            **kwargs,
        )
        tool.setWindowTitle(ds.attrs["itool_title"])
        tool.setGeometry(*ds.attrs["itool_rect"])
        return tool

    @classmethod
    def from_file(cls, filename: str | os.PathLike, **kwargs) -> typing.Self:
        """Restore a window from a file saved using :meth:`to_file`.

        Parameters
        ----------
        filename
            The name of the file.
        **kwargs
            Additional keyword arguments passed to the constructor.
        """
        return cls.from_dataset(xr.load_dataset(filename, engine="h5netcdf"), **kwargs)

    def duplicate(self, **kwargs) -> typing.Self:
        """Create a duplicate of the current ImageTool window.

        This method creates a new instance of the ImageTool with the same data and state
        as the current one.

        Parameters
        ----------
        kwargs
            Additional keyword arguments passed to the constructor of the new ImageTool.

        Returns
        -------
        tool
            The duplicated ImageTool window.

        """
        kwargs["state"] = self.slicer_area.state.copy()
        new_tool = self.__class__(self.slicer_area._data.copy(), **kwargs)
        new_tool.setGeometry(self.geometry())
        return new_tool

    @QtCore.Slot()
    def move_to_manager(self) -> None:
        from erlab.interactive.imagetool.manager import show_in_manager

        show_in_manager(self.to_dataset())
        self.close()

    # TODO: this is ugly and temporary, fix it
    def widget_box(self, widget: QtWidgets.QWidget, **kwargs) -> QtWidgets.QGroupBox:
        """Create a box that surrounds the given widget.

        Parameters
        ----------
        widget
            The widget to be added to the box.
        **kwargs
            Additional keyword arguments passed to :class:`QtWidgets.QGroupBox`

        Returns
        -------
        group
            The created widget box.

        """
        group = QtWidgets.QGroupBox(**kwargs)
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred
        )
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 3)
        group_layout.setSpacing(3)
        group_layout.addWidget(widget)
        return group

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        # For some reason, the select all and copy shortcuts don't work in vbox menu
        # widgets. The events start at the ItoolGraphicsLayoutWidget and is never passed
        # to menu widgets so we need to intercept them at a higher level.
        if event is not None and event.type() == QtCore.QEvent.Type.ShortcutOverride:
            event = typing.cast("QtGui.QKeyEvent", event)
            focused = QtWidgets.QApplication.focusWidget()
            if isinstance(
                focused, QtWidgets.QAbstractSpinBox | QtWidgets.QLineEdit
            ) and (
                event.matches(QtGui.QKeySequence.StandardKey.SelectAll)
                or event.matches(QtGui.QKeySequence.StandardKey.Copy)
            ):
                QtWidgets.QApplication.sendEvent(focused, event)
                return True

        return super().eventFilter(obj, event)

    def closeEvent(self, evt: QtGui.QCloseEvent | None) -> None:
        self.slicer_area.close_associated_windows()
        super().closeEvent(evt)


class ImageTool(BaseImageTool):
    """The ImageTool window class.

    This class adds the menubar with keyboard shortcuts to :class:`BaseImageTool
    <erlab.interactive.imagetool.BaseImageTool>`. Use this class to create an ImageTool
    window.

    Instead of instantiating this class directly, users should use the convenience
    function :func:`itool <erlab.interactive.imagetool.itool>`.

    Parameters
    ----------
    data
        The data to be displayed.
    **kwargs
        Additional keyword arguments to :class:`BaseImageTool
        <erlab.interactive.imagetool.BaseImageTool>`.

    Signals
    -------
    sigTitleChanged(str)
        Emitted when the title of the window is changed by setting a new data or file.
        It is *not* emitted when the title is changed by other means such as
        :meth:`setWindowTitle`.

    """

    sigTitleChanged = QtCore.Signal(str)  #: :meta private:

    def __init__(self, data=None, **kwargs) -> None:
        super().__init__(data, **kwargs)
        self.__recent_name_filter: str | None = None
        self.__recent_directory: str | None = None

        self._dask_menu = erlab.interactive._dask.DaskMenu(self, "Dask")
        self._history_menu = erlab.interactive.imagetool._history.HistoryMenu(
            self.slicer_area, "History"
        )
        self.initialize_actions()
        self.setMenuBar(ItoolMenuBar(self))

        self.slicer_area.sigDataChanged.connect(self._update_title)
        self._update_title()
        self.slicer_area.installEventFilter(self)

    @property
    def _recent_name_filter(self) -> str | None:
        if self.slicer_area._manager_instance is not None:
            return self.slicer_area._manager_instance._recent_name_filter
        return self.__recent_name_filter

    @_recent_name_filter.setter
    def _recent_name_filter(self, value: str | None) -> None:
        if self.slicer_area._manager_instance is not None:
            self.slicer_area._manager_instance._recent_name_filter = value
        self.__recent_name_filter = value

    @property
    def _recent_directory(self) -> str | None:
        if self.slicer_area._manager_instance is not None:
            return self.slicer_area._manager_instance._recent_directory
        return self.__recent_directory

    @_recent_directory.setter
    def _recent_directory(self, value: str | None) -> None:
        if self.slicer_area._manager_instance is not None:
            self.slicer_area._manager_instance._recent_directory = value
        self.__recent_directory = value

    def initialize_actions(self) -> None:
        self.open_act = QtWidgets.QAction("&Open...", self)
        self.open_act.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_act.triggered.connect(self._open_file)
        self.open_act.setIcon(QtGui.QIcon.fromTheme("document-open"))

        self.save_act = QtWidgets.QAction("&Save As...", self)
        self.save_act.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.save_act.triggered.connect(self._export_file)
        self.save_act.setIcon(QtGui.QIcon.fromTheme("document-save-as"))

        self.close_act = QtWidgets.QAction("&Close", self)
        self.close_act.setShortcut("Ctrl+W")
        self.close_act.triggered.connect(self.hide)
        self.close_act.setIcon(QtGui.QIcon.fromTheme("window-close"))

        self.remove_act = QtWidgets.QAction("Remove from Manager", self)
        self.remove_act.setShortcut(QtGui.QKeySequence.StandardKey.Delete)
        self.remove_act.triggered.connect(self.slicer_area.remove_from_manager)
        self.remove_act.setVisible(self.slicer_area._in_manager)

        self._dask_menu.addSeparator()
        self._dask_menu.addAction(self.slicer_area.compute_act)
        self._dask_menu.addAction(self.slicer_area.chunk_auto_act)
        self._dask_menu.addAction(self.slicer_area.chunk_act)

    @property
    def mnb(self) -> ItoolMenuBar:
        return typing.cast("ItoolMenuBar", self.menuBar())

    def _update_title(self) -> None:
        if self.slicer_area._data is not None:
            title = self.slicer_area.display_name
            self.setWindowTitle(title)
            self.sigTitleChanged.emit(title)

    @QtCore.Slot()
    def _open_file(self, *, native: bool = True) -> None:
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        valid_loaders: dict[str, tuple[Callable, dict]] = (
            erlab.interactive.utils.file_loaders()
        )
        dialog.setNameFilters(valid_loaders.keys())
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if self._recent_name_filter is not None:
            dialog.selectNameFilter(self._recent_name_filter)

        if self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)

        if dialog.exec():
            fname = dialog.selectedFiles()[0]

            # This also sets the recent name filter & dir for the manager through the
            # setter method
            self._recent_name_filter = dialog.selectedNameFilter()
            self._recent_directory = os.path.dirname(fname)

            fn, kargs = valid_loaders[self._recent_name_filter]

            try:
                with erlab.interactive.utils.wait_dialog(self, "Loading..."):
                    self.slicer_area.set_data(
                        fn(fname, **kargs), file_path=fname, load_func=(fn, kargs, 0)
                    )
            except Exception:
                erlab.interactive.utils.MessageDialog.critical(
                    self,
                    "Error",
                    "An error occurred while loading the file.",
                    "Try again with a different loader.",
                )
                self._open_file()
            else:
                self.slicer_area.view_all()

    @QtCore.Slot()
    def _export_file(self, *, native: bool = True) -> None:
        if self.slicer_area._data is None:
            raise ValueError("Data is Empty!")
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        # To avoid importing erlab.io, we define the following functions here
        def _add_igor_scaling(darr: xr.DataArray) -> xr.DataArray:
            scaling = [[1, 0]]
            for i in range(darr.ndim):
                coord: npt.NDArray = np.asarray(darr[darr.dims[i]].values)
                delta = coord[1] - coord[0]
                scaling.append([delta, coord[0]])
            if darr.ndim == 4:
                scaling[0] = scaling.pop(-1)
            darr.attrs["IGORWaveScaling"] = scaling
            return darr

        def _to_netcdf(darr: xr.DataArray, file: str, **kwargs) -> None:
            darr.to_netcdf(file, **kwargs)

        def _to_hdf5(darr: xr.DataArray, file: str, **kwargs) -> None:
            _to_netcdf(_add_igor_scaling(darr), file, **kwargs)

        valid_savers: dict[str, tuple[Callable, dict[str, typing.Any]]] = {
            "xarray HDF5 Files (*.h5)": (
                _to_hdf5,
                {"engine": "h5netcdf", "invalid_netcdf": True},
            ),
            "NetCDF Files (*.nc *.nc4 *.cdf)": (_to_netcdf, {}),
        }

        dialog.setNameFilters(valid_savers.keys())
        dialog.setDirectory(f"{self.slicer_area._data.name}.h5")
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if dialog.exec():
            files = dialog.selectedFiles()
            fn, kargs = valid_savers[dialog.selectedNameFilter()]
            with erlab.interactive.utils.wait_dialog(self, "Saving..."):
                fn(self.slicer_area._data, files[0], **kargs)


class ItoolMenuBar(erlab.interactive.utils.DictMenuBar):
    def __init__(self, tool: ImageTool) -> None:
        super().__init__(tool)
        self.createMenus()

    @property
    def image_tool(self) -> ImageTool:
        return typing.cast("ImageTool", self.parent())

    @property
    def slicer_area(self) -> erlab.interactive.imagetool.core.ImageSlicerArea:
        return self.image_tool.slicer_area

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    def _generate_menu_kwargs(self) -> dict:
        guideline_actions = self.slicer_area.main_image._guideline_actions
        release_notes_action, open_docs_action, report_issue_action = (
            erlab.interactive.utils.make_help_actions(self.image_tool)
        )
        menu_kwargs: dict[str, typing.Any] = {
            "fileMenu": {
                "title": "&File",
                "actions": {
                    "openAct": self.image_tool.open_act,
                    "saveAsAct": self.image_tool.save_act,
                    "sep0": {"separator": True},
                    "closeAct": self.image_tool.close_act,
                    "removeAct": self.image_tool.remove_act,
                    "sep1": {"separator": True},
                    "moveToManagerAct": {
                        "text": "Move to Manager",
                        "triggered": self.image_tool.move_to_manager,
                        "shortcut": "Ctrl+Shift+M",
                    },
                    "SettingsAct": {
                        "text": "Settings",
                        "triggered": self._settings,
                        "shortcut": QtGui.QKeySequence.StandardKey.Preferences,
                        "sep_after": True,
                    },
                },
            },
            "editMenu": {
                "title": "&Edit",
                "actions": {
                    "undoAct": self.slicer_area.undo_act,
                    "redoAct": self.slicer_area.redo_act,
                    "historyMenu": {"menu": self.image_tool._history_menu},
                    "sep": {"separator": True},
                    "&Copy Cursor Values": {
                        "shortcut": "Ctrl+Shift+C",
                        "triggered": self._copy_cursor_val,
                    },
                    "&Copy Cursor Indices": {
                        "shortcut": "Ctrl+Alt+C",
                        "triggered": self._copy_cursor_idx,
                        "sep_after": True,
                    },
                    "Edit Coordinates": {
                        "triggered": self._assign_coords,
                        "sep_after": True,
                    },
                    "Rotate": {"triggered": self._rotate},
                    "Rotation Guidelines": {
                        "actions": {
                            f"guide{i}": act for i, act in enumerate(guideline_actions)
                        },
                        "sep_after": True,
                    },
                    "Crop": {"triggered": self._crop},
                    "Crop to View": {
                        "triggered": self._crop_to_view,
                        "tooltip": "Crop to the current axes view range",
                    },
                    "Average": {"triggered": self._average},
                    "Symmetrize": {"triggered": self._symmetrize},
                    "Correct With Edge...": {"triggered": self._correct_with_edge},
                },
            },
            "viewMenu": {
                "title": "&View",
                "actions": {
                    "viewAllAct": self.slicer_area.view_all_act,
                    "transposeAct": self.slicer_area.transpose_act,
                    "sep0": {"separator": True},
                    "associatedAct": self.slicer_area.associated_coords_act,
                    "sep1": {"separator": True},
                    "snapCursorAct": self.array_slicer.snap_act,
                    "addCursorAct": self.slicer_area.add_cursor_act,
                    "remCursorAct": self.slicer_area.rem_cursor_act,
                    "toggleCursorAct": self.slicer_area.toggle_cursor_act,
                    "cursorMoveMenu": {"title": "Cursor Control", "actions": {}},
                    "cursorColorAct": self.slicer_area.cursor_color_act,
                    "cursorColorCoordAct": self.slicer_area.cursor_colors_by_coord_act,
                    "sep2": {"separator": True},
                    "colorInvertAct": self.slicer_area.reverse_act,
                    "highContrastAct": self.slicer_area.high_contrast_act,
                    "zeroCenterAct": self.slicer_area.zero_centered_act,
                    "lockLevelsAct": self.slicer_area.lock_levels_act,
                    "sep3": {"separator": True},
                    "ktoolAct": self.slicer_area.ktool_act,
                    "meshtoolAct": self.slicer_area.meshtool_act,
                    "sep4": {"separator": True},
                    "Normalize": {"triggered": self._normalize},
                    "resetAct": {
                        "text": "Reset",
                        "triggered": self._reset_filters,
                        "sep_after": True,
                    },
                },
            },
            "daskMenu": {"menu": self.image_tool._dask_menu},
            "helpMenu": {
                "title": "&Help",
                "actions": {
                    "releaseNotesAct": release_notes_action,
                    "help_sep": {"separator": True},
                    "openDocsAct": open_docs_action,
                    "reportIssueAct": report_issue_action,
                },
            },
        }

        menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
            "centerCursorAct"
        ] = self.slicer_area.center_act
        for i, ((t, s), axis, amount) in enumerate(
            zip(
                (
                    ("Shift Current Cursor Up", "Shift+Up"),
                    ("Shift Current Cursor Down", "Shift+Down"),
                    ("Shift Current Cursor Right", "Shift+Right"),
                    ("Shift Current Cursor Left", "Shift+Left"),
                    ("Shift Current Cursor Up × 10", "Ctrl+Shift+Up"),
                    ("Shift Current Cursor Down × 10", "Ctrl+Shift+Down"),
                    ("Shift Current Cursor Right × 10", "Ctrl+Shift+Right"),
                    ("Shift Current Cursor Left × 10", "Ctrl+Shift+Left"),
                ),
                (1, 1, 0, 0) * 2,
                (1, -1, 1, -1, 10, -10, 10, -10),
                strict=True,
            )
        ):
            menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
                f"ShiftCursorAct{i}"
            ] = {
                "text": t,
                "shortcut": s,
                "triggered": lambda *,
                ax=self.slicer_area.main_image.display_axis[axis],
                d=amount: self.slicer_area.step_index(ax, d),
            }
        menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
            "centerAllCursorsAct"
        ] = self.slicer_area.center_all_act
        for i, ((t, s), axis, amount) in enumerate(
            zip(
                (
                    ("Shift Cursors Up", "Alt+Shift+Up"),
                    ("Shift Cursors Down", "Alt+Shift+Down"),
                    ("Shift Cursors Right", "Alt+Shift+Right"),
                    ("Shift Cursors Left", "Alt+Shift+Left"),
                    ("Shift Cursors Up × 10", "Ctrl+Alt+Shift+Up"),
                    ("Shift Cursors Down × 10", "Ctrl+Alt+Shift+Down"),
                    ("Shift Cursors Right × 10", "Ctrl+Alt+Shift+Right"),
                    ("Shift Cursors Left × 10", "Ctrl+Alt+Shift+Left"),
                ),
                (1, 1, 0, 0) * 2,
                (1, -1, 1, -1, 10, -10, 10, -10),
                strict=True,
            )
        ):
            menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
                f"ShiftAllCursorAct{i}"
            ] = {
                "text": t,
                "shortcut": s,
                "triggered": lambda *,
                ax=self.slicer_area.main_image.display_axis[axis],
                d=amount: self.slicer_area.step_index_all(ax, d),
            }
        return menu_kwargs

    def createMenus(self) -> None:
        menu_kwargs = self._generate_menu_kwargs()
        self.add_items(**menu_kwargs)

        # Disable/Enable menus based on context
        self.menu_dict["fileMenu"].aboutToShow.connect(self._file_menu_visibility)
        self.menu_dict["viewMenu"].aboutToShow.connect(self._view_menu_visibility)

    @QtCore.Slot()
    def _file_menu_visibility(self) -> None:
        if self.slicer_area._in_manager:
            visible: bool = False
        else:
            from erlab.interactive.imagetool.manager import is_running

            visible = is_running()

        self.slicer_area.compute_act.setEnabled(self.slicer_area.data_chunked)
        self.action_dict["moveToManagerAct"].setVisible(visible)

    @QtCore.Slot()
    def _view_menu_visibility(self) -> None:
        self.slicer_area.refresh_actions_enabled()
        self.action_dict["resetAct"].setEnabled(
            self.slicer_area._applied_func is not None
        )

    def execute_dialog(self, dialog_cls: type[QtWidgets.QDialog]) -> None:
        dialog = dialog_cls(self.slicer_area)
        dialog.setModal(True)
        self.slicer_area.add_tool_window(
            dialog, update_title=False, transfer_to_manager=False
        )

    @QtCore.Slot()
    def _settings(self) -> None:
        self.execute_dialog(erlab.interactive._options.OptionDialog)

    @QtCore.Slot()
    def _rotate(self) -> None:
        self.execute_dialog(erlab.interactive.imagetool.dialogs.RotationDialog)

    @QtCore.Slot()
    def _crop(self) -> None:
        self.execute_dialog(erlab.interactive.imagetool.dialogs.CropDialog)

    @QtCore.Slot()
    def _average(self) -> None:
        self.execute_dialog(erlab.interactive.imagetool.dialogs.AverageDialog)

    @QtCore.Slot()
    def _symmetrize(self) -> None:
        self.execute_dialog(erlab.interactive.imagetool.dialogs.SymmetrizeDialog)

    @QtCore.Slot()
    def _correct_with_edge(self) -> None:
        self.execute_dialog(erlab.interactive.imagetool.dialogs.EdgeCorrectionDialog)

    @QtCore.Slot()
    def _crop_to_view(self) -> None:
        self.execute_dialog(erlab.interactive.imagetool.dialogs.CropToViewDialog)

    @QtCore.Slot()
    def _normalize(self) -> None:
        self.execute_dialog(erlab.interactive.imagetool.dialogs.NormalizeDialog)

    @QtCore.Slot()
    def _assign_coords(self) -> None:
        self.execute_dialog(erlab.interactive.imagetool.dialogs.AssignCoordsDialog)

    @QtCore.Slot()
    def _reset_filters(self) -> None:
        self.slicer_area.apply_func(None)

    @QtCore.Slot()
    def _copy_cursor_val(self) -> None:
        erlab.interactive.utils.copy_to_clipboard(
            str(
                erlab.utils.misc._convert_to_native(
                    self.slicer_area.array_slicer._values
                )
            )
        )

    @QtCore.Slot()
    def _copy_cursor_idx(self) -> None:
        erlab.interactive.utils.copy_to_clipboard(
            str(
                erlab.utils.misc._convert_to_native(
                    self.slicer_area.array_slicer._indices
                )
            )
        )
