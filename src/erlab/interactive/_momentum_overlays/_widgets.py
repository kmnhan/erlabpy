"""Movable pocket and image overlays."""

import math

import pyqtgraph as pg
from qtpy import QtCore

from erlab.interactive.utils import qt_is_valid


class PocketROI(pg.EllipseROI):
    """Translate, stretch, and rotate an ellipse about its center."""

    sigSelected = QtCore.Signal(object)

    def __init__(self, center, radii, rotation=0.0, **kwargs):
        kwargs.setdefault("removable", True)
        super().__init__((0, 0), (1, 1), **kwargs)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.sigClicked.connect(self._selected)
        self.setGeometry(center, radii, rotation)

    def _selected(self, *_args) -> None:
        self.sigSelected.emit(self)

    def geometry(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        center = self.mapToParent(self.size() * 0.5)
        return (center.x(), center.y()), tuple(self.size() * 0.5), float(self.angle())

    def setGeometry(self, center, radii, rotation) -> None:
        rx, ry = (max(1e-6, float(value)) for value in radii)
        angle = math.radians(rotation)
        self.setSize((2 * rx, 2 * ry), update=False)
        self.setAngle(rotation, update=False)
        self.setPos(
            (
                center[0] - rx * math.cos(angle) + ry * math.sin(angle),
                center[1] - rx * math.sin(angle) - ry * math.cos(angle),
            ),
            update=False,
        )
        self.stateChanged()

    def getMenu(self):
        menu = super().getMenu()
        if self.removable:
            self.remove_action = menu.remAct
            self.remove_action.setObjectName("remove_pocket")
            self.remove_action.setData("remove_pocket")
        return menu

    def removeClicked(self) -> None:
        # Commit the edit before another command can run. The overlay editor
        # defers graphics teardown until the menu event handler has returned.
        self._emitRemoveRequest()

    def _emitRemoveRequest(self) -> None:
        if qt_is_valid(self):
            self.sigRemoveRequested.emit(self)


class ImageROI(pg.ROI):
    """Translate, rotate, or uniformly scale an image about its center."""

    def __init__(self, center, width, aspect, rotation=0.0, **kwargs):
        super().__init__((0, 0), (1, 1), **kwargs)
        self._aspect = max(1e-6, float(aspect))
        self.addScaleHandle((1, 1), (0.5, 0.5), lockAspect=True)
        self.addRotateHandle((1, 0.5), (0.5, 0.5))
        self.setGeometry(center, width, rotation=rotation)

    def geometry(self) -> tuple[tuple[float, float], float, float]:
        center = self.mapToParent(self.size() * 0.5)
        return (center.x(), center.y()), float(self.size().x()), float(self.angle())

    def setGeometry(self, center, width, aspect=None, rotation=None) -> None:
        if aspect is not None:
            self._aspect = max(1e-6, float(aspect))
        if rotation is None:
            rotation = self.angle()
        width = max(1e-6, float(width))
        height = width * self._aspect
        angle = math.radians(rotation)
        self.setSize((width, height), update=False)
        self.setAngle(rotation, update=False)
        self.setPos(
            (
                center[0] - width / 2 * math.cos(angle) + height / 2 * math.sin(angle),
                center[1] - width / 2 * math.sin(angle) - height / 2 * math.cos(angle),
            ),
            update=False,
        )
        self.stateChanged()
