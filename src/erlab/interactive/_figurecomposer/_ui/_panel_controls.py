"""Shared controls used by Figure Composer side panels."""

from __future__ import annotations

from qtpy import QtCore, QtWidgets


def _step_toolbar_button(
    parent: QtWidgets.QWidget,
    object_name: str,
    text: str,
    tooltip: str,
) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton(parent)
    button.setObjectName(object_name)
    button.setText(text)
    button.setToolTip(tooltip)
    button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Minimum,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    return button
