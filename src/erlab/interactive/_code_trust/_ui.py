"""Reusable Qt widgets for executable-document trust."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtWidgets

from erlab.interactive._code_trust._api import manifest_review_text

if typing.TYPE_CHECKING:
    from erlab.interactive._code_trust._core import CodeTrustManifest


class _CodeTrustBanner(QtWidgets.QFrame):
    """Persistent warning for a document whose executable content is paused."""

    review_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("code_trust_banner")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setProperty("codeTrustWarning", True)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        label = QtWidgets.QLabel(
            "Stored executable content is paused because this document is not trusted.",
            self,
        )
        label.setWordWrap(True)
        layout.addWidget(label, 1)

        self.review_button = QtWidgets.QPushButton("Review and Trust…", self)
        self.review_button.setObjectName("code_trust_review_button")
        self.review_button.clicked.connect(self.review_requested)
        layout.addWidget(self.review_button)
        self.setFocusProxy(self.review_button)


def create_code_trust_banner(
    parent: QtWidgets.QWidget | None = None,
) -> _CodeTrustBanner:
    """Create the standard paused-code banner for one document window."""
    return _CodeTrustBanner(parent)


def confirm_code_trust(
    parent: QtWidgets.QWidget,
    manifest: CodeTrustManifest,
    *,
    document_name: str,
    object_name: str,
    window_title: str,
) -> bool:
    """Show the standard executable-content review dialog."""
    subject = document_name.lower()
    message = QtWidgets.QMessageBox(parent)
    message.setObjectName(object_name)
    message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    message.setWindowTitle(window_title)
    message.setText(f"Trust stored executable content in this {subject}?")
    message.setInformativeText(
        "Some listed content can access files, start programs, and use your "
        "Python environment. When present, a serialized lmfit code payload is "
        "identified by a digest because its source cannot be displayed. Trust "
        f"this {subject} only if you know its source."
    )
    details = manifest_review_text(manifest)
    if details:
        message.setDetailedText(details)
    trust_button = message.addButton(
        f"Trust {document_name} and Run Code",
        QtWidgets.QMessageBox.ButtonRole.AcceptRole,
    )
    cancel_button = message.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    message.setDefaultButton(typing.cast("QtWidgets.QPushButton", cancel_button))
    message.exec()
    return message.clickedButton() is trust_button
