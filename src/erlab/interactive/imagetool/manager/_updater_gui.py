from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import zipfile

import packaging.version
import requests
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive.imagetool.manager import _get_updater_settings
from erlab.interactive.imagetool.manager._updater_core import (
    REPO,
    add_update_tmp_dir,
    fetch_latest_release,
    get_full_changelog_from,
    get_install_root,
    verify_sha256,
)


class Downloader(QtCore.QThread):
    progress = QtCore.Signal(int, int)  # bytes_received, total_bytes (-1 if unknown)
    finished_ok = QtCore.Signal(str)  # path to zip
    failed = QtCore.Signal(str)  # error message

    def __init__(
        self,
        url: str,
        out_path: pathlib.Path,
        total_bytes: int | None,
        headers: dict[str, str],
    ):
        super().__init__()
        self.url = url
        self.out_path = out_path
        self.total_bytes = total_bytes or -1
        self.headers = headers
        self._stop = False

    def run(self):
        try:
            with requests.get(
                self.url, stream=True, headers=self.headers, timeout=30
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", self.total_bytes))
                received = 0
                with open(self.out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if self._stop or self.isInterruptionRequested():
                            return
                        if chunk:
                            f.write(chunk)
                            received += len(chunk)
                            self.progress.emit(received, total)
            self.finished_ok.emit(str(self.out_path))
        except Exception as e:
            self.failed.emit(f"Download failed: {e}")

    def stop(self):
        self._stop = True
        self.requestInterruption()


class Extractor(QtCore.QThread):
    progress = QtCore.Signal(int, int)  # bytes_processed, total_bytes (-1 if unknown)
    finished_ok = QtCore.Signal(str)  # path to extracted dir
    failed = QtCore.Signal(str)

    def __init__(self, zip_path: pathlib.Path, out_dir: pathlib.Path):
        super().__init__()
        self.zip_path = zip_path
        self.out_dir = out_dir
        self._stop = False

    def run(self):
        try:
            with zipfile.ZipFile(self.zip_path) as zf:
                infos = zf.infolist()
                total = sum(info.file_size for info in infos if not info.is_dir())
                processed = 0
                for info in infos:
                    if self._stop or self.isInterruptionRequested():
                        return
                    zf.extract(info, self.out_dir)
                    if info.is_dir():
                        continue
                    processed += info.file_size
                    self.progress.emit(processed, total or -1)
            self.finished_ok.emit(str(self.out_dir))
        except Exception as e:
            self.failed.emit(f"Extraction failed: {e}")

    def stop(self):
        self._stop = True
        self.requestInterruption()


class AutoUpdater(QtCore.QObject):
    def __init__(self, current_version: str | None = None):
        super().__init__()
        if current_version is None:
            current_version = erlab.__version__
        self.current_version = current_version

    def check_for_updates(
        self, parent: erlab.interactive.imagetool.manager.ImageToolManager
    ):
        try:
            info = fetch_latest_release()
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                parent, "Update", "Failed to check for updates."
            )
            return

        if not info:
            QtWidgets.QMessageBox.information(
                parent, "Update", "No suitable asset for this platform."
            )
            return

        new = packaging.version.Version(info.tag)
        cur = packaging.version.Version(self.current_version)
        if new <= cur:
            msg_box = parent._make_icon_msgbox()
            msg_box.setText("Up to date!")
            msg_box.setInformativeText(
                f"You are running the latest version (v{self.current_version})."
            )

            msg_box.exec()
            return

        # Show changelog in a custom dialog with Markdown rendering
        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("")
        dlg.setModal(True)

        vbox = QtWidgets.QVBoxLayout(dlg)

        title_label = QtWidgets.QLabel(
            "A new version of ImageTool Manager is available!", dlg
        )
        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)
        title_label.setWordWrap(True)
        vbox.addWidget(title_label)

        info_label = QtWidgets.QLabel(
            f"{info.tag} is available—you have v{self.current_version}. "
            "Would you like to download it now?",
            dlg,
        )
        info_label.setWordWrap(True)
        vbox.addWidget(info_label)

        browser = QtWidgets.QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        body_text = get_full_changelog_from(self.current_version)
        browser.setMarkdown(body_text)
        browser.setMinimumSize(500, 300)
        vbox.addWidget(browser, 1)

        btns = QtWidgets.QDialogButtonBox(dlg)
        btns.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Yes
            | QtWidgets.QDialogButtonBox.StandardButton.No
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)

        if not dlg.exec():
            return

        install_root = get_install_root()
        if sys.platform == "darwin" and not self._macos_location_is_writable(
            install_root
        ):
            QtWidgets.QMessageBox.warning(
                parent,
                "Move to Applications",
                "ImageTool Manager needs to live in the Applications folder "
                "to install updates.\n\n"
                f"Current location: {install_root}\n\n"
                "Please move it to the Applications folder and try again.",
            )
            return

        match QtWidgets.QMessageBox.question(
            parent,
            "Update",
            "The application will download and install the update. "
            "The application will close and relaunch during the process. "
            "Make sure to save your work. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        ):
            case QtWidgets.QMessageBox.StandardButton.No:
                return
            case _:
                pass

        # Choose temp zip path
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="imagetool-manager-update-"))
        add_update_tmp_dir(tmpdir)

        zippath = tmpdir / info.asset.name

        # Download with progress dialog
        token = os.environ.get("ERLAB_GITHUB_TOKEN")
        headers = {
            "User-Agent": f"{REPO}-updater",
            "Accept": "application/octet-stream",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        progress = QtWidgets.QProgressDialog("Downloading…", "Cancel", 0, 100, parent)
        progress.setAutoClose(False)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        total_hint = info.asset.size if info.asset.size > 0 else None
        dl = Downloader(info.asset.download_url, zippath, total_hint, headers)

        factor = 1024 * 1024  # Bytes to MiB

        def _on_prog(done: int, total: int):
            if total <= 0:
                progress.setLabelText(f"Downloading…<br>{done / factor:.1f} MiB")
                progress.setRange(0, 0)  # busy
            else:
                progress.setRange(0, total)
                progress.setValue(done)
                progress.setLabelText(
                    f"Downloading…<br>{done / factor:.1f} / {total / factor:.1f} MiB"
                )

        def _on_fail(msg: str):
            progress.cancel()
            QtWidgets.QMessageBox.critical(parent, "Download failed", msg)

        def _on_ok(path: str):
            progress.close()
            with erlab.interactive.utils.wait_dialog(parent, "Verifying download…"):
                verify_result = verify_sha256(pathlib.Path(path), info.asset.digest)
            if not verify_result:
                QtWidgets.QMessageBox.critical(
                    parent, "Integrity error", "SHA256 mismatch. Aborting."
                )
                return
            self._extract_and_update(pathlib.Path(path), parent)

        dl.progress.connect(_on_prog)
        dl.failed.connect(_on_fail)
        dl.finished_ok.connect(_on_ok)

        def on_cancel():
            dl.stop()

        progress.canceled.connect(on_cancel)
        dl.start()

    def _extract_and_update(
        self,
        zip_path: pathlib.Path,
        parent: erlab.interactive.imagetool.manager.ImageToolManager,
    ):
        install_root = get_install_root()

        self._confirm_install_ready(parent)

        tmpdir = zip_path.parent

        extract_dir = tmpdir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with erlab.interactive.utils.wait_dialog(parent, "Extracting…"):
            if sys.platform == "darwin":
                subprocess.run(
                    ["/usr/bin/ditto", "-x", "-k", zip_path, str(extract_dir)],
                    check=True,
                )
            else:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_dir)

        self._apply_update(extract_dir, install_root, parent)

    def _apply_update(
        self,
        extract_dir: pathlib.Path,
        install_root: pathlib.Path,
        parent: QtWidgets.QWidget | None,
    ):
        pid = os.getpid()
        tmpdir = extract_dir.parent
        settings = _get_updater_settings()
        settings.setValue("version_before_update", self.current_version)
        settings.sync()

        if sys.platform == "darwin":
            new_app = None
            for p in extract_dir.glob("*.app"):
                new_app = p
                break
            if not new_app:
                for p in extract_dir.iterdir():
                    if p.is_dir():
                        candidate = next(p.glob("*.app"), None)
                        if candidate:
                            new_app = candidate
                            break
            if not new_app:
                raise RuntimeError("`.app` bundle not found in extracted zip.")

            # Set executable permissions on extracted files
            app_binary = new_app / "Contents" / "MacOS" / new_app.stem
            st = app_binary.stat()
            app_binary.chmod(st.st_mode | stat.S_IEXEC)

            script = _macos_helper_script(
                new_app=new_app.resolve(), old_app=install_root.resolve(), pid=pid
            )

            script_path = tmpdir / "apply_update.sh"
            script_path.write_text(script, encoding="utf-8")
            script_path.chmod(0o755)

            try:
                subprocess.Popen(
                    ["/bin/bash", str(script_path.resolve())], close_fds=True
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    parent, "Update", f"Failed to start updater: {e}"
                )
                return

        elif sys.platform.startswith("win"):
            # Find installer .exe in extracted dir
            exe = None
            for p in extract_dir.rglob("*.exe"):
                exe = p
                break
            if not exe:
                raise RuntimeError("Installer .exe not found in extracted zip.")

            # Start installer
            subprocess.Popen([str(exe), "/log"], cwd=str(extract_dir))
        else:
            QtWidgets.QMessageBox.critical(
                parent,
                "Update",
                "Auto-update helper not implemented for this OS and architecture.",
            )
            return

        # Close current manager
        manager = erlab.interactive.imagetool.manager._manager_instance
        if manager:
            manager.remove_all_tools()
            manager.close()

        # Quit current app; helper will take over and relaunch
        qapp = QtWidgets.QApplication.instance()
        if qapp:
            qapp.quit()

    @staticmethod
    def _confirm_install_ready(
        parent: erlab.interactive.imagetool.manager.ImageToolManager,
    ) -> None:
        msg = parent._make_icon_msgbox()
        msg.setWindowTitle("Updating ImageTool Manager")
        msg.setText("Ready to install")
        install_btn = msg.addButton(
            "Install Update"
            if sys.platform.startswith("win")
            else "Install and Relaunch",
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        msg.setDefaultButton(install_btn)
        msg.exec()

    @staticmethod
    def _macos_location_is_writable(app_path: pathlib.Path) -> bool:
        resolved = app_path.resolve()
        parent_dir = resolved.parent
        in_applications = False
        try:
            in_applications = resolved.is_relative_to(pathlib.Path("/Applications"))
        except AttributeError:
            in_applications = str(resolved).startswith("/Applications")
        writable = os.access(parent_dir, os.W_OK)
        return in_applications and writable


def _macos_helper_script(new_app: pathlib.Path, old_app: pathlib.Path, pid: int) -> str:
    return f"""#!/bin/bash
set -euo pipefail
NEW_APP="{new_app}"
APP_PATH="{old_app}"
PID={pid}

echo "[updater] Waiting for PID $PID to exit…"
if [ "$PID" -gt 0 ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 0.2; done
fi

echo "[updater] Removing quarantine attribute from new app (if any)."
xattr -dr com.apple.quarantine "$NEW_APP" 2>/dev/null || true

copy_bundle() {{
  src="$1"; dst="$2"
  rm -rf "$dst.tmp" || true
  /usr/bin/ditto "$src" "$dst.tmp"
  rm -rf "$dst" || true
  mv "$dst.tmp" "$dst"
  chmod +x "$dst/Contents/MacOS/"*
}}

echo "[updater] Attempting in-place update: $APP_PATH"
if copy_bundle "$NEW_APP" "$APP_PATH"; then
  echo "[updater] Updated in place."
  /usr/bin/open -a "$APP_PATH"
  exit 0
fi

echo "[updater] In-place failed (likely permissions)."
"""
