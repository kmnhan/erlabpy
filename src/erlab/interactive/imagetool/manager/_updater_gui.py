from __future__ import annotations

import os
import pathlib
import platform
import stat
import subprocess
import sys
import tempfile
import zipfile

import packaging.version
import requests
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive.imagetool.manager._updater_core import (
    REPO,
    fetch_latest_release,
    get_full_changelog_from,
    verify_sha256,
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def runtime_platform() -> str:
    osname = platform.system()  # "Windows", "Darwin", "Linux"
    machine = platform.machine() or platform.processor()
    # Normalize a few common aliases
    m = machine.lower()
    if m in {"amd64", "x86_64"}:
        if osname == "Darwin":
            return "macos-intel"
        if osname == "Windows":
            return "windows-amd64"
    elif m in {"aarch64", "arm64"}:
        if osname == "Darwin":
            return "macos-arm"
        if osname == "Windows":
            return "windows-arm"
    return f"{osname}-{machine}"


def get_install_root() -> pathlib.Path:
    """Return the root directory where the current app is installed.

    For macOS, this is the path to the .app bundle. For other platforms, the directory
    of the executable is returned.
    """
    if not erlab.utils.misc._IS_PACKAGED:
        return pathlib.Path("/Applications/ImageTool Manager.app").resolve()
    exe = pathlib.Path(sys.executable).resolve()
    if sys.platform == "darwin":
        bundle = exe.parents[2]
    else:
        bundle = exe.parent
    return bundle


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
                        if self._stop:
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


class AutoUpdater(QtCore.QObject):
    def __init__(self, current_version: str | None = None):
        super().__init__()
        if current_version is None:
            current_version = erlab.__version__
        self.current_version = current_version

    def check_for_updates(self, parent: QtWidgets.QWidget):
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
            QtWidgets.QMessageBox.information(
                parent,
                "Up to date",
                f"You are running the latest version (v{self.current_version}).",
            )
            return

        # Show changelog in a custom dialog with Markdown rendering
        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("Update available")
        dlg.setModal(True)

        vbox = QtWidgets.QVBoxLayout(dlg)

        title_label = QtWidgets.QLabel(
            f"Version {info.tag} is available. You have v{self.current_version}.", dlg
        )
        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)
        title_label.setWordWrap(True)
        vbox.addWidget(title_label)

        info_label = QtWidgets.QLabel(
            "Do you want to download and install it now?", dlg
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

        match QtWidgets.QMessageBox.question(
            parent,
            "Update",
            "The application will download and install the update automatically. "
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
        zippath = tmpdir / info.asset.name

        # Download with progress dialog
        token = os.environ.get("ERLAB_GITHUB_TOKEN")
        headers = {
            "User-Agent": f"{REPO}-updater",
            "Accept": "application/octet-stream",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        progress = QtWidgets.QProgressDialog(
            "Downloading update…", "Cancel", 0, 100, parent
        )
        progress.setAutoClose(False)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        total_hint = info.asset.size if info.asset.size > 0 else None
        dl = Downloader(info.asset.download_url, zippath, total_hint, headers)

        def _on_prog(done: int, total: int):
            if total <= 0:
                progress.setLabelText(f"Downloading… {done // (1024 * 1024)} MB")
                progress.setRange(0, 0)  # busy
            else:
                progress.setRange(0, total)
                progress.setValue(done)

        def _on_fail(msg: str):
            progress.cancel()
            QtWidgets.QMessageBox.critical(parent, "Download failed", msg)

        def _on_ok(path: str):
            progress.close()
            if not verify_sha256(pathlib.Path(path), info.asset.digest):
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

    def _extract_and_update(self, zip_path: pathlib.Path, parent: QtWidgets.QWidget):
        install_root = get_install_root()

        tmpdir = zip_path.parent

        print(f"Creating extraction dir in {tmpdir}")
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
            print(f"Set executable permissions on {app_binary}")

            script = _macos_helper_script(
                new_app=new_app.resolve(),
                old_app=install_root.resolve(),
                tmp_dir=tmpdir.resolve(),
                pid=pid,
            )

            script_dir = pathlib.Path(
                tempfile.mkdtemp(prefix="imagetool-manager-update-script-")
            )
            script_path = pathlib.Path(script_dir) / "apply_update.sh"
            script_path.write_text(script, encoding="utf-8")
            script_path.chmod(0o755)

            try:
                subprocess.Popen(["/bin/bash", str(script_path)], close_fds=True)
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


def _macos_helper_script(
    new_app: pathlib.Path,
    old_app: pathlib.Path,
    tmp_dir: pathlib.Path,
    pid: int,
) -> str:
    return f"""#!/bin/bash
set -euo pipefail
NEW_APP="{new_app}"
APP_PATH="{old_app}"
TMPDIR="{tmp_dir}"
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
  {('/usr/bin/open -a "$APP_PATH"')}
  exit 0
fi

# Cleanup payloads
echo "[updater] In-place failed (likely permissions)."

"""
