"""Inspect widget destruction threads on the compatibility 266 Qt wheel."""

import gc
import json
import sys
import threading

import PySide6
from PySide6 import QtCore, QtWidgets


class PythonWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.timer = QtCore.QTimer(self)
        self.timer.start(1000)


app = QtWidgets.QApplication([])
gui_thread = threading.get_ident()
results = []
print(f"Python: {sys.version}", flush=True)
print(f"PySide6: {PySide6.__version__}", flush=True)
print(f"GUI thread: {gui_thread}", flush=True)

for widget_class in (QtWidgets.QWidget, PythonWidget):
    for trigger in ("reference", "gc"):
        destruction_threads = []
        widget = widget_class()
        widget.destroyed.connect(
            lambda _obj=None, observed=destruction_threads: observed.append(
                threading.get_ident()
            ),
            QtCore.Qt.ConnectionType.DirectConnection,
        )
        if trigger == "gc":
            widget.cycle = widget
        holder = [widget]
        del widget

        def release(owned=holder, cause=trigger):
            owned.clear()
            if cause == "gc":
                gc.collect()

        worker = threading.Thread(target=release)
        worker.start()
        worker.join()
        app.processEvents()
        result = {
            "class": widget_class.__name__,
            "trigger": trigger,
            "destruction_threads": destruction_threads,
            "destroyed_on_gui_thread": destruction_threads == [gui_thread],
        }
        results.append(result)
        print(json.dumps(result), flush=True)

if any(not result["destroyed_on_gui_thread"] for result in results):
    raise SystemExit(2)
