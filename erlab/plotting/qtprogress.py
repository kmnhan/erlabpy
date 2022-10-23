import contextlib
import sys

import joblib
import numpy as np
from joblib import Parallel, delayed
from PySide6 import QtCore, QtGui, QtWidgets


@contextlib.contextmanager
def joblib_qt(widget):
    
    class worker(QtCore.QThread):
        sigProgressUpdated = QtCore.Signal(int)
        def __init__(self):
            QtCore.QThread.__init__(self, parent=QtWidgets.QApplication.instance())
            
    
    thread = worker()
    thread.sigProgressUpdated.connect(widget.progress.setValue)
    # widget.connect(thread, thread.sigProgressUpdated, widget.progress.setValue)
    
    # dialog = QtWidgets.QProgressDialog(
        # "doing something", None, 0, 10000, parent=widget
    # )
    # dialog.setWindowModality(QtCore.Qt.WindowModal)
    # dialog.reset()
    class QtBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        @QtCore.Slot(str)
        def __call__(self, a):
            thread.start()
            # dialog.setValue(self.parallel.n_completed_tasks + self.batch_size)
            thread.sigProgressUpdated.emit(self.parallel.n_completed_tasks + self.batch_size)
            return super().__call__(a)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = QtBatchCompletionCallback
    try:
        yield None
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback


class MyWidget(QtWidgets.QWidget):

    # sigProgressUpdated = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.progress = QtWidgets.QProgressDialog(
            "doing something", None, 0, 1000000, parent=self
        )
        self.progress.setWindowModality(QtCore.Qt.WindowModal)
        # self.sigProgressUpdated.connect(self.progress.setValue)
        self.progress.reset()
        self.button = QtWidgets.QPushButton("Click me!")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.button)
        self.button.clicked.connect(self.do_iter)
    
    def do_iter(self):
        with joblib_qt(self) as p:
            Parallel(n_jobs=3)(delayed(np.sqrt)(i**2) for i in range(1000000))


if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    widget = MyWidget()
    # widget.resize(800, 600)
    widget.show()
    widget.activateWindow()
    widget.raise_()
    QtCore.QTimer.singleShot(600000, qapp.quit)
    qapp.exec()
