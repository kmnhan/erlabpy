"""Helper functions for parallel processing."""
import contextlib
import joblib
import joblib._parallel_backends
# from tqdm import tqdm, tqdm_notebook
import tqdm
import time
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

__all__ = ["joblib_progress"]

def is_notebook():
    # http://stackoverflow.com/questions/34091701/determine-if-were-in-an-ipython-notebook-session
    if "IPython" not in sys.modules:  # IPython hasn't been imported
        return False
    from IPython import get_ipython

    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), "kernel", None) is not None

@contextlib.contextmanager
def joblib_progress(file=None, notebook=None, dynamic_ncols=True, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar given as
    argument"""
    
    if file is None:
        file = sys.stdout
    
    if notebook is None:
        notebook = is_notebook()
        
    if notebook:
        tqdm_object = tqdm.notebook.tqdm(iterable=None, dynamic_ncols=dynamic_ncols, file=file)
    else:
        tqdm_object = tqdm.tqdm(iterable=None, dynamic_ncols=dynamic_ncols, file=file, **kwargs)

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


@contextlib.contextmanager
def joblib_qt(signal):
    class QtBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *a, **kwa):
            # dialog.setValue(self.parallel.n_completed_tasks + self.batch_size)
            signal.emit(self.parallel.n_completed_tasks + self.batch_size)
            print("callbackemit")
            return super().__call__(*a, **kwa)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = QtBatchCompletionCallback
    try:
        yield None
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        # tqdm_object.close()


class Emitter(QtCore.QThread):
    """Emitter waits for data from the capitalization process and emits a signal for the UI to update its text."""

    ui_data_available = QtCore.Signal(int)

    def __init__(self, batchcallback):
        super().__init__()
        self.batchcallback = batchcallback

    def run(self):
        while True:
            self.ui_data_available.emit(
                self.batchcallback.parallel.n_completed_tasks
                + self.batchcallback.batch_size
            )


class MultiCallback:
    def __init__(self, *callbacks):
        self.callbacks = [cb for cb in callbacks if cb]

    def __call__(self, out):
        for cb in self.callbacks:
            cb(out)


# class QtBackend(joblib._parallel_backends.MultiprocessingBackend):


class joblib_pg(pg.ProgressDialog):  # joblib.parallel_backend):
    sigProgressUpdated = QtCore.Signal(int)

    def __init__(
        self,
        labelText,
        minimum=0,
        maximum=100,
        cancelText="Cancel",
        parent=None,
        wait=0,
        busyCursor=False,
        disable=False,
        nested=False,
    ):
        pg.ProgressDialog.__init__(
            self,
            labelText=labelText,
            minimum=minimum,
            maximum=maximum,
            cancelText=cancelText,
            parent=parent,
            wait=wait,
            busyCursor=busyCursor,
            disable=disable,
            nested=nested,
        )

        # class QtBackend(joblib._parallel_backends.LokyBackend):
        #     def callback(self, result):

        #         pass
        #         # print("\tImmediateResult function %s" % result)

        #     def apply_async(self, func, callback=None):
        #         cbs = MultiCallback(callback, self.callback)
        #         return super().apply_async(func, cbs)

        # joblib.register_parallel_backend("qtbackend", QtBackend)
        # joblib.parallel_backend.__init__(self, backend="qtbackend")

        self.sigProgressUpdated.connect(self.setValue)
        # self.sigProgressUpdated.connect(self.increment_progress)

    @QtCore.Slot(int)
    def increment_progress(self):
        super().setValue(self.value() + 1)

    @QtCore.Slot(int)
    def setValue(self, val):
        super().setValue(val)

    @QtCore.Slot()
    def __enter__(self):
        pg_dialog = self

        def pg_print_progress(self):
            if self.n_completed_tasks > pg_dialog.value():
                print(self.n_completed_tasks)
                # n_completed = self.n_completed_tasks - pg_dialog.value()
                # for i in range(n_completed):
                # pg_dialog.sigProgressUpdated.emit()
                pg_dialog.sigProgressUpdated.emit(self.n_completed_tasks)

        # class QtBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):

        #     # sigBCC = QtCore.Signal(int)

        # def __init__(self, *a, **kwa):
        #     super().__init__(*a, **kwa)
        #     self._last_val = 0

        #         self.pg_prog_emitter = Emitter(self)
        #         self.pg_prog_emitter.daemon = False
        #         self.pg_prog_emitter.start()
        #         self.pg_prog_emitter.ui_data_available.connect(pg_dialog.setValue)

        # def __call__(self, *a, **kwa):
        #     new_val = self.parallel.n_completed_tasks + self.batch_size
        #     delta = new_val - self._last_val

        #     for _ in range(delta):
        #         pg_dialog.sigProgressUpdated.emit()
        #     self._last_val = new_val
        #     #         # sigBCC.emit(self.parallel.n_completed_tasks + self.batch_size)
        #     #         # print("callbackemit")
        #     return super().__call__(*a, **kwa)
        # self.original_batch_callback = joblib.parallel.BatchCompletionCallBack
        # joblib.parallel.BatchCompletionCallBack = QtBatchCompletionCallback

        self.original_print_progress = joblib.parallel.Parallel.print_progress
        joblib.parallel.Parallel.print_progress = pg_print_progress

        # joblib.parallel_backend.__enter__(self)
        return pg.ProgressDialog.__enter__(self)

    @QtCore.Slot()
    def __exit__(self, *args, **kwargs):
        joblib.parallel.Parallel.print_progress = self.original_print_progress
        # joblib.parallel.BatchCompletionCallBack = self.original_batch_callback
        pg.ProgressDialog.__exit__(self, *args, **kwargs)
        # joblib.parallel_backend.__exit__(self, *args, **kwargs)

    # def qt_print_progress(self):
    #     if self.n_completed_tasks > dialog.value():
    #         signal.emit(self.n_completed_tasks)
    #         # dialog.setValue(self.n_completed_tasks)
    #         # pass
    # original_print_progress = joblib.parallel.Parallel.print_progress
    # joblib.parallel.Parallel.print_progress = qt_print_progress

    # try:
    #     yield dialog
    # finally:
    #     joblib.parallel.Parallel.print_progress = original_print_progress
    #     # tqdm_object.close()
