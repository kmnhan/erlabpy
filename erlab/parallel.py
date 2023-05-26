"""Helper functions for parallel processing."""

__all__ = ["joblib_progress", "joblib_progress_qt"]

import contextlib
import sys

import joblib
import joblib._parallel_backends
import tqdm
import tqdm.notebook
from qtpy import QtCore


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
        tqdm_object = tqdm.notebook.tqdm(
            iterable=None, dynamic_ncols=dynamic_ncols, file=file, **kwargs
        )
    else:
        tqdm_object = tqdm.tqdm(
            iterable=None, dynamic_ncols=dynamic_ncols, file=file, **kwargs
        )

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
def joblib_progress_qt(signal: QtCore.Signal):
    class QtBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *a, **kwa):
            signal.emit(self.parallel.n_completed_tasks + self.batch_size)
            return super().__call__(*a, **kwa)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = QtBatchCompletionCallback
    try:
        yield None
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
