"""Helper functions for parallel processing."""

__all__ = ["joblib_progress", "joblib_progress_qt"]

import contextlib
import sys

import joblib
import joblib._parallel_backends
import tqdm.auto
from qtpy import QtCore


@contextlib.contextmanager
def joblib_progress(file=None, **kwargs):
    """Patches joblib to report into a tqdm progress bar."""
    if file is None:
        file = sys.stdout

    tqdm_object = tqdm.auto.tqdm(iterable=None, file=file, **kwargs)

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
    """Context manager for interactive windows.

    The number of completed tasks are emitted by the given signal.
    """

    def qt_print_progress(self):
        signal.emit(self.n_completed_tasks)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = qt_print_progress

    try:
        yield None
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
