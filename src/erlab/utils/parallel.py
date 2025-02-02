"""Helper functions for parallel processing."""

__all__ = ["joblib_progress", "joblib_progress_qt"]

import contextlib
import sys
import typing

if typing.TYPE_CHECKING:
    import joblib
    import tqdm.auto as tqdm
else:
    import lazy_loader as _lazy

    from erlab.utils.misc import LazyImport

    joblib = _lazy.load("joblib")
    tqdm = LazyImport("tqdm.auto")


@contextlib.contextmanager
def joblib_progress(file=None, **kwargs):
    """Patches joblib to report into a tqdm progress bar."""
    import joblib

    if file is None:
        file = sys.stdout

    tqdm_object = tqdm.tqdm(iterable=None, file=file, **kwargs)

    def tqdm_print_progress(self) -> None:
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
def joblib_progress_qt(signal):
    """Context manager for interactive windows.

    The number of completed tasks are emitted by the given signal.
    """

    def qt_print_progress(self) -> None:
        signal.emit(self.n_completed_tasks)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = qt_print_progress

    try:
        yield None
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
