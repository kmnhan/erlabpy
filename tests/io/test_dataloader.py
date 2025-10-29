import errno
import os
import pathlib
import re
import threading
import time

import numpy as np
import pytest

import erlab
from erlab.io.dataloader import LoaderNotFoundError, UnsupportedFileError


def test_loader(
    qtbot, example_loader, example_data_dir: pathlib.Path, manager_context
) -> None:
    wrong_file = example_data_dir / "data_010.nc"

    with erlab.io.loader_context("example", example_data_dir):
        erlab.io.load(1)

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            str(
                FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), "some_nonexistent_dir"
                )
            )
        ).replace("some_nonexistent_dir", ".*some_nonexistent_dir"),
    ):
        erlab.io.loaders.set_data_dir("some_nonexistent_dir")

    # Test if the reprs are working
    assert repr(erlab.io.loaders).startswith("Name")
    assert erlab.io.loaders._repr_html_().startswith("<div><style>")

    # Test magic methods
    assert len(erlab.io.loaders) > 0
    assert "example" in erlab.io.loaders
    assert isinstance(erlab.io.loaders["example"], example_loader)
    for k, v in erlab.io.loaders.items():
        assert erlab.io.loaders[k] is v

    with pytest.raises(
        LoaderNotFoundError,
        match="Loader for name or alias nonexistent_loader not found in the registry",
    ):
        erlab.io.set_loader("nonexistent_loader")
    with pytest.raises(
        AttributeError,
        match="Loader for name or alias nonexistent_loader not found in the registry",
    ):
        _ = erlab.io.loaders.nonexistent_loader

    # Set loader
    erlab.io.set_loader("example")
    erlab.io.set_data_dir(example_data_dir)
    erlab.io.load(2)

    erlab.io.set_loader(None)

    # Set with attribute
    erlab.io.loaders.current_loader = "example"
    erlab.io.loaders.current_data_dir = example_data_dir
    erlab.io.load(2)

    # Loading nonexistent indices
    nonexistent_file = "data_099.h5"
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            str(
                FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), nonexistent_file
                )
            )
        ).replace(re.escape(nonexistent_file), f".*{re.escape(nonexistent_file)}"),
    ):
        erlab.io.load("data_099.h5")

    with pytest.raises(
        UnsupportedFileError,
        match=UnsupportedFileError._make_msg(
            "example",
            pathlib.Path(wrong_file),
            erlab.io.loaders.example.extensions,
        ),
    ):
        erlab.io.load(wrong_file, single=True)

    # Test if coordinate_attrs are correctly assigned
    mfdata = erlab.io.load(1)
    assert np.allclose(mfdata["sample_temp"].values, 20.0 + np.arange(10))

    df = erlab.io.summarize(display=False)
    assert len(df.index) == 7

    # Test if pretty printing works
    erlab.io.loaders.current_loader.get_styler(df)._repr_html_()

    # Interactive summary
    box = erlab.io.loaders.current_loader._isummarize(df)
    btn_box = box.children[0].children[0]
    assert len(btn_box.children) == 3  # prev, next, load full
    btn_box.children[2].click()  # load full
    btn_box.children[1].click()  # next
    btn_box.children[1].click()  # next
    del box, btn_box

    # Interactive summary with imagetool manager
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        box = erlab.io.loaders.current_loader._isummarize(df)
        btn_box = box.children[0].children[0]
        assert len(btn_box.children) == 4  # prev, next, load full, imagetool
        assert box.children[0].children[1].value == "data_001_S001"
        btn_box.children[3].click()  # imagetool

        qtbot.wait_until(lambda: manager.ntools == 1)

        # Archive nd remove
        manager._imagetool_wrappers[0].archive()
        manager.remove_imagetool(0)
        qtbot.wait_until(lambda: manager.ntools == 0)
        manager.close()


def test_thread_safety():
    potential_loaders = list(erlab.io.loaders.keys())

    def worker(loader_name, thread_id):
        erlab.io.set_loader(loader_name)
        time.sleep(0.1)
        current = erlab.io.loaders.current_loader.name
        if current != loader_name:
            raise RuntimeError(
                f"Thread {thread_id}: Expected loader '{loader_name}', "
                f"but got '{current}'"
            )

    threads = []
    for i, loader_name in enumerate(potential_loaders):
        t = threading.Thread(target=worker, args=(loader_name, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
