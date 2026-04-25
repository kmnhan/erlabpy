import errno
import importlib
import os
import pathlib
import re
import threading
import time
import typing

import numpy as np
import pytest

import erlab
from erlab.io.dataloader import LoaderNotFoundError, UnsupportedFileError


def test_loader(example_loader, example_data_dir: pathlib.Path, monkeypatch) -> None:
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

    shown_data: list[typing.Any] = []

    monkeypatch.setattr(erlab.interactive.imagetool.manager, "is_running", lambda: True)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "show_in_manager",
        lambda data: shown_data.append(data),
    )

    box = erlab.io.loaders.current_loader._isummarize(df)
    btn_box = box.children[0].children[0]
    assert len(btn_box.children) == 4  # prev, next, load full, imagetool
    assert box.children[0].children[1].value == "data_001_S001"
    btn_box.children[3].click()  # imagetool

    assert len(shown_data) == 1
    assert shown_data[0].name == "data_001_S001"


def test_loader_registry_survives_dataloader_reload(
    example_loader, example_data_dir: pathlib.Path
) -> None:
    cached_loaders = erlab.io.loaders
    cached_load = erlab.io.load
    cached_set_loader = erlab.io.set_loader

    cached_set_loader("example")
    erlab.io.set_data_dir(example_data_dir)

    importlib.reload(erlab.io.dataloader)

    assert "example" in erlab.io.loaders
    assert isinstance(erlab.io.loaders["example"], example_loader)
    assert isinstance(cached_loaders, erlab.io.dataloader.LoaderRegistry)
    assert "example" in cached_loaders
    assert cached_loaders["example"] is erlab.io.loaders["example"]
    assert erlab.io.loaders.current_loader is not None
    assert erlab.io.loaders.current_loader.name == "example"
    assert cached_loaders.current_loader is erlab.io.loaders.current_loader
    assert erlab.io.loaders.current_data_dir == example_data_dir
    assert cached_loaders.current_data_dir == example_data_dir

    cached_load(2)
    erlab.io.load(2)

    erlab.io.set_loader(None)
    erlab.io.set_data_dir(None)


def test_lazy_namespace_exports_shared_loader_registry() -> None:
    import erlab.io._namespace as namespace

    assert isinstance(namespace.loaders, erlab.io.dataloader.LoaderRegistry)
    assert namespace.loaders._state is erlab.io.dataloader.loaders._state
    assert namespace.load.__self__ is namespace.loaders
    assert namespace.loader_context.__self__ is namespace.loaders
    assert namespace.set_data_dir.__self__ is namespace.loaders
    assert namespace.set_loader.__self__ is namespace.loaders
    assert namespace.extend_loader.__self__ is namespace.loaders
    assert namespace.summarize.__self__ is namespace.loaders
    assert namespace.loaders._lock is namespace.loaders._state.lock


def test_loader_extensions_keyword_matches_context(
    example_loader, example_data_dir: pathlib.Path
) -> None:
    extensions = {"additional_coords": {"gui_extra": 7.0}}

    with erlab.io.loader_context("example", example_data_dir):
        with erlab.io.extend_loader(**extensions):
            expected = erlab.io.load(2)

        actual = erlab.io.load(
            2,
            loader_extensions={"additional_coords": {"gui_extra": 7.0}},
        )
        assert "gui_extra" in actual.coords
        np.testing.assert_equal(actual["gui_extra"].values, 7.0)

        import xarray.testing

        xarray.testing.assert_identical(actual, expected)
        assert "gui_extra" not in erlab.io.load(2).coords

    loader = erlab.io.loaders["example"]
    assert "gui_extra" not in loader.additional_coords


def test_loader_extensions_validation(
    example_loader, example_data_dir: pathlib.Path
) -> None:
    with erlab.io.loader_context("example", example_data_dir):
        loaded = erlab.io.load(2, loader_extensions={"coordinate_attrs": ["LensMode"]})
        assert "LensMode" in loaded.coords

        with pytest.raises(TypeError):
            erlab.io.load(2, loader_extensions=[])

        with pytest.raises(TypeError):
            erlab.io.load(2, loader_extensions={"unknown": ()})

        with pytest.raises(TypeError):
            erlab.io.load(2, loader_extensions={"additional_coords": ()})

        with pytest.raises(TypeError, match="not a string"):
            erlab.io.load(2, loader_extensions={"coordinate_attrs": "theta"})


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
