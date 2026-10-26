import contextlib
import struct
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

import erlab
from erlab.io.plugins import mbs

_ENERGY_HEADER = "Start K.E.\t10\nStep Size\t0.5\n"
_KRX_HEADER = (
    _ENERGY_HEADER + "ScaleMin\t-1\nScaleMult\t1\n"
    "MapStartX\t-2\nMapEndX\t2\nMapStartY\t10\nMapEndY\t14\nDATA:\n"
)


def _write_krx(
    path: Path,
    frames: npt.ArrayLike,
    *,
    word_size: int = 8,
    map_size: tuple[int, ...] | None = None,
    header: str | list[str] = _KRX_HEADER,
) -> Path:
    """Write small KRX files with word-based image pointers and byte-sized headers."""
    images = np.asarray(frames, dtype="<i4")
    n_images, size_y, size_x = images.shape
    word_format = "I" if word_size == 4 else "Q"
    metadata = (
        b""
        if map_size is None
        else struct.pack(
            f"<{len(map_size) + 2}{word_format}",
            len(map_size),
            len(map_size),
            *map_size,
        )
    )
    position = (1 + 3 * n_images) * word_size + len(metadata)
    headers = [header] * n_images if isinstance(header, str) else header
    entries = []
    blocks = []
    for image, image_header in zip(images, headers, strict=True):
        entries.extend((position // 4, size_y, size_x))
        encoded = image_header.encode("utf-8")
        block = image.tobytes() + struct.pack("<I", len(encoded)) + encoded
        block += bytes(-len(block) % 4)
        blocks.append(block)
        position += len(block)
    path.write_bytes(
        struct.pack(f"<{1 + len(entries)}{word_format}", 3 * n_images, *entries)
        + metadata
        + b"".join(blocks)
    )
    return path


def _check_metadata_only(path: Path, loaded: xr.DataArray) -> None:
    metadata = mbs.load_krax(path, without_values=True)
    assert metadata.sizes == loaded.sizes
    assert metadata.attrs == loaded.attrs
    assert metadata.dtype == np.int32
    xr.testing.assert_identical(
        metadata.coords.to_dataset(), loaded.coords.to_dataset()
    )
    backing = metadata.data
    while isinstance(backing.base, np.ndarray):
        backing = backing.base
    assert backing.size == 1


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("mbs")
    erlab.io.set_data_dir(test_data_dir / "mbs")
    return test_data_dir / "mbs"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("FIX00001_00000.txt", "FIX00001_00000.h5"),
        (1, "FIX00001_00000.h5"),
    ],
)
def test_load(expected_dir, args, expected) -> None:
    loaded = erlab.io.load(**args) if isinstance(args, dict) else erlab.io.load(args)

    xr.testing.assert_identical(
        loaded, xr.load_dataarray(expected_dir / expected, engine="h5netcdf")
    )


@pytest.mark.parametrize("separator", ["\t", ":"])
def test_text_header_metadata(tmp_path, separator) -> None:
    path = tmp_path / "spectrum.txt"
    path.write_text(
        "Detector export\n\n"
        f"Start K.E.            {separator}10\n"
        f"start k.e.            {separator}999\n"
        f"STEP SIZE            {separator}0.5\n"
        f"Enabled{separator}Yes\nDisabled{separator}No\n"
        f"Lens{separator}L4SAng1d5\n"
        "TIMESTAMP: 2026-01-13 10:26\n"
        "\t123\n\t\n"
        "  DATA:\t\n11\n13\n17\n"
    )
    loaded = mbs.load_text(path)
    np.testing.assert_array_equal(loaded.values, [11, 13, 17])
    np.testing.assert_array_equal(loaded.energies, [10, 10.5, 11])
    assert loaded.attrs == {
        "Start K.E.": 10,
        "STEP SIZE": 0.5,
        "Enabled": True,
        "Disabled": False,
        "Lens": "L4SAng1d5",
        "TIMESTAMP:": "2026-01-13 10:26",
        "Unknown": 123,
    }


@pytest.mark.parametrize("marker", ["DATA:\n", ""])
@pytest.mark.parametrize(
    ("header", "body", "energies"),
    [
        pytest.param(_ENERGY_HEADER, "11\n13\n17\n", [10, 10.5, 11], id="intensity"),
        pytest.param(
            _ENERGY_HEADER + "NoS\t1\n",
            "9.99 11\n10.49 13\n10.99 17\n",
            [10, 10.5, 11],
            id="rounded-energy-column",
        ),
        pytest.param(
            "", "10 11\n10.25 13\n10.75 17\n", [10, 10.25, 10.75], id="energy-column"
        ),
        pytest.param("", "11\n13\n17\n", [0, 1, 2], id="unscaled"),
        pytest.param(_ENERGY_HEADER, "11\n", [10], id="single-point"),
    ],
)
def test_text_spectra(tmp_path, marker, header, body, energies) -> None:
    path = tmp_path / "spectrum.txt"
    path.write_text("\n" + header + marker + body)
    warning = (
        pytest.warns(UserWarning, match="energy scale")
        if not header and " " not in body
        else contextlib.nullcontext()
    )
    with warning:
        loaded = mbs.load_text(path)
    assert loaded.dims == ("energies",)
    assert loaded.dtype == np.int32
    np.testing.assert_array_equal(loaded.values, [11, 13, 17][: len(energies)])
    np.testing.assert_array_equal(loaded.energies, energies)


@pytest.mark.parametrize(
    ("header", "energies", "warn"),
    [
        ("Start K.E.\t10\nStep Size\t0.5\nEnd K.E.\t99\n", [10, 10.5, 11], False),
        ("Start K.E.\t10\nEnd K.E.\t12\n", [10, 11, 12], False),
        ("Start K.E.\t10\nStep Size\t0\nEnd K.E.\t12\n", [10, 11, 12], False),
        ("Start K.E.\t10\nStep Size\tinvalid\nEnd K.E.\t12\n", [10, 11, 12], False),
        ("Start K.E.\t10\nStep Size\t-0.5\n", [10, 9.5, 9], False),
        ("Start K.E.\t\nStep Size\t0.5\n", [0, 1, 2], True),
        ("Start K.E.\tNaN\nStep Size\t0.5\n", [0, 1, 2], True),
        ("Start K.E.\t10\nEnd K.E.\t10\n", [0, 1, 2], True),
    ],
)
def test_text_energy_scale_fallbacks(tmp_path, header, energies, warn) -> None:
    path = tmp_path / "spectrum.txt"
    path.write_text(header + "DATA:\n11\n13\n17\n")
    with (
        pytest.warns(UserWarning, match="energy scale")
        if warn
        else contextlib.nullcontext()
    ):
        loaded = mbs.load_text(path)
    np.testing.assert_array_equal(loaded.energies, energies)
    np.testing.assert_array_equal(loaded.values, [11, 13, 17])


@pytest.mark.parametrize(
    ("header", "angles", "warn"),
    [
        ("ScaleMin\t-1\nScaleMult\t0.5\nScaleMax\t99\n", [-1, -0.5, 0], False),
        ("ScaleMin\t-1\nScaleMax\t1\n", [-1, 0, 1], False),
        ("XScaleMin\t-1\nXScaleMult\t0.5\n", [-1, -0.5, 0], False),
        ("SY\t10\nEndY\t14\n", [10, 12, 14], False),
        ("Start Line\t10\nEnd Line\t14\n", [10, 12, 14], False),
        ("", [0, 1, 2], True),
    ],
)
def test_text_angle_scale_fallbacks(tmp_path, header, angles, warn) -> None:
    path = tmp_path / "spectrum.txt"
    path.write_text(_ENERGY_HEADER + header + "NoS\t3\nDATA:\n11 12 13\n21 22 23\n")
    with (
        pytest.warns(UserWarning, match="angle scale")
        if warn
        else contextlib.nullcontext()
    ):
        loaded = mbs.load_text(path)
    assert loaded.dims == ("angles", "energies")
    np.testing.assert_array_equal(loaded.angles, angles)
    np.testing.assert_array_equal(loaded.values, [[11, 21], [12, 22], [13, 23]])


@pytest.mark.parametrize("slices", ["NoS\t2\n", ""])
def test_text_energy_column_with_multiple_angles(tmp_path, slices) -> None:
    path = tmp_path / "spectrum.txt"
    path.write_text(
        _ENERGY_HEADER
        + "ScaleMin\t-1\nScaleMult\t2\n"
        + slices
        + "DATA:\n10 11 12\n10.5 21 22\n11 31 32\n"
    )
    loaded = mbs.load_text(path)
    np.testing.assert_array_equal(loaded.values, [[11, 21, 31], [12, 22, 32]])
    np.testing.assert_array_equal(loaded.energies, [10, 10.5, 11])
    np.testing.assert_array_equal(loaded.angles, [-1, 1])


def test_text_stale_slice_count(tmp_path) -> None:
    path = tmp_path / "spectrum.txt"
    path.write_text(
        _ENERGY_HEADER + "ScaleMin\t-1\nScaleMult\t1\nNoS\t99\n"
        "DATA:\n11 12 13\n21 22 23\n31 32 33\n41 42 43\n"
    )
    loaded = mbs.load_text(path)
    assert loaded.sizes == {"angles": 3, "energies": 4}
    np.testing.assert_array_equal(
        loaded.values, [[11, 21, 31, 41], [12, 22, 32, 42], [13, 23, 33, 43]]
    )


def test_text_constant_energy_map(tmp_path) -> None:
    path = tmp_path / "map.txt"
    path.write_text(
        "NoS\t4\nScaleMin\t-3\nScaleMult\t2\nMapStartX\t-2\nMapEndX\t2\n"
        "DATA:\n11 12 13\n21 22 23\n31 32 33\n41 42 43\n"
    )
    loaded = mbs.load_text(path)
    assert loaded.dims == ("angles", "defl_angles")
    np.testing.assert_array_equal(loaded.angles, [-3, -1, 1, 3])
    np.testing.assert_array_equal(loaded.defl_angles, [-2, 0, 2])
    np.testing.assert_array_equal(
        loaded.values, [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
    )


@pytest.mark.parametrize(
    "contents", ["", "Description\n\n", "DATA:\n", "\t123\nDATA:\n"]
)
def test_text_missing_data(tmp_path, contents) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(contents)
    warning = (
        pytest.warns(UserWarning, match="input contained no data")
        if "DATA:" in contents
        else contextlib.nullcontext()
    )
    with warning, pytest.raises(ValueError, match="does not contain data"):
        mbs.load_text(path)


@pytest.mark.parametrize("word_size", [4, 8])
def test_krax_long_header_and_first_image_metadata(tmp_path, word_size) -> None:
    frames = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    first = _KRX_HEADER.replace(
        "DATA:\n", "TIMESTAMP:\tfirst\nComment\t" + "μ" * 1500 + "\nDATA:\n"
    )
    second = _KRX_HEADER.replace("Start K.E.\t10", "Start K.E.\t99").replace(
        "DATA:\n", "TIMESTAMP:\tsecond\nDATA:\n"
    )
    path = _write_krx(
        tmp_path / "map.krx", frames, word_size=word_size, header=[first, second]
    )
    loaded = mbs.load_krax(path)
    np.testing.assert_array_equal(loaded.values, frames)
    np.testing.assert_array_equal(loaded.energies, [10, 10.5, 11])
    np.testing.assert_array_equal(loaded.defl_angles, [-2, 2])
    assert loaded.attrs["TIMESTAMP:"] == "first"
    assert loaded.attrs["Comment"] == "μ" * 1500
    assert "MapSize" not in loaded.attrs
    _check_metadata_only(path, loaded)


@pytest.mark.parametrize("word_size", [4, 8])
@pytest.mark.parametrize(
    ("map_size", "dimension", "coordinates"),
    [
        (None, "defl_angles", [-2, 0, 2]),
        ((2, 3), "defl_angles", [-2, 0, 2]),
        ((2, 3, 3, 1), "defl_angles", [-2, 0, 2]),
        ((2, 3, 1, 3), "defl_angles_y", [10, 12, 14]),
    ],
)
def test_krax_scan_dimensions(
    tmp_path, word_size, map_size, dimension, coordinates
) -> None:
    frames = np.arange(18, dtype=np.int32).reshape(3, 2, 3)
    path = _write_krx(
        tmp_path / "map.krx", frames, word_size=word_size, map_size=map_size
    )
    loaded = mbs.load_krax(path)
    assert loaded.dims == (dimension, "angles", "energies")
    np.testing.assert_array_equal(loaded.coords[dimension], coordinates)
    np.testing.assert_array_equal(loaded.values, frames)
    _check_metadata_only(path, loaded)


@pytest.mark.parametrize("word_size", [4, 8])
def test_krax_two_axis_map_order(tmp_path, word_size) -> None:
    frames = np.arange(36, dtype=np.int32).reshape(6, 2, 3)
    path = _write_krx(
        tmp_path / "map.krx", frames, word_size=word_size, map_size=(2, 3, 2, 3)
    )
    loaded = mbs.load_krax(path)
    assert loaded.dims == ("defl_angles_y", "defl_angles", "angles", "energies")
    np.testing.assert_array_equal(loaded.defl_angles, [-2, 2])
    np.testing.assert_array_equal(loaded.defl_angles_y, [10, 12, 14])
    for (iy, ix), frame_index in np.ndenumerate([[0, 1], [2, 3], [4, 5]]):
        np.testing.assert_array_equal(
            loaded.isel(defl_angles_y=iy, defl_angles=ix), frames[frame_index]
        )
    _check_metadata_only(path, loaded)


@pytest.mark.parametrize("word_size", [4, 8])
@pytest.mark.parametrize(("n_spin", "map_x"), [(2, 3), (2, 5), (1, 1)])
def test_krax_separate_spin_images(tmp_path, word_size, n_spin, map_x) -> None:
    mapped = map_x == 3
    n_images = n_spin * 3 if mapped else n_spin
    frames = np.arange(n_images * 6, dtype=np.int32).reshape(n_images, 2, 3)
    path = _write_krx(
        tmp_path / "spin.krx",
        frames,
        word_size=word_size,
        map_size=(2, 3, map_x, 1, n_spin),
    )
    loaded = mbs.load_krax(path)
    assert loaded.dims == (
        ("spin", "defl_angles", "angles", "energies")
        if mapped
        else ("spin", "angles", "energies")
    )
    np.testing.assert_array_equal(loaded.spin, list(range(n_spin)))
    for spin in range(n_spin):
        for image in range(3 if mapped else 1):
            selected = loaded.isel(spin=spin)
            if mapped:
                selected = selected.isel(defl_angles=image)
            np.testing.assert_array_equal(
                selected, frames[spin * 3 + image] if mapped else frames[spin]
            )
    assert "SpinChannels" not in loaded.attrs
    _check_metadata_only(path, loaded)


@pytest.mark.parametrize("word_size", [4, 8])
@pytest.mark.parametrize("mapped", [False, True])
def test_krax_packed_spin_rows(tmp_path, word_size, mapped) -> None:
    n_images = 3 if mapped else 1
    frames = np.arange(n_images * 12, dtype=np.int32).reshape(n_images, 4, 3)
    path = _write_krx(
        tmp_path / "spin.krx",
        frames,
        word_size=word_size,
        map_size=(4, 3, 3 if mapped else 5, 1, 2),
    )
    loaded = mbs.load_krax(path)
    assert loaded.dims == (
        ("spin", "defl_angles", "angles", "energies")
        if mapped
        else ("spin", "angles", "energies")
    )
    np.testing.assert_array_equal(loaded.angles, [-1, 0])
    for spin, rows in enumerate((slice(0, 2), slice(2, 4))):
        for image in range(n_images):
            selected = loaded.isel(spin=spin)
            if mapped:
                selected = selected.isel(defl_angles=image)
            np.testing.assert_array_equal(selected, frames[image, rows])
    _check_metadata_only(path, loaded)


def test_krax_spin_angle_scale(tmp_path) -> None:
    frames = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    path = _write_krx(
        tmp_path / "spin.krx",
        frames,
        map_size=(2, 3, 1, 1, 2),
        header=_ENERGY_HEADER + "S0ScaleMin\t-5\nS0ScaleMult\t2\nDATA:\n",
    )
    loaded = mbs.load_krax(path)
    np.testing.assert_array_equal(loaded.angles, [-5, -3])
    np.testing.assert_array_equal(loaded.values, frames)


@pytest.mark.parametrize(
    ("shape", "dimensions"),
    [((1, 1, 3), ("energies",)), ((1, 2, 1), ("angles",)), ((1, 1, 1), ())],
)
def test_krax_singleton_dimensions(tmp_path, shape, dimensions) -> None:
    frames = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    path = _write_krx(tmp_path / "spectrum.krx", frames)
    loaded = mbs.load_krax(path)
    assert loaded.dims == dimensions
    np.testing.assert_array_equal(
        loaded.values, np.arange(np.prod(shape)).reshape(loaded.shape)
    )
    _check_metadata_only(path, loaded)


@pytest.mark.parametrize("header", [_KRX_HEADER.removesuffix("DATA:\n"), ""])
def test_krax_optional_header_fields(tmp_path, header) -> None:
    frames = np.arange(6, dtype=np.int32).reshape(1, 2, 3)
    path = _write_krx(tmp_path / "spectrum.krx", frames, header=header)
    with (
        pytest.warns(UserWarning, match="MBS .* scale")
        if not header
        else contextlib.nullcontext()
    ):
        loaded = mbs.load_krax(path)
    np.testing.assert_array_equal(loaded.values, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(
        loaded.energies, [10, 10.5, 11] if header else [0, 1, 2]
    )
    np.testing.assert_array_equal(loaded.angles, [-1, 0] if header else [0, 1])


@pytest.mark.parametrize("map_size", [(0, 999, 2, 1), (0, 999, 2, 1, 123, 456)])
def test_krax_ignores_unused_map_metadata(tmp_path, map_size) -> None:
    frames = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    path = _write_krx(tmp_path / "map.krx", frames, map_size=map_size)
    contents = bytearray(path.read_bytes())
    struct.pack_into("<Q", contents, 8 * (1 + 3 * 2 + 1), 999)
    path.write_bytes(contents)
    np.testing.assert_array_equal(mbs.load_krax(path).values, frames)


def test_krax_missing_deflector_scale(tmp_path) -> None:
    frames = np.arange(18, dtype=np.int32).reshape(3, 2, 3)
    path = _write_krx(
        tmp_path / "map.krx",
        frames,
        map_size=(2, 3, 3, 1),
        header=_ENERGY_HEADER + "ScaleMin\t-1\nScaleMult\t1\nDATA:\n",
    )
    with pytest.warns(UserWarning, match="deflector X scale"):
        loaded = mbs.load_krax(path)
    np.testing.assert_array_equal(loaded.defl_angles, [0, 1, 2])
    np.testing.assert_array_equal(loaded.values, frames)


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        ("truncated", "Unexpected end"),
        ("empty-index", "index size"),
        ("fractional-index", "index size"),
        ("oversized-index", "index size"),
        ("zero-width", "same nonzero dimensions"),
        ("zero-height", "same nonzero dimensions"),
        ("unequal-images", "same nonzero dimensions"),
        ("index-overlap", "image data exceeds"),
        ("image-overlap", "image data exceeds"),
        ("incomplete-metadata", "Incomplete MBS map dimensions"),
        ("oversized-metadata", "map metadata overlaps"),
        ("oversized-header", "header length exceeds"),
    ],
)
def test_krax_rejects_malformed_files(tmp_path, damage, match) -> None:
    frames = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    path = _write_krx(tmp_path / "invalid.krx", frames, map_size=(2, 3, 2, 1))
    contents = bytearray(path.read_bytes())
    index_end = 8 * 7
    first_image = struct.unpack_from("<Q", contents, 8)[0] * 4
    if damage == "truncated":
        del contents[6:]
    elif damage == "empty-index":
        struct.pack_into("<Q", contents, 0, 0)
    elif damage == "fractional-index":
        struct.pack_into("<Q", contents, 0, 5)
    elif damage == "oversized-index":
        struct.pack_into("<Q", contents, 0, len(contents) * 3)
    elif damage == "zero-width":
        struct.pack_into("<Q", contents, 24, 0)
    elif damage == "zero-height":
        struct.pack_into("<Q", contents, 16, 0)
    elif damage == "unequal-images":
        struct.pack_into("<Q", contents, 40, 4)
    elif damage == "index-overlap":
        struct.pack_into("<Q", contents, 8, index_end // 4 - 1)
    elif damage == "image-overlap":
        struct.pack_into("<Q", contents, 32, (first_image + frames[0].nbytes - 4) // 4)
    elif damage == "incomplete-metadata":
        struct.pack_into("<Q", contents, 8, (index_end + 4) // 4)
    elif damage == "oversized-metadata":
        struct.pack_into("<Q", contents, index_end, 99)
    elif damage == "oversized-header":
        struct.pack_into("<I", contents, first_image + frames[0].nbytes, len(contents))
    path.write_bytes(contents)
    with pytest.raises(ValueError, match=match):
        mbs.load_krax(path)


@pytest.mark.parametrize(
    ("shape", "map_size", "match"),
    [
        ((2, 2, 3), (2, 3, 3, 1), "image count"),
        ((3, 3, 3), (3, 3, 4, 1, 2), "spin channels"),
        ((1, 3, 3), (3, 3, 4, 1, 2), "spin channels"),
    ],
)
def test_krax_rejects_incompatible_layouts(tmp_path, shape, map_size, match) -> None:
    frames = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    path = _write_krx(tmp_path / "invalid.krx", frames, map_size=map_size)
    with pytest.raises(ValueError, match=match):
        mbs.load_krax(path)
