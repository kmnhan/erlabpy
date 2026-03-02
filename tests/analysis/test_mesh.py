import numpy as np
import pytest
import xarray as xr

from erlab.analysis import mesh
from erlab.io.exampledata import generate_gold_edge


def test_find_peaks_detects_known_locations() -> None:
    arr = np.zeros((64, 64))
    arr[10, 50] = 10.0
    arr[25, 45] = 8.0

    peaks = mesh.find_peaks(arr, bins=1, n_peaks=2, min_distance=5)

    assert tuple(peaks[0]) == (32, 32)
    assert {tuple(p) for p in peaks[1:]} == {(25, 45), (10, 50)}


def test_find_peaks_reflects_and_plots_upper_half_points() -> None:
    import matplotlib

    matplotlib.use("Agg")

    arr = np.zeros((16, 16))
    arr[2, 2] = 5.0

    peaks = mesh.find_peaks(arr, bins=1, n_peaks=1, min_distance=1, plot=True)

    assert tuple(peaks[0]) == (8, 8)
    assert peaks[1, 0] >= arr.shape[0] // 2
    assert (peaks[1] != np.array([2, 2])).any()

    import matplotlib.pyplot as plt

    plt.close("all")


def test_find_local_maxima_stops_when_num_peaks_reached() -> None:
    image = np.zeros((9, 9))
    image[4, 4] = 9.0
    image[1, 1] = 8.0
    image[1, 7] = 7.0

    coords = mesh._find_local_maxima(image, min_distance=1, num_peaks=2)

    assert coords.shape == (2, 2)
    assert tuple(coords[0]) == (4, 4)
    assert {tuple(c) for c in coords[1:]} <= {(1, 1), (1, 7)}


def test_higher_order_peaks_respects_bounds() -> None:
    first_order = np.array([[16, 16], [16, 20], [16, 12]])

    peaks = mesh.higher_order_peaks(
        first_order, order=2, shape=(32, 32), only_upper=True, include_center=True
    )

    assert tuple(peaks[0]) == (16, 16)
    assert len(peaks) == len({tuple(p) for p in peaks})
    assert (peaks[:, 0] <= 16).all()
    assert {tuple(p) for p in peaks[:3]} == {(16, 16), (16, 20), (16, 12)}


def test_extract_blob_mask_and_fit_moments() -> None:
    rng = np.random.default_rng(0)
    surface = rng.normal(0, 0.02, (20, 20)) + 0.1
    surface[9:12, 7:10] += 1.5

    mask = mesh._extract_blob_mask(surface, (10, 8), roi_hw=4, k=1.0)
    center, sig_major, sig_minor, theta = mesh._fit_moments(mask, (10, 8))

    assert mask.any()
    assert np.allclose(center, (10, 8), atol=1.0)
    assert sig_major > 0
    assert sig_minor > 0
    assert -np.pi / 2 <= theta <= np.pi / 2

    gaussian_notch = mesh._rotated_gaussian_notch(
        surface.shape, center, sig_major, sig_minor, theta
    )
    assert gaussian_notch.shape == surface.shape
    assert 0.0 <= gaussian_notch.min() < gaussian_notch.max() <= 1.0


def test_extract_blob_mask_handles_empty_roi_and_moment_fallback() -> None:
    rng = np.random.default_rng(42)
    arr = rng.uniform(0.0, 0.01, (6, 6))

    mask = mesh._extract_blob_mask(arr, (2, 2), roi_hw=1, k=1_000.0)
    assert not mask.any()

    fallback = mesh._fit_moments(mask, (2, 2))
    center, sig_major, sig_minor, theta = fallback
    assert center == (0, 0)
    assert sig_major == pytest.approx(2.0)
    assert sig_minor == pytest.approx(2.0)
    assert theta == pytest.approx(0.0)

    notch = mesh._rotated_gaussian_notch(arr.shape, center, sig_major, sig_minor, theta)
    assert notch.shape == arr.shape
    assert np.isfinite(notch).all()


def test_auto_correct_curvature_reduces_edge_variation() -> None:
    alpha = np.linspace(-1, 1, 40)
    ev = np.linspace(0, 1, 50)
    data = np.zeros((alpha.size, ev.size))

    for idx, _alpha in enumerate(alpha):
        edge = 20 + idx // 8
        data[idx, edge:] = 1.0

    arr = xr.DataArray(data, coords={"alpha": alpha, "eV": ev}, dims=("alpha", "eV"))
    edge_positions = (arr.diff("eV") > 0).argmax("eV")

    shift_arr, corrected = mesh.auto_correct_curvature(arr, poly_deg=2)
    corrected_edges = (corrected.diff("eV") > 0).argmax("eV")

    assert corrected.shape == arr.shape
    assert shift_arr.max() > 0
    assert corrected_edges.std().item() < edge_positions.std().item()


def test_remove_mesh_full_output_shapes_and_symmetry() -> None:
    height = width = 32
    alpha = np.arange(height)
    ev = np.arange(width)

    yy, xx = np.meshgrid(alpha, ev, indexing="ij")
    base = np.exp(-((yy - height / 2) ** 2 + (xx - width / 2) ** 2) / 200)
    mesh_pattern = 1 + 0.1 * np.cos(2 * np.pi * ev / 8)
    mesh_pattern = np.tile(mesh_pattern, (height, 1))
    data = base * mesh_pattern

    arr = xr.DataArray(data, coords={"alpha": alpha, "eV": ev}, dims=("alpha", "eV"))
    center = np.array([height // 2, width // 2])
    first_order = np.array(
        [center, center + np.array([0, 4]), center + np.array([0, -4])]
    )

    expected_peaks = mesh.higher_order_peaks(
        first_order,
        order=1,
        shape=arr.shape,
        only_upper=True,
        include_center=False,
    )

    (
        corrected,
        mesh_da,
        shift_arr,
        log_mag,
        log_mag_corr,
        peaks,
        mask,
    ) = mesh.remove_mesh(
        arr,
        first_order_peaks=first_order,
        order=1,
        n_pad=0,
        roi_hw=8,
        k=0.1,
        feather=1.0,
        full_output=True,
    )

    assert corrected.shape == arr.shape == mesh_da.shape
    assert shift_arr is None
    assert np.array_equal(peaks, expected_peaks)
    assert mask.shape == arr.shape
    assert mask.min() < 1.0
    assert np.allclose(mask, np.flip(np.flip(mask, axis=0), axis=1))
    assert np.isfinite(corrected).all()
    assert log_mag.shape == arr.shape
    assert log_mag_corr.shape == arr.shape


def test_remove_mesh_auto_peaks_with_edge_correction() -> None:
    alpha = np.linspace(-1, 1, 32)
    ev = np.linspace(0, 1, 32)
    yy, xx = np.meshgrid(alpha, ev, indexing="ij")
    data = (1 + 0.2 * np.cos(2 * np.pi * xx * 4)) * (1 + 0.05 * yy)
    arr = xr.DataArray(data, coords={"alpha": alpha, "eV": ev}, dims=("alpha", "eV"))

    corrected, mesh_da = mesh.remove_mesh(
        arr,
        order=1,
        n_pad=8,
        roi_hw=6,
        k=0.1,
        feather=0.5,
        undo_edge_correction=True,
    )

    assert corrected.shape == arr.shape == mesh_da.shape
    assert np.isfinite(corrected).all()
    assert np.isfinite(mesh_da).all()


def test_remove_mesh_validates_dimensions() -> None:
    bad = xr.DataArray(np.ones((4, 4)), dims=("x", "y"))

    with pytest.raises(ValueError, match=r"alpha.*eV"):
        mesh.remove_mesh(bad)


def test_remove_mesh_unknown_method_raises() -> None:
    arr = xr.DataArray(
        np.ones((8, 8)),
        coords={"alpha": range(8), "eV": range(8)},
        dims=("alpha", "eV"),
    )

    with pytest.raises(ValueError, match="Unknown mesh removal method"):
        mesh.remove_mesh(
            arr,
            first_order_peaks=[[4, 4], [4, 5], [5, 4]],
            method="invalid",  # type: ignore[arg-type]
        )


def test_remove_mesh_rejects_degenerate_first_order_peaks() -> None:
    arr = xr.DataArray(
        np.ones((8, 8)),
        coords={"alpha": range(8), "eV": range(8)},
        dims=("alpha", "eV"),
    )

    with pytest.raises(ValueError, match="distinct first-order peaks"):
        mesh.remove_mesh(arr, first_order_peaks=[[4, 4], [0, 0], [0, 0]])


def test_remove_mesh_rejects_invalid_auto_detected_peaks(monkeypatch) -> None:
    arr = xr.DataArray(
        np.ones((8, 8)),
        coords={"alpha": range(8), "eV": range(8)},
        dims=("alpha", "eV"),
    )

    def _bad_find_peaks(*args, **kwargs):
        return np.array([[4, 4], [0, 0], [0, 0]], dtype=np.intp)

    monkeypatch.setattr(mesh, "find_peaks", _bad_find_peaks)

    with pytest.raises(ValueError, match="distinct first-order peaks"):
        mesh.remove_mesh(arr, first_order_peaks=None, n_pad=0)


@pytest.mark.parametrize("method", ["constant", "gaussian", "circular"])
def test_remove_mesh_on_realistic_gold_edge_with_mesh(method) -> None:
    shape = (928, 1064)

    clean = generate_gold_edge(shape, noise=False, add_mesh=False, count=2_000_000)
    clean = clean.transpose("alpha", "eV")

    meshy = generate_gold_edge(
        shape,
        noise=False,
        add_mesh=True,
        count=2_000_000,
        mesh_params={"pitch": 12.0, "duty": 0.85, "rotate": 22.0, "amplitude": 0.2},
    ).transpose("alpha", "eV")

    corrected, mesh_da = mesh.remove_mesh(
        meshy,
        first_order_peaks=[[464, 532], [378, 571], [344, 475]],
        order=6,
        n_pad=90,
        roi_hw=18,
        k=0.1,
        feather=2.0,
        method=method,
    )
    before_std = float((meshy / clean).values.std())
    after_std = float((corrected / clean).values.std())

    assert corrected.shape == clean.shape == mesh_da.shape
    assert before_std > 0.01
    assert after_std < before_std
    assert 0.99 < mesh_da.mean() < 1.01
