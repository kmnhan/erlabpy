import importlib
import pathlib
import subprocess
import sys
import textwrap
import types

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

import erlab.analysis as era

_denoise = importlib.import_module("erlab.analysis.image.ml._denoise")


class _IdentitySession:
    def __init__(self):
        self.inputs = []
        self.loaded_threads = []

    def get_inputs(self):
        return [
            types.SimpleNamespace(
                name="image", type="tensor(float)", shape=[1, 1, "height", "width"]
            )
        ]

    def get_outputs(self):
        return [
            types.SimpleNamespace(
                name="prediction", type="tensor(float)", shape=[1, 1, "h", "w"]
            )
        ]

    def run(self, output_names, input_feed):
        assert output_names == ["prediction"]
        assert set(input_feed) == {"image"}
        image = input_feed["image"]
        assert image.dtype == np.float32
        assert image.flags.c_contiguous
        self.inputs.append(image.copy())
        return [image.copy()]


class _AsymmetricSession(_IdentitySession):
    def run(self, output_names, input_feed):
        image = super().run(output_names, input_feed)[0]
        energy_weight = np.linspace(0.5, 1.5, image.shape[2], dtype=np.float32)
        return [image * energy_weight[None, None, :, None] + 10.0]


@pytest.fixture
def identity_session(monkeypatch):
    session = _IdentitySession()

    def load_session(model_path, num_threads):
        session.loaded_threads.append(num_threads)
        return pathlib.Path("identity.onnx"), session

    monkeypatch.setattr(_denoise, "_load_session", load_session)
    return session


@pytest.fixture
def asymmetric_session(monkeypatch):
    session = _AsymmetricSession()

    def load_session(model_path, num_threads):
        session.loaded_threads.append(num_threads)
        return pathlib.Path("asymmetric.onnx"), session

    monkeypatch.setattr(_denoise, "_load_session", load_session)
    return session


@pytest.fixture
def image():
    return xr.DataArray(
        np.arange(1.0, 13.0).reshape(3, 4),
        dims=("eV", "alpha"),
        coords={
            "eV": xr.DataArray([2.0, 1.0, 0.0], dims="eV", attrs={"units": "eV"}),
            "alpha": [0.0, 2.0, 4.0, 6.0],
            "angle_offset": ("alpha", [0.1, 0.2, 0.3, 0.4]),
            "hv": 21.2,
        },
        name="spectrum",
        attrs={"sample": {"label": "test"}},
    )


@pytest.fixture
def deferred_image(image):
    scales = xr.DataArray([1.0, 3.0, 7.0], dims="beta", coords={"beta": [-1, 0.2, 2.7]})
    eager = (image.drop_vars("angle_offset") * scales).transpose("alpha", "beta", "eV")
    reads = []

    def read_plane(index):
        reads.append(index)
        return eager.isel(beta=index).transpose("eV", "alpha").values.copy()

    planes = [
        da.from_delayed(dask.delayed(read_plane)(index), shape=(3, 4), dtype=float)
        for index in range(3)
    ]
    lazy = eager.copy(data=da.stack(planes).transpose(2, 0, 1))
    return lazy, eager, reads


def test_denoise_imports_do_not_require_onnxruntime():
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import xarray as xr
        import erlab.analysis as era

        assert "erlab.analysis.image.ml" not in sys.modules
        image = era.image
        assert callable(image.gaussian_filter)
        assert {"gaussian_filter", "ml"} <= set(image.__all__)
        assert {"gaussian_filter", "ml", "_ndsavgol_coeffs"} <= set(dir(image))
        assert "erlab.analysis.image.ml" not in sys.modules
        ml = image.ml
        assert "erlab.analysis.image.ml._denoise" not in sys.modules
        denoise = ml.denoise
        from erlab.analysis.image.ml import denoise as imported
        assert denoise is imported
        assert callable(denoise)
        assert "onnxruntime" not in sys.modules

        sys.modules["onnxruntime"] = None
        try:
            data = xr.DataArray(np.ones((4, 6)), dims=("eV", "alpha"))
            denoise(data, resize=(4, 6), tta=False)
        except ImportError as exc:
            assert "erlab[ml]" in str(exc)
        else:
            raise AssertionError("Denoising must require ONNX Runtime")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("tta", [False, True])
def test_denoise_preserves_data_coordinates_and_metadata(image, identity_session, tta):
    original = image.copy(deep=True)
    result = era.image.ml.denoise(image, resize=image.shape, tta=tta)

    np.testing.assert_allclose(result.values, image.values, rtol=1e-6)
    xr.testing.assert_identical(result.coords.to_dataset(), image.coords.to_dataset())
    assert result.dims == image.dims
    assert result.name == "spectrum_denoised"
    assert result.attrs["sample"] == image.attrs["sample"]
    assert result.attrs["denoising_model"] == "arpes-denoiser.onnx"
    assert result.attrs["denoising_tta"] == int(tta)
    assert result.attrs["denoising_along"] == "alpha"
    assert len(identity_session.inputs) == (8 if tta else 1)
    assert {value.shape for value in identity_session.inputs} == (
        {(1, 1, 3, 4), (1, 1, 4, 3)} if tta else {(1, 1, 3, 4)}
    )
    for value in identity_session.inputs:
        assert value.sum() == pytest.approx(10_000)

    result.attrs["sample"]["label"] = "changed"
    result.eV.attrs["units"] = "changed"
    result.values[0, 0] = -1.0
    xr.testing.assert_identical(image, original)


def test_denoise_resizes_coordinates_and_drops_only_affected_auxiliary(
    image, identity_session
):
    with pytest.warns(UserWarning, match="angle_offset"):
        result = era.image.ml.denoise(
            image, resize=(3, 8), restore_size=False, tta=False
        )

    np.testing.assert_allclose(result.values, np.repeat(image.values / 2, 2, axis=1))
    assert result.sum().item() == pytest.approx(image.sum().item())
    xr.testing.assert_identical(result.eV, image.eV)
    np.testing.assert_allclose(result.alpha, np.arange(8) - 0.5)
    assert "angle_offset" not in result.coords
    assert result.hv.item() == image.hv.item()


def test_denoise_resizes_descending_axis_and_restores_original_grid(
    image, identity_session
):
    resized = era.image.ml.denoise(image, resize=(6, 4), restore_size=False, tta=False)
    np.testing.assert_allclose(resized.eV, [2.25, 1.75, 1.25, 0.75, 0.25, -0.25])
    xr.testing.assert_identical(resized.angle_offset, image.angle_offset)

    restored = era.image.ml.denoise(image, resize=(6, 8), tta=False)
    np.testing.assert_allclose(restored.values, image.values, rtol=1e-6)
    xr.testing.assert_identical(restored.coords.to_dataset(), image.coords.to_dataset())


def test_denoise_supports_alternative_along_dimension(image, identity_session):
    data = image.rename(alpha="beta").transpose("beta", "eV")
    result = era.image.ml.denoise(data, along="beta", resize=(3, 4), tta=False)

    assert result.dims == data.dims
    assert result.attrs["denoising_along"] == "beta"
    np.testing.assert_allclose(result.values, data.values, rtol=1e-6)
    xr.testing.assert_identical(result.coords.to_dataset(), data.coords.to_dataset())
    assert identity_session.inputs[0].shape == (1, 1, 3, 4)


@pytest.mark.parametrize("restore_size", [False, True])
def test_denoise_transpose_keeps_model_plane_and_resize_order(
    image, asymmetric_session, restore_size
):
    data = image.drop_vars("angle_offset")
    expected = era.image.ml.denoise(
        data, resize=(6, 8), restore_size=restore_size, tta=False
    )
    transposed = data.transpose("alpha", "eV")
    result = era.image.ml.denoise(
        transposed, resize=(6, 8), restore_size=restore_size, tta=False
    )

    assert result.dims == transposed.dims
    xr.testing.assert_identical(result.transpose(*expected.dims), expected)
    assert len(asymmetric_session.inputs) == 2
    assert asymmetric_session.inputs[0].shape == (1, 1, 6, 8)
    np.testing.assert_array_equal(
        asymmetric_session.inputs[0], asymmetric_session.inputs[1]
    )


@pytest.mark.parametrize("ndim", [3, 4])
def test_denoise_batches_match_independent_slices(image, asymmetric_session, ndim):
    scales = xr.DataArray(
        [1.0, 4.0, 20.0], dims="temperature", coords={"temperature": [10, 40, 300]}
    )
    data = image * scales
    extra_dims = ["temperature"]
    if ndim == 4:
        data = data * xr.DataArray(
            [0.5, 3.0], dims="polarization", coords={"polarization": ["s", "p"]}
        )
        extra_dims.append("polarization")
    data = data.transpose(*extra_dims[::-1], "alpha", "eV")
    original = data.copy(deep=True)

    result = era.image.ml.denoise(data, resize=(6, 8), tta=False)

    assert result.dims == data.dims
    xr.testing.assert_identical(result.coords.to_dataset(), data.coords.to_dataset())
    slice_shape = tuple(data.sizes[dim] for dim in extra_dims)
    assert len(asymmetric_session.inputs) == np.prod(slice_shape)
    for tensor in asymmetric_session.inputs:
        assert tensor.shape == (1, 1, 6, 8)
        assert tensor.sum() == pytest.approx(10_000)
    for index in np.ndindex(slice_shape):
        selection = dict(zip(extra_dims, index, strict=True))
        expected = era.image.ml.denoise(data.isel(selection), resize=(6, 8), tta=False)
        xr.testing.assert_allclose(result.isel(selection), expected)
    xr.testing.assert_identical(data, original)


def test_denoise_resizes_only_core_dimensions(image, identity_session):
    data = image.expand_dims(scan=["forward", "reverse"], delay=[-1.0, 0.2, 2.7])
    data = data.assign_coords(
        scan_quality=("scan", [1, 2]),
        frame_id=(("scan", "delay"), np.arange(6).reshape(2, 3)),
        energy_offset=("eV", [0.1, 0.2, 0.3]),
        drift=(("delay", "alpha"), np.ones((3, 4))),
    ).transpose("scan", "alpha", "delay", "eV")
    original = data.copy(deep=True)

    with pytest.warns(UserWarning, match="angle_offset"):
        result = era.image.ml.denoise(
            data, resize=(6, 8), restore_size=False, tta=False
        )

    assert result.dims == data.dims
    assert result.sizes == {"scan": 2, "alpha": 8, "delay": 3, "eV": 6}
    np.testing.assert_allclose(result.eV, [2.25, 1.75, 1.25, 0.75, 0.25, -0.25])
    np.testing.assert_allclose(result.alpha, np.arange(8) - 0.5)
    for name in ("scan", "delay", "scan_quality", "frame_id", "hv"):
        xr.testing.assert_identical(result[name], data[name])
    assert not {"angle_offset", "energy_offset", "drift"}.intersection(result.coords)
    xr.testing.assert_identical(data, original)


@pytest.mark.parametrize("restore_size", [False, True])
def test_denoise_dask_is_lazy_and_matches_eager(
    deferred_image, asymmetric_session, restore_size
):
    lazy, eager, reads = deferred_image
    original_data = lazy.data
    original_chunks = lazy.chunks
    result = era.image.ml.denoise(
        lazy, resize=(6, 8), restore_size=restore_size, tta=False
    )

    assert isinstance(result.data, da.Array)
    assert reads == asymmetric_session.inputs == asymmetric_session.loaded_threads == []
    assert result.dims == lazy.dims
    assert result.chunksizes["beta"] == lazy.chunksizes["beta"]
    assert result.chunksizes["eV"] == ((3,) if restore_size else (6,))
    assert result.chunksizes["alpha"] == ((4,) if restore_size else (8,))
    xr.testing.assert_identical(result.beta, eager.beta)

    computed = result.compute(scheduler="synchronous")
    assert sorted(reads) == [0, 1, 2]
    assert len(asymmetric_session.inputs) == 3
    assert asymmetric_session.loaded_threads == [1, 1, 1]
    expected = era.image.ml.denoise(
        eager, resize=(6, 8), restore_size=restore_size, tta=False, num_threads=1
    )
    xr.testing.assert_identical(computed, expected)
    assert lazy.data is original_data
    assert lazy.chunks == original_chunks
    xr.testing.assert_identical(lazy.compute(scheduler="synchronous"), eager)


def test_denoise_dask_computes_only_selected_planes(deferred_image, asymmetric_session):
    lazy, eager, reads = deferred_image
    result = era.image.ml.denoise(lazy, resize=(3, 4), tta=False)
    assert reads == asymmetric_session.loaded_threads == []

    selected = result.isel(beta=1).compute(scheduler="synchronous")

    assert reads == [1]
    assert len(asymmetric_session.inputs) == 1
    expected = era.image.ml.denoise(eager.isel(beta=1), resize=(3, 4), tta=False)
    xr.testing.assert_identical(selected, expected)


@pytest.mark.parametrize("dim", ["eV", "alpha"])
def test_denoise_dask_requires_single_core_chunks(image, identity_session, dim):
    lazy = image.chunk({dim: 1}).transpose("alpha", "eV")
    with pytest.raises(ValueError, match=dim):
        era.image.ml.denoise(lazy, resize=(3, 4), tta=False)
    assert identity_session.inputs == identity_session.loaded_threads == []


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("requested", [None, 0, 7])
def test_denoise_dask_limits_inference_threads(
    image, identity_session, lazy, requested
):
    data = image.chunk({"eV": -1, "alpha": -1}) if lazy else image
    options = {} if requested is None else {"num_threads": requested}
    result = era.image.ml.denoise(data, resize=(3, 4), tta=False, **options)
    if lazy:
        assert identity_session.loaded_threads == []
        result.compute(scheduler="synchronous")
    expected = 1 if lazy else (0 if requested is None else requested)
    assert identity_session.loaded_threads == [expected]


@pytest.mark.parametrize("invalid", ["zero", "nan"])
def test_denoise_dask_validates_slices_only_when_computed(
    image, identity_session, invalid
):
    bad = xr.zeros_like(image) if invalid == "zero" else xr.full_like(image, np.nan)
    lazy = xr.concat([image, bad], dim="beta").chunk({"beta": 1})
    result = era.image.ml.denoise(lazy, resize=(3, 4), tta=False)
    assert identity_session.inputs == identity_session.loaded_threads == []

    with pytest.raises(ValueError, match=r"finite|positive"):
        result.isel(beta=1).compute(scheduler="synchronous")
    assert identity_session.inputs == identity_session.loaded_threads == []


@pytest.mark.parametrize("invalid", ["zero", "nan"])
def test_denoise_rejects_one_invalid_slice(image, identity_session, invalid):
    bad = xr.zeros_like(image) if invalid == "zero" else xr.full_like(image, np.nan)
    data = xr.concat([image, bad], dim=xr.IndexVariable("scan", ["valid", "invalid"]))
    with pytest.raises(ValueError, match=r"finite|positive"):
        era.image.ml.denoise(data, resize=(3, 4), tta=False)


@pytest.mark.parametrize(
    ("rename", "along", "match"),
    [
        ({"eV": "energy"}, "alpha", "eV"),
        ({"alpha": "beta"}, "alpha", "alpha"),
        ({}, "eV", "along"),
    ],
)
def test_denoise_requires_distinct_core_dimensions(image, rename, along, match):
    with pytest.raises(ValueError, match=match):
        era.image.ml.denoise(image.rename(rename), along=along)


@pytest.mark.parametrize("restore_size", [False, True])
@pytest.mark.parametrize("restore_scale", [False, True])
def test_denoise_coarse_prediction_preserves_predicted_intensity(
    monkeypatch, restore_size, restore_scale
):
    class CoarseSession(_IdentitySession):
        def run(self, output_names, input_feed):
            image = super().run(output_names, input_feed)[0]
            return [
                np.full(
                    (1, 1, image.shape[2] // 2, image.shape[3] // 2),
                    image.mean() / 2,
                    dtype=np.float32,
                )
            ]

    session = CoarseSession()
    monkeypatch.setattr(
        _denoise,
        "_load_session",
        lambda model_path, num_threads: (pathlib.Path("coarse.onnx"), session),
    )
    image = xr.DataArray(np.full((3, 4), 2.0), dims=("eV", "alpha"))
    result = era.image.ml.denoise(
        image,
        resize=(6, 8),
        restore_scale=restore_scale,
        restore_size=restore_size,
        tta=True,
    )

    assert result.shape == ((3, 4) if restore_size else (6, 8))
    assert result.attrs["denoising_resize"] == (6, 8)
    expected_total = 12.0 if restore_scale else 5_000.0
    np.testing.assert_allclose(result.values, expected_total / result.size, rtol=1e-6)
    assert result.name == "denoised"


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"resize": True}, ValueError),
        ({"resize": (2, 0)}, ValueError),
        ({"resize": (2, 3, 4)}, ValueError),
        ({"resize": 2.5}, ValueError),
        ({"tta": 1}, TypeError),
        ({"restore_size": 1}, TypeError),
        ({"restore_scale": 1}, TypeError),
        ({"num_threads": True}, ValueError),
        ({"num_threads": -1}, ValueError),
        ({"along": 1}, TypeError),
    ],
)
def test_denoise_rejects_invalid_options(image, kwargs, exception):
    with pytest.raises(exception):
        era.image.ml.denoise(image, **kwargs)


@pytest.mark.parametrize(
    ("values", "exception"),
    [
        (np.ones((2, 3), dtype=int), TypeError),
        (np.ones(3), TypeError),
        (np.ones((0, 3)), ValueError),
        (np.full((2, 3), np.nan), ValueError),
        (np.zeros((2, 3)), ValueError),
    ],
)
def test_denoise_rejects_invalid_data(values, exception):
    dims = ("eV", "alpha")[: values.ndim]
    with pytest.raises(exception):
        era.image.ml.denoise(xr.DataArray(values, dims=dims), resize=(2, 3))


def test_denoise_rejects_unlabeled_array():
    with pytest.raises(TypeError, match=r"xarray\.DataArray"):
        era.image.ml.denoise(np.ones((2, 3)))


def test_denoise_rejects_nonuniform_coordinates_even_when_restoring(image):
    image = image.assign_coords(eV=[2.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="uniformly spaced"):
        era.image.ml.denoise(image, resize=(6, 8), restore_size=True)


@pytest.mark.parametrize("tta", [False, True])
def test_denoise_bundled_model(image, tta):
    pytest.importorskip("onnxruntime", exc_type=ModuleNotFoundError)
    original = image.copy(deep=True)

    result = era.image.ml.denoise(image, resize=(32, 48), tta=tta, num_threads=1)

    assert result.shape == image.shape
    assert result.dims == image.dims
    assert np.isfinite(result.values).all()
    assert result.attrs["denoising_model"] == "arpes-denoiser.onnx"
    xr.testing.assert_identical(result.coords.to_dataset(), image.coords.to_dataset())
    xr.testing.assert_identical(image, original)


def test_denoise_bundled_model_with_dask_processes():
    pytest.importorskip("onnxruntime", exc_type=ModuleNotFoundError)
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import xarray as xr
        import erlab.analysis as era

        if __name__ == "__main__":
            data = xr.DataArray(
                np.arange(1.0, 129.0).reshape(2, 8, 8),
                dims=("beta", "eV", "alpha"),
                coords={"beta": [0.0, 0.2], "eV": np.arange(8), "alpha": np.arange(8)},
            )
            lazy = era.image.ml.denoise(
                data.chunk({"beta": 1}), resize=(8, 8), tta=False
            )
            assert "onnxruntime" not in sys.modules
            result = lazy.compute(scheduler="processes", num_workers=2)
            expected = era.image.ml.denoise(
                data, resize=(8, 8), tta=False, num_threads=1
            )
            xr.testing.assert_identical(result, expected)
            assert np.isfinite(result.values).all()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
