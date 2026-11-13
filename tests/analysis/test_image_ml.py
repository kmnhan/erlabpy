import concurrent.futures
import contextlib
import importlib
import pathlib
import subprocess
import sys
import textwrap
import threading
import types

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

import erlab.analysis as era

_denoise = importlib.import_module("erlab.analysis.image.ml._denoise")


@pytest.fixture(scope="module", autouse=True)
def isolated_coreml_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("coreml-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_denoise, "_coreml_cache_root", lambda: root)
        yield
    _denoise._cached_session.cache_clear()


class _IdentitySession:
    def __init__(self):
        self.inputs = []
        self.loaded_threads = []
        self.loaded_providers = []
        self.providers = ["CPUExecutionProvider"]
        self.input_shape = [1, 1, "height", "width"]
        self.input_name = "image"
        self.output_name = "prediction"
        self.fallback_enabled = True

    def enable_fallback(self):
        self.fallback_enabled = True

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return [
            types.SimpleNamespace(
                name=self.input_name, type="tensor(float)", shape=self.input_shape
            )
        ]

    def get_outputs(self):
        return [
            types.SimpleNamespace(
                name=self.output_name, type="tensor(float)", shape=[1, 1, "h", "w"]
            )
        ]

    def run(self, output_names, input_feed):
        assert output_names == [self.output_name]
        assert set(input_feed) == {self.input_name}
        image = input_feed[self.input_name]
        assert image.dtype == np.float32
        assert image.flags.c_contiguous
        assert all(
            not isinstance(expected, int) or actual == expected
            for actual, expected in zip(image.shape, self.input_shape, strict=True)
        )
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

    def load_session(model_path, num_threads, provider, *, shape=None):
        session.loaded_threads.append(num_threads)
        session.loaded_providers.append(provider)
        return pathlib.Path("identity.onnx"), session

    monkeypatch.setattr(_denoise, "_load_session", load_session)
    return session


@pytest.fixture
def asymmetric_session(monkeypatch):
    session = _AsymmetricSession()

    def load_session(model_path, num_threads, provider, *, shape=None):
        session.loaded_threads.append(num_threads)
        session.loaded_providers.append(provider)
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


@pytest.fixture
def onnx_runtime(monkeypatch, tmp_path):
    sessions = []

    class SessionOptions:
        def __init__(self):
            self.overrides = {}

        def add_free_dimension_override_by_name(self, name, value):
            self.overrides[name] = value

    def create_session(path, *, sess_options, providers, enable_fallback=True):
        session = _IdentitySession()
        session.path = path
        session.options = sess_options
        session.providers = [p[0] if isinstance(p, tuple) else p for p in providers]
        session.provider_options = dict(p for p in providers if isinstance(p, tuple))
        session.constructor_fallback_enabled = enable_fallback
        session.fallback_enabled = enable_fallback
        session.input_shape = [
            sess_options.overrides.get(n, n) for n in session.input_shape
        ]
        sessions.append(session)
        return session

    runtime = types.SimpleNamespace(
        __version__="test-runtime",
        SessionOptions=SessionOptions,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=object()),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=object()),
        InferenceSession=create_session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        sessions=sessions,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setattr(_denoise, "_coreml_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(
        _denoise, "_lock_coreml_cache", lambda path: contextlib.nullcontext()
    )
    _denoise._cached_session.cache_clear()
    yield runtime
    _denoise._cached_session.cache_clear()


@pytest.mark.parametrize(
    ("available", "provider", "expected"),
    [
        (["CPUExecutionProvider"], "auto", ["CPUExecutionProvider"]),
        (
            ["AzureExecutionProvider", "CPUExecutionProvider"],
            "auto",
            ["CPUExecutionProvider"],
        ),
        (
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            "auto",
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            [
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
            ],
            "auto",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "cpu",
            ["CPUExecutionProvider"],
        ),
        (
            [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "coreml",
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    ],
)
def test_denoise_selects_execution_provider(
    onnx_runtime, available, provider, expected
):
    onnx_runtime.get_available_providers = lambda: available
    _, session = _denoise._load_session(None, 3, provider)

    assert session.providers == expected
    assert session.options.intra_op_num_threads == 3
    assert session.options.inter_op_num_threads == 1


@pytest.mark.parametrize("provider", ["cuda", "coreml"])
def test_denoise_rejects_unavailable_provider(onnx_runtime, provider):
    with pytest.raises(ValueError, match="not available"):
        _denoise._load_session(None, 0, provider)
    assert onnx_runtime.sessions == []


def test_denoise_session_cache_tracks_provider_and_model(onnx_runtime, tmp_path):
    onnx_runtime.get_available_providers = lambda: [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    path = tmp_path / "model.onnx"
    path.write_bytes(b"original")
    _, automatic = _denoise._load_session(path, 0, "auto")
    assert _denoise._load_session(path, 0, "auto")[1] is automatic
    path.write_bytes(b"replacement model")
    _, replacement = _denoise._load_session(path, 0, "auto")
    assert replacement is not automatic
    assert _denoise._load_session(path, 0, "cpu")[1] is not replacement
    assert _denoise._load_session(path, 1, "auto")[1] is not replacement
    assert len(onnx_runtime.sessions) == 4


@pytest.fixture
def coreml_runtime(monkeypatch, onnx_runtime):
    onnx_runtime.get_available_providers = lambda: [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setattr(_denoise.platform, "mac_ver", lambda: ("27.2", (), ""))
    return onnx_runtime


@pytest.mark.parametrize(
    "changed", ["model", "shape", "runtime", "options", "machine", "macos", "kernel"]
)
def test_denoise_coreml_disk_cache_separates_configurations(
    monkeypatch, onnx_runtime, changed
):
    options = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
    arguments = [b"weights", (9, 11), "1.31", options]
    original = _denoise._coreml_cache_directory(*arguments)
    assert original.parent == _denoise._coreml_cache_root()
    assert original == _denoise._coreml_cache_directory(*arguments)
    if changed == "model":
        arguments[0] = b"new weights"
    elif changed == "shape":
        arguments[1] = (11, 9)
    elif changed == "runtime":
        arguments[2] = "1.32"
    elif changed == "options":
        arguments[3] = options | {"MLComputeUnits": "CPUOnly"}
    elif changed == "machine":
        monkeypatch.setattr(_denoise.platform, "machine", lambda: "other-architecture")
    elif changed == "macos":
        monkeypatch.setattr(_denoise.platform, "mac_ver", lambda: ("28.0", (), ""))
    else:
        monkeypatch.setattr(_denoise.platform, "version", lambda: "other-kernel-build")
    assert original != _denoise._coreml_cache_directory(*arguments)


def test_denoise_coreml_disk_cache_is_independent_of_installation(
    monkeypatch, coreml_runtime, tmp_path
):
    paths = [tmp_path / "notebook.onnx", tmp_path / "frozen-resource.onnx"]
    sessions = []
    for path in paths:
        path.write_bytes(b"identical bundled graph and weights")
        stat = path.stat()
        if path == paths[1]:
            monkeypatch.setattr(sys, "frozen", True, raising=False)
            monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        sessions.append(
            _denoise._cached_session(
                str(path),
                stat.st_mtime_ns,
                stat.st_size,
                1,
                "CoreMLExecutionProvider",
                (9, 11),
            )
        )

    assert sessions[0].path == sessions[1].path == paths[0].read_bytes()
    first, second = [s.provider_options["CoreMLExecutionProvider"] for s in sessions]
    assert first["ModelCacheDirectory"] == second["ModelCacheDirectory"]
    assert all(not session.constructor_fallback_enabled for session in sessions)
    assert all(session.fallback_enabled for session in sessions)


@pytest.mark.parametrize("failure", ["directory", "lock"])
def test_denoise_coreml_disk_cache_access_failure_is_optional(
    monkeypatch, coreml_runtime, tmp_path, failure
):
    if failure == "directory":
        root = tmp_path / "not-a-directory"
        root.write_text("blocked")
        monkeypatch.setattr(_denoise, "_coreml_cache_root", lambda: root)
    else:

        def denied(path):
            raise PermissionError("lock unavailable")

        monkeypatch.setattr(_denoise, "_lock_coreml_cache", denied)
    with pytest.warns(RuntimeWarning, match="Could not access the CoreML disk cache"):
        _, session = _denoise._load_session(None, 1, "coreml", shape=(9, 11))

    assert (
        "ModelCacheDirectory" not in session.provider_options["CoreMLExecutionProvider"]
    )
    assert session.constructor_fallback_enabled
    assert session.fallback_enabled


def test_denoise_coreml_corrupt_cache_recovers_after_unlock(
    monkeypatch, coreml_runtime
):
    class NativeFailure(Exception):
        pass

    _, healthy = _denoise._load_session(None, 1, "coreml", shape=(9, 11))
    cache = pathlib.Path(
        healthy.provider_options["CoreMLExecutionProvider"]["ModelCacheDirectory"]
    )
    (cache / "incomplete-model").write_bytes(b"invalid")
    unrelated = cache.parent / "other-configuration"
    unrelated.mkdir()
    (unrelated / "keep").touch()
    _denoise._cached_session.cache_clear()
    create_session = coreml_runtime.InferenceSession
    locked = False
    calls = []

    @contextlib.contextmanager
    def lock(path):
        nonlocal locked
        assert path == cache.with_suffix(".lock")
        locked = True
        try:
            yield
        finally:
            locked = False

    def corrupted(*args, **kwargs):
        calls.append(kwargs.get("enable_fallback", True))
        if not calls[-1]:
            assert locked
            assert (cache / "incomplete-model").exists()
            raise NativeFailure("invalid compiled model")
        assert not locked
        assert not cache.exists()
        return create_session(*args, **kwargs)

    monkeypatch.setattr(_denoise, "_lock_coreml_cache", lock)
    coreml_runtime.InferenceSession = corrupted
    with pytest.warns(RuntimeWarning, match="Could not load the CoreML disk cache"):
        _, recovered = _denoise._load_session(None, 1, "coreml", shape=(9, 11))

    assert calls == [False, True]
    assert recovered.constructor_fallback_enabled
    assert recovered.fallback_enabled
    assert (unrelated / "keep").exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="CoreML uses the macOS file-lock API"
)
def test_denoise_coreml_lock_excludes_other_openers_and_releases(tmp_path):
    path = tmp_path / "cache.lock"
    started, acquired = threading.Event(), threading.Event()

    def other_opener():
        started.set()
        with _denoise._lock_coreml_cache(path):
            acquired.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with _denoise._lock_coreml_cache(path):
            future = executor.submit(other_opener)
            assert started.wait(5)
            assert not acquired.wait(0.05)
        future.result(timeout=5)
    assert acquired.is_set()
    assert path.is_file()
    with (
        pytest.raises(ValueError, match="release on failure"),
        _denoise._lock_coreml_cache(path),
    ):
        raise ValueError("release on failure")
    with _denoise._lock_coreml_cache(path):
        assert path.is_file()


@pytest.mark.parametrize(
    ("provider", "mac_version", "custom"),
    [
        ("cpu", "27.2", False),
        ("cuda", "27.2", False),
        ("coreml", "11.7", False),
        ("coreml", "", False),
        ("coreml", "27.2", True),
    ],
)
def test_denoise_keeps_dynamic_sessions(
    monkeypatch, onnx_runtime, tmp_path, provider, mac_version, custom
):
    onnx_runtime.get_available_providers = lambda: [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setattr(_denoise.platform, "mac_ver", lambda: (mac_version, (), ""))
    path = None
    if custom:
        path = tmp_path / "custom.onnx"
        path.write_bytes(b"custom")
    _, session = _denoise._load_session(path, 1, provider, shape=(9, 11))
    _, resized = _denoise._load_session(path, 1, provider, shape=(11, 9))

    assert resized is session
    assert session.options.overrides == {}
    assert session.provider_options == {}


@pytest.mark.parametrize("fallback", [False, True])
def test_denoise_coreml_specializes_both_orientations(
    monkeypatch, onnx_runtime, image, fallback
):
    onnx_runtime.get_available_providers = lambda: [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setattr(_denoise.platform, "mac_ver", lambda: ("27.2", (), ""))
    if fallback:
        create_session = onnx_runtime.InferenceSession

        def cpu_fallback(*args, **kwargs):
            session = create_session(*args, **kwargs)
            session.providers = ["CPUExecutionProvider"]
            return session

        onnx_runtime.InferenceSession = cpu_fallback
    options = {"resize": (9, 12), "num_threads": 1, "provider": "auto"}
    result = era.image.ml.denoise(image, **options)
    sessions = onnx_runtime.sessions

    assert len(sessions) == 2
    for session, shape in zip(sessions, [(9, 12), (12, 9)], strict=True):
        assert session.input_shape == [1, 1, *shape]
        assert len(session.inputs) == 4
        assert all(tensor.shape == (1, 1, *shape) for tensor in session.inputs)
        provider_options = session.provider_options["CoreMLExecutionProvider"].copy()
        cache_directory = pathlib.Path(provider_options.pop("ModelCacheDirectory"))
        assert cache_directory.parent == _denoise._coreml_cache_root()
        assert cache_directory.is_dir()
        assert isinstance(session.path, bytes)
        assert not session.constructor_fallback_enabled
        assert session.fallback_enabled
        assert provider_options == {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "0",
        }
    xr.testing.assert_allclose(result, image.rename("spectrum_denoised"))
    repeated = era.image.ml.denoise(image, **options)
    lazy = era.image.ml.denoise(image.chunk(), **options)
    assert len(sessions) == 2
    xr.testing.assert_identical(lazy.compute(scheduler="threads"), repeated)
    assert len(sessions) == 2
    era.image.ml.denoise(image, **(options | {"resize": (9, 9)}))
    assert len(sessions) == 3
    assert len(sessions[-1].inputs) == 8


@pytest.mark.parametrize("parallel", [False, True])
def test_denoise_static_tta_uses_each_session_signature(tta_executor, parallel):
    image = np.arange(99, dtype=np.float32).reshape(9, 11)
    primary, transposed = _AsymmetricSession(), _AsymmetricSession()
    primary.input_shape = [1, 1, 9, 11]
    transposed.input_shape = [1, 1, 11, 9]
    transposed.input_name = "transposed_image"
    transposed.output_name = "transposed_prediction"

    result = _denoise._predict(
        primary, image, True, parallel=parallel, transposed_session=transposed
    )
    expected = _denoise._predict(_AsymmetricSession(), image, True)

    np.testing.assert_array_equal(result, expected)
    assert len(primary.inputs) == len(transposed.inputs) == 4


def test_denoise_custom_static_model_rejects_rectangular_tta(monkeypatch, image):
    session = _IdentitySession()
    session.input_shape = [1, 1, 9, 11]
    monkeypatch.setattr(
        _denoise,
        "_load_session",
        lambda *args, **kwargs: (pathlib.Path("custom.onnx"), session),
    )
    with pytest.raises(ValueError, match="does not accept"):
        era.image.ml.denoise(image, resize=(9, 11), model_path="custom.onnx")
    assert session.inputs == []


@pytest.fixture
def tta_executor(monkeypatch):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        monkeypatch.setattr(_denoise, "_cached_tta_executor", lambda pid: executor)
        yield executor


@pytest.mark.parametrize("shape", [(7, 9), (8, 10)])
def test_parallel_tta_preserves_order_and_output_bits(tta_executor, shape):
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    transforms = _denoise._tta_transforms(True)
    inputs = {
        (
            (image.T if transpose else image).shape,
            np.ascontiguousarray(
                np.flip(image.T if transpose else image, axis=flip_axes)
            ).tobytes(),
        ): (index, transpose, flip_axes)
        for index, (transpose, flip_axes) in enumerate(transforms)
    }
    scales = [1e16, 1.0, -1e16, 1.0, 2.0, 3.0, 4.0, 5.0]

    class Session(_IdentitySession):
        def __init__(self, delay_first):
            super().__init__()
            self.delay_first = delay_first
            self.later_finished = threading.Event()
            self.completed = []

        def run(self, output_names, input_feed):
            tensor = super().run(output_names, input_feed)[0][0, 0]
            index, transpose, flip_axes = inputs[tensor.shape, tensor.tobytes()]
            if self.delay_first and index == 0:
                assert self.later_finished.wait(timeout=5)
            prediction = np.full(
                ((shape[0] + 1) // 2, (shape[1] + 1) // 2),
                scales[index],
                dtype=np.float32,
            )
            prediction[0, 0] = -0.0
            prediction = np.flip(
                prediction.T if transpose else prediction, axis=flip_axes
            )
            self.completed.append(index)
            if index != 0:
                self.later_finished.set()
            return [prediction[None, None]]

    serial = _denoise._predict(Session(False), image, True)
    parallel_session = Session(True)
    parallel = _denoise._predict(parallel_session, image, True, parallel=True)

    assert parallel_session.completed[0] != 0
    assert sorted(parallel_session.completed) == list(range(8))
    assert parallel.tobytes() == serial.tobytes()
    assert parallel[0, 0].view(np.uint64) == 0


@pytest.mark.parametrize("error", ["shape", "empty", "nonfinite", "inconsistent"])
def test_parallel_tta_preserves_model_errors(tta_executor, error):
    class Session(_IdentitySession):
        def run(self, output_names, input_feed):
            tensor = super().run(output_names, input_feed)[0]
            if error == "shape":
                return [tensor[0]]
            if error == "empty":
                return [tensor[:, :, :0]]
            if error == "nonfinite":
                return [tensor * np.nan]
            return [np.ones((1, 1, 2, 3), dtype=np.float32)]

    image = np.ones((7, 9), dtype=np.float32)
    exception = FloatingPointError if error == "nonfinite" else ValueError
    with pytest.raises(exception) as serial:
        _denoise._predict(Session(), image, True)
    with pytest.raises(exception) as parallel:
        _denoise._predict(Session(), image, True, parallel=True)
    assert str(parallel.value) == str(serial.value)


@pytest.mark.parametrize(
    ("provider", "num_threads", "tta", "shape", "custom", "lazy", "parallel"),
    [
        ("CoreMLExecutionProvider", 0, True, (128, 128), False, False, True),
        ("CoreMLExecutionProvider", 3, True, (128, 128), False, False, False),
        ("CPUExecutionProvider", 0, True, (128, 128), False, False, True),
        ("CoreMLExecutionProvider", 0, True, (128, 128), True, False, False),
        ("CoreMLExecutionProvider", 0, False, (128, 128), False, False, False),
        ("CoreMLExecutionProvider", 0, True, (7, 9), False, False, True),
        ("CPUExecutionProvider", 1, True, (128, 128), False, False, False),
        ("CPUExecutionProvider", 3, True, (128, 128), False, False, False),
        ("CPUExecutionProvider", 0, True, (128, 128), False, True, False),
        ("CoreMLExecutionProvider", 0, True, (128, 128), False, True, False),
        ("CUDAExecutionProvider", 0, True, (128, 128), False, False, False),
    ],
)
def test_denoise_parallel_tta_eligibility(
    monkeypatch, tmp_path, provider, num_threads, tta, shape, custom, lazy, parallel
):
    class StatefulSession(_IdentitySession):
        def run(self, output_names, input_feed):
            tensor = super().run(output_names, input_feed)[0]
            return [np.full_like(tensor, len(self.inputs))]

    session = StatefulSession()
    session.providers = (
        [provider]
        if provider == "CPUExecutionProvider"
        else [provider, "CPUExecutionProvider"]
    )
    monkeypatch.setattr(
        _denoise,
        "_load_session",
        lambda *args, **kwargs: (pathlib.Path("model.onnx"), session),
    )
    requests = []

    class Executor:
        def map(self, function, values):
            return map(function, values)

    def executor(pid):
        assert parallel, "This model and configuration must run serially"
        requests.append(pid)
        return Executor()

    monkeypatch.setattr(_denoise, "_cached_tta_executor", executor)
    data = xr.DataArray(np.ones(shape), dims=("eV", "alpha"))
    if lazy:
        data = data.chunk({"eV": -1, "alpha": -1})
    result = era.image.ml.denoise(
        data,
        resize=shape,
        model_path=tmp_path / "custom.onnx" if custom else None,
        num_threads=num_threads,
        tta=tta,
        restore_scale=False,
    ).compute(scheduler="synchronous")

    assert bool(requests) == parallel
    assert len(session.inputs) == (8 if tta else 1)
    np.testing.assert_array_equal(result.values, 4.5 if tta else 1.0)


@pytest.mark.parametrize(("cpu_count", "workers"), [(None, 1), (1, 1), (2, 2), (8, 4)])
def test_parallel_tta_shares_a_bounded_executor(monkeypatch, cpu_count, workers):
    real_executor = concurrent.futures.ThreadPoolExecutor
    executors = []
    active = 0
    max_active = 0
    worker_threads = set()
    lock = threading.Lock()
    workers_running = threading.Event()
    release = threading.Event()

    def create_executor(**kwargs):
        executor = real_executor(**kwargs)
        executors.append(executor)
        return executor

    class Session(_IdentitySession):
        def run(self, output_names, input_feed):
            nonlocal active, max_active
            with lock:
                worker_threads.add(threading.get_ident())
                active += 1
                max_active = max(max_active, active)
                if active == workers:
                    workers_running.set()
            try:
                assert release.wait(timeout=5)
                return super().run(output_names, input_feed)
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr(_denoise.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", create_executor)
    _denoise._cached_tta_executor.cache_clear()
    image = np.arange(63, dtype=np.float32).reshape(7, 9)
    try:
        with real_executor(max_workers=2) as callers:
            results = [
                callers.submit(_denoise._predict, Session(), image, True, parallel=True)
                for _ in range(2)
            ]
            try:
                assert workers_running.wait(timeout=5)
            finally:
                release.set()
            for result in results:
                np.testing.assert_array_equal(result.result(timeout=5), image)
        assert len(executors) == 1
        assert max_active == len(worker_threads) == workers
    finally:
        release.set()
        for executor in executors:
            executor.shutdown(wait=True)
        _denoise._cached_tta_executor.cache_clear()


@pytest.mark.parametrize("provider", ["auto", "cpu", "cuda", "coreml"])
def test_denoise_passes_provider_to_dask_workers(image, identity_session, provider):
    result = era.image.ml.denoise(
        image.chunk({"eV": -1, "alpha": -1}),
        resize=image.shape,
        provider=provider,
        tta=False,
    )
    assert identity_session.loaded_providers == []
    result = result.compute(scheduler="synchronous")
    assert identity_session.loaded_providers == [provider]
    assert result.attrs["denoising_provider"] == provider
    np.testing.assert_allclose(result.values, image.values, rtol=1e-6)


def test_denoise_dask_discovers_providers_when_computed(image, onnx_runtime):
    result = era.image.ml.denoise(
        image.chunk({"eV": -1, "alpha": -1}), resize=image.shape, tta=False
    )
    assert onnx_runtime.sessions == []
    onnx_runtime.get_available_providers = lambda: [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    computed = result.compute(scheduler="synchronous")
    assert len(onnx_runtime.sessions) == 1
    assert onnx_runtime.sessions[0].providers == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    np.testing.assert_allclose(computed.values, image.values, rtol=1e-6)


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
        lambda model_path, num_threads, provider, *, shape=None: (
            pathlib.Path("coarse.onnx"),
            session,
        ),
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
        ({"provider": "invalid"}, ValueError),
        ({"provider": None}, ValueError),
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
@pytest.mark.parametrize("provider", ["auto", "cpu"])
def test_denoise_bundled_model(image, tta, provider):
    pytest.importorskip("onnxruntime", exc_type=ModuleNotFoundError)
    original = image.copy(deep=True)

    result = era.image.ml.denoise(
        image, resize=(32, 48), tta=tta, num_threads=1, provider=provider
    )

    assert result.shape == image.shape
    assert result.dims == image.dims
    assert np.isfinite(result.values).all()
    assert result.attrs["denoising_model"] == "arpes-denoiser.onnx"
    assert result.attrs["denoising_provider"] == provider
    xr.testing.assert_identical(result.coords.to_dataset(), image.coords.to_dataset())
    xr.testing.assert_identical(image, original)


@pytest.mark.parametrize("provider", ["cpu", "coreml", "cuda"])
@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (
            (5, 7),
            [
                [315.4223328, 318.5719299, 323.4724121, 325.0528259],
                [318.4394836, 321.5350952, 326.4729309, 328.0229797],
                [320.9434204, 324.0326843, 329.0254211, 330.6137390],
            ],
        ),
        ((1, 1), [[10027.3759766]]),
        ((1, 7), [[1212.8768311, 1512.5834961, 1944.8887939, 2146.7272949]]),
        ((3, 3), [[1201.9035645, 1201.9089355], [1201.9049072, 1201.9106445]]),
        ((4, 4), [[504.1811218, 504.1822205], [504.1805725, 504.1817322]]),
    ],
)
def test_denoise_optimized_model_preserves_original_predictions(
    provider, shape, expected
):
    ort = pytest.importorskip("onnxruntime", exc_type=ModuleNotFoundError)
    if provider != "cpu":
        accelerator = {
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
        }[provider]
        if accelerator not in ort.get_available_providers():
            pytest.skip(f"{accelerator} is unavailable")
    # CPU reference predictions from the original model before graph rewriting.
    image = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    image = (image / image.sum()) * 10000
    _, session = _denoise._load_session(None, 1, provider, shape=shape)
    if provider == "cuda":
        # Compare full-precision outputs; the CUDA provider defaults to TF32.
        path = (
            pathlib.Path(_denoise.__file__).with_name("_data")
            / _denoise.DEFAULT_MODEL_NAME
        )
        session = ort.InferenceSession(
            str(path),
            providers=[
                ("CUDAExecutionProvider", {"use_tf32": "0"}),
                "CPUExecutionProvider",
            ],
        )

    result = _denoise._predict(session, image, False)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("tta", [False, True])
def test_denoise_bundled_model_with_dask_threads(image, tta):
    pytest.importorskip("onnxruntime", exc_type=ModuleNotFoundError)
    data = image * xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="beta")
    expected = era.image.ml.denoise(data, resize=(32, 48), tta=tta, num_threads=1)
    lazy = era.image.ml.denoise(data.chunk({"beta": 1}), resize=(32, 48), tta=tta)

    result = lazy.compute(scheduler="threads", num_workers=4)

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("provider", ["auto", "cpu"])
@pytest.mark.parametrize("restore_scale", [False, True])
@pytest.mark.parametrize("restore_size", [False, True])
def test_denoise_parallel_bundled_model_matches_serial_bits(
    monkeypatch, provider, restore_scale, restore_size
):
    pytest.importorskip("onnxruntime", exc_type=ModuleNotFoundError)
    data = xr.DataArray(np.arange(1.0, 100.0).reshape(9, 11), dims=("eV", "alpha"))
    options = {
        "resize": (129, 131),
        "provider": provider,
        "restore_scale": restore_scale,
        "restore_size": restore_size,
    }
    result = era.image.ml.denoise(data, **options)
    original_predict = _denoise._predict
    monkeypatch.setattr(
        _denoise,
        "_predict",
        lambda session, image, tta, **kwargs: original_predict(
            session, image, tta, **(kwargs | {"parallel": False})
        ),
    )
    expected = era.image.ml.denoise(data, **options)

    assert result.values.tobytes() == expected.values.tobytes()
    xr.testing.assert_identical(result, expected)


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
