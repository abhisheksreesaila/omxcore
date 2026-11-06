"""F-209: the Python path's GPU check asks for a usable CUDA device, not for a library file.

On an Apple Mac the Metal build of libmxpipeline_gpu.so exists and MAX sees a Metal accelerator, but the Python path
drives CUDA only, so it must say "no GPU" there (it said yes, and every -gpu case crashed with "libcuda not found")."""
import omxcore.custom_ops as co
import omxcore.pipeline_kernels as pk


class FakeCuda:
    """A libcuda whose cuInit returns `init` and which reports `devices` devices."""

    def __init__(self, init=0, devices=1):
        self.init, self.devices = init, devices

    def cuInit(self, flags):
        return self.init

    def cuDeviceGetCount(self, ref):
        ref._obj.value = self.devices
        return 0


def _driver(monkeypatch, lib):
    """pk.ctypes.CDLL returns `lib`, or raises OSError (no libcuda) when lib is None; the asked-once answer is reset."""
    def cdll(name):
        if lib is None:
            raise OSError(f"{name}: cannot open shared object file")
        return lib
    monkeypatch.setattr(pk.ctypes, "CDLL", cdll)
    monkeypatch.setattr(pk, "_CUDA", None)
    monkeypatch.setattr(pk, "_find", lambda name, env: "/some/libmxpipeline_gpu.so")     # the library file is there


def test_no_libcuda_means_no_gpu_though_the_library_file_exists(monkeypatch):
    _driver(monkeypatch, None)
    assert pk.cuda_usable() is False
    assert pk.gpu_available() is False


def test_a_driver_whose_cuinit_fails_means_no_gpu(monkeypatch):
    _driver(monkeypatch, FakeCuda(init=100))                 # CUDA_ERROR_NO_DEVICE
    assert pk.gpu_available() is False


def test_a_driver_with_no_devices_means_no_gpu(monkeypatch):
    _driver(monkeypatch, FakeCuda(devices=0))
    assert pk.gpu_available() is False


def test_a_working_driver_and_the_library_mean_gpu(monkeypatch):
    _driver(monkeypatch, FakeCuda(devices=2))
    assert pk.gpu_available() is True


def test_a_working_driver_without_the_library_means_no_gpu(monkeypatch):
    _driver(monkeypatch, FakeCuda())
    monkeypatch.setattr(pk, "_find", lambda name, env: None)
    assert pk.gpu_available() is False


class FakeAccelerator:
    architecture_name = "fake"

    def __init__(self, api):
        self.api = api


class FakeMaxDriver:
    def __init__(self, api):
        self.api = api

    def accelerator_count(self):
        return 1

    def Accelerator(self):
        return FakeAccelerator(self.api)


def test_max_accelerator_accepts_a_cuda_device_and_refuses_a_metal_one(monkeypatch):
    monkeypatch.setattr(co, "_ACCELERATOR", {})
    monkeypatch.setattr(co, "driver", FakeMaxDriver("metal"))
    assert co._max_accelerator() is None
    monkeypatch.setattr(co, "_ACCELERATOR", {})
    monkeypatch.setattr(co, "driver", FakeMaxDriver("cuda"))
    assert co._max_accelerator().api == "cuda"
