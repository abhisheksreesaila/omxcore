"""F-209: the Python path's GPU check asks for a usable CUDA driver, not for a library file.

On an Apple Mac the Metal build of libmxpipeline_gpu.so exists but the Python path drives CUDA only, so
gpu_available() must be False there (it was True, and every -gpu case crashed with "libcuda not found")."""
import omxcore.pipeline_kernels as pk


def test_no_cuda_driver_means_no_gpu_even_with_the_library_present(monkeypatch):
    monkeypatch.setattr(pk, "_cuda_usable", lambda: False)
    monkeypatch.setattr(pk, "_find", lambda name, env: "/some/libmxpipeline_gpu.so")
    assert pk.gpu_available() is False


def test_a_usable_driver_and_the_library_mean_gpu(monkeypatch):
    monkeypatch.setattr(pk, "_cuda_usable", lambda: True)
    monkeypatch.setattr(pk, "_find", lambda name, env: "/some/libmxpipeline_gpu.so")
    assert pk.gpu_available() is True


def test_a_usable_driver_without_the_library_means_no_gpu(monkeypatch):
    monkeypatch.setattr(pk, "_cuda_usable", lambda: True)
    monkeypatch.setattr(pk, "_find", lambda name, env: None)
    assert pk.gpu_available() is False


def test_this_machines_answer_matches_whether_cuda_opens():
    import ctypes
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        n = ctypes.c_int(0)
        has = lib.cuInit(0) == 0 and lib.cuDeviceGetCount(ctypes.byref(n)) == 0 and n.value > 0
    except OSError:
        has = False
    assert pk._cuda_usable() is has
