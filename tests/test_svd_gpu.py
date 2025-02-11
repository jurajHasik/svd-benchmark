import multiprocessing
import warnings
import pytest
import time
import numpy as np
import cupy as cp
import logging
from .conftest import TYPES
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




cupy_types= dict(zip(TYPES,(cp.float32, cp.float64, cp.complex64, cp.complex128)))

def test_svd_cupy(benchmark, size, dtype):
    logger.info(f"cupy.cuda.runtime.getDevice {cp.cuda.runtime.getDevice()}")

    def setup():
        a = cp.random.rand(size, size).astype(cupy_types[dtype])
        if cp.iscomplexobj(a):
            a = a + 1j*cp.random.rand(size, size).astype(cupy_types[dtype])
        return ((a, ), {})
    
    def _wrap_cuda_events(a):
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()

        start_gpu.record()
        start_cpu = time.perf_counter()
        out = cp.linalg.svd(a, full_matrices=True)
        end_gpu.record()
        end_gpu.synchronize()
        end_cpu = time.perf_counter()
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        t_cpu = end_cpu - start_cpu
        # print(f"GPU time: {t_gpu} ms CPU time: {t_cpu} s")
        return *out, t_gpu, t_cpu

    benchmark.pedantic(_wrap_cuda_events, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


try:
    import arrayfire as af
    af_types= dict(zip(TYPES,(af.Dtype.f32, af.Dtype.f64, af.Dtype.c32, af.Dtype.c64)))
except Exception as e:
    # See
    # https://arrayfire.org/docs/#gsc.tab=0
    #
    # and
    # https://github.com/arrayfire/arrayfire-python
    af = None
    warnings.warn(f"ArrayFire import failed with {e}")

@pytest.mark.skipif(af is None, reason="ArrayFire not available")
def test_svd_arrayfire(benchmark, size, dtype):
    af.set_backend('cuda')
    logger.info(f"ArrayFire {af.device.info_str()}")

    def setup():
        a = af.randu(size, size, dtype=af_types[dtype])
        return ((a, ), {})
    
    def _wrap_cuda_events(a):
        af.sync()
        start_cpu = time.perf_counter()

        out = af.lapack.svd(a)

        af.sync()
        end_cpu = time.perf_counter()
        t_cpu = end_cpu - start_cpu
        # print(f"CPU time: {t_cpu} s")
        return *out, t_cpu

    benchmark.pedantic(_wrap_cuda_events, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


try:
    import torch
    torch_types= dict(zip(TYPES,(torch.float32, torch.float64, torch.complex64, torch.complex128)))
except Exception as e:
    torch = None
    warnings.warn(f"torch import failed with {e}")

@pytest.mark.skipif(torch is None, reason="ArrayFire not available")
@pytest.mark.parametrize("driver", ['gesvd', 'gesvdj', 'gesvda',])
def test_svd_torch(benchmark, size, dtype, driver):
    logger.info(f"torch.backends.cuda.preferred_linalg_library {torch.backends.cuda.preferred_linalg_library()}")
    logger.info(f"torch.set_num_threads {multiprocessing.cpu_count()}")
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_default_device('cuda')

    def setup():
        a = torch.rand(size, size, dtype=torch_types[dtype])
        return ((a, ), {})
    
    def _wrap_cuda_events(a):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start_cpu = time.perf_counter()

        out = torch.linalg.svd(a, full_matrices=True, driver=driver)

        end_event.record()
        torch.cuda.synchronize()
        end_cpu = time.perf_counter()
        t_gpu = start_event.elapsed_time(end_event)
        t_cpu = end_cpu - start_cpu
        # print(f"GPU time: {t_gpu} ms CPU time: {t_cpu} s")
        return *out, t_gpu, t_cpu

    benchmark.pedantic(_wrap_cuda_events, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


try:
    import cuquantum.cutensornet as cutn
except Exception as e:
    cutn = None
    warnings.warn(f"cuTensornet import failed with {e}")

@pytest.mark.skipif(cutn is None, reason="cuTensornet not available")
@pytest.mark.parametrize("driver", ['gesvd', 'gesvdj', 'gesvdp',])
def test_svd_cutensornet(benchmark, size, dtype, driver):
    logger.info(f"cupy.cuda.runtime.getDevice {cp.cuda.runtime.getDevice()}")

    def setup():
        a = cp.random.rand(size, size).astype(cupy_types[dtype])
        if cp.iscomplexobj(a):
            a = a + 1j*cp.random.rand(size, size).astype(cupy_types[dtype])
        return ((a, ), {})

    if torch:
        def setup_torch():
            a = torch.rand(size, size, dtype=torch_types[dtype])
            return ((a, ), {})
    
    def _wrap_cuda_events(a):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start_cpu = time.perf_counter()

        method = cutn.tensor.SVDMethod(algorithm=driver)
        out = cutn.tensor.decompose("ij->ix,xj", a, method=method, return_info=True)

        end_event.record()
        torch.cuda.synchronize()
        end_cpu = time.perf_counter()
        t_gpu = start_event.elapsed_time(end_event)
        t_cpu = end_cpu - start_cpu
        # print(f"GPU time: {t_gpu} ms CPU time: {t_cpu} s")
        return *out, t_gpu, t_cpu

    benchmark.pedantic(_wrap_cuda_events, setup=setup, iterations=1, warmup_rounds=1, rounds=5)