import time
import numpy as np
import scipy
import pytest
import multiprocessing
import warnings
import logging
from .conftest import TYPES
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_np_svd(benchmark, size, dtype):
    # numpy.show_config()

    def setup():
        a = np.random.rand(size, size).astype(dtype)
        if np.iscomplexobj(a):
            a = a + 1j*np.random.rand(size, size).astype(dtype)
        return ((a, ), {})

    np_svd= lambda x: np.linalg.svd(x, full_matrices=True, compute_uv=True)

    benchmark.pedantic(np_svd, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


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
    af.set_backend('cpu')
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


@pytest.mark.parametrize("driver", ['gesdd', 'gesvd',])
def test_scipy_svd(benchmark, size, dtype, driver):

    def setup():
        a = np.random.rand(size, size).astype(dtype)
        if np.iscomplexobj(a):
            a = a + 1j*np.random.rand(size, size).astype(dtype)
        return ((a,), {})

    scipy_svd= lambda x: scipy.linalg.svd(x, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver=driver)

    benchmark.pedantic(scipy_svd, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


try:
    import torch
    torch_types= dict(zip(TYPES,(torch.float32, torch.float64, torch.complex64, torch.complex128)))
except Exception as e:
    torch = None
    warnings.warn(f"torch import failed with {e}")

@pytest.mark.skipif(torch is None, reason="ArrayFire not available")
def test_torch_svd(benchmark, size, dtype):
    logger.info(f"torch.set_num_threads {multiprocessing.cpu_count()}")
    if dtype==torch.float16:
        pytest.skip("PyTorch does not support float16 for SVD")
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_default_device('cpu')

    def setup():
        a = torch.rand(size, size, dtype=torch_types[dtype])
        return ((a, ), {})

    torch_svd= lambda x: torch.linalg.svd(x, full_matrices=True)

    benchmark.pedantic(torch_svd, setup=setup, iterations=1, warmup_rounds=1, rounds=5)
