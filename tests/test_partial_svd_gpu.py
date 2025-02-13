import pytest
import numpy as np
import cupy
import time
import warnings
from scipy.sparse.linalg import svds as scipy_svds
from scipy.sparse.linalg import LinearOperator
import cupyx.scipy.sparse.linalg as cupy_spla
from .conftest import TYPES
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



cupy_types= dict(zip(TYPES,(cupy.float32, cupy.float64, cupy.complex64, cupy.complex128)))

# size = x*(D^2)*(D^2), with x*(D^2) being the number of singular values to compute
# hence the ratio is 1/D^2

@pytest.mark.parametrize("driver", ['arpack', 'lobpcg', 'propack',])
def test_scipy_svds(benchmark, size, ratio, dtype, driver):
    #print(f"cupy.cuda.runtime.getDevice {cupy.cuda.runtime.getDevice()}")

    def setup():
        a = cupy.random.rand(size, size).astype(cupy_types[dtype])
        if cupy.iscomplexobj(a):
            a = a + 1j*cupy.random.rand(size, size).astype(cupy_types[dtype])
        
        op = LinearOperator(
            (size, size),
            matvec=lambda x: cupy.asnumpy(a @ cupy.asarray(x)),
            rmatvec=lambda x: cupy.asnumpy(a.conj().T @ cupy.asarray(x)),
            dtype=a.dtype,
        )
        
        return ((op,), {})
    
    svds= lambda x: scipy_svds(x, k=int(size*ratio), ncv=None, tol=0, which='LM', v0=None, maxiter=None, \
                                    return_singular_vectors=True, solver=driver, options=None) # rng=None,

    def _wrap_cuda_events(a):
        start_gpu = cupy.cuda.Event()
        end_gpu = cupy.cuda.Event()

        start_gpu.record()
        start_cpu = time.perf_counter()
        out = scipy_svds(a)
        end_gpu.record()
        end_gpu.synchronize()
        end_cpu = time.perf_counter()
        t_gpu = cupy.cuda.get_elapsed_time(start_gpu, end_gpu)
        t_cpu = end_cpu - start_cpu
        # print(f"GPU time: {t_gpu} ms CPU time: {t_cpu} s")
        return *out, t_gpu, t_cpu

    benchmark.pedantic(_wrap_cuda_events, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


def test_svd_cupy(benchmark, size, ratio, dtype):
    logger.info(f"cupy.cuda.runtime.getDevice {cupy.cuda.runtime.getDevice()}")

    def setup():
        a = cupy.random.rand(size, size).astype(cupy_types[dtype])
        if cupy.iscomplexobj(a):
            a = a + 1j*cupy.random.rand(size, size).astype(cupy_types[dtype])
        return ((a, ), {})
    
    def _wrap_cuda_events(a):
        start_gpu = cupy.cuda.Event()
        end_gpu = cupy.cuda.Event()

        start_gpu.record()
        start_cpu = time.perf_counter()
        out = cupy_spla.svds(a, k=int(size*ratio), ncv=None, tol=0, which='LM', maxiter=None, return_singular_vectors=True)
        end_gpu.record()
        end_gpu.synchronize()
        end_cpu = time.perf_counter()
        t_gpu = cupy.cuda.get_elapsed_time(start_gpu, end_gpu)
        t_cpu = end_cpu - start_cpu
        # print(f"GPU time: {t_gpu} ms CPU time: {t_cpu} s")
        return *out, t_gpu, t_cpu

    benchmark.pedantic(_wrap_cuda_events, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


try:
    import slepc4py
    from slepc4py import SLEPc
    from petsc4py import PETSc
except Exception as e:
    slepc4py = None
    warnings.warn(f"SLEPc/PETSc import failed with {e}")

#
# see https://gitlab.com/slepc/slepc/-/blob/main/src/binding/slepc4py/demo/ex4.py?ref_type=heads
#
# TODO: Add support for complex numbers, CUDA
# see https://abhigupta.io/2021/12/08/installing-petsc-complex.html for complex number support
# see https://petsc.org/release/install/install/#installing-petsc-to-use-gpus-and-accelerators
@pytest.mark.skipif(slepc4py is None, reason="SLEPc not available")
def test_svds_slepc(benchmark, size, ratio, dtype):
    if 'complex' in dtype:
        pytest.skip("SLEPc/PETSc installed via pip does not support complex numbers")

    def setup():
        arr = cupy.random.rand(size, size).astype(cupy_types[dtype])
        if cupy.iscomplexobj(arr):
            arr += 1j * cupy.random.rand(size, size).astype(cupy_types[dtype])
        return (arr,), {}

    def run_slepc_svds(mat):
        A = PETSc.Mat().createDenseCUDA(size=mat.shape, array=mat,) #comm=SLEPc.COMM_SELF)
        A.assemble()

        solver = SLEPc.SVD().create()#comm=SLEPc.COMM_SELF)
        solver.setOperator(A)
        solver.setDimensions(int(size * ratio))
        solver.setType(SLEPc.SVD.Type.TRLANCZOS)
        solver.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
        solver.solve()

        nconv= solver.getConverged()
        assert nconv >= int(size*ratio), f"Expected {int(size*ratio)} converged singular values, got {nconv}"	
        
        # Preallocate numpy arrays for S, U, V
        svals = cupy.empty(nconv, dtype=mat.real.dtype)
        left_vecs = cupy.empty((mat.shape[0], nconv), dtype=mat.dtype)
        right_vecs = cupy.empty((mat.shape[1], nconv), dtype=mat.dtype)

        v, u = A.createVecs()
        for i in range(nconv):
            svals[i] = solver.getSingularTriplet(i, u, v)
            left_vecs[:, i] = u.getArray()
            right_vecs[:, i] = v.getArray()
            
        return left_vecs, svals, right_vecs

    benchmark.pedantic(run_slepc_svds, setup=setup, iterations=1, warmup_rounds=1, rounds=5)