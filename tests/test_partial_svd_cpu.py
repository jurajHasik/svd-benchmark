import pytest
import numpy as np
import warnings
from scipy.sparse.linalg import svds as scipy_svds


# size = x*(D^2)*(D^2), with x*(D^2) being the number of singular values to compute
# hence the ratio is 1/D^2

@pytest.mark.parametrize("driver", ['arpack', 'lobpcg', 'propack',])
def test_scipy_svds(benchmark, size, ratio, dtype, driver):

    def setup():
        a = np.random.rand(size, size).astype(dtype)
        if np.iscomplexobj(a):
            a = a + 1j*np.random.rand(size, size).astype(dtype)
        return ((a,), {})

    svds = lambda x: scipy_svds(x, k=int(size*ratio), ncv=None, tol=0, which='LM', v0=None, maxiter=None, \
                                    return_singular_vectors=True, solver=driver, options=None) # rng=None,

    benchmark.pedantic(svds, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


try:
    import primme
except Exception as e:
    # See
    # https://pypi.org/project/primme/
    primme = None
    warnings.warn(f"primme import failed with {e}")

@pytest.mark.skipif(primme is None, reason="primme not available")
def test_svds_primme(benchmark, size, ratio, dtype):
    
    def setup():
        a = np.random.rand(size, size).astype(dtype)
        if np.iscomplexobj(a):
            a = a + 1j*np.random.rand(size, size).astype(dtype)
        return ((a,), {})
    
    svds_primme = lambda x: primme.svds(x, k=int(size*ratio), ncv=None, tol=0, which='LM', v0=None, maxiter=None, \
                            return_singular_vectors=True, precAHA=None, precAAH=None, precAug=None, u0=None, \
                            orthou0=None, orthov0=None, return_stats=False, maxBlockSize=0, method=None, methodStage1=None, \
                            methodStage2=None, return_history=False, convtest=None,)
    
    benchmark.pedantic(svds_primme, setup=setup, iterations=1, warmup_rounds=1, rounds=5)


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
# TODO: Add support for complex numbers
# see https://abhigupta.io/2021/12/08/installing-petsc-complex.html for complex number support
@pytest.mark.skipif(slepc4py is None, reason="SLEPc not available")
def test_svds_slepc(benchmark, size, ratio, dtype):
    if 'complex' in dtype:
        pytest.skip("SLEPc/PETSc installed via pip does not support complex numbers")

    def setup():
        arr = np.random.rand(size, size).astype(dtype)
        if np.iscomplexobj(arr):
            arr += 1j * np.random.rand(size, size).astype(dtype)
        return (arr,), {}

    def run_slepc_svds(mat):
        A = PETSc.Mat().createDense(size=mat.shape, array=mat,) #comm=SLEPc.COMM_SELF)
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
        svals = np.empty(nconv, dtype=mat.real.dtype)
        left_vecs = np.empty((mat.shape[0], nconv), dtype=mat.dtype)
        right_vecs = np.empty((mat.shape[1], nconv), dtype=mat.dtype)

        v, u = A.createVecs()
        for i in range(nconv):
            svals[i] = solver.getSingularTriplet(i, u, v)
            left_vecs[:, i] = u.getArray()
            right_vecs[:, i] = v.getArray()
            
        return left_vecs, svals, right_vecs

    benchmark.pedantic(run_slepc_svds, setup=setup, iterations=1, warmup_rounds=1, rounds=5)