import cupy as cp
import cupyx.scipy.sparse.linalg as spla


def svd_cupy(A):
    U, S, VT = cp.linalg.svd(A, full_matrices=True)
    return U, S, VT

def svds_cupy(A,k):
    U_s, S_s, VT_s = spla.svds(A, k)
    return U_s, S_s, VT_s