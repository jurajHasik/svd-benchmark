import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds

def svd_sparse_arpack(A, k=6):
    """Sparse SVD using ARPACK."""
    U, s, Vt = svds(A, k=k, solver='arpack')
    return U, s, Vt

def svd_sparse_lobpcg(A, k=6):
    """Sparse SVD using LOBPCG (for certain matrix types)."""
    U, s, Vt = svds(A, k=k, solver='lobpcg')
    return U, s, Vt

def svd_sparse_propack(A, k=6):
    """Sparse SVD using PROPACK."""
    U, s, Vt = svds(A, k=k, solver='propack')
    return U, s, Vt