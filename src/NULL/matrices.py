import numpy as np

def rforwardsolve(A, b, d):
    """
    Given a nonsingular lower triangular d-banded matrix and a right hand side b,
    x is computed so that Ax = b.
    :param A: nxn matrix d-banded
    :param b: n-veector, right hand side
    :param d: band-width of A
    :return: n-vector x
    """

    n = len(b)
    if np.iscomplexobj(A) or np.iscomplexobj(b):
        A = A.astype('complex128')
        b = b.astype('complex128')
    x = b
    x[0] = b[0] / A[0, 0]
    for k in range(1, n):
        lk = max(0, k-d)
        x[k] = (b[k] - np.dot(A[k, lk : k], x[lk : k])) / A[k, k]

    return x

def rbackwardsolve(A, b, d):
    """
    Given a nonsingular upper triangular d-banded matrix and a right hand side b,
    x is computed so that Ax = b.
    :param A: nxn matrix d-banded
    :param b: n-veector, right hand side
    :param d: band-width of A
    :return: n-vector x
    """

    n = len(b)
    if np.iscomplexobj(A) or np.iscomplexobj(b):
        A = A.astype('complex128')
        b = b.astype('complex128')
    x = b
    x[n-1] = b[n-1] / A[n-1, n-1]

    for k in range(n-2, -1, -1):
        uk = min(n-1, k+d)
        x[k] = (b[k] - np.dot(A[k, k+1:uk+1], x[k+1:uk+1])) / A[k, k]

    return x

def L1U(A, d):
    """
    Given a matrix A with non-singular leading submatrices with bandwidth d, computes
    the matrices L, U such that A = LU
    :param A: nxn matrix d-banded
    :param d: bandwidth
    :return: L, U
    """

    n, _ = A.shape
    L = np.eye(n, n, dtype=A.dtype)
    U = np.zeros((n, n), dtype=A.dtype)

    U[0, 0] = A[0, 0]
    for k in range(1, n):
        km = max(0, k-d)
        L[k, km : k] = rforwardsolve(U[km:k, km:k].T, A[k, km:k].T, d).T
        U[km:k+1, k] = rforwardsolve(L[km:k+1, km:k+1], A[km:k+1, k], d)

    return L, U

