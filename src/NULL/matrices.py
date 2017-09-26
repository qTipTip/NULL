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
        print(A[k, k])
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
        L[k, km : k] = np.transpose(rforwardsolve(np.transpose(U[km:k, km:k]),\
                                                  np.transpose(A[k, km:k]), d))
        U[km:k+1, k] = rforwardsolve(L[km:k+1, km:k+1], A[km:k+1, k], d)
    return L, U

def PLU(A):
    """
    Given a matrix A computes the matrices P, L, U such that A = PLU where P is
    a permutation matrix.
    :param A: nxn matrix
    :return: PLU matrix
    """
    n, _ = A.shape
    P = np.eye(n, n, dtype=A.dtype)
    L = np.eye(n, n, dtype=A.dtype) 
    U = np.copy(A)
    
    for k in range(n-1):
        i = np.argmax(np.abs(U[k:, k])) + k # index of row with highest absolute value in column k

        U[[k, i], k:] = U[[i, k], k:]
        L[[k, i], :k] = L[[i, k], :k]
        P[[k, i], :] = P[[i, k], :]

        for j in range(k+1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k : n] = U[j, k : n] - L[j, k]*U[k, k:n]

    return P, L, U

def LU_solve(A, d, b):
    """
    Given a matrix A with bandwidth d and a right hand side b, computes x such
    that Ax = b using the L1U factorization.
    :param A: nxn matrix
    :return: PLU matrix
    """

    L, U = L1U(A, d)

    y = rforwardsolve(L, b, d)
    x = rbackwardsolve(U, y, d)

    return x

def housegen(x):
    """
    Given an n-vector x, returns the vector u and constant a such that
    (I - uu*)x = a*e_1
    """

    a = np.linalg.norm(x)
    if a == 0:
        u = x
        u[0] = np.sqrt(2)
        return u, a
    
    if x[0] == 0:
        r = 1
    else:
        r = x[0] / abs(x[0])

    u = np.conj(r) * x / a
    u[0] = u[0] + 1
    u = u / np.sqrt(u[0])
    
    a = -r*a

    return u, a
