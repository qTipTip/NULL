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
    x = b.copy()
    x[0] = x[0] / A[0, 0]
    for k in range(1, n):
        lk = max(0, k-d)
        x[k] = b[k] - np.dot(A[k, lk : k], x[lk : k])
        x[k] = x[k] / A[k, k] 
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
    x = b.copy()
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
        P[:, [k, i]] = P[:, [i, k]]

        for j in range(k+1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k : n] = U[j, k : n] - L[j, k]*U[k, k:n]

    return P, L, U

def LU_solve(A, d, b):
    """
    Given a matrix A with bandwidth d and nonsingular principal submatrices and
    a right hand side b, computes x such that Ax = b using the L1U
    factorization.
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

def housetriang(A, B):
    """
    Given a (mxn) matrix A, and a set of right hand sides B (mxr),
    computes the householder transformations H1, ..., Hs such that
    R = Hs...H1A is upper trapezoidal and C = Hs...H1B.
    """

    m, n = A.shape
    try:
        _, r = B.shape
    except:
        r = 1

    A = np.hstack((A.copy(), B.copy()))

    for k in range(min(n, m-1)):
        v, A[k, k] = housegen(A[k:m, k])
        v = np.reshape(v, (m-k, 1))
        C = A[k:m, k+1:n+r]
        A[k:m, k+1:n+r] = C -  v * np.dot(np.conjugate(v).T, C)

    R = np.triu(A[:, 0:n])
    C = A[:, n:n+r]

    return R, C

def gaussian_elimination(A, b):
    """
    Given a (nxn) matrix A and a right hand side b, computes x such that Ax =
    b. """
    
    m, n = A.shape
    U = A.copy() 
    b = b.copy()

    # forward sweep, reduce A to a upper triangular matrix
    for k in range(min(m, n)):
        swap = np.argmax(np.abs(U[k:, k])) + k
        if U[swap, k] == 0:
            raise ValueError('Singular matrix')
        U[[k, swap], :] = U[[swap, k], :]
        b[[k, swap]] = b[[swap, k]]
        
        for i in range(k + 1, m):
            factor = U[i, k] / U[k, k]
            b[i] = b[i] - factor*b[k]
            U[i, k+1:] = U[i, k+1:] - U[k, k+1:] * factor
            U[i, k] = 0
    
    # solve by back subistitution
    x = rbackwardsolve(U, b, m)

    return x

def gaussian_elimination_pivots(A, b):
    """
    Given an nxn matrix A and a right hand side b, computes
    the matrices P, L, U such that A = PLU,
    then computes x such that LUx = (P.T)b.
    """

    P, L, U = PLU(A)
    n,_ = A.shape
    y = rforwardsolve(L, (P.T).dot(b), n)
    x = rbackwardsolve(U, y, n)

    return x

def housetriang_solve(A, b):
    """
    Given an nxn matrix A and a right hand side b, computes the matrix
    R and the vector c such that
    Rx = c, where R is upper triangular. Hence can be solved by back-substitution.
    :param A: nxn matrix A
    :param b: right hand side
    :return: x such that Ax = b
    """

    n, _ = A.shape
    b = np.reshape(b.copy(), (n, 1))
    R, c = housetriang(A, b)
    x = np.reshape(rbackwardsolve(R, c, n), (n,))


    return x
