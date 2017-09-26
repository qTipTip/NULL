from matrices import gaussian_elimination, gaussian_elimination_pivots, PLU

import numpy as np

A = np.array([
    [1, 1, 2],
    [2, 2, 1], 
    [1, 2, 3]
], dtype=np.float64)
b = np.array([9, 9, 14], dtype=np.float64)

P, L, U = PLU(A)

print(P, L, U)
print(P.dot(L).dot(U))

print(gaussian_elimination(A, b))
print(gaussian_elimination_pivots(A, b))
