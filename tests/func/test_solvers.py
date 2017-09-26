import numpy as np
import pytest

from NULL import PLU, rforwardsolve, rbackwardsolve, LU_solve, gaussian_elimination, gaussian_elimination_pivots, \
    housetriang
from NULL.matrices import housetriang_solve
from .fixtures import linear_system_3x3


@pytest.mark.parametrize("A, d, b, x", linear_system_3x3())
def test_PLU_solve(A, d, b, x):

    P, L, U = PLU(A)
    
    y = rforwardsolve(L, P.T.dot(b), 3)
    computed_x = rbackwardsolve(U, y, 3)

    np.testing.assert_almost_equal(computed_x, x)

@pytest.mark.lu
@pytest.mark.parametrize("A, d, b, x", linear_system_3x3())
@pytest.mark.skip(reason='Not LU-factorizable')
def test_LU_solve(A, d, b, x):
    computed_x = LU_solve(A, d, b)
    np.testing.assert_almost_equal(computed_x, x)

@pytest.mark.solvers
@pytest.mark.parametrize("A, d, b, x", linear_system_3x3())
def test_gaussian_elimination(A, d, b, x):
    computed_x = gaussian_elimination(A, b)
    np.testing.assert_almost_equal(computed_x, x)

@pytest.mark.solvers
@pytest.mark.parametrize("A, d, b, x", linear_system_3x3())
def test_gaussian_elimination_pivots(A, d, b, x):
    computed_x = gaussian_elimination_pivots(A, b)
    np.testing.assert_almost_equal(computed_x, x)

@pytest.mark.solvers
@pytest.mark.parametrize("A, d, b, x", linear_system_3x3())
def test_householder_solver(A, d, b, x):
    computed_x = housetriang_solve(A, b)
    np.testing.assert_almost_equal(computed_x, x)