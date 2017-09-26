import pytest
import numpy as np
from src.NULL.matrices import rforwardsolve, rbackwardsolve, L1U, PLU, housegen
from .fixtures import get_banded_lower_systems, get_banded_upper_systems, get_banded_matrices, get_PLU_matrices, get_householder_vectors


@pytest.mark.forward
@pytest.mark.solvers
@pytest.mark.parametrize("A, b, x, d", get_banded_lower_systems())
def test_rforwardsolve(A, b, x, d):
    computed_x = rforwardsolve(A, b, d)
    np.testing.assert_almost_equal(computed_x, x)

@pytest.mark.backward
@pytest.mark.solvers
@pytest.mark.parametrize("A, b, x, d", get_banded_upper_systems())
def test_rbackwardsolve(A, b, x, d):
    computed_x = rbackwardsolve(A, b, d)
    np.testing.assert_almost_equal(computed_x, x)


@pytest.mark.decomposition
@pytest.mark.parametrize("A, d, L, U", get_banded_matrices())
def test_L1U(A, d, L, U):
    computed_L, computed_U = L1U(A, d)
    np.testing.assert_almost_equal(computed_L, L)
    np.testing.assert_almost_equal(computed_U, U)


@pytest.mark.decomposition
@pytest.mark.parametrize("A, P, L, U", get_PLU_matrices())
def test_PLU(A, P, L, U):
    computed_P, computed_L, computed_U = PLU(A)
    
    np.testing.assert_almost_equal(computed_P, P)
    np.testing.assert_almost_equal(computed_L, L, decimal=6)
    np.testing.assert_almost_equal(computed_U, U, decimal=6)

    computed_A = P.T.dot(L).dot(U)
    np.testing.assert_almost_equal(computed_A, A, decimal=4)

@pytest.mark.transformation
@pytest.mark.parametrize("x, u, a", get_householder_vectors())
def test_housegen(x, u, a):
    computed_u, computed_a = housegen(x)

    np.testing.assert_almost_equal(computed_u, u)
    np.testing.assert_almost_equal(computed_a, a) 
