import pytest
import numpy as np
from src.NULL.matrices import rforwardsolve, rbackwardsolve
from .fixtures import get_banded_lower_systems, get_banded_upper_systems


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
