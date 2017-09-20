import pytest
import numpy as np
from src.NULL.matrices import rforwardsolve, rbackwardsolve
from .fixtures import get_banded_lower_systems, get_banded_upper_systems


@pytest.mark.forward
@pytest.mark.solvers
def test_rforwardsolve(get_banded_lower_systems):
    for A, b, x, d in get_banded_lower_systems:
        computed_x = rforwardsolve(A, b, d)
        np.testing.assert_almost_equal(computed_x, x)

@pytest.mark.backward
@pytest.mark.solvers
def test_rbackwardsolve(get_banded_upper_systems):
    for A, b, x, d in get_banded_upper_systems:
        computed_x = rbackwardsolve(A, b, d)
        np.testing.assert_almost_equal(computed_x, x)
