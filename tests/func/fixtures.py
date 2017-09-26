import pytest
import numpy as np

@pytest.fixture
def linear_system_3x3():

    A = np.array([
        [1.0, 1, 2],
        [2, 2, 1],
        [1, 2, 3]
    ])

    b = np.array([9.0, 9, 14])
    x = np.array([1, 2, 3])
    d = 2
    return (
        (A, d, b, x),
    )
