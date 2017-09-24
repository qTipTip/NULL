import pytest
import numpy as np
import scipy.linalg as la

@pytest.fixture()
def get_banded_lower_systems():
    """ returns linear systems of lower triangular matrices
    with right hand side and expected solution """

    A1 = np.array([
        [1, 0],
        [3, 2],
    ])
    b1 = np.array([3, 4.2j])
    x1 = np.array([3, 2.1j - 4.5])
    d1 = 1

    A2 = np.array([
        [3.2j, 0, 0],
        [1, 5, 0],
        [0, 2, 1]
    ])
    b2 = np.array([1, 2, 3])
    x2 = np.array(
        [-1j / 3.2, (2 + 1j / 3.2) / 5, 11/5 - 1j/8]
    )
    d2 = 1

    A3 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    b3 = np.array([
        1, 2, 3j
    ])
    x3 = np.array([1, 2, 3j])
    d3 = 0

    return (
        (A1, b1, x1, d1),
        (A2, b2, x2, d2),
        (A3, b3, x3, d3)
    )

@pytest.fixture()
def get_banded_upper_systems():

    A1 = np.array([
        [1, 3],
        [0, 2]
    ])
    b1 = np.array([3, 4.2j])
    x1 = np.array([3 - 6.3j, 2.1j])
    d1 = 1


    A2 = np.array([
        [3.2j, 1, 0],
        [0, 5, 2],
        [0, 0, 1]
    ])
    b2 = np.array([1, 2, 3])
    x2 = np.array([-9j/16, -4/5, 3])
    d2 = 1

    A3 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    b3 = np.array([
        1, 2, 3j
    ])
    x3 = np.array([1, 2, 3j])
    d3 = 0

    return (
        (A1, b1, x1, d1),
        (A2, b2, x2, d2),
        (A3, b3, x3, d3)
    )

@pytest.fixture()
def get_banded_matrices():

    A1 = np.array([[1.2]])
    L1 = np.array([[1]])
    U1 = np.array([[1.2]])
    d1 = 0

    A2 = np.array([
        [2.3j, 1],
        [2, 4]
    ])
    L2 = np.array([
        [1, 0],
        [-2j / 2.3, 1]
    ])
    U2 = np.array([
        [2.3j, 1],
        [0, 4 + 2j / 2.3]
    ])
    d2 = 1

    return (
        (A1, d1, L1, U1),
        (A2, d2, L2, U2)
    )

@pytest.fixture()
def get_PLU_matrices():

    A1 = np.array([
        [0, 1],
        [1, 1]
    ])
    d1 = 1
    P1 = np.array([
        [0, 1],
        [1, 0]
    ])
    L1 = np.array([
        [1, 0],
        [0, 1]
    ])
    U1 = np.array([
        [1, 1],
        [0, 1]
    ])


    A2 = np.array([
        [0, 2, 1],
        [5.2j, 2, 2],
        [1.3, 5, 2]
    ])
    d2 = 2
    P2, L2, U2 = la.lu(A2)

    return (
        (A1, d1, P1, L1, U1),
        pytest.mark.xfail(reason='not sure if this is computed properly\
                          analytically')((A2, d2, P2, L2, U2))
    )
                          
