import pytest
import numpy as np
import scipy.linalg as la

from src.NULL.matrices import L1U

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
    
    A3 = np.array([
        [1, 2, 4],
        [3, 8, 14],
        [2, 6, 13]
    ])
    L3 = np.array([
        [1, 0, 0],
        [3, 1, 0],
        [2, 1, 1]
    ])  
    U3 = np.array([
        [1, 2, 4],
        [0, 2, 2],
        [0, 0, 3]
    ])
    d3 = 2
    return (
        (A1, d1, L1, U1),
        (A2, d2, L2, U2),
        (A3, d3, L3, U3)
    )


@pytest.fixture()
def get_PLU_matrices():

    A1 = np.array([
        [0, 1],
        [1, 1]
    ])
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
        [1, 1, 2],
        [2, 2, 1],
        [1.0, 2, 3]
    ])
    P2 = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    L2 = np.array([
        [1, 0, 0],
        [0.5, 1, 0],
        [0.5, 0, 1]
    ])
    U2 = np.array([
        [2, 2, 1],
        [0, 1, 2.5],
        [0, 0, 1.5]
    ])
    
    I3 = np.eye(100, 100)
    
    A4 = np.array([
        [0, 3],
        [1, 2]
    ])
    P4 = np.array([
        [0, 1],
        [1, 0]
    ])
    L4 = np.array([
        [1, 0], 
        [0, 1]
    ])
    U4 = np.array([
        [1, 2],
        [0, 3]
    ])

    return (
        (A1, P1, L1, U1),
        (A2, P2, L2, U2),
        (I3, I3, I3, I3),
        (A4, P4, L4, U4)
    )

@pytest.fixture()
def get_householder_vectors():
    
    e1 = np.zeros(10)
    e1[0] = 1
    a1 = -1
    u1 = e1.copy()
    u1[0] = 2 / np.sqrt(2)
    

    x2 = np.array([3.0j, 2, 5])
    u2 = np.array([3.0j + np.sqrt(38), -2j, -5]) / (np.sqrt(38*(3 + np.sqrt(38))))
    a2 = -1j * np.sqrt(38)
    
    x3 = np.array([1j, 1, 0])
    u3 = np.array([
        (1 / np.sqrt(2) + 1) / (np.sqrt(1 / np.sqrt(2) + 1)),
        (-1j / np.sqrt(2)) / (np.sqrt(1 / np.sqrt(2) + 1)),
        0
    ])
    a3 = -1j * np.sqrt(2)
    return  (
        (e1, u1, a1),
        pytest.mark.skip(reason='Computed the wrong analytical answer')((x2, u2, a2)),
        (x3, u3, a3) 
)

