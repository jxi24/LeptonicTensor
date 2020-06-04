import numpy as np
from . import lorentz_tensor

METRIC_TENSOR = np.diag([1, -1, -1, -1])
IDENTITY = np.diag([1, 1, 1, 1])


def eps(i, j, k, l):
    return -(i-j)*(i-k)*(i-l)*(j-k)*(j-l)*(k-l)/12


EPS = np.fromfunction(eps, (4, 4, 4, 4), dtype=int)

DELTA = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

GAMMA_0 = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]])

GAMMA_1 = np.array([[0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, -1, 0, 0],
                    [-1, 0, 0, 0]])

GAMMA_2 = np.array([[0, 0, 0, -1j],
                    [0, 0, 1j, 0],
                    [0, 1j, 0, 0],
                    [-1j, 0, 0, 0]])

GAMMA_3 = np.array([[0, 0, 1, 0],
                    [0, 0, 0, -1],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0]])

CHARGE_CONJ = 1j*np.einsum('ij,jk->ik', GAMMA_2, GAMMA_0)

GAMMA_5 = 1j*np.einsum('ij,jk,kl,lm->im', GAMMA_0, GAMMA_1, GAMMA_2, GAMMA_3)
GAMMA_MU = np.array([GAMMA_0, GAMMA_1, GAMMA_2, GAMMA_3])


class ChargeConj(lorentz_tensor.Tensor):
    def __init__(self, i, j):
        super().__init__(CHARGE_CONJ, (i, j))


class Gamma(lorentz_tensor.Tensor):
    def __init__(self, mu, i, j):
        super().__init__(GAMMA_MU, (mu, i, j))


class Gamma5(lorentz_tensor.Tensor):
    def __init__(self, i, j):
        super().__init__(GAMMA_5, (i, j))


class Metric(lorentz_tensor.Tensor):
    def __init__(self, mu, nu):
        super().__init__(METRIC_TENSOR, (mu, nu))


class Momentum(lorentz_tensor.Tensor):
    def __init__(self, p, mu, n):
        super().__init__(p[n], [mu])


class Identity(lorentz_tensor.Tensor):
    def __init__(self, i, j):
        super().__init__(IDENTITY, (i, j))


class ProjP(lorentz_tensor.Tensor):
    def __init__(self, i, j):
        array = (IDENTITY + GAMMA_5)/2
        super().__init__(array, (i, j))


class ProjM(lorentz_tensor.Tensor):
    def __init__(self, i, j):
        array = (IDENTITY - GAMMA_5)/2
        super().__init__(array, (i, j))


class Epsilon(lorentz_tensor.Tensor):
    def __init__(self, i, j, k, l):
        super().__init__(EPS, (i, j, k, l))
