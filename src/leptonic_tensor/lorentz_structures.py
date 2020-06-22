import numpy as np
import lorentz_tensor as lt

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


class ChargeConj(lt.Tensor):
    def __init__(self, i, j):
        if not isinstance(i, lt.Index):
            i = lt.SpinIndex(i)
        if not isinstance(j, lt.Index):
            j = lt.SpinIndex(j)
        super().__init__(CHARGE_CONJ, (i, j))


class Gamma(lt.Tensor):
    def __init__(self, mu, i, j):
        if not isinstance(mu, lt.Index):
            mu = lt.LorentzIndex(mu)
        if not isinstance(i, lt.Index):
            i = lt.SpinIndex(i)
        if not isinstance(j, lt.Index):
            j = lt.SpinIndex(j)
        super().__init__(GAMMA_MU, (mu, i, j))


class Gamma5(lt.Tensor):
    def __init__(self, i, j):
        if not isinstance(i, lt.Index):
            i = lt.SpinIndex(i)
        if not isinstance(j, lt.Index):
            j = lt.SpinIndex(j)
        super().__init__(GAMMA_5, (i, j))


class Metric(lt.Tensor):
    def __init__(self, mu, nu):
        if not isinstance(mu, lt.Index):
            mu = lt.LorentzIndex(mu)
        if not isinstance(nu, lt.Index):
            nu = lt.LorentzIndex(nu)
        super().__init__(METRIC_TENSOR, (mu, nu))


# TODO: Figure out how to handle momentum correctly
class Momentum(lt.Tensor):
    def __init__(self, p, mu, n):
        super().__init__(p[n], [mu])


class Identity(lt.Tensor):
    def __init__(self, i, j):
        if not isinstance(i, lt.Index):
            i = lt.SpinIndex(i)
        if not isinstance(j, lt.Index):
            j = lt.SpinIndex(j)
        super().__init__(IDENTITY, (i, j))


class LorentzIdentity(lt.Tensor):
    def __init__(self, mu, nu):
        if not isinstance(mu, lt.Index):
            mu = lt.LorentzIndex(mu)
        if not isinstance(nu, lt.Index):
            nu = lt.LorentzIndex(nu)
        super().__init__(IDENTITY, (mu, nu))


class ProjP(lt.Tensor):
    def __init__(self, i, j):
        if not isinstance(i, lt.Index):
            i = lt.SpinIndex(i)
        if not isinstance(j, lt.Index):
            j = lt.SpinIndex(j)
        array = (IDENTITY + GAMMA_5)/2
        super().__init__(array, (i, j))


class ProjM(lt.Tensor):
    def __init__(self, i, j):
        if not isinstance(i, lt.Index):
            i = lt.SpinIndex(i)
        if not isinstance(j, lt.Index):
            j = lt.SpinIndex(j)
        array = (IDENTITY - GAMMA_5)/2
        super().__init__(array, (i, j))


class Epsilon(lt.Tensor):
    def __init__(self, mu1, mu2, mu3, mu4):
        if not isinstance(mu1, lt.Index):
            mu1 = lt.LorentzIndex(mu1)
        if not isinstance(mu2, lt.Index):
            mu2 = lt.LorentzIndex(mu2)
        if not isinstance(mu3, lt.Index):
            mu3 = lt.LorentzIndex(mu3)
        if not isinstance(mu4, lt.Index):
            mu4 = lt.LorentzIndex(mu4)
        super().__init__(EPS, (mu1, mu2, mu3, mu4))
