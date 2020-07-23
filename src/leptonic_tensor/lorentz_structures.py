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

CHARGE_CONJ = -1j*GAMMA_2

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
    def __init__(self, p, mu):
        if not isinstance(mu, lt.Index):
            mu = lt.LorentzIndex(mu)
        super().__init__(p, [mu])


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


# class SpinorV(lt.Tensor):
#     def __init__(self, p, i, spin):
#         if not isinstance(i, lt.Index):
#             i = lt.SpinIndex(i)
#         m = np.sqrt(p[0]**2-(np.sum(p[1:]**2))+0j)
#         if spin == 0:
#             vector = np.array([
#                 m+p[0]-p[3], -p[1]-1j*p[2], -m-p[0]-p[3], -p[1]-1j*p[2]
#             ])
#         else:
#             vector = np.array([
#                 -p[1]+1j*p[2], m+p[0]+p[3],  -p[1]+1j*p[2], -m-p[0]+p[3]
#             ])
#         vector /= np.sqrt(2*(p[0]+m))
#         super().__init__(vector, [i])
# 
# 
# class SpinorVBar(lt.Tensor):
#     def __init__(self, p, i, spin):
#         if not isinstance(i, lt.Index):
#             i = lt.SpinIndex(i)
#         m = np.sqrt(p[0]**2-(np.sum(p[1:]**2))+0j)
#         if spin == 0:
#             vector = np.array([
#                 -m-p[0]-p[3], -p[1]-1j*p[2], m+p[0]-p[3], -p[1]-1j*p[2]
#             ])
#         else:
#             vector = np.array([
#                 -p[1]+1j*p[2], -m-p[0]+p[3],  -p[1]+1j*p[2], m+p[0]+p[3]
#             ])
#         vector /= np.sqrt(2*(p[0]+m))
#         super().__init__(vector.conjugate(), [i])
# 
# 
# class SpinorU(lt.Tensor):
#     def __init__(self, p, i, spin):
#         if not isinstance(i, lt.Index):
#             i = lt.SpinIndex(i)
#         m = np.sqrt(p[0]**2-(np.sum(p[1:]**2))+0j)
#         if spin == 0:
#             vector = np.array([
#                 m+p[0]-p[3], -p[1]-1j*p[2], m+p[0]+p[3], p[1]+1j*p[2]
#             ])
#         else:
#             vector = np.array([
#                 -p[1]+1j*p[2], m+p[0]+p[3], p[1]-1j*p[2], m+p[0]-p[3]
#             ])
#         vector /= np.sqrt(2*(p[0]+m))
#         super().__init__(vector, [i])
# 
# 
# class SpinorUBar(lt.Tensor):
#     def __init__(self, p, i, spin):
#         if not isinstance(i, lt.Index):
#             i = lt.SpinIndex(i)
#         m = np.sqrt(p[0]**2-(np.sum(p[1:]**2))+0j)
#         if spin == 0:
#             vector = np.array([
#                 m+p[0]+p[3], p[1]+1j*p[2], m+p[0]-p[3], -p[1]-1j*p[2]
#             ])
#         else:
#             vector = np.array([
#                 p[1]-1j*p[2], m+p[0]-p[3], -p[1]+1j*p[2], m+p[0]+p[3]
#             ])
#         vector /= np.sqrt(2*(p[0]+m))
#         super().__init__(vector.conjugate(), [i])


class WeylSpinor:
    def __init__(self, mr, p):
        if abs(mr) != 1:
            raise ValueError("mr should be +/- 1")
        rpp = np.sqrt(p[0]+p[3]+0j)
        rpm = np.sqrt(p[0]-p[3]+0j)
        pt = p[1] + 1j*p[2]
        self.mu = np.array([rpp, rpm])
        if abs(pt) > 0:
            self.mu[1] = (pt.real + 1j*mr*pt.imag)/rpp

    def __getitem__(self, i):
        if i > 2:
            raise IndexError()
        return self.mu[i]

    def __neg__(self):
        self.mu = -self.mu
        return self

    def __str__(self):
        return f'<{self.mu[0], self.mu[1]}>'


class Spinor(lt.Tensor):
    def __init__(self, p, i, mr, hel=0, spin=1, bar=1):
        if not isinstance(i, lt.Index):
            i = lt.SpinIndex(i)
        self.bar = bar
        self.u = np.zeros(4, dtype=np.complex)
        if np.all(p[1:] == 0):
            rte = np.sqrt(p[0]+0j)
            if (mr > 0) ^ (hel < 0):  # u+(p, m) / v-(p, m)
                self.u[2] = rte
            else:  # u-(p, m) / v+(p, m)
                self.u[1] = -rte
            sgn = 1 if mr > 0 else -1
            r = 0 if (mr > 0) ^ (hel < 0) else 2
            self.u[0 + r] = sgn*self.u[2 - r]
            self.u[1 + r] = sgn*self.u[3 - r]
            self.on = 3
        else:
            ph = p.copy()
            ps = np.sqrt(np.sum(p[1:]**2))
            ph[0] = -ps if p[0] < 0 else ps
            if (mr > 0) ^ (hel < 0):  # u+(p, m) / v-(p, m)
                sh = WeylSpinor(1, ph)
                self.u[2] = sh[0]
                self.u[3] = sh[1]
                self.on = 2
            else:  # u-(p, m) / v+(p, m)
                sh = WeylSpinor(-1, ph)
                if p[0] < 0:
                    sh = -sh
                self.u[0] = sh[1]
                self.u[1] = -sh[0]
                self.on = 1
            m2 = p[0]**2 - np.sum(p[1:]**2)
            if m2 > 1e-8:
                sgn = 1 if (mr > 0) ^ (spin < 0) else -1
                omp = np.sqrt(p[0]+ph[0]+0j)/(2*ph[0])
                omm = np.sqrt(p[0]-ph[0]+0j)/(2*ph[0])
                r = 0 if (mr > 0) ^ (hel < 0) else 2
                self.u[0 + r] = sgn*omm*self.u[2 - r]
                self.u[1 + r] = sgn*omm*self.u[3 - r]
                self.u[2 - r] *= omp
                self.u[3 - r] *= omp
                self.on = 3

        if self.bar < 0:
            self.bar = 1
            self.Bar()

        super().__init__(self.u, [i])

    def Bar(self):
        self.u = np.array([self.u[2], self.u[3], self.u[0], self.u[1]])
        self.u = self.u.conjugate()
        self.on = (self.on & 1) << 1 | (self.on & 2) >> 1


def SpinorU(p, i, hel):
    return Spinor(p, i, 1, 2*hel-1)


def SpinorUBar(p, i, hel):
    return Spinor(p, i, 1, 2*hel-1, bar=-1)


def SpinorV(p, i, hel):
    return Spinor(p, i, 1, -(2*hel-1))


def SpinorVBar(p, i, hel):
    return Spinor(p, i, 1, -(2*hel-1), bar=-1)
