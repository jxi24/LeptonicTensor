import numpy as np
import lorentz_tensor as lt
from utils import Dot

METRIC_TENSOR = np.diag([1, -1, -1, -1])
IDENTITY = np.diag([1, 1, 1, 1])


def eps(i, j, k, l):
    return -(i-j)*(i-k)*(i-l)*(j-k)*(j-l)*(k-l)/12

def eps2(i, j, k, l):
    return np.sign(l-i)*np.sign(k-i)*np.sign(j-i)*np.sign(l-j)*np.sign(k-j)*np.sign(l-k)


EPS = np.fromfunction(eps, (4, 4, 4, 4), dtype=int)
EPS2 = np.fromfunction(eps2, (4, 4, 4, 4), dtype=int)

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
PROJ_M = (IDENTITY - GAMMA_5)/2
PROJ_P = (IDENTITY + GAMMA_5)/2

FFV1 = GAMMA_MU
FFV2 = np.einsum('mij,jk->mik', GAMMA_MU, PROJ_M)
FFV3 = np.einsum('mij,jk->mik', GAMMA_MU, PROJ_P)
VVS1 = METRIC_TENSOR
FFS1 = GAMMA_5
FFS2 = IDENTITY
FFS3 = PROJ_M
FFS4 = PROJ_P
SSS1 = 1

class VVV:
    def __init__(self, name, mom):
        self.name = name
        if self.name == 'VVV1':
            P = mom[0]
            VVV1 = np.einsum("i,jk->jki", P, METRIC_TENSOR)
            self.lorentz_tensor = VVV1
        elif self.name == 'VVV2':
            P = mom[1]
            VVV2 = np.einsum("i,jk->jki", P, METRIC_TENSOR)
            self.lorentz_tensor = VVV2
        elif self.name == 'VVV3':
            P = mom[2]
            VVV3 = np.einsum("i,jk->jki", P, METRIC_TENSOR)
            self.lorentz_tensor = VVV3
        elif self.name == 'VVV4':
            P = mom[0]
            VVV4 = np.einsum("i,jk->jik", P, METRIC_TENSOR)
            self.lorentz_tensor = VVV4
        elif self.name == 'VVV5':
            P = mom[1]
            VVV5 = np.einsum("i,jk->jik", P, METRIC_TENSOR)
            self.lorentz_tensor = VVV5
        elif self.name == 'VVV6':
            P = mom[2]
            VVV6 = np.einsum("i,jk->jik", P, METRIC_TENSOR)
            self.lorentz_tensor = VVV6
        elif self.name == 'VVV7':
            P = mom[1]
            VVV7 = np.einsum("i,jk->ijk", P, METRIC_TENSOR)
            self.lorentz_tensor = VVV7
        elif self.name == 'VVV8':
            P = mom[2]
            VVV8 = np.einsum("i,jk->ijk", P, METRIC_TENSOR)
            self.lorentz_tensor = VVV8

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
    # def __init__(self, p, mu):
    #     if not isinstance(mu, lt.Index):
    #         mu = lt.LorentzIndex(mu)
    #     super().__init__(p, [mu])
    def __init__(self, mu, N):
        if not isinstance(mu, lt.Index):
            mu = lt.LorentzIndex(mu)
        self.id = N
        p = np.zeros(4, dtype=np.complex)
        super().__init__(p, [mu], label = self.id)


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
        rpp = np.sqrt(p[:, 0]+p[:, 3]+0j)
        rpm = np.sqrt(p[:, 0]-p[:, 3]+0j)
        pt = p[:, 1] + 1j*p[:, 2]
        self.mu = np.stack([rpp, rpm], axis=-1)
        sv = abs(1e-8*p[:, 0])
        mask = np.logical_and(np.logical_or(abs(pt.real) > sv, abs(pt.imag) > sv),
                              np.logical_or(abs(rpp.real) > sv, abs(rpp.imag) > sv))
        self.mu[:, 1] = np.where(mask,
                                 np.divide(pt.real + 1j*mr*pt.imag,
                                           rpp, out=np.zeros_like(self.mu[:, 1]), where=rpp!=0),
                                 self.mu[:, 1])

    def __getitem__(self, *args):
        return self.mu[args]

    def __neg__(self):
        self.mu = -self.mu
        return self

    def __str__(self):
        return str(self.mu)

    def __mul__(self, other):
        return self.mu[:, 0]*other.mu[:, 1]-self.mu[:, 1]*other.mu[:, 0]


class Spinor:
    def __init__(self, p, mr, hel=0, spin=1, bar=1):
        self.bar = bar
        self.u = np.zeros((p.shape[0], 4), dtype=np.complex)
        if np.all(p[:, 1:] <= 1e-8):
            rte = np.sqrt(p[:, 0]+0j)
            if (mr > 0) ^ (hel < 0):  # u+(p, m) / v-(p, m)
                self.u[:, 2] = rte
            else:  # u-(p, m) / v+(p, m)
                self.u[:, 1] = -rte
            sgn = 1 if mr > 0 else -1
            r = 0 if (mr > 0) ^ (hel < 0) else 2
            self.u[:, 0 + r] = sgn*self.u[:, 2 - r]
            self.u[:, 1 + r] = sgn*self.u[:, 3 - r]
            self.on = 3
        else:
            ph = p.copy()
            ps = np.sqrt(np.sum(p[:, 1:]**2, axis=-1))
            ph[:, 0] = np.where(p[:, 0] < 0, -ps, ps)
            if (mr > 0) ^ (hel < 0):  # u+(p, m) / v-(p, m)
                sh = WeylSpinor(1, ph).mu
                self.u[:, 2] = sh[:, 0]
                self.u[:, 3] = sh[:, 1]
                self.on = 2
            else:  # u-(p, m) / v+(p, m)
                sh = WeylSpinor(-1, ph).mu
                sh = np.where(p[:, 0, np.newaxis] < 0, -sh, sh)
                self.u[:, 0] = sh[:, 1]
                self.u[:, 1] = -sh[:, 0]
                self.on = 1
            # m2 = Dot(p, p)
            # self.u = np.where(m2[:, None] > 1e-8, self.Massive(p, ph, mr, hel, spin), self.u)
            self.u = self.Massive(p, ph, mr, hel, spin)

        if self.bar < 0:
            self.bar = 1
            self.Bar()

    def Massive(self, p, ph, mr, hel, spin):
        u = self.u.copy()
        sgn = 1 if (mr > 0) ^ (spin < 0) else -1
        omp = np.sqrt(p[:, 0]+ph[:, 0]+0j)/(2*ph[:, 0])
        omm = np.sqrt(p[:, 0]-ph[:, 0]+0j)/(2*ph[:, 0])
        r = 0 if (mr > 0) ^ (hel < 0) else 2
        u[:, 0 + r] = sgn*omm*u[:, 2 - r]
        u[:, 1 + r] = sgn*omm*u[:, 3 - r]
        u[:, 2 - r] *= omp
        u[:, 3 - r] *= omp
        return u

    def Bar(self):
        # self.u[:] = np.transpose(self.u[:], (2, 3, 0, 1))
        # self.u[:, 2:3], self.u[:, 0:1] = self.u[:, 0:1], self.u[:, 2:3]
        self.u[:, [2, 0]] = self.u[:, [0, 2]]
        self.u[:, [3, 1]] = self.u[:, [1, 3]]
        self.u = self.u.conjugate()
        self.on = (self.on & 1) << 1 | (self.on & 2) >> 1


def SpinorU(p, hel):
    return Spinor(p, 1, hel)


def SpinorUBar(p, hel):
    return Spinor(p, 1, hel, bar=-1)


def SpinorV(p, hel):
    return Spinor(p, 1, -hel)


def SpinorVBar(p, hel):
    return Spinor(p, 1, -hel, bar=-1)


class PolarizationVector:
    def __init__(self, p, hel=0, conj=1):
        self.conj = conj
        batch = np.size(p, axis=0)
        self.k = np.tile(np.array([1, 1, 0, 0]), (batch, 1))
        self.kp = WeylSpinor(1, self.k)
        self.km = WeylSpinor(-1, self.k)

        # mass2 = p[:,0]**2 - np.sum(p[:,1:]**2)
        # if mass2.all() == 0:
        mass2 = Dot(p, p)    
        if mass2.all() <= 1e-8:
            if hel == 0:
                raise RuntimeError("Invalid helicity for massless particle")

            self.epsilon = (self._ep(p, batch) if hel == 1
                            else self._em(p, batch))
        else:
            if hel == 0:
                self.epsilon = self._eml(p, batch)
            elif hel == 1:
                self.epsilon = self._emp(p, batch)
            elif hel == -1:
                self.epsilon = self._emm(p, batch)

    @staticmethod
    def _vt(a, b, batch):
        eps = np.zeros((batch,4), dtype=np.complex)
        eps[:,0] = a.mu[:,0]*b.mu[:,0]+a.mu[:,1]*b.mu[:,1]
        eps[:,3] = a.mu[:,0]*b.mu[:,0]-a.mu[:,1]*b.mu[:,1]
        eps[:,1] = a.mu[:,0]*b.mu[:,1]+a.mu[:,1]*b.mu[:,0]
        eps[:,2] = 1j*(a.mu[:,0]*b.mu[:,1]-a.mu[:,1]*b.mu[:,0])
        return eps

    def _em(self, p, batch):
        pp = WeylSpinor(1, p)
        eps = self._vt(pp, self.km, batch)
        # print("eps: ", np.shape(eps))
        # print("kp: ", np.shape(self.kp.mu))
        # print("pp: ", np.shape(pp.mu))
        # print("kp*pp: ", np.shape(self.kp*pp))
        # print("sqrt(2)*conj(kp*pp): ", np.shape(np.sqrt(2)*np.conjugate(self.kp*pp)))
        eps = np.einsum("bi, b -> bi", eps, 1/(np.sqrt(2)*np.conjugate(self.kp*pp)))
        # print("new eps: ", np.shape(eps))
        return eps

    def _ep(self, p, batch):
        pm = WeylSpinor(-1, p)
        eps = self._vt(self.kp, pm, batch)
        eps = np.einsum("bi, b -> bi", eps, 1/(np.sqrt(2)*np.conjugate(self.km*pm)))
        # eps /= np.sqrt(2)*np.conjugate(self.km*pm)
        return eps

    def _emm(self, p, batch):
        mass2 = p[:,0]**2 - np.sum(p[:,1:]**2)
        kappa = mass2/(2*(self.k[:,0]*p[:,0]-np.sum(self.k[:,1:]*p[:,1:])))
        # print(np.shape(p), np.shape(self.k), np.shape(kappa))
        return self._em(p-kappa*self.k, batch)

    def _emp(self, p, batch):
        mass2 = p[:,0]**2 - np.sum(p[:,1:]**2)
        kappa = mass2/(2*(self.k[:,0]*p[:,0]-np.sum(self.k[:,1:]*p[:,1:])))
        return self._ep(p-kappa*self.k, batch)

    def _eml(self, p, batch):
        mass2 = p[:,0]**2 - np.sum(p[:,1:]**2)
        dot = 2*(self.k[:,0]*p[:,0]-np.sum(self.k[:,1:]*p[:,1:]))
        kappa = mass2/dot
        b = p - kappa*self.k
        bm = WeylSpinor(-1, b)
        bp = WeylSpinor(1, b)
        eps = self._vt(bp, bm, batch) - kappa*self._vt(self.kp, self.km, batch)
        eps /= 2*np.sqrt(mass2)
        return eps

    def Conjugate(self):
        self.epsilon = np.conjugate(self.epsilon)
