import numpy as np
import leptonic_tensor.lorentz_structures as ls
import leptonic_tensor.lorentz_tensor as lt
from leptonic_tensor.ufo_grammer import UFOParser


ufo = UFOParser()


def test_decimal():
    assert(ufo('10') == 10)
    assert(ufo('0') == 0)


def test_float():
    assert(ufo('45.545') == 45.545)
    assert(ufo('45.e-5') == 45.e-5)


def test_imaginary():
    assert(ufo('3j') == 3j)
    assert(ufo('1.13j') == 1.13j)


def test_complex():
    assert(ufo('3+3j') == 3+3j)
    assert(ufo('2.4+4.2j') == 2.4+4.2j)


def test_ufo_metric():
    # \eta^{\mu\nu} \eta_{\mu\nu} = 4
    assert(ufo("Metric(0, 1)*Metric(0, 1)") == 4)

    # \eta^{\mu\nu} \eta_{\nu\rho} = \delta^\mu_\rho
    assert(ufo("Metric(0, 1)*Metric(1, 2)") == ls.LorentzIdentity(0, 2))


def test_charge_conj():
    # C*C = -I_4
    assert(ufo("C(0, 1)*C(1, 2)") == -ls.Identity(0, 2))


# Testing gamma matrix identities from:
# - https://en.wikipedia.org/wiki/Gamma_matrices#Identities
def test_gamma():
    # \gamma^\mu \gamma_\mu = 4 I_4
    lhs = ufo("Gamma(0, 1, 2)*Metric(0, 3)*Gamma(3, 2, 4)")
    rhs = 4*ls.Identity(1, 4)
    assert(lhs == rhs)

    # \gamma^\mu \gamma^\nu \gamma_\mu = -2 \gamma^\nu
    lhs = ufo("Gamma(0, 3, 4)*Metric(0, 2)*Gamma(1, 4, 5)*Gamma(2, 5, 6)")
    rhs = -2*ls.Gamma(1, 3, 6)
    assert(lhs == rhs)

    # \gamma^\mu \gamma^\nu \gamma^\rho \gamma_\mu = 4 \eta^{\mu\rho} I_4
    lhs = ufo("Gamma(0,4,5)*Metric(0,3)*Gamma(1,5,6)"
              "*Gamma(2,6,7)*Gamma(3,7,8)")
    rhs = 4*ls.Metric(1, 2)*ls.Identity(4, 8)
    assert(lhs == rhs)

    # \gamma^\mu \gamma^\nu gamma^\rho = \eta^{\mu\nu}\gamma^\rho
    #                                  + \eta^{\nu\rho}\gamma^\mu
    #                                  - \eta^{\mu\rho}\gamma^\nu
    #                                  - i\eps^{\sigma\mu\nu\rho}
    #                                    * \gamma_\sigma\gamma5
    lhs = ufo("Gamma(0, 3, 4)*Gamma(1, 4, 5)*Gamma(2, 5, 6)")
    rhs = ls.Metric(0, 1)*ls.Gamma(2, 3, 6)
    rhs += ls.Metric(1, 2)*ls.Gamma(0, 3, 6)
    rhs -= ls.Metric(0, 2)*ls.Gamma(1, 3, 6)
    rhs -= (1j*ls.Epsilon(0, 1, 2, 7)*ls.Metric(7, 8)
            * ls.Gamma(8, 3, 9)*ls.Gamma5(9, 6))
    assert(lhs == rhs)


# Testing trace gamma matrix identities from:
# - https://en.wikipedia.org/wiki/Gamma_matrices#Identities
def test_trace_gamma():
    # Tr(\gamma^\mu) = 0
    lhs = ufo("Gamma(0, 1, 1)").reduce()
    rhs = lt.Tensor(np.zeros(4), [lt.LorentzIndex(0)])
    assert(lhs == rhs)

    # TODO: Get this working, and ensure it is quick
    # Trace of any product of an odd number of \gamma^\mu is zero
    # Trace of \gamma5 * any product of an odd number of \gamma^\mu is zero
    # Tested here for 3, 5, 7, 9, 11, ..., nmax
    # nmax = 3
    # for i in range(3, nmax+1, 2):
    #     lhs = 1
    #     for j in range(0, i):
    #         lhs *= ls.Gamma(j, i+j, 2*i+j)
    #     rhs = lt.Tensor(np.zeros(i*[4]), [k for k in range(i)])
    #     assert(lhs == rhs)

    # Tr(\gamma^\mu\gamma^\nu) = 4\eta^{\mu\nu}
    lhs = ufo("Gamma(0, 1, 2)*Gamma(3, 2, 1)")
    rhs = 4*ls.Metric(0, 3)
    assert(lhs == rhs)

    # Tr(\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma)
    # = 4*( \eta^{\mu\nu}\eta^{\rho\sigma}
    #     - \eta^{\mu\rho}\eta^{\nu\sigma}
    #     + \eta^{\mu\sigma}\eta^{\nu\rho})
    lhs = ufo("Gamma(0, 4, 5)*Gamma(1, 5, 6)*Gamma(2, 6, 7)*Gamma(3, 7, 4)")
    rhs = ls.Metric(0, 1)*ls.Metric(2, 3)
    rhs -= ls.Metric(0, 2)*ls.Metric(1, 3)
    rhs += ls.Metric(0, 3)*ls.Metric(1, 2)
    rhs *= 4
    assert(lhs == rhs)

    # Tr(\gamma5) = 0
    lhs = ufo("Gamma5(0, 0)").reduce()
    assert(lhs == 0)

    # Tr(\gamma^\mu\gamma^\nu\gamma5) = 0
    lhs = ufo("Gamma(0, 2, 3)*Gamma(1, 3, 4)*Gamma5(4, 2)")
    rhs = lt.Tensor(np.zeros([4, 4]), (lt.LorentzIndex(0), lt.LorentzIndex(1)))
    assert(lhs == rhs)

    # Tr(\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma)
    # = 4i\eps^{\mu\nu\rho\sigma}
    lhs = ufo("Gamma(0, 4, 5)*Gamma(1, 5, 6)*Gamma(2, 6, 7)"
              "*Gamma(3, 7, 8)*Gamma5(8, 4)")
    rhs = 4j*ls.Epsilon(0, 1, 2, 3)
    assert(lhs == rhs)


def test_projections():
    # P_L^2 = P_L
    assert(ufo("ProjM(0, 1)*ProjM(1, 2)") == ls.ProjM(0, 2))

    # P_R^2 = P_R
    assert(ufo("ProjP(0, 1)*ProjP(1, 2)") == ls.ProjP(0, 2))

    # P_L * P_R = P_R * P_L = 0
    term1 = ufo("ProjM(0, 1)*ProjP(1, 2)")
    term2 = ufo("ProjP(0, 1)*ProjM(1, 2)")
    term3 = lt.Tensor(np.zeros([4, 4]), (lt.SpinIndex(0), lt.SpinIndex(2)))
    assert(term1 == term2 == term3)


def test_parameters():
    ufo("MU_R := 91.188")
    mu_r = ufo("MU_R")
    assert(mu_r == 91.188)
    ufo("aEWM1 := 127.8")
    ufo("aEW := 1 / aEWM1")
    ufo("ee := cmath.sqrt(aEW)")
    ufo("GC_1 := ee*complex(0, -1)")
    GC_1 = ufo("GC_1")
    result = np.sqrt(1.0/127.8)*complex(0, -1)
    assert(GC_1 == result)
