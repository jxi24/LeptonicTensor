import numpy as np
import leptonic_tensor.lorentz_structures as ls
import leptonic_tensor.lorentz_tensor as lt


def test_metric():
    # \eta^{\mu\nu} \eta_{\mu\nu} = 4
    assert(ls.Metric(0, 1)*ls.Metric(0, 1) == 4)

    # \eta^{\mu\nu} \eta_{\nu\rho} = \delta^\mu_\rho
    assert(ls.Metric(0, 1)*ls.Metric(1, 2) == ls.LorentzIdentity(0, 2))


def test_charge_conj():
    # C*C = -I_4
    assert(ls.ChargeConj(0, 1)*ls.ChargeConj(1, 2) == -ls.Identity(0, 2))


# Testing gamma matrix identities from:
# - https://en.wikipedia.org/wiki/Gamma_matrices#Identities
def test_gamma():
    # \gamma^\mu \gamma_\mu = 4 I_4
    lhs = ls.Gamma(0, 1, 2)*ls.Metric(0, 3)*ls.Gamma(3, 2, 4)
    rhs = 4*ls.Identity(1, 4)
    assert(lhs == rhs)

    # \gamma^\mu \gamma^\nu \gamma_\mu = -2 \gamma^\nu
    lhs = ls.Gamma(0, 3, 4)*ls.Metric(0, 2)*ls.Gamma(1, 4, 5)*ls.Gamma(2, 5, 6)
    rhs = -2*ls.Gamma(1, 3, 6)
    print(lhs, rhs)
    assert(lhs == rhs)

    # \gamma^\mu \gamma^\nu \gamma^\rho \gamma_\mu = 4 \eta^{\mu\rho} I_4
    lhs = ls.Gamma(0, 4, 5)*ls.Metric(0, 3)*ls.Gamma(1, 5, 6)
    lhs *= ls.Gamma(2, 6, 7)*ls.Gamma(3, 7, 8)
    rhs = 4*ls.Metric(1, 2)*ls.Identity(4, 8)
    assert(lhs == rhs)

    # \gamma^\mu \gamma^\nu gamma^\rho = \eta^{\mu\nu}\gamma^\rho
    #                                  + \eta^{\nu\rho}\gamma^\mu
    #                                  - \eta^{\mu\rho}\gamma^\nu
    #                                  - i\eps^{\sigma\mu\nu\rho}
    #                                    * \gamma_\sigma\gamma5
    lhs = ls.Gamma(0, 3, 4)*ls.Gamma(1, 4, 5)*ls.Gamma(2, 5, 6)
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
    # lhs = ls.Gamma(0, 1, 1).reduce()
    # rhs = lt.Tensor(np.zeros(4), [0])
    # assert(lhs == rhs)

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
    lhs = ls.Gamma(0, 1, 2)*ls.Gamma(3, 2, 1)
    rhs = 4*ls.Metric(0, 3)
    assert(lhs == rhs)

    # Tr(\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma)
    # = 4*( \eta^{\mu\nu}\eta^{\rho\sigma}
    #     - \eta^{\mu\rho}\eta^{\nu\sigma}
    #     + \eta^{\mu\sigma}\eta^{\nu\rho})
    lhs = ls.Gamma(0, 4, 5)*ls.Gamma(1, 5, 6)*ls.Gamma(2, 6, 7)
    lhs *= ls.Gamma(3, 7, 4)
    rhs = ls.Metric(0, 1)*ls.Metric(2, 3)
    rhs -= ls.Metric(0, 2)*ls.Metric(1, 3)
    rhs += ls.Metric(0, 3)*ls.Metric(1, 2)
    rhs *= 4
    assert(lhs == rhs)

    # Tr(\gamma5) = 0
    # lhs = ls.Gamma5(0, 0).reduce()
    # assert(lhs == 0)

    # Tr(\gamma^\mu\gamma^\nu\gamma5) = 0
    lhs = ls.Gamma(0, 2, 3)*ls.Gamma(1, 3, 4)*ls.Gamma5(4, 2)
    rhs = lt.Tensor(np.zeros([4, 4]), (0, 1))
    assert(lhs == rhs)

    # Tr(\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma)
    # = 4i\eps^{\mu\nu\rho\sigma}
    lhs = ls.Gamma(0, 4, 5)*ls.Gamma(1, 5, 6)*ls.Gamma(2, 6, 7)
    lhs *= ls.Gamma(3, 7, 8)*ls.Gamma5(8, 4)
    rhs = 4j*ls.Epsilon(0, 1, 2, 3)
    assert(lhs == rhs)


def test_projections():
    # P_L^2 = P_L
    assert(ls.ProjM(0, 1)*ls.ProjM(1, 2) == ls.ProjM(0, 2))

    # P_R^2 = P_R
    assert(ls.ProjP(0, 1)*ls.ProjP(1, 2) == ls.ProjP(0, 2))

    # P_L * P_R = P_R * P_L = 0
    term1 = ls.ProjM(0, 1)*ls.ProjP(1, 2)
    term2 = ls.ProjP(0, 1)*ls.ProjM(1, 2)
    term3 = lt.Tensor(np.zeros([4, 4]), (lt.SpinIndex(0), lt.SpinIndex(2)))
    assert(term1 == term2 == term3)
