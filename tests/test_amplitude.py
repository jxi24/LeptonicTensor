import numpy as np
import leptonic_tensor.lorentz_structures as ls
import leptonic_tensor.lorentz_tensor as lt
from leptonic_tensor.ufo_grammer import UFOParser

ufo = UFOParser()
ufo("aEWM1 := 127.8")
ufo("aEW := 1 / aEWM1")
ufo("ee := 2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)")
ee = ufo("ee")


def test_spinor():
    mom = np.array([[np.sqrt(300), 10, 0, 10]])
    pslash = ls.Momentum(mom, 2, 0)*ls.Metric(2, 3)*ls.Gamma(3, 0, 1)
    outer_u = (ls.SpinorU(mom, 0, 0, 0)*ls.SpinorUBar(mom, 1, 0, 0)
               + ls.SpinorU(mom, 0, 0, 1)*ls.SpinorUBar(mom, 1, 0, 1))
    outer_v = (ls.SpinorV(mom, 0, 0, 0)*ls.SpinorVBar(mom, 1, 0, 0)
               + ls.SpinorV(mom, 0, 0, 1)*ls.SpinorVBar(mom, 1, 0, 1))
    p = mom[0]
    mass = np.sqrt(p[0]**2-(np.sum(p[1:]**2)))
    pslash_pm = pslash + mass*ls.Identity(0, 1)
    pslash_mm = pslash - mass*ls.Identity(0, 1)
    assert(np.allclose(pslash_pm._array, outer_u._array))
    assert(np.allclose(pslash_mm._array, outer_v._array))

    sum_u1 = ls.SpinorUBar(mom, 0, 0, 0)*ls.SpinorU(mom, 0, 0, 0)
    sum_u2 = ls.SpinorUBar(mom, 0, 0, 1)*ls.SpinorU(mom, 0, 0, 0)
    sum_u3 = ls.SpinorUBar(mom, 0, 0, 0)*ls.SpinorU(mom, 0, 0, 1)
    sum_u4 = ls.SpinorUBar(mom, 0, 0, 1)*ls.SpinorU(mom, 0, 0, 1)

    assert(abs(sum_u1.real() - 2*mass) < 1e-8)
    assert(abs(sum_u2.real()) < 1e-8)
    assert(abs(sum_u3.real()) < 1e-8)
    assert(abs(sum_u4.real() - 2*mass) < 1e-8)

    sum_v1 = ls.SpinorVBar(mom, 0, 0, 0)*ls.SpinorV(mom, 0, 0, 0)
    sum_v2 = ls.SpinorVBar(mom, 0, 0, 1)*ls.SpinorV(mom, 0, 0, 0)
    sum_v3 = ls.SpinorVBar(mom, 0, 0, 0)*ls.SpinorV(mom, 0, 0, 1)
    sum_v4 = ls.SpinorVBar(mom, 0, 0, 1)*ls.SpinorV(mom, 0, 0, 1)

    assert(abs(sum_v1.real() + 2*mass) < 1e-8)
    assert(abs(sum_v2.real()) < 1e-8)
    assert(abs(sum_v3.real()) < 1e-8)
    assert(abs(sum_v4.real() + 2*mass) < 1e-8)


def test_spinor_conjugate():
    costheta = 2*np.random.uniform()-1
    sintheta = np.sqrt(1-costheta**2)
    phi = 2*np.pi*np.random.uniform()
    mom = np.array(
        [[10, 0, 0, 10],
         [10, 0, 0, -10],
         [10, 10*sintheta*np.cos(phi),
          10*sintheta*np.sin(phi), 10*costheta],
         [10, -10*sintheta*np.cos(phi),
          -10*sintheta*np.sin(phi), -10*costheta]])

    vbguconj1 = (ls.SpinorVBar(mom, 1, 1, 1)
                 * ls.Gamma(0, 1, 0)
                 * ls.SpinorU(mom, 0, 0, 0)).conjugate()
    vbguconj2 = (ls.SpinorVBar(mom, 1, 1, 0)
                 * ls.Gamma(0, 1, 0)
                 * ls.SpinorU(mom, 0, 0, 1)).conjugate()
    ubgv1 = (ls.SpinorUBar(mom, 1, 0, 0)
             * ls.Gamma(0, 1, 0)
             * ls.SpinorV(mom, 0, 1, 1))
    ubgv2 = (ls.SpinorUBar(mom, 1, 0, 1)
             * ls.Gamma(0, 1, 0)
             * ls.SpinorV(mom, 0, 1, 0))

    assert(vbguconj1 == ubgv1)
    assert(vbguconj2 == ubgv2)


def test_eemumu_amplitude():
    costheta = 2*np.random.uniform()-1
    # costheta = 1.0
    sintheta = np.sqrt(1-costheta**2)
    phi = 2*np.pi*np.random.uniform()
    # phi = 0.0
    mom = np.array(
        [[10, 0, 0, 10],
         [10, 0, 0, -10],
         [10, 10*sintheta*np.cos(phi),
          10*sintheta*np.sin(phi), 10*costheta],
         [10, -10*sintheta*np.cos(phi),
          -10*sintheta*np.sin(phi), -10*costheta]])

    q = mom[0, :] + mom[1, :]
    mom = np.append(mom, [q], axis=0)
    mom = np.append(mom, [mom[0] - mom[2]], axis=0)
    mom = np.append(mom, [mom[0] - mom[3]], axis=0)
    q2 = float((ls.Momentum(mom, 0, 4)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 4)))
    t = float((ls.Momentum(mom, 0, 5)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 5)))
    u = float((ls.Momentum(mom, 0, 6)*ls.Metric(0, 1)*ls.Momentum(mom, 1, 6)))
    total = 0
    total2 = 0
    for spin0 in range(2):
        for spin1 in range(2):
            for spin2 in range(2):
                for spin3 in range(2):
                    total += (ls.SpinorVBar(mom, 1, 1, spin1)
                              * ls.Gamma(0, 1, 0)
                              * ls.SpinorU(mom, 0, 0, spin0)
                              * ls.SpinorUBar(mom, 3, 0, spin0)
                              * ls.Gamma(1, 3, 4)
                              * ls.SpinorV(mom, 4, 1, spin1)

                              * ls.SpinorUBar(mom, 5, 3, spin3)
                              * ls.Gamma(0, 5, 6)
                              * ls.SpinorV(mom, 6, 2, spin2)
                              * ls.SpinorVBar(mom, 7, 2, spin2)
                              * ls.Gamma(1, 7, 8)
                              * ls.SpinorU(mom, 8, 3, spin3))

                    tmp = 1j*(ls.SpinorVBar(mom, 1, 1, spin1)
                              * ls.Gamma(0, 1, 0)
                              * ls.SpinorU(mom, 0, 0, spin0)
                              * ls.SpinorUBar(mom, 3, 3, spin3)
                              * ls.Gamma(1, 3, 2)
                              * ls.SpinorV(mom, 2, 2, spin2)
                              * ls.Metric(0, 1))
                    total2 += tmp*tmp.conjugate()
    result1 = total*ee**4/q2**2
    result2 = total2*ee**4/q2**2
    exact = 8*ee**4*(t*t+u*u)/q2**2
    assert(np.allclose(result1.real(), exact))
    assert(np.allclose(result2.real(), exact))
