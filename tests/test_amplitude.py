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
    mom = np.array([np.sqrt(200.0, dtype=np.float64), 10, 0, 10], dtype=np.float64)
    mom2 = np.array([np.sqrt(200), 0, 10, 10])
    pslash = ls.Momentum(mom, 2)*ls.Metric(2, 3)*ls.Gamma(3, 0, 1)
    outer_u = (ls.SpinorU(mom, 0, 0)*ls.SpinorUBar(mom, 1, 0)
               + ls.SpinorU(mom, 0, 1)*ls.SpinorUBar(mom, 1, 1))
    outer_v = (ls.SpinorV(mom, 0, 0)*ls.SpinorVBar(mom, 1, 0)
               + ls.SpinorV(mom, 0, 1)*ls.SpinorVBar(mom, 1, 1))
    outer_u2 = (ls.SpinorUBar(mom, 0, 0)*ls.SpinorU(mom, 1, 0)
                + ls.SpinorUBar(mom, 0, 1)*ls.SpinorU(mom, 1, 1))
    assert(np.allclose(np.transpose(outer_u2._array), outer_u._array))

    p = mom
    mass2 = p[0]**2-(np.sum(p[1:]**2))
    if abs(mass2) < 1e-6:
        mass = 0
    else:
        mass = np.sqrt(mass2)
    pslash_pm = pslash + mass*ls.Identity(0, 1)
    pslash_mm = pslash - mass*ls.Identity(0, 1)
    assert(np.allclose(pslash_pm._array, outer_u._array))
    assert(np.allclose(pslash_mm._array, outer_v._array))

    sum_u1 = ls.SpinorUBar(mom, 0, 0)*ls.SpinorU(mom, 0, 0)
    sum_u2 = ls.SpinorUBar(mom, 0, 1)*ls.SpinorU(mom, 0, 0)
    sum_u3 = ls.SpinorUBar(mom, 0, 0)*ls.SpinorU(mom, 0, 1)
    sum_u4 = ls.SpinorUBar(mom, 0, 1)*ls.SpinorU(mom, 0, 1)

    assert(abs(sum_u1.real() - 2*mass) < 1e-8)
    assert(abs(sum_u2.real()) < 1e-8)
    assert(abs(sum_u3.real()) < 1e-8)
    assert(abs(sum_u4.real() - 2*mass) < 1e-8)

    sum_v1 = ls.SpinorVBar(mom, 0, 0)*ls.SpinorV(mom, 0, 0)
    sum_v2 = ls.SpinorVBar(mom, 0, 1)*ls.SpinorV(mom, 0, 0)
    sum_v3 = ls.SpinorVBar(mom, 0, 0)*ls.SpinorV(mom, 0, 1)
    sum_v4 = ls.SpinorVBar(mom, 0, 1)*ls.SpinorV(mom, 0, 1)

    assert(abs(sum_v1.real() + 2*mass) < 1e-8)
    assert(abs(sum_v2.real()) < 1e-8)
    assert(abs(sum_v3.real()) < 1e-8)
    assert(abs(sum_v4.real() + 2*mass) < 1e-8)

    s1 = ls.Spinor(mom, 0, 1)
    s2 = ls.Spinor(mom, 0, -1)
    s1b = ls.Spinor(mom, 1, 1, bar=-1)
    s2b = ls.Spinor(mom, 1, -1, bar=-1)
    result = 2*mom
    assert(np.allclose(result, (s1b*ls.Gamma(0, 1, 0)*s1)._array))
    assert(np.allclose(result, (s2b*ls.Gamma(0, 1, 0)*s2)._array))


def test_interference():
    # costheta = 2*np.random.uniform()-1
    costheta = 0
    sintheta = np.sqrt(1-costheta**2)
    # phi = 2*np.pi*np.random.uniform()
    phi = 0
    mom = np.array(
        [[-10, 0, 0, -10],
         [-10, 0, 0, 10],
         [10, 10*sintheta*np.cos(phi),
          10*sintheta*np.sin(phi), 10*costheta],
         [10, -10*sintheta*np.cos(phi),
          -10*sintheta*np.sin(phi), -10*costheta]])
    p12 = mom[0] + mom[1]
    p13 = mom[0] + mom[2]
    p14 = mom[0] + mom[3]
    s = p12[0]**2 - np.sum(p12[1:]**2)
    t = p13[0]**2 - np.sum(p13[1:]**2)
    u = p14[0]**2 - np.sum(p14[1:]**2)
    assert(np.allclose(s+t+u, 0))
    exact = 8*((u*u + s*s)/t**2 + 2*u*u/(s*t) + (u*u + t*t)/s**2)
    total = 0
    diag1 = 0
    diag2 = 0
    diag3 = 0
    for spin0 in range(2):
        for spin1 in range(2):
            for spin2 in range(2):
                for spin3 in range(2):
                    term1 = (ls.SpinorUBar(mom[2], 2, spin2)
                             * ls.Gamma(0, 2, 0)
                             * ls.SpinorV(mom[0], 0, spin0)
                             * ls.SpinorUBar(mom[1], 1, spin1)
                             * ls.Gamma(1, 1, 3)
                             * ls.SpinorV(mom[3], 3, spin3)
                             * ls.Metric(0, 1) / s
                             + ls.SpinorUBar(mom[1], 1, spin1)
                             * ls.Gamma(2, 1, 0)
                             * ls.SpinorV(mom[0], 0, spin0)
                             * ls.SpinorUBar(mom[2], 2, spin2)
                             * ls.Gamma(3, 2, 3)
                             * ls.SpinorV(mom[3], 3, spin3)
                             * ls.Metric(2, 3) / t)*1j
                    term2 = (ls.SpinorUBar(mom[2], 2, spin2)
                             * ls.SpinorV(mom[0], 0, spin0)
                             * ls.SpinorUBar(mom[1], 1, spin1)
                             * ls.SpinorV(mom[3], 3, spin3)
                             * (ls.Gamma(1, 1, 3)
                                * ls.Gamma(0, 2, 0)
                                * ls.Metric(0, 1) / s
                                + ls.Gamma(0, 1, 0)
                                * ls.Gamma(1, 2, 3)
                                * ls.Metric(0, 1) / t))*1j
                    d1 = (ls.SpinorUBar(mom[2], 2, spin2)
                          * ls.Gamma(4, 2, 0)
                          * ls.SpinorV(mom[0], 0, spin0)
                          * ls.SpinorUBar(mom[1], 1, spin1)
                          * ls.Gamma(5, 1, 3)
                          * ls.SpinorV(mom[3], 3, spin3)
                          * ls.Metric(4, 5) / s)*1j
                    d2 = (ls.SpinorUBar(mom[1], 1, spin1)
                          * ls.Gamma(2, 1, 0)
                          * ls.SpinorV(mom[0], 0, spin0)
                          * ls.SpinorUBar(mom[2], 2, spin2)
                          * ls.Gamma(3, 2, 3)
                          * ls.SpinorV(mom[3], 3, spin3)
                          * ls.Metric(2, 3) / t)*1j
                    # print(d1*d1.conjugate(), d2*d2.conjugate(), 2*d1*d2.conjugate())
                    diag1 += complex(d1)*complex(d1).conjugate()
                    diag2 += complex(d2)*complex(d2).conjugate()
                    diag3 += 2*complex(d2)*complex(d1).conjugate()
                    total += complex(term1)*complex(term1).conjugate()
                    result = complex(term1 - term2)
                    assert(np.allclose(abs(result), 0))
    print(diag1, diag2, diag3)
    print(8*(u**2+t**2)/s**2, 8*(u**2+s**2)/t**2, 16*u**2/(s*t))
    assert(np.allclose(total, exact))


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

    vbguconj1 = (ls.SpinorVBar(mom[1], 1, 1)
                 * ls.Gamma(0, 1, 0)
                 * ls.SpinorU(mom[0], 0, 0)).conjugate()
    vbguconj2 = (ls.SpinorVBar(mom[1], 1, 0)
                 * ls.Gamma(0, 1, 0)
                 * ls.SpinorU(mom[0], 0, 1)).conjugate()
    ubgv1 = (ls.SpinorUBar(mom[0], 1, 0)
             * ls.Gamma(0, 1, 0)
             * ls.SpinorV(mom[1], 0, 1))
    ubgv2 = (ls.SpinorUBar(mom[0], 1, 1)
             * ls.Gamma(0, 1, 0)
             * ls.SpinorV(mom[1], 0, 0))

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
    q2 = float((ls.Momentum(mom[4], 0)*ls.Metric(0, 1)*ls.Momentum(mom[4], 1)))
    t = float((ls.Momentum(mom[5], 0)*ls.Metric(0, 1)*ls.Momentum(mom[5], 1)))
    u = float((ls.Momentum(mom[6], 0)*ls.Metric(0, 1)*ls.Momentum(mom[6], 1)))
    total = 0
    total2 = 0
    for spin0 in range(2):
        for spin1 in range(2):
            for spin2 in range(2):
                for spin3 in range(2):
                    total += (ls.SpinorVBar(mom[1], 1, spin1)
                              * ls.Gamma(0, 1, 0)
                              * ls.SpinorU(mom[0], 0, spin0)
                              * ls.SpinorUBar(mom[0], 3, spin0)
                              * ls.Gamma(1, 3, 4)
                              * ls.SpinorV(mom[1], 4, spin1)

                              * ls.SpinorUBar(mom[3], 5, spin3)
                              * ls.Gamma(0, 5, 6)
                              * ls.SpinorV(mom[2], 6, spin2)
                              * ls.SpinorVBar(mom[2], 7, spin2)
                              * ls.Gamma(1, 7, 8)
                              * ls.SpinorU(mom[3], 8, spin3))

                    tmp = 1j*(ls.SpinorVBar(mom[1], 1, spin1)
                              * ls.Gamma(0, 1, 0)
                              * ls.SpinorU(mom[0], 0, spin0)
                              * ls.SpinorUBar(mom[3], 3, spin3)
                              * ls.Gamma(1, 3, 2)
                              * ls.SpinorV(mom[2], 2, spin2)
                              * ls.Metric(0, 1))
                    total2 += tmp*tmp.conjugate()
    result1 = total*ee**4/q2**2
    result2 = total2*ee**4/q2**2
    exact = 8*ee**4*(t*t+u*u)/q2**2
    assert(np.allclose(result1.real(), exact))
    assert(np.allclose(result2.real(), exact))
