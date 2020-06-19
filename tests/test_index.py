import leptonic_tensor.lorentz_tensor as lt


def test_index():
    sindex0, sindex1 = lt.SpinIndex(0), lt.SpinIndex(1)
    lindex0, lindex1 = lt.LorentzIndex(0), lt.LorentzIndex(1)
    # SpinIndex(0) != LorentzIndex(0)
    assert(sindex0 != lindex0)

    # SpinIndex(0) == SpinIndex(0) != SpinIndex(1)
    assert(sindex0 == sindex0 != sindex1)

    # LorentzIndex(0) == LorentzIndex(0) != LorentzIndex(1)
    assert(lindex0 == lindex0 != lindex1)

    # To integer
    assert(int(lindex0) == 0)

    # To string
    assert(str(lindex0) == 'A')
    assert(str(sindex0) == 'a')
