import numpy as np
from leptonic_tensor.ufo_grammer import UFOParser
import leptonic_tensor.Models.SM_NLO as model


ufo = UFOParser(model)


def test_complex_functions():
    assert(ufo('complexconjugate(1+1j)') == 1-1j)
    assert(ufo('re(10)') == 10)
    assert(ufo('im(10j)') == 10)


def test_trig_functions():
    assert(ufo('sec(0.5)') == 1.0/np.cos(0.5))
    assert(ufo('asec(5)') == np.arccos(1.0/5))
    assert(ufo('csc(0.5)') == 1.0/np.sin(0.5))
    assert(ufo('acsc(5)') == np.arcsin(1.0/5))
    assert(ufo('cot(0.5)') == 1.0/np.tan(0.5))
