from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
from sympy.physics.hep.gamma_matrices import kahane_simplify
from sympy.tensor.tensor import tensor_indices

i0, i1, i2 = tensor_indices('i0:3', LorentzIndex)
ta = G(i0)*G(-i0)
print(kahane_simplify(ta))
tb = G(i0)*G(i1)*G(-i0)
print(kahane_simplify(tb))
