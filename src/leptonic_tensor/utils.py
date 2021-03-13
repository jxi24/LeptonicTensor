import numpy as np


def Pt(mom):
    return np.sqrt(mom[:, 1]**2 + mom[:, 2]**2)


def CosTheta(mom):
    return np.cos(np.arctan2(Pt(mom), mom[:, 3]))


def Dot(mom1, mom2):
    return (mom1[:, 0]*mom2[:, 0])[:, None]-np.sum(mom1[:, 1:]*mom2[:, 1:],
                                                   axis=-1, keepdims=True)


def ctz(inp):
    return (inp & -inp).bit_length() - 1


def next_permutation(inp):
    t = inp | (inp - 1)
    return (t+1) | (((~t & -~t) - 1) >> (ctz(inp) + 1))


def set_bits(inp, setbits, size):
    iset = 0
    for i in range(size):
        if(inp & (1 << i)):
            setbits[iset] = (inp & (1 << i))
            iset += 1
