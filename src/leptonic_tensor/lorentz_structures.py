import numpy as np

METRIC_TENSOR = np.diag([1, -1, -1, -1])
IDENTITY = np.diag([1, 1, 1, 1])


def eps(i, j, k, l):
    return (i-j)*(i-k)*(i-l)*(j-k)*(j-l)*(k-l)/12


EPS = np.fromfunction(eps, (4, 4, 4, 4), dtype=int)

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

GAMMA_5 = 1j*np.einsum('ij,jk,kl,lm->im', GAMMA_0, GAMMA_1, GAMMA_2, GAMMA_3)
GAMMA_MU = np.array([GAMMA_0, GAMMA_1, GAMMA_2, GAMMA_3])

print('Identities from Wikipedia:')
print('--------------------------')
print(r'gamma^\mu gamma_\mu = 4I_4: {}'.format(
    np.array_equal(np.einsum('aij,ab,bjl->il', GAMMA_MU,
                             METRIC_TENSOR, GAMMA_MU),
                   4*IDENTITY)))
print(r'gamma^\mu gamma^\nu gamma_\mu = -2 gamma^\nu: {}'.format(
    np.array_equal(np.einsum('aij,cjk,ab,bkl->cil',
                             GAMMA_MU, GAMMA_MU,
                             METRIC_TENSOR, GAMMA_MU),
                   -2*GAMMA_MU)))
print(r'gamma^\mu gamma^\nu gamma^\rho gamma_\mu = 4\eta^\nu\rho I_4: {}'.format(
    np.array_equal(np.einsum('aij,bjk,ckl,ad,dlm->bcim',
                             GAMMA_MU, GAMMA_MU, GAMMA_MU,
                             METRIC_TENSOR, GAMMA_MU),
                   4*np.einsum('ab,ij->abij', METRIC_TENSOR, IDENTITY))))
print(r'Tr(gamma^\mu) = 0: {}'.format(
    np.array_equal(np.einsum('aii->', GAMMA_MU), 0)))
print(r'Tr(gamma^\mu gamma^\nu) = 4\eta^\mu\nu: {}'.format(
    np.array_equal(np.einsum('aij,bji->ab', GAMMA_MU, GAMMA_MU),
                   4*METRIC_TENSOR)))
print(r'Tr(gamma^\mu gamma^\nu gamma^\rho) = 0: {}'.format(
    np.array_equal(np.einsum('aij,bjk,cki->', GAMMA_MU, GAMMA_MU, GAMMA_MU), 0)))
print(r'Tr(gamma^5) = 0: {}'.format(
    np.array_equal(np.einsum('ii', GAMMA_5), 0)))
print(r'Tr(gamma^\mu gamma^\nu gamma^5) = 0: {}'.format(
    np.array_equal(np.einsum('aij,bjk,ki->', GAMMA_MU, GAMMA_MU, GAMMA_5), 0)))
print(r'Tr(gamma^\mu gamma^\nu gamma^\rho gamma^\sigma gamma^5) = 4i\esp^\mu\nu\rho\sigma: {}'.format(
    np.array_equal(np.einsum('aij,bjk,ckl,dlm,mi->abcd', GAMMA_MU, GAMMA_MU, GAMMA_MU, GAMMA_MU, GAMMA_5),
                   4j*EPS)))
